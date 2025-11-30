import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from fs_net import FsResNet18
from collections import OrderedDict
from horizon_plugin_pytorch.quantization import (
    QuantStub,
    prepare,
    set_fake_quantize,
    FakeQuantState,
)
from horizon_plugin_pytorch.quantization.qconfig_template import qat_8bit_weight_16bit_act_qconfig_setter
from horizon_plugin_pytorch.march import set_march
import horizon_plugin_pytorch
from hbdk4.compiler import convert, save, March, hbm_perf, compile
import os

march = March.nash_e

set_march(march)

def rgb2yuv444_bt709_full_range(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[0.213,0.715, 0.072],[-0.117, -0.394, 0.511],[0.511, -0.464, -0.047]])
    bias = np.array([0.0, 128.0, 128.0])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.minimum(np.maximum(np.matmul(trans_mat, src_seq) + bias_seq,0),255)
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3))).astype(np.uint8)
    return dst

det_heads= {
    'hm_pillar': 1,
    'pillar_points': 40,
    'pillar_cls':2*6, # two lines, 6 classes
    'hm_det':1,
    'det_points':20,
    'det_cls':18
    }
        
seg_heads= {
    'seg_fs':1+27,
    'seg_vehped':1+4
    }
float_model = FsResNet18(det_heads, seg_heads, stage="Qat")
float_model = float_model.to("cuda:0")
example_input = torch.rand(1, 3, 768, 960, device="cuda:0")
qat_model = prepare(float_model, example_input, qat_8bit_weight_16bit_act_qconfig_setter)
qat_model_params = torch.load("fs_calib-checkpoint_20251119.ckpt")
new_state_dict = OrderedDict()
for k, v in qat_model_params.items():
    name = k[7:] if k.startswith('module.') else k  # 去掉'module.'
    new_state_dict[name] = v
qat_model.load_state_dict(new_state_dict)
hbir_qat_model = horizon_plugin_pytorch.quantization.hbdk4.export(qat_model, example_input)
# hbir_quantized_model = convert(hbir_qat_model, march, advice=False)
save(hbir_qat_model, "./compile/qat.bc")

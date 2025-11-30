import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from hat.models.task_modules.bevformer.utils import get_clone_module

from collections import OrderedDict
from horizon_plugin_pytorch.quantization import (
    QuantStub,
    prepare,
    set_fake_quantize,
    FakeQuantState,
)
from horizon_plugin_pytorch.quantization.qconfig_template import qat_8bit_weight_16bit_act_qconfig_setter, default_qat_fixed_act_qconfig_setter
from horizon_plugin_pytorch.march import March, set_march
import horizon_plugin_pytorch
from hbdk4.compiler import convert, save, hbm_perf, compile#, March
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.dirname(current_dir))
from ultralytics import YOLO

# march = March.nash_e
march = March.NASH_E

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

import random
def set_seed(seed: int = 42):
    """固定随机数种子，确保结果可复现"""
    # Python 内置随机数
    random.seed(seed)
    # NumPy 随机数
    np.random.seed(seed)
    # PyTorch CPU 随机数
    torch.manual_seed(seed)
    # PyTorch GPU 随机数（单卡和多卡）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保使用确定性算法（可能会降低性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# 固定种子
# set_seed(1234)

torch.backends.cudnn.enabled = True

from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import select_device
import torch

def load_trained_model(ckpt_path: str, device_str="0"):
    # 1. 选设备
    device = select_device(device_str)

    # 2. 使用框架提供的 load_checkpoint
    model, ckpt = load_checkpoint(ckpt_path, device=device)

    # model 就是保存时的 ema 模型（YOLO11DetectionModel）
    # model.eval()
    return model

model_name = "yolo11"

# resnetj6e
if model_name == "resnetj6e":
    from ultralytics.cfg.models_py.resnet_j6e_wrapper import QATResNetJ6EDetectionModel

    pretrained_model_path = "runs/detect_d4q/minieye-driving-d4q-resnetj6e-qat8/weights/last.pt"
    pretrained_model = load_trained_model(pretrained_model_path, device_str="0")

    pretrained_state_dict = pretrained_model.state_dict()

    float_model = QATResNetJ6EDetectionModel(args={"train_3d":True}, nc=37, ch=3, scale='n').cuda()
    detection_head = float_model.model.detect
    detection_head.qat = True 

# yolo11
elif model_name == "yolo11":
    from ultralytics.cfg.models_py.yolo11_wrapper import QATYOLO11DetectionModel
    # pretrained_model_path = "runs/detect_d4q/minieye-driving-d4q-python-model-test14/weights/last.pt"
    # pretrained_model = load_trained_model(pretrained_model_path, device_str="0")

    # pretrained_state_dict = pretrained_model.state_dict()

    float_model = QATYOLO11DetectionModel(args={"train_3d":True}, nc=37, ch=3, scale='s').cuda()

    # float_model.load_state_dict(pretrained_state_dict)

    detection_head = float_model.model[-1]
    detection_head.qat = True


float_model.eval()
# detection_head.dfl = nn.Identity()  # 禁用 DFL 层以简化量化过程

# model.model 访问的是底层的 PyTorch nn.Module
print("\n开始检查并修改模型的inplace属性...")
modified_layers_count = 0
for module in float_model.modules():
    # 检查模块是否有 'inplace' 属性并且其值为 True
    if hasattr(module, 'inplace') and module.inplace:
        # 常见的带有 inplace 参数的层是激活函数，如 SiLU, ReLU 等
        print(f"  - 找到 inplace=True 的层: {module}")
        
        # 将 inplace 属性修改为 False
        module.inplace = False
        modified_layers_count += 1
        print(f"    -> 已修改为 inplace={module.inplace}")

if modified_layers_count > 0:
    print(f"\n修改完成！总共修改了 {modified_layers_count} 个层的 inplace 属性。")
else:
    print("\n检查完成，没有找到需要修改的 inplace=True 的层。")

example_input = torch.rand(1, 3, 384, 960, device="cuda:0")

# output = float_model(example_input)

qat_model = prepare(float_model, example_input, qat_8bit_weight_16bit_act_qconfig_setter)
qat_state = qat_model.state_dict()
# ------------------- 解决方案开始 -------------------
print("\n开始修复量化模型的自定义属性...")
# 1. 获取原始模型中所有带名称的模块
original_named_modules = dict(float_model.model.named_modules())

# 2. 获取量化后模型中所有带名称的模块
qat_named_modules = dict(qat_model.model.named_modules())

# restored_count = 0
# # 3. 遍历原始模型的每一个命名模块
# for name, orig_module in original_named_modules.items():
    
#     # 检查原始模块是否有我们关心的 '.f' 属性
#     if not hasattr(orig_module, 'f'):
#         continue

#     # 在量化模型中查找同名模块
#     # prepare 函数可能会包装原始模块，但通常会保留其名称
#     if name in qat_named_modules:
#         qat_module = qat_named_modules[name]
        
#         # 核心检查：如果量化后的模块丢失了 '.f' 属性
#         if not hasattr(qat_module, 'f'):
#             # 从原始模块恢复丢失的自定义属性
#             qat_module.f = orig_module.f
#             qat_module.i = orig_module.i
#             # 标记类型，方便调试
#             qat_module.type = getattr(orig_module, 'type', 'Unknown') + '_QuantizedWrapper'
            
#             print(f"  - 模块 '{name}' (类型: {type(qat_module).__name__}) 丢失了属性。")
#             print(f"    -> 已恢复属性: f={qat_module.f}, i={qat_module.i}")
#             restored_count += 1
            
# # ------------------- 解决方案结束 -------------------
            
# qat_model_params = torch.load("yolo11s_calib-checkpoint_20251119.ckpt")


qat_model_params = torch.load(f"{model_name}s_calib-checkpoint_20251125.ckpt")

# 分别打印 float_model, qat_model, qat_model_params state_dict()的keys
with open("float_model_keys.txt", "w") as f:
    for k in float_model.state_dict().keys():
        f.write(k + "\n")
with open("qat_model_keys.txt", "w") as f:
    for k in qat_model.state_dict().keys():
        f.write(k + "\n")
# with open("qat_model_params_keys.txt", "w") as f:
#     for k in qat_model_params.keys():
#         f.write(k + "\n")

new_state_dict = OrderedDict()
for k, v in qat_model_params.items():
    name = k[7:] if k.startswith('module.') else k  # 去掉'module.'
    if name in qat_state:
        new_state_dict[name] = v
qat_model.load_state_dict(new_state_dict, strict=True)

hbir_qat_model = horizon_plugin_pytorch.quantization.hbdk4.export(qat_model, example_input)
# hbir_quantized_model = convert(hbir_qat_model, march, advice=False)
save(hbir_qat_model, f"./J6E_qat/compile/{model_name}/qat.bc")

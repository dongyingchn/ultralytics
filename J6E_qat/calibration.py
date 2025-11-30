import os
import torch
from torch.utils import data
from torch.nn import DataParallel
from horizon_plugin_pytorch.march import March, set_march
from horizon_plugin_pytorch.quantization import (
    prepare,
    set_fake_quantize,
    FakeQuantState,
)
from horizon_plugin_pytorch.quantization.qconfig_template import calibration_8bit_weight_16bit_act_qconfig_setter

from fs_net import FsResNet18
from collections import OrderedDict
from fs_dataset import FsSegDataset, FsVehiclePedDataset

###############################################################################
# The user can modify the following parameters as required.
# 1. The batch_size used for Calibration.
calib_batch_size = 32
# 2. The batch_size used for Validation.
eval_batch_size = 8
# 3. The amount of data used by Calibration, configured to inf to use all data.
num_examples = 200 # float("inf")
# 4. Code name of the target hardware platform.
march = March.NASH_E
# 5. Example input for model tracing and export hbir.
example_input = torch.rand(1, 3, 768, 960, device="cuda:0")
################################################################################

# Before model transformation, the hardware platform on which the model will be executed must be set up.
set_march(march)

# load model
# Define Model
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

pretrain = torch.load('./run/Float/experiment_3/2_checkpoint.pth.tar')['state_dict']

new_state_dict = OrderedDict()
for k, v in pretrain.items():
    name = k[7:] if k.startswith('module.') else k  # 去掉'module.'
    new_state_dict[name] = v
float_model.load_state_dict(new_state_dict)
float_model = float_model.to("cuda:0")

# Transform the model into the Calibration state to characterize the numerical distribution of the data at each location statistically.
calib_model = prepare(float_model, example_input, calibration_8bit_weight_16bit_act_qconfig_setter)


# Prepare the dataset.
train_freespace_dataset = FsSegDataset(data_root="/data/parking_fs", 
                                        file_list="train_list_seg_other.txt",
                                        img_size=None,
                                        train_phase=True)
train_vehicle_ped_dataset = FsVehiclePedDataset(data_root="/data/parking_fs", 
                                            file_list="train_list_seg_car.txt",
                                            img_size=None,
                                            train_phase=True)

calib_freespace_loader = data.DataLoader(
        train_freespace_dataset,
        batch_size=calib_batch_size,
        sampler=data.RandomSampler(train_freespace_dataset),
        num_workers=8,
        pin_memory=True)

calib_vehicle_ped_loader = data.DataLoader(
        train_vehicle_ped_dataset,
        batch_size=calib_batch_size,
        sampler=data.RandomSampler(train_vehicle_ped_dataset),
        num_workers=8,
        pin_memory=True)

# Perform Calibration process (no backward required).
# Note the control of the model state here, the model needs to be in the eval state for the behavior of Bn to match the requirements.
calib_model.eval()
set_fake_quantize(calib_model, FakeQuantState.CALIBRATION)
with torch.no_grad():
    cnt = 0
    for sample_fs, sample_vp in zip(calib_freespace_loader, calib_vehicle_ped_loader):

        for k in sample_fs:
             sample_fs[k] = sample_fs[k].to("cuda:0")
        for k in sample_vp:
             sample_vp[k] = sample_vp[k].to("cuda:0")

        calib_model(sample_fs['input'])
        calib_model(sample_vp['input'])

        print(".", end="", flush=True)
        cnt += sample_fs['input'].size(0)
        if cnt >= num_examples:
            break
        print("Calibration {}/{} samples".format(cnt, num_examples))

# Test pseudo-quantization accuracy.
# Note the control of the model state here.
"""
calib_model.eval()
set_fake_quantize(calib_model, FakeQuantState.VALIDATION)
"""


# Saving Calibration Model Parameters.
torch.save(
    calib_model.state_dict(),
    os.path.join("./", "fs_calib-checkpoint_20251119.ckpt"),
)
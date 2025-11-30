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
from horizon_plugin_pytorch.quantization.qconfig_template import calibration_8bit_weight_16bit_act_qconfig_setter, default_calibration_qconfig_setter

from collections import OrderedDict

current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.dirname(current_dir))

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

torch.backends.cudnn.enabled = True

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
example_input = torch.rand(1, 3, 384, 960, dtype=torch.float32, device="cuda:0")
################################################################################

# Before model transformation, the hardware platform on which the model will be executed must be set up.
set_march(march)

# load model
# Define Model
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

# Transform the model into the Calibration state to characterize the numerical distribution of the data at each location statistically.
calib_model = prepare(float_model, example_input, calibration_8bit_weight_16bit_act_qconfig_setter) # calibration_8bit_weight_16bit_act_qconfig_setter

with open("calib_model_keys.txt", "w") as f:
    for k in calib_model.state_dict().keys():
        f.write(k + "\n")

# ------------------- 解决方案开始 -------------------
print("\n开始修复量化模型的自定义属性...")
# 1. 获取原始模型中所有带名称的模块
original_named_modules = dict(float_model.model.named_modules())

# 2. 获取量化后模型中所有带名称的模块
calibrated_named_modules = dict(calib_model.model.named_modules())

# restored_count = 0
# # 3. 遍历原始模型的每一个命名模块
# for name, orig_module in original_named_modules.items():
    
#     # 检查原始模块是否有我们关心的 '.f' 属性
#     if not hasattr(orig_module, 'f'):
#         continue

#     # 在量化模型中查找同名模块
#     # prepare 函数可能会包装原始模块，但通常会保留其名称
#     if name in calibrated_named_modules:
#         calib_module = calibrated_named_modules[name]
        
#         # 核心检查：如果量化后的模块丢失了 '.f' 属性
#         if not hasattr(calib_module, 'f'):
#             # 从原始模块恢复丢失的自定义属性
#             calib_module.f = orig_module.f
#             calib_module.i = orig_module.i
#             # 标记类型，方便调试
#             calib_module.type = getattr(orig_module, 'type', 'Unknown') + '_QuantizedWrapper'
            
#             print(f"  - 模块 '{name}' (类型: {type(calib_module).__name__}) 丢失了属性。")
#             print(f"    -> 已恢复属性: f={calib_module.f}, i={calib_module.i}")
#             restored_count += 1
            
# # ------------------- 解决方案结束 -------------------

# if restored_count > 0:
#     print(f"\n修复完成！总共为 {restored_count} 个模块恢复了自定义属性。")
# else:
#     print("\n检查完成，未发现需要修复的模块。")

# # ------------------- 解决方案结束 -------------------

# Prepare the dataset.
args = dict(
    model=float_model, # pretrained_model_path
    data="minieye-driving-D4Q.yaml",
    epochs=100,
    imgsz=960,
    device="cuda:0"
)

# 3. 创建一个 Trainer 实例
# 对于目标检测任务，我们使用 DetectionTrainer
# 如果是其他任务（如分类、分割），你需要使用对应的 Trainer
# 例如：ClassificationTrainer, SegmentationTrainer
trainer = DetectionTrainer(overrides=args)
# trainer.setup_model()

# 4. 构建数据集
# trainer.build_dataset 方法会根据 'data' 参数加载数据集
# 这里我们只需要数据集路径，trainer内部会处理
data_path = trainer.data['val']

# 5. 获取 dataloader
# get_dataloader 方法接收数据集路径和批次大小
# 'mode' 参数可以是 'train' 或 'val'
train_loader = trainer.get_dataloader_for_qat(dataset_path=data_path, batch_size=16, rank=-1, mode="train")

# 现在你可以使用 train_loader 和 val_loader 了
print(f"成功获取 Dataloader!")
print(f"训练集 Dataloader: {train_loader}")

# 你可以迭代 dataloader 来检查数据
# for i, batch in enumerate(train_loader):
#     print(f"Batch {i}:")
#     # batch 是一个字典，包含了 'img', 'labels' 等信息
#     print(f"  Image batch shape: {batch['img'].shape}")
#     break

# Perform Calibration process (no backward required).
# Note the control of the model state here, the model needs to be in the eval state for the behavior of Bn to match the requirements.
calib_model.eval()
set_fake_quantize(calib_model, FakeQuantState.CALIBRATION)
with torch.no_grad():
    cnt = 0
    for sample in train_loader:

        # for k in sample_fs:
        #      sample_fs[k] = sample_fs[k].to("cuda:0")
        # for k in sample_vp:
        #      sample_vp[k] = sample_vp[k].to("cuda:0")
        sample = trainer.preprocess_batch(sample)
        preds = calib_model(sample['img'])

        print(".", end="", flush=True)
        cnt += sample['img'].size(0)
        if cnt >= num_examples:
            break
        print("Calibration {}/{} samples".format(cnt, num_examples))

# Test pseudo-quantization accuracy.
# Note the control of the model state here.
"""
calib_model.eval()
set_fake_quantize(calib_model, FakeQuantState.VALIDATION)
"""

calib_model.eval()
set_fake_quantize(calib_model, FakeQuantState.VALIDATION)
with torch.no_grad():
    cnt = 0
    for sample in train_loader:

        # for k in sample_fs:
        #      sample_fs[k] = sample_fs[k].to("cuda:0")
        # for k in sample_vp:
        #      sample_vp[k] = sample_vp[k].to("cuda:0")
        sample = trainer.preprocess_batch(sample)
        preds = calib_model(sample['img'])
        cnt += sample['img'].size(0)


# Saving Calibration Model Parameters.
torch.save(
    calib_model.state_dict(),
    os.path.join("./", f"{model_name}s_calib-checkpoint_20251125.ckpt"),
)
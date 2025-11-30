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
from horizon_plugin_pytorch.quantization.qconfig_template import calibration_8bit_weight_16bit_act_qconfig_setter, qat_8bit_weight_16bit_act_qconfig_setter

from collections import OrderedDict

current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.dirname(current_dir))

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.nn.tasks import DetectionModel

from horizon_plugin_pytorch.quantization import (
    QuantStub,
    prepare,
    set_fake_quantize,
    FakeQuantState,
)
from torch.quantization import DeQuantStub

class QATReadyDetectionModel(DetectionModel):
    def __init__(self, cfg):

        object.__setattr__(self, '_initialized', False)
        super().__init__(cfg)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        object.__setattr__(self, '_initialized', True)

    def predict(self, x, *args, **kwargs):
        if self._initialized:
            x = self.quant(x)
        x = super().predict(x, *args, **kwargs)
        if self._initialized:
            if isinstance(x, (list, tuple)):
                for i in range(len(x)):
                    if isinstance(x[i], (list, tuple)):
                        for j in range(len(x[i])):
                            x[i][j] = self.dequant(x[i][j])
                    else:
                        x[i] = self.dequant(x[i])
            else:
                x = self.dequant(x)

        return x

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
pretrained_model_path = "./runs/detect_d4q/minieye-driving-d4q-yolo11s-full_image-all/weights/best.pt"
model = YOLO(pretrained_model_path)

# Perform validation before QAT transformation
model.model.model[-1].qat = True
result = model.val(data="minieye-driving-D4Q-debug.yaml", imgsz=960, rect=True, batch=eval_batch_size)

# Create a QATReadyDetectionModel instance
float_model = QATReadyDetectionModel("yolo11s-minieye-2d.yaml")
# Load pretrained weights
float_model.model.load_state_dict(model.model.model.state_dict())
float_model.model[-1].qat = True
# float_model.training = False
float_model = float_model.to("cuda:0")

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

# Perform Calibration process (no backward required).
# Note the control of the model state here, the model needs to be in the eval state for the behavior of Bn to match the requirements.
calib_model.eval()
set_fake_quantize(calib_model, FakeQuantState.CALIBRATION)

# Prepare the dataset.
args = dict(
    model=pretrained_model_path, 
    data="minieye-driving-D4Q-debug.yaml",
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
train_loader = trainer.get_dataloader_for_qat(dataset_path=data_path, batch_size=16, rank=-1, mode="val")

# 现在你可以使用 train_loader 和 val_loader 了
print(f"成功获取 Dataloader!")
print(f"训练集 Dataloader: {train_loader}")

with torch.no_grad():
    cnt = 0
    for sample in train_loader:
        
        sample = trainer.preprocess_batch(sample)
        calib_model(sample['img'])

        # preds_post = validator.postprocess(preds)

         # Update metrics

        print(".", end="", flush=True)
        cnt += sample['img'].size(0)
        if cnt >= num_examples:
            break
        print("Calibration {}/{} samples".format(cnt, num_examples))

calib_model.eval()
set_fake_quantize(calib_model, FakeQuantState.VALIDATION)

# trainer.test_loader = train_loader
# validator = trainer.get_validator()
# validator.init_metrics(calib_model)

# save_dir = "./J6E_qat/calibration/yolo11s"
# os.makedirs(save_dir, exist_ok=True)
# trainer.save_dir = save_dir

# with torch.no_grad():
#     cnt = 0
#     for sample in train_loader:
        
#         sample = trainer.preprocess_batch(sample)
#         preds = calib_model(sample['img'])

#         cnt += sample['img'].size(0)

#         preds_post = validator.postprocess(preds)
#         # Update metrics
#         validator.update_metrics(preds_post, sample)

# validator.finalize_metrics()
# validator.print_results()

# calib_model.model[-1].qat = False
model.model = calib_model
result = model.val(data="minieye-driving-D4Q-debug.yaml", imgsz=960, rect=True, batch=eval_batch_size)

# Saving Calibration Model Parameters.
torch.save(
    calib_model.state_dict(),
    os.path.join("./", "yolo11s_calib-checkpoint_20251126.ckpt"),
)
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn

# from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, C3, C2f, SPPF, Concat, ConvTranspose
# 请确认 C3k2 和 C2PSA 是否是自定义模块或者在特定版本中存在
# 如果它们是自定义的，需要在这里导入。以下代码假设它们是ultralytics.nn.modules的一部分。
# 如果 C3k2 和 C2PSA 不存在，您可能需要用 C2f 或其他可用模块替换它们。
from ultralytics.nn.modules import C3k2, C2PSA # 假设的导入路径
from ultralytics.nn.modules import Detect, Detect3D

from ultralytics.utils.loss import v8DetectionLoss, v8Detection3DLoss
from ultralytics.utils.ops import make_divisible  # 导入通道对齐函数

from typing import Dict, List

def make_scaler(scale: str = "n"):
    # 对应 yolo11.yaml 的 scales: [depth, width, max_channels]
    scales = {
        "n": (0.50, 0.25, 1024),
        "s": (0.50, 0.50, 1024),
        "m": (0.50, 1.00, 512),
        "l": (1.00, 1.00, 512),
        "x": (1.00, 1.50, 512),
    }
    d, w, max_ch = scales[scale]

    def get_width(ch):
        ch = int(ch * w)
        return min(ch, max_ch)

    def get_depth(r):
        return max(1, round(r * d))

    return get_width, get_depth, max_ch

class YOLO11Backbone(nn.Module):
    """
    返回三个特征图：
        P3: 输出对应原来 b4_save
        P4: 输出对应原来 b6_save
        P5: 输出对应原来 b10_save
    """
    def __init__(self, ch=3, scale="n"):
        super().__init__()
        get_width, get_depth, _ = make_scaler(scale)

        self.b0 = Conv(ch, get_width(64), 3, 2)
        self.b1 = Conv(get_width(64), get_width(128), 3, 2)
        self.b2 = C3k2(get_width(128), get_width(256), n=get_depth(2), shortcut=False)
        self.b3 = Conv(get_width(256), get_width(256), 3, 2)
        self.b4_save = C3k2(get_width(256), get_width(512), n=get_depth(2), shortcut=False)  # P3
        self.b5 = Conv(get_width(512), get_width(512), 3, 2)
        self.b6_save = C3k2(get_width(512), get_width(512), n=get_depth(2), shortcut=True)  # P4
        self.b7 = Conv(get_width(512), get_width(1024), 3, 2)
        self.b8 = C3k2(get_width(1024), get_width(1024), n=get_depth(2), shortcut=True)
        self.b9 = SPPF(get_width(1024), get_width(1024), 5)
        self.b10_save = C2PSA(get_width(1024), get_width(1024), n=get_depth(2))  # P5
        # self.b10_save = C2f(get_width(1024), get_width(1024), n=get_depth(2))  # P5

        # 记录通道数，给 head / detect 使用
        self.c3 = get_width(512)   # P3
        self.c4 = get_width(512)   # P4
        self.c5 = get_width(1024)  # P5

    def forward(self, x):
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        p3 = self.b4_save(x)   # P3

        x = self.b5(p3)
        p4 = self.b6_save(x)   # P4

        x = self.b7(p4)
        x = self.b8(x)
        x = self.b9(x)
        p5 = self.b10_save(x)  # P5

        return p3, p4, p5

class YOLO11Head(nn.Module):
    """
    接收 backbone 的 P3, P4, P5，输出 head 的 P3、P4、P5，用于 Detect。
    """
    def __init__(self, backbone: YOLO11Backbone, scale="n"):
        super().__init__()
        get_width, get_depth, _ = make_scaler(scale)

        c3 = backbone.c3  # 512
        c4 = backbone.c4  # 512
        c5 = backbone.c5  # 1024

        # 11-13
        self.h11_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.h11_upsample = ConvTranspose(c5, c5, k=2, s=2)
        self.h12_concat = Concat(1)
        # x11(=upsample P5): c5; concat with P4: c5 + c4
        self.h13_save = C3k2(c5 + c4, get_width(512), n=get_depth(2), shortcut=False)

        # 14-16
        self.h14_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.h14_upsample = ConvTranspose(get_width(512), get_width(512), k=2, s=2)
        self.h15_concat = Concat(1)
        # x14(=upsample p4_head): get_width(512); concat with P3: get_width(512) + c3
        self.h16_p3_out = C3k2(get_width(512) + c3, get_width(256), n=get_depth(2), shortcut=False)

        # 17-19
        self.h17_downsample = Conv(get_width(256), get_width(256), 3, 2)
        self.h18_concat = Concat(1)
        # x17: get_width(256); concat with p4_head(get_width(512)) => get_width(256) + get_width(512)
        self.h19_p4_out = C3k2(get_width(256) + get_width(512), get_width(512), n=get_depth(2), shortcut=False)

        # 20-22
        self.h20_downsample = Conv(get_width(512), get_width(512), 3, 2)
        self.h21_concat = Concat(1)
        # x20: get_width(512); concat with P5(c5) => get_width(512) + c5
        self.h22_p5_out = C3k2(get_width(512) + c5, get_width(1024), n=get_depth(2), shortcut=True)

        # 显式记录 head 输出通道，给 Detect 用
        self.c3_out = get_width(256)
        self.c4_out = get_width(512)
        self.c5_out = get_width(1024)

    def forward(self, p3_backbone, p4_backbone, p5_backbone):
        # 11: upsample P5
        x11 = self.h11_upsample(p5_backbone)
        # 12: concat with backbone P4
        x12 = self.h12_concat((x11, p4_backbone))
        # 13: head P4
        p4_head = self.h13_save(x12)

        # 14: upsample
        x14 = self.h14_upsample(p4_head)
        # 15: concat with P3
        x15 = self.h15_concat((x14, p3_backbone))
        # 16: P3 out
        p3_out = self.h16_p3_out(x15)

        # 17: downsample
        x17 = self.h17_downsample(p3_out)
        # 18: concat with head P4
        x18 = self.h18_concat((x17, p4_head))
        # 19: P4 out
        p4_out = self.h19_p4_out(x18)

        # 20: downsample
        x20 = self.h20_downsample(p4_out)
        # 21: concat with P5_backbone
        x21 = self.h21_concat((x20, p5_backbone))
        # 22: P5 out
        p5_out = self.h22_p5_out(x21)

        return [p3_out, p4_out, p5_out]

# 2. 直接继承 nn.Module
class YOLO11(nn.Module):
    """
    YOLOv11 object detection model, implemented as a standalone nn.Module.
    This version does NOT inherit from ultralytics.nn.tasks.DetectionModel.
    """

    # 3. 在类级别定义缩放配置
    _scales = {
        # [depth, width, max_channels]
        'n': [0.50, 0.25, 1024],
        's': [0.50, 0.50, 1024],
        'm': [0.50, 1.00, 512],
        'l': [1.00, 1.00, 512],
        'x': [1.00, 1.50, 512],
    }

    def __init__(self, scale='s', ch=3, nc=80):
        """
        Initializes the YOLOv11 model.

        Args:
            cfg (dict, optional): Configuration dictionary. Kept for framework compatibility (i.e. saving metadata).
            ch (int): Number of input channels.
            nc (int): Number of classes.
        """
        super().__init__()

        # 4. 获取缩放因子
        depth_multiple, width_multiple, max_channels = self._scales[scale]

        # 封装一个辅助函数来动态计算重复次数
        def get_depth(n):
            return max(round(n * depth_multiple), 1) if n > 1 else n
        
        # 封装一个辅助函数来动态计算通道数
        def get_width(c):
            return make_divisible(min(c * width_multiple, max_channels), 8)

        # 3. 手动设置所有训练器依赖的属性
        self.yaml = {'scale': scale, 'nc': nc}
        self.nc = nc      # 类别数量 (必须)
        self.ch = ch      # 输入通道
        self.names = {i: f'class_{i}' for i in range(nc)}  # 类别名称 (必须)
        
        # 这个属性至关重要！定义模型每个输出头的步长 (必须)
        self.stride = torch.tensor([8., 16., 32.]) 
        
        # 这个属性会被训练器自动设置，这里先初始化为 None
        self.args = None 
        
        # 损失函数将在 init_criterion 中被初始化 (必须)
        self.criterion = None 

        # 1) Backbone
        self.backbone = YOLO11Backbone(ch=ch, scale=scale)

        # 2) Head
        self.head = YOLO11Head(backbone=self.backbone, scale=scale)

        # 3) Detect
        detect_channels = (self.head.c3_out, self.head.c4_out, self.head.c5_out)
        self.detect = Detect3D(nc=self.nc, ch=detect_channels)

        self._initialize_strides(img_size=640)
        
        # 初始化权重
        self.detect.bias_init()

    def _initialize_strides(self, img_size: int = 640):
        """
        通过一次 dummy forward 推断各检测层 stride，并写入 self.stride/self.detect.stride。
        """
        # 确保在 CPU 上也能跑，不影响之后再 .to(device)
        device = next(self.parameters()).device
        x = torch.zeros(1, 3, img_size, img_size, device=device)

        self.eval()
        with torch.no_grad():
            # 如果你的 forward 直接返回 Detect 的 list 输出，可以直接：outs = self(x)
            # 这里按你之前的实现，backbone 输出 p3, p4, p5，然后 head.detect 接收它们：
            p3, p4, p5 = self.backbone(x)
            _, outs = self.detect(self.head(p3, p4, p5))

        # 根据特征图尺寸计算 stride：stride = 输入尺寸 / 特征图尺寸
        strides = []
        in_size = img_size

        if isinstance(outs, tuple):
            outs = outs[0]

        for o in outs:
            # o: [B, C, H, W]
            s = in_size // o.shape[-1]
            strides.append(s)

        self.stride = torch.tensor(strides, device=device).float()
        if hasattr(self.detect, "stride"):
            self.detect.stride = self.stride

        self.train()  # 恢复训练模式（后面 Trainer 还会再调用 .train()）

    def forward(self, x):
        
        p3, p4, p5 = self.backbone(x)
        p3_out, p4_out, p5_out = self.head(p3, p4, p5)
        preds = self.detect([p3_out, p4_out, p5_out])
        return preds

    # 7. 手动实现 loss 方法 (必须)
    def loss(self, batch: Dict, preds: List[torch.Tensor] = None):
        """
        计算损失。这个方法会被训练器在每个训练迭代中调用。
        """
        if self.criterion is None:
            self.criterion = self.init_criterion()

        # 如果没有预先计算的preds，就通过forward获取
        preds = self(batch['img']) if preds is None else preds
        
        return self.criterion(preds, batch)

    # 8. 手动实现 init_criterion 方法 (必须)
    def init_criterion(self):
        """
        初始化损失函数。这个方法会被训练器在`_setup_train`阶段调用。
        """
        # v8DetectionLoss会从传入的模型实例(self)中获取stride, nc等信息来配置自己
        return v8Detection3DLoss(self)

    # 9. (推荐) 实现权重加载方法
    def load(self, weights_path, verbose=True):
        """从 .pt 文件加载权重。"""
        from ultralytics.nn.tasks import load_checkpoint
        ckpt = load_checkpoint(weights_path, map_location='cpu')['model'].float().state_dict()
        
        # 过滤掉不匹配的层
        csd = {k: v for k, v in ckpt.items() if k in self.state_dict() and self.state_dict()[k].shape == v.shape}
        self.load_state_dict(csd, strict=False)
        
        if verbose:
            print(f'Transferred {len(csd)}/{len(self.state_dict())} items from {weights_path}')

# === 模型测试 (可选，用于独立验证) ===
if __name__ == '__main__':
    from ultralytics.utils.torch_utils import model_info
    
    # 模拟创建模型实例
    model = YOLO11(nc=80)
    model.eval()

    # 打印模型信息
    model_info(model, imgsz=640, verbose=True)
    
    # 创建一个虚拟输入
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # 测试前向传播
    try:
        predictions, (train_outputs) = model(dummy_input)
        print("\nInference test successful!")
        print(f"Inference output shape: {predictions.shape}")
        print("\nTraining mode forward pass successful!")
        print(f"Training output shapes: {[p.shape for p in train_outputs]}")
        
    except Exception as e:
        import traceback
        print(f"\nAn error occurred during forward pass test: {e}")
        traceback.print_exc()
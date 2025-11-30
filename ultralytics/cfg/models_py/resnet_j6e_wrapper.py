import torch
import torch.nn as nn

from ultralytics.nn.tasks import BaseModel  # 复用已有 BaseModel 能力（forward/predict/loss 等）
from ultralytics.utils.loss import v8Detection3DLoss, v8DetectionLoss  # 按 enable_3d 分支选择
from ultralytics.cfg.models_py.resnet_j6e import ResNetJ6eModel  # 你的 Python 版 YOLO11


class ResNetJ6EDetectionModel(BaseModel):
    """
    一个包装类，将自定义 YOLO11 模型“适配”为类似 DetectionModel 的结构，便于：
    - v8Detection3DLoss / v8DetectionLoss 通过 model.model[-1] 访问 Detect3D 头；
    - Trainer / Validator 使用 BaseModel 的 forward/predict/loss 接口；
    - Validator 调用 model(img, augment=...) 时与 BaseModel 保持兼容。
    """

    def __init__(self, args, nc: int, ch: int, scale: str = "n"):
        """
        Args:
            args: trainer.args（SimpleNamespace），其中包含训练超参
            nc: 类别数
            ch: 输入通道数（一般为 3）
            scale: YOLO11 规模，如 'n', 's', 'm', 'l', 'x'
        """
        super().__init__()
        self.args = args
        self.nc = nc
        self.ch = ch
        self.scale = scale

        # 1. 实例化你自己的 YOLO11
        #    假设 YOLO11(nc=..., ch=..., scale=...) 可用；如果你的 __init__ 没有 ch 参数，请去掉 ch=ch。
        
        model = ResNetJ6eModel(n_class=nc)

        # self.stem_conv = model.model.stem_conv
        # self.stage0 = model.model.stage0
        # self.stage1 = model.model.stage1
        # self.stage2 = model.model.stage2

        # self.stage3 = model.model.stage3
        # self.refinenet2 = model.model.refinenet2
        # self.refinenet1 = model.model.refinenet1

        # self.detect = model.model.detect

        # self.model = nn.ModuleList([self.stem_conv, self.stage0, self.stage1, self.stage2, self.stage3, self.refinenet2, self.refinenet1, self.detect])

        self.model = model.model
        # self.detect = self.model.detect  # Detect3D 头

        # 3. 构造一个与 DetectionModel 相似的 self.model（nn.Sequential 或 ModuleList）
        #    注意：BaseModel._predict_once 假定 self.model 是一个“按 yaml parse”后的结构，
        #    但这里我们只需要满足 v8Detection3DLoss 中的 model.model[-1] 访问 Detect 头即可；
        #    不打算复用 BaseModel._predict_once 的 DAG 功能，因此我们自定义 predict。
        #
        #    为了兼容 BaseModel._apply 中的：
        #       m = self.model[-1]
        #       if isinstance(m, Detect or Detect3D):
        #           m.stride = fn(m.stride)
        #           ...
        #    这里我们用 ModuleList/Sequential 均可；不走 _predict_once。
        # self.model = nn.ModuleList([self.backbone, self.head, self.detect])

        # 4. 设置关键属性，供 loss / Trainer / Validator 使用
        # stride：从 core 透传；如果 core 未设置，可以自行初始化为 [8, 16, 32] 或根据实际推断
        self.stride = getattr(self.model, "stride", torch.tensor([8.0, 16.0, 32.0]))
        # enable_3d：让 DetectionModel.init_criterion 风格的逻辑可重用
        self.enable_3d = isinstance(self.model.detect, type(getattr(self.model.detect, "__class__", None))) or self.args.get("train_3d", False)
        # names：Trainer.set_model_attributes 中会覆盖；这里预先给一个默认的
        self.names = {i: f"{i}" for i in range(self.nc)}

        # 5. end2end：如果你的 Detect3D/YOLO11 支持 E2E one2many，可在 core.detect 上加 end2end 标志
        self.end2end = getattr(self.model.detect, "end2end", False)

        # 6. 如果 Detect 头需要 bias_init，并且 stride 已正确初始化，可以在此调用一次
        if hasattr(self.model.detect, "bias_init") and hasattr(self.model.detect, "stride"):
            # 防御：确保 stride 没有 0
            if torch.all(self.model.detect.stride > 0):
                self.model.detect.bias_init()

    # ----------------------------------------------------
    # 预测接口：兼容 BaseModel.forward(x, augment=...) 调用
    # ----------------------------------------------------

    def predict(self, x, profile: bool = False, visualize: bool = False, augment: bool = False, embed=None):
        """
        兼容 BaseModel.predict 接口。

        Args:
            x (torch.Tensor): 输入图像，形状 [B, C, H, W]
            profile (bool): 是否逐层 profile（这里不使用）
            visualize (bool): 是否可视化特征（这里不使用）
            augment (bool): 是否进行 TTA（这里暂不支持，直接忽略）
            embed (list): 需要返回 embedding 的层索引集合（这里不支持，忽略）

        Returns:
            preds: 检测头输出，与原 Ultralytics Detect3D 输出格式保持一致（list[Tensor] 或 Tensor）
        """
        # 如果后续你希望真正支持 TTA，可以在 augment=True 时做多尺度/翻转再合并结果。
        # 目前先简单忽略 augment 参数，走单尺度前向。
        _ = profile, visualize, embed  # 未使用

        # 按你在 YOLO11.forward 中的逻辑：backbone -> head -> detect
        layer_1 = self.model.stem_conv(x)
        layer_1 = self.model.stage0(layer_1)
        layer_2 = self.model.stage1(layer_1)
        layer_3 = self.model.stage2(layer_2)

        path_3 = self.model.stage3(layer_3)
        path_2 = self.model.refinenet2(path_3, layer_3)
        path_1 = self.model.refinenet1(path_2, layer_2)

        output = self.model.detect([path_1, path_2, path_3])
        return output

    # ----------------------------------------------------
    # forward：直接复用 BaseModel.forward（已实现 dict/img 分支）
    # 这里只需确保签名兼容 Validator 调用 (x, augment=...)
    # ----------------------------------------------------

    def forward(self, x, *args, **kwargs):
        """
        覆盖 forward 仅为了显式加入 augment 参数，转发给 BaseModel.forward。

        BaseModel.forward(x, *args, **kwargs) 会：
        - 如果 x 是 dict：return self.loss(x, *args, **kwargs)
        - 否则：return self.predict(x, *args, **kwargs)

        Validator 调用：model(batch["img"], augment=augment)
        """
        return super().forward(x, *args, **kwargs)

    # ----------------------------------------------------
    # 损失函数：适配 v8Detection3DLoss / v8DetectionLoss
    # ----------------------------------------------------

    def init_criterion(self):
        """
        初始化损失函数。

        仿照 DetectionModel.init_criterion：
        - 如果是 end2end：E2EDetectLoss（此处略）
        - 如果 enable_3d：v8Detection3DLoss(self)
        - 否则：v8DetectionLoss(self)

        你当前是 3D 检测（Detect3D + v8Detection3DLoss），这里直接走 v8Detection3DLoss。
        """
        # 如果你有 end2end 结构，可以在此加上 E2EDetectLoss 分支
        if getattr(self, "end2end", False):
            from ultralytics.utils.loss import E2EDetectLoss

            return E2EDetectLoss(self)

        # 3D 检测
        if getattr(self, "enable_3d", False):
            return v8Detection3DLoss(self)

        # 普通 2D 检测
        return v8DetectionLoss(self)

    def loss(self, batch, preds=None):
        """
        复用 BaseModel.loss 逻辑，但显示声明一下类型，方便阅读。
        """
        return super().loss(batch, preds)

class QATResNetJ6EDetectionModel(ResNetJ6EDetectionModel):
    def __init__(self, args, nc: int, ch: int, scale: str = "n"):
        super().__init__(args, nc, ch, scale)

        from horizon_plugin_pytorch.quantization import QuantStub
        from torch.quantization import DeQuantStub

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x, *args, **kwargs):
        x = self.quant(x)
        output = super().forward(x, *args, **kwargs)
        output = self.dequant(output)
        return output
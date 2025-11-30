from ultralytics.cfg.models_py.resnet_j6e_wrapper import ResNetJ6EDetectionModel  # 你的封装类，或直接 YOLO11


def build_model(args, data):
    """
    Factory function to build YOLO11 model.

    Args:
        args: trainer.args (SimpleNamespace) with training hyperparameters.
        data: dict with dataset info, must contain 'nc' and 'channels'.

    Returns:
        nn.Module: model instance, e.g. YOLO11DetectionModel
    """
    nc = data["nc"]
    ch = data["channels"]
    # 可选：从 args 里读一个自定义 scale，例如 model_scale='n/s/m/...'
    scale = getattr(args, "model_scale", "n")
    # 根据你的实现调整构造签名
    return ResNetJ6EDetectionModel(args=args, nc=nc, ch=ch, scale=scale)
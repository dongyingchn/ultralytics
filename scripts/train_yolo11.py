import torch
import os

# import sys
# current_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.append("/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/")  # 添加上级目录到路径
# print(sys.path)

from ultralytics.models.yolo.detect.train import DetectionTrainer
# from ultralytics.cfg.models_py.yolo11 import YOLO11
# from ultralytics.cfg.models_py.yolo11_wrapper import YOLO11DetectionModel
# from ultralytics.cfg.models_py.resnet_j6e_wrapper import ResNetJ6EDetectionModel

from ultralytics import SETTINGS

from flops import measure_with_thop

SETTINGS["tensorboard"] = True

def main():
    # 训练参数，根据你当前项目自己调整
    overrides = dict(
        task="detect",                 # 任务类型
        mode="train",
        model="/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/ultralytics/cfg/models_py/yolo11_interface.py",                    # 关键：让 BaseTrainer 不去解析 .pt/.yaml
        data="minieye-driving-D4Q.yaml",             # 或你的自定义数据配置
        epochs=200,
        imgsz=960,
        batch=512,
        workers=6,
        device=[0,1,2,3,4,5,6,7],                      # '0' / '0,1' / 'cpu' 等
        # 下面任选，根据你现有 config 习惯补充 / 覆盖
        lr0=0.001,
        lrf=0.01,
        bgr=1,
        optimizer="AdamW",
        cos_lr=False,
        rect=True,
        project='runs/detect_d4q',
        name='minieye-driving-d4q-yolo11s_c2psa_upsample-full_image-all-200epochs-fix_dataset',
        train_3d=True,
        resume=False,
    )

    # 初始化 DetectionTrainer（这一步里 self.model 还只是 None）
    trainer = DetectionTrainer(overrides=overrides)

    # 用你自己的 YOLO11 Python 模型替换掉 trainer.model
    # 注意：要用数据集的 nc 和通道数初始化
    # nc = trainer.data["nc"]
    # ch = trainer.data["channels"]  # 一般是 3
    # # 如果你的 YOLO11 构造函数只需要 nc、scale，可以忽略 ch
    # # model = YOLO11DetectionModel(args=trainer.args, nc=nc, ch=ch, scale="n")
    # model = ResNetJ6EDetectionModel(args=trainer.args, nc=nc, ch=ch, scale="n")

    # # 把模型挂到 trainer 上
    # trainer.model = model

    # 现在开始训练：BaseTrainer.setup_model() 会检测到 self.model 已是 nn.Module，直接跳过 cfg 加载
    trainer.train()

def eval():
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import select_device, TORCH_2_4, ModelEMA
    from ultralytics.utils.checks import check_amp

    overrides = dict(
        task="detect",                 # 任务类型
        mode="eval",
        model="/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/ultralytics/cfg/models_py/yolo11_interface.py",                    # 关键：让 BaseTrainer 不去解析 .pt/.yaml
        data="minieye-driving-D4Q.yaml",             # 或你的自定义数据配置
        epochs=100,
        imgsz=960,
        batch=64,
        workers=6,
        device=[0],                      # '0' / '0,1' / 'cpu' 等
        # 下面任选，根据你现有 config 习惯补充 / 覆盖
        lr0=0.001,
        optimizer="AdamW",
        rect=True,
        project='runs/detect_d4q',
        name='minieye-driving-d4q-yolo11s_c2f_deconv-full_image-all',
        train_3d=True,
        resume=False,
    )

    # 初始化 DetectionTrainer（这一步里 self.model 还只是 None）
    trainer = DetectionTrainer(overrides=overrides)
    # trainer._setup_ddp()
    # trainer._setup_train()  # 准备训练相关属性（data、batch_size 等）

    # ckpt = trainer.setup_model()  # 如果 overrides.model 指向 .pt，会返回 ckpt（或 None）

    # 把模型移到 trainer.device（setup_model 有时已做，但保险起见）
    # trainer.model = trainer.model.to(trainer.device).eval()

    # 手动加载 checkpoint => 得到 model（ema 或 model
    model_path = "/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/runs/detect_d4q/minieye-driving-d4q-yolo11s_c2f_deconv-full_image-all/weights/best.pt"
    model, ckpt = load_checkpoint(model_path, device=select_device(trainer.args.device, trainer.args.batch))

    # 注入 trainer 并准备
    trainer.model = model.eval().to(trainer.device)
    trainer.set_model_attributes()

    # 下面是补上 EMA 的代码：
    trainer.ema = ModelEMA(trainer.model)  # 创建 EMA 管理器，内部会有 .ema 属性指向模型副本

    # 如果 checkpoint 中包含 EMA 模型（Ultralytics save 会有 ckpt['ema']），优先加载它：
    if ckpt is not None and "ema" in ckpt and ckpt["ema"] is not None:
        # ckpt["ema"] 是一个模型对象（float），我们把权重载入到 trainer.ema.ema
        trainer.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
        # 恢复更新计数（可选）
        trainer.ema.updates = ckpt.get("updates", getattr(trainer.ema, "updates", 0))

    # 5) 初始化 AMP 标志和 scaler（validator 要用）
    trainer.amp = bool(check_amp(trainer.model))  # check_amp 返回能否使用 amp
    # set scaler consistent with ultralytics: use new API when TORCH_2_4 else legacy
    trainer.scaler = (
        torch.amp.GradScaler("cuda", enabled=trainer.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=trainer.amp)
    )

    # 4) 手动创建 test_loader（与 _setup_train 中逻辑一致）
    #    注意：val loader batch_size 通常是 train batch 的 2 倍（DetectionTrainer 的实现中）
    batch_size = trainer.batch_size if trainer.args.task == "obb" else trainer.batch_size * 2
    trainer.test_loader = trainer.get_dataloader(
        trainer.data.get("val") or trainer.data.get("test"),
        batch_size=batch_size,
        rank=-1,
        mode="val",
    )

    # 初始化 criterion（loss 计算器）
    if not hasattr(trainer, "criterion") or trainer.criterion is None:
        trainer.criterion = trainer.model.init_criterion()

    device = select_device(trainer.args.device)  # e.g. "cuda:0" or "cpu"
    # 用 test_loader 的第一个 batch 试算一次 loss 来得到 loss_items 的 shape
    try:
        it = iter(trainer.test_loader)
        batch = next(it)
        # 将 batch 的 tensors 移动到 device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        # 预测并计算 loss -> 得到 (loss, loss_items)
        with torch.no_grad():
            batch = trainer.preprocess_batch(batch)
            preds = trainer.model(batch["img"].to(device))
            loss_out = trainer.criterion(preds, batch)
        if isinstance(loss_out, tuple) and len(loss_out) >= 2:
            loss_tensor, loss_items = loss_out[0], loss_out[1]
        else:
            # If criterion returns scalar loss only
            loss_tensor = loss_out if torch.is_tensor(loss_out) else torch.tensor(loss_out, device=device)
            loss_items = torch.tensor([loss_tensor.item()], device=device)
    except Exception as e:
        print(f"试算 loss 时出错：{e}")
        loss_tensor = torch.tensor(0.0, device=device)
        loss_items = torch.tensor([0.0], device=device)

    # set trainer fields expected by validate()
    trainer.loss = loss_tensor.sum() if torch.is_tensor(loss_tensor) else torch.tensor(loss_tensor, device=device)
    trainer.loss_items = loss_items

    class SimpleStopper:
        """Minimal stopper compatible with validator usage."""
        def __init__(self):
            self.stop = False           # validator/Trainer 读取以判断是否提前停止
            self.best_fitness = 0.0     # 用于记录最优 fitness
            self.best_results = None    # 可选：记录最优 metrics
            self.patience = None        # 可选：early stop patience
            self._history = []
            self.possible_stop  = True

        def update(self, metrics):
            """
            Called by training loop / validator to report current metrics.
            metrics: whatever validator passes (typically a sequence or dict).
            """
            # 如果你希望自动计算 fitness，可在这里实现。为保险起见这里只记录。
            self._history.append(metrics)
            # 示例：更新 best_fitness if metrics is (fitness,):
            try:
                # 若 metrics 是 list/tuple 且最后一个元素为 fitness
                fitness = metrics if isinstance(metrics, (int, float)) else (metrics[-1] if isinstance(metrics, (list,tuple)) else None)
                if fitness is not None and fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_results = metrics
            except Exception:
                pass

        def should_stop(self):
            return self.stop

    # attach to trainer
    trainer.stopper = SimpleStopper()

    # # 创建 validator 并运行验证
    trainer.validator = trainer.get_validator()
    metrics, fitness = trainer.validate()
    # metrics = trainer.validator()
    print(metrics, fitness)


if __name__ == "__main__":
    main()
    # eval()
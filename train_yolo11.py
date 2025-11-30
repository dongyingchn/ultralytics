import torch
import os

# import sys
# current_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.append("/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/")  # 添加上级目录到路径
# print(sys.path)

from ultralytics.models.yolo.detect.train import DetectionTrainer
# from ultralytics.cfg.models_py.yolo11 import YOLO11
# from ultralytics.cfg.models_py.yolo11_wrapper import YOLO11DetectionModel
from ultralytics.cfg.models_py.resnet_j6e_wrapper import ResNetJ6EDetectionModel

def main():
    # 训练参数，根据你当前项目自己调整
    overrides = dict(
        task="detect",                 # 任务类型
        mode="train",
        model="/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/ultralytics/cfg/models_py/resnet_j6e_socket.py",                    # 关键：让 BaseTrainer 不去解析 .pt/.yaml
        data="minieye-driving-D4Q.yaml",             # 或你的自定义数据配置
        epochs=100,
        imgsz=960,
        batch=16, #512,
        workers=6,
        device=[0,1],                      # '0' / '0,1' / 'cpu' 等
        # 下面任选，根据你现有 config 习惯补充 / 覆盖
        lr0=0.001,
        optimizer="AdamW",
        rect=True,
        project='runs/detect_d4q',
        name='minieye-driving-d4q-resnetj6e-full_image-all',
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


if __name__ == "__main__":
    main()
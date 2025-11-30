import torch
import torch.nn as nn
from torch.nn import init
from ultralytics import YOLO
from ultralytics import SETTINGS

from flops import measure_with_thop

SETTINGS["tensorboard"] = True

# model = YOLO("/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/yolo11n.pt")  # load a pretrained model
model = YOLO("yolo11s-minieye-2d.yaml")  # build a new model from scratch
# model = YOLO("/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/runs/detect_d4q/minieye-driving-d4q-yolo11s-full_image-all-lr0.002-depth50/weights/last.pt")  # build a new model from scratch

measure_with_thop(model, input_size=(384,960), device='cuda' if torch.cuda.is_available() else 'cpu')

# results = model.train(data="minieye-driving-2d.yaml", epochs=100, imgsz=960, batch=96, device=[0,1,2,3,4,5])  # train the model
# results = model.train(data="minieye-driving-2d.yaml", epochs=100, imgsz=960, batch=32, device=[6,7])  # train the model

def init_weights(net, init_type='normal', init_gain=0.02):
    """初始化网络权重
    
    Args:
        net: 网络模型
        init_type: 初始化类型: normal | xavier | kaiming | orthogonal
        init_gain: 初始化的缩放因子
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'初始化方法 [{init_type}] 未实现')
            
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)
    print(f'使用 {init_type} 初始化网络权重')

def load_partial_weights(model, ckpt_path=None, migrate_head_conv=False):
    """
    加载预训练权重并忽略 shape 不匹配。
    可选：迁移原头部中间卷积到新 cv4 的前两层。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # yolo11n.pt 中通常包含 'model' 或 'ema'；新版本多用 'model' 或 'ema'
    sd = ckpt.get("model")  # 兼容不同存储键
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()

    model_sd = model.state_dict()
    loaded, skipped = [], []

    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded.append(k)
        else:
            skipped.append(k)

    model.load_state_dict(model_sd, strict=False)
    print(f"[INFO] Loaded {len(loaded)} keys, skipped {len(skipped)} keys (shape mismatch or not found).")
    print(f"[INFO] skipped keys: {skipped}")

    # 迁移头部中间卷积权重到 cv4（可选）
    if migrate_head_conv:
        old_head_prefix = "model.{}.m".format(len(model.model)-1)  # 原 head 常见命名
        # 实际 key 例子：model.22.m.0.cv2.conv.weight 等
        # 新 head cv4[i][0].conv / cv4[i][1].conv
        with torch.no_grad():
            for scale_i, seq in enumerate(model.model[-1].cv4):
                # seq: Sequential(Conv, Conv, final Conv)
                first_conv = seq[0].conv
                second_conv = seq[1].conv

                # 找原 head 对应尺度的一二层卷积（如果命名一致）
                k1 = f"{old_head_prefix}.{scale_i}.cv2.conv.weight"
                k1b = f"{old_head_prefix}.{scale_i}.cv2.conv.bias"
                k2 = f"{old_head_prefix}.{scale_i}.cv3.conv.weight"
                k2b = f"{old_head_prefix}.{scale_i}.cv3.conv.bias"

                # 如果形状兼容就拷贝
                if k1 in sd and sd[k1].shape == first_conv.weight.shape:
                    first_conv.weight.copy_(sd[k1])
                    if first_conv.bias is not None and k1b in sd and sd[k1b].shape == first_conv.bias.shape:
                        first_conv.bias.copy_(sd[k1b])
                if k2 in sd and sd[k2].shape == second_conv.weight.shape:
                    second_conv.weight.copy_(sd[k2])
                    if second_conv.bias is not None and k2b in sd and sd[k2b].shape == second_conv.bias.shape:
                        second_conv.bias.copy_(sd[k2b])

    init_weights(model.model[-1].cv3, init_type='kaiming')
    init_weights(model.model[-1].cv4, init_type='kaiming')
    # 初始化新增 3D 输出层（cv4 每个尺度最后一个 1×1 conv）
    # for seq in model.model[-1].cv4:
    #     out_conv = seq[-1]
    #     nn.init.kaiming_normal_(out_conv.weight, mode="fan_out", nonlinearity="linear")
    #     if out_conv.bias is not None:
    #         nn.init.zeros_(out_conv.bias)

    return model

# model.model = load_partial_weights(
#     model.model,
#     ckpt_path="/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/yolo11s.pt",
#     migrate_head_conv=False,
# )

results = model.train(data="minieye-driving-D4Q.yaml", imgsz=960, 
                    batch=512, device=[0,1,2,3,4,5,6,7], 
                    # batch=8, device=[0,1], 
                    workers=6,
                    epochs=300, 
                    lr0=0.01,
                    lrf=0.001,
                    cls=0.8, # original 0.5
                    bgr=1,
                    optimizer='SGD',
                    cos_lr=True,
                    rect=True,
                    project='runs/detect_d4q',
                    name='minieye-driving-d4q-yolo11s-full_image-all-lr0.01-depth50-cls0.8',
                    train_3d=True,
                    resume=False,)  # train the model

# test two ROI
# results = model.train(data="minieye-driving-D4Q.yaml", epochs=100, imgsz=960, 
#                       batch=512, device=[0,1,2,3,4,5,6,7], 
#                     #   batch=16, device=[0], 
#                       workers=4,
#                       optimizer='AdamW', lr0=0.001, rect=True,
#                       project='runs/detect_d4q',
#                       name='minieye-driving-d4q-full_image-all',
#                       train_3d=True,
#                       resume=False,)  # train the model
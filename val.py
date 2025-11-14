import torch
import torch.nn as nn
from torch.nn import init
from ultralytics import YOLO

# model = YOLO("/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/yolo11n.pt")  # load a pretrained model
model = YOLO("/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/runs/detect/train16/weights/best.pt")  # build a new model from scratch

# results = model.train(data="minieye-driving-2d.yaml", epochs=100, imgsz=960, batch=96, device=[0,1,2,3,4,5])  # train the model
# results = model.train(data="minieye-driving-2d.yaml", epochs=100, imgsz=960, batch=32, device=[6,7])  # train the model

results = model.val(data="minieye-driving-2d.yaml", imgsz=960, rect=True)  # train the model
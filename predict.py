import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("/data1/ydong/projects/monocular_3d_object_detection/ultralytics/runs/detect/train4/weights/best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images

img_path = "/mnt/mono3d/ydong_data/Detection/2Ddetection/images/val/2021-06-04-08-46-47_0_rear_Cam_1_fish_camera_encoded_mbuf_72_3456.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (960, 540))
results = model.predict(source=img, save=True, save_txt=True)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result_train4.jpg")  # save to disk
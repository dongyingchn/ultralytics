import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/runs/detect/train10/weights/best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images

img_path = "/mnt/mono3d/xdzhu_data/Mono3d/Mono3d_4face_2m_g1m3/driving/G1M3_FDL2232/20250912/images/G1M3_FDL2232_20250912_seq_53_camera4_002777_68656.jpg"
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
    # result.show()  # display to screen
    result.save(filename="result_train10.jpg")  # save to disk
import os
import cv2
import numpy as np

import random
from tqdm import tqdm

def get_image_list():
    random.seed(42)

    data_dir = "/mnt/mono3d/xdzhu_data/Mono3d/Mono3d_4face_2m_g1m3/driving/G1M3_FDL2232/20250912"
    data_dir = "/mnt/mono3d/swji_data/mono3d_test/mono3d/driving"

    vehicle_names = os.listdir(data_dir)
    vehicle_names.sort()

    train_list = []
    val_list = []

    for vehicle_name in vehicle_names:
        if vehicle_name.startswith("."):
            continue
        vehicle_dir = os.path.join(data_dir, vehicle_name)
        if not os.path.isdir(vehicle_dir):
            continue
        
        date_names = os.listdir(vehicle_dir)
        date_names.sort()
        for date_name in date_names:
            if date_name.startswith("."):
                continue
            date_dir = os.path.join(vehicle_dir, date_name)
            if not os.path.isdir(date_dir):
                continue

            image_dir = os.path.join(date_dir, "images")
            img_list = os.listdir(image_dir)
            img_list.sort()
            img_list = [img_name for img_name in img_list if img_name.endswith(".jpg")]
            random.shuffle(img_list)

            num_all = len(img_list)
            num_train = int(num_all * 0.9)
            
            for i in tqdm(range(num_all), desc=f"Processing {vehicle_name}/{date_name}"):
                img_name = img_list[i]
                img_path = os.path.join(image_dir, img_name)

                if i < num_train:
                    train_list.append(img_path)
                else:
                    val_list.append(img_path)

    save_dir = os.path.join("./train_data/D4Q")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_txt_path = os.path.join(save_dir, f"train_20251114_{len(train_list)}.txt")
    with open(train_txt_path, "w") as f:
        for img_path in train_list:
            f.write(f"{img_path}\n")
    print(f"Total training images: {len(train_list)}")

    val_txt_path = os.path.join(save_dir, f"val_20251114_{len(val_list)}.txt")
    with open(val_txt_path, "w") as f:
        for img_path in val_list:
            f.write(f"{img_path}\n")
    print(f"Total validation images: {len(val_list)}")

def yuv444_bt601_full_range2rgb(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[1.000, 0.000, 1.402],[1.000, -0.344136, -0.714136],[1.000, 1.772, 0.000]])
    bias = np.array([0, 128, 128])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.minimum(np.maximum(np.matmul(trans_mat, src_seq-bias_seq),0),255)
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3))).astype(np.uint8)
    return dst

def check_annos():
    img_path = "/mnt/mono3d/swji_data/mono3d_test/mono3d/driving/D4Q_51/20250712/images/D4Q_51_20250712_seq_95_camera4_000632_80721.jpg"
    img_yuv444 = cv2.imread(img_path)
    h,w = img_yuv444.shape[:2]
    img = yuv444_bt601_full_range2rgb(img_yuv444, w, h)
    img = img[:,:,::-1].copy()

    label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        line = line.split(" ")
        cls = line[0]
        xc_norm, yc_norm = float(line[1]), float(line[2])
        w_norm, h_norm = float(line[3]), float(line[4])

        x1, y1 = int((xc_norm - w_norm / 2) * w), int((yc_norm - h_norm / 2) * h)
        x2, y2 = int((xc_norm + w_norm / 2) * w), int((yc_norm + h_norm / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(cls), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    img_name = img_path.split("/")[-1]
    cv2.imwrite(img_name.replace(".jpg", "_annos.jpg"), img)

if __name__ == "__main__":
    # get_image_list()
    check_annos()
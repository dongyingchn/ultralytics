import os
import cv2
import numpy as np

import random
from tqdm import tqdm

def get_image_list():
    random.seed(42)

    data_dir = "/mnt/mono3d/xdzhu_data/Mono3d/Mono3d_4face_2m_g1m3/driving/G1M3_FDL2232/20250912"
    data_dir = "/mnt/mono3d/swji_data/mono3d_test/mono3d/driving"
    data_dir = "/mnt/mono3d/swji_data/Mono3d_4face_8m_d4q_bt601full/driving"
    data_dir = "/mnt/mono3d/swji_data/mono3d_test/test_data_20251128"

    vehicle_names = os.listdir(data_dir)
    vehicle_names.sort()

    train_list = []
    val_list = []

    train_ratio = 0.0

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
            if date_name not in ['20250530']:
                continue
            date_dir = os.path.join(vehicle_dir, date_name)
            if not os.path.isdir(date_dir):
                continue

            image_dir = os.path.join(date_dir, "images")
            img_list = os.listdir(image_dir)
            img_list.sort()
            img_list = [img_name for img_name in img_list if img_name.endswith(".jpg")]
            # random.shuffle(img_list)

            num_all = len(img_list)
            num_train = int(num_all * train_ratio)
            
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

    # train_txt_path = os.path.join(save_dir, f"train_20250530_{len(train_list)}.txt")
    # with open(train_txt_path, "w") as f:
    #     for img_path in train_list:
    #         f.write(f"{img_path}\n")
    # print(f"Total training images: {len(train_list)}")

    val_txt_path = os.path.join(save_dir, f"val_D4Q_33_20250530_{len(val_list)}.txt")
    with open(val_txt_path, "w") as f:
        for img_path in val_list:
            f.write(f"{img_path}\n")
    print(f"Total validation images: {len(val_list)}")

def get_image_list_2d():
    random.seed(42)

    data_dir = "/mnt/mono3d/swji_data/mono3d_test/Detection"
    
    train_list = []
    val_list = []
    
    image_dir = os.path.join(data_dir, "images")
    img_list = os.listdir(image_dir)
    img_list.sort()
    img_list = [img_name for img_name in img_list if img_name.endswith(".jpg")]
    random.shuffle(img_list)

    num_all = len(img_list)
    num_train = int(num_all * 0.9)
    
    for i in tqdm(range(num_all), desc=f"Processing"):
        img_name = img_list[i]
        img_path = os.path.join(image_dir, img_name)

        if i < num_train:
            train_list.append(img_path)
        else:
            val_list.append(img_path)

    save_dir = os.path.join("./train_data/D4Q")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_txt_path = os.path.join(save_dir, f"train_20251116_{len(train_list)}.txt")
    with open(train_txt_path, "w") as f:
        for img_path in train_list:
            f.write(f"{img_path}\n")
    print(f"Total training images: {len(train_list)}")

    val_txt_path = os.path.join(save_dir, f"val_20251116_{len(val_list)}.txt")
    with open(val_txt_path, "w") as f:
        for img_path in val_list:
            f.write(f"{img_path}\n")
    print(f"Total validation images: {len(val_list)}")

def combine_2d_and_3d():
    train_file_3d = "./train_data/D4Q/train_20251114_195583.txt"
    train_file_2d = "./train_data/D4Q/train_20251116_44640.txt"

    val_file_3d = "./train_data/D4Q/val_20251114_21732.txt"
    val_file_2d = "./train_data/D4Q/val_20251116_4960.txt"

    combined_lines = []
    train_file_combined = "./train_data/D4Q/train_all.txt"
    for file_path in [
        train_file_3d,
        train_file_2d,
        val_file_3d,
        val_file_2d,
    ]:
        with open(file_path, "r") as f:
            lines = f.readlines()
        combined_lines += lines

    with open(train_file_combined, "w") as f:
        f.writelines(combined_lines)

if __name__ == "__main__":
    # get_image_list()
    # get_image_list_2d()
    combine_2d_and_3d()
import os
import random
from tqdm import tqdm

random.seed(42)

data_dir = "/mnt/mono3d/xdzhu_data/Mono3d/Mono3d_4face_2m_g1m3/driving/G1M3_FDL2232/20250912"
image_dir = os.path.join(data_dir, "images")
img_list = os.listdir(image_dir)
img_list.sort()
img_list = [img_name for img_name in img_list if img_name.endswith(".jpg")]
random.shuffle(img_list)

num_all = len(img_list)
num_train = int(num_all * 0.9)

save_dir = os.path.join("./train_data")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_txt_path = os.path.join(save_dir, "train.txt")
with open(train_txt_path, "w") as f:
    for i in tqdm(range(num_train)):
        img_name = img_list[i]
        img_path = os.path.join(image_dir, img_name)
        f.write(f"{img_path}\n")

print(f"Total images: {num_all}, Training images: {num_train}")


val_txt_path = os.path.join(save_dir, "val.txt")
with open(val_txt_path, "w") as f:
    for i in tqdm(range(num_train, num_all)):
        img_name = img_list[i]
        img_path = os.path.join(image_dir, img_name)
        f.write(f"{img_path}\n")
print(f"Validation images: {num_all - num_train}")
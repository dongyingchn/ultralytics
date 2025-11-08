import os
import cv2
import random
import shutil
from tqdm import tqdm

def vis_anno():
    random.seed(0)

    data_dir = "/mnt/mono3d/ydong_data/Detection/2Ddetection"
    image_dir = os.path.join(data_dir, "images", "val")
    anno_dir = os.path.join(data_dir, "labels", "val")

    img_list = os.listdir(image_dir)
    img_list.sort()
    random.shuffle(img_list)

    for img_name in img_list[:20]:
        img_path = os.path.join(image_dir, img_name)
        anno_path = os.path.join(anno_dir, img_name.replace(".jpg", ".txt"))

        if not os.path.exists(anno_path):
            print(f"{img_name} has no annotation")
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(anno_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                if len(line) != 6:
                    print(f"{img_name} has wrong annotation format")
                    break
                cls, cx, cy, wn, hn, attr = map(float, line)
                x1 = int((cx - wn / 2) * w)
                y1 = int((cy - hn / 2) * h)
                x2 = int((cx + wn / 2) * w)
                y2 = int((cy + hn / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, str(int(cls))+' '+str(int(attr)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("img", img)
        cv2.waitKey(0)

def anno_statistics():
    data_dir = "/mnt/mono3d/xdzhu_data/Detection/2Ddetection"
    anno_dir = os.path.join(data_dir, "labels")
    anno_list = os.listdir(anno_dir)
    anno_list.sort()

    cls_dict = {}
    for anno_name in tqdm(anno_list):
        anno_path = os.path.join(anno_dir, anno_name)
        with open(anno_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                cls = int(line[0])
                if cls not in cls_dict:
                    cls_dict[cls] = 0
                cls_dict[cls] += 1

    for cls, count in cls_dict.items():
        print(f"Class {cls}: {count} instances")

def gen_tmp_dataset():
    data_dir = "/mnt/dst/mono3d/xdzhu_data/Detection/2Ddetection"
    image_dir = os.path.join(data_dir, "images")
    anno_dir = os.path.join(data_dir, "labels")

    save_dir = "/mnt/dst/mono3d/ydong_data/Detection/2Ddetection"
    save_image_dir = os.path.join(save_dir, "images")
    save_anno_dir = os.path.join(save_dir, "labels")

    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_anno_dir, exist_ok=True)

    anno_list = os.listdir(anno_dir)
    anno_list.sort()

    cls_dict = {}
    for anno_name in tqdm(anno_list):
        anno_path = os.path.join(anno_dir, anno_name)

        with open(anno_path, "r") as f:

            cls_unique = []
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                cls = int(line[0])

                if cls not in cls_unique:
                    cls_unique.append(cls)
                else:
                    continue

                if cls not in cls_dict:
                    cls_dict[cls] = []
                
                if anno_path not in cls_dict[cls]:
                    cls_dict[cls].append(anno_path)

    for key in range(14, 0, -1):
        anno_list = cls_dict[key]
        random.shuffle(anno_list)

        num = 0
        for anno_path in tqdm(anno_list, desc=f"processing class {key}"):
            anno_name = os.path.basename(anno_path)
            save_anno_path = os.path.join(save_anno_dir, anno_name)
            if not os.path.exists(save_anno_path):
                shutil.copy(anno_path, save_anno_path)
                save_image_path = os.path.join(save_image_dir, anno_name.replace(".txt", ".jpg"))
                shutil.copy(os.path.join(image_dir, anno_name.replace(".txt", ".jpg")), save_image_path)
                num += 1

            if num >= 5000:
                break

def split_train_val():
    random.seed(0)
    data_dir = "/mnt/dst/mono3d/ydong_data/Detection/2Ddetection"
    image_dir = os.path.join(data_dir, "images")
    anno_dir = os.path.join(data_dir, "labels")

    train_img_dir = os.path.join(image_dir, "train")
    val_img_dir = os.path.join(image_dir, "val")
    train_anno_dir = os.path.join(anno_dir, "train")
    val_anno_dir = os.path.join(anno_dir, "val")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_anno_dir, exist_ok=True)
    os.makedirs(val_anno_dir, exist_ok=True)

    img_list = os.listdir(image_dir)
    img_list.sort()
    img_list = [img for img in img_list if img.endswith(".jpg")]

    random.shuffle(img_list)

    for idx, img_name in enumerate(img_list):
        if idx < 5000:
            shutil.move(os.path.join(image_dir, img_name), os.path.join(val_img_dir, img_name))
            shutil.move(os.path.join(anno_dir, img_name.replace(".jpg", ".txt")), os.path.join(val_anno_dir, img_name.replace(".jpg", ".txt")))
        else:
            shutil.move(os.path.join(image_dir, img_name), os.path.join(train_img_dir, img_name))
            shutil.move(os.path.join(anno_dir, img_name.replace(".jpg", ".txt")), os.path.join(train_anno_dir, img_name.replace(".jpg", ".txt")))

if __name__ == "__main__":
    vis_anno()
    # anno_statistics()
    # gen_tmp_dataset()
    # split_train_val()
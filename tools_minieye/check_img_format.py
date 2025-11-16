# import imghdr

data_file = "/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/train_data/D4Q/val_20251114_21732.txt"

with open(data_file, "r") as f:
    img_paths = f.readlines()

for img_path in img_paths[:]:
    img_path = img_path.strip()
    
    with open(img_path,'rb') as f:
        f.seek(-2, 2)
        tail = f.read()

    if tail != b"\xff\xd9":
        print(tail, img_path)  # 正常应为 b'\xff\xd9'
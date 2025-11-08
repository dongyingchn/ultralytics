import os
import cv2
import json
import logging
from tqdm import tqdm

def visualize_2d_3d_asso_res():
    vehicle_name = "D4Q_51"
    date_name = "20250712"
    data_dir = f"/mnt/mono3d/ydong_data/Detection/Mono3D/{vehicle_name}/{date_name}"

    save_dir = f"/data1/dongying/Mono3d/visualization/{vehicle_name}/{date_name}"

    seq_list = os.listdir(data_dir)
    seq_list.sort()

    for seq_name in seq_list[::3]:
        seq_dir = os.path.join(data_dir, seq_name)

        img_dir = os.path.join(seq_dir, "images", "camera4")
        anno_dir = os.path.join(seq_dir, "annotations")

        img_list = os.listdir(img_dir)
        img_list.sort()

        save_seq_dir = os.path.join(save_dir, seq_name)
        if not os.path.exists(save_seq_dir):
            os.makedirs(save_seq_dir)

        for img_name in tqdm(img_list[:], desc=f"Processing sequence {seq_name}"):
            img_path = os.path.join(img_dir, img_name)
            anno_path = os.path.join(anno_dir, img_name.replace(".jpg", ".json"))

            if not os.path.exists(anno_path):
                print(f"Annotation file not found for image: {img_name}")
                continue

            with open(anno_path, "r") as f:
                annotations = json.load(f)

            img = cv2.imread(img_path)

            asso_list = annotations.get("asso_list", [])
            if len(asso_list) == 0:
                logging.info(f"No associations found in annotation for image: {img_name}")
                continue

            for asso in asso_list[:]:
                camera_mea = asso.get("camera_mea", {})
                lidar_mea = asso.get("lidar_mea")
                if lidar_mea is None:
                    id = -1
                else:
                    id = lidar_mea[8]

                cls = camera_mea.get("cls")
                subcls = camera_mea.get("subcls")

                # if cls not in ['vehicle', 'pedestrian', 'bicycle', "bicyclis"]: #['kVehicle', 'kPed', 'kBike', 'kCyclist']:
                #     continue
                # if subcls in ["unknown"]: #['kNegative', 'kPedInvalidCls']:
                #     continue

                # x_lt = camera_mea.get("reg_pt_x") if camera_mea.get("reg_pt_x") is not None else camera_mea.get("det_pt_x")
                # y_lt = camera_mea.get("reg_pt_y") if camera_mea.get("reg_pt_y") is not None else camera_mea.get("det_pt_y")
                # width = camera_mea.get("reg_pt_width") if camera_mea.get("reg_pt_width") is not None else camera_mea.get("det_pt_width")
                # height = camera_mea.get("reg_pt_height") if camera_mea.get("reg_pt_height") is not None else camera_mea.get("det_pt_height")

                x_lt = camera_mea.get("det_pt_x")
                y_lt = camera_mea.get("det_pt_y")
                width = camera_mea.get("det_pt_width")
                height = camera_mea.get("det_pt_height")
                
                x1 = int(x_lt)
                y1 = int(y_lt)
                x2 = int(x_lt + width)
                y2 = int(y_lt + height)
                label = f"{cls}-{subcls} ID:{id}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)

            # cv2.imshow("Image with Annotations", img)
            # key = cv2.waitKey(0)
            # if key == ord('q'):
            #     break
            
            save_path = os.path.join(save_seq_dir, img_name)
            cv2.imwrite(save_path, img)

def create_video_from_images(image_folder, output_video_path, fps=30):
    # 获取图像文件列表，并按文件名排序
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()

    if not images:
        print("指定路径下没有找到图像文件")
        return

    # 获取第一张图像的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码器和输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 遍历图像文件并写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 释放视频写入器
    video_writer.release()
    print(f"视频已保存到 {output_video_path}")

def gen_video():
    # 示例调用
    vehicle_name = "D4Q_51"
    date_name = "20250712"
    data_dir = f"/data1/dongying/Mono3d/visualization/{vehicle_name}/{date_name}"
    seq_list = os.listdir(data_dir)
    seq_list.sort()
    for seq_name in tqdm(seq_list):
        if seq_name not in ["seq_83", "seq_255"]:
            continue
        seq_dir = os.path.join(data_dir, seq_name)
        
        video_name = f"{vehicle_name}-{date_name}-{seq_name}.mp4"  # 替换为输出视频路径
        output_video_path = os.path.join(data_dir, video_name)
        create_video_from_images(seq_dir, output_video_path, fps=15)


if __name__ == "__main__":
    # visualize_2d_3d_asso_res()
    gen_video()
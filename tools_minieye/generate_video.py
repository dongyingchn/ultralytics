import os
import cv2

from tqdm import tqdm

def generate_video_from_images(image_paths, output_video_path, fps=20, width=1920, height=1080):

    # Get list of image files in the folder
    # image_files = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    # image_files.sort()  # Ensure images are in order

    if not image_paths:
        print("No images found in the specified folder.")
        return

    # Read the first image to get dimensions
    # first_image_path = image_paths[0]
    # first_image = cv2.imread(first_image_path)
    # height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as needed
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_path in tqdm(image_paths, desc="Generating video"):
        # image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (width, height))  # Ensure consistent size
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":

    model_version = "detect_minieye-driving-d4q-yolo11s-full_image-all"
    image_folder = f"./output_val/{model_version}"  # Replace with your image folder path
    output_dir = f"./output_videos/{model_version}"
    os.makedirs(output_dir, exist_ok=True)
    fps = 20                                     # Frames per second

    img_list = os.listdir(image_folder)
    img_list.sort()

    for seq_name in ['seq_145', 'seq_160', 'seq_174']:
        seq_img_list = [img for img in img_list if seq_name in img]

        for cate in ['2dbox', '3dbox_face', 'bev']:
            seq_img_list_cate = [img for img in seq_img_list if cate in img]

            seq_img_path_list = [os.path.join(image_folder, img) for img in seq_img_list_cate]

            output_video_path = f"./output_videos/{model_version}/{seq_name}_{cate}.mp4"

            if cate == 'bev':
                width, height = 1000, 1000
            else:
                width, height = 1920, 1080
            generate_video_from_images(seq_img_path_list[::2], output_video_path, fps, width, height)
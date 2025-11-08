import os
import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("/data1/ydong/projects/monocular_3d_object_detection/ultralytics/runs/detect/train4/weights/best.pt")  # pretrained YOLO11n model

video_path = "/data3/tmp/ydong/data_coding_problems/D4Q/data/20250731210417/d4q.1/camera4.bin"
cap = cv2.VideoCapture(video_path)

frame_id = 0
save_dir = "./runs/track/"
os.makedirs(save_dir, exist_ok=True)
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        frame = cv2.resize(frame, (1920, 1080))
        results = model.track(frame, persist=True, tracker="botsort.yaml")

        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            # masks = result.masks  # Masks object for segmentation masks outputs
            # keypoints = result.keypoints  # Keypoints object for pose outputs
            # probs = result.probs  # Probs object for classification outputs
            # obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen

            # Save the results to disk
            frame_name = f"frame_{frame_id:06d}.jpg"
            save_path = os.path.join(save_dir, frame_name)
            result.save(filename=save_path)  # save to disk

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        frame_id += 1

        # # Display the annotated frame
        # cv2.imshow("YOLO11 Tracking", annotated_frame)

        # # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
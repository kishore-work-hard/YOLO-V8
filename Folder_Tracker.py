from ultralytics import YOLO
import os
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use other variants like yolov8s.pt, yolov8m.pt, etc.

# Paths for input images and output folder
input_folder = 'path/to/your/image/folder'
output_folder = 'path/to/save/tracking/results'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Sort images to maintain frame order
image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# Initialize tracking (Using ByteTrack, supported by YOLOv8)
# Track state will be automatically maintained across frames
for frame_number, filename in enumerate(image_files):
    image_path = os.path.join(input_folder, filename)

    # Read image
    frame = cv2.imread(image_path)

    # Perform object tracking
    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml")  # Tracker is persisted across frames

    # Draw the tracking results on the frame
    annotated_frame = results[0].plot()

    # Save the annotated frame
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, annotated_frame)

    print(f"Processed frame {frame_number + 1}/{len(image_files)}: {filename}")

print("Tracking completed and results saved.")

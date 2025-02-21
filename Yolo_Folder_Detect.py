from ultralytics import YOLO
import os

# Path to the YOLOv8 model (you can use 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.)
model = YOLO('yolov8n.pt')  # Using the nano model for speed; you can change it based on accuracy needs.

# Folder containing images
input_folder = 'path/to/your/image/folder'
output_folder = 'path/to/save/results'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all images in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        
        # Perform detection
        results = model(image_path)

        # Save results
        results.save(filename=os.path.join(output_folder, filename))

print("Detection completed and results saved.")

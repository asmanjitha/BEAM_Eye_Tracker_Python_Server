from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained YOLOv11 model (assuming it's available as "yolov11.pt")
model = YOLO("yolov8s.pt")


# Load an image
image_path = "image.png"  # Change this to your image file path
image = cv2.imread(image_path)


# Run inference
results = model(image)


# Draw bounding boxes on the image
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class index

        # Get class label
        label = model.names[cls]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

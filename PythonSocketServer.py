import asyncio
import websockets
import cv2
import numpy as np
import torch
from ultralytics import YOLO  # Updated to use YOLOv8
import base64

# Load YOLOv8 Model (Change to YOLOv11 once available)
model_path = "yolo11m.pt"  # Using YOLOv8 nano model for efficiency
model = YOLO(model_path)

# WebSocket Server Handler
async def handle_connection(websocket):
    print("✅ Client connected!")

    try:
        while True:
            # Receive image data from Unity
            image_data = await websocket.recv()
            print(f"📷 Received {len(image_data)} bytes")

            # Convert bytes to numpy image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            received_image_path = "received_input.jpg"
            cv2.imwrite(received_image_path, img)

            if img is None:
                print("❌ Failed to decode image")
                continue

            # Perform object detection with YOLOv8
            results = model.predict(img)
            print(results)

            # Extract bounding boxes
            bounding_boxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    label = model.names[class_id]
                    bounding_boxes.append(f"{label} {x1} {y1} {x2} {y2}")

                    # Draw bounding boxes on the image
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save detected image (Optional)
            detected_image_path = "detected_output.jpg"
            cv2.imwrite(detected_image_path, img)
            print(f"✅ Detection complete! Image saved: {detected_image_path}")

            # Send bounding box data back to Unity
            response = "\n".join(bounding_boxes)
            await websocket.send(response)
            print(f"📩 Sent bounding box data to Unity")
    
    except websockets.exceptions.ConnectionClosed:
        print("❌ Connection closed by Unity")

    finally:
        print("🔌 Client disconnected")

# Start WebSocket Server
async def main():
    server = await websockets.serve(handle_connection, "localhost", 4200)
    print("🚀 WebSocket Server started at ws://localhost:4200")
    await server.wait_closed()

asyncio.run(main())

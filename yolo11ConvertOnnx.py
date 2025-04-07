# import torch
# from models.yolo import Model  # Adjust the import based on your YOLOv11 implementation

# # Load the YOLOv11 model
# model = Model(cfg='path_to_yolov11_config.cfg')  # Provide the correct path to your config
# model.load_state_dict(torch.load('yolov11s.pt'))

# # Set the model to evaluation mode
# model.eval()

# # Dummy input for the model (adjust input size as needed)
# dummy_input = torch.randn(1, 3, 640, 640)

# # Export the model to ONNX
# torch.onnx.export(model, dummy_input, 'yolov11.onnx', opset_version=12)


from ultralytics import YOLO

# Load YOLOv11 PyTorch model
model = YOLO("yolov8m.pt")  # Ensure you have the correct YOLOv11 weights

# Export the model with simplifications
model.export(format="onnx", simplify=True, opset=11)



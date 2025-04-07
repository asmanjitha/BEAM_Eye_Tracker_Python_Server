import onnx

# Load ONNX model
model = onnx.load("yolo11m.onnx")

# Check model integrity
onnx.checker.check_model(model)
print("Model is valid!")

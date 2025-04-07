import onnx

# Load ONNX model
model = onnx.load("yolov8m.onnx")

# Print all output layer names
output_names = [output.name for output in model.graph.output]
print("Model Output Layers:", output_names)

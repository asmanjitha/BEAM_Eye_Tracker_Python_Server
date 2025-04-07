import onnx
from onnx import helper, shape_inference

# Load the ONNX model
model_path = "yolo11m.onnx"
model = onnx.load(model_path)

# Remove MatMul layers
nodes_to_remove = [node for node in model.graph.node if node.op_type == "MatMul"]
for node in nodes_to_remove:
    model.graph.node.remove(node)

# Save the modified ONNX model
onnx.save(model, "yolo11m_fixed.onnx")

# Run shape inference to fix model
onnx_model = onnx.load("yolo11m_fixed.onnx")
onnx_model = shape_inference.infer_shapes(onnx_model)
onnx.save(onnx_model, "yolo11m_fixed.onnx")

print("MatMul layers removed successfully!")

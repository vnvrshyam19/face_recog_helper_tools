import onnx

# Load the ONNX model
model = onnx.load("modified_model_with_static_shapes.onnx")

# Check the output shapes of the model
for output in model.graph.output:
    print(f"Output name: {output.name}, shape: {output.type.tensor_type.shape}")

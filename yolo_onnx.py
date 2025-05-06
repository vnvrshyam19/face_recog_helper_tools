import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto

onnx_model_path = "yolov8n-face.onnx"
model = onnx.load(onnx_model_path)
graph = model.graph

# Original output name
original_output_name = graph.output[0].name
graph.ClearField("output")  # Remove old output

transpose_output = "transposed_output"
transpose_node = helper.make_node(
    "Transpose",
    inputs=[original_output_name],
    outputs=[transpose_output],
    perm=[0, 2, 1],
    name="Transpose_to_8400x20"
)
graph.node.append(transpose_node)

# Utility to add a Slice node
def add_slice_node(name, input_tensor, start, end, axis, output_name):
    starts_name = f"{name}_starts"
    ends_name = f"{name}_ends"
    axes_name = f"{name}_axes"

    graph.initializer.append(numpy_helper.from_array(np.array([start], dtype=np.int64), starts_name))
    graph.initializer.append(numpy_helper.from_array(np.array([end], dtype=np.int64), ends_name))
    graph.initializer.append(numpy_helper.from_array(np.array([axis], dtype=np.int64), axes_name))

    slice_node = helper.make_node(
        "Slice",
        inputs=[input_tensor, starts_name, ends_name, axes_name],
        outputs=[output_name],
        name=name
    )
    graph.node.append(slice_node)

# Add slice nodes to split the transposed output
add_slice_node("slice_boxes", transpose_output, 0, 4, 2, "boxes")         # last dim: [0:4]
add_slice_node("slice_scores", transpose_output, 4, 5, 2, "scores")       # last dim: [4:5]
add_slice_node("slice_landmarks", transpose_output, 5, 20, 2, "landmarks")# last dim: [5:20]

# Define new output shapes: (1, 8400, ?)
boxes_out = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 8400, 4])
scores_out = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 8400, 1])
landmarks_out = helper.make_tensor_value_info("landmarks", TensorProto.FLOAT, [1, 8400, 15])

# Register new outputs
graph.output.extend([boxes_out, scores_out, landmarks_out])

# Save updated model
onnx.save(model, "yolov8n-face_modified.onnx")
print("✅ Saved model as 'yolov8n-face_modified.onnx'")


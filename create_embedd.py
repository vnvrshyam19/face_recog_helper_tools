import os
import numpy as np
import onnxruntime as ort
import cv2

image_path = "/home/Shyam/Shyam_25/img_db/Tanmay/WhatsApp Image 2025-04-30 at 11.10.42.jpeg"
path = "/home/Shyam/Shyam_25/buffalo_l/"
out_npy_path = os.path.join(path, image_path.replace("jpeg", "npy"))
print(out_npy_path)
raw_image = cv2.imread(image_path)
raw_image = cv2.resize(raw_image,(112,112))
assert raw_image.shape[0] == 112 and raw_image.shape[1] == 112
input_data = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
image_data = np.array(raw_image, dtype=np.float32)
image_data -= 127.5
image_data /= 128.0
image_data = np.transpose(image_data, (2, 0, 1))
image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
image = np.array(image_data, dtype=np.float32, order="C")

session = ort.InferenceSession("/home/Shyam/Shyam_25/buffalo_l/w600k_r50.onnx")
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]
embeddings = session.run(output_names, {input_name: image})
preds = embeddings[0][0]
res = np.reshape(preds,(1,-1))
# print(res)
norm=np.linalg.norm(res)                    
normal_array = res / norm
normal_array = np.reshape(normal_array,(-1,1))
print(normal_array)
print(normal_array.shape)
np.save(out_npy_path, normal_array)
import cv2
import numpy as np
import onnxruntime as ort

# Load image
image_path = "/home/Shyam/Shyam_25/img_db/Shyam/imagepreview1724565210929.jpg"
img = cv2.imread(image_path)
img_resized = cv2.resize(img, (640, 640))  # assuming 640x640 input
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# Prepare input
img_input = img_rgb.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))  # (HWC to CHW)
img_input = np.expand_dims(img_input, axis=0)

# Run ONNX model
session = ort.InferenceSession("yolov8n-face_modified.onnx")
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]
boxes, scores, landmarks = session.run(output_names, {input_name: img_input})

# Apply sigmoid to scores
print(scores)
probs = 1 / (1 + np.exp(-scores))
probs = probs[0, :, 0]
boxes = boxes[0]
landmarks = landmarks[0]

# Get index of best score
best_idx = np.argmax(probs)
best_score = probs[best_idx]

if best_score > 0.5:
    box = boxes[best_idx]  # [x, y, w, h] or [x1, y1, x2, y2] depending on model
    landmark = landmarks[best_idx].reshape(5, 3)

    # Assuming box is [x_center, y_center, width, height]
    x_center, y_center, w, h = box
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    # Draw bbox
    img_out = img_resized.copy()
    cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_out, f"{best_score:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Draw landmarks
    for (lx, ly, _) in landmark:
        cv2.circle(img_out, (int(lx), int(ly)), 3, (0, 0, 255), -1)
    print(f"Best Score : {best_score}")
    print(box)
    print(landmark)
    # Show image
    cv2.imshow("Detection", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ No detection above threshold")

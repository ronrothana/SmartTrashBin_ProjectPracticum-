import torch
import cv2
import numpy as np


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.25
model.iou = 0.45
model.max_det = 1000

recyclable = {"keyboard", "mouse", "remote", "toothbrush", "book"}
non_recyclable = {"person", "dog", "cat", "cow", "sandwich", "umbrella"}

# Start webcam capture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)

    for *box, conf, cls in results.pred[0]:
        cls = int(cls)
        label = model.names[cls]

        if label in recyclable:
            classification = "Recyclable"
            color = (0, 255, 0)
        elif label in non_recyclable:
            classification = "Non-Recyclable"
            color = (0, 0, 255)
        else:
            classification = "Unknown"
            color = (0, 255, 255)

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label}: {classification}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Real-Time Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
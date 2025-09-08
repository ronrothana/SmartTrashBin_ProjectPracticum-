# to test with all lists
from pathlib import Path

import cv2
import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5s")

print("\nYOLOv5s can detect the following classes:")
for i, name in model.names.items():
    print(f"{i}: {name}")

model.conf = 0.25
model.iou = 0.45
model.max_det = 1000

base_dir = Path(__file__).resolve().parent
img_path = base_dir / "images" / "Keyboard.png"

results = model(str(img_path))

recyclable = {
    "bottle",
    "wine glass",
    "cup",
    "book",
    "cardboard",
    "fork",
    "spoon",
    "knife",
    "bowl",
    "banana",
    "apple",
    "orange",
    "broccoli",
    "carrot",
    "pizza",
    "donut",
    "cake",
    "laptop",
    "cell phone",
    "keyboard",
    "mouse",
    "remote",
    "toothbrush",
}

non_recyclable = {
    "person",
    "dog",
    "cat",
    "toilet",
    "couch",
    "bed",
    "chair",
    "tv",
    "scissors",
    "teddy bear",
    "vase",
    "toaster",
    "sink",
    "microwave",
    "oven",
    "hair drier",
    "sports ball",
    "frisbee",
    "bench",
    "zebra",
    "bear",
    "cow",
    "elephant",
    "giraffe",
    "hot dog",
    "sandwich",
    "skateboard",
    "surfboard",
    "snowboard",
    "kite",
    "handbag",
    "suitcase",
    "tie",
    "umbrella",
}


img = cv2.imread(str(img_path))


for *box, conf, cls in results.pred[0]:
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
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"{label}: {classification}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

output_path = base_dir / "results" / "labeled_result.jpg"
output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), img)
print(f"\n Image saved with updated labels: {output_path}")


cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

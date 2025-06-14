# To show all the object classes YOLOv5s can detect:

import torch
from pathlib import Path

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

print("\nYOLOv5s can detect the following classes:")
for i, name in model.names.items():
    print(f"{i}: {name}")

model.conf = 0.25  
model.iou = 0.45   
model.max_det = 1000 

base_dir = Path(__file__).resolve().parent
img_path = base_dir / 'images' / 'Keyboard.png'

results = model(str(img_path))  

detected_names = results.names
detected_classes = results.pred[0][:, -1].tolist()  
detected_labels = [detected_names[int(i)] for i in detected_classes]

recyclable = {
    "bottle", "wine glass", "cup", "book", "cardboard",
    "fork", "spoon", "knife", "bowl", "banana", "apple", "orange",
    "broccoli", "carrot", "pizza", "donut", "cake",
    "laptop", "cell phone", "keyboard", "mouse", "remote", "toothbrush"
}

non_recyclable = {
    "person", "dog", "cat", "toilet", "couch", "bed", "chair", "tv",
    "scissors", "teddy bear", "vase", "toaster", "sink", "microwave", "oven",
    "hair drier", "sports ball", "frisbee", "bench", "zebra", "bear", "cow",
    "elephant", "giraffe", "hot dog", "sandwich", "skateboard", "surfboard",
    "snowboard", "kite", "handbag", "suitcase", "tie", "umbrella"
}

print("\nDetected Labels and Classification:")
for label in detected_labels:
    if label in recyclable:
        print(f"{label}: Recyclable")
    elif label in non_recyclable:
        print(f"{label}: Non-Recyclable")
    else:
        print(f"{label}: Unknown / Not Classified")

results.show()

results.save(save_dir=base_dir / 'results')

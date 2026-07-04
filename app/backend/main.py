from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


model = YOLO("/Users/mac/Detect_Drill_Bit/Models/yolo_v12/results/runs/detect/train/weights/best.pt")
CLASSES = ["Broken", "Chipped", "Scratched", "Severe_Rust", "Tip_Wear"]

@app.post("/predict")
async def predict_image(file_name: UploadFile):
    img = await file_name.read()
    array_1d = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(array_1d, cv2.IMREAD_COLOR)
    
    detections = []
    results = model(img)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            class_id = int(box.cls[0].item())
            class_name = CLASSES[class_id] if class_id < len(CLASSES) else "Unknown"
            detections.append({
                "box": [x1.item(), y1.item(), x2.item(), y2.item()],
                "conf": conf.item(),
                "class_id": class_id,
                "class_name": class_name
            })

    return detections
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
from ultralytics import YOLO
from mangum import Mangum

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Drill Bit Defect Detection API", "endpoints": ["/predict", "/compare"]}

# Lazy load models with ONNX for memory efficiency
model_after_aug = None
model_before_aug = None
CLASSES = ["Broken", "Chipped", "Scratched", "Severe_Rust", "Tip_Wear"]

def get_models():
    global model_after_aug, model_before_aug
    if model_after_aug is None:
        model_path = os.path.join(os.path.dirname(__file__), "../app/backend/models/best_after_aug_int8.onnx")
        model_after_aug = YOLO(model_path)
    if model_before_aug is None:
        model_path = os.path.join(os.path.dirname(__file__), "../app/backend/models/best_before_aug_int8.onnx")
        model_before_aug = YOLO(model_path)
    return model_after_aug, model_before_aug

def run_detection(model, img):
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

@app.post("/predict")
async def predict_image(file: UploadFile):
    img = await file.read()
    array_1d = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(array_1d, cv2.IMREAD_COLOR)
    
    model_after, _ = get_models()
    detections = run_detection(model_after, img)
    return detections

@app.post("/compare")
async def compare_models(file: UploadFile):
    img = await file.read()
    array_1d = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(array_1d, cv2.IMREAD_COLOR)
    
    model_before, model_after = get_models()
    detections_before = run_detection(model_before, img)
    detections_after = run_detection(model_after, img)
    
    return {
        "before_aug": detections_before,
        "after_aug": detections_after
    }

# Vercel handler
handler = Mangum(app)

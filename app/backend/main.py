from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
from ultralytics import YOLO

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Drill Bit Defect Detection API", "endpoints": ["/predict", "/compare"]}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Load both models with relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_AFTER = os.path.join(BASE_DIR, "models/best_after_aug.pt")
MODEL_PATH_BEFORE = os.path.join(BASE_DIR, "models/best_before_aug.pt")

# Lazy load models to avoid startup issues
model_after_aug = None
model_before_aug = None
CLASSES = ["Broken", "Chipped", "Scratched", "Severe_Rust", "Tip_Wear"]

def get_models():
    global model_after_aug, model_before_aug
    if model_after_aug is None:
        model_after_aug = YOLO(MODEL_PATH_AFTER)
    if model_before_aug is None:
        model_before_aug = YOLO(MODEL_PATH_BEFORE)
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
async def predict_image(file_name: UploadFile):
    img = await file_name.read()
    array_1d = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(array_1d, cv2.IMREAD_COLOR)
    
    model_after, _ = get_models()
    detections = run_detection(model_after, img)
    return detections

@app.post("/compare")
async def compare_models(file_name: UploadFile):
    img = await file_name.read()
    array_1d = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(array_1d, cv2.IMREAD_COLOR)
    
    model_before, model_after = get_models()
    detections_before = run_detection(model_before, img)
    detections_after = run_detection(model_after, img)
    
    return {
        "before_aug": detections_before,
        "after_aug": detections_after
    }
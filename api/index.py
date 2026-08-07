from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import onnxruntime as ort
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
session_after = None
session_before = None
CLASSES = ["Broken", "Chipped", "Scratched", "Severe_Rust", "Tip_Wear"]

def preprocess_image(img, input_size=640):
    # Resize and normalize image
    img_resized = cv2.resize(img, (input_size, input_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = img_normalized.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_transposed, axis=0)
    return img_batch.astype(np.float32)

def get_sessions():
    global session_after, session_before
    if session_after is None:
        model_path = os.path.join(os.path.dirname(__file__), "../app/backend/models/best_after_aug_int8.onnx")
        session_after = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    if session_before is None:
        model_path = os.path.join(os.path.dirname(__file__), "../app/backend/models/best_before_aug_int8.onnx")
        session_before = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session_after, session_before

def run_detection(session, img, original_size):
    detections = []
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Preprocess
    input_tensor = preprocess_image(img)
    
    # Run inference
    outputs = session.run(output_names, {input_name: input_tensor})
    
    # Process outputs (YOLO format: [batch, 4+num_classes, num_anchors])
    output = outputs[0][0]  # Remove batch dimension
    num_classes = len(CLASSES)
    
    # Filter detections with confidence threshold
    conf_threshold = 0.25
    for i in range(output.shape[1]):
        # Get box coordinates and confidence
        box = output[:4, i]
        class_probs = output[4:, i]
        max_conf = np.max(class_probs)
        
        if max_conf > conf_threshold:
            class_id = np.argmax(class_probs)
            x1, y1, x2, y2 = box
            
            # Scale to original image size
            h, w = original_size
            x1 = (x1 / 640) * w
            y1 = (y1 / 640) * h
            x2 = (x2 / 640) * w
            y2 = (y2 / 640) * h
            
            detections.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(max_conf),
                "class_id": int(class_id),
                "class_name": CLASSES[class_id] if class_id < len(CLASSES) else "Unknown"
            })
    
    return detections

@app.post("/predict")
async def predict_image(file: UploadFile):
    img = await file.read()
    array_1d = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(array_1d, cv2.IMREAD_COLOR)
    original_size = img.shape[:2]
    
    session_after, _ = get_sessions()
    detections = run_detection(session_after, img, original_size)
    return detections

@app.post("/compare")
async def compare_models(file: UploadFile):
    img = await file.read()
    array_1d = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(array_1d, cv2.IMREAD_COLOR)
    original_size = img.shape[:2]
    
    session_before, session_after = get_sessions()
    detections_before = run_detection(session_before, img, original_size)
    detections_after = run_detection(session_after, img, original_size)
    
    return {
        "before_aug": detections_before,
        "after_aug": detections_after
    }

# Vercel handler
handler = Mangum(app)

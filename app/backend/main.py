from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = FastAPI()

MODEL_PATH = "/app/models/best.onnx"

CLASSES = [
    "Broken",
    "Chipped",
    "Scratched",
    "Severe_Rust",
    "Tip_Wear"
]

CONF_THRESHOLD = 0.25

# Load model 1 lần khi server khởi động
model = YOLO(MODEL_PATH)


@app.get("/", status_code=200)
def root():
    return {
        "message": "Drill Bit Defect Detection API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "best.onnx"
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    # =========================
    # READ IMAGE
    # =========================

    image_bytes = await image.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    original_width, original_height = image.size


    # =========================
    # ULTRALYTICS PREDICTION
    # =========================

    results = model.predict(
        source=np.array(image),
        imgsz=640,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    result = results[0]


    # =========================
    # GET BOXES
    # =========================

    detections = []

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)


        for box, confidence, class_id in zip(
            boxes,
            confidences,
            class_ids
        ):

            x1, y1, x2, y2 = box


            # Clamp về ảnh gốc
            x1 = max(0, min(original_width, x1))
            y1 = max(0, min(original_height, y1))
            x2 = max(0, min(original_width, x2))
            y2 = max(0, min(original_height, y2))


            # Bỏ bbox không hợp lệ
            if x2 <= x1 or y2 <= y1:
                continue


            detections.append({
                "class_id": int(class_id),
                "class_name": CLASSES[int(class_id)],
                "confidence": round(
                    float(confidence),
                    4
                ),
                "bbox": {
                    "x1": int(round(x1)),
                    "y1": int(round(y1)),
                    "x2": int(round(x2)),
                    "y2": int(round(y2))
                }
            })


    # =========================
    # RESPONSE
    # =========================

    return {
        "success": True,
        "image_size": {
            "width": original_width,
            "height": original_height
        },
        "detections": detections
    }
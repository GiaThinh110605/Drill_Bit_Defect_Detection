from ultralytics import YOLO
import os

# Load models
model_after = YOLO("app/backend/models/best_after_aug.pt")
model_before = YOLO("app/backend/models/best_before_aug.pt")

# Export to ONNX for smaller size and faster inference
print("Exporting best_after_aug.pt to ONNX...")
model_after.export(format="onnx", opset=12, simplify=True)

print("Exporting best_before_aug.pt to ONNX...")
model_before.export(format="onnx", opset=12, simplify=True)

print("Export complete!")

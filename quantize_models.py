from ultralytics import YOLO
import os

# Load PyTorch models
model_after = YOLO("app/backend/models/best_after_aug.pt")
model_before = YOLO("app/backend/models/best_before_aug.pt")

# Export with quantization (int8) for smaller size
print("Quantizing best_after_aug.pt to int8 ONNX...")
model_after.export(format="onnx", opset=12, simplify=True, int8=True)

print("Quantizing best_before_aug.pt to int8 ONNX...")
model_before.export(format="onnx", opset=12, simplify=True, int8=True)

print("Quantization complete!")

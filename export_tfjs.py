from ultralytics import YOLO

# Load PyTorch models
model_after = YOLO("app/backend/models/best_after_aug.pt")
model_before = YOLO("app/backend/models/best_before_aug.pt")

# Export to TensorFlow.js
print("Exporting best_after_aug.pt to TensorFlow.js...")
model_after.export(format="tfjs")

print("Exporting best_before_aug.pt to TensorFlow.js...")
model_before.export(format="tfjs")

print("Export complete!")

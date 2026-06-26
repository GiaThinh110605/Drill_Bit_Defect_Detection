import json, os, cv2
import matplotlib.pyplot as plt

JSON_PATH = "/Users/mac/Detect_Drill_Bit/mui_khoan/train_preprocessed/cropped_width_annotations.json"
IMAGE_DIR = "/Users/mac/Detect_Drill_Bit/mui_khoan/train_preprocessed/cropped_images"

# 1. Đọc dữ liệu
with open(JSON_PATH, "r") as f:
    coco = json.load(f)

# 2. In kích thước toàn bộ ảnh
print("--- KÍCH THƯỚC ẢNH ---")
for img in coco["images"]:
    print(f"{img['file_name']}: {img['width']}x{img['height']}")

# 3. Trực quan hóa ảnh đầu tiên kèm Box
if coco["images"]:
    img_info = coco["images"][0]
    img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
    
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Vẽ các box thuộc về ảnh này
        for ann in coco["annotations"]:
            if ann["image_id"] == img_info["id"]:
                x, y, w, h = map(int, ann["bbox"])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Hiển thị
        plt.figure(figsize=(5, 6))
        plt.imshow(img)
        plt.title(f"Test: {img_info['file_name']}")
        plt.axis("on")
        plt.show()
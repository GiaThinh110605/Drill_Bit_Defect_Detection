import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import matplotlib.patches as patches
import numpy as np

class IndustrialEDA:
    def __init__(self, annotation_path, image_dir=None):
        with open(annotation_path, "r") as f:
            self.coco_data = json.load(f)
        self.image_dir = image_dir
        self.categories = {
            cat["id"]: cat["name"] for cat in self.coco_data["categories"]
        }

    def get_class_distribution(self):
        cat_counts = {}
        for caterogies in self.coco_data['annotations']:
            cat_id = caterogies["category_id"]
            if cat_id in cat_counts:
                cat_counts[cat_id] += 1
            else:
                cat_counts[cat_id] = 1
            
        return cat_counts
    
    def get_full_path(self):
        image_path = {}
        for root, directories, files in os.walk(self.image_dir):
            for file_name in files:
                if file_name.lower().endswith(('.jpg', 'jpeg', '.png')):
                    image_path[file_name] = os.path.join(root, file_name)
        return image_path
    
    def visualize_some_images(self, n_samples=5):

        image_path = self.get_full_path()
        show = 0
        for annotation in self.coco_data["annotations"]:
            if show >= 5:
                break
            image_id = annotation["image_id"]
            
            image_info = next((img for img in self.coco_data.get("images") if img["id"] == image_id), None)
            file_name = image_info["file_name"]
            
            full_path = image_path.get(file_name)
            img = Image.open(full_path)
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.imshow(img)
            
            category_id = annotation["category_id"]
            class_name = self.categories[category_id]
            ax.set_title("{}, {}".format(category_id, class_name))

            x, y, w, h = annotation["bbox"]
            rect = patches.Rectangle(xy=(x, y), width=w, height=h, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)

            plt.show()
            show += 1
    
    def resolution_distribution(self):
        resolution_map = {}
        for annotation in self.coco_data["images"]:
            if (annotation["width"], annotation["height"]) in resolution_map:
                resolution_map[(annotation["width"], annotation["height"])] += 1
            else: 
                resolution_map[(annotation["width"], annotation["height"])] = 1
        return resolution_map

    def analyze_image_quality(self):
        image_path = self.get_full_path()

        brightness_list = []
        contrast_list = []
        blurrieness_list = []
        file_names = []
        for image_info in self.coco_data["images"]:
            file_name = image_info["file_name"]
            full_path = image_path.get(file_name)

            # Nhằm việc chuyển đổi RGB sang Gray scale để tránh tính toán nhiều,
            # tránh gây nhiễu bởi độ sáng Gray = 0.299R + 0.587G + 0.114B
            image_gray = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            
            # độ sáng = trung bình của từng pixel trong ảnh
            brightness = np.mean(image_gray)
            # độ tương phản = độ lệch chuẩn của từng pixel trong ảnh
            contrast = np.std(image_gray)
            # độ mờ = đọ hàm bậc 2 của ảnh, sau đó lấy phương sai(độ phân tán của bức ảnh) 
            # ảnh sắc nét: có nhiều cạnh rõ ràng -> phương sai rất cao
            # ảnh bị nhòe/mờ: các cạnh bị mịn hóa, thay đổi độ sáng diễn ra từ từ -> phương sai rất thấp
            laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()

            brightness_list.append(brightness)
            contrast_list.append(contrast)
            blurrieness_list.append(laplacian_var)
            file_names.append(file_name)
            
            return {
                "file_name": file_names,
                "brightness": brightness_list,
                "contrast": contrast_list,
                "blurrieness": blurrieness_list
            }

    def plot_quality_distribution(self, metrics):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].hist(metrics["brightness"], bins=20, color="gold", edgecolor="black")
        axes[0].set_title("Phân phối độ sáng")
        axes[0].set_xlabel("Giá trị trung bình pixel (0-255)")
        axes[0].set_ylabel("Số lượng ảnh")

        axes[1].hist(metrics["contrast"], bins=20, color="gold", edgecolor="black")
        axes[1].set_title("Phân phối độ tương phản")
        axes[1].set_xlabel("Độ tương phản")
        axes[1].set_ylabel("Số lượng ảnh")

        axes[2].hist(metrics["blurrieness"], bins=20, color="gold", edgecolor="black")
        axes[2].set_title("Phân phối độ mờ")
        axes[2].set_xlabel("Độ mờ")
        axes[2].set_ylabel("Số lượng ảnh")
        plt.show()
            

if __name__ == "__main__":
    eda = IndustrialEDA("/Users/mac/Detect_Drill_Bit/mui_khoan/train/_annotations.coco.json", "/Users/mac/Detect_Drill_Bit/mui_khoan/train")
    print("categories: ", eda.get_class_distribution())
    # eda.visualize_some_images()
    print(eda.resolution_distribution())
    metrics = eda.analyze_image_quality()
    eda.plot_quality_distribution(metrics)
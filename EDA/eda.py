from json import decoder
import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import matplotlib.patches as patches

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
    
    def visualize_some_images(self, n_samples=5):
        image_path = {}
        for root, directories, files in os.walk(self.image_dir):
            for file_name in files:
                if file_name.lower().endswith(('.jpg', 'jpeg', '.png')):
                    image_path[file_name] = os.path.join(root, file_name)

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

            print(x, y, w, h)
            plt.show()
            show += 1
            

if __name__ == "__main__":
    eda = IndustrialEDA("/Users/mac/Detect_Drill_Bit/mui_khoan/train/_annotations.coco.json", "/Users/mac/Detect_Drill_Bit/mui_khoan/train")
    print(eda.coco_data.keys())
    eda.get_class_distribution()
    eda.visualize_some_images()
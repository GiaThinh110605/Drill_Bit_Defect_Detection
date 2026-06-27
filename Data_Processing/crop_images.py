import json
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt



class CropImages:
    def __init__(self, annotation_path, image_dir, output_dir):
        with open(annotation_path, "r") as f:
            self.coco_data = json.load(f)
        self.image_dir = image_dir
        self.output_dir = output_dir
        
        self.out_img_dir = os.path.join(output_dir, "cropped_images")
        os.makedirs(self.out_img_dir, exist_ok=True)

        self.img_annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_annotations:
                self.img_annotations[img_id] = []
            self.img_annotations[img_id].append(ann)
        
    def get_full_path(self):
        image_path = {}
        for root, directories, files in os.walk(self.image_dir):
            for file_name in files:
                if file_name.lower().endswith(('.jpg', 'jpeg', '.png')):
                    image_path[file_name] = os.path.join(root, file_name)
        return image_path

    def crop_width_only(self, padding_ratio=0.15):
        # Lấy map đường dẫn thông qua hàm phụ trợ
        image_path_map = self.get_full_path()

        new_images = []
        new_annotations = []
        
        new_coco_data = {
            "info": self.coco_data.get("info", {}),
            "licenses": self.coco_data.get("licenses", []),
            "categories": self.coco_data.get("categories", []),
            "images": [],
            "annotations": []
        }

        for img_info in self.coco_data["images"]:
            img_id = img_info["id"]
            file_name = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"] 
            
            full_path = image_path_map.get(file_name)
            anns = self.img_annotations.get(img_id, [])
            
            if not full_path or not os.path.exists(full_path) or not anns:
                continue

            img = cv2.imread(full_path)
            if img is None:
                continue

            # --- LOGIC MỚI: CẮT CỐ ĐỊNH 1/4 CHÍNH GIỮA ---
            quarter = img_w // 4
            
            crop_x1 = quarter          # Bỏ 1/2 bên trái
            crop_x2 = img_w - quarter  # Bỏ 1/2 bên phải
            
            crop_y1 = 0
            crop_y2 = img_h 

            crop_w = crop_x2 - crop_x1
            crop_h = img_h 

            # Tiến hành cắt lát dọc ảnh
            cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            out_img_path = os.path.join(self.out_img_dir, file_name)
            cv2.imwrite(out_img_path, cropped_img)

            # Cập nhật trường 'images' (Kích thước width giờ đây sẽ đồng đều chằn chặn)
            updated_img_info = img_info.copy()
            updated_img_info["width"] = crop_w
            updated_img_info["height"] = crop_h 
            new_images.append(updated_img_info)

            # Tính toán lại tọa độ BBox cho ảnh mới
            for ann in anns:
                updated_ann = ann.copy()
                x, y, w, h = ann["bbox"]

                # Tọa độ X mới sau khi tịnh tiến sang trái một khoảng crop_x1
                new_x = x - crop_x1
                new_y = y 
                
                # Kiểm tra nếu nhãn lỗi nằm hoàn toàn ngoài vùng 1/3 chính giữa (nếu có nhãn nhiễu ở rìa)
                # Giữ lại và clip biên nếu nhãn chạm nhẹ vào vùng cắt
                if new_x + w < 0 or new_x > crop_w:
                    continue  # Bỏ qua nhãn này vì đã bị cắt mất
                
                # Giới hạn tọa độ nằm trong không gian ảnh mới [0, crop_w]
                safe_x = max(0.0, new_x)
                safe_w = min(w, crop_w - safe_x)

                updated_ann["bbox"] = [round(safe_x, 2), round(new_y, 2), round(safe_w, 2), round(h, 2)]
                updated_ann["area"] = round(safe_w * h, 2)
                
                new_annotations.append(updated_ann)

        new_coco_data["images"] = new_images
        new_coco_data["annotations"] = new_annotations

        output_json_path = os.path.join(self.output_dir, "cropped_width_annotations.json")
        with open(output_json_path, "w") as f_out:
            json.dump(new_coco_data, f_out, indent=4)
            
        print(f"Xử lý xong! Đã cắt cố định 1/4 chính giữa cho {len(new_images)} ảnh.")
        print(f"Kích thước đầu ra cố định: {crop_w}x{crop_h}")
            

if __name__ == "__main__":
    # Đường dẫn gốc tới thư mục mui_khoan
    DATA_ROOT = "/Users/mac/Detect_Drill_Bit/mui_khoan"
    
    # Chỉ định đích danh tập test cần quét
    TARGET_SPLIT = "test"
    split_dir = os.path.join(DATA_ROOT, TARGET_SPLIT)
    
    # Quét đệ quy tìm tất cả các nhóm con bên trong tập test (Bright_Field, Dark_Field, side, top...)
    count_folder = 0
    for root, dirs, files in os.walk(split_dir):
        for file in files:
            if file == "_annotations.coco.json":
                count_folder += 1
                # Đường dẫn file JSON hiện tại
                anno_path = os.path.join(root, file)
                # Thư mục ảnh chính là thư mục chứa file JSON đó
                image_dir = root
                
                # Tự động map cấu trúc sang thư mục preprocessed tương ứng
                # Ví dụ: mui_khoan/test/Dark_Field/side -> mui_khoan/test_preprocessed/Dark_Field/side
                relative_path = os.path.relpath(root, DATA_ROOT)
                path_parts = relative_path.split(os.sep)
                path_parts[0] = f"{path_parts[0]}_preprocessed"
                
                output_dir = os.path.join(DATA_ROOT, *path_parts)
                
                print(f"\n[{count_folder}] Đang xử lý nhóm: {relative_path}")
                print(f"   + Input JSON: {anno_path}")
                print(f"   + Output Dir: {output_dir}")
                
                # Thực thi tiền xử lý cắt ảnh
                preprocessor = CropImages(anno_path, image_dir, output_dir)
                preprocessor.crop_width_only()

    print(f"\n Hoàn thành! Đã quét và xử lý xong toàn bộ {count_folder} thư mục con của tập test.")
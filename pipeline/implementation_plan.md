# End-to-End Roadmap for Drill Bit Defect Detection

Based on the dataset found at `/Users/mac/Detect_Drill_Bit/mui_khoan`, it is an object detection dataset formatted in COCO format (from Roboflow). The images are categorized into `train`, `valid`, and `test` sets, and further separated into `Bright_Field` and `Dark_Field` lighting conditions.

The dataset includes the following defect categories:
- `drill` (base class)
- `Broken`
- `Chipped`
- `Scratched`
- `Severe_Rust`
- `Tip_Wear`

Since you want to code it yourself to learn, here is a complete, step-by-step roadmap tailored to this specific dataset, organized into 5 phases.

---

# GIAI ĐOẠN 1: EDA, TIỀN XỬ LÝ & BASELINE

## 1. EDA Ban Đầu (Dữ liệu Thô)

Trước khi xây dựng model, cần hiểu đặc điểm dữ liệu. Viết script (Python, Pandas, Matplotlib, OpenCV/PIL) để thực hiện:

- **Mở file COCO JSON soi cấu trúc annotation:**
  - Parse `_annotations.coco.json` để hiểu cấu trúc categories, images, annotations
  - Kiểm tra mapping giữa image_id và annotation_id

- **Phát hiện ảnh trùng, ảnh hỏng, BBox tràn viền/size=0:**
  - **Data Leakage Check:** Tính MD5 hash cho tất cả ảnh trong train/valid/test để phát hiện ảnh trùng lặp giữa các tập
    - ✅ **Kết quả hiện tại:** 38 ảnh trùng giữa train và valid (cần xử lý)
  - **Corrupted Images Check:** Kiểm tra ảnh không thể đọc được bằng OpenCV/PIL
  - **BBox Validation:** Kiểm tra bbox có negative coordinates, out of bounds, width/height = 0
    - ✅ **Kết quả hiện tại:** Không có bbox lỗi
  - **Duplicate Annotations:** Kiểm tra annotation trùng lặp (cùng image_id, category_id, bbox)
    - ✅ **Kết quả hiện tại:** Không có duplicate annotations

- **Kiểm tra chia tập train/val/test ban đầu:**
  - Verify ratio ảnh giữa các tập (thường 70/20/10 hoặc 80/10/10)
  - Đếm số lượng images và annotations trong mỗi split

- **Phân tích kích thước BBox theo class:**
  - ✅ **Kết quả hiện tại:** Chipped/Tip_Wear có bbox nhỏ hơn nhiều (1.47-1.77%) so với Broken/Severe_Rust (6.52-7.08%)
  - Cần cân nhắc khi thiết kế anchor boxes

## 2. Tiền xử lý Data & Anti-Leakage

Chuẩn bị dataset để model có thể digest dễ dàng.

- **Chuyển COCO JSON → YOLO Format (.txt):**
  - Convert `_annotations.coco.json` sang YOLO format
  - Format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)
  - Tạo file `data.yaml` với paths và class names

- **Chuẩn hóa BBox [x_center, y_center, w, h]:**
  - Convert từ COCO format [x, y, w, h] sang YOLO format [x_center, y_center, w, h] normalized

- **Xóa triệt để ảnh trùng lặp ở Valid/Test (Check MD5/Phash):**
  - ⚠️ **Cần xử lý:** 38 ảnh trùng train-valid hiện tại
  - Xóa ảnh trùng từ valid set (giữ trong train) hoặc ngược lại
  - Re-split dataset nếu cần

- **Lọc bỏ các BBox rác/lỗi phát hiện ở B1:**
  - ✅ Không cần lọc thêm (annotation chất lượng tốt)

- **Cắt viền đen (Black Border Removal):**
  - ⚠️ **Cần xử lý:** Ảnh có viền đen xung quanh vùng mũi khoan
  - Crop vùng trung tâm hoặc detect vùng mũi khoan để cắt bỏ viền

- **Tăng cường độ sáng/tương phản (Brightness/Contrast Enhancement):**
  - ⚠️ **Cần xử lý:** Độ sáng và độ tương phản ảnh còn thấp
  - Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Hoặc các kỹ thuật enhancement khác cho defect detection

## 3. Visualize Data Chuẩn Hóa

- **Biểu đồ phân bổ Class (Cân bằng / Mất cân bằng):**
  - ✅ **Đã có:** Train: Scratched(1367), Severe_Rust(1205), Chipped(1139), Tip_Wear(934), Broken(802)
  - Dataset mất cân bằng nhẹ, Scratched và Severe_Rust chiếm đa số

- **Heatmap vị trí lỗi trên bề mặt mũi khoan:**
  - ⏳ **Cần thêm:** Visualize vị trí các bbox trên bề mặt mũi khoan
  - Phân tích xem defect thường xuất hiện ở vị trí nào (tip, body, edge)

- **Scatter plot phân bố kích thước BBox (nhỏ/lớn):**
  - ✅ **Đã có:** Mean bbox area 7,819 px², ratio 3.9%
  - Large objects: 4,528 (83%), Small objects: 919 (17%)

- **Overlay BBox kiểm tra lại nhãn bằng mắt:**
  - Visualize random images với bbox overlay để kiểm tra quality annotation

## 4. Train Baseline Model

- **Model:** YOLOv12n (nano version) trên Dataset sạch, chưa Augmentation
- **Environment Setup:** Install ultralytics package
- **Create data.yaml:** Point to dataset paths và class names
- **Transfer Learning:** Start với pre-trained model
- **Hyperparameters:**
  - epochs: 50-100
  - imgsz: 448 hoặc 640
  - batch: 16 hoặc 32
  - optimizer: AdamW hoặc SGD

---

# GIAI ĐOẠN 2: PHÂN TÍCH LỖI (DATA-CENTRIC)

## 5. Inference Baseline & Error Analysis

- Chạy inference baseline model trên test set
- Thu thập predictions và ground truth

## 6. Visualize & Phân loại Lỗi

- **Visualize ảnh FP (False Positives - Mô hình nhìn nhầm):**
  - Model detect defect nhưng thực tế không có
  - Phân tích nguyên nhân (reflection, noise, background)

- **Visualize ảnh FN (False Negatives - Bỏ sót lỗi nhỏ):**
  - Model bỏ qua lỗi thực tế
  - Thường là defect nhỏ hoặc khó nhìn

- **Visualize BBox lệch (IoU < 0.5):**
  - BBox dự đoán lệch so với ground truth
  - Phân tích pattern lệch

- **Confusion Matrix & Class PR-Curves:**
  - Phân tích class confusion (Broken vs Chipped, etc.)
  - Precision-Recall curves cho từng class

---

# GIAI ĐOẠN 3: TỐI ƯU DỮ LIỆU (DATA IMPROVEMENT)

## 7. Target Data Improvement

- **Hard Example Mining (Gom ảnh FP/FN):**
  - Collect các ảnh model predict sai
  - Add vào training set với higher weight

- **Copy-Paste Tiny Defect:**
  - Copy defect từ ảnh này paste sang ảnh khác
  - Tăng số lượng defect nhỏ (Chipped, Tip_Wear)

- **Negative Background Samples:**
  - Add ảnh không có defect để giảm FP

- **Advanced Augmentations:**
  - Mosaic, MixUp
  - Domain matching (Bright ↔ Dark field transfer)
  - Class-aware oversampling cho minority classes

## 8. Re-Visualize & Train Data Mới

- **Overlay BBox kiểm tra ảnh sau Augment:**
  - Sanity check augmented data
  - Đảm bảo bbox không bị cắt hoặc lệch

- **Biểu đồ cân bằng Class mới:**
  - Kiểm tra class distribution sau augmentation

- **Train & Evaluate Ablation Study:**
  - Train với data mới
  - Compare với baseline
  - Ablation study từng augmentation technique

---

# GIAI ĐOẠN 4: CẢI TIẾN MÔ HÌNH & QUANTUM

## 9. Model Architecture Optimization

- **Custom YOLOv12-AHFIN / Inner-CIoU:**
  - Custom loss function cho defect detection
  - Attention mechanism cho small objects

- **Model Distillation (Teacher → Student):**
  - Train large model làm teacher
  - Distill knowledge sang smaller student model

## 10. Hybrid Quantum (QEDL-YOLOv12)

- **Quantum Feature Encoding:**
  - Extract features từ classical backbone
  - Encode sang quantum state

- **Fusion (Classical + Quantum):**
  - Hybrid classical-quantum architecture
  - Quantum circuit cho classification head

**Research Approach:**
1. Feature Extraction: Extract embeddings từ trained YOLO model
2. Dimension Reduction: PCA/Autoencoder để giảm xuống 4-8 features
3. Quantum Encoding: Angle/Amplitude embedding
4. VQC: Variational Quantum Circuit với rotation gates
5. Hybrid Training: PennyLane + PyTorch backpropagation

---

# GIAI ĐOẠN 5: TRIỂN KHAI SẢN PHẨM

## 11. RESTful API (FastAPI Backend)

- **Setup:** Install fastapi, uvicorn, python-multipart
- **Endpoints:**
  - POST /predict: Accept image upload
- **Logic:**
  1. Receive uploaded image
  2. Pass to YOLO model for inference
  3. Format results (bbox, class, confidence)
  4. Return JSON response
- **Testing:** uvicorn main:app --reload, test at http://localhost:8000/docs

## 12. Web Interface (React + Vite UI)

- **Tech Stack:** React + Vite cho modern, fast UI
- **Features:**
  - Drag-and-drop image upload
  - Display image with overlay bounding boxes
  - Show confidence scores và class names
  - Responsive design
- **Integration:** Fetch/axios để gọi FastAPI backend
- **Canvas:** HTML5 canvas để draw bounding boxes

## 13. Docker & Deploy (CI/CD, Cloud Deployment)

- **Dockerfile for API:**
  - Base: python:3.10-slim
  - Install system dependencies (libgl1-mesa-glx for OpenCV)
  - Copy requirements.txt và install packages
  - Copy FastAPI code và model weights
  - Expose port 8000, CMD uvicorn

- **Dockerfile for UI:**
  - Base: node:alpine
  - Install dependencies, build React app
  - Expose port 3000, CMD serve

- **Docker Compose:**
  - Spin up cả API và UI với single command
  - Configure networking giữa containers

- **CI/CD:**
  - GitHub Actions hoặc GitLab CI
  - Automated testing, building, deployment

- **Cloud Deployment Options:**
  - **Hugging Face Spaces:** Streamlit/Gradio hoặc Docker
  - **Render/Railway:** Easy free tiers
  - **AWS ECS/GCP Cloud Run:** Production-grade

---

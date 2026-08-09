# KẾT QUẢ BASELINE YOLOv12 - NHẬN DIỆN LỖI MŨI KHOAN

## 2. DỮ LIỆU

### 2.1 Tổng quan dataset
Dataset được thu thập từ các mũi khoan thực tế trong quá trình sản xuất, bao gồm các ảnh chụp dưới hai điều kiện ánh sáng khác nhau: Bright_Field (ánh sáng sáng) và Dark_Field (ánh sáng tối). Dataset được chia thành 3 tập: train, validation và test theo tỷ lệ 80:10:10.

### 2.2 Phân chia dataset

#### Tập Training
- **Tổng số ảnh:** 4,364 ảnh
- **Bright_Field:** 2,182 ảnh
- **Dark_Field:** 2,182 ảnh
- **Tổng số annotations:** 5,447 bounding boxes

#### Tập Validation
- **Tổng số ảnh:** 882 ảnh
- **Bright_Field:** 441 ảnh
- **Dark_Field:** 441 ảnh
- **Tổng số annotations:** 1,027 bounding boxes

#### Tập Test
- **Tổng số ảnh:** 704 ảnh
- **Bright_Field:** 352 ảnh
- **Dark_Field:** 352 ảnh
- **Tổng số annotations:** 1,522 bounding boxes (758 Bright_Field + 764 Dark_Field)

### 2.3 Phân phối theo loại lỗi

Dataset bao gồm 5 loại lỗi chính trên mũi khoan:

| Loại lỗi | Số lượng (Train) | Tỷ lệ (%) | Mô tả |
|----------|------------------|-----------|-------|
| Scratched | 1,367 | 25.1% | Trầy xước bề mặt mũi khoan |
| Severe_Rust | 1,205 | 22.1% | Gỉ sét nặng trên bề mặt |
| Chipped | 1,139 | 20.9% | Mẻ nứt cạnh mũi khoan |
| Tip_Wear | 934 | 17.1% | Mòn đầu mũi khoan |
| Broken | 802 | 14.7% | Mẻ vỡ nghiêm trọng |
| **Tổng** | **5,447** | **100%** | |

### 2.4 Đặc điểm dataset

#### Điều kiện ánh sáng
- **Bright_Field:** Ánh sáng sáng, chi tiết rõ nét, phù hợp cho phát hiện lỗi nhỏ
- **Dark_Field:** Ánh sáng tối, tăng độ tương phản, phù hợp cho phát hiện lỗi bề mặt

#### Góc chụp
- **Side view:** Góc chụp ngang, quan sát cạnh mũi khoan
- **Top view:** Góc chụp từ trên xuống, quan sát đỉnh mũi khoan

#### Độ phân giải
- Kích thước ảnh gốc: Đa dạng, từ 1920x1080 đến 4096x2160 pixels
- Kích thước ảnh sau resize: 640x640 pixels (cho training)

### 2.5 Annotation
- **Format:** COCO (Common Objects in Context)
- **Loại annotation:** Bounding box (x, y, width, height)
- **Công cụ annotation:** LabelImg / CVAT
- **Kiểm soát chất lượng:** Double-check bởi 2 chuyên gia

## 1. CẤU HÌNH TRAINING

### Thông số mô hình
- **Mô hình:** YOLOv12n (nano version)
- **Task:** Object Detection
- **Số lượng classes:** 5 (Broken, Chipped, Scratched, Severe_Rust, Tip_Wear)
- **Kích thước ảnh:** 640x640 pixels

### Thông số training
- **Epochs:** 100
- **Batch size:** 32
- **Optimizer:** AdamW
- **Learning rate:** 0.001 (initial), 0.0001 (final)
- **Learning rate schedule:** Cosine annealing
- **Warmup epochs:** 3
- **Weight decay:** 0.0005
- **Momentum:** 0.937
- **Image augmentation:** 
  - Horizontal flip: 0.5
  - Vertical flip: 0.5
  - Translation: 0.1
  - Scale: 0.1
  - HSV_V: 0.2
  - Mosaic: 1.0
  - Auto augment: RandAugment
  - Erasing: 0.3

### Thông số dataset
- **Train images:** 4,364 images
- **Validation images:** 844 images
- **Test images:** 352 images (Bright_Field + Dark_Field)

## 2. KẾT QUẢ TRAINING

### Kết quả cuối cùng (Epoch 100)
- **Precision (P):** 0.76418
- **Recall (R):** 0.74053
- **mAP50:** 0.75270
- **mAP50-95:** 0.40013
- **Training time:** 5,750.59 seconds (~1.6 hours)

### Training Loss
- **Box loss:** 0.87041
- **Class loss:** 0.48954
- **DFL loss:** 1.06081

### Validation Loss
- **Box loss:** 1.64691
- **Class loss:** 1.13023
- **DFL loss:** 1.76225

## 3. ĐƯỜNG CONVERGENCE

### Epochs quan trọng
- **Epoch 10:** mAP50 = 0.62, mAP50-95 = 0.296
- **Epoch 20:** mAP50 = 0.653, mAP50-95 = 0.324
- **Epoch 30:** mAP50 = 0.67, mAP50-95 = 0.347
- **Epoch 50:** mAP50 = 0.729, mAP50-95 = 0.40
- **Epoch 70:** mAP50 = 0.766, mAP50-95 = 0.411
- **Epoch 90:** mAP50 = 0.768, mAP50-95 = 0.404
- **Epoch 100:** mAP50 = 0.753, mAP50-95 = 0.400

### Quan sát
- Mô hình đạt hiệu suất tốt nhất khoảng epoch 70-80
- Sau epoch 80, hiệu suất có xu hướng giảm nhẹ (overfitting nhẹ)
- Training loss giảm ổn định trong suốt quá trình

## 4. KẾT QUẢ THEO CLASS

### Confusion Matrix
- **Broken:** Precision cao, Recall trung bình
- **Chipped:** Precision và Recall cân bằng
- **Scratched:** Precision cao, Recall thấp
- **Severe_Rust:** Precision trung bình, Recall cao
- **Tip_Wear:** Precision thấp, Recall trung bình

*(Chi tiết confusion matrix xem file confusion_matrix.png)*

## 5. BIỂU ĐỒ KẾT QUẢ

### Các file biểu đồ có sẵn
- `BoxF1_curve.png` - Đường cong F1-score
- `BoxPR_curve.png` - Precision-Recall curve
- `BoxP_curve.png` - Precision theo epoch
- `BoxR_curve.png` - Recall theo epoch
- `confusion_matrix.png` - Confusion matrix
- `confusion_matrix_normalized.png` - Confusion matrix chuẩn hóa
- `results.png` - Tổng hợp kết quả training
- `labels.jpg` - Phân phối labels trong dataset
- `train_batch*.jpg` - Ví dụ training batches
- `val_batch*_labels.jpg` - Validation labels
- `val_batch*_pred.jpg` - Validation predictions

## 6. SO SÁNH VỚI CÁC PHƯƠNG PHÁP KHÁC

### YOLOv12 Baseline
- **mAP50:** 0.753
- **mAP50-95:** 0.400
- **FPS:** ~12.2 (CPU), ~45+ (GPU)
- **Model size:** ~5.6 MB
- **Parameters:** ~2.5M

### YOLOv12_eca (Epoch 113 - đang train)
- **mAP50:** 0.739
- **mAP50-95:** 0.40
- **FPS:** ~11.3 (CPU)
- **Model size:** ~6.5 MB
- **Parameters:** ~2.57M

*(Lưu ý: YOLOv12_eca chưa hoàn thành training)*

## 7. THÔNG TIN FILE CHECKPOINT

### Weights file
- **Path:** `/Users/mac/Detect_Drill_Bit/Models_Before_Handle_Data/yolo_v12/results/runs/detect/train/weights/best.pt`
- **Size:** ~5.6 MB
- **Format:** PyTorch checkpoint

### File saved
- `best.pt` - Checkpoint tốt nhất (theo mAP50)
- `last.pt` - Checkpoint epoch cuối cùng
- `epoch100.pt` - Checkpoint epoch 100

## 8. ĐÁNH GIÁ

### Ưu điểm
- Tốc độ training nhanh (~1.6 giờ cho 100 epochs)
- Model size nhỏ (~5.6 MB), phù hợp edge computing
- Hiệu suất tốt (mAP50 = 0.753)
- Tốc độ inference cao (FPS ~12.2 trên CPU)

### Hạn chế
- mAP50-95 còn thấp (0.400) - khó phát hiện chính xác bounding box
- Một số class (Tip_Wear) có precision thấp
- Có dấu hiệu overfitting nhẹ sau epoch 80

### Ứng dụng
- Phù hợp cho hệ thống edge computing do model size nhỏ
- Có thể deploy trên thiết bị nhúng như Jetson Nano
- Tốc độ inference cao phù hợp real-time detection

## 9. KẾT LUẬN

YOLOv12 baseline cho thấy hiệu suất tốt với mAP50 = 0.753 và model size nhỏ (~5.6 MB), phù hợp cho các ứng dụng edge computing trong nhận diện lỗi mũi khoan. Tuy nhiên, mAP50-95 còn thấp (0.400) cho thấy cần cải thiện độ chính xác của bounding box.

Các hướng cải thiện có thể:
1. Thử các kiến trúc attention khác (ECA, SE, CBAM)
2. Điều chỉnh hyperparameters
3. Tăng cường augmentation
4. Fusion với HQNN để cải thiện classification accuracy

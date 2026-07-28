# EDA - Exploratory Data Analysis

## 1. Data Leakage Check

Phát hiện **38 ảnh trùng lặp** giữa tập train và valid (data leakage nghiêm trọng):
- Train ↔ Valid: 38 ảnh trùng
- Train ↔ Test: 0
- Valid ↔ Test: 0

Cần xử lý data leakage này trước khi training model để tránh overfitting.

## 2. Phân bố Class (Class Distribution)

### Train set
- Scratched: 1,367 objects
- Severe_Rust: 1,205 objects
- Chipped: 1,139 objects
- Tip_Wear: 934 objects
- Broken: 802 objects

### Validation set
- Tip_Wear: 246 objects
- Scratched: 217 objects
- Severe_Rust: 210 objects
- Broken: 190 objects
- Chipped: 164 objects

### Test set
**Dark_Field:**
- Severe_Rust: 235 objects
- Scratched: 203 objects
- Broken: 128 objects
- Chipped: 105 objects
- Tip_Wear: 93 objects

**Bright_Field:**
- Severe_Rust: 231 objects
- Scratched: 200 objects
- Broken: 127 objects
- Chipped: 108 objects
- Tip_Wear: 92 objects

**Insight:** Dataset có sự mất cân bằng class, với Scratched và Severe_Rust chiếm đa số.

## 3. Kiểm tra Annotation Quality

Tất cả các tập (train, valid, test) đều:
- ✅ Không có bbox lỗi (negative coordinates, out of bounds)
- ✅ Không có bbox quá nhỏ (< 5px)
- ✅ Không có duplicate annotations

Annotation chất lượng tốt, không cần xử lý thêm.

## 4. Object Size Analysis

### Thống kê chung (Train set)
- Mean bbox area: 7,819 px²
- Mean ratio to image: 3.9%
- Min ratio: 0.14%
- Max ratio: 31.4%

### Phân bố theo class (theo mean ratio)
- Chipped: 1.47% (nhỏ nhất)
- Tip_Wear: 1.77%
- Scratched: 3.20%
- Severe_Rust: 6.52%
- Broken: 7.08% (lớn nhất)

### Phân loại kích thước
- Large objects: 4,528 (83%)
- Small objects: 919 (17%)

**Insight:** Các defect có kích thước khác nhau đáng kể theo loại defect, cần cân nhắc khi thiết kế anchor boxes.

## 5. Duplicate Images Check

- Không có duplicate images trong cùng một tập
- Có 38 duplicate giữa train và valid (đã báo ở mục 1)

## 6. Các vấn đề khác cần lưu ý

- Độ sáng và độ tương phản của bức ảnh còn khá thấp -> tăng cường data làm sáng bức ảnh hơn, với những kỹ thuật mới lạ có thể áp dụng cho tình huống này
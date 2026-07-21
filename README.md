<div align="center">
  <h1>🛠️ Drill Bit Defect Detection</h1>
  <p>Hệ thống tự động phát hiện và phân loại các lỗi trên mũi khoan sử dụng trí tuệ nhân tạo (YOLO) kết hợp giao diện Web hiện đại.</p>
  
  <a href="https://canva.link/zr4io2buqf924sv">📚 Tài liệu quá trình thực hiện (Canva)</a>
</div>

---

## 🚀 Giới thiệu
Dự án này tập trung vào việc giải quyết bài toán kiểm định chất lượng mũi khoan trong công nghiệp. Bằng việc phân tích dữ liệu hình ảnh (EDA) và áp dụng mô hình Deep Learning (YOLOv12), hệ thống có khả năng nhận diện chính xác các lỗi phổ biến trên mũi khoan, giúp tiết kiệm thời gian và nâng cao độ chính xác so với việc kiểm tra thủ công.

## 📊 Phân tích Dữ liệu (EDA) & Trực quan hóa
Quá trình **Exploratory Data Analysis (EDA)** được thực hiện nhằm hiểu rõ phân phối của dữ liệu và các đặc trưng của từng loại lỗi trước khi tiến hành huấn luyện mô hình.

<div align="center">
  <h3>Các loại lỗi trên mũi khoan</h3>
  <table>
    <tr>
      <td align="center">
        <img src="EDA/visualize_errors/Broken.png" width="300px;" alt="Broken Defect"/>
        <br /><b>Broken (Gãy)</b>
      </td>
      <td align="center">
        <img src="EDA/visualize_errors/Serve_Rust.png" width="300px;" alt="Severe Rust Defect"/>
        <br /><b>Severe Rust (Rỉ sét nặng)</b>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="EDA/visualize_errors/Tip_Wear.png" width="300px;" alt="Tip Wear Defect"/>
        <br /><b>Tip Wear (Mòn đầu)</b>
      </td>
      <td align="center">
        <img src="EDA/visualize_errors/scratched.png" width="300px;" alt="Scratched Defect"/>
        <br /><b>Scratched (Trầy xước)</b>
      </td>
    </tr>
  </table>
  
  <h3>Phân phối & Tương quan Dữ liệu</h3>
  <table>
    <tr>
      <td align="center">
        <img src="EDA/visualize_errors/Histogram.png" width="400px;" alt="Histogram"/>
        <br /><b>Biểu đồ phân phối lỗi (Histogram)</b>
      </td>
      <td align="center">
        <img src="EDA/visualize_errors/Heatmap.png" width="400px;" alt="Heatmap"/>
        <br /><b>Bản đồ nhiệt (Heatmap)</b>
      </td>
    </tr>
  </table>
</div>

## 🛠️ Quá trình Tiền xử lý (Data Preprocessing)
- Chuyển đổi định dạng Annotation từ COCO sang YOLO chuẩn.
- Crop (cắt) vùng lỗi chính trên ảnh để tối ưu hóa đầu vào, giúp quá trình training diễn ra nhanh hơn và tối ưu hóa tài nguyên phần cứng (RAM/VRAM).

## 💡 Kỹ năng đúc kết được
- **Kỹ năng Debug:** Rèn luyện thói quen print và trực quan hóa (visualize) dữ liệu mẫu trước khi chạy dữ liệu thực tế để tránh làm hỏng tập dữ liệu.
- **Kỹ năng Nghiên cứu:** Đọc hiểu và tham khảo các tài liệu, bài báo khoa học (Research Papers) về Computer Vision.
- **Giao tiếp Kỹ thuật:** Nâng cao khả năng diễn giải, giải thích code logic do bản thân/AI viết một cách mạch lạc cho người khác hiểu.

## ⚙️ Công nghệ & Công cụ
- **AI & Deep Learning:** Python, PyTorch, YOLOv12, OpenCV.
- **Backend:** FastAPI.
- **Frontend:** React, Vite.
- **DevOps & MLOps:** Docker, GitHub Actions (CI/CD), Hugging Face Spaces (Deployment).
- **Trợ lý AI:** Sử dụng Antigravity, Gemini để hỗ trợ lên kế hoạch, đọc hiểu source code phức tạp và gỡ lỗi (debug).

## 🔗 Tài liệu tham khảo
- Hướng dẫn chuyển đổi dữ liệu COCO sang định dạng YOLO: [Link tới file json mẫu](original-data/train/_annotations.coco.json)
- Slide fastapi: [Link tới file slide](https://canva.link/840ya322qx6ytve)
- Slide deep learning: [Link tới file slide](https://canva.link/6xdb3n1gpijx1kw)
- Quantum model: [Link tới file slide](https://canva.link/xbnml6xwiar58hc)
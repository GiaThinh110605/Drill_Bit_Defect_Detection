- Data annotation là quá trình gán nhãn dữ liệu cho 
    - Hình ảnh: 
        - bounding box: hộp bao quanh đối tượng để có thể nhận diện và phân loại đối tượng đó
        - segmentation: phân chia hình ảnh thành các khu vực nhỏ hơn, giúp mô hình phân biệt các đối tượng và nền
    - Văn bản: 
        - Entity Recognition: nhận diện và gắn nhãn cho các thực thể trong văn bản
        - Sentiment Analysis: gắn nhãn cho cảm xúc hoặc thái độ trong văn bản, giúp mô hình xác định được cảm xúc của người viết 
    - Video:
        - Object Tracking: theo dõi chuyển động của đối tượng trong video, giúp mô hình nhận diện và phân loại đối tượng qua từng khung hình
    - Âm thanh:
        - Speech Recognition: chuyển đổi âm thanh nói thành văn bản và gắn nhãn các phần trong đoạn âm thanh để mô hình có thể hiểu và phân tích lời nói
    -> tùy vào nhu cầu để mô hình có thể dự đoán tốt hơn


(https://docs.ultralytics.com/vi/guides/coco-to-yolo#h%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-chuy%E1%BB%83n-%C4%91%E1%BB%95i-t%E1%BB%ABng-b%C6%B0%E1%BB%9Bc)
- Chuyển đổi định dạng COCO sang định dạng YOLO:
    - Tại sao cần chuyển đổi:
        - Tốc độ đọc dữ liệu: cpu có thể đọc song song
        - Tự động khớp mọi kích thước: 
            - coco dùng (Xmin, Ymin, W, H)
            - yolo dùng 0 đến 1 (Xcenter, Ycenter, W, H)
        - Tối ưu hóa cho GPU
    - COCO JSON                                     YOLO TXT
    - Một file json cho tất cả các ảnh              Một tệp .txt cho mỗi ảnh
    - [x_min, y_min, width, height]                 class x_center y_center width height
    - category_id (bất kỳ chỉ số nào)               chỉ số bắt đầu từ 0
    - các mảng đa giác trong trường segmentation    tọa độ đa giác sau ClassID
    - [x, y, visibilit...] theo pixel               [x, y, visibility, ...] đã chuẩn hóa



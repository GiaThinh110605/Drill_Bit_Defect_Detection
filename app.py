import cv2
import gradio as gr
from ultralytics import YOLO

# Load model từ đường dẫn trong project của bạn
model = YOLO("Models_After_Handle_Data/yolov12_carafe/best.pt")


def predict_image(image):
  results = model(image)
  annotated_frame = results[0].plot()
  # Chuyển hệ màu BGR sang RGB cho Gradio hiển thị đúng màu
  annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
  return annotated_frame


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload ảnh cần kiểm tra"),
    outputs=gr.Image(label="Kết quả phát hiện"),
    title="Drill Bit Detection (YOLOv12)",
    description="Hệ thống nhận diện lỗi/mũi khoan sử dụng YOLOv12.",
)

if __name__ == "__main__":
  demo.launch()
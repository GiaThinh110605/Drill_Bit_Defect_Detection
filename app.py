import __main__
import cv2
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import spaces


# 1. Khai báo lại class CARAFE trực tiếp trong app.py
class CARAFE(nn.Module):

  def __init__(self, c1, scale_factor=2, k_encoder=3, up_kernel=3):
    super(CARAFE, self).__init__()
    self.scale_factor = scale_factor
    self.up_kernel = up_kernel
    self.k_encoder = k_encoder
    self.down = nn.Conv2d(c1, c1 // 4, 1)
    self.encoder = nn.Conv2d(
        c1 // 4,
        self.up_kernel**2 * self.scale_factor**2,
        self.k_encoder,
        padding=self.k_encoder // 2,
    )
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    N, C, H, W = x.size()
    kernel_tensor = self.down(x)
    kernel_tensor = self.encoder(kernel_tensor)
    kernel_tensor = F.pixel_shuffle(kernel_tensor, self.scale_factor)
    kernel_tensor = self.softmax(kernel_tensor)

    x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
    x_unfold = F.unfold(x, self.up_kernel, padding=self.up_kernel // 2)

    x_unfold = x_unfold.view(N, C, self.up_kernel**2, -1)
    kernel_tensor = kernel_tensor.view(N, 1, self.up_kernel**2, -1)

    out = (x_unfold * kernel_tensor).sum(dim=2)
    out = out.view(N, C, H * self.scale_factor, W * self.scale_factor)
    return out


# 2. Đăng ký CARAFE vào không gian tên toàn cục (__main__) để PyTorch nhận diện
__main__.CARAFE = CARAFE

# 3. Load mô hình YOLO của bạn
model = YOLO("Models_After_Handle_Data/yolov12_carafe/best.pt")


@spaces.GPU
def predict_image(image):
  results = model(image)
  annotated_frame = results[0].plot()
  # Chuyển đổi hệ màu BGR sang RGB cho Gradio hiển thị chuẩn xác
  annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
  return annotated_frame


# 4. Giao diện Gradio
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Tải ảnh cần kiểm tra lỗi"),
    outputs=gr.Image(label="Kết quả phát hiện"),
    title="Drill Bit Detection (YOLOv12 - CARAFE)",
    description=(
        "Hệ thống nhận diện lỗi sử dụng mô hình tùy chỉnh YOLOv12 tích hợp"
        " CARAFE."
    ),
)

if __name__ == "__main__":
  demo.launch()
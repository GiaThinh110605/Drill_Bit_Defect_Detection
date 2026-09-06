FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY app/backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/backend .

COPY Models_After_Handle_Data/yolov12_carafe/best.onnx /app/models/best.onnx

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
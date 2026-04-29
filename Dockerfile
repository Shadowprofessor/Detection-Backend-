FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n model weights
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create snapshot directory
RUN mkdir -p snapshots

# Copy app code
COPY . .

EXPOSE 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]

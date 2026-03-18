# LaMa Inpainting Service for macOS (CPU)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy LaMa project
COPY lama /tmp/lama-build
RUN mv /tmp/lama-build/* /app/ && rm -rf /tmp/lama-build

# Install Python dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-service.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-service.txt

# Download pre-trained model
RUN wget -O /tmp/big-lama.zip https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip && \
    unzip /tmp/big-lama.zip -d /app/ && \
    rm /tmp/big-lama.zip

# Install Flask for web service
RUN pip install --no-cache-dir flask flask-cors pillow

# Copy web service and templates
COPY lama_service.py /app/
COPY templates /app/templates

# Create directories for uploads and outputs
RUN mkdir -p /app/uploads /app/outputs

ENV PYTHONPATH=/app
ENV TORCH_HOME=/app

EXPOSE 5000

CMD ["python", "lama_service.py"]

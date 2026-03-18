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

# Copy LaMa project (rarely changes)
COPY lama /tmp/lama-build
RUN mv /tmp/lama-build/* /app/ && rm -rf /tmp/lama-build

# Install Python dependencies (rarely changes)
COPY requirements-service.txt /tmp/
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /tmp/requirements-service.txt && \
    pip install --no-cache-dir flask flask-cors pillow

# Download pre-trained model (rarely changes, cache it)
RUN --mount=type=cache,target=/tmp/cache \
    wget -O /tmp/big-lama.zip https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip && \
    unzip /tmp/big-lama.zip -d /app/ && \
    rm /tmp/big-lama.zip

# Generate test images (rarely changes)
COPY generate_test_images.py /app/
RUN python generate_test_images.py

# Create directories for uploads and outputs
RUN mkdir -p /app/uploads /app/outputs

# Copy application code (frequently changes - mounted as volume in dev)
COPY lama_service.py /app/
COPY templates /app/templates

ENV PYTHONPATH=/app
ENV TORCH_HOME=/app

EXPOSE 5000

CMD ["python", "lama_service.py"]

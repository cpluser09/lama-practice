# LaMa Image Inpainting Service - Deployment Guide

Deployment guide for LaMa image inpainting service on macOS.

## Table of Contents

- [Quick Start](#quick-start)
- [Deployment Options](#deployment-options)
- [macOS with MPS (GPU Acceleration)](#macos-with-mps-gpu-acceleration)
- [Docker (CPU)](#docker-cpu)
- [API Usage](#api-usage)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### macOS MPS (GPU) - Apple Silicon Only

```bash
git clone https://github.com/cpluser09/lama-practice.git
cd lama-practice
git submodule update --init --recursive
pip3 install -r requirements-service.txt
./launch_gpu_service_mac.sh
# Service: http://localhost:5002
```

### Docker (CPU)

```bash
git clone https://github.com/cpluser09/lama-practice.git
cd lama-practice
docker-compose up -d --build
# Service: http://localhost:5001
```

---

## Deployment Options

| Feature | macOS MPS (GPU) | Docker (CPU) |
|---------|-----------------|--------------|
| **Hardware** | Apple Silicon (M1/M2/M3/M4) | Any x86_64 / ARM |
| **Acceleration** | Metal GPU | CPU only |
| **Runtime** | Native Python | Docker container |
| **Port** | 5002 | 5001 |
| **Speed (512x384)** | ~2s | ~7s |
| **Speed (1500x2000)** | ~25s | ~111s |
| **Speedup** | 3.5-4x faster than CPU | 1x (baseline) |

### Which to Choose?

- **Use MPS GPU** if you have Apple Silicon Mac for 3-4x faster processing
- **Use Docker CPU** for Intel Mac or when containerization is preferred

---

## macOS with MPS (GPU Acceleration)

### System Requirements

- **Hardware**: Apple Silicon (M1/M2/M3/M4)
- **OS**: macOS 12.0+ (Monterey or later)
- **Python**: 3.9+
- **PyTorch**: 2.0+ with MPS support

### Installation

```bash
# 1. Clone repository
git clone https://github.com/cpluser09/lama-practice.git
cd lama-practice

# 2. Initialize submodule
git submodule update --init --recursive

# 3. Create virtual environment (dependencies installed to project venv/ directory)
python3 -m venv venv
source venv/bin/activate

# 4. Download pre-trained model (~363MB)
curl -L -o big-lama.zip "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
unzip big-lama.zip
rm big-lama.zip

# 5. Install dependencies
pip3 install -r requirements-service.txt

# 6. Start service
chmod +x launch_gpu_service_mac.sh
./launch_gpu_service_mac.sh
```

> **Note**: The virtual environment (`venv/`) stores all dependencies in the project directory, isolating them from system Python. To exit the virtual environment, run `deactivate`.

### Verify GPU Acceleration

```bash
curl http://localhost:5002/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "LaMa Inpainting",
  "device": "MPS (Metal Performance Shaders, Apple Silicon GPU)"
}
```

### Performance Benchmarks

| Resolution | CPU Time | MPS Time | Speedup |
|------------|----------|----------|---------|
| 512x384 | ~7s | ~2s | 3.5x |
| 1024x768 | ~15s | ~4s | 3.8x |
| 1500x2000 | ~111s | ~25s | 4.4x |
| 2048x1536 | ~180s | ~45s | 4.0x |

---

## Docker (CPU)

### System Requirements

- **Docker**: Docker Desktop for Mac
- **RAM**: 8GB+ recommended
- **Disk**: 5GB+ for image and model

### Installation

```bash
# 1. Clone repository
git clone https://github.com/cpluser09/lama-practice.git
cd lama-practice

# 2. Start service
docker-compose up -d --build
```

### Verify Service

```bash
curl http://localhost:5001/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "LaMa Inpainting",
  "device": "CPU (no GPU detected)"
}
```

### Development Mode (Hot Reload)

```bash
# Use dev compose file for code changes without rebuild
docker-compose -f docker-compose.dev.yml up -d

# Reload after code changes
docker-compose -f docker-compose.dev.yml restart lama-service
```

---

## API Usage

### Health Check

```bash
curl http://localhost:5002/health
```

### Inpaint Image

```bash
curl -X POST http://localhost:5002/inpaint \
  -F "image=@input.jpg" \
  -F "mask=@mask.png" \
  -o output.png
```

### Without Mask (Auto-generated)

```bash
curl -X POST http://localhost:5002/inpaint \
  -F "image=@input.jpg" \
  -o output.png
```

### Response Headers

```
X-Processing-Time: 2.15      # Total time (seconds)
X-Inference-Time: 1.93       # Model inference time (seconds)
X-Input-Resolution: 512x384   # Original image size
X-Output-Resolution: 512x384  # Output image size
```

---

## Troubleshooting

### MPS Not Available

**Check**:
```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

**Solution**:
- Ensure macOS 12.0+ and Apple Silicon hardware
- Reinstall PyTorch: `pip3 install --upgrade torch`

### Port Already in Use

**macOS Port 5000 Conflict**:
```bash
# Check what's using port 5000
lsof -i :5000
```

Port 5000 is used by AirPlay Receiver on macOS. The MPS service uses port 5002 to avoid this conflict.

**Docker Port Conflict**:
```bash
# Stop existing container
docker-compose down

# Or use different port in docker-compose.yml
ports:
  - "5001:5000"
```

### Out of Memory

**Symptom**: `RuntimeError: out of memory`

**Solution**:
- Large images are automatically resized if > 4096px
- Restart service: `./launch_gpu_service_mac.sh` or `docker-compose restart`

### Slow Performance on GPU

**Check GPU is being used**:
```bash
curl -s http://localhost:5002/health | grep device
```

Should return `"MPS (Metal Performance Shaders...)"`, not `"CPU"`.

---

## Files Reference

| File | Purpose |
|------|---------|
| `launch_gpu_service_mac.sh` | MPS GPU service (Apple Silicon) |
| `launch-cpu-service_docker.py` | CPU service (Docker) |
| `docker-compose.yml` | Docker deployment (CPU) |
| `reload-docker.sh` | Quick reload Docker service |
| `requirements-service.txt` | Python dependencies |

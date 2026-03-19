#!/usr/bin/env python3
"""
LaMa Image Inpainting Service - macOS GPU Version
Run directly on host for MPS (Metal Performance Shaders) support
"""

import os
import subprocess
import sys
from pathlib import Path

# Auto-detect and use venv if available
SYS_PATH = Path(__file__).parent
VENV_PYTHON = SYS_PATH / "venv" / "bin" / "python"

# If venv exists but we're not using it, re-run with venv python
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

# Setup paths
LAMA_PATH = SYS_PATH / "lama"
sys.path.insert(0, str(SYS_PATH))
sys.path.insert(0, str(LAMA_PATH))

# Check and install dependencies before importing them
def check_and_install_dependencies():
    """Check if required packages are installed, install if missing."""
    # Package name to import name mapping for special cases
    PACKAGE_TO_MODULE = {
        'opencv-python': 'cv2',
        'opencv-python-headless': 'cv2',
        'pillow': 'PIL',
        'pyyaml': 'yaml',
        'scikit-image': 'skimage',
        'scikit-learn': 'sklearn',
        'pytorch-lightning': 'pytorch_lightning',
        'gin-config': 'gin',
        'flatten-dict': 'flatten_dict',
    }

    def get_module_name(package_line: str) -> str:
        """Convert package name to module import name."""
        # Strip comments and whitespace
        package_line = package_line.split('#')[0].strip()
        if not package_line:
            return None

        # Extract package name (remove version specs like >=2.0, ==1.2.9, <2.0)
        for sep in ['>=', '==', '<=', '~= ', '<', '>', '  ', ' ']:
            if sep in package_line:
                package_line = package_line.split(sep)[0]
        pkg_name = package_line.strip()

        # Check special mappings first
        if pkg_name in PACKAGE_TO_MODULE:
            return PACKAGE_TO_MODULE[pkg_name]

        # Default: replace hyphens with underscores
        return pkg_name.replace('-', '_')

    # Parse requirements file
    requirements_file = SYS_PATH / "requirements-service.txt"
    if not requirements_file.exists():
        print(f"Warning: {requirements_file} not found, skipping dependency check")
        return

    with open(requirements_file) as f:
        package_lines = f.readlines()

    missing_packages = []
    for line in package_lines:
        module_name = get_module_name(line)
        if module_name:
            try:
                __import__(module_name)
            except ImportError:
                # Get package name for installation (reuse get_module_name logic)
                pkg_line = line.split('#')[0].strip()
                if pkg_line:
                    for sep in ['>=', '==', '<=', '~= ', '<', '>', '  ', ' ']:
                        if sep in pkg_line:
                            pkg_line = pkg_line.split(sep)[0]
                    pkg_name = pkg_line.strip()
                    missing_packages.append(pkg_name)

    if missing_packages:
        print("=" * 60)
        print(f"Missing dependencies: {', '.join(missing_packages)}")
        print("Installing dependencies...")
        print("=" * 60)

        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "-r", str(requirements_file)
            ])
            print("Dependencies installed successfully!")
            print("Please run the script again to start the service.")
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            print("Please run manually: pip3 install -r requirements-service.txt")
            sys.exit(1)


check_and_install_dependencies()

# Now import all dependencies
import io
import logging
import os
import traceback

import cv2
import numpy as np
import torch
import yaml
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/outputs'
MODEL_PATH = SYS_PATH / 'big-lama'
MAX_IMAGE_SIZE = 4096
DEFAULT_IMAGE_SIZE = 1024
STATIC_FOLDER = SYS_PATH / 'static'
TEST_FOLDER = STATIC_FOLDER / 'test'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_FOLDER))
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Global model variable
model = None
device = None


def get_optimal_device():
    """Auto-detect and return the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device (no GPU detected)")
    return device


def load_model():
    """Load the LaMa model on optimal device"""
    global model, device
    if model is not None:
        return model

    device = get_optimal_device()

    train_config_path = str(MODEL_PATH / 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = str(MODEL_PATH / 'models' / 'best.ckpt')
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load model to CPU first, then move to target device
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully on {device}")
    return model


def generate_default_mask(image_size):
    """Generate a default center square mask"""
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)

    width, height = image_size
    mask_size = min(width, height) // 4
    x1 = (width - mask_size) // 2
    y1 = (height - mask_size) // 2
    x2 = x1 + mask_size
    y2 = y1 + mask_size
    draw.rectangle([x1, y1, x2, y2], fill=255)

    return np.array(mask)


def resize_image_if_needed(image_np, max_size=MAX_IMAGE_SIZE, default_size=DEFAULT_IMAGE_SIZE):
    """Resize image if it's too large and ensure dimensions are multiples of 8"""
    h, w = image_np.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        logger.info(f"Resizing image from {image_np.shape} to ({new_h}, {new_w})")
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    # Ensure dimensions are multiples of 8
    new_h = h - (h % 8)
    new_w = w - (w % 8)

    if new_h != h or new_w != w:
        logger.info(f"Adjusting image size from ({h}, {w}) to ({new_h}, {new_w}) for model compatibility")
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image_np


def inpaint_image(image_np, mask_np):
    """Run inpainting on image with mask

    Returns: (result_bgr, inference_time)
    """
    import time

    model_obj = load_model()

    # Resize if needed
    image_np = resize_image_if_needed(image_np)
    mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]))

    # Convert to tensor format expected by model
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)

    mask_tensor = torch.from_numpy(mask_np).float() / 255.0
    if len(mask_tensor.shape) == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    mask_tensor = mask_tensor.unsqueeze(0)

    batch = {
        'image': image_tensor,
        'mask': mask_tensor
    }

    # Measure pure inference time
    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1

        inference_start = time.time()
        batch = model_obj(batch)
        inference_time = time.time() - inference_start

        result = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

    result = np.clip(result * 255, 0, 255).astype('uint8')
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return result_bgr, inference_time


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    global device
    device_info = str(device)
    if device.type == 'cuda':
        device_info = f"CUDA ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)"
    elif device.type == 'mps':
        device_info = "MPS (Metal Performance Shaders, Apple Silicon GPU)"

    return jsonify({
        'status': 'healthy',
        'service': 'LaMa Inpainting',
        'device': device_info
    })


@app.route('/inpaint', methods=['POST'])
def inpaint():
    """Inpaint an image with a mask"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']

        # Read and decode image
        image_bytes = image_file.read()
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        if image_np is None:
            return jsonify({'error': 'Invalid image format'}), 400

        logger.info(f"Received image of size {image_np.shape}")

        # Handle mask
        if 'mask' in request.files and request.files['mask'].filename:
            mask_file = request.files['mask']
            mask_bytes = mask_file.read()
            mask_np = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

            if mask_np.shape[:2] != image_np.shape[:2]:
                mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]))

            _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        else:
            mask_np = generate_default_mask((image_np.shape[1], image_np.shape[0]))

        # Run inpainting
        logger.info(f"Processing image of size {image_np.shape}")
        import time
        start_time = time.time()
        result_np, inference_time = inpaint_image(image_np, mask_np)
        total_time = time.time() - start_time

        # Encode result
        _, buffer = cv2.imencode('.png', result_np)
        result_bytes = io.BytesIO(buffer.tobytes())

        logger.info(f"Inpainting completed - Total: {total_time:.2f}s, Inference: {inference_time:.2f}s")
        response = send_file(result_bytes, mimetype='image/png')
        response.headers['X-Processing-Time'] = f'{total_time:.2f}'
        response.headers['X-Inference-Time'] = f'{inference_time:.2f}'
        response.headers['X-Input-Resolution'] = f'{image_np.shape[1]}x{image_np.shape[0]}'
        response.headers['X-Output-Resolution'] = f'{result_np.shape[1]}x{result_np.shape[0]}'
        return response

    except Exception as e:
        logger.error(f"Error during inpainting: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Web interface for image inpainting"""
    return render_template('index.html')


if __name__ == '__main__':
    # Check if running in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if not in_venv:
        logger.warning("=" * 60)
        logger.warning("NOT running in a virtual environment!")
        logger.warning("Dependencies are installed to system Python.")
        logger.warning("Recommended: Create a venv for isolation:")
        logger.warning("  python3 -m venv venv")
        logger.warning("  ./launch_gpu_service_mac.sh  # will auto-use venv")
        logger.warning("=" * 60)

    # Preload model
    logger.info("Loading LaMa model...")
    load_model()
    logger.info("Starting Flask server...")
    logger.info(f"GPU acceleration enabled: {device}")
    logger.info("Access at http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)

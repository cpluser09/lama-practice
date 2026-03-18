#!/usr/bin/env python3
"""
LaMa Image Inpainting Web Service
A simple Flask API for image inpainting using LaMa model
"""

import io
import logging
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

# Set threading env vars for CPU optimization
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Setup paths
SYS_PATH = Path(__file__).parent
saicinpainting_path = SYS_PATH / "saicinpainting"
sys.path.insert(0, str(SYS_PATH))

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/outputs'
MODEL_PATH = '/app/big-lama'
MAX_IMAGE_SIZE = 4096  # Limit image size for performance
DEFAULT_IMAGE_SIZE = 1024  # Default size for large images

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Increase max content length for large images
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Global model variable
model = None
device = None


def load_model():
    """Load the LaMa model"""
    global model, device
    if model is not None:
        return model

    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    train_config_path = os.path.join(MODEL_PATH, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(MODEL_PATH, 'models', 'best.ckpt')
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
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

    # If image is larger than max_size, resize it
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        logger.info(f"Resizing image from {image_np.shape} to ({new_h}, {new_w})")
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    # Ensure dimensions are multiples of 8 (required by LaMa model)
    new_h = h - (h % 8)
    new_w = w - (w % 8)

    if new_h != h or new_w != w:
        logger.info(f"Adjusting image size from ({h}, {w}) to ({new_h}, {new_w}) for model compatibility")
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image_np


def inpaint_image(image_np, mask_np):
    """Run inpainting on image with mask"""
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

    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1
        batch = model_obj(batch)
        result = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

    result = np.clip(result * 255, 0, 255).astype('uint8')
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return result_bgr


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'LaMa Inpainting',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })


@app.route('/inpaint', methods=['POST'])
def inpaint():
    """
    Inpaint an image with a mask using multipart/form-data.

    Parameters:
    - image: The image file to inpaint
    - mask: (optional) The mask file where white pixels indicate areas to inpaint

    Returns: The inpainted image
    """
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

            # Resize mask to match image size
            if mask_np.shape[:2] != image_np.shape[:2]:
                mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]))

            # Threshold to binary
            _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        else:
            # Generate default mask
            mask_np = generate_default_mask((image_np.shape[1], image_np.shape[0]))

        # Run inpainting
        logger.info(f"Processing image of size {image_np.shape}")
        result_np = inpaint_image(image_np, mask_np)

        # Encode result
        _, buffer = cv2.imencode('.png', result_np)
        result_bytes = io.BytesIO(buffer.tobytes())

        logger.info("Inpainting completed successfully")
        return send_file(result_bytes, mimetype='image/png')

    except Exception as e:
        logger.error(f"Error during inpainting: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Web interface for image inpainting"""
    return render_template('index.html')


if __name__ == '__main__':
    # Preload model
    logger.info("Loading LaMa model...")
    load_model()
    logger.info("Starting Flask server...")
    # Increase timeout for large images
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

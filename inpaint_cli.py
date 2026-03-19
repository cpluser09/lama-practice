#!/usr/bin/env python3
"""LaMa CLI Inpainting - macOS MPS/GPU Version"""

# Compatibility patches - MUST be first
import sys
import time

# Patch 1: pkg_resources compatibility for Python 3.13
try:
    import pkg_resources
except ImportError:
    # Create a fake pkg_resources module
    class _FakePkgResources:
        class DistributionNotFound(Exception):
            pass
    sys.modules['pkg_resources'] = _FakePkgResources()

import argparse
from pathlib import Path

# Setup paths
LAMA_PATH = Path(__file__).parent / "lama"
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(LAMA_PATH))

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

# Fix PyTorch 2.6+ weights_only issue - monkeypatch torch.load
import torch
import functools
_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint


def get_optimal_device():
    """Auto-detect best device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path, device):
    """Load LaMa model"""
    train_config_path = model_path / 'config.yaml'
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = model_path / 'models' / 'best.ckpt'
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    model.eval()
    return model


def resize_to_multiple_of_8(image_np):
    """Resize image to dimensions that are multiples of 8 for LaMa model"""
    h, w = image_np.shape[:2]
    new_h = h - (h % 8)
    new_w = w - (w % 8)

    if new_h != h or new_w != w:
        return cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA), (h, w)
    return image_np, (h, w)


def inpaint(model, image_path, mask_path, output_path, device):
    """Run inpainting, returns (total_time, inference_time)"""
    # Read images
    image_np = cv2.imread(str(image_path))
    mask_np = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image_np is None or mask_np is None:
        raise ValueError("Failed to load images")

    # Store original size
    orig_h, orig_w = image_np.shape[:2]

    # Resize mask to match image
    if mask_np.shape[:2] != image_np.shape[:2]:
        mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]))

    # Resize to multiples of 8 for LaMa model
    image_np, original_shape = resize_to_multiple_of_8(image_np)
    mask_np, _ = resize_to_multiple_of_8(mask_np)

    # Convert to RGB tensor
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)

    mask_tensor = torch.from_numpy(mask_np).float() / 255.0
    if len(mask_tensor.shape) == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    mask_tensor = mask_tensor.unsqueeze(0)

    batch = {'image': image_tensor, 'mask': mask_tensor}

    # Run inference with timing
    with torch.no_grad():
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1

        # Measure pure inference time
        inference_start = time.time()
        batch = model(batch)
        inference_time = time.time() - inference_start

        result = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

    result = np.clip(result * 255, 0, 255).astype('uint8')

    # Resize back to original dimensions if needed
    if result.shape[:2] != (orig_h, orig_w):
        result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_path), result_bgr)
    print(f"Saved result to {output_path}")
    print(f"Pure inference time: {inference_time:.2f} seconds")

    return inference_time


def main():
    parser = argparse.ArgumentParser(description='LaMa Image Inpainting CLI')
    parser.add_argument('--image', '-i', required=True, help='Input image path')
    parser.add_argument('--mask', '-m', required=True, help='Mask image path (white = to inpaint)')
    parser.add_argument('--output', '-o', required=True, help='Output image path')
    parser.add_argument('--model', default='big-lama', help='Model directory')

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path

    device = get_optimal_device()
    print(f"Using device: {device}")

    print("Loading model...")
    model = load_model(model_path, device)

    print(f"Processing: {args.image}")
    start_time = time.time()
    inference_time = inpaint(model, args.image, args.mask, args.output, device)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Inpainting completed in {total_time:.2f} seconds (inference: {inference_time:.2f}s, overhead: {total_time - inference_time:.2f}s)")
    print("Done!")


if __name__ == '__main__':
    main()

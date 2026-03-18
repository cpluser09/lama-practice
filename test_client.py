#!/usr/bin/env python3
"""
Test client for LaMa Inpainting Service
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import requests


def create_sample_image():
    """Create a sample image for testing"""
    # Create a simple test image with some content
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200

    # Add some colored rectangles
    cv2.rectangle(img, (50, 50), (200, 150), (255, 0, 0), -1)
    cv2.rectangle(img, (250, 100), (400, 200), (0, 255, 0), -1)
    cv2.rectangle(img, (100, 250), (300, 350), (0, 0, 255), -1)

    # Add text
    cv2.putText(img, 'LaMa Test', (150, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return img


def create_sample_mask(size):
    """Create a sample mask (white area will be inpainted)"""
    mask = np.zeros(size, dtype=np.uint8)

    # Create a white rectangle in the middle
    cv2.rectangle(mask, (200, 120), (300, 220), 255, -1)

    return mask


def test_service(base_url='http://localhost:5000'):
    """Test the LaMa inpainting service"""

    print(f"Testing LaMa Service at {base_url}")
    print("-" * 50)

    # Health check
    print("1. Health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Make sure the service is running!")
        return False

    # Create test images
    print("\n2. Creating test images...")
    test_img = create_sample_image()
    test_mask = create_sample_mask((test_img.shape[1], test_img.shape[0]))

    cv2.imwrite('test_input.jpg', test_img)
    cv2.imwrite('test_mask.png', test_mask)
    print("   Created test_input.jpg and test_mask.png")

    # Test without mask
    print("\n3. Testing inpainting without mask...")
    try:
        with open('test_input.jpg', 'rb') as f:
            response = requests.post(
                f"{base_url}/inpaint",
                files={'image': f},
                timeout=60
            )

        if response.status_code == 200:
            with open('test_output_no_mask.png', 'wb') as f:
                f.write(response.content)
            print("   Saved result to test_output_no_mask.png")
        else:
            print(f"   Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test with mask
    print("\n4. Testing inpainting with mask...")
    try:
        with open('test_input.jpg', 'rb') as img_f, open('test_mask.png', 'rb') as mask_f:
            response = requests.post(
                f"{base_url}/inpaint",
                files={'image': img_f, 'mask': mask_f},
                timeout=60
            )

        if response.status_code == 200:
            with open('test_output_with_mask.png', 'wb') as f:
                f.write(response.content)
            print("   Saved result to test_output_with_mask.png")
        else:
            print(f"   Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 50)
    print("Test complete! Check the output images:")
    print("  - test_input.jpg (original)")
    print("  - test_mask.png (mask)")
    print("  - test_output_no_mask.png (result without mask)")
    print("  - test_output_with_mask.png (result with mask)")
    print("=" * 50)

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test LaMa Inpainting Service')
    parser.add_argument('--url', default='http://localhost:5000',
                        help='Service URL (default: http://localhost:5000)')

    args = parser.parse_args()

    test_service(args.url)

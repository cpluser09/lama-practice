#!/usr/bin/env python3
"""
Generate test images for LaMa inpainting demonstration.
Creates various test patterns covering different inpainting scenarios.
"""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_text_removal_sample():
    """Create an image with text that needs to be removed"""
    img = Image.new('RGB', (512, 384), (135, 206, 235))  # Sky blue background
    draw = ImageDraw.Draw(img)

    # Draw a simple landscape
    draw.rectangle([50, 250, 462, 380], fill=(34, 139, 34))  # Grass
    draw.ellipse([200, 150, 312, 230], fill=(255, 200, 150))  # Sun

    # Add text overlay
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            font = ImageFont.load_default()

    draw.text((256, 100), "WATERMARK", fill=(255, 255, 255), anchor="mm", font=font)
    draw.text((256, 320), "Sample Text 123", fill=(255, 255, 255), anchor="mm", font=font)

    return img


def create_object_removal_sample():
    """Create an image with objects to remove"""
    img = Image.new('RGB', (512, 384), (240, 248, 255))
    draw = ImageDraw.Draw(img)

    # Background pattern
    for i in range(0, 512, 20):
        draw.line([(i, 0), (i, 384)], fill=(220, 230, 240))

    # Draw shapes representing objects to remove
    draw.rectangle([100, 100, 150, 150], fill=(255, 100, 100))  # Red box
    draw.ellipse([350, 200, 400, 250], fill=(100, 100, 255))  # Blue circle
    draw.polygon([(256, 50), (280, 100), (232, 100)], fill=(100, 255, 100))  # Green triangle

    return img


def create_scratch_repair_sample():
    """Create an image with scratch marks"""
    img = Image.new('RGB', (512, 384), (255, 248, 220))
    draw = ImageDraw.Draw(img)

    # Draw a photo-like scene
    draw.rectangle([50, 50, 462, 334], outline=(139, 69, 19), width=20)

    # Add scratch marks
    for i in range(5):
        y = 100 + i * 40
        draw.line([(200, y), (312, y)], fill=(50, 50, 50), width=2)

    # Add some spot damage
    import random
    for _ in range(10):
        x = random.randint(100, 400)
        y = random.randint(80, 300)
        r = random.randint(3, 8)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(80, 80, 80))

    return img


def create_face_restoration_sample():
    """Create a simple face-like image with missing parts"""
    img = Image.new('RGB', (512, 384), (255, 228, 196))
    draw = ImageDraw.Draw(img)

    # Simple face outline
    draw.ellipse([156, 84, 356, 284], outline=(139, 90, 43), width=3)

    # Eyes
    draw.ellipse([186, 134, 216, 164], fill=(60, 60, 60))
    draw.ellipse([296, 134, 326, 164], fill=(60, 60, 60))

    # Nose area (to be inpainted)
    draw.ellipse([236, 174, 276, 214], outline=(200, 180, 160), width=2)

    # Mouth
    draw.arc([186, 214, 326, 254], 0, 180, fill=(180, 80, 80), width=3)

    return img


def create_watermark_sample():
    """Create an image with watermark"""
    img = Image.new('RGB', (512, 384), (30, 30, 50))
    draw = ImageDraw.Draw(img)

    # Draw some content
    draw.rectangle([50, 50, 200, 200], fill=(100, 150, 200))
    draw.ellipse([280, 100, 450, 250], fill=(200, 100, 150))

    # Add diagonal watermark
    for i in range(-200, 600, 100):
        draw.text((i, 200), "PREVIEW", fill=(255, 255, 255), anchor="mm")
        draw.line([(i-50, 180), (i+50, 220)], fill=(255, 255, 255), width=20)

    return img


def create_old_photo_sample():
    """Create an image simulating old photo damage"""
    img = Image.new('RGB', (512, 384), (200, 180, 150))
    draw = ImageDraw.Draw(img)

    # Draw a vintage-style border
    draw.rectangle([20, 20, 492, 364], outline=(100, 80, 50), width=10)

    # Create missing/damaged areas (white for inpainting)
    damaged_areas = [
        (80, 80, 120, 120),
        (380, 60, 420, 100),
        (150, 280, 200, 330),
        (340, 300, 390, 350),
        (230, 150, 280, 200),
    ]

    for area in damaged_areas:
        draw.rectangle(area, fill=(255, 255, 255))

    # Add fold lines
    draw.line([(256, 0), (256, 384)], fill=(180, 160, 130), width=3)

    return img


def main():
    output_dir = Path(__file__).parent / 'static' / 'test'
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = [
        ('text_removal.jpg', create_text_removal_sample),
        ('object_removal.jpg', create_object_removal_sample),
        ('scratch_repair.jpg', create_scratch_repair_sample),
        ('face_restoration.jpg', create_face_restoration_sample),
        ('watermark.jpg', create_watermark_sample),
        ('old_photo.jpg', create_old_photo_sample),
    ]

    for filename, generator in test_images:
        img = generator()
        output_path = output_dir / filename
        img.save(output_path, 'JPEG')
        print(f'Generated: {output_path}')

    print(f'\nAll test images generated in: {output_dir}')


if __name__ == '__main__':
    main()

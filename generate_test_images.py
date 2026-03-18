#!/usr/bin/env python3
"""
Generate test images for LaMa inpainting demonstration.
Creates various test patterns covering different inpainting scenarios.
Each test image includes a corresponding mask file.
"""

import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_text_removal_sample():
    """Create an image with text that needs to be removed"""
    img = Image.new('RGB', (512, 384), (135, 206, 235))  # Sky blue background
    mask = Image.new('L', (512, 384), 0)
    draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)

    # Draw a simple landscape
    draw.rectangle([50, 250, 462, 380], fill=(34, 139, 34))  # Grass
    draw.ellipse([200, 150, 312, 230], fill=(255, 200, 150))  # Sun

    # Add text overlay - also mark on mask
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            font = ImageFont.load_default()

    text1 = "WATERMARK"
    text2 = "Sample Text 123"

    # Draw text on image
    draw.text((256, 100), text1, fill=(255, 255, 255), anchor="mm", font=font)
    draw.text((256, 320), text2, fill=(255, 255, 255), anchor="mm", font=font)

    # Draw white rectangles on mask for text areas (approximate)
    mask_draw.rectangle([100, 80, 412, 130], fill=255)  # WATERMARK area
    mask_draw.rectangle([100, 300, 412, 345], fill=255)  # Sample Text area

    return img, mask


def create_object_removal_sample():
    """Create an image with objects to remove"""
    img = Image.new('RGB', (512, 384), (240, 248, 255))
    mask = Image.new('L', (512, 384), 0)
    draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)

    # Background pattern
    for i in range(0, 512, 20):
        draw.line([(i, 0), (i, 384)], fill=(220, 230, 240))

    # Draw shapes representing objects to remove - also mark on mask
    shapes = [
        ([100, 100, 150, 150], 'rect'),
        ([350, 200, 400, 250], 'ellipse'),
        ([(236, 50), (270, 90), (232, 90)], 'polygon'),
    ]

    for shape, shape_type in shapes:
        if shape_type == 'rect':
            draw.rectangle(shape, fill=(255, 100, 100))  # Red box
            mask_draw.rectangle(shape, fill=255)
        elif shape_type == 'ellipse':
            draw.ellipse(shape, fill=(100, 100, 255))  # Blue circle
            mask_draw.ellipse(shape, fill=255)
        elif shape_type == 'polygon':
            draw.polygon(shape, fill=(100, 255, 100))  # Green triangle
            mask_draw.polygon(shape, fill=255)

    return img, mask


def create_scratch_repair_sample():
    """Create an image with scratch marks"""
    img = Image.new('RGB', (512, 384), (255, 248, 220))
    mask = Image.new('L', (512, 384), 0)
    draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)

    # Draw a photo-like scene
    draw.rectangle([50, 50, 462, 334], outline=(139, 69, 19), width=20)

    # Add scratch marks
    for i in range(5):
        y = 100 + i * 40
        draw.line([(200, y), (312, y)], fill=(50, 50, 50), width=2)
        mask_draw.line([(200, y), (312, y)], fill=255, width=4)

    # Add some spot damage
    for _ in range(10):
        x = random.randint(100, 400)
        y = random.randint(80, 300)
        r = random.randint(3, 8)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(80, 80, 80))
        mask_draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    return img, mask


def create_face_restoration_sample():
    """Create a simple face-like image with missing parts"""
    img = Image.new('RGB', (512, 384), (255, 228, 196))
    mask = Image.new('L', (512, 384), 0)
    draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)

    # Simple face outline
    draw.ellipse([156, 84, 356, 284], outline=(139, 90, 43), width=3)

    # Eyes
    draw.ellipse([186, 134, 216, 164], fill=(60, 60, 60))
    draw.ellipse([296, 134, 326, 164], fill=(60, 60, 60))

    # Nose area (to be inpainted) - mark on mask
    nose_area = [236, 174, 276, 214]
    draw.ellipse(nose_area, outline=(200, 180, 160), width=2)
    mask_draw.ellipse(nose_area, fill=255)

    # Mouth
    draw.arc([186, 214, 326, 254], 0, 180, fill=(180, 80, 80), width=3)

    return img, mask


def create_watermark_sample():
    """Create an image with watermark"""
    img = Image.new('RGB', (512, 384), (30, 30, 50))
    mask = Image.new('L', (512, 384), 0)
    draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)

    # Draw some content
    draw.rectangle([50, 50, 200, 200], fill=(100, 150, 200))
    draw.ellipse([280, 100, 450, 250], fill=(200, 100, 150))

    # Add diagonal watermark - also mark on mask
    for i in range(-200, 600, 100):
        draw.text((i, 200), "PREVIEW", fill=(255, 255, 255), anchor="mm")
        draw.line([(i-50, 180), (i+50, 220)], fill=(255, 255, 255), width=20)
        # Mark on mask - wider area for watermark
        mask_draw.rectangle([i-60, 170, i+60, 230], fill=255)

    return img, mask


def create_old_photo_sample():
    """Create an image simulating old photo damage"""
    img = Image.new('RGB', (512, 384), (200, 180, 150))
    mask = Image.new('L', (512, 384), 0)
    draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)

    # Draw a vintage-style border
    draw.rectangle([20, 20, 492, 364], outline=(100, 80, 50), width=10)

    # Create missing/damaged areas (white for inpainting) - also mark on mask
    damaged_areas = [
        (80, 80, 120, 120),
        (380, 60, 420, 100),
        (150, 280, 200, 330),
        (340, 300, 390, 350),
        (230, 150, 280, 200),
    ]

    for area in damaged_areas:
        draw.rectangle(area, fill=(255, 255, 255))
        mask_draw.rectangle(area, fill=255)

    # Add fold lines
    draw.line([(256, 0), (256, 384)], fill=(180, 160, 130), width=3)
    mask_draw.line([(256, 0), (256, 384)], fill=200, width=5)

    return img, mask


def main():
    output_dir = Path(__file__).parent / 'static' / 'test'
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = [
        ('text_removal', create_text_removal_sample),
        ('object_removal', create_object_removal_sample),
        ('scratch_repair', create_scratch_repair_sample),
        ('face_restoration', create_face_restoration_sample),
        ('watermark', create_watermark_sample),
        ('old_photo', create_old_photo_sample),
    ]

    for name, generator in test_images:
        img, mask = generator()

        # Save image
        img_path = output_dir / f'{name}.jpg'
        img.save(img_path, 'JPEG')
        print(f'Generated: {img_path}')

        # Save mask
        mask_path = output_dir / f'{name}_mask.png'
        mask.save(mask_path, 'PNG')
        print(f'Generated: {mask_path}')

    print(f'\nAll test images and masks generated in: {output_dir}')


if __name__ == '__main__':
    main()

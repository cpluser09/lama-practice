#!/usr/bin/env python3
"""Run inpainting tests on all patterns and collect performance data"""

import subprocess
import re
from pathlib import Path
import json

# Test patterns directory
TEST_DIR = Path("/Volumes/Samsung_T3/PW/pattern/inpainting")
OUTPUT_DIR = Path("/tmp/inpainting_test_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model path
MODEL_PATH = "big-lama"

# Resolutions to test
WIDTHS = [64, 128, 256, 512, 1024, 2048, 4096]

def run_inpaint_test(width, height):
    """Run inpainting test and return timing data"""
    image_file = TEST_DIR / f"test_pattern_{width}x{height}.jpg"
    mask_file = TEST_DIR / f"test_pattern_{width}x{height}_mask.jpg"
    output_file = OUTPUT_DIR / f"result_{width}x{height}.png"

    if not image_file.exists() or not mask_file.exists():
        return None

    cmd = [
        "venv/bin/python",
        "inpaint_cli.py",
        "-i", str(image_file),
        "-m", str(mask_file),
        "-o", str(output_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent
        )

        output = result.stdout + result.stderr

        # Extract both total time and inference time
        match = re.search(r'Inpainting completed in ([\d.]+) seconds \(inference: ([\d.]+)s, overhead: ([\d.]+)s\)', output)
        if match:
            total_time = float(match.group(1))
            inference_time = float(match.group(2))
            overhead = float(match.group(3))
            return {
                'width': width,
                'height': height,
                'resolution': f'{width}x{height}',
                'pixels': width * height,
                'total_time': total_time,
                'inference_time': inference_time,
                'overhead': overhead
            }
    except subprocess.TimeoutExpired:
        print(f"Timeout for {width}x{height}")
    except Exception as e:
        print(f"Error for {width}x{height}: {e}")

    return None

def main():
    print("=" * 60)
    print("LaMa Inpainting Performance Test")
    print("=" * 60)
    print()

    results = []

    for width in WIDTHS:
        # Calculate height (same aspect ratio as original)
        aspect_ratio = 5152 / 7728
        height = int(width * aspect_ratio)
        resolution = f'{width}x{height}'

        print(f"Testing {resolution}...", end=" ", flush=True)

        data = run_inpaint_test(width, height)
        if data:
            results.append(data)
            print(f"✓ total:{data['total_time']:.2f}s inf:{data['inference_time']:.2f}s")
        else:
            print("✗ Failed")

    print()
    print("=" * 60)

    # Sort by resolution
    results.sort(key=lambda x: x['width'])

    # Save results as JSON
    with open(OUTPUT_DIR / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate markdown report
    generate_markdown_report(results)

    print(f"\nResults saved to {OUTPUT_DIR}/")

def generate_markdown_report(results):
    """Generate markdown statistical report"""

    report = []
    report.append("# LaMa Inpainting Performance Report")
    report.append("")
    report.append(f"**Test Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Test Configuration")
    report.append("- **Device:** MPS (Apple Silicon GPU)")
    report.append("- **Model:** big-lama")
    report.append("- **Test Images:** 7 resolutions (64px to 4096px width)")
    report.append("")

    # Summary table
    report.append("## Results Summary")
    report.append("")
    report.append("| Resolution | Pixels | Total Time (s) | Inference (s) | Overhead (s) | Inf Pixels/sec |")
    report.append("|------------|--------|----------------|---------------|--------------|----------------|")

    for r in results:
        inf_pixels_per_sec = r['pixels'] / r['inference_time']
        report.append(f"| {r['resolution']} | {r['pixels']:,} | {r['total_time']:.2f} | {r['inference_time']:.2f} | {r['overhead']:.2f} | {inf_pixels_per_sec:,.0f} |")

    report.append("")

    # Statistics
    if results:
        total_times = [r['total_time'] for r in results]
        inf_times = [r['inference_time'] for r in results]
        overheads = [r['overhead'] for r in results]
        pixels = [r['pixels'] for r in results]
        pixels_per_sec = [r['pixels'] / r['inference_time'] for r in results]

        report.append("## Statistics")
        report.append("")
        report.append("### Total Time")
        report.append(f"- **Min:** {min(total_times):.2f}s")
        report.append(f"- **Max:** {max(total_times):.2f}s")
        report.append(f"- **Avg:** {sum(total_times)/len(total_times):.2f}s")
        report.append("")
        report.append("### Pure Inference Time")
        report.append(f"- **Min:** {min(inf_times):.2f}s")
        report.append(f"- **Max:** {max(inf_times):.2f}s")
        report.append(f"- **Avg:** {sum(inf_times)/len(inf_times):.2f}s")
        report.append("")
        report.append("### Overhead (pre/post processing)")
        report.append(f"- **Min:** {min(overheads):.2f}s")
        report.append(f"- **Max:** {max(overheads):.2f}s")
        report.append(f"- **Avg:** {sum(overheads)/len(overheads):.2f}s")
        report.append(f"- **Avg % of Total:** {sum(overheads)/sum(total_times)*100:.1f}%")
        report.append("")
        report.append("### Throughput (Inference)")
        report.append(f"- **Min:** {min(pixels_per_sec):,.0f} pixels/sec")
        report.append(f"- **Max:** {max(pixels_per_sec):,.0f} pixels/sec")
        report.append(f"- **Avg:** {sum(pixels_per_sec)/len(pixels_per_sec):,.0f} pixels/sec")
        report.append("")

        # Performance analysis
        report.append("## Performance Analysis")
        report.append("")

        # Calculate per-pixel time trend
        small = results[0]
        large = results[-1]
        speedup = large['pixels'] / small['pixels']
        total_time_ratio = large['total_time'] / small['total_time']
        inf_time_ratio = large['inference_time'] / small['inference_time']

        report.append(f"Scaling from {small['resolution']} to {large['resolution']}:")
        report.append(f"- Pixel count increased by: {speedup:.1f}x")
        report.append(f"- Total time increased by: {total_time_ratio:.1f}x")
        report.append(f"- Inference time increased by: {inf_time_ratio:.1f}x")
        report.append(f"- Linear scaling would be: {speedup:.1f}x")
        report.append("")

        if inf_time_ratio < speedup:
            efficiency = (speedup / inf_time_ratio - 1) * 100
            report.append(f"✓ **GPU efficiency gain:** {efficiency:.1f}% better than linear scaling (inference)")
        else:
            overhead = (inf_time_ratio / speedup - 1) * 100
            report.append(f"⚠ **Overhead:** {overhead:.1f}% worse than linear scaling (inference)")

    report.append("")

    # Print report
    markdown_text = "\n".join(report)
    print(markdown_text)

    # Save report
    report_file = OUTPUT_DIR / "performance_report.md"
    with open(report_file, 'w') as f:
        f.write(markdown_text)

    print(f"\nReport saved to: {report_file}")

if __name__ == "__main__":
    main()

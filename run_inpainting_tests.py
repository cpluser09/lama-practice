#!/usr/bin/env python3
"""Run inpainting tests on all patterns and collect performance data"""

import argparse
import subprocess
import re
from pathlib import Path
import json
import sys
import time


def parse_mbt_file(path: Path) -> dict[str, str]:
    """Parse an MBT file into a dictionary of key-value pairs."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";") or line.startswith("//"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def run_inpaint_test(mbt_file: Path, output_dir: Path, model_override: Path | None = None):
    """Run inpainting test using an MBT file and return timing data"""
    mbt_name = mbt_file.stem

    cmd = [
        sys.executable,
        "inpaint_cli.py",
        "--mbt", str(mbt_file)
    ]

    # Override model path if provided
    if model_override:
        cmd.extend(["--model", str(model_override)])

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).parent
        )
        end_time = time.time()

        output = result.stdout + result.stderr

        # Parse MBT file to get test info
        mbt_config = parse_mbt_file(mbt_file)

        # Extract both total time and inference time
        match = re.search(r'Inpainting completed in ([\d.]+) seconds \(inference: ([\d.]+)s, overhead: ([\d.]+)s\)', output)
        if match:
            total_time = float(match.group(1))
            inference_time = float(match.group(2))
            overhead = float(match.group(3))

            # Try to get resolution from image path or name
            resolution = mbt_config.get("Name", mbt_name)

            return {
                'name': mbt_name,
                'mbt_file': str(mbt_file),
                'resolution': resolution,
                'total_time': total_time,
                'inference_time': inference_time,
                'overhead': overhead,
                'wall_time': end_time - start_time,
                'success': result.returncode == 0,
                'image_path': mbt_config.get("ImagePath", ""),
                'model_path': mbt_config.get("ModelPath", "")
            }
        else:
            # Still return some data even if we couldn't parse the times
            return {
                'name': mbt_name,
                'mbt_file': str(mbt_file),
                'resolution': mbt_name,
                'total_time': None,
                'inference_time': None,
                'overhead': None,
                'wall_time': end_time - start_time,
                'success': result.returncode == 0,
                'error': output[-500:] if output else "No output",
                'returncode': result.returncode
            }

    except subprocess.TimeoutExpired:
        print(f"Timeout for {mbt_name}")
        return {
            'name': mbt_name,
            'mbt_file': str(mbt_file),
            'resolution': mbt_name,
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"Error for {mbt_name}: {e}")
        return {
            'name': mbt_name,
            'mbt_file': str(mbt_file),
            'resolution': mbt_name,
            'success': False,
            'error': str(e)
        }


def find_mbt_files(directory: Path) -> list[Path]:
    """Find all .mbt files in a directory (recursive)."""
    if directory.is_file() and directory.suffix == '.mbt':
        return [directory]
    return sorted(directory.rglob("*.mbt"))


def main():
    parser = argparse.ArgumentParser(description='LaMa Inpainting Performance Test using MBT files')
    parser.add_argument('mbt_path', nargs='?',
                        help='Path to MBT file or directory containing MBT files')
    parser.add_argument('--output-dir', '-o', default='/tmp/inpainting_test_results',
                        help='Output directory for results')
    parser.add_argument('--filter', '-f', action='append', default=[],
                        help='Filter MBT files by keyword (can be used multiple times)')
    parser.add_argument('--model', '-M', default='big-lama',
                        help='Override model path (ignores ModelPath in MBT files, default: big-lama)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get model override path
    model_override = Path(args.model)
    if not model_override.is_absolute():
        model_override = (Path(__file__).parent / model_override).resolve()
    if not model_override.exists():
        print(f"Warning: Model path does not exist: {model_override}", file=sys.stderr)

    # Determine MBT path
    if args.mbt_path:
        mbt_path = Path(args.mbt_path)
    else:
        # Default paths to check
        default_paths = [
            Path("/Volumes/BACH/Yf/gh/lama-coreml/tests/unit/perf/mbt"),
            Path("/Volumes/BACH/Yf/gh/lama-coreml/tests/mbt"),
            Path(__file__).parent,
        ]
        mbt_path = None
        for p in default_paths:
            if p.exists():
                mbt_path = p
                break
        if not mbt_path:
            print("Error: No MBT path specified and no default found.", file=sys.stderr)
            parser.print_help()
            sys.exit(1)

    # Find MBT files
    mbt_files = find_mbt_files(mbt_path)

    if not mbt_files:
        print(f"Error: No .mbt files found in {mbt_path}", file=sys.stderr)
        sys.exit(1)

    # Apply filters if any
    if args.filter:
        filtered = []
        for mbt_file in mbt_files:
            if any(f.lower() in mbt_file.name.lower() for f in args.filter):
                filtered.append(mbt_file)
        mbt_files = filtered
        if not mbt_files:
            print(f"Error: No .mbt files matched filters: {args.filter}", file=sys.stderr)
            sys.exit(1)

    print("=" * 70)
    print("LaMa Inpainting Performance Test (MBT)")
    print("=" * 70)
    print(f"MBT path: {mbt_path}")
    print(f"Found {len(mbt_files)} MBT file(s)")
    print(f"Output directory: {output_dir}")
    if args.filter:
        print(f"Filters: {args.filter}")
    if model_override:
        print(f"Model override: {model_override}")
    print()

    results = []

    for idx, mbt_file in enumerate(mbt_files, 1):
        mbt_name = mbt_file.stem
        print(f"[{idx}/{len(mbt_files)}] Testing {mbt_name}...", end=" ", flush=True)

        data = run_inpaint_test(mbt_file, output_dir, model_override)
        results.append(data)

        if data.get('success') and data.get('inference_time') is not None:
            print(f"✓ total:{data['total_time']:.2f}s inf:{data['inference_time']:.2f}s")
        elif data.get('success'):
            print(f"✓ (no timing data)")
        else:
            print(f"✗ Failed: {data.get('error', 'Unknown error')}")

    print()
    print("=" * 70)

    # Save results as JSON
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate markdown report
    generate_markdown_report(results, output_dir)

    print(f"\nResults saved to {output_dir}/")


def generate_markdown_report(results, output_dir: Path):
    """Generate markdown statistical report"""

    report = []
    report.append("# LaMa Inpainting Performance Report (MBT)")
    report.append("")
    report.append(f"**Test Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Test Configuration")
    report.append("- **Device:** MPS (Apple Silicon GPU)")
    report.append(f"- **Total Tests:** {len(results)}")
    report.append("")

    # Summary table
    report.append("## Results Summary")
    report.append("")
    report.append("| # | Test Name | Total Time (s) | Inference (s) | Overhead (s) | Status |")
    report.append("|---|-----------|----------------|---------------|--------------|--------|")

    successful_results = [r for r in results if r.get('success') and r.get('inference_time') is not None]

    for idx, r in enumerate(results, 1):
        status = "✓" if r.get('success') else "✗"
        total_time = f"{r['total_time']:.2f}" if r.get('total_time') is not None else "-"
        inf_time = f"{r['inference_time']:.2f}" if r.get('inference_time') is not None else "-"
        overhead = f"{r['overhead']:.2f}" if r.get('overhead') is not None else "-"
        report.append(f"| {idx} | {r['name']} | {total_time} | {inf_time} | {overhead} | {status} |")

    report.append("")

    # Statistics
    if successful_results:
        total_times = [r['total_time'] for r in successful_results]
        inf_times = [r['inference_time'] for r in successful_results]
        overheads = [r['overhead'] for r in successful_results]

        report.append("## Statistics")
        report.append("")
        report.append(f"**Successful tests:** {len(successful_results)}/{len(results)}")
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

    report.append("")

    # Print report
    markdown_text = "\n".join(report)
    print(markdown_text)

    # Save report
    report_file = output_dir / "performance_report.md"
    with open(report_file, 'w') as f:
        f.write(markdown_text)

    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()

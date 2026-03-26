#!/usr/bin/env python3
"""Run inpainting tests on all patterns and collect performance data"""

import argparse
import subprocess
import re
from pathlib import Path
import json
import sys
import time
from statistics import mean, stdev


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


def run_single_test(mbt_file: Path, output_dir: Path, model_override: Path | None = None):
    """Run a single inpainting test and return timing data"""
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

        # Extract both total time and inference time
        match = re.search(r'Inpainting completed in ([\d.]+) seconds \(inference: ([\d.]+)s, overhead: ([\d.]+)s\)', output)
        if match:
            total_time = float(match.group(1))
            inference_time = float(match.group(2))
            overhead = float(match.group(3))

            return {
                'total_time': total_time,
                'inference_time': inference_time,
                'overhead': overhead,
                'wall_time': end_time - start_time,
                'success': result.returncode == 0,
            }
        else:
            return {
                'success': False,
                'error': output[-500:] if output else "No output",
                'returncode': result.returncode
            }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def run_inpaint_test(mbt_file: Path, output_dir: Path, model_override: Path | None = None, num_runs: int = 1):
    """Run inpainting test multiple times and return aggregated timing data"""
    mbt_name = mbt_file.stem

    # Parse MBT file to get test info
    mbt_config = parse_mbt_file(mbt_file)
    resolution = mbt_config.get("Name", mbt_name)

    runs = []
    all_success = True

    for run_idx in range(num_runs):
        if num_runs > 1:
            print(f"  Run {run_idx + 1}/{num_runs}...", end=" ", flush=True)

        single_result = run_single_test(mbt_file, output_dir, model_override)
        runs.append(single_result)

        if not single_result.get('success'):
            all_success = False
            if num_runs > 1:
                print("✗ Failed")
        elif num_runs > 1:
            print(f"✓ inf:{single_result['inference_time']:.2f}s")

    # Aggregate results
    successful_runs = [r for r in runs if r.get('success') and r.get('inference_time') is not None]

    result = {
        'name': mbt_name,
        'mbt_file': str(mbt_file),
        'resolution': resolution,
        'runs': num_runs,
        'successful_runs': len(successful_runs),
        'success': all_success and len(successful_runs) > 0,
        'all_runs': runs,
        'image_path': mbt_config.get("ImagePath", ""),
        'model_path': mbt_config.get("ModelPath", "")
    }

    if successful_runs:
        total_times = [r['total_time'] for r in successful_runs]
        inference_times = [r['inference_time'] for r in successful_runs]
        overheads = [r['overhead'] for r in successful_runs]
        wall_times = [r['wall_time'] for r in successful_runs]

        result['total_time'] = mean(total_times)
        result['total_time_min'] = min(total_times)
        result['total_time_max'] = max(total_times)
        result['total_time_std'] = stdev(total_times) if len(total_times) > 1 else 0

        result['inference_time'] = mean(inference_times)
        result['inference_time_min'] = min(inference_times)
        result['inference_time_max'] = max(inference_times)
        result['inference_time_std'] = stdev(inference_times) if len(inference_times) > 1 else 0

        result['overhead'] = mean(overheads)
        result['overhead_min'] = min(overheads)
        result['overhead_max'] = max(overheads)
        result['overhead_std'] = stdev(overheads) if len(overheads) > 1 else 0

        result['wall_time'] = mean(wall_times)

    return result


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
    parser.add_argument('--runs', '-r', type=int, default=1,
                        help='Number of runs per test case (default: 1)')

    args = parser.parse_args()

    if args.runs < 1:
        print("Error: --runs must be at least 1", file=sys.stderr)
        sys.exit(1)

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
    print(f"Runs per test: {args.runs}")
    print(f"Output directory: {output_dir}")
    if args.filter:
        print(f"Filters: {args.filter}")
    if model_override:
        print(f"Model override: {model_override}")
    print()

    results = []

    for idx, mbt_file in enumerate(mbt_files, 1):
        mbt_name = mbt_file.stem
        print(f"[{idx}/{len(mbt_files)}] Testing {mbt_name}...")

        data = run_inpaint_test(mbt_file, output_dir, model_override, args.runs)
        results.append(data)

        if data.get('success') and data.get('inference_time') is not None:
            if args.runs == 1:
                print(f"  ✓ total:{data['total_time']:.2f}s inf:{data['inference_time']:.2f}s")
            else:
                print(f"  ✓ total:{data['total_time']:.2f}s (±{data['total_time_std']:.2f}s) "
                      f"inf:{data['inference_time']:.2f}s (±{data['inference_time_std']:.2f}s) "
                      f"[{data['successful_runs']}/{data['runs']} ok]")
        elif data.get('success'):
            print(f"  ✓ (no timing data)")
        else:
            print(f"  ✗ Failed: {data.get('error', 'Unknown error')}")

    print()
    print("=" * 70)

    # Save results as JSON
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate markdown report
    generate_markdown_report(results, output_dir, args.runs)

    print(f"\nResults saved to {output_dir}/")


def generate_markdown_report(results, output_dir: Path, num_runs: int):
    """Generate markdown statistical report"""

    report = []
    report.append("# LaMa Inpainting Performance Report (MBT)")
    report.append("")
    report.append(f"**Test Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Test Configuration")
    report.append("- **Device:** MPS (Apple Silicon GPU)")
    report.append(f"- **Total Tests:** {len(results)}")
    if num_runs > 1:
        report.append(f"- **Runs per test:** {num_runs}")
    report.append("")

    # Summary table
    report.append("## Results Summary")
    report.append("")

    if num_runs == 1:
        report.append("| # | Test Name | Total Time (s) | Inference (s) | Overhead (s) | Status |")
        report.append("|---|-----------|----------------|---------------|--------------|--------|")
    else:
        report.append("| # | Test Name | Total Time (s) | Inference (s) | Overhead (s) | Success |")
        report.append("|---|-----------|----------------|---------------|--------------|---------|")

    successful_results = [r for r in results if r.get('success') and r.get('inference_time') is not None]

    for idx, r in enumerate(results, 1):
        status = "✓" if r.get('success') else "✗"

        if r.get('inference_time') is not None:
            if num_runs == 1:
                total_time = f"{r['total_time']:.2f}"
                inf_time = f"{r['inference_time']:.2f}"
                overhead = f"{r['overhead']:.2f}"
                report.append(f"| {idx} | {r['name']} | {total_time} | {inf_time} | {overhead} | {status} |")
            else:
                total_time = f"{r['total_time']:.2f} ±{r['total_time_std']:.2f}"
                inf_time = f"{r['inference_time']:.2f} ±{r['inference_time_std']:.2f}"
                overhead = f"{r['overhead']:.2f} ±{r['overhead_std']:.2f}"
                success_info = f"{r['successful_runs']}/{r['runs']}"
                report.append(f"| {idx} | {r['name']} | {total_time} | {inf_time} | {overhead} | {success_info} |")
        else:
            if num_runs == 1:
                report.append(f"| {idx} | {r['name']} | - | - | - | {status} |")
            else:
                success_info = f"{r['successful_runs']}/{r['runs']}"
                report.append(f"| {idx} | {r['name']} | - | - | - | {success_info} |")

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

        # Per-test detailed stats when multiple runs
        if num_runs > 1:
            report.append("## Per-Test Detailed Statistics")
            report.append("")
            report.append("| Test Name | Metric | Mean (s) | Min (s) | Max (s) | Std Dev (s) |")
            report.append("|-----------|--------|----------|---------|---------|-------------|")

            for r in successful_results:
                report.append(f"| {r['name']} | **Total** | {r['total_time']:.4f} | {r['total_time_min']:.4f} | {r['total_time_max']:.4f} | {r['total_time_std']:.4f} |")
                report.append(f"| {r['name']} | **Inference** | {r['inference_time']:.4f} | {r['inference_time_min']:.4f} | {r['inference_time_max']:.4f} | {r['inference_time_std']:.4f} |")
                report.append(f"| {r['name']} | **Overhead** | {r['overhead']:.4f} | {r['overhead_min']:.4f} | {r['overhead_max']:.4f} | {r['overhead_std']:.4f} |")

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

#!/usr/bin/env python3
"""Master script to run all SteadyText benchmarks."""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils.reporting import generate_markdown_report, generate_csv_export
from benchmarks.utils.plotting import plot_speed_results, plot_accuracy_results


def run_speed_benchmarks(args):
    """Run speed benchmarks."""
    print("\n" + "=" * 80)
    print("Running Speed Benchmarks")
    print("=" * 80 + "\n")

    cmd = [
        sys.executable,
        "benchmarks/speed/run_speed_benchmarks.py",
        "--generation-iterations",
        str(args.generation_iterations),
        "--embedding-iterations",
        str(args.embedding_iterations),
    ]

    if args.skip_warmup:
        cmd.append("--skip-warmup")

    if args.quick:
        # Quick mode with reduced iterations
        cmd.extend(
            [
                "--generation-iterations",
                "10",
                "--embedding-iterations",
                "10",
                "--model-loading-iterations",
                "1",
                "--concurrent-iterations",
                "2",
                "--skip-streaming",
                "--skip-cache",
                "--skip-concurrent",
            ]
        )

    # Output file
    speed_output = Path(args.output_dir) / f"speed_benchmark_{args.timestamp}.json"
    cmd.extend(["-o", str(speed_output)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Speed benchmark failed: {result.stderr}")
        return None

    print(result.stdout)
    return speed_output


def run_accuracy_benchmarks(args):
    """Run accuracy benchmarks."""
    print("\n" + "=" * 80)
    print("Running Accuracy Benchmarks")
    print("=" * 80 + "\n")

    cmd = [
        sys.executable,
        "benchmarks/accuracy/run_accuracy_benchmarks.py",
        "--benchmarks",
        "simple" if args.quick else "all",
    ]

    if args.verify_determinism:
        cmd.append("--verify-determinism")

    # Output file
    accuracy_output = (
        Path(args.output_dir) / f"accuracy_benchmark_{args.timestamp}.json"
    )
    cmd.extend(["-o", str(accuracy_output)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Accuracy benchmark failed: {result.stderr}")
        # For accuracy, we might still want to continue even if LightEval isn't available
        if "LightEval is not installed" in result.stderr:
            print("Note: LightEval not available, only simple tests were run")
        else:
            return None

    print(result.stdout)
    return accuracy_output


def generate_reports(speed_results, accuracy_results, args):
    """Generate reports and visualizations."""
    print("\n" + "=" * 80)
    print("Generating Reports")
    print("=" * 80 + "\n")

    # Markdown report
    report_file = Path(args.output_dir) / f"benchmark_report_{args.timestamp}.md"
    generate_markdown_report(
        speed_results=str(speed_results) if speed_results else None,
        accuracy_results=str(accuracy_results) if accuracy_results else None,
        output_file=str(report_file),
    )

    # CSV exports
    if speed_results:
        csv_file = speed_results.with_suffix(".csv")
        generate_csv_export(str(speed_results), str(csv_file))

    if accuracy_results:
        csv_file = accuracy_results.with_suffix(".csv")
        generate_csv_export(str(accuracy_results), str(csv_file))

    # Plots
    if not args.skip_plots:
        print("\nGenerating plots...")

        if speed_results:
            try:
                plot_speed_results(str(speed_results))
            except Exception as e:
                print(f"Warning: Failed to generate speed plots: {e}")

        if accuracy_results:
            try:
                plot_accuracy_results(str(accuracy_results))
            except Exception as e:
                print(f"Warning: Failed to generate accuracy plots: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)

    print(f"\nAll results saved to: {args.output_dir}")
    print(f"\nMain report: {report_file}")

    if speed_results:
        print(f"Speed results: {speed_results}")

    if accuracy_results:
        print(f"Accuracy results: {accuracy_results}")

    # Print key metrics
    if speed_results and speed_results.exists():
        with open(speed_results, "r") as f:
            speed_data = json.load(f)

        if "results" in speed_data and "generation" in speed_data["results"]:
            gen_times = speed_data["results"]["generation"].get("times", [])
            if gen_times:
                avg_time = sum(gen_times) / len(gen_times)
                print(f"\nAverage generation time: {avg_time * 1000:.2f} ms")

    if accuracy_results and accuracy_results.exists():
        with open(accuracy_results, "r") as f:
            acc_data = json.load(f)

        if "tests" in acc_data and "determinism" in acc_data["tests"]:
            det_rate = acc_data["tests"]["determinism"].get("determinism_rate", 0)
            print(f"Determinism rate: {det_rate * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run all SteadyText benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python benchmarks/run_all_benchmarks.py
  
  # Quick benchmarks for CI
  python benchmarks/run_all_benchmarks.py --quick
  
  # Only speed benchmarks
  python benchmarks/run_all_benchmarks.py --only speed
  
  # Custom output directory
  python benchmarks/run_all_benchmarks.py --output-dir results/custom
        """,
    )

    # Benchmark selection
    parser.add_argument(
        "--only", choices=["speed", "accuracy"], help="Run only specific benchmark type"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks with reduced iterations",
    )

    # Configuration
    parser.add_argument(
        "--generation-iterations",
        type=int,
        default=100,
        help="Number of generation iterations (default: 100)",
    )
    parser.add_argument(
        "--embedding-iterations",
        type=int,
        default=100,
        help="Number of embedding iterations (default: 100)",
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Verify determinism in accuracy tests",
    )

    # Options
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warmup runs")
    parser.add_argument(
        "--skip-plots", action="store_true", help="Skip generating plots"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for all results",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamp to args for consistent naming
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 80}")
    print("SteadyText Comprehensive Benchmarks")
    print(f"{'=' * 80}")
    print(f"Timestamp: {args.timestamp}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")

    # Preload models once
    print("\nPreloading models...")
    import steadytext

    steadytext.preload_models(verbose=True)

    # Run benchmarks
    speed_results = None
    accuracy_results = None

    try:
        if not args.only or args.only == "speed":
            speed_results = run_speed_benchmarks(args)

        if not args.only or args.only == "accuracy":
            accuracy_results = run_accuracy_benchmarks(args)

        # Generate reports
        if speed_results or accuracy_results:
            generate_reports(speed_results, accuracy_results, args)
        else:
            print("\nNo benchmark results to report")

    except KeyboardInterrupt:
        print("\n\nBenchmarks interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running benchmarks: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

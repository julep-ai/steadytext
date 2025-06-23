"""Report generation utilities for benchmarks."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd


def generate_markdown_report(
    speed_results: Optional[str] = None,
    accuracy_results: Optional[str] = None,
    output_file: str = "benchmark_report.md",
):
    """Generate a comprehensive markdown report from benchmark results."""
    report_lines = [
        "# SteadyText Benchmark Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n---\n",
    ]

    # Speed benchmark results
    if speed_results and Path(speed_results).exists():
        with open(speed_results, "r") as f:
            speed_data = json.load(f)

        report_lines.extend(format_speed_section(speed_data))

    # Accuracy benchmark results
    if accuracy_results and Path(accuracy_results).exists():
        with open(accuracy_results, "r") as f:
            accuracy_data = json.load(f)

        report_lines.extend(format_accuracy_section(accuracy_data))

    # Write report
    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report generated: {output_file}")


def format_speed_section(data: Dict[str, Any]) -> List[str]:
    """Format speed benchmark results for the report."""
    lines = [
        "## Speed Benchmarks",
        "",
        f"**Test Date**: {data.get('timestamp', 'Unknown')}",
        "",
    ]

    if "args" in data:
        lines.extend(
            [
                "### Configuration",
                "",
                f"- Generation iterations: {data['args'].get('generation_iterations', 'N/A')}",
                f"- Embedding iterations: {data['args'].get('embedding_iterations', 'N/A')}",
                f"- Batch sizes tested: {data['args'].get('batch_sizes', 'N/A')}",
                "",
            ]
        )

    if "results" not in data:
        lines.append("*No results available*\n")
        return lines

    results = data["results"]

    # Summary table
    lines.extend(
        [
            "### Performance Summary",
            "",
            "| Operation | Mean Time (ms) | P95 (ms) | P99 (ms) | Throughput (ops/s) | Memory (MB) |",
            "|-----------|----------------|----------|----------|-------------------|-------------|",
        ]
    )

    for op_name, op_data in results.items():
        if "times" in op_data and op_data["times"]:
            times_ms = [t * 1000 for t in op_data["times"]]
            mean_time = sum(times_ms) / len(times_ms)
            p95 = sorted(times_ms)[int(len(times_ms) * 0.95)]
            p99 = sorted(times_ms)[int(len(times_ms) * 0.99)]
            throughput = 1000 / mean_time  # ops per second
            memory = max(op_data.get("memory_usage", [0]))

            lines.append(
                f"| {op_name} | {mean_time:.2f} | {p95:.2f} | {p99:.2f} | "
                f"{throughput:.2f} | {memory:.2f} |"
            )

    lines.append("")

    # Key findings
    lines.extend(format_speed_key_findings(results))

    return lines


def format_accuracy_section(data: Dict[str, Any]) -> List[str]:
    """Format accuracy benchmark results for the report."""
    lines = [
        "\n## Accuracy Benchmarks",
        "",
        f"**Test Date**: {data.get('timestamp', 'Unknown')}",
        "",
    ]

    # Simple test results
    if "tests" in data:
        lines.extend(["### Simple Accuracy Tests", ""])

        tests = data["tests"]

        if "determinism" in tests:
            det = tests["determinism"]
            lines.extend(
                [
                    "#### Determinism",
                    f"- All outputs deterministic: {'✓' if det['all_deterministic'] else '✗'}",
                    f"- Determinism rate: {det['determinism_rate'] * 100:.1f}%",
                    "",
                ]
            )

        if "quality" in tests:
            lines.extend(["#### Quality Checks", ""])
            for check, passed in tests["quality"].items():
                lines.append(f"- {check}: {'✓' if passed else '✗'}")
            lines.append("")

        if "embeddings" in tests:
            emb = tests["embeddings"]
            lines.extend(
                [
                    "#### Embedding Quality",
                    f"- Similar text similarity: {emb['similar_texts_similarity']:.3f}",
                    f"- Different text similarity: {emb['different_texts_similarity']:.3f}",
                    f"- Similarity ordering correct: {'✓' if emb['similarity_check_passed'] else '✗'}",
                    "",
                ]
            )

    # LightEval results
    if "standard_benchmarks" in data:
        lines.extend(format_lighteval_results(data["standard_benchmarks"], "Standard"))

    if "custom_benchmarks" in data:
        lines.extend(format_lighteval_results(data["custom_benchmarks"], "Custom"))

    return lines


def format_lighteval_results(results: Dict[str, Any], prefix: str) -> List[str]:
    """Format LightEval benchmark results."""
    lines = [f"\n### {prefix} LightEval Benchmarks", ""]

    if "results" not in results:
        lines.append("*No results available*\n")
        return lines

    task_results = results["results"]

    # Create a table of results
    lines.extend(["| Task | Metric | Score |", "|------|--------|-------|"])

    for task, metrics in task_results.items():
        for metric, value in metrics.items():
            if isinstance(value, float):
                score = f"{value:.3f}"
            else:
                score = str(value)
            lines.append(f"| {task} | {metric} | {score} |")

    lines.append("")

    # Determinism report if available
    if "determinism_report" in results:
        report = results["determinism_report"]
        lines.extend(
            [
                "#### Determinism Verification",
                f"- Checks performed: {report.get('checks_performed', 0)}",
                f"- Determinism rate: {report.get('determinism_rate', 0) * 100:.1f}%",
                f"- Average generation time: {report.get('average_generation_time', 0):.3f}s",
                "",
            ]
        )

    return lines


def format_speed_key_findings(results: Dict[str, Any]) -> List[str]:
    """Extract and format key findings from speed results."""
    lines = ["### Key Findings", ""]

    # Find fastest and slowest operations
    operation_times = []
    for op_name, op_data in results.items():
        if "times" in op_data and op_data["times"]:
            mean_time = sum(op_data["times"]) / len(op_data["times"])
            operation_times.append((op_name, mean_time))

    if operation_times:
        operation_times.sort(key=lambda x: x[1])
        fastest = operation_times[0]
        slowest = operation_times[-1]

        lines.extend(
            [
                f"- **Fastest operation**: {fastest[0]} ({fastest[1] * 1000:.2f} ms avg)",
                f"- **Slowest operation**: {slowest[0]} ({slowest[1] * 1000:.2f} ms avg)",
                "",
            ]
        )

    # Cache performance
    if "cache_hit" in results and "cache_miss" in results:
        hit_time = sum(results["cache_hit"]["times"]) / len(
            results["cache_hit"]["times"]
        )
        miss_time = sum(results["cache_miss"]["times"]) / len(
            results["cache_miss"]["times"]
        )
        speedup = miss_time / hit_time if hit_time > 0 else 0

        lines.extend(
            [
                f"- **Cache speedup**: {speedup:.2f}x faster with cache hits",
                f"- **Cache hit time**: {hit_time * 1000:.2f} ms",
                f"- **Cache miss time**: {miss_time * 1000:.2f} ms",
                "",
            ]
        )

    # Concurrent scaling
    concurrent_results = [(k, v) for k, v in results.items() if "concurrent" in k]
    if concurrent_results:
        lines.append("- **Concurrent scaling**:")
        for op_name, op_data in concurrent_results:
            if "workers" in op_name and "times" in op_data and op_data["times"]:
                workers = op_name.split("_")[-2]
                throughput = 1.0 / (sum(op_data["times"]) / len(op_data["times"]))
                lines.append(f"  - {workers} workers: {throughput:.2f} ops/s")
        lines.append("")

    return lines


def generate_csv_export(results_file: str, output_file: str):
    """Export benchmark results to CSV format."""
    with open(results_file, "r") as f:
        data = json.load(f)

    rows = []

    # Speed benchmarks
    if "results" in data:
        for op_name, op_data in data["results"].items():
            if "times" in op_data and op_data["times"]:
                row = {
                    "timestamp": data.get("timestamp", ""),
                    "benchmark_type": "speed",
                    "operation": op_name,
                    "iterations": op_data.get("iterations", 0),
                    "mean_time_ms": sum(op_data["times"])
                    / len(op_data["times"])
                    * 1000,
                    "min_time_ms": min(op_data["times"]) * 1000,
                    "max_time_ms": max(op_data["times"]) * 1000,
                    "memory_peak_mb": max(op_data.get("memory_usage", [0])),
                    "errors": op_data.get("errors", 0),
                    "cache_hits": op_data.get("cache_hits", 0),
                    "cache_misses": op_data.get("cache_misses", 0),
                }
                rows.append(row)

    # Accuracy benchmarks
    if "tests" in data:
        for test_name, test_results in data["tests"].items():
            for metric, value in test_results.items():
                row = {
                    "timestamp": data.get("timestamp", ""),
                    "benchmark_type": "accuracy",
                    "operation": f"{test_name}_{metric}",
                    "value": value,
                }
                rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"Results exported to CSV: {output_file}")
    else:
        print("No data to export")


def compare_runs(run1_file: str, run2_file: str, output_file: str):
    """Compare two benchmark runs and generate a comparison report."""
    with open(run1_file, "r") as f:
        run1 = json.load(f)

    with open(run2_file, "r") as f:
        run2 = json.load(f)

    report_lines = [
        "# Benchmark Comparison Report",
        "",
        f"**Run 1**: {Path(run1_file).stem} ({run1.get('timestamp', 'Unknown')})",
        f"**Run 2**: {Path(run2_file).stem} ({run2.get('timestamp', 'Unknown')})",
        "",
        "## Performance Comparison",
        "",
        "| Operation | Run 1 (ms) | Run 2 (ms) | Change (%) |",
        "|-----------|------------|------------|------------|",
    ]

    # Compare speed results
    if "results" in run1 and "results" in run2:
        results1 = run1["results"]
        results2 = run2["results"]

        common_ops = set(results1.keys()) & set(results2.keys())

        for op in sorted(common_ops):
            if "times" in results1[op] and "times" in results2[op]:
                time1 = sum(results1[op]["times"]) / len(results1[op]["times"]) * 1000
                time2 = sum(results2[op]["times"]) / len(results2[op]["times"]) * 1000
                change = ((time2 - time1) / time1) * 100

                # Format with color coding
                if change > 10:
                    change_str = f"🔴 +{change:.1f}%"
                elif change < -10:
                    change_str = f"🟢 {change:.1f}%"
                else:
                    change_str = f"🟡 {change:+.1f}%"

                report_lines.append(
                    f"| {op} | {time1:.2f} | {time2:.2f} | {change_str} |"
                )

    report_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- 🟢 Improvement (>10% faster)",
            "- 🟡 Similar performance (±10%)",
            "- 🔴 Regression (>10% slower)",
        ]
    )

    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Comparison report generated: {output_file}")

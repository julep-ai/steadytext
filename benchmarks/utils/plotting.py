"""Plotting utilities for benchmark visualization."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_speed_results(results_file: str, output_dir: Optional[str] = None) -> None:
    """Create plots from speed benchmark results."""
    with open(results_file, "r") as f:
        data = json.load(f)

    results = data["results"]

    if output_dir is None:
        output_path = Path(results_file).parent
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Operation timing comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    operations = []
    mean_times = []
    p95_times = []
    p99_times = []

    for op_name, op_data in results.items():
        if "times" in op_data and op_data["times"]:
            operations.append(op_name)
            times_ms = [t * 1000 for t in op_data["times"]]
            mean_times.append(np.mean(times_ms))
            p95_times.append(np.percentile(times_ms, 95))
            p99_times.append(np.percentile(times_ms, 99))

    x = np.arange(len(operations))
    width = 0.25

    ax.bar(x - width, mean_times, width, label="Mean", alpha=0.8)
    ax.bar(x, p95_times, width, label="P95", alpha=0.8)
    ax.bar(x + width, p99_times, width, label="P99", alpha=0.8)

    ax.set_ylabel("Time (ms)")
    ax.set_title("Operation Timing Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(operations, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "operation_timing.png", dpi=300)
    plt.close()

    # 2. Throughput comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    throughputs = []
    for op_name, op_data in results.items():
        if "times" in op_data and op_data["times"]:
            mean_time = np.mean(op_data["times"])
            throughput = 1.0 / mean_time if mean_time > 0 else 0
            throughputs.append((op_name, throughput))

    throughputs.sort(key=lambda x: x[1], reverse=True)
    ops, tps = zip(*throughputs[:10])  # Top 10

    ax.barh(ops, tps, alpha=0.8)
    ax.set_xlabel("Throughput (operations/second)")
    ax.set_title("Top 10 Operations by Throughput")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path / "throughput_comparison.png", dpi=300)
    plt.close()

    # 3. Memory usage
    fig, ax = plt.subplots(figsize=(10, 6))

    operations = []
    memory_peaks = []

    for op_name, op_data in results.items():
        if "memory_usage" in op_data and op_data["memory_usage"]:
            operations.append(op_name)
            memory_peaks.append(max(op_data["memory_usage"]))

    if operations:
        ax.bar(operations, memory_peaks, alpha=0.8)
        ax.set_ylabel("Peak Memory Usage (MB)")
        ax.set_title("Memory Usage by Operation")
        ax.set_xticklabels(operations, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_path / "memory_usage.png", dpi=300)
        plt.close()

    # 4. Cache performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Cache hit rates
    cache_ops = []
    hit_rates = []

    for op_name, op_data in results.items():
        if "cache_hits" in op_data or "cache_misses" in op_data:
            hits = op_data.get("cache_hits", 0)
            misses = op_data.get("cache_misses", 0)
            total = hits + misses
            if total > 0:
                cache_ops.append(op_name)
                hit_rates.append(hits / total * 100)

    if cache_ops:
        ax1.bar(cache_ops, hit_rates, alpha=0.8)
        ax1.set_ylabel("Cache Hit Rate (%)")
        ax1.set_title("Cache Hit Rates by Operation")
        ax1.set_xticklabels(cache_ops, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.set_ylim(0, 105)

    # Cache vs no-cache timing
    if "cache_hit" in results and "cache_miss" in results:
        hit_times = [t * 1000 for t in results["cache_hit"]["times"]]
        miss_times = [t * 1000 for t in results["cache_miss"]["times"]]

        ax2.boxplot([miss_times, hit_times], labels=["Cache Miss", "Cache Hit"])
        ax2.set_ylabel("Time (ms)")
        ax2.set_title("Cache Performance Impact")
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path / "cache_performance.png", dpi=300)
    plt.close()

    # 5. Concurrent scaling
    fig, ax = plt.subplots(figsize=(10, 6))

    worker_counts = []
    throughputs = []

    for op_name, op_data in results.items():
        if "concurrent" in op_name and "workers" in op_name:
            workers = int(op_name.split("_")[-2])
            mean_time = np.mean(op_data["times"]) if op_data["times"] else 0
            throughput = 1.0 / mean_time if mean_time > 0 else 0
            worker_counts.append(workers)
            throughputs.append(throughput)

    if worker_counts:
        sorted_data = sorted(zip(worker_counts, throughputs))
        workers, tps = zip(*sorted_data)

        ax.plot(workers, tps, "o-", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Workers")
        ax.set_ylabel("Throughput (ops/sec)")
        ax.set_title("Concurrent Request Scaling")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(workers)

        # Add ideal scaling line
        if len(workers) > 1:
            base_throughput = tps[0]
            ideal_scaling = [base_throughput * w for w in workers]
            ax.plot(workers, ideal_scaling, "--", alpha=0.5, label="Ideal Scaling")
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / "concurrent_scaling.png", dpi=300)
    plt.close()

    print(f"Plots saved to: {output_path}")


def plot_accuracy_results(results_file: str, output_dir: Optional[str] = None) -> None:
    """Create plots from accuracy benchmark results."""
    with open(results_file, "r") as f:
        data = json.load(f)

    if output_dir is None:
        output_dir_path = Path(results_file).parent
    else:
        output_dir_path = Path(output_dir)

    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Check what type of results we have
    if "tests" in data:
        # Simple accuracy tests
        plot_simple_accuracy_results(data["tests"], output_dir_path)

    if "standard_benchmarks" in data:
        # LightEval standard benchmarks
        plot_lighteval_results(data["standard_benchmarks"], output_dir_path, "standard")

    if "custom_benchmarks" in data:
        # Custom benchmarks
        plot_lighteval_results(data["custom_benchmarks"], output_dir_path, "custom")

    print(f"Plots saved to: {output_dir_path}")


def plot_simple_accuracy_results(tests: Dict[str, Any], output_dir: Path) -> None:
    """Plot results from simple accuracy tests."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Determinism results
    ax = axes[0]
    if "determinism" in tests:
        det_data = tests["determinism"]
        labels = ["Deterministic", "Non-deterministic"]
        sizes = [det_data["determinism_rate"], 1 - det_data["determinism_rate"]]
        colors = ["green", "red"]

        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
        ax.set_title("Determinism Test Results")

    # 2. Quality checks
    ax = axes[1]
    if "quality" in tests:
        quality_data = tests["quality"]
        checks = list(quality_data.keys())
        results = [1 if quality_data[k] else 0 for k in checks]

        ax.bar(checks, results, alpha=0.8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Pass (1) / Fail (0)")
        ax.set_title("Quality Checks")
        ax.set_xticklabels(checks, rotation=45, ha="right")

    # 3. Embedding similarity
    ax = axes[2]
    if "embeddings" in tests:
        emb_data = tests["embeddings"]
        similarities = [
            ("Similar texts", emb_data["similar_texts_similarity"]),
            ("Different texts", emb_data["different_texts_similarity"]),
        ]

        labels, values = zip(*similarities)
        ax.bar(labels, values, alpha=0.8)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Embedding Similarity Test")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "simple_accuracy_results.png", dpi=300)
    plt.close()


def plot_lighteval_results(
    results: Dict[str, Any], output_dir: Path, prefix: str
) -> None:
    """Plot LightEval benchmark results."""
    if "results" not in results:
        return

    task_results = results["results"]

    # Extract metrics for each task
    task_names = []
    metric_values = {}

    for task, metrics in task_results.items():
        task_names.append(task)
        for metric, value in metrics.items():
            if metric not in metric_values:
                metric_values[metric] = []
            metric_values[metric].append(value)

    # Create a plot for each metric
    for metric, values in metric_values.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(task_names, values, alpha=0.8)
        ax.set_ylabel(metric)
        ax.set_title(f"{prefix.title()} Benchmarks - {metric}")
        ax.set_xticklabels(task_names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        safe_metric = metric.replace("/", "_").replace(":", "_")
        plt.savefig(output_dir / f"{prefix}_{safe_metric}.png", dpi=300)
        plt.close()

    # Determinism report if available
    if "determinism_report" in results:
        report = results["determinism_report"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Determinism rate
        if "determinism_rate" in report:
            labels = ["Deterministic", "Non-deterministic"]
            sizes = [report["determinism_rate"], 1 - report["determinism_rate"]]
            colors = ["green", "red"]

            ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
            ax1.set_title("Determinism Verification")

        # Generation times
        if "average_generation_time" in report:
            ax2.bar(["Average"], [report["average_generation_time"]], alpha=0.8)
            ax2.set_ylabel("Time (seconds)")
            ax2.set_title("Average Generation Time")
            ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_determinism_report.png", dpi=300)
        plt.close()


def create_comparison_plot(results_files: List[str], output_file: str) -> None:
    """Create comparison plots across multiple benchmark runs."""
    all_data = []

    for file in results_files:
        with open(file, "r") as f:
            data = json.load(f)
            data["filename"] = Path(file).stem
            all_data.append(data)

    # Extract common metrics for comparison
    df_data = []

    for data in all_data:
        run_name = data["filename"]
        timestamp = data.get("timestamp", "unknown")

        # Extract speed metrics if available
        if "results" in data:
            for op, metrics in data["results"].items():
                if "times" in metrics and metrics["times"]:
                    df_data.append(
                        {
                            "run": run_name,
                            "timestamp": timestamp,
                            "operation": op,
                            "mean_time_ms": np.mean(metrics["times"]) * 1000,
                            "p95_time_ms": np.percentile(metrics["times"], 95) * 1000,
                            "type": "speed",
                        }
                    )

    if df_data:
        df = pd.DataFrame(df_data)

        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 8))

        # Group by operation and plot
        operations = df["operation"].unique()
        x = np.arange(len(operations))
        width = 0.8 / len(all_data)

        for i, run in enumerate(df["run"].unique()):
            run_data = df[df["run"] == run]
            means = [
                (
                    run_data[run_data["operation"] == op]["mean_time_ms"].values[0]
                    if len(run_data[run_data["operation"] == op]) > 0
                    else 0
                )
                for op in operations
            ]

            ax.bar(x + i * width, means, width, label=run, alpha=0.8)

        ax.set_ylabel("Mean Time (ms)")
        ax.set_title("Performance Comparison Across Runs")
        ax.set_xticks(x + width * (len(all_data) - 1) / 2)
        ax.set_xticklabels(operations, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"Comparison plot saved to: {output_file}")

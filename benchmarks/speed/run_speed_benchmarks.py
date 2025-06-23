#!/usr/bin/env python3
"""Run SteadyText speed benchmarks."""

import json
import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.speed.benchmark_speed import (
    SpeedBenchmark,
    format_result,
    format_results_table,
)
import steadytext


def load_test_data():
    """Load test prompts and texts for benchmarking."""
    # Default test prompts of varying lengths
    prompts = [
        "Hello world",
        "Write a function to calculate fibonacci numbers",
        "Explain quantum computing in simple terms",
        "What are the key principles of object-oriented programming?",
        "Describe the process of photosynthesis in detail, including the light and dark reactions",
        "Compare and contrast different sorting algorithms including their time and space complexity",
        "Analyze the themes in Shakespeare's Hamlet and their relevance to modern society",
        "Explain how neural networks work, from basic perceptrons to deep learning architectures",
    ]

    # Test texts for embeddings
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a high-level programming language",
        "Data structures and algorithms are fundamental to computer science",
        "Natural language processing enables computers to understand human language",
        "Deep learning has revolutionized computer vision applications",
        "Reinforcement learning teaches agents through reward and punishment",
        "Transfer learning allows models to leverage pre-trained knowledge",
        "Gradient descent optimizes neural network parameters",
        "Attention mechanisms improved sequence-to-sequence models",
    ] * 10  # Repeat to have more data

    return prompts, texts


def run_all_benchmarks(args):
    """Run all speed benchmarks."""
    print(f"\n{'=' * 80}")
    print(
        f"SteadyText Speed Benchmarks - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"{'=' * 80}\n")

    # Preload models if requested
    if not args.skip_warmup:
        print("Preloading models...")
        steadytext.preload_models(verbose=True)
        print()

    benchmark = SpeedBenchmark(warmup_iterations=args.warmup_iterations)
    prompts, texts = load_test_data()

    results = {}
    all_results = []

    # 1. Model loading benchmark
    if not args.skip_model_loading:
        print("\n1. Benchmarking model loading...")
        result = benchmark.benchmark_model_loading(
            iterations=args.model_loading_iterations
        )
        results["model_loading"] = result.__dict__
        all_results.append(result)
        print(format_result(result))

    # 2. Generation benchmark
    print("\n2. Benchmarking text generation...")
    result = benchmark.benchmark_generation(
        prompts=prompts,
        iterations=args.generation_iterations,
        warmup=not args.skip_warmup,
    )
    results["generation"] = result.__dict__
    all_results.append(result)
    print(format_result(result))

    # 3. Streaming generation benchmark
    if not args.skip_streaming:
        print("\n3. Benchmarking streaming generation...")
        result = benchmark.benchmark_generation_streaming(
            prompts=prompts,
            iterations=args.streaming_iterations,
            warmup=not args.skip_warmup,
        )
        results["generation_streaming"] = result.__dict__
        all_results.append(result)
        print(format_result(result))

    # 4. Embedding benchmark
    print("\n4. Benchmarking embeddings...")
    embedding_results = benchmark.benchmark_embedding(
        texts=texts,
        iterations=args.embedding_iterations,
        warmup=not args.skip_warmup,
        batch_sizes=args.batch_sizes,
    )
    for batch_size, result in embedding_results.items():
        results[f"embedding_batch_{batch_size}"] = result.__dict__
        all_results.append(result)
        print(format_result(result))

    # 5. Cache performance benchmark
    if not args.skip_cache:
        print("\n5. Benchmarking cache performance...")
        miss_result, hit_result = benchmark.benchmark_cache_performance(
            prompts=prompts[:20],  # Use subset for cache testing
            iterations=100,
        )
        results["cache_miss"] = miss_result.__dict__
        results["cache_hit"] = hit_result.__dict__
        all_results.extend([miss_result, hit_result])
        print(format_result(miss_result))
        print(format_result(hit_result))

    # 6. Concurrent request benchmark
    if not args.skip_concurrent:
        print("\n6. Benchmarking concurrent requests...")
        concurrent_results = benchmark.benchmark_concurrent(
            operation=steadytext.generate,
            args_list=[(p,) for p in prompts],
            max_workers=args.max_workers,
            iterations_per_worker=args.concurrent_iterations,
        )
        for workers, result in concurrent_results.items():
            results[f"concurrent_{workers}_workers"] = result.__dict__
            all_results.append(result)
            print(format_result(result))

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(format_results_table(all_results))

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save detailed JSON results
        with open(output_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "args": vars(args),
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {output_path}")

        # Save summary markdown
        summary_path = output_path.with_suffix(".md")
        with open(summary_path, "w") as f:
            f.write("# SteadyText Speed Benchmark Results\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n\n")
            f.write("```\n")
            f.write(format_results_table(all_results))
            f.write("\n```\n\n")

            f.write("## Detailed Results\n\n")
            for result in all_results:
                f.write(f"### {result.operation}\n\n")
                f.write("```\n")
                f.write(format_result(result))
                f.write("\n```\n\n")

        print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run SteadyText speed benchmarks")

    # Iteration counts
    parser.add_argument(
        "--generation-iterations",
        type=int,
        default=100,
        help="Number of generation benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--streaming-iterations",
        type=int,
        default=50,
        help="Number of streaming benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--embedding-iterations",
        type=int,
        default=100,
        help="Number of embedding benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--model-loading-iterations",
        type=int,
        default=3,
        help="Number of model loading iterations (default: 3)",
    )
    parser.add_argument(
        "--concurrent-iterations",
        type=int,
        default=10,
        help="Iterations per worker for concurrent benchmarks (default: 10)",
    )

    # Configuration
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 10, 50],
        help="Batch sizes for embedding benchmarks (default: 1 10 50)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Worker counts for concurrent benchmarks (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )

    # Skip options
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warmup runs")
    parser.add_argument(
        "--skip-model-loading", action="store_true", help="Skip model loading benchmark"
    )
    parser.add_argument(
        "--skip-streaming",
        action="store_true",
        help="Skip streaming generation benchmark",
    )
    parser.add_argument(
        "--skip-cache", action="store_true", help="Skip cache performance benchmark"
    )
    parser.add_argument(
        "--skip-concurrent",
        action="store_true",
        help="Skip concurrent request benchmark",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"benchmarks/speed/results/speed_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Output file path for results",
    )

    args = parser.parse_args()

    try:
        run_all_benchmarks(args)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running benchmarks: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

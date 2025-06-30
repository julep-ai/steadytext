#!/usr/bin/env python3
"""Run SteadyText accuracy benchmarks using LightEval."""

import json
import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from benchmarks.accuracy.lighteval_runner import (
        SteadyTextEvaluator,
        SteadyTextConfig,
        LIGHTEVAL_AVAILABLE,
    )
    from benchmarks.accuracy.custom_tasks import register_steadytext_tasks
except ImportError as e:
    print(f"Error importing modules: {e}")
    LIGHTEVAL_AVAILABLE = False

import steadytext


def run_standard_benchmarks(args):
    """Run standard LightEval benchmarks."""
    print("\nRunning standard NLP benchmarks...")

    config = SteadyTextConfig(
        model_name="steadytext",
        deterministic=True,
        verify_determinism=args.verify_determinism,
        max_length=512,
    )

    evaluator = SteadyTextEvaluator(config)

    # Select tasks based on args
    if args.tasks:
        tasks = args.tasks
    else:
        # Default standard tasks
        tasks = [
            "leaderboard|truthfulqa:mc|0|0",
            "leaderboard|gsm8k|0|0",
            "leaderboard|hellaswag|0|0",
            "leaderboard|arc:easy|0|0",
        ]

    # Run evaluation
    results = evaluator.evaluate(
        tasks=tasks,
        num_shots=args.num_shots,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_details=args.save_details,
    )

    return results


def run_custom_benchmarks(args):
    """Run SteadyText-specific custom benchmarks."""
    print("\nRunning SteadyText-specific benchmarks...")

    # Register custom tasks
    register_steadytext_tasks()

    config = SteadyTextConfig(
        model_name="steadytext",
        deterministic=True,
        verify_determinism=True,
        max_length=512,
    )

    evaluator = SteadyTextEvaluator(config)

    # Custom tasks
    custom_tasks = [
        "steadytext_determinism",
        "steadytext_consistency",
        "steadytext_fallback",
        "steadytext_performance_regression",
    ]

    # Run evaluation
    results = evaluator.evaluate(
        tasks=custom_tasks,
        num_shots=0,  # Custom tasks don't use few-shot
        batch_size=1,  # Process one at a time for custom metrics
        output_dir=args.output_dir,
        save_details=True,
    )

    return results


def run_simple_accuracy_tests(args):
    """Run simple accuracy tests without LightEval."""
    print("\nRunning simple accuracy tests (LightEval not available)...")

    results = {"timestamp": datetime.now().isoformat(), "tests": {}}

    # Test 1: Determinism
    print("1. Testing determinism...")
    test_prompts = [
        "Write a Python function",
        "Explain machine learning",
        "What is recursion?",
    ]

    determinism_results = []
    for prompt in test_prompts:
        outputs = [steadytext.generate(prompt) for _ in range(3)]
        is_deterministic = all(o == outputs[0] for o in outputs)
        determinism_results.append(is_deterministic)

    results["tests"]["determinism"] = {
        "all_deterministic": all(determinism_results),
        "determinism_rate": sum(determinism_results) / len(determinism_results),
    }

    # Test 2: Output quality (basic checks)
    print("2. Testing output quality...")
    quality_checks = []

    # Check if code generation produces code-like output
    code_output = steadytext.generate("Write a Python function to sort a list")
    if code_output is not None:
        has_code_markers = any(
            marker in code_output for marker in ["def ", "return", ":", "(", ")"]
        )
        quality_checks.append(("code_generation", has_code_markers))
    else:
        quality_checks.append(("code_generation", False))

    # Check if explanations are reasonable length
    explanation = steadytext.generate("Explain quantum computing")
    if explanation is not None:
        reasonable_length = 100 < len(explanation) < 2000
        quality_checks.append(("explanation_length", reasonable_length))
    else:
        quality_checks.append(("explanation_length", False))

    results["tests"]["quality"] = {check[0]: check[1] for check in quality_checks}

    # Test 3: Embedding quality
    print("3. Testing embeddings...")
    import numpy as np

    # Similar texts should have similar embeddings
    text1 = "Machine learning is a type of artificial intelligence"
    text2 = "ML is a subset of AI"
    text3 = "The weather is nice today"

    emb1 = steadytext.embed(text1)
    emb2 = steadytext.embed(text2)
    emb3 = steadytext.embed(text3)

    # Cosine similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_related = cosine_sim(emb1, emb2)
    sim_unrelated = cosine_sim(emb1, emb3)

    results["tests"]["embeddings"] = {
        "similar_texts_similarity": float(sim_related),
        "different_texts_similarity": float(sim_unrelated),
        "similarity_check_passed": sim_related > sim_unrelated,
    }

    return results


def format_results_summary(results: dict) -> str:
    """Format results as a readable summary."""
    lines = [
        f"\n{'=' * 80}",
        "ACCURACY BENCHMARK RESULTS",
        f"{'=' * 80}\n",
    ]

    if "tests" in results:
        # Simple test results
        lines.append("## Simple Accuracy Tests\n")

        for test_name, test_results in results["tests"].items():
            lines.append(f"### {test_name.title()}")
            for key, value in test_results.items():
                lines.append(f"  - {key}: {value}")
            lines.append("")

    if "results" in results:
        # LightEval results
        lines.append("## LightEval Benchmark Results\n")

        for task, metrics in results["results"].items():
            lines.append(f"### {task}")
            for metric, value in metrics.items():
                lines.append(f"  - {metric}: {value}")
            lines.append("")

    if "determinism_report" in results:
        # Determinism verification
        lines.append("## Determinism Verification\n")
        report = results["determinism_report"]
        lines.append(f"  - Checks performed: {report.get('checks_performed', 0)}")
        lines.append(f"  - Determinism rate: {report.get('determinism_rate', 0):.2%}")
        lines.append(
            f"  - Avg generation time: {report.get('average_generation_time', 0):.3f}s"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run SteadyText accuracy benchmarks")

    # Benchmark selection
    parser.add_argument(
        "--benchmarks",
        choices=["standard", "custom", "all", "simple"],
        default="all",
        help="Which benchmarks to run (default: all)",
    )

    # Task configuration
    parser.add_argument("--tasks", nargs="+", help="Specific LightEval tasks to run")
    parser.add_argument(
        "--num-shots",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)",
    )

    # Options
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Verify determinism during evaluation",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save detailed results for each example",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"benchmarks/accuracy/results/accuracy_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Output file path for results",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Directory for LightEval outputs"
    )

    args = parser.parse_args()

    # Check if LightEval is available
    if not LIGHTEVAL_AVAILABLE and args.benchmarks != "simple":
        print("\nWarning: LightEval is not installed. Install with:")
        print("  pip install lighteval")
        print("\nFalling back to simple accuracy tests...")
        args.benchmarks = "simple"

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"\n{'=' * 80}")
        print(
            f"SteadyText Accuracy Benchmarks - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"{'=' * 80}\n")

        # Preload models
        print("Preloading models...")
        steadytext.preload_models(verbose=True)

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
        }

        # Run selected benchmarks
        if args.benchmarks in ["all", "simple"] and not LIGHTEVAL_AVAILABLE:
            results = run_simple_accuracy_tests(args)
            all_results.update(results)

        elif args.benchmarks in ["standard", "all"]:
            results = run_standard_benchmarks(args)
            all_results["standard_benchmarks"] = results

        if args.benchmarks in ["custom", "all"] and LIGHTEVAL_AVAILABLE:
            results = run_custom_benchmarks(args)
            all_results["custom_benchmarks"] = results

        # Print summary
        print(format_results_summary(all_results))

        # Save results
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")

        # Save markdown summary
        summary_path = output_path.with_suffix(".md")
        with open(summary_path, "w") as f:
            f.write("# SteadyText Accuracy Benchmark Results\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(format_results_summary(all_results))

        print(f"Summary saved to: {summary_path}")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running benchmarks: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

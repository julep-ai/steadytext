#!/usr/bin/env python3
"""Quick test to verify benchmarks are working."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_speed_benchmark():
    """Test basic speed benchmark functionality."""
    print("Testing speed benchmark module...")

    try:
        from benchmarks.speed.benchmark_speed import SpeedBenchmark, format_result
        import steadytext

        # Preload models
        print("Preloading models...")
        steadytext.preload_models(verbose=False)

        # Create benchmark instance
        benchmark = SpeedBenchmark(warmup_iterations=1)

        # Test generation benchmark
        print("\nTesting generation benchmark...")
        result = benchmark.benchmark_generation(
            prompts=["Hello world", "Test prompt"], iterations=5, warmup=True
        )

        print(format_result(result))

        # Test embedding benchmark
        print("\nTesting embedding benchmark...")
        embedding_results = benchmark.benchmark_embedding(
            texts=["Test text 1", "Test text 2"], iterations=5, batch_sizes=[1, 2]
        )

        for batch_size, result in embedding_results.items():
            print(f"\nBatch size {batch_size}:")
            print(f"  Mean time: {result.mean_time * 1000:.2f} ms")
            print(f"  Throughput: {result.throughput:.2f} ops/sec")

        print("\n✓ Speed benchmarks working correctly")

    except Exception as e:
        print(f"✗ Speed benchmark test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_accuracy_benchmark():
    """Test basic accuracy benchmark functionality."""
    print("\n\nTesting accuracy benchmark module...")

    try:
        # Test simple accuracy tests (no LightEval required)
        print("\nRunning simple accuracy tests...")

        import steadytext
        import numpy as np

        # Test determinism
        prompt = "Test determinism"
        outputs = [steadytext.generate(prompt) for _ in range(3)]
        all_same = all(o == outputs[0] for o in outputs)
        print(f"  Determinism test: {'✓' if all_same else '✗'}")

        # Test embeddings
        emb1 = steadytext.embed("Machine learning")
        emb2 = steadytext.embed("ML")
        emb3 = steadytext.embed("Weather")

        # Cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_related = cosine_sim(emb1, emb2)
        sim_unrelated = cosine_sim(emb1, emb3)

        print(f"  Related similarity: {sim_related:.3f}")
        print(f"  Unrelated similarity: {sim_unrelated:.3f}")
        print(f"  Similarity ordering: {'✓' if sim_related > sim_unrelated else '✗'}")

        # Try LightEval if available
        try:
            from benchmarks.accuracy.lighteval_runner import LIGHTEVAL_AVAILABLE

            if LIGHTEVAL_AVAILABLE:
                print("\n  LightEval is available")
            else:
                print(
                    "\n  LightEval not available (install with: pip install lighteval)"
                )
        except Exception:
            print("\n  Could not check LightEval availability")

        print("\n✓ Accuracy benchmarks working correctly")

    except Exception as e:
        print(f"✗ Accuracy benchmark test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def main():
    """Run all benchmark tests."""
    print("=" * 60)
    print("SteadyText Benchmark Test Suite")
    print("=" * 60)

    all_passed = True

    # Test speed benchmarks
    if not test_speed_benchmark():
        all_passed = False

    # Test accuracy benchmarks
    if not test_accuracy_benchmark():
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All benchmark tests passed!")
        print("\nYou can now run the full benchmarks:")
        print("  python benchmarks/run_all_benchmarks.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

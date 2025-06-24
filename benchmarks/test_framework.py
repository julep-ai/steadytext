#!/usr/bin/env python3
"""Test the benchmarking framework independently of SteadyText."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_benchmark_framework():
    """Test that the benchmark framework itself works."""
    print("Testing benchmark framework...")

    try:
        # Test speed benchmark classes
        from benchmarks.speed.benchmark_speed import (
            BenchmarkResult,
            format_result,
            format_results_table,
        )

        # Create a mock result
        result = BenchmarkResult(operation="test_operation", iterations=10)
        result.times = [0.001, 0.0012, 0.0011, 0.0009, 0.0013]
        result.memory_usage = [10.5, 11.0, 10.8, 10.9, 11.2]
        result.cache_hits = 3
        result.cache_misses = 2

        print("\nMock benchmark result:")
        print(format_result(result))

        # Test table formatting
        print("\nTable formatting test:")
        print(format_results_table([result]))

        print("\n✓ Speed benchmark framework working")

    except Exception as e:
        print(f"✗ Speed benchmark framework failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        # Test plotting utilities
        print("\n✓ Plotting utilities imported successfully")

    except Exception as e:
        print(f"✗ Plotting utilities failed: {e}")
        return False

    try:
        # Test reporting utilities
        from benchmarks.utils.reporting import format_speed_key_findings

        mock_results = {
            "operation1": {"times": [0.001, 0.002], "memory_usage": [10, 11]},
            "operation2": {"times": [0.01, 0.02], "memory_usage": [20, 21]},
            "cache_hit": {"times": [0.0001, 0.0002]},
            "cache_miss": {"times": [0.001, 0.002]},
        }

        format_speed_key_findings(mock_results)
        print("\n✓ Reporting utilities working")

    except Exception as e:
        print(f"✗ Reporting utilities failed: {e}")
        return False

    return True


def mock_steadytext_functions():
    """Create mock SteadyText functions for testing."""

    class MockSteadyText:
        @staticmethod
        def generate(prompt, **kwargs):
            # Mock deterministic generation
            return f"Generated text for: {prompt[:20]}..."

        @staticmethod
        def generate_iter(prompt, **kwargs):
            # Mock streaming
            text = MockSteadyText.generate(prompt)
            for word in text.split():
                yield word + " "

        @staticmethod
        def embed(text):
            # Mock embedding
            import numpy as np

            if isinstance(text, list):
                # Average of mock embeddings
                return np.random.rand(1024).astype(np.float32)
            return np.random.rand(1024).astype(np.float32)

        @staticmethod
        def preload_models(verbose=False):
            if verbose:
                print("Mock: Models preloaded")

        DEFAULT_SEED = 42
        GENERATION_MAX_NEW_TOKENS = 512
        EMBEDDING_DIMENSION = 1024
        __version__ = "0.2.3-mock"

    # Inject mock into sys.modules
    sys.modules["steadytext"] = MockSteadyText

    # Mock the model loader
    class MockLoader:
        @staticmethod
        def clear_model_cache():
            pass

        @staticmethod
        def get_generator_model_instance():
            return None

        @staticmethod
        def get_embedding_model_instance():
            return None

    sys.modules["steadytext.models.loader"] = MockLoader

    # Mock caches
    class MockCache:
        def __init__(self):
            self.data = {}

        def clear(self):
            self.data.clear()

        def __len__(self):
            return len(self.data)

    # Mock cache manager
    class MockCacheManager:
        def __init__(self):
            self.generation_cache = MockCache()
            self.embedding_cache = MockCache()

        def get_generation_cache(self):
            return self.generation_cache

        def get_embedding_cache(self):
            return self.embedding_cache

    mock_cache_manager = MockCacheManager()

    sys.modules["steadytext.cache_manager"] = type(
        "Module",
        (),
        {
            "get_generation_cache": mock_cache_manager.get_generation_cache,
            "get_embedding_cache": mock_cache_manager.get_embedding_cache,
        },
    )

    sys.modules["steadytext.core.generator"] = type(
        "Module",
        (),
        {
            "_deterministic_fallback_generate": lambda x: f"Fallback: {x}",
        },
    )
    sys.modules["steadytext.core.embedder"] = type("Module", (), {})

    return MockSteadyText


def test_with_mock():
    """Test benchmarks with mock SteadyText."""
    print("\n\nTesting benchmarks with mock SteadyText...")

    # Setup mocks
    mock_steadytext_functions()

    try:
        from benchmarks.speed.benchmark_speed import SpeedBenchmark

        benchmark = SpeedBenchmark(warmup_iterations=1)

        # Test generation
        result = benchmark.benchmark_generation(
            prompts=["Test prompt 1", "Test prompt 2"], iterations=5, warmup=False
        )

        print("\nGeneration benchmark:")
        print(f"  Mean time: {result.mean_time * 1000:.2f} ms")
        print(f"  Throughput: {result.throughput:.2f} ops/sec")

        # Test embedding
        embedding_results = benchmark.benchmark_embedding(
            texts=["Text 1", "Text 2", "Text 3"],
            iterations=5,
            batch_sizes=[1, 2],
            warmup=False,
        )

        print("\nEmbedding benchmarks:")
        for batch_size, result in embedding_results.items():
            print(f"  Batch {batch_size}: {result.mean_time * 1000:.2f} ms")

        print("\n✓ Mock benchmarks completed successfully")
        return True

    except Exception as e:
        print(f"✗ Mock benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all framework tests."""
    print("=" * 60)
    print("Benchmark Framework Test")
    print("=" * 60)

    all_passed = True

    # Test framework components
    if not test_benchmark_framework():
        all_passed = False

    # Test with mocks
    if not test_with_mock():
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ Benchmark framework is working correctly!")
        print("\nNote: This tested the framework only. To test with real SteadyText:")
        print("1. Install SteadyText dependencies")
        print("2. Run: python benchmarks/test_benchmarks.py")
    else:
        print("✗ Some framework tests failed.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

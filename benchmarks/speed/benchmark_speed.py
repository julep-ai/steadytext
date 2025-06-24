"""Core speed benchmarking module for SteadyText.

This module provides utilities to measure:
- Generation speed (tokens/second)
- Embedding speed (embeddings/second)
- Cache performance
- Memory usage
- Concurrent request handling
"""

import time
import gc
import statistics
from typing import List, Dict, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import psutil
import os


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    operation: str
    iterations: int
    times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0

    @property
    def mean_time(self) -> float:
        return statistics.mean(self.times) if self.times else 0.0

    @property
    def median_time(self) -> float:
        return statistics.median(self.times) if self.times else 0.0

    @property
    def p95_time(self) -> float:
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99_time(self) -> float:
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def throughput(self) -> float:
        """Operations per second."""
        if not self.times or self.mean_time == 0:
            return 0.0
        return 1.0 / self.mean_time

    @property
    def memory_peak_mb(self) -> float:
        return max(self.memory_usage) if self.memory_usage else 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class SpeedBenchmark:
    """Main speed benchmarking class."""

    def __init__(self, warmup_iterations: int = 5):
        self.warmup_iterations = warmup_iterations
        self.process = psutil.Process(os.getpid())

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def _warmup(self, func: Callable, *args, **kwargs):
        """Perform warmup runs."""
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        gc.collect()

    def benchmark_generation(
        self, prompts: List[str], iterations: int = 100, warmup: bool = True
    ) -> BenchmarkResult:
        """Benchmark text generation speed."""
        import steadytext

        result = BenchmarkResult(operation="generation", iterations=iterations)

        if warmup:
            self._warmup(steadytext.generate, prompts[0])

        # Clear cache to measure cold performance
        from steadytext.cache_manager import get_generation_cache

        generation_cache = get_generation_cache()
        initial_size = len(generation_cache)
        generation_cache.clear()

        for i in range(iterations):
            prompt = prompts[i % len(prompts)]

            mem_before = self._get_memory_mb()
            start_time = time.perf_counter()

            try:
                steadytext.generate(prompt)
                elapsed = time.perf_counter() - start_time
                result.times.append(elapsed)

                # Check if this was a cache hit
                cache_size_after = len(generation_cache)
                if cache_size_after == initial_size:
                    result.cache_hits += 1
                else:
                    result.cache_misses += 1
                    initial_size = cache_size_after

            except Exception:
                result.errors += 1
                continue

            mem_after = self._get_memory_mb()
            result.memory_usage.append(mem_after - mem_before)

        return result

    def benchmark_generation_streaming(
        self, prompts: List[str], iterations: int = 50, warmup: bool = True
    ) -> BenchmarkResult:
        """Benchmark streaming generation speed."""
        import steadytext

        result = BenchmarkResult(
            operation="generation_streaming", iterations=iterations
        )

        if warmup:
            # Warmup with streaming
            list(steadytext.generate_iter(prompts[0]))

        for i in range(iterations):
            prompt = prompts[i % len(prompts)]

            mem_before = self._get_memory_mb()
            start_time = time.perf_counter()

            try:
                tokens = []
                for token in steadytext.generate_iter(prompt):
                    tokens.append(token)

                elapsed = time.perf_counter() - start_time
                result.times.append(elapsed)

            except Exception:
                result.errors += 1
                continue

            mem_after = self._get_memory_mb()
            result.memory_usage.append(mem_after - mem_before)

        return result

    def benchmark_embedding(
        self,
        texts: List[str],
        iterations: int = 100,
        warmup: bool = True,
        batch_sizes: List[int] = [1, 10, 50],
    ) -> Dict[int, BenchmarkResult]:
        """Benchmark embedding speed with different batch sizes."""
        import steadytext

        results = {}

        for batch_size in batch_sizes:
            result = BenchmarkResult(
                operation=f"embedding_batch_{batch_size}", iterations=iterations
            )

            if warmup:
                self._warmup(steadytext.embed, texts[:batch_size])

            # Clear embedding cache
            from steadytext.cache_manager import get_embedding_cache

            embedding_cache = get_embedding_cache()
            embedding_cache.clear()

            for i in range(iterations):
                batch_start = (i * batch_size) % len(texts)
                batch_end = min(batch_start + batch_size, len(texts))
                batch = texts[batch_start:batch_end]

                mem_before = self._get_memory_mb()
                start_time = time.perf_counter()

                try:
                    if batch_size == 1:
                        steadytext.embed(batch[0])
                    else:
                        steadytext.embed(batch)

                    elapsed = time.perf_counter() - start_time
                    result.times.append(elapsed)

                except Exception:
                    result.errors += 1
                    continue

                mem_after = self._get_memory_mb()
                result.memory_usage.append(mem_after - mem_before)

            results[batch_size] = result

        return results

    def benchmark_concurrent(
        self,
        operation: Callable,
        args_list: List[Tuple],
        max_workers: List[int] = [1, 2, 4, 8],
        iterations_per_worker: int = 10,
    ) -> Dict[int, BenchmarkResult]:
        """Benchmark concurrent request handling."""
        results = {}

        for workers in max_workers:
            # Get operation name safely
            operation_name = getattr(operation, "__name__", str(operation))
            result = BenchmarkResult(
                operation=f"concurrent_{operation_name}_workers_{workers}",
                iterations=workers * iterations_per_worker,
            )

            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []

                for i in range(workers * iterations_per_worker):
                    args = args_list[i % len(args_list)]
                    future = executor.submit(operation, *args)
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        _ = future.result()
                    except Exception:
                        result.errors += 1

            total_time = time.perf_counter() - start_time
            result.times = [
                total_time / (workers * iterations_per_worker)
            ] * result.iterations

            results[workers] = result

        return results

    def benchmark_model_loading(self, iterations: int = 5) -> BenchmarkResult:
        """Benchmark model loading time."""
        import steadytext

        result = BenchmarkResult(operation="model_loading", iterations=iterations)

        from steadytext.models.loader import clear_model_cache

        for _ in range(iterations):
            # Clear model cache to force reload
            clear_model_cache()
            gc.collect()

            mem_before = self._get_memory_mb()
            start_time = time.perf_counter()

            try:
                # Preload models
                steadytext.preload_models(verbose=False)
                elapsed = time.perf_counter() - start_time
                result.times.append(elapsed)

            except Exception:
                result.errors += 1
                continue

            mem_after = self._get_memory_mb()
            result.memory_usage.append(mem_after - mem_before)

        return result

    def benchmark_cache_performance(
        self, prompts: List[str], iterations: int = 100
    ) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Benchmark cache hit vs miss performance."""
        import steadytext

        # Clear cache first
        from steadytext.cache_manager import get_generation_cache

        generation_cache = get_generation_cache()
        generation_cache.clear()

        # First pass - all cache misses
        miss_result = BenchmarkResult(operation="cache_miss", iterations=len(prompts))

        for prompt in prompts:
            start_time = time.perf_counter()
            _ = steadytext.generate(prompt)
            elapsed = time.perf_counter() - start_time
            miss_result.times.append(elapsed)
            miss_result.cache_misses += 1

        # Second pass - all cache hits
        hit_result = BenchmarkResult(operation="cache_hit", iterations=len(prompts))

        for prompt in prompts:
            start_time = time.perf_counter()
            _ = steadytext.generate(prompt)
            elapsed = time.perf_counter() - start_time
            hit_result.times.append(elapsed)
            hit_result.cache_hits += 1

        return miss_result, hit_result


def format_result(result: BenchmarkResult) -> str:
    """Format a benchmark result as a readable string."""
    lines = [
        f"\n{'=' * 60}",
        f"Operation: {result.operation}",
        f"Iterations: {result.iterations}",
        f"{'=' * 60}",
        "Timing Statistics:",
        f"  Mean:   {result.mean_time * 1000:.2f} ms",
        f"  Median: {result.median_time * 1000:.2f} ms",
        f"  P95:    {result.p95_time * 1000:.2f} ms",
        f"  P99:    {result.p99_time * 1000:.2f} ms",
        f"  Throughput: {result.throughput:.2f} ops/sec",
        "",
        "Memory Usage:",
        f"  Peak: {result.memory_peak_mb:.2f} MB",
        "",
        "Cache Performance:",
        f"  Hits:   {result.cache_hits}",
        f"  Misses: {result.cache_misses}",
        f"  Hit Rate: {result.cache_hit_rate * 100:.1f}%",
        "",
        f"Errors: {result.errors}",
        f"{'=' * 60}",
    ]
    return "\n".join(lines)


def format_results_table(results: List[BenchmarkResult]) -> str:
    """Format multiple results as a comparison table."""
    from tabulate import tabulate

    headers = [
        "Operation",
        "Mean (ms)",
        "P95 (ms)",
        "P99 (ms)",
        "Throughput",
        "Memory (MB)",
        "Errors",
    ]
    rows = []

    for result in results:
        rows.append(
            [
                result.operation,
                f"{result.mean_time * 1000:.2f}",
                f"{result.p95_time * 1000:.2f}",
                f"{result.p99_time * 1000:.2f}",
                f"{result.throughput:.2f}",
                f"{result.memory_peak_mb:.2f}",
                result.errors,
            ]
        )

    return tabulate(rows, headers=headers, tablefmt="grid")

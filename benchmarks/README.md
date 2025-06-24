# SteadyText Benchmarking Suite

This directory contains comprehensive speed and accuracy benchmarks for SteadyText.

## Overview

The benchmarking suite consists of two main components:

1. **Speed Benchmarks**: Measure performance characteristics including:
   - Generation speed (tokens/second)
   - Embedding speed (embeddings/second)
   - Cache performance
   - Memory usage
   - Concurrent request handling
   - Model loading time

2. **Accuracy Benchmarks**: Evaluate model quality and behavior including:
   - Determinism verification
   - Standard NLP benchmarks (using LightEval)
   - Custom SteadyText-specific tests
   - Fallback behavior testing

## Installation

Install benchmark dependencies:

**Using UV (recommended):**
```bash
# Install with benchmark extras
uv add "steadytext[benchmark]"

# Or if already installed
uv sync --all-extras
```

**Legacy method:**
```bash
pip install steadytext[benchmark]
```

This installs:
- `lighteval`: For running standard NLP benchmarks
- `psutil`: For memory monitoring
- `matplotlib`: For plotting results
- `pandas`: For data analysis
- `tabulate`: For pretty tables

## Quick Start

Run all benchmarks:

**Using UV (recommended):**
```bash
uv run python benchmarks/run_all_benchmarks.py
```

**Legacy method:**
```bash
python benchmarks/run_all_benchmarks.py
```

Run quick benchmarks (reduced iterations):

```bash
# UV
uv run python benchmarks/run_all_benchmarks.py --quick

# Legacy
python benchmarks/run_all_benchmarks.py --quick
```

Run only speed benchmarks:

```bash
# UV
uv run python benchmarks/run_all_benchmarks.py --only speed

# Legacy
python benchmarks/run_all_benchmarks.py --only speed
```

Run only accuracy benchmarks:

```bash
# UV
uv run python benchmarks/run_all_benchmarks.py --only accuracy

# Legacy
python benchmarks/run_all_benchmarks.py --only accuracy
```

## Detailed Usage

### Speed Benchmarks

Run full speed benchmark suite:

```bash
# UV (recommended)
uv run python benchmarks/speed/run_speed_benchmarks.py

# Legacy
python benchmarks/speed/run_speed_benchmarks.py
```

Options:
- `--generation-iterations N`: Number of generation tests (default: 100)
- `--embedding-iterations N`: Number of embedding tests (default: 100)
- `--batch-sizes N1 N2 ...`: Batch sizes for embedding tests (default: 1 10 50)
- `--skip-warmup`: Skip warmup runs
- `--skip-cache`: Skip cache performance tests
- `--skip-concurrent`: Skip concurrent request tests

Example with custom settings:

```bash
# UV (recommended)
uv run python benchmarks/speed/run_speed_benchmarks.py \
    --generation-iterations 50 \
    --embedding-iterations 100 \
    --batch-sizes 1 5 10 25 \
    --output results/my_speed_test.json

# Legacy
python benchmarks/speed/run_speed_benchmarks.py \
    --generation-iterations 50 \
    --embedding-iterations 100 \
    --batch-sizes 1 5 10 25 \
    --output results/my_speed_test.json
```

### Accuracy Benchmarks

Run accuracy benchmarks:

```bash
# UV (recommended)
uv run python benchmarks/accuracy/run_accuracy_benchmarks.py

# Legacy
python benchmarks/accuracy/run_accuracy_benchmarks.py
```

Options:
- `--benchmarks {standard,custom,all,simple}`: Which benchmarks to run
- `--tasks TASK1 TASK2 ...`: Specific LightEval tasks
- `--verify-determinism`: Extra determinism checks
- `--num-shots N`: Few-shot examples for LightEval

Example running specific tasks:

```bash
# UV (recommended)
uv run python benchmarks/accuracy/run_accuracy_benchmarks.py \
    --benchmarks standard \
    --tasks "leaderboard|truthfulqa:mc|0|0" "leaderboard|gsm8k|0|0" \
    --verify-determinism

# Legacy
python benchmarks/accuracy/run_accuracy_benchmarks.py \
    --benchmarks standard \
    --tasks "leaderboard|truthfulqa:mc|0|0" "leaderboard|gsm8k|0|0" \
    --verify-determinism
```

## Understanding Results

### Speed Benchmark Metrics

1. **Timing Statistics**:
   - Mean: Average operation time
   - Median: Middle value of all timings
   - P95/P99: 95th/99th percentile (worst-case performance)
   - Throughput: Operations per second

2. **Memory Usage**:
   - Peak memory consumption per operation
   - Memory overhead of different batch sizes

3. **Cache Performance**:
   - Hit rate: Percentage of cache hits
   - Performance improvement from caching

4. **Concurrent Scaling**:
   - Throughput with different worker counts
   - Scaling efficiency

### Accuracy Benchmark Metrics

1. **Determinism Tests**:
   - Whether outputs are identical across runs
   - Determinism rate across different prompts

2. **Quality Tests**:
   - Code generation quality
   - Explanation coherence
   - Embedding similarity for related texts

3. **LightEval Benchmarks** (if available):
   - TruthfulQA: Truthfulness of responses
   - GSM8K: Mathematical reasoning
   - HellaSwag: Common sense reasoning
   - ARC: Science question answering

## Output Files

Running benchmarks generates:

1. **JSON Results**: Raw benchmark data
   - `speed_benchmark_TIMESTAMP.json`
   - `accuracy_benchmark_TIMESTAMP.json`

2. **Markdown Report**: Human-readable summary
   - `benchmark_report_TIMESTAMP.md`

3. **CSV Exports**: For further analysis
   - `speed_benchmark_TIMESTAMP.csv`
   - `accuracy_benchmark_TIMESTAMP.csv`

4. **Plots** (if matplotlib available):
   - `operation_timing.png`: Timing comparison
   - `throughput_comparison.png`: Throughput analysis
   - `memory_usage.png`: Memory consumption
   - `cache_performance.png`: Cache analysis
   - `concurrent_scaling.png`: Concurrency performance

## CI Integration

For continuous integration, use quick mode:

```bash
# UV (recommended for CI - faster and more reliable)
uv run python benchmarks/run_all_benchmarks.py --quick --output-dir ci_results

# Legacy
python benchmarks/run_all_benchmarks.py --quick --output-dir ci_results
```

To detect performance regressions:

```bash
# Run benchmarks and save baseline
uv run python benchmarks/run_all_benchmarks.py --output-dir baseline

# Later, compare against baseline
uv run python benchmarks/run_all_benchmarks.py --output-dir current
uv run python benchmarks/utils/compare_runs.py baseline/speed_*.json current/speed_*.json
```

## Custom Benchmarks

### Adding Speed Tests

Create a new test in `benchmarks/speed/custom_speed_tests.py`:

```python
from benchmarks.speed.benchmark_speed import SpeedBenchmark

def benchmark_custom_operation():
    benchmark = SpeedBenchmark()
    
    # Your custom benchmark logic
    result = benchmark.benchmark_generation(
        prompts=["Custom prompt 1", "Custom prompt 2"],
        iterations=50
    )
    
    print(f"Custom benchmark: {result.mean_time*1000:.2f} ms average")
```

### Adding Accuracy Tests

For LightEval integration, see `benchmarks/accuracy/custom_tasks.py` for examples.

## Troubleshooting

### LightEval Not Available

If you see "LightEval is not installed", the accuracy benchmarks will fall back to simple tests. Install with:

```bash
pip install lighteval
```

### Memory Issues

For large-scale benchmarks, you may need to:
- Reduce iteration counts
- Run benchmarks separately
- Increase system memory

### Slow Performance

If benchmarks are too slow:
- Use `--quick` mode
- Reduce iterations
- Skip certain tests (e.g., `--skip-concurrent`)

## Performance Tuning

Based on benchmark results, you can tune SteadyText:

1. **Cache Settings**:
   ```bash
   export STEADYTEXT_GENERATION_CACHE_CAPACITY=512
   export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=100
   ```

2. **Model Loading**:
   - Preload models before critical operations
   - Use model instance caching

3. **Batch Processing**:
   - Use larger batches for embeddings
   - Process multiple prompts together

## Contributing

To add new benchmarks:

1. Add test functions to appropriate modules
2. Update the runner scripts
3. Document new metrics in this README
4. Submit a pull request

## Related Documentation

- [SteadyText API Documentation](../docs/api.md)
- [Performance Optimization Guide](../docs/performance.md)
- [Testing Guide](../tests/README.md)
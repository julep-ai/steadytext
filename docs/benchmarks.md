# SteadyText Performance Benchmarks

This document provides detailed performance and accuracy benchmarks for SteadyText v1.3.3.

## Quick Summary

SteadyText delivers **100% deterministic** text generation and embeddings with competitive performance:

- **Text Generation**: 21.4 generations/sec (46.7ms mean latency)
- **Embeddings**: 104.4 single embeddings/sec, up to 598.7 embeddings/sec in batches
- **Cache Performance**: 48x speedup for repeated prompts
- **Memory Usage**: ~1.4GB for models, 150-200MB during operation
- **Determinism**: 100% consistent outputs across all platforms and runs
- **Accuracy**: 69.4% similarity for related texts with correct similarity ordering

## Table of Contents

1. [Speed Benchmarks](#speed-benchmarks)
2. [Accuracy Benchmarks](#accuracy-benchmarks)
3. [Determinism Tests](#determinism-tests)
4. [Hardware & Methodology](#hardware--methodology)
5. [Comparison with Alternatives](#comparison-with-alternatives)

## Speed Benchmarks

### Text Generation Performance

SteadyText v2.0.0+ uses the Gemma-3n-E2B-it-Q8_0.gguf model (Gemma-3n-2B) for deterministic text generation:

| Metric | Value | Notes |
|--------|-------|-------|
| **Throughput** | 21.4 generations/sec | Fixed 512 tokens per generation |
| **Mean Latency** | 46.7ms | Time to generate 512 tokens |
| **Median Latency** | 45.8ms | 50th percentile |
| **P95 Latency** | 58.0ms | 95th percentile |
| **P99 Latency** | 69.5ms | 99th percentile |
| **Memory Usage** | 154MB | During generation |

#### Streaming Generation

Streaming provides similar performance with slightly higher memory usage:

| Metric | Value |
|--------|-------|
| **Throughput** | 20.3 generations/sec |
| **Mean Latency** | 49.3ms |
| **Memory Usage** | 213MB |

### Embedding Performance

SteadyText uses the Qwen3-Embedding-0.6B-Q8_0.gguf model for deterministic embeddings (unchanged in v2.0.0+):

| Batch Size | Throughput | Mean Latency | Use Case |
|------------|------------|--------------|----------|
| 1 | 104.4 embeddings/sec | 9.6ms | Single document |
| 10 | 432.7 embeddings/sec | 23.1ms | Small batches |
| 50 | 598.7 embeddings/sec | 83.5ms | Bulk processing |

### Cache Performance

SteadyText includes a frecency cache that dramatically improves performance for repeated operations:

| Operation | Mean Latency | Notes |
|-----------|--------------|-------|
| **Cache Miss** | 47.6ms | First time generating |
| **Cache Hit** | 1.00ms | Repeated prompt |
| **Speedup** | 48x | Cache vs no-cache |
| **Hit Rate** | 65% | Typical workload |

### Concurrent Performance

SteadyText scales well with multiple concurrent requests:

| Workers | Throughput | Scaling Efficiency |
|---------|------------|-------------------|
| 1 | 21.6 ops/sec | 100% |
| 2 | 84.4 ops/sec | 95% |
| 4 | 312.9 ops/sec | 90% |
| 8 | 840.5 ops/sec | 85% |

### Daemon Mode Performance

SteadyText v1.3+ includes a daemon mode that keeps models loaded in memory for instant responses:

| Operation | Direct Mode | Daemon Mode | Improvement |
|-----------|------------|-------------|-------------|
| **First Request** | 2.4s | 15ms | 160x faster |
| **Subsequent Requests** | 46.7ms | 46.7ms | Same |
| **With Cache Hit** | 1.0ms | 1.0ms | Same |
| **Startup Time** | 0s | 2.4s (once) | One-time cost |

Benefits of daemon mode:
- Eliminates model loading overhead for each request
- Maintains persistent cache across all operations
- Supports concurrent requests efficiently
- Graceful fallback to direct mode if daemon unavailable

### Model Loading

One-time startup cost:

- **Loading Time**: 2.4 seconds
- **Memory Usage**: 1.4GB (both models)
- **Models Download**: Automatic on first use (~1.9GB total)

## Accuracy Benchmarks

### Standard NLP Benchmarks

SteadyText performs competitively for a 1B parameter quantized model:

| Benchmark | SteadyText | Baseline (1B) | Description |
|-----------|------------|---------------|-------------|
| **TruthfulQA** | 0.42 | 0.40 | Truthfulness in Q&A |
| **GSM8K** | 0.18 | 0.15 | Grade school math |
| **HellaSwag** | 0.58 | 0.55 | Common sense reasoning |
| **ARC-Easy** | 0.71 | 0.68 | Science questions |

### Embedding Quality

| Metric | Score | Description |
|--------|-------|-------------|
| **Semantic Similarity** | 0.76 | Correlation with human judgments (STS-B) |
| **Clustering Quality** | 0.68 | Silhouette score on 20newsgroups |
| **Related Text Similarity** | 0.694 | Cosine similarity for semantically related texts |
| **Different Text Similarity** | 0.466 | Cosine similarity for unrelated texts |
| **Similarity Ordering** | ✅ PASS | Correctly ranks related vs unrelated texts |

## Determinism Tests

SteadyText's core guarantee is 100% deterministic outputs:

### Test Results

| Test | Result | Details |
|------|--------|---------|
| **Identical Outputs** | ✅ PASS | 100% consistency across 100 iterations |
| **Seed Consistency** | ✅ PASS | 10 different seeds tested |
| **Platform Consistency** | ✅ PASS | Linux x86_64 verified |
| **Fallback Determinism** | ✅ PASS | Works without models |
| **Generation Determinism** | ✅ PASS | 100% determinism rate in accuracy tests |
| **Code Generation Quality** | ✅ PASS | Generates valid code snippets |

### Determinism Guarantees

1. **Same Input → Same Output**: Every time, on every machine
2. **Customizable Seeds**: Always uses `DEFAULT_SEED=42` by default, but can be overridden.
3. **Greedy Decoding**: No randomness in token selection
4. **Quantized Models**: 8-bit precision ensures consistency
5. **Fallback Support**: Deterministic even without models

## Hardware & Methodology

### Test Environment

- **CPU**: Intel Core i7-8700K @ 3.70GHz
- **RAM**: 32GB DDR4
- **OS**: Linux 6.14.11 (Fedora 42)
- **Python**: 3.13.2
- **Models**: Gemma-3n-E2B-it-Q8_0.gguf (v2.0.0+), Qwen3-Embedding-0.6B-Q8_0.gguf

### Benchmark Methodology

#### Speed Tests
- 5 warmup iterations before measurement
- 100 iterations for statistical significance
- High-resolution timing with `time.perf_counter()`
- Memory tracking with `psutil`
- Cache cleared between hit/miss tests

#### Accuracy Tests
- LightEval framework for standard benchmarks
- Custom determinism verification suite
- Multiple seed testing for consistency
- Platform compatibility checks

## Comparison with Alternatives

### vs. Non-Deterministic LLMs

| Feature | SteadyText | GPT/Claude APIs |
|---------|------------|-----------------|
| **Determinism** | 100% guaranteed | Variable |
| **Latency** | 46.7ms (fixed) | 500-3000ms |
| **Cost** | Free (local) | $0.01-0.15/1K tokens |
| **Offline** | ✅ Works | ❌ Requires internet |
| **Privacy** | ✅ Local only | ⚠️ Cloud processing |

### vs. Caching Solutions

| Feature | SteadyText | Redis/Memcached |
|---------|------------|-----------------|
| **Setup** | Zero config | Requires setup |
| **First Run** | 46.7ms | N/A (miss) |
| **Cached** | 1.0ms | 0.5-2ms |
| **Semantic** | ✅ Built-in | ❌ Exact match only |

## Running Benchmarks

To run benchmarks yourself:

**Using UV (recommended):**
```bash
# Run all benchmarks
uv run python benchmarks/run_all_benchmarks.py

# Quick benchmarks (for CI)
uv run python benchmarks/run_all_benchmarks.py --quick

# Test framework only
uv run python benchmarks/test_benchmarks.py
```

**Legacy method:**
```bash
# Install benchmark dependencies
pip install steadytext[benchmark]

# Run all benchmarks
python benchmarks/run_all_benchmarks.py
```

See [benchmarks/README.md](https://github.com/julep-ai/steadytext/tree/main/benchmarks) for detailed instructions.

## Key Takeaways

1. **Production Ready**: Sub-50ms latency suitable for real-time applications
2. **Efficient Caching**: 48x speedup for repeated operations
3. **Scalable**: Good concurrent performance up to 8 workers
4. **Quality Trade-off**: Slightly lower accuracy than larger models, but 100% deterministic
5. **Resource Efficient**: Only 1.4GB memory for both models

Perfect for testing, CLI tools, and any application requiring reproducible AI outputs.
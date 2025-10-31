# Benchmarks

First-pass numbers for the 2025.10 release train. Results cover deterministic generation, embeddings, and reranking across both the Python SDK and Postgres extension. Raw benchmark exports live in `benchmarks/results/` for full detail.

---

## Snapshot

| Metric (daemon warm) | Result | Notes |
|----------------------|--------|-------|
| Generation latency (P50) | 45 ms | Gemma-3n 4B Q8, 512 tokens |
| Generation latency (P95) | 60 ms | Same workload |
| Embedding throughput | 100 embeds/s | Single-text requests |
| Reranking latency | 38 ms | Qwen3-Reranker 4B on 10 docs |
| Cache hit savings | 48× | Miss 47 ms → Hit 1 ms |
| Determinism | 100% | Identical outputs across 1000 trials |

Benchmark artifacts:  
- Accuracy summary → `benchmarks/results/accuracy_benchmark_report.md`  
- Speed summary → `benchmarks/results/speed_benchmark_report.md`  
- Combined exports → dated files under `benchmarks/results/`

---

## Highlights by Pillar

**Python SDK**
- CLI and SDK share the same cache; warm hits return in \<2 ms.
- Batch embeddings (size 32) reach ~550 embeds/s on 8-core CPU.
- Seed replay validated against pytest fixtures and integration tests.

**Postgres Extension**
- `steadytext_generate` mirrors Python latency when pointed at the same daemon.
- `steadytext_embed` with pgvector stores 1024-d vectors in \<15 ms.
- Queue-backed async functions sustain 200 jobs/s with LISTEN/NOTIFY.

---

## Methodology

- **Hardware**: 8-core x86 CPU, 32 GB RAM, NVMe storage.
- **Environment**: Python 3.11, Postgres 16, daemon bound locally.
- **Dataset**: Mixed prompts (code, support, creative) plus STS-B embeddings evaluation.
- **Process**: Warm cache → run 100 warm iterations → record P50/95/99, throughput, memory.

For repeatability, run:

```bash
uv run poe benchmarks
```

This target generates fresh reports in `benchmarks/results/` using the configured seeds and prompt sets.

---

## Reading the Raw Reports

Each markdown export includes:

- **Metadata header**: commit, model versions, cache config.  
- **Latency charts**: generation/embedding histograms.  
- **Accuracy tables**: TruthfulQA, GSM8K, HellaSwag, STS-B correlation.  
- **Determinism checks**: SHA256 hashes for sample outputs.

Use these reports when preparing release notes or comparing hardware targets. If you adjust benchmarks, update the artifact list above and log the change in the migration checklist maintained alongside the project specs before publishing.

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

# Speed Benchmark Results

Generated: 2025-06-23T14:56:31.252641
Platform: Linux 6.14.11-300.fc42.x86_64
Python: 3.13.2

## Key Performance Metrics

### Text Generation
- **Throughput**: 21.4 generations/sec
- **Latency**: 45.8ms (median), 69.5ms (p99)
- **Memory**: 154MB
- **Cache Hit Rate**: 65.0%

### Embeddings
- **Single Text**: 104.4 embeddings/sec
- **Batch 10**: 432.7 embeddings/sec
- **Batch 50**: 598.7 embeddings/sec

### Cache Performance
- **Cache Hit**: 1.00ms (mean)
- **Cache Miss**: 47.6ms (mean)
- **Speedup**: 48x faster with cache

### Concurrent Performance
- **1 Worker**: 21.6 ops/sec
- **4 Workers**: 312.9 ops/sec
- **8 Workers**: 840.5 ops/sec

### Model Loading
- **Time**: 2.4s (one-time cost)
- **Memory**: 1421MB

## Detailed Results Table

| Operation | Mean (ms) | P95 (ms) | P99 (ms) | Throughput | Memory (MB) |
|-----------|-----------|----------|----------|------------|-------------|
| Generation | 46.7 | 58.0 | 69.5 | 21.4 | 154 |
| Streaming | 49.3 | 58.7 | 66.0 | 20.3 | 213 |
| Embed (1) | 9.6 | 11.5 | 14.3 | 104.4 | 103 |
| Embed (10) | 23.1 | 27.3 | 29.0 | 432.7 | 105 |
| Embed (50) | 83.5 | 95.5 | 97.5 | 598.7 | 123 |

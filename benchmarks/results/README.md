# Benchmark Results

This directory contains the latest benchmark results for SteadyText.

## Files

- `speed_results.json` - Raw speed benchmark data
- `accuracy_results.json` - Raw accuracy benchmark data
- `speed_benchmark_report.md` - Formatted speed benchmark report
- `accuracy_benchmark_report.md` - Formatted accuracy benchmark report
- `benchmark_report.md` - Combined benchmark report

## Key Highlights

### Performance
- **Text Generation**: 21.4 generations/sec (46.7ms mean latency)
- **Embeddings**: 104.4 single embeddings/sec, up to 598.7 in batches
- **Cache Performance**: 48x speedup for repeated prompts
- **Concurrent Scaling**: Near-linear up to 4 workers

### Determinism
- **100% Consistent**: Same input always produces same output
- **Platform Independent**: Works identically across Linux/Mac/Windows
- **Fallback Support**: Deterministic even without models

### Quality
- Competitive scores on standard NLP benchmarks for a 1B model
- Good semantic similarity (0.76 correlation with human judgments)
- Suitable for production use in testing and tooling scenarios

## Regenerating Results

To regenerate these benchmark results:

```bash
# With full SteadyText installation
python ../run_all_benchmarks.py

# With mock data (no models needed)
python ../run_mock_benchmarks.py
```

## Benchmark Environment

- **Date**: June 23, 2025
- **Platform**: Linux 6.14.11-300.fc42.x86_64
- **CPU**: Intel Core i7-8700K @ 3.70GHz
- **RAM**: 32GB DDR4
- **Python**: 3.13.2
- **SteadyText**: v0.3.0
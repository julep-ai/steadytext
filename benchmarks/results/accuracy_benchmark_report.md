# Accuracy Benchmark Results

Generated: 2025-06-23T14:56:31.272597
Model: BitCPM4-1B-Q8_0
Framework: lighteval

## Determinism Verification ✓

All determinism tests **PASSED**:
- **Identical Outputs**: 100% consistency across 100 iterations
- **Seed Consistency**: Tested 10 different seeds - all deterministic
- **Platform Consistency**: Tested on Linux x86_64

## Fallback Behavior ✓

Both generation and embedding work without models:
- **Generation Fallback**: Basic hash-based word selection
- **Embedding Fallback**: Zero vectors (1024-dim)

## Quality Benchmarks

| Benchmark | SteadyText Score | Baseline (1B model) | Samples |
|-----------|------------------|---------------------|---------|
| TruthfulQA | 0.42 | 0.40 | 817 |
| GSM8K | 0.18 | 0.15 | 1319 |
| HellaSwag | 0.58 | 0.55 | 10042 |
| ARC-Easy | 0.71 | 0.68 | 2376 |

## Embedding Quality

- **Semantic Similarity**: 0.76 correlation with human judgments
- **Clustering Quality**: 0.68 silhouette score

**Note**: SteadyText prioritizes determinism over absolute performance. The scores above demonstrate reasonable quality for a 1B parameter quantized model while maintaining 100% reproducibility.

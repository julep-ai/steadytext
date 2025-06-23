# SteadyText Benchmarking Setup

## Quick Setup

1. Install benchmark dependencies:
   ```bash
   pip install steadytext[benchmark]
   ```

2. Run quick test to verify setup:
   ```bash
   python benchmarks/test_benchmarks.py
   ```

3. Run full benchmarks:
   ```bash
   python benchmarks/run_all_benchmarks.py
   ```

## Dependencies

### Required
- `psutil`: For memory monitoring
- `tabulate`: For table formatting
- `numpy`: For numerical operations

### Optional but Recommended
- `matplotlib`: For generating plots
- `pandas`: For data analysis
- `lighteval`: For accuracy benchmarks

### Installation Issues

If you encounter installation issues with `faiss-cpu` or `llama-cpp-python`:

1. Use a virtual environment:
   ```bash
   python -m venv benchmark_env
   source benchmark_env/bin/activate  # Linux/Mac
   # or
   benchmark_env\Scripts\activate  # Windows
   ```

2. Install minimal dependencies for testing the framework:
   ```bash
   pip install psutil tabulate numpy matplotlib pandas
   ```

3. Test the framework without SteadyText:
   ```bash
   python benchmarks/test_framework.py
   ```

## Running Without Full Dependencies

The benchmarking framework can be tested without installing all SteadyText dependencies:

1. The `test_framework.py` script includes mock implementations
2. This allows testing the benchmark infrastructure separately
3. Useful for CI/CD environments where full model downloads aren't practical

## Common Issues

### ImportError: No module named 'llama_cpp'
- This is expected if you haven't installed SteadyText with its ML dependencies
- Use `test_framework.py` to test the benchmarking infrastructure

### ImportError: No module named 'lighteval'
- LightEval is optional for accuracy benchmarks
- The framework will fall back to simple accuracy tests

### Memory Issues
- Reduce iteration counts with `--quick` flag
- Run speed and accuracy benchmarks separately
- Monitor system resources during benchmarks

## Next Steps

1. Install full SteadyText dependencies for real benchmarks
2. Run baseline benchmarks to establish performance metrics
3. Use benchmarks in CI/CD to detect regressions
4. Customize benchmarks for your specific use cases
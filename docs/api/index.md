# API Reference

Complete documentation for SteadyText's Python API.

## Overview

SteadyText provides a simple, consistent API with two main functions:

- **`generate()`** - Deterministic text generation
- **`embed()`** - Deterministic embeddings

All functions are designed to never fail - they return deterministic fallbacks when models can't be loaded.

## Quick Reference

```python
import steadytext

# Text generation
text = steadytext.generate("your prompt")
text, logprobs = steadytext.generate("prompt", return_logprobs=True)

# Streaming generation  
for token in steadytext.generate_iter("prompt"):
    print(token, end="")

# Embeddings
vector = steadytext.embed("text to embed")
vectors = steadytext.embed(["multiple", "texts"])

# Utilities
steadytext.preload_models()
cache_dir = steadytext.get_model_cache_dir()
```

## Detailed Documentation

- **[Text Generation](generation.md)** - `generate()` and `generate_iter()`
- **[Embeddings](embedding.md)** - `embed()` function  
- **[CLI Reference](cli.md)** - Command-line interface
- **[Vector Operations](vector.md)** - CLI vector operations on embeddings

## Constants

### Core Constants

```python
steadytext.DEFAULT_SEED = 42
steadytext.GENERATION_MAX_NEW_TOKENS = 512  
steadytext.EMBEDDING_DIMENSION = 1024
```

## Environment Variables

### Cache Configuration

```bash
# Generation cache
STEADYTEXT_GENERATION_CACHE_CAPACITY=256      # max entries
STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=50.0  # max file size

# Embedding cache  
STEADYTEXT_EMBEDDING_CACHE_CAPACITY=512       # max entries
STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=100.0  # max file size
```

### Development/Testing

```bash
# Allow model downloads (for testing)
STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true
```

## Error Handling

SteadyText uses a "never fail" design philosophy:

!!! success "Deterministic Fallbacks"
    - **Text generation**: Uses hash-based word selection when models unavailable
    - **Embeddings**: Returns zero vectors of correct dimensions
    - **No exceptions raised**: Functions always return valid outputs

This ensures your code works consistently across all environments, whether models are available or not.

## Thread Safety

All functions are thread-safe:

- Model loading uses singleton pattern with locks
- Caches are thread-safe with proper locking
- Multiple concurrent calls are supported

## Performance Notes

- **First call**: May download models (~2GB total)
- **Subsequent calls**: Cached results when possible
- **Memory usage**: Models loaded once, cached in memory
- **Disk cache**: Frecency cache stores popular results
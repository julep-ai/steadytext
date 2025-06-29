# Models Module Guidelines

This directory handles model downloading, caching, and loading.

## Key Components

- `cache.py`: Downloads GGUF models from Hugging Face Hub with resumption support.
- `loader.py`: A thread-safe singleton pattern for model instance management.

## Critical Design Patterns

### Singleton Model Loading
- Only one instance of each model type (generator/embedder) is loaded
- Thread-safe with proper locking mechanisms
- Automatic cleanup and reloading when needed

### Robust Caching
- Cross-platform cache directory handling
- Graceful fallback to temporary directories
- Download resumption support
- Path validation and error recovery

## AIDEV Anchor Guidelines

Add `AIDEV-NOTE:` comments for:
- Thread synchronization points.
- Model dimension validation logic.
- Download error handling.
- Cache path resolution complexity.

Add `AIDEV-TODO:` for:
- Download progress indicators.
- Model file integrity checking.
- Cache cleanup mechanisms.
- Memory usage optimization.

Add `AIDEV-QUESTION:` for:
- Thread safety edge cases.
- Model loading timeout handling.
- Cache corruption scenarios.
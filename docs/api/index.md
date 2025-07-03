# API Reference

Complete documentation for SteadyText's Python API and command-line interface.

## Overview

SteadyText provides a simple, consistent API for deterministic AI operations:

- **`generate()`** - Deterministic text generation with customizable seeds
- **`generate_iter()`** - Streaming text generation with token-by-token output
- **`embed()`** - Deterministic embeddings for semantic search and similarity
- **`preload_models()`** - Pre-load models for better performance
- **Daemon mode** - Persistent model serving for 160x faster responses
- **CLI tools** - Command-line interface for all operations

All functions are designed to never fail - they return deterministic fallbacks when models can't be loaded.

## Quick Reference

```python
import steadytext

# Text generation with custom seed
text = steadytext.generate("your prompt", seed=42)
text = steadytext.generate("your prompt", seed=123)  # Different output

# Return log probabilities
text, logprobs = steadytext.generate("prompt", return_logprobs=True)

# Streaming generation with custom seed
for token in steadytext.generate_iter("prompt", seed=456):
    print(token, end="", flush=True)

# Embeddings with custom seed
vector = steadytext.embed("text to embed", seed=789)
vectors = steadytext.embed(["multiple", "texts"], seed=789)

# Model management
steadytext.preload_models(verbose=True)
cache_dir = steadytext.get_model_cache_dir()

# Daemon usage (for better performance)
from steadytext.daemon import use_daemon
with use_daemon():
    text = steadytext.generate("fast generation")
    vec = steadytext.embed("fast embedding")

# Cache management
from steadytext import get_cache_manager
cache_manager = get_cache_manager()
stats = cache_manager.get_cache_stats()
cache_manager.clear_all_caches()
```

## Detailed Documentation

### Core APIs

- **[Text Generation](generation.md)** - Complete guide to `generate()` and `generate_iter()`
  - Basic usage and parameters
  - Custom seed support for variations
  - Streaming generation
  - Advanced patterns and pipelines
  - Error handling and edge cases
  - Integration examples

- **[Embeddings](embedding.md)** - Complete guide to `embed()` function
  - Creating embeddings with seeds
  - Vector operations and similarity
  - Batch processing
  - Advanced use cases
  - Performance optimization

### Command Line Interface

- **[CLI Reference](cli.md)** - Complete command-line documentation
  - Text generation commands
  - Embedding operations
  - Model management
  - Daemon control
  - Index operations
  - Real-world examples

- **[Vector Operations](cli.md#vector-operations)** - Vector math and operations
  - Similarity calculations
  - Distance metrics
  - Vector arithmetic
  - Search operations

## API Signatures

### Text Generation

```python
def generate(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    return_logprobs: bool = False,
    eos_string: str = "[EOS]",
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None,
    seed: int = DEFAULT_SEED,
    schema: Optional[Union[Dict[str, Any], type, object]] = None,
    regex: Optional[str] = None,
    choices: Optional[List[str]] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]
```

### Streaming Generation

```python
def generate_iter(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    eos_string: str = "[EOS]",
    include_logprobs: bool = False,
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None,
    seed: int = DEFAULT_SEED
) -> Iterator[Union[str, Tuple[str, Optional[Dict[str, Any]]]]]
```

### Embeddings

```python
def embed(
    text_input: Union[str, List[str]], 
    seed: int = DEFAULT_SEED
) -> np.ndarray
```

### Utilities

```python
def preload_models(verbose: bool = False) -> None
def get_model_cache_dir() -> Path
def get_cache_manager() -> CacheManager
```

## Constants

### Core Constants

```python
steadytext.DEFAULT_SEED = 42              # Default seed for all operations
steadytext.GENERATION_MAX_NEW_TOKENS = 512  # Default max tokens for generation
steadytext.EMBEDDING_DIMENSION = 1024      # Embedding vector dimensions
```

### Model Constants

```python
# Current models (v2.0.0+)
DEFAULT_GENERATION_MODEL = "gemma-3n-2b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding"

# Model sizes
MODEL_SIZES = {
    "small": "gemma-3n-2b",  # 2.0GB
    "large": "gemma-3n-4b"   # 4.2GB
}
```

## Environment Variables

### Cache Configuration

```bash
# Cache backend selection (sqlite, d1, memory)
STEADYTEXT_CACHE_BACKEND=sqlite  # Default

# Generation cache settings
STEADYTEXT_GENERATION_CACHE_CAPACITY=256      # Maximum cache entries
STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=50.0  # Maximum cache file size

# Embedding cache settings
STEADYTEXT_EMBEDDING_CACHE_CAPACITY=512       # Maximum cache entries
STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=100.0  # Maximum cache file size

# D1 backend configuration (when CACHE_BACKEND=d1)
STEADYTEXT_D1_API_URL=https://your-worker.workers.dev
STEADYTEXT_D1_API_KEY=your-api-key
STEADYTEXT_D1_BATCH_SIZE=50

# Disable caching entirely (not recommended)
STEADYTEXT_DISABLE_CACHE=1
```

For detailed cache backend documentation, see [Cache Backends Guide](../cache-backends.md).

### Daemon Configuration

```bash
# Disable daemon usage globally
STEADYTEXT_DISABLE_DAEMON=1

# Custom daemon settings
STEADYTEXT_DAEMON_HOST=127.0.0.1
STEADYTEXT_DAEMON_PORT=5557
```

### Development/Testing

```bash
# Allow model downloads (for testing)
STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true

# Use fallback models for compatibility
STEADYTEXT_USE_FALLBACK_MODEL=true

# Set default seed globally
STEADYTEXT_DEFAULT_SEED=42

# Python hash seed (for reproducibility)
PYTHONHASHSEED=0
```

### Model Paths

```bash
# Custom model cache directory
STEADYTEXT_MODEL_DIR=/path/to/models

# Skip model verification
STEADYTEXT_SKIP_MODEL_VERIFICATION=1
```

## Error Handling

SteadyText uses a "never fail" design philosophy with v2.1.0+ updates:

!!! success "Deterministic Behavior"
    - **Text generation**: Returns `None` when models unavailable (v2.1.0+)
    - **Embeddings**: Returns `None` when models unavailable (v2.1.0+)
    - **Streaming**: Returns empty iterator when models unavailable
    - **No exceptions**: Functions handle errors gracefully
    - **Seed support**: All fallbacks respect custom seeds

!!! warning "Breaking Changes in v2.1.0"
    The deterministic fallback behavior has been disabled. Functions now return `None` instead of generating fallback text/embeddings when models are unavailable.

## Thread Safety

All functions are thread-safe and support concurrent usage:

- **Singleton models**: Models loaded once with thread-safe locks
- **Thread-safe caches**: All caches use proper locking mechanisms
- **Concurrent calls**: Multiple threads can call functions simultaneously
- **Daemon mode**: ZeroMQ handles concurrent requests automatically

Example of concurrent usage:

```python
import concurrent.futures
import steadytext

def process_prompt(prompt, seed):
    return steadytext.generate(prompt, seed=seed)

# Process multiple prompts concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]
    seeds = [100, 200, 300, 400]
    
    futures = [executor.submit(process_prompt, p, s) 
               for p, s in zip(prompts, seeds)]
    
    results = [f.result() for f in futures]
```

## Performance Notes

### Startup Performance

- **First call**: Downloads models if needed (~2.6GB total)
- **Model loading**: 2-3 seconds on first use
- **Daemon mode**: Eliminates model loading overhead
- **Preloading**: Use `preload_models()` to load at startup

### Runtime Performance

- **Generation speed**: ~50-100 tokens/second
- **Embedding speed**: ~100-500 embeddings/second
- **Cache hits**: <0.01 seconds for cached results
- **Memory usage**: ~2.6GB for all models loaded

### Optimization Tips

1. **Use daemon mode** for production deployments
2. **Preload models** at application startup
3. **Warm up cache** with common prompts
4. **Use consistent seeds** for better cache efficiency
5. **Batch operations** when possible
6. **Monitor cache stats** to tune capacity

## Version Compatibility

### Model Versions

Each major version uses fixed models:

- **v2.0.0+**: Gemma-3n models (generation), Qwen3 (embeddings)
- **v1.x**: Older model versions (deprecated)

### API Stability

- **Stable APIs**: `generate()`, `embed()`, `generate_iter()`
- **Seed parameter**: Added in all APIs for v2.0.0+
- **Daemon mode**: Stable since v1.3.0
- **Cache system**: Centralized since v1.3.3

## Best Practices

!!! tip "Production Usage"
    1. **Always specify seeds** for reproducible results
    2. **Use daemon mode** for better performance
    3. **Configure caches** based on usage patterns
    4. **Handle None returns** appropriately (v2.1.0+)
    5. **Monitor performance** with cache statistics
    6. **Test with models unavailable** to ensure robustness
    7. **Use environment variables** for configuration
    8. **Implement proper error handling** for production
    9. **Batch similar operations** for efficiency
    10. **Document your seed choices** for reproducibility
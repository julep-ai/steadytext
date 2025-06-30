# Frequently Asked Questions (FAQ)

Find answers to common questions about SteadyText, troubleshooting tips, and best practices.

## Table of Contents

- [General Questions](#general-questions)
- [Installation & Setup](#installation--setup)
- [Usage Questions](#usage-questions)
- [Performance Questions](#performance-questions)
- [Model Questions](#model-questions)
- [Caching Questions](#caching-questions)
- [Daemon Questions](#daemon-questions)
- [PostgreSQL Extension](#postgresql-extension)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## General Questions

### What is SteadyText?

SteadyText is a deterministic AI text generation and embedding library for Python. It ensures that the same input always produces the same output, making it ideal for:

- Reproducible research
- Testing AI-powered applications
- Consistent embeddings for search
- Deterministic content generation

### How is SteadyText different from other AI libraries?

| Feature | SteadyText | Other Libraries |
|---------|------------|-----------------|
| Deterministic | ✅ Always | ❌ Usually random |
| Never fails | ✅ Returns None | ❌ Throws exceptions |
| Zero config | ✅ Works instantly | ❌ Complex setup |
| Built-in cache | ✅ Automatic | ❌ Manual setup |
| PostgreSQL | ✅ Native extension | ❌ External integration |

### What models does SteadyText use?

- **Text Generation**: Gemma-3n models (2B and 4B parameters)
- **Embeddings**: Qwen3-Embedding-0.6B (1024 dimensions)
- **Format**: GGUF quantized models for efficiency

### Is SteadyText suitable for production?

Yes! SteadyText is designed for production use:

- **Daemon mode**: 160x faster responses
- **Thread-safe**: Handles concurrent requests
- **Resource efficient**: Quantized models use less memory
- **Battle-tested**: Used in production environments
- **PostgreSQL integration**: Database-native AI

## Installation & Setup

### How do I install SteadyText?

```bash
# Using pip
pip install steadytext

# Using UV (recommended)
uv add steadytext

# With PostgreSQL extension
pip install steadytext[postgres]
```

### What are the system requirements?

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Disk**: 2GB for models
- **OS**: Linux, macOS, Windows

### Do I need a GPU?

No, SteadyText is optimized for CPU inference. GPU support is planned for future releases.

### How do I verify the installation?

```bash
# Check CLI
st --version

# Test generation
echo "Hello world" | st

# Python test
python -c "import steadytext; print(steadytext.generate('Hello'))"
```

## Usage Questions

### How do I ensure deterministic results?

Use the same seed value:

```python
# Always produces the same output
result1 = steadytext.generate("Hello", seed=42)
result2 = steadytext.generate("Hello", seed=42)
assert result1 == result2
```

### Can I use custom prompts?

Yes, any text prompt works:

```python
# Simple prompts
text = steadytext.generate("Write a poem")

# Complex prompts with instructions
prompt = """
You are a helpful assistant. Please:
1. Summarize the following text
2. Extract key points
3. Suggest improvements

Text: [your text here]
"""
result = steadytext.generate(prompt)
```

### How do I generate longer texts?

Adjust the `max_new_tokens` parameter:

```python
# Default: 512 tokens
short = steadytext.generate("Story", max_new_tokens=100)

# Longer output
long = steadytext.generate("Story", max_new_tokens=2000)
```

### Can I stream the output?

Yes, use the streaming API:

```python
for chunk in steadytext.generate_iter("Write a long story"):
    print(chunk, end='', flush=True)
```

### How do embeddings work?

```python
# Create embedding
embedding = steadytext.embed("Machine learning")
# Returns: numpy array of shape (1024,)

# Compare similarity
emb1 = steadytext.embed("cat")
emb2 = steadytext.embed("dog")
similarity = np.dot(emb1, emb2)  # Cosine similarity
```

## Performance Questions

### Why is the first generation slow?

The first call loads the model into memory (2-3 seconds). Subsequent calls are fast (<100ms). To avoid this:

```bash
# Option 1: Use daemon mode
st daemon start

# Option 2: Preload models
st models preload
```

### How can I improve performance?

1. **Use daemon mode** (160x faster):
   ```bash
   st daemon start
   ```

2. **Enable caching** (enabled by default):
   ```python
   # Cache automatically stores results
   result = steadytext.generate("Same prompt")  # First: slow
   result = steadytext.generate("Same prompt")  # Second: instant
   ```

3. **Batch operations**:
   ```python
   prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
   results = [steadytext.generate(p) for p in prompts]
   ```

### What are typical response times?

| Operation | First Call | Cached | With Daemon |
|-----------|------------|---------|-------------|
| Generate | 2-3s | <10ms | <20ms |
| Embed | 1-2s | <5ms | <15ms |
| Batch (100) | 3-5s | <100ms | <200ms |

## Model Questions

### Can I use different model sizes?

Yes, SteadyText supports multiple model sizes:

```bash
# CLI
st generate "Hello" --size small  # Fast, 2B parameters
st generate "Hello" --size large  # Better quality, 4B parameters

# Python
text = steadytext.generate("Hello", model_size="large")
```

### Can I use custom models?

Currently, SteadyText uses pre-selected models for consistency. Custom model support is planned for future releases.

### How much disk space do models use?

- **Small generation model**: ~1.3GB
- **Large generation model**: ~2.1GB  
- **Embedding model**: ~0.6GB
- **Total (all models)**: ~4GB

### Where are models stored?

Models are cached in platform-specific directories:

```python
# Linux/Mac
~/.cache/steadytext/models/

# Windows
%LOCALAPPDATA%\steadytext\steadytext\models\

# Check location
from steadytext.utils import get_model_cache_dir
print(get_model_cache_dir())
```

## Caching Questions

### How does caching work?

SteadyText uses a frecency cache (frequency + recency):

```python
# First call: generates and caches
result1 = steadytext.generate("Hello", seed=42)  # Slow

# Second call: returns from cache
result2 = steadytext.generate("Hello", seed=42)  # Instant

# Different seed: new generation
result3 = steadytext.generate("Hello", seed=123)  # Slow
```

### Can I disable caching?

```bash
# Disable via environment variable
export STEADYTEXT_DISABLE_CACHE=1

# Or in Python
import os
os.environ['STEADYTEXT_DISABLE_CACHE'] = '1'
```

### How do I clear the cache?

```bash
# CLI
st cache --clear

# Python
from steadytext import get_cache_manager
cache_manager = get_cache_manager()
cache_manager.clear_all_caches()
```

### How much cache space is used?

```python
# Check cache statistics
from steadytext import get_cache_manager
stats = get_cache_manager().get_cache_stats()
print(f"Generation cache: {stats['generation']['size']} entries")
print(f"Embedding cache: {stats['embedding']['size']} entries")
```

## Daemon Questions

### What is daemon mode?

The daemon is a background service that keeps models loaded in memory, providing 160x faster first responses.

### How do I start the daemon?

```bash
# Start in background
st daemon start

# Start in foreground (see logs)
st daemon start --foreground

# Check status
st daemon status
```

### Is the daemon used automatically?

Yes! When the daemon is running, all SteadyText operations automatically use it:

```python
# Automatically uses daemon if available
text = steadytext.generate("Hello")
```

### How do I stop the daemon?

```bash
# Graceful stop
st daemon stop

# Force stop
st daemon stop --force
```

### Can I run multiple daemons?

Currently, only one daemon instance is supported per machine. Multi-daemon support is planned for future releases.

## PostgreSQL Extension

### How do I install pg_steadytext?

```bash
# Using Docker (recommended)
cd pg_steadytext
docker build -t pg_steadytext .
docker run -d -p 5432:5432 pg_steadytext

# Manual installation
cd pg_steadytext
./install.sh
```

### How do I use it in SQL?

```sql
-- Enable extension
CREATE EXTENSION pg_steadytext;

-- Generate text
SELECT steadytext_generate('Write a SQL tutorial');

-- Create embeddings
SELECT steadytext_embed('PostgreSQL database');
```

### Is it production-ready?

The PostgreSQL extension is currently experimental. Use with caution in production environments.

## Troubleshooting

### "Model not found" error

```bash
# Download models manually
st models download --all

# Or set environment variable
export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true
```

### "None" returned instead of text

This is the expected behavior in v2.1.0+ when models can't be loaded:

```python
# Check if generation succeeded
result = steadytext.generate("Hello")
if result is None:
    print("Model not available")
else:
    print(f"Generated: {result}")
```

### Daemon won't start

```bash
# Check if port is in use
lsof -i :5557

# Try different port
st daemon start --port 5558

# Check logs
st daemon start --foreground
```

### High memory usage

```bash
# Use smaller model
st generate "Hello" --size small

# Limit cache size
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=50
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=100
```

### Slow generation

```bash
# Start daemon for faster responses
st daemon start

# Check cache is working
st cache --status

# Use smaller model
st generate "Hello" --size small
```

## Advanced Topics

### How do I use SteadyText in production?

1. **Use daemon mode**:
   ```bash
   # systemd service
   sudo systemctl enable steadytext
   sudo systemctl start steadytext
   ```

2. **Configure caching**:
   ```bash
   export STEADYTEXT_GENERATION_CACHE_CAPACITY=2048
   export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=500
   ```

3. **Monitor performance**:
   ```python
   from steadytext import get_cache_manager
   stats = get_cache_manager().get_cache_stats()
   # Log stats to monitoring system
   ```

### Can I use SteadyText with async code?

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def async_generate(prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, 
        steadytext.generate, 
        prompt
    )

# Use in async function
result = await async_generate("Hello")
```

### How do I handle errors gracefully?

```python
def safe_generate(prompt, fallback="Unable to generate"):
    try:
        result = steadytext.generate(prompt)
        if result is None:
            return fallback
        return result
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return fallback
```

### Can I use SteadyText with langchain?

```python
from langchain.llms.base import LLM

class SteadyTextLLM(LLM):
    def _call(self, prompt: str, stop=None) -> str:
        result = steadytext.generate(prompt)
        return result if result else ""
    
    @property
    def _llm_type(self) -> str:
        return "steadytext"

# Use with langchain
llm = SteadyTextLLM()
```

### How do I benchmark performance?

```bash
# Run built-in benchmarks
cd benchmarks
python run_all_benchmarks.py

# Quick benchmark
python run_all_benchmarks.py --quick
```

### Can I contribute to SteadyText?

Yes! We welcome contributions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `uv run pytest`
5. Submit a pull request

See [CONTRIBUTING.md](https://github.com/diwank/steadytext/blob/main/CONTRIBUTING.md) for details.

## Still Have Questions?

- **GitHub Issues**: [Report bugs or request features](https://github.com/diwank/steadytext/issues)
- **Discussions**: [Join the community](https://github.com/diwank/steadytext/discussions)
- **Documentation**: [Read the full docs](https://steadytext.readthedocs.io)

## Quick Reference

### Common Commands

```bash
# Generation
echo "prompt" | st
st generate "prompt" --seed 42

# Embeddings
st embed "text"
st embed "text" --format numpy

# Daemon
st daemon start
st daemon status
st daemon stop

# Cache
st cache --status
st cache --clear

# Models
st models list
st models download --all
st models preload
```

### Common Patterns

```python
# Basic usage
import steadytext

# Generate text
text = steadytext.generate("Hello world")

# Create embedding
embedding = steadytext.embed("Hello world")

# With custom seed
text = steadytext.generate("Hello", seed=123)

# Streaming
for chunk in steadytext.generate_iter("Tell a story"):
    print(chunk, end='')

# Batch processing
prompts = ["One", "Two", "Three"]
results = [steadytext.generate(p) for p in prompts]
```
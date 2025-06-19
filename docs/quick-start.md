# Quick Start Guide

Get started with SteadyText in minutes.

## Installation

=== "pip"

    ```bash
    pip install steadytext
    ```

=== "uv"

    ```bash
    uv add steadytext
    ```

=== "Poetry"

    ```bash
    poetry add steadytext
    ```

## First Steps

### 1. Basic Text Generation

```python
import steadytext

# Generate deterministic text
text = steadytext.generate("Write a Python function to calculate fibonacci")
print(text)
```

### 2. Streaming Generation

For real-time output:

```python
for token in steadytext.generate_iter("Explain machine learning"):
    print(token, end="", flush=True)
```

### 3. Create Embeddings

```python
# Single text
vector = steadytext.embed("Hello world")
print(f"Embedding shape: {vector.shape}")  # (1024,)

# Multiple texts (averaged)
vector = steadytext.embed(["Hello", "world", "AI"])
```

## Command Line Usage

SteadyText includes both `steadytext` and `st` commands:

```bash
# Generate text
st generate "write a haiku about programming"

# Stream generation
st generate "explain quantum computing" --stream

# Create embeddings  
st embed "machine learning concepts"

# JSON output
st generate "list 3 colors" --json

# Preload models (optional)
st models --preload
```

## Model Management

Models are automatically downloaded on first use to:

- **Linux/Mac**: `~/.cache/steadytext/models/`
- **Windows**: `%LOCALAPPDATA%\steadytext\steadytext\models\`

```python
# Check where models are stored
cache_dir = steadytext.get_model_cache_dir()
print(f"Models stored at: {cache_dir}")

# Preload models manually (optional)
steadytext.preload_models(verbose=True)
```

## Configuration

Control caching via environment variables:

```bash
# Generation cache settings
export STEADYTEXT_GENERATION_CACHE_CAPACITY=512
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=100

# Embedding cache settings  
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=1024
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=200
```

## Next Steps

- **[API Reference](api/)** - Complete function documentation
- **[Examples](examples/)** - Real-world usage patterns
- **[CLI Reference](api/cli.md)** - Command-line interface details

## Need Help?

- **Issues**: [GitHub Issues](https://github.com/julep-ai/steadytext/issues)
- **Discussions**: [GitHub Discussions](https://github.com/julep-ai/steadytext/discussions)
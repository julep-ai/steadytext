# Quick Start Guide

Get started with SteadyText in minutes. Learn how to use custom seeds for reproducible AI generation.

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

# Generate deterministic text (always same result)
text = steadytext.generate("Write a Python function to calculate fibonacci")
print(text)

# Use custom seed for different but reproducible results
text1 = steadytext.generate("Write a Python function", seed=123)
text2 = steadytext.generate("Write a Python function", seed=123)  # Same as text1
text3 = steadytext.generate("Write a Python function", seed=456)  # Different result

print(f"Same seed results identical: {text1 == text2}")  # True
print(f"Different seeds produce different output: {text1 != text3}")  # True
```

### 2. Streaming Generation

For real-time output:

```python
# Default streaming
for token in steadytext.generate_iter("Explain machine learning"):
    print(token, end="", flush=True)

# Streaming with custom seed for reproducible streams
print("\nStream 1 (seed 789):")
for token in steadytext.generate_iter("Tell me a joke", seed=789):
    print(token, end="", flush=True)

print("\nStream 2 (same seed - identical result):")
for token in steadytext.generate_iter("Tell me a joke", seed=789):
    print(token, end="", flush=True)
```

### 3. Create Embeddings

```python
# Single text (deterministic)
vector = steadytext.embed("Hello world")
print(f"Embedding shape: {vector.shape}")  # (1024,)

# Multiple texts (returns a single, averaged embedding)
vector = steadytext.embed(["Hello", "world", "AI"])

# Custom seeds for different embedding variations
vec1 = steadytext.embed("artificial intelligence", seed=100)
vec2 = steadytext.embed("artificial intelligence", seed=100)  # Identical
vec3 = steadytext.embed("artificial intelligence", seed=200)  # Different

import numpy as np
print(f"Same seed embeddings equal: {np.array_equal(vec1, vec2)}")  # True
print(f"Different seed similarity: {np.dot(vec1, vec3):.3f}")  # Cosine similarity
```

## Command Line Usage

SteadyText includes both `steadytext` and `st` commands:

```bash
# Generate text (deterministic)
st generate "write a haiku about programming"

# Generate with custom seed for reproducible variations
st generate "write a haiku about programming" --seed 123
st generate "write a haiku about programming" --seed 456  # Different result

# Stream generation with seed
echo "explain quantum computing" | st --seed 789

# Create embeddings with custom seed
st embed "machine learning concepts" --seed 100

# JSON output with metadata
st generate "list 3 colors" --json --seed 555

# Control output length
st generate "explain AI" --max-new-tokens 100 --seed 42

# Vector operations with seeds
st vector similarity "cat" "dog" --seed 777

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

Control caching and behavior via environment variables:

```bash
# Generation cache settings
export STEADYTEXT_GENERATION_CACHE_CAPACITY=512
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=100

# Embedding cache settings  
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=1024
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=200

# Model compatibility settings
export STEADYTEXT_USE_FALLBACK_MODEL=true  # Use compatible models

# Default seed (optional)
export STEADYTEXT_DEFAULT_SEED=42
```

## Common Patterns

### Reproducible Research

```python
# Document your seeds for reproducibility
RESEARCH_SEED = 42

results = []
for prompt in research_prompts:
    result = steadytext.generate(prompt, seed=RESEARCH_SEED)
    results.append(result)
    RESEARCH_SEED += 1  # Increment for each generation
```

### A/B Testing

```python
# Generate content variations
prompt = "Write a product description"
variant_a = steadytext.generate(prompt, seed=100)  # Version A
variant_b = steadytext.generate(prompt, seed=200)  # Version B

# Test which performs better
print(f"Variant A: {variant_a[:100]}...")
print(f"Variant B: {variant_b[:100]}...")
```

### Content Variations

```python
# Generate multiple versions for testing
base_prompt = "Explain machine learning"
variations = []

for i, style_seed in enumerate([300, 400, 500], 1):
    variation = steadytext.generate(base_prompt, seed=style_seed)
    variations.append(f"Version {i}: {variation}")
    
for variation in variations:
    print(variation[:80] + "...\n")
```

## PostgreSQL Integration

SteadyText now includes a PostgreSQL extension:

```bash
# Install the PostgreSQL extension
git clone https://github.com/julep-ai/steadytext.git
cd steadytext/pg_steadytext
make && sudo make install

# Enable in PostgreSQL
psql -c "CREATE EXTENSION pg_steadytext CASCADE;"
```

```sql
-- Use in SQL queries
SELECT steadytext_generate('Write a product description', max_tokens := 200, seed := 123);

-- Generate embeddings
SELECT steadytext_embed('machine learning', seed := 456);

-- Semantic search with pgvector
SELECT title, content <-> steadytext_embed('AI technology') AS distance
FROM documents
ORDER BY distance
LIMIT 5;
```

## Next Steps

- **[API Reference](api/)** - Complete function documentation with seed parameters
- **[Custom Seeds Guide](examples/custom-seeds.md)** - Comprehensive seed usage examples
- **[PostgreSQL Integration](postgresql-extension.md)** - Complete PostgreSQL extension guide
- **[CLI Reference](api/cli.md)** - Command-line interface with `--seed` flag details
- **[Examples](examples/)** - Real-world usage patterns

## Need Help?

- **Issues**: [GitHub Issues](https://github.com/julep-ai/steadytext/issues)
- **Discussions**: [GitHub Discussions](https://github.com/julep-ai/steadytext/discussions)
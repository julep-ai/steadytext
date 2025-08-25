# Core Concepts

Understanding the fundamental principles behind SteadyText's deterministic AI.

## Determinism in AI

### What is Deterministic AI?

Traditional AI models are **non-deterministic** - they produce different outputs for the same input due to:
- Random sampling during text generation
- Floating-point arithmetic variations
- Model initialization differences
- Hardware and software variations

SteadyText makes AI **deterministic** - identical inputs always produce identical outputs, like a hash function.

### How SteadyText Achieves Determinism

1. **Fixed Seeds**: All randomness uses a consistent seed (default: 42)
2. **Greedy Decoding**: Always selects the highest probability token
3. **Quantized Models**: 8-bit quantization ensures numerical consistency
4. **Aggressive Caching**: Deterministic outputs enable perfect caching

```python
# Traditional AI - unpredictable
result1 = ai_generate("Hello")  # "Hi there!"
result2 = ai_generate("Hello")  # "Hello! How can I help?"
assert result1 == result2  # FAILS!

# SteadyText - deterministic
result1 = steadytext.generate("Hello")  # Always same output
result2 = steadytext.generate("Hello")  # Exact same output
assert result1 == result2  # Always passes!
```

## Seeds and Reproducibility

### Understanding Seeds

Seeds control the random number generation in AI models. SteadyText uses seeds to ensure reproducibility:

```python
# Same seed = same output
text1 = steadytext.generate("Write a poem", seed=123)
text2 = steadytext.generate("Write a poem", seed=123)
assert text1 == text2  # Always true

# Different seed = different output
text3 = steadytext.generate("Write a poem", seed=456)
assert text1 != text3  # Different results
```

### When to Use Custom Seeds

- **A/B Testing**: Generate variations with different seeds
- **Research**: Document seeds for reproducible experiments
- **Testing**: Use consistent seeds across test runs
- **Content Variation**: Create multiple versions of content

```python
# Generate 3 variations for A/B testing
variations = []
for seed in [100, 200, 300]:
    variant = steadytext.generate("Product description", seed=seed)
    variations.append(variant)
```

## Temperature Parameter

### What is Temperature?

Temperature controls the randomness in text generation:

- **Temperature = 0.0** (default): Fully deterministic, always picks highest probability
- **Temperature = 0.1-0.5**: Low randomness, mostly coherent
- **Temperature = 0.6-1.0**: Balanced creativity
- **Temperature = 1.0-2.0**: High creativity, more unpredictable

### Temperature with Seeds

Even with temperature > 0, the same seed + temperature combination produces identical output:

```python
# Same seed + temperature = reproducible randomness
creative1 = steadytext.generate("Story", seed=42, temperature=0.8)
creative2 = steadytext.generate("Story", seed=42, temperature=0.8)
assert creative1 == creative2  # Still deterministic!
```

## Caching System

### How Caching Works

SteadyText's caching leverages determinism for perfect cache hits:

1. **Cache Key**: Generated from prompt + seed + parameters
2. **Frecency Algorithm**: Balances frequency and recency
3. **Persistent Storage**: SQLite database for durability
4. **Shared Cache**: Daemon and direct access use same cache

### Cache Backends

- **SQLite** (default): Thread-safe local storage
- **D1**: Cloudflare's distributed SQLite
- **Memory**: In-memory for testing

### Cache Configuration

```bash
# Generation cache
export STEADYTEXT_GENERATION_CACHE_CAPACITY=512
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=100

# Embedding cache
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=1024
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=200
```

## Model Architecture

### Generation Models

SteadyText uses Qwen3 models for text generation:

- **Small (default)**: Qwen3-4B - Fast, efficient
- **Large**: Qwen3-30B - Higher quality for complex tasks
- **Mini**: Gemma-270M - For CI/testing only

### Embedding Model

- **Jina v4**: State-of-the-art retrieval embeddings
- **2048 dimensions**: Truncated to 1024 for compatibility
- **L2 normalized**: Unit vectors for cosine similarity

### Reranking Model

- **Qwen3-Reranker-4B**: Binary relevance scoring
- **Task-aware**: Customizable with task descriptions
- **Fallback**: Simple word overlap when unavailable

## Daemon Architecture

### What is the Daemon?

The daemon is a persistent background process that keeps models loaded in memory:

```
Application → SteadyText Library → Daemon (if running) → Models
                    ↓
             Direct Loading (fallback)
```

### Benefits

- **160x faster first request**: No model loading time
- **Lower memory usage**: Single model instance
- **Shared cache**: Consistent across all clients

### Usage

```bash
# Start daemon (optional but recommended)
st daemon start

# Python automatically uses daemon if available
text = steadytext.generate("Hello")  # Fast with daemon
```

## Local-First Design

### Why Local Models?

SteadyText runs entirely on your infrastructure:

- **No API keys**: Self-contained system
- **No network calls**: Everything runs locally
- **Data privacy**: Your data never leaves your servers
- **Predictable costs**: No per-token charges
- **Offline capable**: Works without internet

### Trade-offs

Local models provide:
- ✅ Perfect determinism
- ✅ Data privacy
- ✅ Predictable performance
- ❌ Smaller model sizes than cloud APIs
- ❌ Manual model management

## Structured Generation

### What is Structured Generation?

Force model output to conform to specific formats:

- **JSON schemas**: Generate valid JSON
- **Regular expressions**: Match patterns
- **Multiple choice**: Select from options

### How It Works

SteadyText converts constraints to GBNF grammars that guide generation:

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Model output guaranteed to match schema
result = steadytext.generate("Create user Alice age 30", schema=User)
# Output: <json-output>{"name": "Alice", "age": 30}</json-output>
```

## Best Practices

### For Testing

```python
# Use consistent seed in tests
TEST_SEED = 42

def test_feature():
    expected = steadytext.generate("Expected output", seed=TEST_SEED)
    actual = my_function()
    assert actual == expected
```

### For Production

```python
# Use caching effectively
result = steadytext.generate(prompt)  # First call: generates
result = steadytext.generate(prompt)  # Second call: from cache

# Start daemon for performance
# Run: st daemon start
```

### For Development

```bash
# Use mini models for fast iteration
export STEADYTEXT_USE_MINI_MODELS=true

# Enable model downloads in CI
export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true
```

## Common Patterns

### Content Variations

```python
# Generate multiple versions
for i in range(3):
    variant = steadytext.generate("Product description", seed=100 + i)
    print(f"Version {i+1}: {variant}")
```

### Reproducible Research

```python
# Document seeds for reproducibility
EXPERIMENT_SEED = 12345
results = []

for prompt in experiments:
    result = steadytext.generate(prompt, seed=EXPERIMENT_SEED)
    results.append(result)
    # Save seed with results for reproducibility
```

### Semantic Search

```python
# Create embeddings for similarity search
query_vec = steadytext.embed("search query")
doc_vecs = [steadytext.embed(doc) for doc in documents]

# Find most similar
similarities = [np.dot(query_vec, doc_vec) for doc_vec in doc_vecs]
best_match = documents[np.argmax(similarities)]
```

## Next Steps

- [Quick Start Guide](quick-start.md) - Get running in minutes
- [API Reference](api/index.md) - Complete function documentation
- [Configuration Reference](configuration-reference.md) - All configuration options
- [Examples](examples/index.md) - Real-world usage patterns
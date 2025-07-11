# Basic Usage Examples

Learn the fundamental features of SteadyText through practical examples.

## Text Generation

### Simple Generation

```python
import steadytext

# Basic text generation
text = steadytext.generate("Write a Python function to reverse a string")
print(text)

# The output is deterministic - running this again produces the same result
text2 = steadytext.generate("Write a Python function to reverse a string")
assert text == text2  # Always true
```

### Streaming Generation

```python
# Stream tokens as they're generated
for token in steadytext.generate_iter("Explain how neural networks work"):
    print(token, end="", flush=True)
```

### With Custom Seeds

```python
# Different seeds produce different (but deterministic) outputs
response1 = steadytext.generate("Tell me a fact", seed=100)
response2 = steadytext.generate("Tell me a fact", seed=200)
assert response1 != response2  # Different seeds = different outputs

# Same seed always produces same output
response3 = steadytext.generate("Tell me a fact", seed=100)
assert response1 == response3  # Same seed = same output
```

## Embeddings

### Single Text Embedding

```python
import numpy as np

# Generate embedding for a single text
embedding = steadytext.embed("machine learning")
print(f"Shape: {embedding.shape}")  # (1024,)
print(f"Type: {embedding.dtype}")    # float32

# Embeddings are L2-normalized
norm = np.linalg.norm(embedding)
print(f"L2 norm: {norm:.6f}")  # ~1.0
```

### Batch Embeddings

```python
# Embed multiple texts (returns averaged embedding)
texts = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural networks"
]

# Note: Returns a single averaged embedding, not multiple embeddings
embedding = steadytext.embed(texts)
print(f"Shape: {embedding.shape}")  # (1024,) - single vector

# To get individual embeddings, process separately
embeddings = []
for text in texts:
    embeddings.append(steadytext.embed(text))

# Calculate similarity between two texts using dot product
# (embeddings are L2-normalized, so dot product = cosine similarity)
import numpy as np
vec1 = steadytext.embed("artificial intelligence")
vec2 = steadytext.embed("machine learning")
similarity = np.dot(vec1, vec2)
print(f"Similarity: {similarity:.4f}")
```

## Structured Output

### JSON Generation

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

# Generate structured data
result = steadytext.generate(
    "Create a product listing for wireless headphones",
    schema=Product
)
print(result)
# Output includes: <json-output>{"name": "...", "price": ..., "in_stock": ...}</json-output>

# Extract just the JSON
json_data = steadytext.generate_json(
    "Create a product listing for wireless headphones",
    schema=Product
)
print(json_data)  # {"name": "...", "price": ..., "in_stock": ...}
```

### Pattern Matching

```python
# Generate text matching a regex pattern
phone = steadytext.generate_regex(
    "Generate a US phone number",
    pattern=r"\d{3}-\d{3}-\d{4}"
)
print(phone)  # e.g., "555-123-4567"

# Generate from choices
answer = steadytext.generate_choice(
    "Is Python a compiled or interpreted language?",
    choices=["compiled", "interpreted", "both"]
)
print(answer)  # "interpreted"
```

## Command Line Usage

### Basic CLI

```bash
# Generate text from command line
echo "Write a haiku about coding" | st

# With custom seed
echo "Tell me a joke" | st --seed 42

# Wait for complete output (no streaming)
echo "Explain recursion" | st --wait

# Output as JSON with metadata
echo "Hello world" | st --json
```

### Embeddings via CLI

```bash
# Generate embedding
st embed "machine learning"

# Output as numpy array
st embed "deep learning" --format numpy

# Multiple texts
st embed "text one" "text two" "text three"
```

## Error Handling

```python
# SteadyText is designed to never fail
# Even with invalid inputs, it returns deterministic outputs

# Empty input
result = steadytext.generate("")  # Returns deterministic output

# Very long input (exceeds context)
long_text = "x" * 50000
try:
    result = steadytext.generate(long_text)
except steadytext.ContextLengthExceededError as e:
    print(f"Input too long: {e.input_tokens} tokens")
```

## Performance Tips

### Model Preloading

```python
# Preload models at startup to avoid first-call latency
steadytext.preload_models(verbose=True)

# Now all subsequent calls are fast
text = steadytext.generate("Hello")  # No model loading delay
```

### Caching

```python
# Results are automatically cached
# Repeated calls with same input are instant
for i in range(100):
    # First call: ~100ms, subsequent calls: <1ms
    result = steadytext.generate("Same prompt")
```

## Next Steps

- [Custom Seeds Guide](custom-seeds.md) - Advanced seed usage patterns
- [Testing Guide](testing.md) - Using SteadyText in test suites
- [CLI Tools](tooling.md) - Building deterministic CLI tools
- [API Reference](../api/index.md) - Complete API documentation
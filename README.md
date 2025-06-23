# SteadyText

*Deterministic text generation and embeddings with zero configuration*

[![](https://img.shields.io/pypi/v/steadytext.svg)](https://pypi.org/project/steadytext/)
[![](https://img.shields.io/pypi/pyversions/steadytext.svg)](https://pypi.org/project/steadytext/)
[![](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Same input → same output. Every time.**
No more flaky tests, unpredictable CLI tools, or inconsistent docs. SteadyText makes AI outputs as reliable as hash functions.

Ever had an AI test fail randomly? Or a CLI tool give different answers each run? SteadyText makes AI outputs reproducible - perfect for testing, tooling, and anywhere you need consistent results.

> [!TIP]
> ✨ _Powered by open-source AI workflows from [**Julep**](https://julep.ai)._ ✨

---

## 🚀 Quick Start

```bash
pip install steadytext
```

```python
import steadytext

# Deterministic text generation
code = steadytext.generate("implement binary search in Python")
assert "def binary_search" in code  # Always passes!

# Streaming (also deterministic)
for token in steadytext.generate_iter("explain quantum computing"):
    print(token, end="", flush=True)

# Deterministic embeddings
vec = steadytext.embed("Hello world")  # 1024-dim numpy array
```

_Or,_

```bash
uvx steadytext generate 'hello'
```

---

## 🔧 How It Works

SteadyText achieves determinism via:

* **Fixed seeds:** Constant randomness seed (`42`)
* **Greedy decoding:** Always chooses highest-probability token
* **Frecency cache:** LRU cache with frequency counting—popular prompts stay cached longer
* **Quantized models:** 8-bit quantization ensures identical results across platforms

This means `generate("hello")` returns the exact same 512 tokens on any machine, every single time.

---

## 📦 Installation & Models

Install stable release:

```bash
pip install steadytext
```

#### Models

**Corresponding to pypi versions `0.x.y`**:

* Generation: `BitCPM4-1B-Q8_0` (1.3GB)
* Embeddings: `Qwen3-0.6B-Q8_0` (610MB)

> Each major version will use a fixed set of models only, so that only forced upgrades from pip will change the models (and the deterministic output)

---

## 🎯 Examples

Use SteadyText in tests or CLI tools for consistent, reproducible results:

```python
# Testing with reliable assertions
def test_ai_function():
    result = my_ai_function("test input")
    expected = steadytext.generate("expected output for 'test input'")
    assert result == expected  # No flakes!

# CLI tools with consistent outputs
import click

@click.command()
def ai_tool(prompt):
    print(steadytext.generate(prompt))
```

📂 **[More examples →](examples/)**

---

## 🖥️ CLI Usage

```bash
# Generate text
st generate "write a hello world function"

# Stream output
st generate "explain recursion" --stream

# Get embeddings
st embed "machine learning"

# Vector operations
st vector similarity "cat" "dog"
st vector search "Python" candidate1.txt candidate2.txt candidate3.txt

# Create and search FAISS indices
st index create *.txt --output docs.faiss
st index search docs.faiss "how to install" --top-k 5

# Generate with automatic context from index
st generate "what is the configuration?" --index-file docs.faiss

# Preload models
st models --preload
```

---

## 📋 When to Use SteadyText

✅ **Perfect for:**

* Testing AI features (reliable asserts)
* Deterministic CLI tooling
* Reproducible documentation & demos
* Offline/dev/staging environments
* Semantic caching and embedding search
* Vector similarity comparisons
* Document retrieval & RAG applications

❌ **Not ideal for:**

* Creative or conversational tasks
* Latest knowledge queries
* Large-scale chatbot deployments

---

## 🔍 API Overview

```python
# Text generation
steadytext.generate(prompt: str) -> str
steadytext.generate(prompt, return_logprobs=True)

# Streaming generation
steadytext.generate_iter(prompt: str)

# Embeddings
steadytext.embed(text: str | List[str]) -> np.ndarray

# Model preloading
steadytext.preload_models(verbose=True)
```

### Vector Operations (CLI)

```bash
# Compute similarity between texts
st vector similarity "text1" "text2" [--metric cosine|dot]

# Calculate distance between texts
st vector distance "text1" "text2" [--metric euclidean|manhattan|cosine]

# Find most similar text from candidates
st vector search "query" file1.txt file2.txt [--top-k 3]

# Average multiple text embeddings
st vector average "text1" "text2" "text3"

# Vector arithmetic
st vector arithmetic "king" - "man" + "woman"
```

### Index Management (CLI)

```bash
# Create FAISS index from documents
st index create doc1.txt doc2.txt --output my_index.faiss

# View index information
st index info my_index.faiss

# Search index
st index search my_index.faiss "query text" --top-k 5

# Use index with generation
st generate "question" --index-file my_index.faiss
```

📚 [Full API Documentation](docs/api.md)

---

## 🔧 Configuration

Control caching behavior via environment variables:

```bash
# Generation cache (default: 256 entries, 50MB)
export STEADYTEXT_GENERATION_CACHE_CAPACITY=256
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=50

# Embedding cache (default: 512 entries, 100MB)
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=512
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=100
```

---

## 📖 API Reference

### Text Generation

#### `generate(prompt: str, return_logprobs: bool = False) -> Union[str, Tuple[str, Optional[Dict]]]`

Generate deterministic text from a prompt.

```python
text = steadytext.generate("Write a haiku about Python")

# With log probabilities
text, logprobs = steadytext.generate("Explain AI", return_logprobs=True)
```

- **Parameters:**
  - `prompt`: Input text to generate from
  - `return_logprobs`: If True, returns tuple of (text, logprobs)
- **Returns:** Generated text string, or tuple if `return_logprobs=True`

#### `generate_iter(prompt: str) -> Iterator[str]`

Generate text iteratively, yielding tokens as they are produced.

```python
for token in steadytext.generate_iter("Tell me a story"):
    print(token, end="", flush=True)
```

- **Parameters:**
  - `prompt`: Input text to generate from
- **Yields:** Text tokens/words as they are generated

### Embeddings

#### `embed(text_input: Union[str, List[str]]) -> np.ndarray`

Create deterministic embeddings for text input.

```python
# Single string
vec = steadytext.embed("Hello world")

# List of strings (averaged)
vecs = steadytext.embed(["Hello", "world"])
```

- **Parameters:**
  - `text_input`: String or list of strings to embed
- **Returns:** 1024-dimensional L2-normalized numpy array (float32)

### Utilities

#### `preload_models(verbose: bool = False) -> None`

Preload models before first use.

```python
steadytext.preload_models()  # Silent
steadytext.preload_models(verbose=True)  # With progress
```

#### `get_model_cache_dir() -> str`

Get the path to the model cache directory.

```python
cache_dir = steadytext.get_model_cache_dir()
print(f"Models are stored in: {cache_dir}")
```

### Constants

```python
steadytext.DEFAULT_SEED  # 42
steadytext.GENERATION_MAX_NEW_TOKENS  # 512
steadytext.EMBEDDING_DIMENSION  # 1024
```

---

## 🤝 Contributing

Contributions are welcome!
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

* **Code:** MIT
* **Models:** MIT (BitCPM4, Qwen3)

---

Built with ❤️ for developers tired of flaky AI tests.

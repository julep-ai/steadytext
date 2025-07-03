<p align="center">
    <img src="https://github.com/user-attachments/assets/735141f8-56ff-40ce-8a4e-013dbecfe299" alt="SteadyText Logo" height=320 width=480 />
</p>

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

# Deterministic text generation (uses daemon by default)
code = steadytext.generate("implement binary search in Python")
assert "def binary_search" in code  # Always passes!

# Streaming (also deterministic)
for token in steadytext.generate_iter("explain quantum computing"):
    print(token, end="", flush=True)

# Deterministic embeddings (uses daemon by default)
vec = steadytext.embed("Hello world")  # 1024-dim numpy array

# Explicit daemon usage (ensures connection)
from steadytext.daemon import use_daemon
with use_daemon():
    code = steadytext.generate("implement quicksort")
    embedding = steadytext.embed("machine learning")

# Model switching (v2.0.0+)
fast_response = steadytext.generate("Quick task", model="gemma-3n-2b")
quality_response = steadytext.generate("Complex analysis", model="gemma-3n-4b")

# Size-based selection (v2.0.0+)
small = steadytext.generate("Simple task", size="small")      # Gemma-3n-2B (default)
large = steadytext.generate("Complex task", size="large")    # Gemma-3n-4B
```

_Or,_

```bash
echo "hello" | uvx steadytext
```

---

## 🔧 How It Works

SteadyText achieves determinism via:

* **Customizable seeds:** Control determinism with a `seed` parameter, while still defaulting to `42`.
* **Greedy decoding:** Always chooses highest-probability token
* **Frecency cache:** LRU cache with frequency counting—popular prompts stay cached longer
* **Quantized models:** 8-bit quantization ensures identical results across platforms
* **Model switching:** Dynamically switch between models while maintaining determinism (v1.0.0+)
* **Daemon architecture:** Persistent model serving eliminates loading overhead (v1.2.0+)

This means `generate("hello")` returns the exact same 512 tokens on any machine, every single time.

### ⚡ Daemon Architecture (Default)

SteadyText uses a daemon architecture by default for optimal performance:

* **Persistent serving:** Models stay loaded in memory between requests
* **Zero loading overhead:** Skip the 2-3 second model loading time on each call
* **Automatic fallback:** Gracefully falls back to direct model loading if daemon unavailable
* **Centralized caching:** Consistent cache behavior between daemon and direct access
* **Background operation:** Daemon runs silently in the background

```python
# Daemon is used automatically - no setup needed
text = steadytext.generate("Hello world")  # Uses daemon by default

# Explicit daemon usage (ensures connection)
from steadytext.daemon import use_daemon
with use_daemon():
    text = steadytext.generate("Hello world")
    embedding = steadytext.embed("Some text")

# Disable daemon globally
import os
os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
text = steadytext.generate("Hello world")  # Direct model loading
```

---

## 📦 Installation & Models

Install stable release:

```bash
pip install steadytext
```

#### Models

**Default models (v2.0.0)**:

* Generation: `Gemma-3n-E2B-it-Q8_0` (2.0GB) - State-of-the-art 2B model
* Embeddings: `Qwen3-Embedding-0.6B-Q8_0` (610MB) - 1024-dimensional embeddings

**Dynamic model switching (v1.0.0+):**

Switch between different models at runtime:

```python
# Use built-in model registry
text = steadytext.generate("Hello", model="gemma-3n-4b")

# Use size parameter for Gemma-3n models
text = steadytext.generate("Hello", size="large")  # Uses Gemma-3n-4B

# Or specify custom models
text = steadytext.generate(
    "Hello",
    model_repo="ggml-org/gemma-3n-E4B-it-GGUF",
    model_filename="gemma-3n-E4B-it-Q8_0.gguf"
)
```

Available models: `gemma-3n-2b`, `gemma-3n-4b`

Size shortcuts: `small` (2B, default), `large` (4B)

> Each model produces deterministic outputs. The default model remains fixed per major version.

### Breaking Changes in v2.0.0+

* **Gemma-3n models:** Switched from Qwen3 to Gemma-3n for state-of-the-art performance
* **Thinking mode removed:** `thinking_mode` parameter and `--think` flag have been deprecated
* **Model registry updated:** Focus on Gemma-3n models (2B and 4B variants)
* **Reduced context:** Default context window reduced from 3072 to 2048 tokens
* **Reduced output:** Default max tokens reduced from 1024 to 512

### Previous Changes in v1.3.0+

* **Daemon enabled by default:** Use `STEADYTEXT_DISABLE_DAEMON=1` to opt-out
* **Streaming by default:** CLI streams output by default, use `--wait` to disable
* **Quiet by default:** CLI is quiet by default, use `--verbose` for informational output
* **Centralized caching:** Cache system now shared between daemon and direct access
* **New CLI syntax:** Use `echo "prompt" | st` instead of `st generate "prompt"`

---

## ⚡ Performance

SteadyText delivers deterministic AI with production-ready performance:

* **Text Generation**: 21.4 generations/sec (46.7ms latency)
* **Embeddings**: 104-599 embeddings/sec (single to batch-50)
* **Cache Speedup**: 48x faster for repeated prompts
* **Memory**: ~1.4GB models, 150-200MB runtime
* **100% Deterministic**: Same output every time, verified across 100+ test runs
* **Accuracy**: 69.4% similarity for related texts, correct ordering maintained

📊 **[Full benchmarks →](docs/benchmarks.md)**

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

### Daemon Management

```bash
# Daemon commands
st daemon start                    # Start daemon in background
st daemon start --foreground       # Run daemon in foreground
st daemon status                   # Check daemon status
st daemon status --json            # JSON status output
st daemon stop                     # Stop daemon gracefully
st daemon stop --force             # Force stop daemon
st daemon restart                  # Restart daemon

# Daemon configuration
st daemon start --host 127.0.0.1 --port 5557  # Custom host/port
```

### Text Generation

```bash
# Generate text (streams by default, uses daemon automatically)
echo "write a hello world function" | st

# Disable streaming (wait for complete output)
echo "write a function" | st --wait

# Enable verbose output
echo "explain recursion" | st --verbose


# JSON output with metadata
echo "hello world" | st --json

# Get log probabilities
echo "predict next word" | st --logprobs
```

### Other Operations

```bash
# Get embeddings
echo "machine learning" | st embed

# Vector operations
st vector similarity "cat" "dog"
st vector search "Python" candidate1.txt candidate2.txt candidate3.txt

# Create and search FAISS indices
st index create *.txt --output docs.faiss
st index search docs.faiss "how to install" --top-k 5

# Generate with automatic context from index
echo "what is the configuration?" | st --index-file docs.faiss

# Disable daemon for specific command
STEADYTEXT_DISABLE_DAEMON=1 echo "hello" | st

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
import steadytext
import numpy as np
from typing import List, Optional, Dict, Tuple, Iterator, Union

# Text generation with structured output
def generate(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    return_logprobs: bool = False,
    size: Optional[str] = None,
    seed: int = 42,
    schema: Optional[Union[Dict, type]] = None,
    regex: Optional[str] = None,
    choices: Optional[List[str]] = None
) -> Union[str, Tuple[str, Optional[Dict]]]:
    # ... implementation ...

# Streaming generation
def generate_iter(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    seed: int = 42
) -> Iterator[str]:
    # ... implementation ...

# Embeddings
def embed(
    text_input: Union[str, List[str]],
    seed: int = 42
) -> np.ndarray:
    # ... implementation ...

# Model preloading
def preload_models(verbose: bool = False) -> None:
    # ... implementation ...
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
echo "question" | st --index-file my_index.faiss
```

📚 [Full API Documentation](docs/api.md)

---

## 🔧 Configuration

### Cache Configuration

Control caching behavior via environment variables (affects both daemon and direct access):

```bash
# Generation cache (default: 256 entries, 50MB)
export STEADYTEXT_GENERATION_CACHE_CAPACITY=256
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=50

# Embedding cache (default: 512 entries, 100MB)
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=512
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=100
```

### Daemon Configuration

```bash
# Disable daemon globally (use direct model loading)
export STEADYTEXT_DISABLE_DAEMON=1

# Daemon connection settings
export STEADYTEXT_DAEMON_HOST=127.0.0.1
export STEADYTEXT_DAEMON_PORT=5557
```

### Model Downloads

```bash
# Allow model downloads in tests
export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true
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

# List of strings (returns a single, averaged embedding)
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
* **Models:** MIT (Qwen3)

---

## 📈 What's New in v2.0.0+

### Gemma-3n Models (v2.0.0+)
- **State-of-the-art performance** with Gemma-3n model family
- **Size-based selection** - use `size="small"` or `size="large"` parameters
- **Improved quality** while maintaining deterministic outputs
- **Streamlined model registry** focused on proven architectures

### Daemon Architecture (v1.2.0+)
- **Persistent model serving** with ZeroMQ for 10-100x faster repeated calls
- **Automatic fallback** to direct model loading when daemon unavailable
- **Zero configuration** - daemon starts automatically on first use
- **Background operation** - daemon runs silently in the background

### Centralized Cache System (v1.3.0+)
- **Unified caching** - consistent behavior between daemon and direct access
- **Thread-safe SQLite backend** for reliable concurrent access
- **Shared cache files** across all access modes
- **Cache integration** with daemon server for optimal performance

### Improved CLI Experience (v1.3.0+)
- **Streaming by default** - see output as it's generated
- **Quiet by default** - clean output without informational messages
- **New pipe syntax** - `echo "prompt" | st` for better unix integration
- **Daemon management** - built-in commands for daemon lifecycle

---

Built with ❤️ for developers tired of flaky AI tests.

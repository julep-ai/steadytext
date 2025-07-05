# SteadyText

*Deterministic text generation and embeddings with zero configuration*

[![PyPI Version](https://img.shields.io/pypi/v/steadytext.svg)](https://pypi.org/project/steadytext/)
[![Python Versions](https://img.shields.io/pypi/pyversions/steadytext.svg)](https://pypi.org/project/steadytext/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/julep-ai/steadytext/blob/main/LICENSE)

**Same input → same output. Every time.**

No more flaky tests, unpredictable CLI tools, or inconsistent docs. SteadyText makes AI outputs as reliable as hash functions.

Ever had an AI test fail randomly? Or a CLI tool give different answers each run? SteadyText makes AI outputs reproducible - perfect for testing, tooling, and anywhere you need consistent results.

!!! tip "Powered by Julep"
    ✨ _Powered by open-source AI workflows from [**Julep**](https://julep.ai)._ ✨

---

## 🚀 Quick Start

```bash
# Using UV (recommended - 10-100x faster)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv add steadytext

# Or using pip
pip install steadytext
```

=== "Python API"

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
    
    # Structured generation (v2.4.1+)
    from pydantic import BaseModel
    class User(BaseModel):
        name: str
        age: int
    
    result = steadytext.generate("Create user Alice age 30", schema=User)
    # Returns: '...<json-output>{"name": "Alice", "age": 30}</json-output>'
    ```

=== "Command Line"

    ```bash
    # Generate text (pipe syntax)
    echo "hello world" | st

    # Stream output (default)  
    echo "explain recursion" | st

    # Wait for complete output
    echo "explain recursion" | st --wait

    # Get embeddings
    echo "machine learning" | st embed

    # Start daemon for faster responses
    st daemon start
    ```

---

## 🔧 How It Works

SteadyText achieves determinism via:

* **Customizable seeds**: Control determinism with a `seed` parameter, while still defaulting to `42`.
* **Greedy decoding**: Always chooses highest-probability token
* **Frecency cache**: LRU cache with frequency counting—popular prompts stay cached longer
* **Quantized models**: 8-bit quantization ensures identical results across platforms

This means `generate("hello")` returns the exact same 512 tokens on any machine, every single time.

## 🌐 Ecosystem

SteadyText is more than just a library. It's a full ecosystem for deterministic AI:

- **Python Library**: The core `steadytext` library for programmatic use in your applications.
- **Command-Line Interface (CLI)**: A powerful `st` command to use SteadyText from your shell for scripting and automation.
- **Zsh Plugin**: Supercharge your shell with AI-powered command suggestions and history search.
- **PostgreSQL Extension**: Run deterministic AI functions directly within your PostgreSQL database.
- **Cloudflare Worker**: Deploy SteadyText to the edge with a Cloudflare Worker for distributed, low-latency applications.

### Daemon Mode (v1.3+)

SteadyText includes a daemon mode that keeps models loaded in memory for instant responses:

* **160x faster first request**: No model loading overhead
* **Persistent cache**: Shared across all operations
* **Automatic fallback**: Works without daemon if unavailable
* **Zero configuration**: Daemon used by default when available

```bash
# Start daemon
st daemon start

# Check status
st daemon status

# All operations now use daemon automatically
echo "hello" | st  # Instant response!
```

### FAISS Indexing

Create and search vector indexes for retrieval-augmented generation:

```bash
# Create index from documents
st index create *.txt --output docs.faiss

# Search index
st index search docs.faiss "query text" --top-k 5

# Use with generation (automatic with default.faiss)
echo "explain this error" | st --index-file docs.faiss
```

---

## 📦 Installation & Models

Install stable release:

```bash
# Using UV (recommended - 10-100x faster)
uv add steadytext

# Or using pip
pip install steadytext
```

### Models

**Current models (v2.0.0+)**:

* Generation: `Gemma-3n-E2B-it-Q8_0.gguf` (2.0GB) - Gemma-3n-2B (default)
* Generation: `Gemma-3n-E4B-it-Q8_0.gguf` (4.2GB) - Gemma-3n-4B (optional)
* Embeddings: `Qwen3-Embedding-0.6B-Q8_0.gguf` (610MB)

!!! note "Version Stability"
    Each major version will use a fixed set of models only, so that only forced upgrades from pip will change the models (and the deterministic output)

---

## 🎯 Use Cases

!!! success "Perfect for"
    * **Testing AI features**: Reliable asserts that never flake
    * **Deterministic CLI tooling**: Consistent outputs for automation  
    * **Reproducible documentation**: Examples that always work
    * **Offline/dev/staging environments**: No API keys needed
    * **Semantic caching and embedding search**: Fast similarity matching

!!! warning "Not ideal for"
    * Creative or conversational tasks
    * Latest knowledge queries  
    * Large-scale chatbot deployments

---

## 📋 Examples

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

📚 **[Full API Documentation →](api/)**

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

## 🤝 Contributing

Contributions are welcome! See [Contributing Guide](contributing.md) for guidelines.

---

## 📄 License

* **Code**: MIT
* **Models**: MIT (Qwen3)

---

Built with ❤️ for developers tired of flaky AI tests.
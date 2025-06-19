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
    ```

=== "Command Line"

    ```bash
    # Generate text
    st generate "hello world"

    # Stream output  
    st generate "explain recursion" --stream

    # Get embeddings
    st embed "machine learning"

    # Preload models
    st models --preload
    ```

---

## 🔧 How It Works

SteadyText achieves determinism via:

* **Fixed seeds**: Constant randomness seed (`42`)
* **Greedy decoding**: Always chooses highest-probability token
* **Frecency cache**: LRU cache with frequency counting—popular prompts stay cached longer
* **Quantized models**: 8-bit quantization ensures identical results across platforms

This means `generate("hello")` returns the exact same 512 tokens on any machine, every single time.

---

## 📦 Installation & Models

Install stable release:

```bash
pip install steadytext
```

### Models

**Corresponding to pypi versions `0.x.y`**:

* Generation: `BitCPM4-1B-Q8_0` (1.3GB)
* Embeddings: `Qwen3-0.6B-Q8_0` (610MB)

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
* **Models**: MIT (BitCPM4, Qwen3)

---

Built with ❤️ for developers tired of flaky AI tests.
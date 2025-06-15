# SteadyText
*Deterministic text generation and embedding with zero configuration*

[![PyPI](https://img.shields.io/pypi/v/steadytext.svg)](https://pypi.org/project/steadytext/)
[![Python Versions](https://img.shields.io/pypi/pyversions/steadytext.svg)](https://pypi.org/project/steadytext/)
[![License](https://img.shields.io/pypi/l/steadytext.svg)](https://github.com/yourusername/steadytext/blob/main/LICENSE) <!-- Placeholder for license badge -->

SteadyText provides perfectly deterministic text generation and embedding outputs with absolutely zero configuration. Install the package, and it just works.

## 🚀 Quick Start

```python
from steadytext import generate, embed
import numpy as np

# Generate 512 tokens - always returns the same string for the same input
text_output = generate("Once upon a time")
print(text_output)

# Generate embedding - always returns the same np.array of shape (1024,)
embedding_vector = embed("Hello world")
print(embedding_vector.shape)
if embedding_vector.any(): # Avoid norm of zero vector
    print(np.linalg.norm(embedding_vector)) # Should be close to 1.0
```

## ✨ Key Features

- **🎯 Deterministic**: Same input always produces the exact same output.
- **⚡ Zero Configuration**: Works immediately after `pip install steadytext`. No API keys, no model selection, no parameters to tune.
- **📦 Self-Contained Models**: Necessary language models are automatically downloaded on first use.
- **🛡️ Never Fails**: Designed to be extremely robust, providing deterministic fallbacks for any edge cases or errors.
- **📏 Fixed Output Sizes**:
    - `generate()`: Always produces a string derived from approximately 512 tokens (actual token count depends on model's tokenizer for the generated string).
    - `embed()`: Always produces a 1024-dimensional, L2-normalized `numpy.ndarray` (float32).

## 📦 Installation

```bash
pip install steadytext
```

If you want to test the latest development version before it is released on
PyPI, install directly from the GitHub repository:

```bash
pip install git+https://github.com/steadytext/steadytext.git
```

Models will be downloaded automatically to your local cache directory (e.g., `~/.cache/steadytext/models`) the first time you call `generate()` or `embed()`. This is a one-time download.
The generation model is ~1.3GB (openbmb.BitCPM4-1B.Q8_0.gguf).
The embedding model is ~610MB (Qwen3-Embedding-0.6B-Q8_0.gguf).

*(Further sections like 'How It Works', 'API Reference', 'Contributing', 'License' would be added here)*

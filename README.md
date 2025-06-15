# SteadyText
*Deterministic text generation and embedding with zero configuration*

[![PyPI](https://img.shields.io/pypi/v/steadytext.svg)](https://pypi.org/project/steadytext/)
[![Python Versions](https://img.shields.io/pypi/pyversions/steadytext.svg)](https://pypi.org/project/steadytext/)
[![License](https://img.shields.io/pypi/l/steadytext.svg)](https://github.com/yourusername/steadytext/blob/main/LICENSE) <!-- Placeholder for license badge -->

SteadyText provides deterministic text generation and embeddings. Same input, same output. Every time.

No more flaky tests or unpredictable AI. Perfect for testing, CLI tools, and anywhere you need reproducible results.

## 🚀 Quick Start

```python
import steadytext
import numpy as np

# Text generation
text = steadytext.generate("Once upon a time")
print(text[:100])

# With logprobs
text, logprobs = steadytext.generate("Explain quantum computing", return_logprobs=True)

# Streaming
for token in steadytext.generate_iter("The future of AI"):
    print(token, end="", flush=True)

# Embeddings
embedding = steadytext.embed("Hello world")  # 1024-dim vector
```

## Features

- **Deterministic**: Same input → same output, always
- **Zero config**: Just `pip install steadytext` and go
- **Fast**: Frecency cache makes repeated queries instant
- **Reliable**: Fallback mechanisms ensure it never crashes
- **Self-contained**: Models download automatically (~1.9GB total)
- **Fixed outputs**: 512 tokens for text, 1024-dim vectors for embeddings

## 📦 Installation

```bash
pip install steadytext
```

For the latest development version:

```bash
pip install git+https://github.com/steadytext/steadytext.git
```

Models are automatically downloaded on first use to your cache directory:
- **Linux/macOS**: `~/.cache/steadytext/models/`
- **Windows**: `%LOCALAPPDATA%\steadytext\steadytext\models\`

Model sizes:
- Generation model: ~1.3GB (openbmb.BitCPM4-1B.Q8_0.gguf)
- Embedding model: ~610MB (Qwen3-Embedding-0.6B-Q8_0.gguf)

## 🖥️ CLI Usage

Available via `steadytext` or `st`:

```bash
st "write a hello world function in Python"
st "write a story" --stream
st embed "hello world"
st cache --status
st models --preload
```

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

## Why Use SteadyText?

Regular AI models give different outputs each time. This breaks:
- Tests (flaky assertions)
- Build processes (inconsistent outputs)
- Documentation (changes every run)
- Any reproducible workflow

SteadyText fixes this. Plus, with built-in caching, repeated queries are instant.

Configure cache behavior with environment variables:
```bash
# Generation cache (default: 256 entries, 50MB)
export STEADYTEXT_GENERATION_CACHE_CAPACITY=512
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=100.0

# Embedding cache (default: 512 entries, 100MB)  
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=1024
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=200.0
```

## CLI Examples

```bash
# Basic usage - same query always returns same command
$ st "find all .py files modified in last week"
find . -name "*.py" -mtime -7

# Command helper
alias howto='st'
$ howto 'compress directory with progress bar'
tar -cf - directory/ | pv | gzip > directory.tar.gz

# Git automation
gitdo() {
    $(st "git command to $*")
}
$ gitdo 'undo last commit but keep changes'

# Error explanations
$ echo "error: ECONNREFUSED" | st 'make user-friendly'
Unable to connect to the server. Please check your connection.

# Config generation  
$ st 'nginx config for SPA on port 3000' > nginx.conf

# Test data
for i in {1..100}; do
    st "fake user data seed:$i" >> test-users.json
done
```


## More Examples

### ASCII Art
```python
import steadytext

cowsay = lambda what: steadytext.generate(f"Draw ASCII art of a cow saying: {what}")
print(cowsay("Hello World"))  # Same cow every time
```

### Quick CLI Tools
```python
#!/usr/bin/env python3
import sys
import steadytext

def motivate():
    return steadytext.generate("Motivational quote")

def excuse():
    return steadytext.generate("Creative excuse for being late")

if __name__ == "__main__":
    print(locals()[sys.argv[1]]())
```

### Game Content
```python
def generate_npc_dialogue(npc_name, player_level):
    return steadytext.generate(f"NPC {npc_name} greets level {player_level} player")
    
# Same NPC always says the same thing
```

### Testing

### Mock Data
```python
def generate_user_profile(user_id):
    return {
        "bio": steadytext.generate(f"Write bio for user {user_id}"),
        "interests": steadytext.generate(f"List hobbies for user {user_id}")
    }

assert generate_user_profile(123) == generate_user_profile(123)  # Always passes
```

### Test Fixtures
```python
import json

def generate_test_json(schema_name):
    return steadytext.generate(f"Generate valid JSON for {schema_name} schema")
    
def generate_sql_fixture(table_name):
    return steadytext.generate(f"SQL INSERT for {table_name} test data")
```

### AI Mocking
```python
class MockAI:
    def complete(self, prompt):
        return steadytext.generate(prompt)
    
    def embed(self, text):
        return steadytext.embed(text)
```

### Error Messages
```python
def get_user_friendly_error(error_code):
    return steadytext.generate(f"Explain error {error_code} in simple terms")
```

### Semantic Cache Keys
```python
import hashlib

def semantic_cache_key(query):
    embedding = steadytext.embed(query)
    return hashlib.sha256(embedding.tobytes()).hexdigest()
```

### Content Generation
```python
def fake_review(product_id, stars):
    return steadytext.generate(f"Review for product {product_id} with {stars} stars")

def fake_bio(profession):
    return steadytext.generate(f"Professional bio for {profession}")
```

### Auto Documentation
```python
def auto_document_function(func_name, params):
    return steadytext.generate(f"Write docstring for {func_name}({params})")
```

### Fuzz Testing
```python
def generate_fuzz_input(test_name, iteration):
    return steadytext.generate(f"Fuzz input for {test_name} iteration {iteration}")

for i in range(100):
    input_data = generate_fuzz_input("parser_test", i)
    test_parser(input_data)
```

### Mock API Responses
```python
def generate_fake_api_response(endpoint):
    return steadytext.generate(f"Mock response for {endpoint}")
```

### Pseudo Translation
```python
def pseudo_translate(text, language):
    return steadytext.generate(f'Translate "{text}" to {language}')
```

### Story Generation
```python
def generate_story_chapter(book_id, chapter_num):
    return steadytext.generate(f"Chapter {chapter_num} of book {book_id}")
```

### Test Oracles
```python
def generate_expected_output(input_data):
    return steadytext.generate(f"Expected output for: {input_data}")

def test_my_function():
    expected = generate_expected_output("test123")
    actual = my_function("test123")
    assert actual == expected
```

## How It Works

1. Fixed seed (42)
2. Temperature=0, top_k=1
3. Model state reset between calls
4. Hash-based fallbacks if models fail
5. L2-normalized embeddings

Result: AI that behaves like a hash function.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
# SteadyText
*Deterministic text generation and embedding with zero configuration*

[![PyPI](https://img.shields.io/pypi/v/steadytext.svg)](https://pypi.org/project/steadytext/)
[![Python Versions](https://img.shields.io/pypi/pyversions/steadytext.svg)](https://pypi.org/project/steadytext/)
[![License](https://img.shields.io/pypi/l/steadytext.svg)](https://github.com/yourusername/steadytext/blob/main/LICENSE) <!-- Placeholder for license badge -->

SteadyText provides perfectly deterministic text generation and embedding outputs with absolutely zero configuration. Install the package, and it just works.

## 🚀 Quick Start

```python
import steadytext
import numpy as np

# Generate text - always returns the same string for the same input
text = steadytext.generate("Once upon a time")
print(text[:100])  # First 100 characters

# Generate text with logprobs
text, logprobs = steadytext.generate("Explain quantum computing", return_logprobs=True)
print(f"Generated: {text[:50]}...")
print(f"Logprobs available: {logprobs is not None}")

# Stream text generation word by word
for token in steadytext.generate_iter("The future of AI"):
    print(token, end="", flush=True)
print()

# Generate embeddings - always returns the same vector for the same input
embedding = steadytext.embed("Hello world")
print(f"Shape: {embedding.shape}")  # (1024,)
print(f"Norm: {np.linalg.norm(embedding):.4f}")  # ~1.0 (L2-normalized)
```

## ✨ Key Features

- **🎯 Perfectly Deterministic**: Same input always produces the exact same output across runs and machines
- **⚡ Zero Configuration**: Works immediately after `pip install steadytext`. No API keys, no model selection, no parameters to tune
- **📦 Self-Contained Models**: Language models are automatically downloaded on first use (~1.9GB total)
- **🛡️ Never Fails**: Designed to be extremely robust with deterministic fallbacks for any edge cases
- **🔄 Streaming Support**: Generate text iteratively with `generate_iter()` for real-time output
- **📊 Logprobs Access**: Optionally get log probabilities alongside generated text
- **💾 Intelligent Caching**: SQLite-backed frecency cache makes repeated generations instant
- **📏 Fixed Output Sizes**:
  - `generate()`: Always produces up to 512 tokens of text
  - `embed()`: Always returns a 1024-dimensional L2-normalized float32 numpy array

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

## 📖 API Reference

### Text Generation

#### `generate(prompt: str, return_logprobs: bool = False) -> Union[str, Tuple[str, Optional[Dict]]]`

Generate deterministic text from a prompt.

```python
# Basic usage
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
# Silent preload
steadytext.preload_models()

# Verbose preload
steadytext.preload_models(verbose=True)
```

#### `get_model_cache_dir() -> str`

Get the path to the model cache directory.

```python
cache_dir = steadytext.get_model_cache_dir()
print(f"Models are stored in: {cache_dir}")
```

### Constants

```python
# Available constants
steadytext.DEFAULT_SEED  # 42 - Used internally for determinism
steadytext.GENERATION_MAX_NEW_TOKENS  # 512 - Max tokens for generation
steadytext.EMBEDDING_DIMENSION  # 1024 - Embedding vector size
```

## 🚀 Why Caching Matters

SteadyText uses a SQLite-backed frecency cache that persists across runs. This means:

- **Instant Repeated Generations**: Once you generate text for a prompt, subsequent calls with the same prompt return instantly from cache
- **Cross-Session Persistence**: Cache survives program restarts - build up a library of instant responses
- **Smart Eviction**: Frecency (frequency + recency) algorithm keeps your most useful generations
- **Production Ready**: Configurable size limits prevent unbounded growth

Configure cache behavior with environment variables:
```bash
# Generation cache (default: 256 entries, 50MB)
export STEADYTEXT_GENERATION_CACHE_CAPACITY=512
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=100.0

# Embedding cache (default: 512 entries, 100MB)  
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=1024
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=200.0
```

## 💡 CLI Tool Ideas

Imagine a `steadytext` CLI that leverages the cache for instant, deterministic command assistance:

```bash
# Basic usage - same query always returns same command
$ steadytext "find all .py files modified in last week"
find . -name "*.py" -mtime -7

# Build your own deterministic command oracle
alias howto='steadytext'
$ howto 'compress directory with progress bar'
tar -cf - directory/ | pv | gzip > directory.tar.gz

# Parameterizable shell functions
gitdo() {
    $(steadytext "git command to $*")
}
$ gitdo 'undo last commit but keep changes'
$ gitdo 'show commits by author in last month'

# Stable explanations that build your knowledge base
alias explain='steadytext explain'
$ explain 'what does chmod 755 mean'
# Always get the SAME explanation - cached for instant access

# Pipeline-friendly deterministic processing
$ echo "error: ECONNREFUSED" | steadytext 'make user-friendly'
Unable to connect to the server. Please check your connection.

# Reproducible config generation
$ steadytext 'nginx config for SPA on port 3000' > nginx.conf
# Regenerating gives identical config - git-friendly!

# Deterministic test data that's instant after first run
for i in {1..1000}; do
    steadytext "fake user data seed:$i" >> test-users.json
done
```

The killer feature: your AI assistant becomes predictable AND fast. No more:
- Getting different suggestions for the same question
- Waiting for model inference on commands you've asked before  
- Worrying about non-deterministic outputs in scripts

Instead, you build a personal, deterministic command cache that gets faster with use!

## 🎨 Creative Examples & Use Cases

### Fun & Silly Examples

#### Deterministic ASCII Art Generator
```python
import steadytext

# Always generates the same ASCII art for the same input
cowsay = lambda what: steadytext.generate(f"Draw ASCII art of a cow saying: {what}")
fortune_cookie = lambda: steadytext.generate("Write a fortune cookie message")
ascii_banner = lambda text: steadytext.generate(f"Create ASCII art banner for: {text}")

# Your cowsay will always produce the same cow!
print(cowsay("Hello World"))
```

#### CLI Tool Generators
```python
#!/usr/bin/env python3
import sys
import steadytext

# Deterministic CLI tools
def motivate():
    return steadytext.generate("Motivational quote")

def excuse():
    return steadytext.generate("Creative excuse for being late")

def technobabble():
    return steadytext.generate("Technical-sounding nonsense")

if __name__ == "__main__":
    print(locals()[sys.argv[1]]())
```

#### Stable Game Content
```python
# Procedural but predictable game content
def generate_npc_dialogue(npc_name, player_level):
    prompt = f"NPC {npc_name} greets level {player_level} player"
    return steadytext.generate(prompt)

def generate_quest_description(seed):
    return steadytext.generate(f"Epic quest number {seed}")

# Same NPC always says the same thing to same-level players!
```

### Practical Testing Examples

#### Stable Mock Data Generator
```python
# Perfect for tests that need realistic but consistent data
def generate_user_profile(user_id):
    return {
        "bio": steadytext.generate(f"Write bio for user {user_id}"),
        "interests": steadytext.generate(f"List hobbies for user {user_id}"),
        "motto": steadytext.generate(f"Life motto for user {user_id}")
    }

# User 123 will ALWAYS have the same profile
assert generate_user_profile(123) == generate_user_profile(123)
```

#### Test Fixture Generator
```python
import json

def generate_test_json(schema_name):
    prompt = f"Generate valid JSON for {schema_name} schema"
    # Always generates the same test data!
    return steadytext.generate(prompt)

def generate_sql_fixture(table_name):
    return steadytext.generate(f"SQL INSERT for {table_name} test data")

# Same schema = same fixture
assert generate_test_json("user") == generate_test_json("user")
```

#### Deterministic Mocking
```python
class MockAI:
    def complete(self, prompt):
        # Same prompt always returns same response
        return steadytext.generate(prompt)
    
    def embed(self, text):
        return steadytext.embed(text)

# Use in tests - completely deterministic!
def test_ai_pipeline():
    ai = MockAI()
    result = my_ai_function(ai)
    assert result == "expected"  # Always passes
```

### Production Use Cases

#### Stable Error Messages
```python
def get_user_friendly_error(error_code):
    return steadytext.generate(
        f"Explain error {error_code} in simple terms"
    )

# Error explanations are consistent across deployments
error_msg = get_user_friendly_error("E404")
```

#### Semantic Caching with Stable Embeddings
```python
import hashlib

def semantic_cache_key(query):
    # Deterministic embedding as cache key
    embedding = steadytext.embed(query)
    return hashlib.sha256(embedding.tobytes()).hexdigest()

# Similar queries map to same cache entry
cache = {}
key = semantic_cache_key("What's the weather?")
if key not in cache:
    cache[key] = expensive_api_call()
```

#### Reproducible Content Generation
```python
# For mockups and prototypes
def lorem_ipsum_2024(paragraphs=3):
    return steadytext.generate(f"Modern lorem ipsum {paragraphs} paragraphs")

def fake_review(product_id, stars):
    return steadytext.generate(f"Review for product {product_id} with {stars} stars")

def fake_bio(profession):
    return steadytext.generate(f"Professional bio for {profession}")
```

### Development & Debugging

#### Deterministic Code Comments
```python
def auto_document_function(func_name, params):
    prompt = f"Write docstring for {func_name}({params})"
    return steadytext.generate(prompt)

# Same function signature = same documentation
docstring = auto_document_function("calculate_tax", "income, rate")
```

#### Reproducible Fuzzing
```python
def generate_fuzz_input(test_name, iteration):
    return steadytext.generate(
        f"Fuzz input for {test_name} iteration {iteration}"
    )

# Fuzz testing with reproducible inputs
for i in range(100):
    input_data = generate_fuzz_input("parser_test", i)
    # Same iteration always gets same fuzz input
    test_parser(input_data)
```

#### Stable Placeholder Content
```python
# Generate consistent placeholder data for development
def generate_fake_api_response(endpoint):
    return steadytext.generate(f"Mock response for {endpoint}")

# Always get the same mock response
mock_users = generate_fake_api_response("/api/users")
mock_posts = generate_fake_api_response("/api/posts")
```

### Creative Applications

#### Deterministic "Translation"
```python
# Poor man's translation with consistent results
def pseudo_translate(text, language):
    return steadytext.generate(f'Translate "{text}" to {language}')

# Always get the same "translation"
spanish = pseudo_translate("Hello", "Spanish")
assert spanish == pseudo_translate("Hello", "Spanish")
```

#### Procedural Story Generation
```python
def generate_story_chapter(book_id, chapter_num):
    prompt = f"Chapter {chapter_num} of book {book_id}"
    return steadytext.generate(prompt)

# Same book + chapter = same content
chapter = generate_story_chapter("mystery_101", 5)
```

#### Test Oracle Generation
```python
def generate_expected_output(input_data):
    """Generate consistent expected outputs for tests"""
    return steadytext.generate(f"Expected output for: {input_data}")

def test_my_function():
    input_val = "test123"
    expected = generate_expected_output(input_val)
    actual = my_function(input_val)
    # Expected is always the same for this input
    assert actual == expected
```

## 🔧 How It Works

SteadyText achieves perfect determinism through:

1. **Fixed Random Seeds**: All operations use a constant seed (42)
2. **Deterministic Sampling**: Temperature=0, top_k=1 for generation
3. **Model State Reset**: Model cache is cleared between generations
4. **Fallback Mechanisms**: Hash-based text generation when models fail
5. **Normalized Outputs**: Embeddings are always L2-normalized

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
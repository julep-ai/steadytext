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

### Installing from PyPI

```bash
pip install steadytext
```

### Installing from Source (Required for proper llama-cpp-python build)

Due to the specific build requirements for the inference-sh fork of llama-cpp-python, you may need to install from source:

```bash
# Clone the repository
git clone https://github.com/julep-ai/steadytext.git
cd steadytext

# Set required environment variables
export FORCE_CMAKE=1
export CMAKE_ARGS="-DLLAVA_BUILD=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF"

# Install with UV (recommended)
uv sync

# Or install with pip
pip install -e .
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
fast_response = steadytext.generate("Quick task", size="small")  # Gemma-3n-2B
quality_response = steadytext.generate("Complex analysis", size="large")  # Gemma-3n-4B

# Size-based selection (v2.0.0+)
small = steadytext.generate("Simple task", size="small")      # Gemma-3n-2B (default)
large = steadytext.generate("Complex task", size="large")    # Gemma-3n-4B
```

_Or,_

```bash
echo "hello" | uvx steadytext
```

---

## 🐘 PostgreSQL Extension

Transform your PostgreSQL database into an AI-powered system with **pg_steadytext** - the production-ready PostgreSQL extension that brings deterministic AI directly to your SQL queries.

### Key Features

- **Native SQL Functions**: Generate text and embeddings using simple SQL commands
- **Async Processing**: Non-blocking AI operations with queue-based background workers  
- **AI Summarization**: Aggregate functions for intelligent text summarization with TimescaleDB support
- **Structured Generation**: Generate JSON, regex-constrained text, and multiple-choice outputs
- **pgvector Integration**: Seamless compatibility for similarity search and vector operations
- **Built-in Caching**: PostgreSQL-based frecency cache that mirrors SteadyText's performance

### Quick Example

```sql
-- Generate text
SELECT steadytext_generate('Write a product description for wireless headphones');

-- Create embeddings for similarity search
SELECT steadytext_embed('machine learning') <-> steadytext_embed('artificial intelligence');

-- AI-powered summarization
SELECT ai_summarize(content) AS summary
FROM documents
WHERE created_at > NOW() - INTERVAL '1 day'
GROUP BY category;

-- Structured JSON generation
SELECT steadytext_generate_json(
    'Create a user profile',
    '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'::jsonb
);
```

📚 **[Full PostgreSQL Extension Documentation →](pg_steadytext/)**

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

## 🌐 Ecosystem

SteadyText is more than just a library. It's a full ecosystem for deterministic AI:

- **Python Library**: The core `steadytext` library for programmatic use in your applications.
- **Command-Line Interface (CLI)**: A powerful `st` command to use SteadyText from your shell for scripting and automation.
- **PostgreSQL Extension (pg_steadytext)**: Production-ready extension with async processing, AI summarization, and structured generation for SQL-native AI operations.
- **Zsh Plugin**: Supercharge your shell with AI-powered command suggestions and history search.
- **Cloudflare Worker**: Deploy SteadyText to the edge with a Cloudflare Worker for distributed, low-latency applications.

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

## Fun Goofy Sample You Can Try

They will (literally) **always** do this...

```bash
❯ st generate --verbose --size large --eos-string STOP "DON'T SAY STOP (ALL CAPS) NO MATTER WHAT"
Understood. I will not use the word "%
```

> [!NOTE]
> This is by far the single best explanation of recursion in existence.

```bash
$> echo "explain recursion in pig latin" | st --verbose --size large

Okay, let's explain recursion in Pig Latin!  It's a bit tricky to do *in* Pig Latin, but I'll try to explain the concept and then give a Pig Latin-flavored analogy.

**What is Recursion? (In English)**

Recursion is like a set of instructions that calls *itself* to solve a smaller version of the same problem. Think of it like Russian nesting dolls (Matryoshka dolls). Each doll contains a smaller version of itself.

Here's the breakdown:

1. **Base Case:**  This is the *stopping point*.  It's the simplest version of the problem that you can solve directly, *without* calling the function again.  Without a base case, the recursion would go on forever (like an infinite loop!).

2. **Recursive Step:** This is where the function calls *itself*, but with a slightly modified (smaller) version of the original problem.  Each call gets closer to the base case.

**Example (in English):**

Let's say you want to calculate the factorial of a number (e.g., 5! = 5 * 4 * 3 * 2 * 1 = 120).  You can do this recursively:

* **Base Case:** If the number is 1, the factorial is 1.
* **Recursive Step:**  If the number is greater than 1, the factorial is the number multiplied by the factorial of the number minus 1.

**Pig Latin Analogy (Trying to explain it *using* Pig Latin):**

Okay, this is where it gets fun (and a little silly)!  Let's say we want to count the number of "ay" sounds in a word.

Here's how we could *imagine* a recursive Pig Latin function to do this:

\```piglatin
"Ehay-ay"  ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-%
```

---

## ✨ Structured Generation (v2.4.1+)

SteadyText now supports structured generation using llama.cpp's native grammar support, allowing you to force the model's output to conform to a specific format.

- **JSON Generation**: Generate JSON that validates against a schema or Pydantic model.
- **Regex Matching**: Constrain output to a regular expression.
- **Multiple Choice**: Force the output to be one of a list of choices.

### Python API

```python
import steadytext
from pydantic import BaseModel

# JSON generation with a Pydantic model
class User(BaseModel):
    name: str
    email: str

user_json = steadytext.generate(
    "Create a user: name John Doe, email john.doe@example.com",
    schema=User
)
# Output contains: <json-output>{"name": "John Doe", "email": "john.doe@example.com"}</json-output>

# Regex-constrained generation
phone = steadytext.generate("My number is ", regex=r"\(\d{3}\) \d{3}-\d{4}")
# Output: (123) 456-7890

# Multiple choice
response = steadytext.generate("Is this useful?", choices=["Yes", "No"])
# Output: Yes
```

### CLI Support

```bash
# JSON generation with schema
echo "Create a person" | st --schema '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}' --wait

# JSON from schema file
echo "Generate user data" | st --schema user_schema.json --wait

# Regex pattern matching
echo "My phone is" | st --regex '\d{3}-\d{3}-\d{4}' --wait

# Multiple choice selection
echo "Is Python good?" | st --choices "yes,no,maybe" --wait
```

📚 **[Learn more in the Structured Generation Guide](docs/structured-generation.md)**

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
text = steadytext.generate("Hello", size="large")  # Uses Gemma-3n-4B

# Use size parameter for Gemma-3n models
text = steadytext.generate("Hello", size="large")  # Uses Gemma-3n-4B

# Or specify custom models
text = steadytext.generate(
    "Hello",
    model_repo="ggml-org/gemma-3n-E4B-it-GGUF",
    model_filename="gemma-3n-E4B-it-Q8_0.gguf"
)
```

Available models: Gemma-3n models in 2B and 4B variants

Size shortcuts: `small` (2B, default), `large` (4B)

> Each model produces deterministic outputs. The default model remains fixed per major version.

## Version History

| Version | Key Features                                                                                                                            | Default Generation Model                               | Default Embedding Model                                | Default Reranking Model | Python Versions |
| :------ | :-------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- | :----------------------------------------------------- | :---------------------- | :-------------- |
| **2.x** | - **Daemon Mode**: Persistent model serving with ZeroMQ.<br>- **Gemma-3n Models**: Switched to `gemma-3n` for generation.<br>- **Thinking Mode Deprecated**: Removed thinking mode.<br>- **Document Reranking**: Reranking functionality with `Qwen3-Reranker-4B` model (since v2.3.0). | `ggml-org/gemma-3n-E2B-it-GGUF` (gemma-3n-E2B-it-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | `Qwen/Qwen3-Reranker-4B-GGUF` (Qwen3-Reranker-4B-Q8_0.gguf) | `>=3.10, <3.14` |
| **1.x** | - **Model Switching**: Added support for switching models via environment variables.<br>- **Centralized Cache**: Unified cache system.<br>- **CLI Improvements**: Streaming by default, quiet output. | `Qwen/Qwen3-1.7B-GGUF` (Qwen3-1.7B-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | - | `>=3.10, <3.14` |
| **1.0-1.2** | - **Model Switching**: Added support for switching models via environment variables and a model registry.<br>- **Qwen3 Models**: Switched to `qwen3-1.7b` for generation.<br>- **Indexing**: Added support for FAISS indexing. | `Qwen/Qwen3-1.7B-GGUF` (Qwen3-1.7B-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | - | `>=3.10, <3.14` |
| **0.x** | - **Initial Release**: Deterministic text generation and embedding.                                                                      | `Qwen/Qwen1.5-0.5B-Chat-GGUF` (qwen1_5-0_5b-chat-q4_k_m.gguf) | `Qwen/Qwen1.5-0.5B-Chat-GGUF` (qwen1_5-0_5b-chat-q8_0.gguf) | - | `>=3.10`        |

### Breaking Changes in v2.0.0+

* **Gemma-3n models:** Switched from Qwen3 to Gemma-3n for state-of-the-art performance
* **Thinking mode removed:** `thinking_mode` parameter and `--think` flag have been deprecated
* **Model registry updated:** Focus on Gemma-3n models (2B and 4B variants)
* **Reduced context:** Default context window reduced from 3072 to 2048 tokens
* **Reduced output:** Default max tokens reduced from 1024 to 512

### Breaking Changes in v2.3.0+

* **Document Reranking:** Added reranking functionality with the Qwen3-Reranker-4B model
* **Reranking API:** New `steadytext.rerank()` function and `st rerank` CLI command

### Other Notable Changes

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
st daemon start --host 127.0.0.1 --port 5678  # Custom host/port
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

### Model Management

```bash
# List available models
st models list

# Download models
st models download --size small
st models download --model gemma-3n-4b
st models download --all

# Delete models
st models delete --size small
st models delete --model gemma-3n-4b
st models delete --all --force

# Preload models
st models preload
```

### Other Operations

```bash
# Get embeddings
echo "machine learning" | st embed

# Document reranking (v2.3.0+)
st rerank "what is Python?" document1.txt document2.txt document3.txt
st rerank "search query" --file documents.txt --top-k 5 --json

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
# Text generation (uses daemon by default)
steadytext.generate(prompt: str, seed: int = 42) -> str
steadytext.generate(prompt, return_logprobs=True, seed: int = 42)


# Streaming generation
steadytext.generate_iter(prompt: str, seed: int = 42)

# Embeddings (uses daemon by default)
steadytext.embed(text: str | List[str], seed: int = 42) -> np.ndarray

# Document reranking (v2.3.0+)
steadytext.rerank(
    query: str,
    documents: Union[str, List[str]],
    task: str = "Given a web search query, retrieve relevant passages that answer the query",
    return_scores: bool = True,
    seed: int = 42
) -> Union[List[Tuple[str, float]], List[str]]

# Daemon management
from steadytext.daemon import use_daemon
with use_daemon():  # Ensure daemon connection
    text = steadytext.generate("Hello")

# Model preloading
steadytext.preload_models(verbose=True)

# Cache management
from steadytext import get_cache_manager
cache_manager = get_cache_manager()
stats = cache_manager.get_cache_stats()
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
export STEADYTEXT_DAEMON_PORT=5678
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
* **Models:** MIT (Qwen3)

---
## 📈 What's New

### Structured Generation (v2.4.1+)
- **Native llama.cpp grammar support** for JSON, regex, and choice constraints
- **PostgreSQL extension integration** - all structured generation features in SQL
- **Async structured generation** functions for high-performance applications

### PostgreSQL Extension (v1.1.0+)
- **Production-ready SQL functions** for text generation and embeddings
- **Async operations** with queue-based background processing
- **AI summarization** aggregate functions with TimescaleDB support
- **Structured generation** in SQL (JSON schemas, regex patterns, choices)
- **Docker support** for easy deployment

### Document Reranking (v2.3.0+)
- **Reranking support** using Qwen3-Reranker-4B model for query-document relevance scoring
- **Python API** - `steadytext.rerank()` function with customizable task descriptions
- **CLI command** - `st rerank` for command-line reranking operations
- **PostgreSQL functions** - SQL functions for reranking with async support (PostgreSQL extension v1.3.0+)
- **Fallback scoring** - simple word overlap when model unavailable
- **Dedicated cache** - separate frecency cache for reranking results

### Daemon Architecture (v1.2.0+)
- **Persistent model serving** with ZeroMQ for 10-100x faster repeated calls
- **Automatic fallback** to direct model loading when daemon unavailable
- **Zero configuration** - daemon starts automatically on first use
- **Background operation** - daemon runs silently in the background

### Centralized Cache System
- **Unified caching** - consistent behavior between daemon and direct access
- **Thread-safe SQLite backend** for reliable concurrent access
- **Shared cache files** across all access modes
- **Cache integration** with daemon server for optimal performance

### Improved CLI Experience
- **Streaming by default** - see output as it's generated
- **Quiet by default** - clean output without informational messages
- **New pipe syntax** - `echo "prompt" | st` for better unix integration
- **Daemon management** - built-in commands for daemon lifecycle

---

## 🔧 Troubleshooting

### Installation Issues

#### llama-cpp-python Build Errors

If you encounter build errors related to llama-cpp-python, especially with the error "Failed to load model", this is likely due to the package requiring the inference-sh fork with specific CMAKE flags:

```bash
# Set required environment variables before installation
export FORCE_CMAKE=1
export CMAKE_ARGS="-DLLAVA_BUILD=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF"

# Then install
pip install steadytext

# Or install from source
git clone https://github.com/julep-ai/steadytext.git
cd steadytext
uv sync  # or pip install -e .
```

#### Model Loading Issues

If you see "Failed to load model from file" errors:

1. **Try fallback models**: Set `STEADYTEXT_USE_FALLBACK_MODEL=true`
2. **Clear model cache**: `rm -rf ~/.cache/steadytext/models/`
3. **Check disk space**: Models require ~2-4GB per model

### Common Issues

- **"No module named 'llama_cpp'"**: Reinstall with the CMAKE flags above
- **Daemon connection refused**: Check if daemon is running with `st daemon status`
- **Slow first run**: Models download on first use (~2-4GB)

---

Built with ❤️ for developers tired of flaky AI tests.

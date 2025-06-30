# SteadyText API Documentation

This document provides detailed API documentation for SteadyText.

## Core Functions

### Text Generation

#### `steadytext.generate()`

```python
def generate(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    return_logprobs: bool = False,
    eos_string: str = "[EOS]",
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None,
    seed: int = DEFAULT_SEED
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]
```

Generate deterministic text from a prompt.

**Parameters:**
- `prompt` (str): The input text to generate from
- `max_new_tokens` (int, optional): Maximum number of tokens to generate (default: 512)
- `return_logprobs` (bool): If True, returns log probabilities along with the text
- `eos_string` (str): Custom end-of-sequence string to stop generation. Use "[EOS]" for model's default stop tokens
- `model` (str, optional): Model name from built-in registry (deprecated - use `size` parameter instead)
- `model_repo` (str, optional): Custom Hugging Face repository ID (e.g., "ggml-org/gemma-3n-E2B-it-GGUF")
- `model_filename` (str, optional): Custom model filename (e.g., "gemma-3n-E2B-it-Q8_0.gguf")
- `size` (str, optional): Size shortcut for Gemma-3n models: "small" (2B, default), or "large" (4B) - **recommended approach**
- `seed` (int): Random seed for deterministic generation (default: 42)

**Returns:**
- If `return_logprobs=False`: A string containing the generated text
- If `return_logprobs=True`: A tuple of (text, logprobs_dict)

**Example:**
```python
# Simple generation
text = steadytext.generate("Write a Python function")

# With custom seed for reproducible results
text1 = steadytext.generate("Write a story", seed=123)
text2 = steadytext.generate("Write a story", seed=123)  # Same result as text1
text3 = steadytext.generate("Write a story", seed=456)  # Different result

# With log probabilities
text, logprobs = steadytext.generate("Explain AI", return_logprobs=True)

# With custom stop string and seed
text = steadytext.generate("List items until END", eos_string="END", seed=789)

# Limit output length
text = steadytext.generate("Quick summary", max_new_tokens=100)

# Using size parameter (recommended)
text = steadytext.generate("Quick task", size="small")   # Uses Gemma-3n-2B
text = steadytext.generate("Complex task", size="large")  # Uses Gemma-3n-4B

# Using a custom model with seed
text = steadytext.generate(
    "Write code",
    model_repo="ggml-org/gemma-3n-E4B-it-GGUF",
    model_filename="gemma-3n-E4B-it-Q8_0.gguf",
    seed=999
)
```

#### `steadytext.generate_iter()`

```python
def generate_iter(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    eos_string: str = "[EOS]",
    include_logprobs: bool = False,
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None,
    seed: int = DEFAULT_SEED
) -> Iterator[Union[str, Tuple[str, Optional[Dict[str, Any]]]]]
```

Generate text iteratively, yielding tokens as they are produced.

**Parameters:**
- `prompt` (str): The input text to generate from
- `max_new_tokens` (int, optional): Maximum number of tokens to generate (default: 512)
- `eos_string` (str): Custom end-of-sequence string to stop generation. Use "[EOS]" for model's default stop tokens
- `include_logprobs` (bool): If True, yields tuples of (token, logprobs) instead of just tokens
- `model` (str, optional): Model name from built-in registry (deprecated - use `size` parameter instead)
- `model_repo` (str, optional): Custom Hugging Face repository ID
- `model_filename` (str, optional): Custom model filename
- `size` (str, optional): Size shortcut for Gemma-3n models: "small" (2B, default), or "large" (4B) - **recommended approach**
- `seed` (int): Random seed for deterministic generation (default: 42)

**Yields:**
- str: Text tokens/words as they are generated (if `include_logprobs=False`)
- Tuple[str, Optional[Dict[str, Any]]]: (token, logprobs) tuples (if `include_logprobs=True`)

**Example:**
```python
# Simple streaming
for token in steadytext.generate_iter("Tell me a story"):
    print(token, end="", flush=True)

# With custom seed for reproducible streaming
for token in steadytext.generate_iter("Tell me a story", seed=123):
    print(token, end="", flush=True)

# With custom stop string and seed
for token in steadytext.generate_iter("Generate until STOP", eos_string="STOP", seed=456):
    print(token, end="", flush=True)

# With log probabilities
for token, logprobs in steadytext.generate_iter("Explain AI", include_logprobs=True):
    print(token, end="", flush=True)

# Stream with size parameter and custom length
for token in steadytext.generate_iter("Quick response", size="small", max_new_tokens=50):
    print(token, end="", flush=True)

for token in steadytext.generate_iter("Complex task", size="large", seed=789):
    print(token, end="", flush=True)
```

### Embeddings

#### `steadytext.embed()`

```python
def embed(text_input: Union[str, List[str]], seed: int = DEFAULT_SEED) -> np.ndarray
```

Create deterministic embeddings for text input.

**Parameters:**
- `text_input` (Union[str, List[str]]): A string or list of strings to embed
- `seed` (int): Random seed for deterministic embedding generation (default: 42)

**Returns:**
- np.ndarray: A 1024-dimensional L2-normalized float32 numpy array

**Example:**
```python
# Single string
vec = steadytext.embed("Hello world")

# With custom seed for reproducible embeddings
vec1 = steadytext.embed("Hello world", seed=123)
vec2 = steadytext.embed("Hello world", seed=123)  # Same result as vec1
vec3 = steadytext.embed("Hello world", seed=456)  # Different result

# Multiple strings (averaged)
vec = steadytext.embed(["Hello", "world"])

# Multiple strings with custom seed
vec = steadytext.embed(["Hello", "world"], seed=789)
```

### Utility Functions

#### `steadytext.preload_models()`

```python
def preload_models(verbose: bool = False) -> None
```

Preload models before first use to avoid delays.

**Parameters:**
- `verbose` (bool): If True, prints progress information

**Example:**
```python
# Silent preloading
steadytext.preload_models()

# Verbose preloading
steadytext.preload_models(verbose=True)
```

#### `steadytext.get_model_cache_dir()`

```python
def get_model_cache_dir() -> str
```

Get the path to the model cache directory.

**Returns:**
- str: The absolute path to the model cache directory

**Example:**
```python
cache_dir = steadytext.get_model_cache_dir()
print(f"Models are stored in: {cache_dir}")
```

## Constants

### `steadytext.DEFAULT_SEED`
- **Type:** int
- **Value:** 42
- **Description:** The default random seed used for deterministic generation. Can be overridden by the `seed` parameter in generation and embedding functions.

### `steadytext.GENERATION_MAX_NEW_TOKENS`
- **Type:** int
- **Value:** 512
- **Description:** Maximum number of tokens to generate

### `steadytext.EMBEDDING_DIMENSION`
- **Type:** int
- **Value:** 1024
- **Description:** The dimensionality of embedding vectors

## Environment Variables

### Generation Cache

- **`STEADYTEXT_GENERATION_CACHE_CAPACITY`**: Maximum number of cache entries (default: 256)
- **`STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB`**: Maximum cache file size in MB (default: 50.0)

### Embedding Cache

- **`STEADYTEXT_EMBEDDING_CACHE_CAPACITY`**: Maximum number of cache entries (default: 512)
- **`STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB`**: Maximum cache file size in MB (default: 100.0)

### Model Downloads

- **`STEADYTEXT_ALLOW_MODEL_DOWNLOADS`**: Set to "true" to allow automatic model downloads (mainly used for testing)

## Model Switching (v2.0.0+)

SteadyText v2.0.0+ supports model switching with the Gemma-3n model family, allowing you to use different model sizes for different tasks.

### Current Model Registry (v2.0.0+)

The following models are available:

| Size Parameter | Model Name | Parameters | Use Case |
|----------------|------------|------------|----------|
| `small` | `gemma-3n-2b` | 2B | Default, fast tasks |
| `large` | `gemma-3n-4b` | 4B | High quality, complex tasks |

### Model Selection Methods

1. **Using size parameter (recommended)**: `generate("prompt", size="large")`
2. **Custom models**: `generate("prompt", model_repo="...", model_filename="...")`
3. **Environment variables**: Set `STEADYTEXT_DEFAULT_SIZE` or custom model variables

### Deprecated Models (v1.x)

> **Note:** The following models were available in SteadyText v1.x but are deprecated in v2.0.0+:
> - `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`
> - `qwen2.5-0.5b`, `qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b`
>
> Use the `size` parameter with Gemma-3n models instead.

### Model Caching

- Models are cached after first load for efficient switching
- Multiple models can be loaded simultaneously
- Use `clear_model_cache()` to free memory if needed

## Error Handling

All functions are designed to never raise exceptions during normal operation. If models cannot be loaded, deterministic fallback functions are used:

- **Text generation fallback**: Uses hash-based word selection to generate pseudo-random but deterministic text
- **Embedding fallback**: Returns zero vectors of the correct dimension

This ensures that your code never breaks, even in environments where models cannot be downloaded or loaded.

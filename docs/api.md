# SteadyText API Documentation

This document provides detailed API documentation for SteadyText.

## Core Functions

### Text Generation

#### `steadytext.generate()`

```python
def generate(
    prompt: str,
    return_logprobs: bool = False,
    eos_string: str = "[EOS]",
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]
```

Generate deterministic text from a prompt.

**Parameters:**
- `prompt` (str): The input text to generate from
- `return_logprobs` (bool): If True, returns log probabilities along with the text
- `eos_string` (str): Custom end-of-sequence string to stop generation. Use "[EOS]" for model's default stop tokens
- `model` (str, optional): Model name from built-in registry (e.g., "qwen2.5-3b", "qwen3-8b")
- `model_repo` (str, optional): Custom Hugging Face repository ID (e.g., "Qwen/Qwen2.5-3B-Instruct-GGUF")
- `model_filename` (str, optional): Custom model filename (e.g., "qwen2.5-3b-instruct-q8_0.gguf")
- `size` (str, optional): Size shortcut for Qwen3 models: "small" (0.6B), "medium" (1.7B, default), or "large" (4B)

**Returns:**
- If `return_logprobs=False`: A string containing the generated text
- If `return_logprobs=True`: A tuple of (text, logprobs_dict)

**Example:**
```python
# Simple generation
text = steadytext.generate("Write a Python function")

# With log probabilities
text, logprobs = steadytext.generate("Explain AI", return_logprobs=True)

# With custom stop string
text = steadytext.generate("List items until END", eos_string="END")

# Using a model from the registry
text = steadytext.generate("Complex analysis", model="qwen2.5-7b")

# Using size parameter
text = steadytext.generate("Quick task", size="small")  # Uses Qwen3-0.6B
text = steadytext.generate("Complex task", size="large")  # Uses Qwen3-4B

# Using a custom model
text = steadytext.generate(
    "Write code",
    model_repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
    model_filename="qwen2.5-3b-instruct-q8_0.gguf"
)
```

#### `steadytext.generate_iter()`

```python
def generate_iter(
    prompt: str,
    eos_string: str = "[EOS]",
    include_logprobs: bool = False,
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None
) -> Iterator[Union[str, Tuple[str, Optional[Dict[str, Any]]]]]
```

Generate text iteratively, yielding tokens as they are produced.

**Parameters:**
- `prompt` (str): The input text to generate from
- `eos_string` (str): Custom end-of-sequence string to stop generation. Use "[EOS]" for model's default stop tokens
- `include_logprobs` (bool): If True, yields tuples of (token, logprobs) instead of just tokens
- `model` (str, optional): Model name from built-in registry (e.g., "qwen2.5-3b")
- `model_repo` (str, optional): Custom Hugging Face repository ID
- `model_filename` (str, optional): Custom model filename
- `size` (str, optional): Size shortcut for Qwen3 models: "small" (0.6B), "medium" (1.7B, default), or "large" (4B)

**Yields:**
- str: Text tokens/words as they are generated (if `include_logprobs=False`)
- Tuple[str, Optional[Dict[str, Any]]]: (token, logprobs) tuples (if `include_logprobs=True`)

**Example:**
```python
# Simple streaming
for token in steadytext.generate_iter("Tell me a story"):
    print(token, end="", flush=True)

# With custom stop string
for token in steadytext.generate_iter("Generate until STOP", eos_string="STOP"):
    print(token, end="", flush=True)

# With log probabilities
for token, logprobs in steadytext.generate_iter("Explain AI", include_logprobs=True):
    print(token, end="", flush=True)

# Stream with a different model
for token in steadytext.generate_iter("Complex task", model="qwen2.5-7b"):
    print(token, end="", flush=True)

# Stream with size parameter
for token in steadytext.generate_iter("Quick response", size="small"):
    print(token, end="", flush=True)
```

### Embeddings

#### `steadytext.embed()`

```python
def embed(text_input: Union[str, List[str]]) -> np.ndarray
```

Create deterministic embeddings for text input.

**Parameters:**
- `text_input` (Union[str, List[str]]): A string or list of strings to embed

**Returns:**
- np.ndarray: A 1024-dimensional L2-normalized float32 numpy array

**Example:**
```python
# Single string
vec = steadytext.embed("Hello world")

# Multiple strings (averaged)
vec = steadytext.embed(["Hello", "world"])
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
- **Description:** The fixed random seed used for deterministic generation

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

## Model Switching (v1.0.0+)

SteadyText supports dynamic model switching, allowing you to use different models for different tasks without restarting your application.

### Built-in Model Registry

The following models are available in the registry:

| Model Name | Parameters | Use Case |
|------------|------------|----------|
| `qwen3-1.7b` | 1.7B | Default model, balanced performance |
| `qwen3-4b` | 4B | Better quality, moderate speed |
| `qwen3-8b` | 8B | High quality, resource intensive |
| `qwen2.5-0.5b` | 0.5B | Fast, lightweight tasks |
| `qwen2.5-1.5b` | 1.5B | Good balance of speed/quality |
| `qwen2.5-3b` | 3B | Enhanced capabilities |
| `qwen2.5-7b` | 7B | Best quality, slower |

### Model Selection Methods

1. **Using the registry**: `generate("prompt", model="qwen2.5-3b")`
2. **Custom models**: `generate("prompt", model_repo="...", model_filename="...")`
3. **Environment variables**: Set `STEADYTEXT_GENERATION_MODEL_REPO` and `STEADYTEXT_GENERATION_MODEL_FILENAME`

### Model Caching

- Models are cached after first load for efficient switching
- Multiple models can be loaded simultaneously
- Use `clear_model_cache()` to free memory if needed

## Error Handling

All functions are designed to never raise exceptions during normal operation. If models cannot be loaded, deterministic fallback functions are used:

- **Text generation fallback**: Uses hash-based word selection to generate pseudo-random but deterministic text
- **Embedding fallback**: Returns zero vectors of the correct dimension

This ensures that your code never breaks, even in environments where models cannot be downloaded or loaded.
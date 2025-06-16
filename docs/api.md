# SteadyText API Documentation

This document provides detailed API documentation for SteadyText.

## Core Functions

### Text Generation

#### `steadytext.generate()`

```python
def generate(
    prompt: str,
    return_logprobs: bool = False
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]
```

Generate deterministic text from a prompt.

**Parameters:**
- `prompt` (str): The input text to generate from
- `return_logprobs` (bool): If True, returns log probabilities along with the text

**Returns:**
- If `return_logprobs=False`: A string containing the generated text
- If `return_logprobs=True`: A tuple of (text, logprobs_dict)

**Example:**
```python
# Simple generation
text = steadytext.generate("Write a Python function")

# With log probabilities
text, logprobs = steadytext.generate("Explain AI", return_logprobs=True)
```

#### `steadytext.generate_iter()`

```python
def generate_iter(prompt: str) -> Iterator[str]
```

Generate text iteratively, yielding tokens as they are produced.

**Parameters:**
- `prompt` (str): The input text to generate from

**Yields:**
- str: Text tokens/words as they are generated

**Example:**
```python
for token in steadytext.generate_iter("Tell me a story"):
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

## Error Handling

All functions are designed to never raise exceptions during normal operation. If models cannot be loaded, deterministic fallback functions are used:

- **Text generation fallback**: Uses hash-based word selection to generate pseudo-random but deterministic text
- **Embedding fallback**: Returns zero vectors of the correct dimension

This ensures that your code never breaks, even in environments where models cannot be downloaded or loaded.
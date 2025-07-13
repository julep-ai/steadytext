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
    seed: int = DEFAULT_SEED,
    schema: Optional[Union[Dict[str, Any], type, object]] = None,
    regex: Optional[str] = None,
    choices: Optional[List[str]] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]
```

Generate deterministic text from a prompt, with optional structured output.

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
- `schema` (Union[Dict, type, object], optional): JSON schema, Pydantic model, or Python type for structured JSON output.
- `regex` (str, optional): A regular expression to constrain the output.
- `choices` (List[str], optional): A list of strings to choose from.
- `response_format` (Dict, optional): A dictionary specifying the output format (e.g., `{"type": "json_object"}`).

**Returns:**
- If `return_logprobs=False`: A string containing the generated text. For structured JSON output, the JSON is wrapped in `<json-output>` tags.
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

# Structured generation with a regex pattern
phone_number = steadytext.generate("My phone number is: ", regex=r"\d{3}-\d{3}-\d{4}")

# Structured generation with choices
mood = steadytext.generate("I feel", choices=["happy", "sad", "angry"])

# Structured generation with a JSON schema
from pydantic import BaseModel
class User(BaseModel):
    name: str
    age: int

user_json = steadytext.generate("Create a user named John, age 30", schema=User)
# user_json will contain: '... <json-output>{"name": "John", "age": 30}</json-output>'
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

### Structured Generation (v2.4.1+)

These are convenience functions for structured generation using llama.cpp's native grammar support.

#### `steadytext.generate_json()`

```python
def generate_json(
    prompt: str,
    schema: Union[Dict[str, Any], type, object],
    max_tokens: int = 512,
    **kwargs
) -> str
```

Generates a JSON string that conforms to the provided schema.

#### `steadytext.generate_regex()`

```python
def generate_regex(
    prompt: str,
    pattern: str,
    max_tokens: int = 512,
    **kwargs
) -> str
```

Generates a string that matches the given regular expression.

#### `steadytext.generate_choice()`

```python
def generate_choice(
    prompt: str,
    choices: List[str],
    max_tokens: int = 512,
    **kwargs
) -> str
```

Generates a string that is one of the provided choices.

#### `steadytext.generate_format()`

```python
def generate_format(
    prompt: str,
    format_type: type,
    max_tokens: int = 512,
    **kwargs
) -> str
```

Generates a string that conforms to a basic Python type (e.g., `int`, `float`, `bool`).

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

# Multiple strings (returns a single, averaged embedding)
vec = steadytext.embed(["Hello", "world"])

# Multiple strings with custom seed
vec = steadytext.embed(["Hello", "world"], seed=789)
```

### Document Reranking (v2.3.0+)

#### `steadytext.rerank()`

```python
def rerank(
    query: str,
    documents: Union[str, List[str]],
    task: str = "Given a web search query, retrieve relevant passages that answer the query",
    return_scores: bool = True,
    seed: int = DEFAULT_SEED
) -> Union[List[Tuple[str, float]], List[str]]
```

Rerank documents based on their relevance to a query using the Qwen3-Reranker-4B model.

**Parameters:**
- `query` (str): The search query to rerank documents against
- `documents` (Union[str, List[str]]): Single document or list of documents to rerank
- `task` (str): Description of the reranking task for better results (default: "Given a web search query, retrieve relevant passages that answer the query")
- `return_scores` (bool): If True, return (document, score) tuples; if False, just documents (default: True)
- `seed` (int): Random seed for deterministic reranking (default: 42)

**Returns:**
- If `return_scores=True`: List[Tuple[str, float]] - List of (document, score) tuples sorted by relevance (highest score first)
- If `return_scores=False`: List[str] - List of documents sorted by relevance (highest score first)

**Example:**
```python
# Basic reranking
documents = [
    "Python is a programming language",
    "Cats are cute animals",
    "Python snakes are found in Asia"
]
results = steadytext.rerank("Python programming", documents)
# Returns documents sorted by relevance to "Python programming"

# With custom task description
results = steadytext.rerank(
    "customer support issue",
    support_tickets,
    task="support ticket prioritization",
    seed=123
)

# Domain-specific reranking
legal_results = steadytext.rerank(
    "contract breach",
    legal_documents,
    task="legal document retrieval for case research"
)

# Get just documents without scores
sorted_docs = steadytext.rerank(
    "machine learning",
    documents,
    return_scores=False
)
# Returns: ["ML document 1", "ML document 2", ...]
```

**Notes:**
- Uses yes/no token logits for binary relevance scoring
- Falls back to simple word overlap scoring when model is unavailable
- Results are cached for identical query-document pairs
- Task descriptions help the model understand the reranking context

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

### Reranking Cache (v2.3.0+)

- **`STEADYTEXT_RERANKING_CACHE_CAPACITY`**: Maximum number of cache entries (default: 256)
- **`STEADYTEXT_RERANKING_CACHE_MAX_SIZE_MB`**: Maximum cache file size in MB (default: 25.0)

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

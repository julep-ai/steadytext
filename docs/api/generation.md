# Text Generation API

Functions for deterministic text generation.

## generate()

Generate deterministic text from a prompt.

```python
def generate(
    prompt: str,
    return_logprobs: bool = False,
    eos_string: str = "[EOS]"
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Input text to generate from |
| `return_logprobs` | `bool` | `False` | Return log probabilities with text |
| `eos_string` | `str` | `"[EOS]"` | Custom end-of-sequence string |

### Returns

=== "Basic Usage"
    **Returns**: `str` - Generated text (512 tokens max)

=== "With Log Probabilities" 
    **Returns**: `Tuple[str, Optional[Dict]]` - Generated text and log probabilities

### Examples

=== "Simple Generation"

    ```python
    import steadytext

    text = steadytext.generate("Write a Python function")
    print(text)
    # Always returns the same 512-token completion
    ```

=== "With Log Probabilities"

    ```python
    text, logprobs = steadytext.generate(
        "Explain machine learning", 
        return_logprobs=True
    )
    
    print("Generated text:", text)
    print("Log probabilities:", logprobs)
    ```

=== "Custom Stop String"

    ```python
    # Stop generation at custom string
    text = steadytext.generate(
        "List programming languages until STOP",
        eos_string="STOP"
    )
    print(text)
    ```

---

## generate_iter()

Generate text iteratively, yielding tokens as produced.

```python
def generate_iter(
    prompt: str,
    eos_string: str = "[EOS]",
    include_logprobs: bool = False
) -> Iterator[Union[str, Tuple[str, Optional[Dict[str, Any]]]]]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Input text to generate from |
| `eos_string` | `str` | `"[EOS]"` | Custom end-of-sequence string |
| `include_logprobs` | `bool` | `False` | Yield log probabilities with tokens |

### Returns

=== "Basic Streaming"
    **Yields**: `str` - Individual tokens/words

=== "With Log Probabilities"
    **Yields**: `Tuple[str, Optional[Dict]]` - Token and log probabilities

### Examples

=== "Basic Streaming"

    ```python
    import steadytext

    for token in steadytext.generate_iter("Tell me a story"):
        print(token, end="", flush=True)
    ```

=== "With Progress Tracking"

    ```python
    prompt = "Explain quantum computing"
    tokens = []
    
    for token in steadytext.generate_iter(prompt):
        tokens.append(token)
        print(f"Generated {len(tokens)} tokens", end="\r")
        
    print(f"\nComplete! Generated {len(tokens)} tokens")
    print("Full text:", "".join(tokens))
    ```

=== "Custom Stop String"

    ```python
    for token in steadytext.generate_iter(
        "Count from 1 to 10 then say DONE", 
        eos_string="DONE"
    ):
        print(token, end="", flush=True)
    ```

=== "With Log Probabilities"

    ```python
    for token, logprobs in steadytext.generate_iter(
        "Explain AI", 
        include_logprobs=True
    ):
        confidence = logprobs.get('confidence', 0) if logprobs else 0
        print(f"{token} (confidence: {confidence:.2f})", end="")
    ```

---

## Advanced Usage

### Deterministic Behavior

Both functions return identical results for identical inputs:

```python
# These will always be identical
result1 = steadytext.generate("hello world")
result2 = steadytext.generate("hello world") 
assert result1 == result2  # Always passes!

# Streaming produces same tokens in same order
tokens1 = list(steadytext.generate_iter("hello world"))
tokens2 = list(steadytext.generate_iter("hello world"))
assert tokens1 == tokens2  # Always passes!
```

### Caching

Results are automatically cached using a frecency cache (LRU + frequency):

```python
# First call: generates and caches result
text1 = steadytext.generate("common prompt")  # ~2 seconds

# Second call: returns cached result  
text2 = steadytext.generate("common prompt")  # ~0.1 seconds

assert text1 == text2  # Same result, much faster
```

### Fallback Behavior

When models can't be loaded, deterministic fallbacks are used:

```python
# Even without models, these always return the same results
text = steadytext.generate("test prompt")  # Hash-based fallback
assert len(text) > 0  # Always has content

# Fallback is also deterministic
text1 = steadytext.generate("fallback test")
text2 = steadytext.generate("fallback test") 
assert text1 == text2  # Same fallback result
```

### Performance Tips

!!! tip "Optimization Strategies"
    - **Preload models**: Call `steadytext.preload_models()` at startup
    - **Batch processing**: Use `generate()` for multiple prompts rather than streaming individual tokens
    - **Cache warmup**: Pre-generate common prompts to populate cache
    - **Memory management**: Models stay loaded once initialized (singleton pattern)
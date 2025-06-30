# Text Generation API

Functions for deterministic text generation.

## generate()

Generate deterministic text from a prompt.

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

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Input text to generate from |
| `max_new_tokens` | `int` | `512` | Maximum number of tokens to generate |
| `return_logprobs` | `bool` | `False` | Return log probabilities with text |
| `eos_string` | `str` | `"[EOS]"` | Custom end-of-sequence string |
| `model` | `str` | `None` | Model name from registry (deprecated) |
| `model_repo` | `str` | `None` | Custom Hugging Face repository ID |
| `model_filename` | `str` | `None` | Custom model filename |
| `size` | `str` | `None` | Size shortcut: "small" or "large" |
| `seed` | `int` | `42` | Random seed for deterministic generation |

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

=== "Custom Seed"

    ```python
    # Generate with different seeds for variation
    text1 = steadytext.generate("Write a story", seed=123)
    text2 = steadytext.generate("Write a story", seed=123)  # Same as text1
    text3 = steadytext.generate("Write a story", seed=456)  # Different result
    
    print(f"Seed 123: {text1[:50]}...")
    print(f"Seed 456: {text3[:50]}...")
    ```

=== "Custom Length"

    ```python
    # Generate shorter responses
    short_text = steadytext.generate("Explain AI", max_new_tokens=50)
    long_text = steadytext.generate("Explain AI", max_new_tokens=200)
    
    print(f"Short ({len(short_text.split())} words): {short_text}")
    print(f"Long ({len(long_text.split())} words): {long_text}")
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

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Input text to generate from |
| `max_new_tokens` | `int` | `512` | Maximum number of tokens to generate |
| `eos_string` | `str` | `"[EOS]"` | Custom end-of-sequence string |
| `include_logprobs` | `bool` | `False` | Yield log probabilities with tokens |
| `model` | `str` | `None` | Model name from registry (deprecated) |
| `model_repo` | `str` | `None` | Custom Hugging Face repository ID |
| `model_filename` | `str` | `None` | Custom model filename |
| `size` | `str` | `None` | Size shortcut: "small" or "large" |
| `seed` | `int` | `42` | Random seed for deterministic generation |

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

=== "Custom Seed Streaming"

    ```python
    # Reproducible streaming with custom seeds
    print("Stream 1 (seed=123):")
    for token in steadytext.generate_iter("Tell me a joke", seed=123):
        print(token, end="", flush=True)
    
    print("\n\nStream 2 (seed=123 - same result):")
    for token in steadytext.generate_iter("Tell me a joke", seed=123):
        print(token, end="", flush=True)
    
    print("\n\nStream 3 (seed=456 - different result):")
    for token in steadytext.generate_iter("Tell me a joke", seed=456):
        print(token, end="", flush=True)
    ```

=== "Controlled Length Streaming"

    ```python
    # Stream with limited tokens
    token_count = 0
    for token in steadytext.generate_iter("Explain quantum physics", max_new_tokens=30):
        print(token, end="", flush=True)
        token_count += 1
    print(f"\nGenerated {token_count} tokens")
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

Both functions return identical results for identical inputs and seeds:

```python
# Default seed (42) - always identical
result1 = steadytext.generate("hello world")
result2 = steadytext.generate("hello world") 
assert result1 == result2  # Always passes!

# Custom seeds - identical for same seed
result1 = steadytext.generate("hello world", seed=123)
result2 = steadytext.generate("hello world", seed=123)
assert result1 == result2  # Always passes!

# Different seeds produce different results
result1 = steadytext.generate("hello world", seed=123)
result2 = steadytext.generate("hello world", seed=456)
assert result1 != result2  # Different seeds, different results

# Streaming produces same tokens in same order for same seed
tokens1 = list(steadytext.generate_iter("hello world", seed=789))
tokens2 = list(steadytext.generate_iter("hello world", seed=789))
assert tokens1 == tokens2  # Always passes!
```

### Custom Seed Use Cases

```python
# Experimental variations - try different seeds for the same prompt
baseline = steadytext.generate("Write a haiku about programming", seed=42)
variation1 = steadytext.generate("Write a haiku about programming", seed=123)
variation2 = steadytext.generate("Write a haiku about programming", seed=456)

print("Baseline:", baseline)
print("Variation 1:", variation1)
print("Variation 2:", variation2)

# A/B testing - consistent results for testing
test_prompt = "Explain machine learning to a beginner"
version_a = steadytext.generate(test_prompt, seed=100)  # Version A
version_b = steadytext.generate(test_prompt, seed=200)  # Version B

# Reproducible research - document your seeds
research_seed = 42
results = []
for prompt in research_prompts:
    result = steadytext.generate(prompt, seed=research_seed)
    results.append((prompt, result))
    research_seed += 1  # Increment for each prompt
```

### Caching

Results are automatically cached using a frecency cache (LRU + frequency), with seed as part of the cache key:

```python
# First call: generates and caches result for default seed
text1 = steadytext.generate("common prompt")  # ~2 seconds

# Second call with same seed: returns cached result  
text2 = steadytext.generate("common prompt")  # ~0.1 seconds
assert text1 == text2  # Same result, much faster

# Different seed: generates new result and caches separately
text3 = steadytext.generate("common prompt", seed=123)  # ~2 seconds (new cache entry)
text4 = steadytext.generate("common prompt", seed=123)  # ~0.1 seconds (cached)

assert text3 == text4  # Same seed, same cached result
assert text1 != text3  # Different seeds, different results

# Cache keys include seed, so each seed gets its own cache entry
for seed in [100, 200, 300]:
    steadytext.generate("warm up cache", seed=seed)  # Each gets cached separately
```

### Fallback Behavior

When models can't be loaded, deterministic fallbacks are used with seed support:

```python
# Even without models, these return deterministic results based on seed
text1 = steadytext.generate("test prompt", seed=42)  # Hash-based fallback
text2 = steadytext.generate("test prompt", seed=42)  # Same result
text3 = steadytext.generate("test prompt", seed=123) # Different result

assert len(text1) > 0  # Always has content
assert text1 == text2  # Same seed, same fallback
assert text1 != text3  # Different seed, different fallback

# Fallback respects custom seeds for variation
fallback_texts = []
for seed in [100, 200, 300]:
    text = steadytext.generate("fallback test", seed=seed)
    fallback_texts.append(text)

# All different due to different seeds
assert len(set(fallback_texts)) == 3
```

### Performance Tips

!!! tip "Optimization Strategies"
    - **Preload models**: Call `steadytext.preload_models()` at startup
    - **Batch processing**: Use `generate()` for multiple prompts rather than streaming individual tokens
    - **Cache warmup**: Pre-generate common prompts to populate cache
    - **Memory management**: Models stay loaded once initialized (singleton pattern)
    - **Seed management**: Use consistent seeds for reproducible results, different seeds for variation
    - **Length control**: Use `max_new_tokens` to control response length and generation time
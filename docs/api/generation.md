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
    seed: int = DEFAULT_SEED,
    schema: Optional[Union[Dict[str, Any], type, object]] = None,
    regex: Optional[str] = None,
    choices: Optional[List[str]] = None,
    response_format: Optional[Dict[str, Any]] = None,
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

---

## Error Handling and Edge Cases

### Handling Invalid Inputs

```python
import steadytext

# Empty prompt handling
empty_result = steadytext.generate("")
print(f"Empty prompt result: {empty_result[:50]}...")  # Still generates deterministic output

# Very long prompt handling (truncated to model's context window)
long_prompt = "Explain " * 1000 + "machine learning"
result = steadytext.generate(long_prompt)
print(f"Long prompt handled: {len(result)} chars generated")

# Special characters and Unicode
unicode_result = steadytext.generate("Write about 🤖 and 人工智能")
print(f"Unicode handled: {unicode_result[:100]}...")

# Newlines and formatting
multiline = steadytext.generate("""Write a function that:
1. Takes a list
2. Sorts it
3. Returns the result""")
print(f"Multiline prompt: {multiline[:100]}...")
```

### Memory-Efficient Streaming

```python
import sys

def stream_large_generation(prompt: str, max_chunks: int = 100):
    """Stream generation with memory tracking."""
    chunks = []
    total_tokens = 0
    
    for i, token in enumerate(steadytext.generate_iter(prompt)):
        chunks.append(token)
        total_tokens += 1
        
        # Process in batches to manage memory
        if len(chunks) >= max_chunks:
            # Process chunk (e.g., write to file)
            sys.stdout.write("".join(chunks))
            sys.stdout.flush()
            chunks = []
    
    # Process remaining
    if chunks:
        sys.stdout.write("".join(chunks))
    
    print(f"\nGenerated {total_tokens} tokens")

# Use for large generations
stream_large_generation("Write a comprehensive guide to Python programming")
```

### Concurrent Generation

```python
import concurrent.futures
import steadytext

def parallel_generation(prompts: list, max_workers: int = 4):
    """Generate text for multiple prompts in parallel."""
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_prompt = {
            executor.submit(steadytext.generate, prompt, seed=idx): (prompt, idx)
            for idx, prompt in enumerate(prompts)
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_prompt):
            prompt, idx = future_to_prompt[future]
            try:
                result = future.result()
                results[prompt] = result
                print(f"✓ Completed prompt {idx+1}: {prompt[:30]}...")
            except Exception as e:
                print(f"✗ Failed prompt {idx+1}: {e}")
                results[prompt] = None
    
    return results

# Example usage
prompts = [
    "Write a Python function for sorting",
    "Explain machine learning",
    "Create a REST API example",
    "Describe quantum computing"
]

results = parallel_generation(prompts)
for prompt, result in results.items():
    print(f"\n{prompt}:\n{result[:100]}...\n")
```

---

## Advanced Patterns

### Custom Generation Pipeline

```python
import steadytext
import re

class TextGenerator:
    """Custom text generation pipeline with preprocessing and postprocessing."""
    
    def __init__(self, default_seed: int = 42):
        self.default_seed = default_seed
        self.generation_count = 0
    
    def preprocess(self, prompt: str) -> str:
        """Clean and prepare prompt."""
        # Remove extra whitespace
        prompt = " ".join(prompt.split())
        
        # Add context if needed
        if not prompt.endswith((".", "?", "!", ":")):
            prompt += ":"
        
        return prompt
    
    def postprocess(self, text: str) -> str:
        """Clean generated text."""
        # Remove any [EOS] markers
        text = text.replace("[EOS]", "")
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with pre/post processing."""
        # Use incremental seeds for variety
        seed = kwargs.pop('seed', self.default_seed + self.generation_count)
        self.generation_count += 1
        
        # Process
        cleaned_prompt = self.preprocess(prompt)
        raw_output = steadytext.generate(cleaned_prompt, seed=seed, **kwargs)
        final_output = self.postprocess(raw_output)
        
        return final_output

# Usage
generator = TextGenerator()

# Generates different outputs due to incremental seeding
response1 = generator.generate("write a function")
response2 = generator.generate("write a function")  # Different seed
response3 = generator.generate("write a function", seed=100)  # Custom seed

print(f"Response 1: {response1[:50]}...")
print(f"Response 2: {response2[:50]}...")
print(f"Response 3: {response3[:50]}...")
```

### Template-Based Generation

```python
import steadytext
from typing import Dict, Any

class TemplateGenerator:
    """Generate text using templates with variable substitution."""
    
    def __init__(self):
        self.templates = {
            "function": "Write a Python function that {action} for {input_type} and returns {output_type}",
            "explanation": "Explain {concept} in simple terms for {audience}",
            "comparison": "Compare and contrast {item1} and {item2} in terms of {criteria}",
            "tutorial": "Create a step-by-step tutorial on {topic} for {skill_level} programmers"
        }
    
    def generate_from_template(self, template_name: str, variables: Dict[str, Any], 
                             seed: int = 42, **kwargs) -> str:
        """Generate text from a template with variables."""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        # Fill template
        template = self.templates[template_name]
        prompt = template.format(**variables)
        
        # Generate
        return steadytext.generate(prompt, seed=seed, **kwargs)
    
    def batch_generate(self, template_name: str, variable_sets: list, 
                      base_seed: int = 42) -> list:
        """Generate multiple outputs from the same template."""
        results = []
        
        for i, variables in enumerate(variable_sets):
            # Use different seed for each to ensure variety
            result = self.generate_from_template(
                template_name, 
                variables, 
                seed=base_seed + i
            )
            results.append({
                "variables": variables,
                "output": result
            })
        
        return results

# Usage examples
gen = TemplateGenerator()

# Single generation
function_code = gen.generate_from_template(
    "function",
    {
        "action": "calculates factorial",
        "input_type": "positive integer",
        "output_type": "integer"
    }
)
print(f"Generated function:\n{function_code[:200]}...\n")

# Batch generation with variations
tutorials = gen.batch_generate(
    "tutorial",
    [
        {"topic": "async programming", "skill_level": "beginner"},
        {"topic": "decorators", "skill_level": "intermediate"},
        {"topic": "metaclasses", "skill_level": "advanced"}
    ]
)

for tutorial in tutorials:
    print(f"\nTopic: {tutorial['variables']['topic']}")
    print(f"Output: {tutorial['output'][:150]}...")
```

### Context-Aware Generation

```python
import steadytext
from collections import deque

class ContextualGenerator:
    """Maintain context across multiple generations."""
    
    def __init__(self, context_window: int = 5):
        self.context = deque(maxlen=context_window)
        self.base_seed = 42
        self.generation_count = 0
    
    def add_context(self, text: str):
        """Add text to context history."""
        self.context.append(text)
    
    def generate_with_context(self, prompt: str, include_context: bool = True) -> str:
        """Generate text considering previous context."""
        if include_context and self.context:
            # Build context prompt
            context_str = "\n".join(f"Previous: {ctx}" for ctx in self.context)
            full_prompt = f"{context_str}\n\nNow: {prompt}"
        else:
            full_prompt = prompt
        
        # Generate with unique seed
        result = steadytext.generate(
            full_prompt, 
            seed=self.base_seed + self.generation_count
        )
        self.generation_count += 1
        
        # Add to context for next generation
        self.add_context(f"{prompt} -> {result[:100]}...")
        
        return result
    
    def clear_context(self):
        """Reset context history."""
        self.context.clear()
        self.generation_count = 0

# Example: Story continuation
story_gen = ContextualGenerator()

# Generate story parts with context
part1 = story_gen.generate_with_context("Once upon a time in a digital kingdom")
print(f"Part 1: {part1[:150]}...\n")

part2 = story_gen.generate_with_context("The hero discovered a mysterious artifact")
print(f"Part 2 (with context): {part2[:150]}...\n")

part3 = story_gen.generate_with_context("Suddenly, the artifact began to glow")
print(f"Part 3 (with context): {part3[:150]}...\n")

# Generate without context for comparison
story_gen.clear_context()
part3_no_context = story_gen.generate_with_context(
    "Suddenly, the artifact began to glow", 
    include_context=False
)
print(f"Part 3 (no context): {part3_no_context[:150]}...")
```

---

## Debugging and Monitoring

### Generation Analytics

```python
import steadytext
import time
from dataclasses import dataclass
from typing import List

@dataclass
class GenerationMetrics:
    prompt: str
    seed: int
    duration: float
    token_count: int
    cached: bool
    output_preview: str

class GenerationMonitor:
    """Monitor and analyze generation patterns."""
    
    def __init__(self):
        self.metrics: List[GenerationMetrics] = []
    
    def generate_with_metrics(self, prompt: str, seed: int = 42, **kwargs) -> str:
        """Generate text while collecting metrics."""
        start_time = time.time()
        
        # Check if likely cached (by doing a duplicate call)
        _ = steadytext.generate(prompt, seed=seed, **kwargs)
        check_time = time.time() - start_time
        
        # Actual generation
        start_time = time.time()
        result = steadytext.generate(prompt, seed=seed, **kwargs)
        duration = time.time() - start_time
        
        # Determine if it was cached
        cached = duration < check_time * 0.5  # Much faster = likely cached
        
        # Count tokens (approximate)
        token_count = len(result.split())
        
        # Store metrics
        metric = GenerationMetrics(
            prompt=prompt,
            seed=seed,
            duration=duration,
            token_count=token_count,
            cached=cached,
            output_preview=result[:50] + "..."
        )
        self.metrics.append(metric)
        
        return result
    
    def get_summary(self):
        """Get generation performance summary."""
        if not self.metrics:
            return "No generations recorded"
        
        total_time = sum(m.duration for m in self.metrics)
        cached_count = sum(1 for m in self.metrics if m.cached)
        avg_tokens = sum(m.token_count for m in self.metrics) / len(self.metrics)
        
        return f"""
Generation Summary:
- Total generations: {len(self.metrics)}
- Total time: {total_time:.2f}s
- Average time: {total_time/len(self.metrics):.3f}s
- Cached hits: {cached_count} ({cached_count/len(self.metrics)*100:.1f}%)
- Average tokens: {avg_tokens:.0f}
"""

# Example usage
monitor = GenerationMonitor()

# Generate with monitoring
prompts = [
    "Write a Python function",
    "Write a Python function",  # Duplicate - should be cached
    "Explain recursion",
    "Write a Python function",  # Another duplicate
    "Create a class example"
]

for prompt in prompts:
    result = monitor.generate_with_metrics(prompt)
    print(f"Generated for '{prompt[:20]}...': {len(result)} chars")

print(monitor.get_summary())

# Show detailed metrics
print("\nDetailed Metrics:")
for i, metric in enumerate(monitor.metrics, 1):
    print(f"{i}. {metric.prompt[:30]}... - {metric.duration:.3f}s "
          f"{'(cached)' if metric.cached else '(computed)'}")
```

---

## Integration Examples

### Flask Web Service

```python
from flask import Flask, request, jsonify
import steadytext

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_text():
    """API endpoint for text generation."""
    data = request.get_json()
    
    # Extract parameters
    prompt = data.get('prompt', '')
    seed = data.get('seed', 42)
    max_tokens = data.get('max_tokens', 512)
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        # Generate text
        result = steadytext.generate(
            prompt, 
            seed=seed,
            max_new_tokens=max_tokens
        )
        
        return jsonify({
            'prompt': prompt,
            'seed': seed,
            'generated_text': result,
            'token_count': len(result.split())
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate/stream', methods=['POST'])
def stream_text():
    """SSE endpoint for streaming generation."""
    from flask import Response
    
    data = request.get_json()
    prompt = data.get('prompt', '')
    seed = data.get('seed', 42)
    
    def generate():
        yield "data: {\"status\": \"starting\"}\n\n"
        
        for token in steadytext.generate_iter(prompt, seed=seed):
            # Escape token for JSON
            escaped = token.replace('"', '\\"').replace('\n', '\\n')
            yield f"data: {{\"token\": \"{escaped}\"}}\n\n"
        
        yield "data: {\"status\": \"complete\"}\n\n"
    
    return Response(generate(), mimetype="text/event-stream")

# Run with: flask run
```

### Async Generation with asyncio

```python
import asyncio
import steadytext
from concurrent.futures import ThreadPoolExecutor

class AsyncGenerator:
    """Async wrapper for SteadyText generation."""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text asynchronously."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            steadytext.generate,
            prompt,
            *kwargs.values()
        )
        return result
    
    async def generate_many(self, prompts: list, base_seed: int = 42) -> list:
        """Generate multiple texts concurrently."""
        tasks = [
            self.generate_async(prompt, seed=base_seed + i)
            for i, prompt in enumerate(prompts)
        ]
        return await asyncio.gather(*tasks)
    
    def cleanup(self):
        """Cleanup executor."""
        self.executor.shutdown(wait=True)

# Example usage
async def main():
    generator = AsyncGenerator()
    
    # Single async generation
    result = await generator.generate_async("Write async Python code")
    print(f"Single result: {result[:100]}...\n")
    
    # Batch async generation
    prompts = [
        "Explain async/await",
        "Write a coroutine example",
        "Describe event loops",
        "Create an async API client"
    ]
    
    start = asyncio.get_event_loop().time()
    results = await generator.generate_many(prompts)
    duration = asyncio.get_event_loop().time() - start
    
    print(f"Generated {len(results)} texts in {duration:.2f}s")
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\n{i+1}. {prompt}:\n{result[:100]}...")
    
    generator.cleanup()

# Run the async example
if __name__ == "__main__":
    asyncio.run(main())
```
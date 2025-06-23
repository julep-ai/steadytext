# Model Switching in SteadyText

SteadyText 1.0.0 introduces dynamic model switching, allowing you to use different language models without restarting your application or changing environment variables.

## Overview

The model switching feature enables you to:

1. **Use different models for different tasks** - Choose smaller models for speed or larger models for quality
2. **Switch models at runtime** - No need to restart your application
3. **Maintain deterministic outputs** - Each model produces consistent results
4. **Cache multiple models** - Models are cached after first load for efficiency

## Usage Methods

### 1. Using Size Parameter (New!)

The simplest way to choose a model based on your needs:

```python
from steadytext import generate

# Quick, lightweight tasks
text = generate("Simple question", size="small")    # Uses Qwen3-0.6B

# Balanced performance (default)
text = generate("General task", size="medium")      # Uses Qwen3-1.7B

# Complex, high-quality output
text = generate("Complex analysis", size="large")   # Uses Qwen3-4B
```

### 2. Using the Model Registry

For more specific model selection:

```python
from steadytext import generate

# Use a smaller, faster model
text = generate("Explain machine learning", model="qwen2.5-0.5b")

# Use a larger, more capable model
text = generate("Write a detailed essay", model="qwen2.5-7b")
```

Available models in the registry:

| Model Name | Size | Use Case | Size Parameter |
|------------|------|----------|----------------|
| `qwen3-0.6b` | 0.6B | Very fast, simple tasks | `small` |
| `qwen3-1.7b` | 1.7B | Default, balanced performance | `medium` |
| `qwen3-4b` | 4B | Better quality, slower | `large` |
| `qwen3-8b` | 8B | High quality, resource intensive | - |
| `qwen2.5-0.5b` | 0.5B | Fast, lightweight tasks | - |
| `qwen2.5-1.5b` | 1.5B | Good balance of speed/quality | - |
| `qwen2.5-3b` | 3B | Enhanced capabilities | - |
| `qwen2.5-7b` | 7B | Best quality, slower | - |

### 3. Using Custom Models

Specify any GGUF model from Hugging Face:

```python
from steadytext import generate

# Use a custom model
text = generate(
    "Create a Python function",
    model_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
    model_filename="qwen2.5-7b-instruct-q8_0.gguf"
)
```

### 4. Using Environment Variables

Set default models via environment variables:

```bash
export STEADYTEXT_GENERATION_MODEL_REPO="Qwen/Qwen2.5-3B-Instruct-GGUF"
export STEADYTEXT_GENERATION_MODEL_FILENAME="qwen2.5-3b-instruct-q8_0.gguf"
```

## Streaming Generation

Model switching works with streaming generation too:

```python
from steadytext import generate_iter

# Stream with a specific model
for token in generate_iter("Tell me a story", model="qwen2.5-3b"):
    print(token, end="", flush=True)
```

## Model Selection Guide

### For Speed (0.5B - 1.5B models)
- **Use cases**: Chat responses, simple completions, real-time applications
- **Recommended**: `qwen3-0.6b` (size="small"), `qwen2.5-0.5b`, `qwen2.5-1.5b`
- **Trade-off**: Faster generation, simpler outputs

### For Balance (1.7B - 3B models)
- **Use cases**: General text generation, summaries, explanations
- **Recommended**: `qwen3-1.7b` (size="medium", default), `qwen2.5-3b`
- **Trade-off**: Good quality with reasonable speed

### For Quality (4B+ models)
- **Use cases**: Complex reasoning, detailed content, creative writing
- **Recommended**: `qwen3-4b` (size="large"), `qwen2.5-7b`, `qwen3-8b`
- **Trade-off**: Best quality, slower generation

## Performance Considerations

1. **First Load**: The first use of a model downloads it (if not cached) and loads it into memory
2. **Model Caching**: Once loaded, models remain in memory for fast switching
3. **Memory Usage**: Each loaded model uses RAM - consider your available resources
4. **Determinism**: All models maintain deterministic outputs with the same seed

## Examples

### Adaptive Model Selection

```python
from steadytext import generate

def smart_generate(prompt, max_length=100):
    """Use different models based on task complexity."""
    if max_length < 50:
        # Use fast model for short outputs
        return generate(prompt, model="qwen2.5-0.5b")
    elif max_length < 200:
        # Use balanced model
        return generate(prompt, model="qwen3-1.7b")
    else:
        # Use high-quality model for long outputs
        return generate(prompt, model="qwen2.5-7b")
```

### A/B Testing Models

```python
from steadytext import generate

prompts = ["Explain quantum computing", "Write a haiku", "Solve 2+2"]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    
    # Test with small model
    small = generate(prompt, model="qwen2.5-0.5b")
    print(f"Small model: {small[:100]}...")
    
    # Test with large model
    large = generate(prompt, model="qwen2.5-3b")
    print(f"Large model: {large[:100]}...")
```

## Troubleshooting

### Model Not Found
If a model download fails, you'll get deterministic fallback text. Check:
- Internet connection
- Hugging Face availability
- Model name spelling

### Out of Memory
Large models require significant RAM. Solutions:
- Use smaller quantized models
- Clear model cache with `clear_model_cache()`
- Use one model at a time

### Slow First Load
Initial model loading takes time due to:
- Downloading (first time only)
- Loading into memory
- Model initialization

Subsequent uses are much faster as models are cached.
# Model Switching in SteadyText

SteadyText v2.0.0+ supports dynamic model switching with the Gemma-3n model family, allowing you to use different model sizes without restarting your application.

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
text = generate("Simple task", size="small")   # Uses Gemma-3n-2B (default)
text = generate("Complex analysis", size="large")   # Uses Gemma-3n-4B
```

### 2. Using the Model Registry

For more specific model selection:

```python
from steadytext import generate

# Use a smaller, faster model
text = generate("Explain machine learning", size="small")   # Gemma-3n-2B

# Use a larger, more capable model
text = generate("Write a detailed essay", size="large")    # Gemma-3n-4B
```

Available models in the registry (v2.0.0+):

| Model Name | Size | Use Case | Size Parameter |
|------------|------|----------|----------------|
| `gemma-3n-2b` | 2B | Default, fast tasks | `small` |
| `gemma-3n-4b` | 4B | High quality, complex tasks | `large` |

> **Note:** SteadyText v2.0.0+ focuses on the Gemma-3n model family. Previous versions (v1.x) supported Qwen models which are now deprecated.

### 3. Using Custom Models

Specify any GGUF model from Hugging Face:

```python
from steadytext import generate

# Use a custom model
text = generate(
    "Create a Python function",
    model_repo="ggml-org/gemma-3n-E4B-it-GGUF",
    model_filename="gemma-3n-E4B-it-Q8_0.gguf"
)
```

### 4. Using Environment Variables

Set default models via environment variables:

```bash
# Use small model by default
export STEADYTEXT_DEFAULT_SIZE="small"

# Or specify custom model (advanced)
export STEADYTEXT_GENERATION_MODEL_REPO="ggml-org/gemma-3n-E2B-it-GGUF"
export STEADYTEXT_GENERATION_MODEL_FILENAME="gemma-3n-E2B-it-Q8_0.gguf"
```

## Streaming Generation

Model switching works with streaming generation too:

```python
from steadytext import generate_iter

# Stream with a specific model size
for token in generate_iter("Tell me a story", size="large"):
    print(token, end="", flush=True)
```

## Model Selection Guide

### For Speed (2B model)
- **Use cases**: Chat responses, simple completions, real-time applications
- **Recommended**: `gemma-3n-2b` (size="small")
- **Trade-off**: Faster generation, simpler outputs

### For Quality (4B model)
- **Use cases**: Complex reasoning, detailed content, creative writing
- **Recommended**: `gemma-3n-4b` (size="large")
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

def smart_generate(prompt, complexity="medium"):
    """Use different models based on task complexity."""
    if complexity == "low":
        # Use fast model for simple tasks
        return generate(prompt, size="small")
    else:
        # Use high-quality model for complex tasks
        return generate(prompt, size="large")
```

### A/B Testing Models

```python
from steadytext import generate

prompts = ["Explain quantum computing", "Write a haiku", "Solve 2+2"]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    
    # Test with small model
    small = generate(prompt, size="small")
    print(f"Small model: {small[:100]}...")
    
    # Test with large model
    large = generate(prompt, size="large")
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
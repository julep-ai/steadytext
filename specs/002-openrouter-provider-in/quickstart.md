# OpenRouter Provider Quickstart

This guide demonstrates how to use the OpenRouter provider with SteadyText for text generation and embeddings.

## Prerequisites

1. **OpenRouter API Key**: Get your API key from [OpenRouter](https://openrouter.ai/)
2. **Environment Setup**: Set your API key as an environment variable
3. **Unsafe Mode**: Enable unsafe mode for remote provider access

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-your-api-key-here"

# Enable unsafe mode (required for all remote providers)
export STEADYTEXT_UNSAFE_MODE=true
```

## Quick Start - Python API

### Basic Text Generation

```python
from steadytext import generate

# Generate text using Claude 3.5 Sonnet via OpenRouter
response = generate(
    "Explain quantum computing in simple terms",
    model="openrouter:anthropic/claude-3.5-sonnet",
    unsafe_mode=True
)
print(response)
```

### Streaming Generation

```python
from steadytext import generate

# Stream responses for real-time output
for chunk in generate(
    "Write a short story about AI",
    model="openrouter:openai/gpt-4",
    stream=True,
    unsafe_mode=True
):
    print(chunk, end="", flush=True)
```

### Text Embeddings

```python
from steadytext import embed
import numpy as np

# Generate embeddings using OpenRouter
embeddings = embed(
    ["Hello world", "Goodbye world"],
    model="openrouter:openai/text-embedding-3-small",
    unsafe_mode=True
)

# embeddings is a numpy array of shape (2, embedding_dim)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Similarity: {np.dot(embeddings[0], embeddings[1])}")
```

### Advanced Parameters

```python
from steadytext import generate

# Use advanced generation parameters
response = generate(
    "Write a creative poem",
    model="openrouter:anthropic/claude-3.5-sonnet",
    temperature=0.9,      # More creative (0-2)
    max_tokens=150,       # Limit response length
    top_p=0.9,           # Nucleus sampling
    unsafe_mode=True
)
print(response)
```

## Quick Start - Command Line Interface

### Basic Generation

```bash
# Generate text from command line
echo "What is machine learning?" | st generate --model "openrouter:anthropic/claude-3.5-sonnet"

# Or with arguments
st generate --model "openrouter:openai/gpt-4" --prompt "Explain Python decorators"
```

### Streaming Output

```bash
# Stream responses for real-time output
st generate --model "openrouter:anthropic/claude-3.5-sonnet" \
            --stream \
            --prompt "Tell me a joke"
```

### Generate Embeddings

```bash
# Generate embeddings and save to file
echo "Hello world" | st embed --model "openrouter:openai/text-embedding-3-small" \
                               --format json > embeddings.json

# Multiple texts at once
st embed --model "openrouter:openai/text-embedding-3-small" \
         --input "text1.txt,text2.txt" \
         --format numpy \
         --output embeddings.npy
```

### Advanced CLI Usage

```bash
# Use generation parameters
st generate --model "openrouter:anthropic/claude-3.5-sonnet" \
            --temperature 0.8 \
            --max-tokens 200 \
            --top-p 0.95 \
            --prompt "Write a haiku about coding"

# JSON output for programmatic use
st generate --model "openrouter:openai/gpt-4" \
            --format json \
            --prompt "List 3 benefits of cloud computing"
```

## Available Models

OpenRouter provides access to many models. Here are some popular options:

### Chat/Text Generation Models

```python
# Anthropic models
"openrouter:anthropic/claude-3.5-sonnet"
"openrouter:anthropic/claude-3-haiku"

# OpenAI models
"openrouter:openai/gpt-4"
"openrouter:openai/gpt-4-turbo"
"openrouter:openai/gpt-3.5-turbo"

# Meta models
"openrouter:meta-llama/llama-3.1-70b-instruct"
"openrouter:meta-llama/llama-3.1-8b-instruct"

# Google models
"openrouter:google/gemini-pro"
"openrouter:google/gemini-flash"

# Specialized code models
"openrouter:codellama/codellama-70b-instruct"
```

### Embedding Models

```python
# OpenAI embeddings
"openrouter:openai/text-embedding-3-small"
"openrouter:openai/text-embedding-3-large"

# Other embedding providers
"openrouter:voyage/voyage-2"
"openrouter:thenlper/gte-large"
```

### List Available Models

```python
from steadytext.providers.registry import list_remote_models

# Get all available models by provider
models = list_remote_models()
openrouter_models = models.get("openrouter", [])
print(f"OpenRouter has {len(openrouter_models)} models available")
```

## Error Handling and Fallbacks

SteadyText provides robust error handling with deterministic fallbacks:

### Handling API Errors

```python
from steadytext import generate
from steadytext.providers.openrouter import OpenRouterError

try:
    response = generate(
        "Hello world",
        model="openrouter:anthropic/claude-3.5-sonnet",
        unsafe_mode=True
    )
    print(response)
except OpenRouterError as e:
    print(f"OpenRouter error: {e}")
    # SteadyText automatically falls back to deterministic generation
```

### Rate Limiting

```python
# OpenRouter rate limits are handled automatically with retries
response = generate(
    "Complex analysis task",
    model="openrouter:anthropic/claude-3.5-sonnet",
    unsafe_mode=True
)
# Will retry with exponential backoff if rate limited
```

### Offline/Fallback Behavior

```python
# If OpenRouter is unavailable, SteadyText falls back to deterministic generation
response = generate(
    "Hello world",
    model="openrouter:anthropic/claude-3.5-sonnet",
    unsafe_mode=True
)
# Always returns a response, even if API is down
```

## Configuration Options

### Environment Variables

```bash
# Required
export OPENROUTER_API_KEY="sk-or-your-key-here"
export STEADYTEXT_UNSAFE_MODE=true

# Optional performance tuning
export OPENROUTER_TIMEOUT=60          # Request timeout in seconds
export OPENROUTER_MAX_RETRIES=3       # Maximum retry attempts
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"  # Custom endpoint
```

### Python Configuration

```python
from steadytext.providers.openrouter import OpenRouterProvider

# Custom provider configuration
provider = OpenRouterProvider(
    api_key="your-key-here",
    model="anthropic/claude-3.5-sonnet",
    timeout=(30, 120),  # (connect, read) timeouts
    max_retries=3
)

# Use with generate/embed functions
response = generate(
    "Hello world",
    model="openrouter:anthropic/claude-3.5-sonnet",
    unsafe_mode=True,
    provider_config={
        "timeout": (60, 180),
        "max_retries": 5
    }
)
```

## Integration with Existing Code

OpenRouter integrates seamlessly with existing SteadyText code:

### Drop-in Replacement

```python
# Existing code using local models
response = generate("Hello world")

# Updated to use OpenRouter - just add model parameter
response = generate(
    "Hello world",
    model="openrouter:anthropic/claude-3.5-sonnet",
    unsafe_mode=True
)
```

### Mixed Usage

```python
from steadytext import generate, embed

# Use local models for some tasks
local_response = generate("Quick question")

# Use OpenRouter for complex tasks
complex_response = generate(
    "Complex analysis requiring reasoning",
    model="openrouter:anthropic/claude-3.5-sonnet",
    unsafe_mode=True
)

# Use OpenRouter embeddings for semantic search
embeddings = embed(
    ["doc1", "doc2", "doc3"],
    model="openrouter:openai/text-embedding-3-large",
    unsafe_mode=True
)
```

## Performance Tips

1. **Model Selection**: Choose appropriate models for your use case
   - Use smaller/faster models for simple tasks
   - Use larger models for complex reasoning

2. **Caching**: OpenRouter responses can be cached for repeated queries
   ```python
   # Same prompt will use cached response
   response1 = generate("What is Python?", model="openrouter:openai/gpt-4", unsafe_mode=True)
   response2 = generate("What is Python?", model="openrouter:openai/gpt-4", unsafe_mode=True)
   ```

3. **Batch Processing**: Use batch embeddings for multiple texts
   ```python
   # More efficient than individual calls
   embeddings = embed(
       ["text1", "text2", "text3", "text4"],
       model="openrouter:openai/text-embedding-3-small",
       unsafe_mode=True
   )
   ```

4. **Streaming**: Use streaming for long responses to improve perceived performance
   ```python
   for chunk in generate("Write a long essay", stream=True, model="openrouter:anthropic/claude-3.5-sonnet", unsafe_mode=True):
       print(chunk, end="")
   ```

## Next Steps

- Explore [OpenRouter's model documentation](https://openrouter.ai/docs) for model-specific capabilities
- Check [pricing](https://openrouter.ai/models) for cost optimization
- Review [SteadyText documentation](../README.md) for advanced features
- Set up monitoring and logging for production usage

## Support

- OpenRouter-specific issues: Check [OpenRouter documentation](https://openrouter.ai/docs)
- SteadyText integration issues: Open an issue on the [SteadyText repository](https://github.com/steadytext/steadytext)
- Provider configuration: Review the [unsafe mode documentation](../docs/unsafe-mode.md)
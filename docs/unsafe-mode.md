# Unsafe Mode: Remote Models with Best-Effort Determinism

> ⚠️ **WARNING**: Remote models provide only **best-effort determinism**. Results may vary between calls, environments, and over time. For true determinism, use local GGUF models (default SteadyText behavior).

## Overview

SteadyText's unsafe mode allows you to use remote AI models (OpenAI, Cerebras, etc.) that support seed parameters for reproducibility. While these models attempt to provide consistent outputs when given the same seed, they cannot guarantee the same level of determinism as local models.

## Why "Unsafe"?

Remote models are considered "unsafe" because:

- **No Guaranteed Determinism**: Results may vary despite using the same seed
- **External Dependencies**: Relies on third-party APIs that may change
- **Version Changes**: Model updates can alter outputs
- **Infrastructure Variability**: Different servers may produce different results
- **API Costs**: Unlike local models, remote models incur per-token charges

## Prerequisites

To use unsafe mode with OpenAI models, you need to install the OpenAI client:

```bash
pip install openai
# or
pip install steadytext[unsafe]
```

## Enabling Unsafe Mode

Unsafe mode requires explicit opt-in via environment variable:

```bash
export STEADYTEXT_UNSAFE_MODE=true
```

## Supported Providers

### OpenAI

Supported models (all models available through OpenAI API):
- `gpt-4o` and `gpt-4o-mini` (recommended for seed support)
- `gpt-5-mini` and `gpt-5-pro` (reasoning models, temperature automatically set to 1.0)
- `o1-preview` and `o1-mini` (reasoning models, temperature automatically set to 1.0)
- `gpt-4-turbo` and variants
- `gpt-3.5-turbo` and variants
- Any future models accessible via the OpenAI API

**Note on Reasoning Models (v2025.8.17+):**
GPT-5 series and o1 series are reasoning models that require `temperature=1.0`. SteadyText automatically adjusts the temperature for these models.

Setup:
```bash
export OPENAI_API_KEY=your-api-key
```

Note: The provider dynamically supports all models available through your OpenAI account.

### Cerebras

Supported models (all models available through Cerebras Cloud API):
- `llama3.1-8b` and `llama3.1-70b`
- `llama3-8b` and `llama3-70b`
- Any future models accessible via the Cerebras API

Setup:
```bash
export CEREBRAS_API_KEY=your-api-key
```

Note: The provider dynamically supports all models available through your Cerebras account.

## Usage

### Python API

```python
import os
import steadytext

# Enable unsafe mode
os.environ["STEADYTEXT_UNSAFE_MODE"] = "true"

# Use OpenAI
text = steadytext.generate(
    "Explain quantum computing",
    model="openai:gpt-4o-mini",
    seed=42  # Best-effort determinism
)

# Use GPT-5 reasoning models (v2025.8.17+)
text = steadytext.generate(
    "Complex reasoning task",
    model="openai:gpt-5-mini",
    unsafe_mode=True
    # Note: temperature automatically set to 1.0 for reasoning models
)

# Pass custom provider options (v2025.8.17+)
text = steadytext.generate(
    "Creative writing",
    model="openai:gpt-4o-mini",
    unsafe_mode=True,
    options={"top_p": 0.95, "presence_penalty": 0.5}
)

# Use Cerebras
text = steadytext.generate(
    "Write a Python function",
    model="cerebras:llama3.1-8b",
    seed=42
)

# Streaming also supported
for token in steadytext.generate_iter(
    "Tell me a story",
    model="openai:gpt-4o-mini"
):
    print(token, end='')

# Structured generation (v2.6.1+: full support)
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# JSON generation with schemas
result = steadytext.generate(
    "Create a person named Alice, age 30",
    model="openai:gpt-4o-mini",
    schema=Person,
    unsafe_mode=True
)

# Regex-constrained generation
phone = steadytext.generate(
    "My phone number is",
    model="openai:gpt-4o-mini",
    regex=r"\d{3}-\d{3}-\d{4}",
    unsafe_mode=True
)

# Choice-constrained generation
sentiment = steadytext.generate(
    "This product is amazing!",
    model="openai:gpt-4o-mini",
    choices=["positive", "negative", "neutral"],
    unsafe_mode=True
)
```

### CLI

```bash
# Enable unsafe mode
export STEADYTEXT_UNSAFE_MODE=true

# Generate with OpenAI
echo "Explain AI" | st --unsafe-mode --model openai:gpt-4o-mini

# Use GPT-5 reasoning models (v2025.8.17+)
echo "Solve this problem" | st --unsafe-mode --model openai:gpt-5-mini
echo "Complex reasoning" | st --unsafe-mode --model openai:gpt-5-pro

# Generate with Cerebras
echo "Write code" | st --unsafe-mode --model cerebras:llama3.1-8b

# With custom seed for reproducibility
echo "Tell me a story" | st --unsafe-mode --model openai:gpt-4o-mini --seed 123

# Pass custom provider options (v2025.8.17+)
echo "Creative writing" | st --unsafe-mode --model openai:gpt-4o-mini \
    --options '{"top_p": 0.95, "presence_penalty": 0.5}'

# Structured generation with remote models
echo "Create a person" | st --unsafe-mode --model openai:gpt-4o-mini \
    --schema '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}' \
    --wait
```

## Limitations

When using unsafe mode:

1. **Full Structured Output (v2.6.1+)**: Remote models now support JSON schemas, regex patterns, and choice constraints
2. **No Logprobs**: Log probabilities are not available from remote APIs
3. **No Embeddings**: Only generation is supported, not embeddings (except for dedicated embedding providers like VoyageAI and Jina)
4. **Best-Effort Only**: Determinism is not guaranteed despite seed parameters
5. **Reasoning Models (v2025.8.17+)**: GPT-5 and o1 series require `temperature=1.0` and don't support temperature tuning

## Best Practices

1. **Use for Prototyping**: Test ideas with remote models, then switch to local models for production
2. **Document Variability**: Note that outputs may change over time
3. **Set Temperature to 0**: Use `temperature=0` for maximum consistency (except for reasoning models which require `temperature=1.0`)
4. **Version Lock**: Document which model versions you're using
5. **Fallback Planning**: Have a plan for when remote APIs are unavailable
6. **Custom Options**: Use the `options` parameter to fine-tune provider-specific behaviors like `top_p`, `presence_penalty`, etc.

## Warning Messages

When using unsafe mode, you'll see warnings like:

```
======================================================================
UNSAFE MODE WARNING: Using OpenAI (gpt-4o-mini) remote model
======================================================================
You are using a REMOTE model that provides only BEST-EFFORT determinism.
Results may vary between:
  - Different API calls
  - Different environments
  - Different times
  - Provider infrastructure changes

For TRUE determinism, use local GGUF models (default SteadyText behavior).
======================================================================
```

## Custom Provider Options (v2025.8.17+)

You can pass provider-specific parameters using the `options` parameter to fine-tune model behavior:

### Python API
```python
result = steadytext.generate(
    "Write a creative story",
    model="openai:gpt-4o-mini",
    unsafe_mode=True,
    options={
        "top_p": 0.95,           # Nucleus sampling
        "presence_penalty": 0.6,  # Reduce repetition
        "frequency_penalty": 0.3, # Encourage diversity
        "max_tokens": 1000       # Custom token limit
    }
)
```

### CLI
```bash
echo "Write a creative story" | st --unsafe-mode \
    --model openai:gpt-4o-mini \
    --options '{"top_p": 0.95, "presence_penalty": 0.6}'
```

Common options for OpenAI models:
- `top_p`: Nucleus sampling threshold (0-1)
- `presence_penalty`: Reduce repetition (-2.0 to 2.0)
- `frequency_penalty`: Encourage diversity (-2.0 to 2.0)
- `max_tokens`: Override default max token limit
- `stop`: Custom stop sequences (array of strings)

Note: Available options vary by provider. Consult your provider's API documentation for supported parameters.

## Comparison: Local vs Remote

| Feature | Local Models (Default) | Remote Models (Unsafe) |
|---------|----------------------|----------------------|
| Determinism | ✅ Guaranteed | ⚠️ Best-effort only |
| Cost | ✅ Free after download | ❌ Per-token charges |
| Speed | ✅ Fast (local) | ❌ Network latency |
| Privacy | ✅ Fully private | ❌ Data sent to API |
| Offline | ✅ Works offline | ❌ Requires internet |
| Models | Limited selection | Many options |
| Custom Options | Limited | ✅ Full provider API access |

## Troubleshooting

### "Unsafe mode requires STEADYTEXT_UNSAFE_MODE=true"

Set the environment variable:
```bash
export STEADYTEXT_UNSAFE_MODE=true
```

### "Provider not available"

Check your API key:
```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Cerebras  
export CEREBRAS_API_KEY=...
```

### "Model does not support seed parameter"

Use only models listed in the supported models section above.

## Migration Path

1. **Prototype** with remote models for flexibility
2. **Evaluate** outputs and identify core use cases
3. **Switch** to local models for production deployment
4. **Maintain** deterministic outputs over time

Remember: SteadyText's core value is **deterministic** text generation. Use unsafe mode only when you explicitly need remote model capabilities and understand the trade-offs.
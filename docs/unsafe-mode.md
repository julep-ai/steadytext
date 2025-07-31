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
- `gpt-4-turbo` and variants
- `gpt-3.5-turbo` and variants
- Any future models accessible via the OpenAI API

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

# Structured generation (JSON schemas)
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# Generate structured JSON with remote models
result = steadytext.generate(
    "Create a person named Alice, age 30",
    model="openai:gpt-4o-mini",
    schema=Person
)
# Result contains JSON wrapped in <json-output> tags
```

### CLI

```bash
# Enable unsafe mode
export STEADYTEXT_UNSAFE_MODE=true

# Generate with OpenAI
echo "Explain AI" | st --unsafe-mode --model openai:gpt-4o-mini

# Generate with Cerebras
echo "Write code" | st --unsafe-mode --model cerebras:llama3.1-8b

# With custom seed for reproducibility
echo "Tell me a story" | st --unsafe-mode --model openai:gpt-4o-mini --seed 123

# Structured generation with remote models
echo "Create a person" | st --unsafe-mode --model openai:gpt-4o-mini \
    --schema '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}' \
    --wait
```

## Limitations

When using unsafe mode:

1. **Limited Structured Output**: Remote models support JSON schemas but not regex or choices constraints
2. **No Logprobs**: Log probabilities are not available from remote APIs
3. **No Embeddings**: Only generation is supported, not embeddings
4. **Best-Effort Only**: Determinism is not guaranteed despite seed parameters

## Best Practices

1. **Use for Prototyping**: Test ideas with remote models, then switch to local models for production
2. **Document Variability**: Note that outputs may change over time
3. **Set Temperature to 0**: Use `temperature=0` for maximum consistency
4. **Version Lock**: Document which model versions you're using
5. **Fallback Planning**: Have a plan for when remote APIs are unavailable

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

## Comparison: Local vs Remote

| Feature | Local Models (Default) | Remote Models (Unsafe) |
|---------|----------------------|----------------------|
| Determinism | ✅ Guaranteed | ⚠️ Best-effort only |
| Cost | ✅ Free after download | ❌ Per-token charges |
| Speed | ✅ Fast (local) | ❌ Network latency |
| Privacy | ✅ Fully private | ❌ Data sent to API |
| Offline | ✅ Works offline | ❌ Requires internet |
| Models | Limited selection | Many options |

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
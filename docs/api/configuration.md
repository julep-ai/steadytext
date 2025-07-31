# Configuration

SteadyText can be configured through environment variables to customize its behavior for different use cases.

## Environment Variables

### Model Configuration

- `STEADYTEXT_MAX_CONTEXT_WINDOW`: Maximum context window size (default: auto-detected per model)
- `STEADYTEXT_ALLOW_MODEL_DOWNLOADS`: Allow automatic model downloads (default: true)
- `STEADYTEXT_DISABLE_DAEMON`: Disable daemon mode (default: false)
- `STEADYTEXT_UNSAFE_MODE`: Enable unsafe mode for remote models (default: false)

### Remote Model Configuration (Unsafe Mode)

- `OPENAI_API_KEY`: API key for OpenAI models (required for openai:* models)
- `CEREBRAS_API_KEY`: API key for Cerebras models (required for cerebras:* models)

### Cache Configuration

#### Generation Cache
- `STEADYTEXT_GENERATION_CACHE_CAPACITY`: Maximum number of cache entries (default: 256)
- `STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB`: Maximum cache file size in MB (default: 50.0)

#### Embedding Cache
- `STEADYTEXT_EMBEDDING_CACHE_CAPACITY`: Maximum number of cache entries (default: 512)
- `STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB`: Maximum cache file size in MB (default: 100.0)

### Cache Backend Configuration

- `STEADYTEXT_CACHE_BACKEND`: Cache backend type (default: sqlite)
  - `sqlite`: Local SQLite database
  - `d1`: Cloudflare D1 distributed database
  - `memory`: In-memory cache (ephemeral)

#### D1 Backend Configuration
- `STEADYTEXT_D1_API_URL`: D1 API endpoint URL
- `STEADYTEXT_D1_API_KEY`: D1 API authentication key
- `STEADYTEXT_D1_BATCH_SIZE`: Batch size for D1 operations (default: 50)

### Daemon Configuration

- `STEADYTEXT_DAEMON_HOST`: Daemon host address (default: localhost)
- `STEADYTEXT_DAEMON_PORT`: Daemon port (default: 5557)

### Shell Integration Configuration

- `STEADYTEXT_SUGGEST_ENABLED`: Enable shell suggestions (default: 1)
- `STEADYTEXT_SUGGEST_MODEL_SIZE`: Model size for suggestions (default: small)
- `STEADYTEXT_SUGGEST_STRATEGY`: Suggestion strategy (default: context)
- `STEADYTEXT_SUGGEST_ASYNC`: Enable async suggestions (default: 1)

## Configuration Examples

### High-Performance Setup
```bash
export STEADYTEXT_MAX_CONTEXT_WINDOW=32768
export STEADYTEXT_GENERATION_CACHE_CAPACITY=1024
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=200.0
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=2048
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=500.0
```

### Minimal Memory Setup
```bash
export STEADYTEXT_MAX_CONTEXT_WINDOW=2048
export STEADYTEXT_GENERATION_CACHE_CAPACITY=64
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=10.0
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=128
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=20.0
```

### Distributed Cache Setup
```bash
export STEADYTEXT_CACHE_BACKEND=d1
export STEADYTEXT_D1_API_URL=https://your-worker.workers.dev
export STEADYTEXT_D1_API_KEY=your-api-key
export STEADYTEXT_D1_BATCH_SIZE=100
```

### Testing Configuration
```bash
export STEADYTEXT_CACHE_BACKEND=memory
export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true
export STEADYTEXT_DISABLE_DAEMON=true
```

### Remote Model Configuration (Unsafe Mode)
```bash
# Enable unsafe mode
export STEADYTEXT_UNSAFE_MODE=true

# Set API keys
export OPENAI_API_KEY=sk-your-openai-key
export CEREBRAS_API_KEY=your-cerebras-key

# Use remote models
python -c "import steadytext; print(steadytext.generate('Hello', model='openai:gpt-4o-mini'))"
```

## Platform-Specific Configuration

### Linux/macOS
Configuration files and caches are stored in:
- Cache: `~/.cache/steadytext/`
- Models: `~/.cache/steadytext/models/`

### Windows
Configuration files and caches are stored in:
- Cache: `%LOCALAPPDATA%\steadytext\steadytext\`
- Models: `%LOCALAPPDATA%\steadytext\steadytext\models\`

## Advanced Configuration

### Custom Model Paths
You can specify custom model repositories and filenames:

```python
import steadytext

# Use custom model repository
text = steadytext.generate(
    "Hello world",
    model_repo="ggml-org/gemma-3n-E2B-it-GGUF",
    model_filename="gemma-3n-E2B-it-Q8_0.gguf"
)
```

### Context Window Management
```python
import os
import steadytext

# Set maximum context window
os.environ["STEADYTEXT_MAX_CONTEXT_WINDOW"] = "8192"

# Generate with automatic context management
text = steadytext.generate("Your very long prompt here...")
```

### Daemon Management
```python
from steadytext.daemon.client import use_daemon

# Force daemon usage
with use_daemon():
    text = steadytext.generate("Hello world")
```

## Troubleshooting Configuration

### Common Issues

1. **Models not downloading**: Set `STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true`
2. **Cache growing too large**: Reduce `*_CACHE_MAX_SIZE_MB` values
3. **Memory usage high**: Reduce `*_CACHE_CAPACITY` values
4. **Daemon connection issues**: Check `STEADYTEXT_DAEMON_HOST` and `STEADYTEXT_DAEMON_PORT`

### Debug Configuration
```bash
export STEADYTEXT_DEBUG=1
export STEADYTEXT_VERBOSE=1
```

For more troubleshooting help, see the [Troubleshooting Guide](../troubleshooting.md).
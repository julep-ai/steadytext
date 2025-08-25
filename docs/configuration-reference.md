# Configuration Reference

Complete reference for all SteadyText configuration options.

## Environment Variables

SteadyText can be configured through environment variables. All variables are optional with sensible defaults.

### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `STEADYTEXT_DEFAULT_SEED` | `42` | Default seed for deterministic generation |
| `STEADYTEXT_DISABLE_DAEMON` | `false` | Disable daemon usage globally |
| `STEADYTEXT_ALLOW_MODEL_DOWNLOADS` | `false` | Allow automatic model downloads (useful for CI) |
| `STEADYTEXT_USE_MINI_MODELS` | `false` | Use mini models for CI/testing |
| `STEADYTEXT_USE_FALLBACK_MODEL` | `false` | Use fallback models for compatibility |
| `STEADYTEXT_UNSAFE_MODE` | `false` | Enable remote models (non-deterministic) |

### Model Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `STEADYTEXT_GENERATION_SIZE` | `small` | Generation model size: `mini`, `small`, `large` |
| `STEADYTEXT_GENERATION_MODEL` | (varies) | Specific generation model name |
| `STEADYTEXT_GENERATION_MODEL_REPO` | (varies) | Custom HuggingFace repo for generation |
| `STEADYTEXT_GENERATION_MODEL_FILENAME` | (varies) | Custom model filename for generation |
| `STEADYTEXT_EMBEDDING_MODEL` | `jina-v4-retrieval` | Embedding model name |
| `STEADYTEXT_EMBEDDING_MODEL_REPO` | (varies) | Custom HuggingFace repo for embeddings |
| `STEADYTEXT_EMBEDDING_MODEL_FILENAME` | (varies) | Custom model filename for embeddings |
| `STEADYTEXT_RERANKING_MODEL` | `qwen3-reranker-4b` | Reranking model name |
| `STEADYTEXT_RERANKING_MODEL_REPO` | (varies) | Custom HuggingFace repo for reranking |
| `STEADYTEXT_RERANKING_MODEL_FILENAME` | (varies) | Custom model filename for reranking |

### Cache Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `STEADYTEXT_GENERATION_CACHE_CAPACITY` | `256` | Max entries in generation cache |
| `STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB` | `50` | Max size of generation cache in MB |
| `STEADYTEXT_EMBEDDING_CACHE_CAPACITY` | `512` | Max entries in embedding cache |
| `STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB` | `100` | Max size of embedding cache in MB |
| `STEADYTEXT_RERANKING_CACHE_CAPACITY` | `256` | Max entries in reranking cache |
| `STEADYTEXT_RERANKING_CACHE_MAX_SIZE_MB` | `50` | Max size of reranking cache in MB |
| `STEADYTEXT_CACHE_BACKEND` | `sqlite` | Cache backend: `sqlite`, `d1`, `memory` |
| `STEADYTEXT_CACHE_DIR` | (platform-specific) | Directory for cache files |

### Daemon Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `STEADYTEXT_DAEMON_HOST` | `127.0.0.1` | Daemon host address |
| `STEADYTEXT_DAEMON_PORT` | `5678` | Daemon port number |
| `STEADYTEXT_DAEMON_TIMEOUT` | `30` | Daemon connection timeout in seconds |
| `STEADYTEXT_DAEMON_LOG_LEVEL` | `INFO` | Daemon log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Generation Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `STEADYTEXT_GENERATION_MAX_TOKENS` | `512` | Default max tokens for generation |
| `STEADYTEXT_GENERATION_TEMPERATURE` | `0.0` | Default temperature (0.0 = deterministic) |
| `STEADYTEXT_GENERATION_TOP_K` | `1` | Top-k sampling (1 = greedy) |
| `STEADYTEXT_GENERATION_TOP_P` | `0.95` | Top-p (nucleus) sampling |
| `STEADYTEXT_GENERATION_MIN_P` | `0.05` | Min-p sampling threshold |
| `STEADYTEXT_GENERATION_EOS_STRING` | `[EOS]` | End-of-sequence string |

### Embedding Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `STEADYTEXT_EMBEDDING_MODE` | `passage` | Default mode: `query` or `passage` |
| `STEADYTEXT_EMBEDDING_DIMENSION` | `1024` | Embedding dimension (truncated from 2048) |
| `STEADYTEXT_EMBEDDING_NORMALIZE` | `true` | L2-normalize embeddings |

### Remote Model Configuration (Unsafe Mode)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (none) | OpenAI API key for remote models |
| `CEREBRAS_API_KEY` | (none) | Cerebras API key for remote models |
| `VOYAGE_API_KEY` | (none) | VoyageAI API key for embeddings |
| `JINA_API_KEY` | (none) | Jina AI API key for embeddings |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL |
| `CEREBRAS_BASE_URL` | `https://api.cerebras.ai/v1` | Cerebras API base URL |

### Logging and Debugging

| Variable | Default | Description |
|----------|---------|-------------|
| `STEADYTEXT_LOG_LEVEL` | `WARNING` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `STEADYTEXT_DEBUG` | `false` | Enable debug mode with verbose output |
| `STEADYTEXT_TRACE` | `false` | Enable trace-level logging |

## Configuration Files

### Model Registry

Models can be configured in code using the model registry:

```python
from steadytext.utils import MODEL_REGISTRY

# Available models
print(MODEL_REGISTRY.keys())
# ['gemma-mini-270m', 'qwen3-4b', 'qwen3-30b', 'jina-v4-retrieval', ...]

# Model details
model_info = MODEL_REGISTRY['qwen3-4b']
print(model_info['repo'])      # HuggingFace repo
print(model_info['filename'])  # Model filename
```

### Size Mappings

Size shortcuts map to specific models:

```python
from steadytext.utils import SIZE_TO_MODEL

print(SIZE_TO_MODEL)
# {
#   'mini': 'gemma-mini-270m',   # ~97MB for CI/testing
#   'small': 'qwen3-4b',          # 3.9GB default
#   'large': 'qwen3-30b'          # 12GB advanced
# }
```

## Configuration Precedence

Configuration is resolved in this order (highest to lowest priority):

1. **Function parameters**: Direct arguments to functions
2. **Environment variables**: System environment settings
3. **Defaults**: Built-in default values

Example:
```python
import os
os.environ['STEADYTEXT_GENERATION_MAX_TOKENS'] = '256'

# Function parameter overrides environment
text = steadytext.generate("Hello", max_new_tokens=128)  # Uses 128

# Environment variable used when no parameter
text = steadytext.generate("Hello")  # Uses 256 from env

# Default used when neither specified
# (Would use 512 if env var not set)
```

## Common Configuration Scenarios

### Development Environment

```bash
# Fast iteration with mini models
export STEADYTEXT_USE_MINI_MODELS=true
export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true
export STEADYTEXT_LOG_LEVEL=DEBUG
```

### Testing/CI Environment

```bash
# Reproducible tests with mini models
export STEADYTEXT_USE_MINI_MODELS=true
export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true
export STEADYTEXT_DEFAULT_SEED=12345
export STEADYTEXT_DISABLE_DAEMON=true
```

### Production Environment

```bash
# Optimized for performance
export STEADYTEXT_GENERATION_CACHE_CAPACITY=1024
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=200
export STEADYTEXT_DAEMON_HOST=0.0.0.0
export STEADYTEXT_DAEMON_PORT=5678
export STEADYTEXT_LOG_LEVEL=WARNING
```

### Offline Environment

```bash
# No model downloads, use existing models
export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=false
export STEADYTEXT_CACHE_DIR=/path/to/shared/cache
```

### High-Memory System

```bash
# Use large models with bigger caches
export STEADYTEXT_GENERATION_SIZE=large
export STEADYTEXT_GENERATION_CACHE_CAPACITY=2048
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=500
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=4096
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=1000
```

## Docker Configuration

### Using Docker Compose

```yaml
version: '3.8'
services:
  steadytext:
    image: steadytext:latest
    environment:
      - STEADYTEXT_GENERATION_SIZE=small
      - STEADYTEXT_DAEMON_HOST=0.0.0.0
      - STEADYTEXT_DAEMON_PORT=5678
      - STEADYTEXT_CACHE_DIR=/cache
      - STEADYTEXT_LOG_LEVEL=INFO
    volumes:
      - ./cache:/cache
      - ./models:/root/.cache/steadytext/models
    ports:
      - "5678:5678"
```

### Using Docker Run

```bash
docker run -d \
  -e STEADYTEXT_GENERATION_SIZE=large \
  -e STEADYTEXT_DAEMON_HOST=0.0.0.0 \
  -e STEADYTEXT_CACHE_DIR=/cache \
  -v $(pwd)/cache:/cache \
  -v $(pwd)/models:/root/.cache/steadytext/models \
  -p 5678:5678 \
  steadytext:latest
```

## Kubernetes Configuration

### ConfigMap Example

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: steadytext-config
data:
  STEADYTEXT_GENERATION_SIZE: "small"
  STEADYTEXT_DAEMON_HOST: "0.0.0.0"
  STEADYTEXT_DAEMON_PORT: "5678"
  STEADYTEXT_CACHE_CAPACITY: "1024"
  STEADYTEXT_LOG_LEVEL: "INFO"
```

### Deployment Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: steadytext
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: steadytext
        image: steadytext:latest
        envFrom:
        - configMapRef:
            name: steadytext-config
        volumeMounts:
        - name: cache
          mountPath: /cache
        - name: models
          mountPath: /root/.cache/steadytext/models
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: steadytext-cache
      - name: models
        persistentVolumeClaim:
          claimName: steadytext-models
```

## Troubleshooting Configuration

### Viewing Current Configuration

```python
import os
import steadytext

# Check environment variables
for key, value in os.environ.items():
    if key.startswith('STEADYTEXT_'):
        print(f"{key}={value}")

# Check model paths
print(f"Model cache: {steadytext.get_model_cache_dir()}")

# Check cache stats
from steadytext import get_cache_manager
cache = get_cache_manager()
print(cache.get_cache_stats())
```

### Common Issues

#### Models Not Loading
```bash
# Check model directory
ls -la ~/.cache/steadytext/models/

# Clear and re-download
rm -rf ~/.cache/steadytext/models/
export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true
```

#### Cache Not Working
```bash
# Check cache directory permissions
ls -la ~/.cache/steadytext/caches/

# Clear cache
rm -rf ~/.cache/steadytext/caches/*.db
```

#### Daemon Connection Failed
```bash
# Check daemon status
st daemon status

# Restart daemon with specific config
export STEADYTEXT_DAEMON_HOST=127.0.0.1
export STEADYTEXT_DAEMON_PORT=5678
st daemon restart
```

## See Also

- [Concepts](concepts.md) - Understanding core concepts
- [API Reference](api/index.md) - Function documentation
- [Examples](examples/index.md) - Usage examples
- [Troubleshooting](troubleshooting.md) - Common problems and solutions
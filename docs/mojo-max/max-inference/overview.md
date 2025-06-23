# MAX Inference Overview

MAX is a high-performance inference framework that enables you to deploy AI models with state-of-the-art performance. It provides an OpenAI-compatible API and supports 500+ models out of the box.

## Key Features

### 1. Universal Model Support
- **500+ Pre-optimized Models**: LLMs, vision models, embeddings, and more
- **Multiple Formats**: GGUF, PyTorch, Hugging Face models
- **Automatic Optimization**: Hardware-specific optimizations applied automatically

### 2. OpenAI-Compatible API
- Drop-in replacement for OpenAI endpoints
- Support for chat completions, embeddings, and more
- Compatible with existing OpenAI client libraries

### 3. Production-Ready
- High-performance serving with batching
- Multi-GPU support with tensor parallelism
- Kubernetes-ready with scaling capabilities
- Docker container for easy deployment

### 4. Hardware Flexibility
- **CPUs**: Optimized for x86_64 processors
- **GPUs**: NVIDIA GPUs (unlimited in Community Edition)
- **Other Accelerators**: AMD GPUs (up to 8 in Community Edition)

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Python API    │     │   REST API      │
│  (Offline Mode) │     │ (Server Mode)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
            ┌────────▼────────┐
            │   MAX Engine    │
            │                 │
            │ • Model Loading │
            │ • Optimization  │
            │ • Execution     │
            └────────┬────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼────┐            ┌────▼────┐
    │   CPU   │            │   GPU   │
    └─────────┘            └─────────┘
```

## Supported Model Types

### Large Language Models (LLMs)
- **Llama Family**: Llama 3.1, Llama 3.2, CodeLlama
- **Mistral Models**: Mistral, Mixtral, Codestral
- **Other Popular**: Qwen, Phi, Gemma, DeepSeek

### Vision Models
- **CLIP**: Image embeddings and classification
- **Stable Diffusion**: Image generation
- **Vision Transformers**: Image classification

### Specialized Models
- **Embedding Models**: Text embeddings for RAG
- **Code Models**: Code generation and completion
- **Audio Models**: Whisper for transcription

## Performance Features

### 1. Quantization Support
```python
# Use quantized models for better performance/memory trade-off
pipeline = PipelineConfig(
    model_path="model.Q4_K_M.gguf",  # 4-bit quantized
)
```

### 2. Continuous Batching
- Dynamic batching of requests
- Optimal GPU utilization
- Reduced latency for concurrent requests

### 3. KV Cache Optimization
- Efficient memory management
- Paged attention for long contexts
- Cache sharing across requests

### 4. Speculative Decoding
```python
# Use a draft model for faster generation
pipeline = PipelineConfig(
    model_path="main_model.gguf",
    draft_model_path="draft_model.gguf",
)
```

## Deployment Options

### 1. Local Development
```bash
# Quick start with CLI
max serve --model-path=modularai/Llama-3.1-8B-Instruct-GGUF

# Python API for integration
from max.entrypoints.llm import LLM
llm = LLM(PipelineConfig(model_path="..."))
```

### 2. Docker Deployment
```bash
# Run with Docker
docker run -p 8000:8000 modular/max \
  max serve --model-path=MODEL_ID

# With GPU support
docker run --gpus all -p 8000:8000 modular/max \
  max serve --model-path=MODEL_ID
```

### 3. Kubernetes/Cloud
- Pre-built Kubernetes manifests
- Auto-scaling support
- Multi-node inference for large models

## Model Management

### Model Sources
1. **Hugging Face Hub**: Direct model IDs
2. **Local Files**: Pre-downloaded models
3. **Custom Models**: Your fine-tuned models

### Model Caching
```python
# Models are cached locally after first download
# Default: ~/.cache/modular/models/

# Custom cache directory
import os
os.environ["MODULAR_CACHE_DIR"] = "/path/to/cache"
```

## API Compatibility

### OpenAI Endpoints
- `/v1/chat/completions` - Chat generation
- `/v1/completions` - Text completion
- `/v1/embeddings` - Text embeddings
- `/v1/models` - List available models

### Extended Features
- Structured output with JSON schema
- Function calling for agents
- Custom sampling parameters
- Log probabilities access

## Integration Examples

### Basic Chat Completion
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="model-name",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Streaming Response
```python
stream = client.chat.completions.create(
    model="model-name",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
```

### Embeddings
```python
embeddings = client.embeddings.create(
    model="model-name",
    input=["Text to embed", "Another text"]
)
```

## Monitoring and Debugging

### Logging
```bash
# Enable debug logging
MODULAR_LOG_LEVEL=debug max serve --model-path=MODEL

# Log to file
max serve --model-path=MODEL --log-file=server.log
```

### Metrics
- Request latency
- Throughput (tokens/second)
- GPU utilization
- Memory usage

### Health Checks
```bash
# Check server status
curl http://localhost:8000/health

# List loaded models
curl http://localhost:8000/v1/models
```

## Best Practices

1. **Model Selection**
   - Use quantized models for better memory efficiency
   - Choose model size based on available hardware
   - Test different quantization levels (Q4, Q5, Q8)

2. **Performance Tuning**
   - Adjust batch size for throughput vs latency
   - Use tensor parallelism for large models
   - Enable GPU acceleration when available

3. **Production Deployment**
   - Use Docker for reproducible deployments
   - Implement proper health checks
   - Monitor resource usage
   - Set up auto-scaling based on load

## Comparison with Alternatives

| Feature | MAX | vLLM | TGI | Ollama |
|---------|-----|------|-----|--------|
| OpenAI API | ✅ | ✅ | ✅ | ✅ |
| Model Support | 500+ | Many | Many | Limited |
| Quantization | ✅ | Limited | ✅ | ✅ |
| Multi-GPU | ✅ | ✅ | ✅ | ❌ |
| CPU Optimized | ✅ | ❌ | Limited | ✅ |
| Custom Ops | ✅ | ❌ | ❌ | ❌ |

## Next Steps

- Try the [Quickstart Guide](../getting-started/quickstart.md)
- Learn about [Serving Models](serving-models.md)
- Explore [Offline Inference](offline-inference.md)
- Check [Deployment Options](deployment.md)
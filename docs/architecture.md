# SteadyText Architecture

This document provides a comprehensive overview of SteadyText's architecture, design decisions, and implementation details.

## Table of Contents

- [Overview](#overview)
- [Core Principles](#core-principles)
- [System Architecture](#system-architecture)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Model Architecture](#model-architecture)
- [Caching Architecture](#caching-architecture)
- [Daemon Architecture](#daemon-architecture)
- [Extension Points](#extension-points)
- [Performance Architecture](#performance-architecture)
- [Security Architecture](#security-architecture)
- [Design Patterns](#design-patterns)
- [Technology Stack](#technology-stack)

## Overview

SteadyText is designed as a deterministic AI text generation and embedding library with a focus on reproducibility, performance, and reliability. The architecture supports multiple deployment modes (direct, daemon, PostgreSQL extension) while maintaining consistent behavior across all interfaces.

### Key Architectural Goals

1. **Determinism**: Same input always produces same output
2. **Performance**: Sub-second response times with caching
3. **Reliability**: Never fails, graceful degradation
4. **Simplicity**: Minimal configuration, intuitive APIs
5. **Extensibility**: Support for custom models and integrations

## Core Principles

### 1. Never Fail Philosophy

```python
# Traditional approach (can fail)
def generate_text(prompt):
    if not model_loaded:
        raise ModelNotLoadedError()
    return model.generate(prompt)

# SteadyText approach (never fails)
def generate_text(prompt):
    if not model_loaded:
        return None  # v2.1.0+ behavior
    return model.generate(prompt)
```

### 2. Deterministic by Design

All operations use fixed seeds and deterministic algorithms:

```python
# Seed propagation through the stack
DEFAULT_SEED = 42

def set_deterministic_environment(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
```

### 3. Lazy Loading

Models are loaded only when first used:

```python
class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
        return cls._instance
    
    def get_model(self):
        if self.model is None:
            self.model = self._load_model()
        return self.model
```

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Applications                         │
├─────────────────────────────────────────────────────────────┤
│                    Interface Layer                           │
│  ┌─────────────┬──────────────┬──────────────┬───────────┐ │
│  │  Python API │   CLI Tools   │  PostgreSQL  │    REST   │ │
│  │             │  (st/steadytext)│  Extension  │    API    │ │
│  └─────────────┴──────────────┴──────────────┴───────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Core Layer                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Unified Processing Engine                  │   │
│  │  ┌──────────────┬────────────────┬────────────────┐ │   │
│  │  │  Generator   │    Embedder    │  Vector Ops    │ │   │
│  │  └──────────────┴────────────────┴────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                        │
│  ┌──────────────┬────────────────┬─────────────────────┐   │
│  │ Model Loader │  Cache Manager  │   Daemon Service    │   │
│  └──────────────┴────────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    Storage Layer                             │
│  ┌──────────────┬────────────────┬─────────────────────┐   │
│  │ Model Files  │  Cache Files    │   Index Files       │   │
│  │   (GGUF)     │   (SQLite)      │    (FAISS)         │   │
│  └──────────────┴────────────────┴─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Deployment Options                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Direct Mode (Default)                           │
│     └─> Application → SteadyText → Models           │
│                                                      │
│  2. Daemon Mode (Recommended for Production)        │
│     └─> Application → Client → Daemon → Models      │
│                                                      │
│  3. PostgreSQL Extension                             │
│     └─> SQL → pg_steadytext → Daemon → Models       │
│                                                      │
│  4. Container/Kubernetes                             │
│     └─> Service → Pod → Container → Daemon          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Component Architecture

### Core Components

#### 1. Generator Component

```python
# steadytext/core/generator.py
class DeterministicGenerator:
    """Core text generation component."""
    
    def __init__(self):
        self.model = None
        self.config = GenerationConfig()
        self.cache = get_cache_manager().generation_cache
    
    def generate(self, prompt: str, seed: int = 42) -> Optional[str]:
        # Check cache first
        cache_key = self._compute_cache_key(prompt, seed)
        if cached := self.cache.get(cache_key):
            return cached
        
        # Load model lazily
        if self.model is None:
            self.model = ModelLoader().get_generation_model()
            if self.model is None:
                return None  # v2.1.0+ behavior
        
        # Generate with deterministic settings
        result = self._generate_deterministic(prompt, seed)
        
        # Cache result
        self.cache.set(cache_key, result)
        
        return result
```

#### 2. Embedder Component

```python
# steadytext/core/embedder.py
class DeterministicEmbedder:
    """Core embedding component."""
    
    def embed(self, text: str, seed: int = 42) -> Optional[np.ndarray]:
        # Similar pattern: cache → model → generate → cache
        cache_key = self._compute_cache_key(text, seed)
        if cached := self.cache.get(cache_key):
            return cached
        
        if self.model is None:
            self.model = ModelLoader().get_embedding_model()
            if self.model is None:
                return None
        
        # Generate L2-normalized embeddings
        embedding = self._embed_deterministic(text, seed)
        embedding = self._normalize_l2(embedding)
        
        self.cache.set(cache_key, embedding)
        return embedding
```

#### 3. Cache Manager

```python
# steadytext/cache_manager.py
class CacheManager:
    """Centralized cache management."""
    
    _instance = None
    
    def __init__(self):
        self.generation_cache = FrecencyCache(
            capacity=int(os.getenv('STEADYTEXT_GENERATION_CACHE_CAPACITY', 256)),
            max_size_mb=float(os.getenv('STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB', 50))
        )
        self.embedding_cache = FrecencyCache(
            capacity=int(os.getenv('STEADYTEXT_EMBEDDING_CACHE_CAPACITY', 512)),
            max_size_mb=float(os.getenv('STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB', 100))
        )
```

## Data Flow

### Generation Flow

```
User Request
    │
    ▼
API Layer (generate())
    │
    ├─> Check Input Validity
    │
    ├─> Compute Cache Key
    │
    ├─> Check Cache ──────┐
    │                     │ (hit)
    │ (miss)              ▼
    ▼                 Return Cached
Model Loading
    │
    ├─> Check Model Status
    │
    │ (not loaded)
    ├─> Load Model ───────┐
    │                     │ (fail)
    │ (loaded)            ▼
    ▼                 Return None
Generate Text
    │
    ├─> Set Deterministic Seed
    │
    ├─> Configure Sampling
    │
    ├─> Run Inference
    │
    ▼
Cache Result
    │
    ▼
Return Result
```

### Embedding Flow

```
Text Input → Tokenization → Model Inference → Raw Embedding
                                                    │
                                                    ▼
                                            L2 Normalization
                                                    │
                                                    ▼
                                            1024-dim Vector
                                                    │
                                                    ▼
                                               Cache & Return
```

## Model Architecture

### Model Selection

```python
MODEL_REGISTRY = {
    'generation': {
        'small': {
            'repo': 'ggml-org/gemma-3n-E2B-it-GGUF',
            'filename': 'gemma-3n-E2B-it-Q8_0.gguf',
            'context_length': 8192,
            'vocab_size': 256128
        },
        'large': {
            'repo': 'ggml-org/gemma-3n-E4B-it-GGUF',
            'filename': 'gemma-3n-E4B-it-Q8_0.gguf',
            'context_length': 8192,
            'vocab_size': 256128
        }
    },
    'embedding': {
        'default': {
            'repo': 'Qwen/Qwen3-Embedding-0.6B-GGUF',
            'filename': 'qwen3-embedding-0.6b-q8_0.gguf',
            'dimension': 1024
        }
    }
}
```

### Model Loading Strategy

1. **Lazy Loading**: Models loaded on first use
2. **Singleton Pattern**: One model instance per type
3. **Thread Safety**: Locks prevent concurrent loading
4. **Graceful Fallback**: Returns None if loading fails

### Model Configuration

```python
GENERATION_CONFIG = {
    'max_tokens': 512,
    'temperature': 0.0,  # Deterministic
    'top_k': 1,          # Greedy decoding
    'top_p': 1.0,
    'repeat_penalty': 1.0,
    'seed': 42,
    'n_threads': 4,
    'n_batch': 512,
    'use_mlock': True,
    'use_mmap': True
}
```

## Caching Architecture

### Cache Design

```
┌─────────────────────────────────────────────┐
│            Cache Manager                     │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────┐    ┌──────────────┐      │
│  │ Generation   │    │  Embedding   │      │
│  │   Cache      │    │    Cache     │      │
│  └──────┬───────┘    └──────┬───────┘      │
│         │                    │               │
│         ▼                    ▼               │
│  ┌─────────────────────────────────┐        │
│  │     Frecency Algorithm          │        │
│  │  (Frequency + Recency scoring)  │        │
│  └─────────────────────────────────┘        │
│                    │                         │
│                    ▼                         │
│  ┌─────────────────────────────────┐        │
│  │    SQLite Backend (Disk)        │        │
│  │  - Thread-safe                  │        │
│  │  - Persistent                   │        │
│  │  - Size-limited                 │        │
│  └─────────────────────────────────┘        │
└─────────────────────────────────────────────┘
```

### Cache Key Generation

```python
def compute_cache_key(prompt: str, seed: int, **kwargs) -> str:
    """Generate deterministic cache key."""
    # Include all parameters that affect output
    key_parts = [
        prompt,
        str(seed),
        str(kwargs.get('max_tokens', 512)),
        str(kwargs.get('eos_string', '[EOS]'))
    ]
    
    # Use SHA256 for consistent hashing
    key_string = '|'.join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()
```

### Cache Eviction Strategy

1. **Frecency Score**: Combines frequency and recency
2. **Size Limits**: Respects configured max size
3. **TTL**: Optional time-to-live for entries
4. **Atomic Operations**: Thread-safe updates

## Daemon Architecture

### Daemon Design

```
┌─────────────────────────────────────────────┐
│              Daemon Process                  │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────────────────────────┐       │
│  │      ZeroMQ REP Server           │       │
│  │   Listening on tcp://*:5557      │       │
│  └────────────┬─────────────────────┘       │
│               │                              │
│               ▼                              │
│  ┌──────────────────────────────────┐       │
│  │      Request Router              │       │
│  │  - generate                      │       │
│  │  - generate_iter                 │       │
│  │  - embed                         │       │
│  │  - ping                          │       │
│  │  - shutdown                      │       │
│  └────────────┬─────────────────────┘       │
│               │                              │
│               ▼                              │
│  ┌──────────────────────────────────┐       │
│  │    Model Instance Pool           │       │
│  │  - Gemma-3n (generation)         │       │
│  │  - Qwen3 (embedding)             │       │
│  └──────────────────────────────────┘       │
│                                              │
└─────────────────────────────────────────────┘
```

### Communication Protocol

```python
# Request format
{
    "id": "unique-request-id",
    "type": "generate",
    "prompt": "Hello world",
    "seed": 42,
    "max_tokens": 512
}

# Response format
{
    "id": "unique-request-id",
    "success": true,
    "result": "Generated text...",
    "cached": false,
    "error": null
}
```

### Connection Management

```python
class DaemonClient:
    def __init__(self, host='127.0.0.1', port=5557):
        self.context = zmq.Context()
        self.socket = None
        self.connected = False
    
    def connect(self):
        """Establish connection with retry logic."""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        
        # Test connection
        if self._ping():
            self.connected = True
        else:
            self._fallback_to_direct()
```

## Extension Points

### Custom Models

```python
# Register custom model
from steadytext.models import register_model

register_model(
    'custom-gen',
    repo='myorg/custom-model-GGUF',
    filename='model.gguf',
    model_type='generation'
)

# Use custom model
text = generate("Hello", model='custom-gen')
```

### Custom Embedders

```python
class CustomEmbedder:
    def embed(self, text: str) -> np.ndarray:
        # Custom embedding logic
        return np.random.randn(1024)

# Register embedder
steadytext.register_embedder('custom', CustomEmbedder())
```

### Plugin System

```python
# Future: Plugin architecture
class SteadyTextPlugin:
    def on_generate_start(self, prompt: str): pass
    def on_generate_complete(self, result: str): pass
    def on_embed_start(self, text: str): pass
    def on_embed_complete(self, embedding: np.ndarray): pass

# Register plugin
steadytext.register_plugin(MyPlugin())
```

## Performance Architecture

### Optimization Strategies

1. **Model Preloading**
   ```python
   # Preload models at startup
   steadytext.preload_models()
   ```

2. **Connection Pooling**
   ```python
   # Daemon connection pool
   pool = ConnectionPool(size=10)
   ```

3. **Batch Processing**
   ```python
   # Process multiple requests efficiently
   results = steadytext.batch_generate(prompts)
   ```

4. **Memory Mapping**
   ```python
   # GGUF models use mmap for efficiency
   config = {'use_mmap': True, 'use_mlock': True}
   ```

### Benchmarking Architecture

```python
class BenchmarkFramework:
    def __init__(self):
        self.metrics = {
            'latency': [],
            'throughput': [],
            'memory': [],
            'cache_hits': []
        }
    
    def run_benchmark(self, workload):
        """Execute standardized benchmark."""
        for operation in workload:
            with self.measure():
                operation.execute()
```

## Security Architecture

### Security Layers

1. **Input Validation**
   ```python
   def validate_input(prompt: str) -> bool:
       # Length limits
       if len(prompt) > MAX_PROMPT_LENGTH:
           return False
       # Character validation
       if contains_invalid_chars(prompt):
           return False
       return True
   ```

2. **Process Isolation**
   - Daemon runs in separate process
   - Limited system access
   - Resource quotas

3. **Communication Security**
   - Local-only ZeroMQ by default
   - Optional TLS for remote connections
   - Request authentication

4. **Model Security**
   - Verified model checksums
   - Restricted model loading paths
   - No arbitrary code execution

## Design Patterns

### 1. Singleton Pattern

Used for model instances and cache manager:

```python
class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
```

### 2. Factory Pattern

For creating different model types:

```python
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, size: str):
        if model_type == 'generation':
            return GenerationModel(size)
        elif model_type == 'embedding':
            return EmbeddingModel()
```

### 3. Strategy Pattern

For different generation strategies:

```python
class GenerationStrategy(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: pass

class GreedyStrategy(GenerationStrategy):
    def generate(self, prompt: str) -> str:
        # Greedy decoding implementation
        pass

class BeamSearchStrategy(GenerationStrategy):
    def generate(self, prompt: str) -> str:
        # Beam search implementation
        pass
```

### 4. Observer Pattern

For event handling:

```python
class EventManager:
    def __init__(self):
        self.listeners = defaultdict(list)
    
    def subscribe(self, event: str, callback):
        self.listeners[event].append(callback)
    
    def notify(self, event: str, data: Any):
        for callback in self.listeners[event]:
            callback(data)
```

## Technology Stack

### Core Technologies

- **Python 3.8+**: Primary language
- **llama-cpp-python**: GGUF model inference
- **NumPy**: Numerical operations
- **SQLite**: Cache storage
- **ZeroMQ**: IPC for daemon
- **FAISS**: Vector indexing

### Development Tools

- **UV**: Package management
- **pytest**: Testing framework
- **ruff**: Linting
- **mypy**: Type checking
- **mkdocs**: Documentation

### Model Format

- **GGUF**: Efficient model storage
- **Quantization**: INT8 for efficiency
- **Compression**: Built-in GGUF compression

## Future Architecture

### Planned Enhancements

1. **Distributed Architecture**
   - Multiple daemon instances
   - Load balancing
   - Horizontal scaling

2. **GPU Support**
   - CUDA acceleration
   - Metal Performance Shaders
   - Vulkan compute

3. **Streaming Architecture**
   - WebSocket support
   - Server-sent events
   - Real-time generation

4. **Cloud Native**
   - Kubernetes operators
   - Service mesh integration
   - Cloud-specific optimizations

### Architecture Evolution

```
Current (Monolithic)          Future (Microservices)
┌─────────────┐              ┌─────────────┐
│  SteadyText │              │   Gateway   │
│   Library   │              └──────┬──────┘
└─────────────┘                     │
                           ┌────────┴────────┐
                           │                 │
                     ┌─────▼────┐    ┌──────▼─────┐
                     │Generation│    │ Embedding  │
                     │  Service │    │  Service   │
                     └──────────┘    └────────────┘
```

## Conclusion

SteadyText's architecture prioritizes:

1. **Simplicity**: Easy to understand and use
2. **Reliability**: Predictable behavior
3. **Performance**: Fast response times
4. **Extensibility**: Easy to extend and customize

The modular design allows for future enhancements while maintaining backward compatibility and consistent behavior across all deployment modes.
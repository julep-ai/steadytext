# Error Handling Guide

Learn how to handle errors gracefully in SteadyText, implement robust fallback strategies, and build resilient applications.

## Overview

SteadyText follows a "never fail" philosophy (with v2.1.0+ updates):

- **Functions return `None`** when models are unavailable (v2.1.0+)
- **No exceptions** are raised during normal operations
- **Graceful degradation** with predictable behavior
- **Clear error indicators** for proper handling
- **Deterministic fallbacks** respect seed values

!!! warning "Breaking Change in v2.1.0"
    The deterministic fallback behavior has been disabled. Functions now return `None` instead of generating fallback text/embeddings when models are unavailable.

## Table of Contents

- [Error Types and Handling](#error-types-and-handling)
- [Generation Error Handling](#generation-error-handling)
- [Embedding Error Handling](#embedding-error-handling)
- [Streaming Error Handling](#streaming-error-handling)
- [Daemon Error Handling](#daemon-error-handling)
- [CLI Error Handling](#cli-error-handling)
- [Production Patterns](#production-patterns)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Recovery Strategies](#recovery-strategies)
- [Best Practices](#best-practices)

## Error Types and Handling

### Common Error Scenarios

```python
import steadytext
import logging
from typing import Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_generation_result(result: Optional[str], prompt: str) -> str:
    """Handle generation result with proper error checking."""
    if result is None:
        logger.error(f"Generation failed for prompt: {prompt}")
        # Implement your fallback strategy
        return f"[Error: Unable to generate response for: {prompt}]"
    
    if not result.strip():
        logger.warning(f"Empty generation for prompt: {prompt}")
        return "[Error: Empty response generated]"
    
    return result

# Usage example
prompt = "Write a summary"
result = steadytext.generate(prompt, seed=42)
handled_result = handle_generation_result(result, prompt)
print(handled_result)
```

### Error Categories

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Any

class ErrorType(Enum):
    """Types of errors in SteadyText operations."""
    MODEL_NOT_LOADED = "model_not_loaded"
    INVALID_INPUT = "invalid_input"
    DAEMON_UNAVAILABLE = "daemon_unavailable"
    CACHE_ERROR = "cache_error"
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    UNKNOWN = "unknown"

@dataclass
class SteadyTextError:
    """Structured error information."""
    error_type: ErrorType
    message: str
    context: dict
    recoverable: bool
    suggested_action: Optional[str] = None

class ErrorHandler:
    """Centralized error handling for SteadyText operations."""
    
    def __init__(self):
        self.error_log = []
        self.error_counts = {error_type: 0 for error_type in ErrorType}
    
    def handle_error(self, error_type: ErrorType, message: str, 
                     context: dict = None, recoverable: bool = True) -> SteadyTextError:
        """Handle and log an error."""
        error = SteadyTextError(
            error_type=error_type,
            message=message,
            context=context or {},
            recoverable=recoverable,
            suggested_action=self._get_suggested_action(error_type)
        )
        
        self.error_log.append(error)
        self.error_counts[error_type] += 1
        
        logger.error(f"{error_type.value}: {message}", extra=context)
        
        return error
    
    def _get_suggested_action(self, error_type: ErrorType) -> str:
        """Get suggested action for error type."""
        actions = {
            ErrorType.MODEL_NOT_LOADED: "Run 'st models download' to download models",
            ErrorType.INVALID_INPUT: "Check input format and constraints",
            ErrorType.DAEMON_UNAVAILABLE: "Start daemon with 'st daemon start'",
            ErrorType.CACHE_ERROR: "Clear cache with 'st cache --clear'",
            ErrorType.TIMEOUT: "Increase timeout or retry operation",
            ErrorType.MEMORY_ERROR: "Reduce batch size or restart daemon",
            ErrorType.UNKNOWN: "Check logs for more information"
        }
        return actions.get(error_type, "")
    
    def get_error_summary(self) -> dict:
        """Get summary of all errors."""
        return {
            "total_errors": len(self.error_log),
            "error_counts": dict(self.error_counts),
            "recent_errors": self.error_log[-10:],
            "most_common": max(self.error_counts.items(), key=lambda x: x[1])
        }

# Global error handler
error_handler = ErrorHandler()
```

## Generation Error Handling

### Basic Error Handling

```python
import steadytext
from typing import Optional

def safe_generate(prompt: str, seed: int = 42, max_retries: int = 3) -> Optional[str]:
    """Generate text with retry logic and error handling."""
    for attempt in range(max_retries):
        try:
            result = steadytext.generate(prompt, seed=seed)
            
            if result is None:
                logger.warning(f"Generation returned None (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    error_handler.handle_error(
                        ErrorType.MODEL_NOT_LOADED,
                        "Generation failed after all retries",
                        {"prompt": prompt, "seed": seed, "attempts": max_retries}
                    )
                    return None
            
            return result
            
        except Exception as e:
            logger.exception(f"Unexpected error in generation: {e}")
            error_handler.handle_error(
                ErrorType.UNKNOWN,
                str(e),
                {"prompt": prompt, "seed": seed, "attempt": attempt + 1}
            )
            
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                return None
    
    return None

# Usage
result = safe_generate("Write a poem", seed=123)
if result:
    print(result)
else:
    print("Failed to generate text. Please check the error log.")
```

### Advanced Generation Error Handling

```python
import steadytext
from typing import Optional, Dict, Any, Callable
import time
import hashlib

class RobustGenerator:
    """Robust text generation with comprehensive error handling."""
    
    def __init__(self, 
                 fallback_strategy: str = "template",
                 cache_fallbacks: bool = True,
                 alert_threshold: int = 5):
        self.fallback_strategy = fallback_strategy
        self.cache_fallbacks = cache_fallbacks
        self.alert_threshold = alert_threshold
        self.fallback_cache = {}
        self.consecutive_failures = 0
        self.success_callbacks = []
        self.failure_callbacks = []
    
    def on_success(self, callback: Callable):
        """Register success callback."""
        self.success_callbacks.append(callback)
    
    def on_failure(self, callback: Callable):
        """Register failure callback."""
        self.failure_callbacks.append(callback)
    
    def generate(self, prompt: str, seed: int = 42, **kwargs) -> str:
        """Generate text with comprehensive error handling."""
        start_time = time.time()
        
        try:
            # Attempt generation
            result = steadytext.generate(prompt, seed=seed, **kwargs)
            
            if result is not None:
                # Success
                self.consecutive_failures = 0
                self._notify_success(prompt, result, time.time() - start_time)
                return result
            
            # Generation failed
            self.consecutive_failures += 1
            self._notify_failure(prompt, "Generation returned None")
            
            # Check alert threshold
            if self.consecutive_failures >= self.alert_threshold:
                self._trigger_alert(f"Generation failures exceeded threshold: {self.consecutive_failures}")
            
            # Apply fallback strategy
            return self._apply_fallback(prompt, seed)
            
        except Exception as e:
            self.consecutive_failures += 1
            self._notify_failure(prompt, str(e))
            return self._apply_fallback(prompt, seed)
    
    def _apply_fallback(self, prompt: str, seed: int) -> str:
        """Apply fallback strategy based on configuration."""
        # Check cache first
        cache_key = self._get_cache_key(prompt, seed)
        if self.cache_fallbacks and cache_key in self.fallback_cache:
            return self.fallback_cache[cache_key]
        
        # Generate fallback
        if self.fallback_strategy == "template":
            fallback = self._template_fallback(prompt)
        elif self.fallback_strategy == "hash":
            fallback = self._hash_fallback(prompt, seed)
        elif self.fallback_strategy == "empty":
            fallback = ""
        elif self.fallback_strategy == "error":
            fallback = f"[Error: Unable to generate response for: {prompt[:50]}...]"
        else:
            fallback = "[Generation failed]"
        
        # Cache fallback
        if self.cache_fallbacks:
            self.fallback_cache[cache_key] = fallback
        
        return fallback
    
    def _template_fallback(self, prompt: str) -> str:
        """Generate template-based fallback."""
        templates = {
            "summary": "This is a summary of the requested content.",
            "explanation": "This explains the requested concept.",
            "code": "# Code implementation would go here",
            "story": "Once upon a time, there was a story to be told.",
            "default": "Response generated for: {}"
        }
        
        # Detect prompt type
        prompt_lower = prompt.lower()
        for key in templates:
            if key in prompt_lower:
                return templates[key].format(prompt[:30] + "...")
        
        return templates["default"].format(prompt[:30] + "...")
    
    def _hash_fallback(self, prompt: str, seed: int) -> str:
        """Generate deterministic hash-based fallback."""
        # Create deterministic hash
        hash_input = f"{prompt}:{seed}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        
        return f"[Fallback response {hash_value} for: {prompt[:30]}...]"
    
    def _get_cache_key(self, prompt: str, seed: int) -> str:
        """Generate cache key for fallback."""
        return f"{prompt}:{seed}"
    
    def _notify_success(self, prompt: str, result: str, duration: float):
        """Notify success callbacks."""
        for callback in self.success_callbacks:
            try:
                callback(prompt, result, duration)
            except Exception as e:
                logger.error(f"Error in success callback: {e}")
    
    def _notify_failure(self, prompt: str, error: str):
        """Notify failure callbacks."""
        for callback in self.failure_callbacks:
            try:
                callback(prompt, error)
            except Exception as e:
                logger.error(f"Error in failure callback: {e}")
    
    def _trigger_alert(self, message: str):
        """Trigger alert for critical errors."""
        logger.critical(f"ALERT: {message}")
        # Implement your alerting mechanism here
        # e.g., send email, Slack message, PagerDuty alert
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "consecutive_failures": self.consecutive_failures,
            "fallback_cache_size": len(self.fallback_cache),
            "fallback_strategy": self.fallback_strategy
        }

# Usage example
generator = RobustGenerator(fallback_strategy="template")

# Register callbacks
generator.on_success(lambda p, r, d: logger.info(f"Generated in {d:.2f}s"))
generator.on_failure(lambda p, e: logger.error(f"Failed: {e}"))

# Generate with robust error handling
result = generator.generate("Write a technical blog post", seed=42)
print(result)

# Check stats
stats = generator.get_stats()
print(f"Generator stats: {stats}")
```

## Embedding Error Handling

### Basic Embedding Error Handling

```python
import steadytext
import numpy as np
from typing import Optional

def safe_embed(text: str, seed: int = 42) -> Optional[np.ndarray]:
    """Safely generate embeddings with error handling."""
    try:
        embedding = steadytext.embed(text, seed=seed)
        
        if embedding is None:
            logger.error(f"Embedding returned None for text: {text[:50]}...")
            return None
        
        # Validate embedding
        if not isinstance(embedding, np.ndarray):
            logger.error(f"Invalid embedding type: {type(embedding)}")
            return None
        
        if embedding.shape != (1024,):
            logger.error(f"Invalid embedding shape: {embedding.shape}")
            return None
        
        # Check for NaN or Inf values
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            logger.error("Embedding contains NaN or Inf values")
            return None
        
        return embedding
        
    except Exception as e:
        logger.exception(f"Error generating embedding: {e}")
        return None

# Usage with fallback
def get_embedding_with_fallback(text: str, seed: int = 42) -> np.ndarray:
    """Get embedding with zero-vector fallback."""
    embedding = safe_embed(text, seed=seed)
    
    if embedding is None:
        logger.warning("Using zero-vector fallback for embedding")
        # Return zero vector with correct shape
        return np.zeros(1024, dtype=np.float32)
    
    return embedding
```

### Advanced Embedding Error Handling

```python
import steadytext
import numpy as np
from typing import List, Optional, Dict, Tuple
import hashlib

class RobustEmbedder:
    """Robust embedding generation with comprehensive error handling."""
    
    def __init__(self, 
                 fallback_method: str = "zero",
                 cache_embeddings: bool = True,
                 similarity_threshold: float = 0.95):
        self.fallback_method = fallback_method
        self.cache_embeddings = cache_embeddings
        self.similarity_threshold = similarity_threshold
        self.embedding_cache = {}
        self.error_count = 0
        self.success_count = 0
    
    def embed(self, text: str, seed: int = 42) -> np.ndarray:
        """Generate embedding with error handling."""
        # Check cache
        cache_key = self._get_cache_key(text, seed)
        if self.cache_embeddings and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Attempt embedding
            embedding = steadytext.embed(text, seed=seed)
            
            if embedding is not None and self._validate_embedding(embedding):
                self.success_count += 1
                
                # Cache successful embedding
                if self.cache_embeddings:
                    self.embedding_cache[cache_key] = embedding
                
                return embedding
            
            # Embedding failed
            self.error_count += 1
            return self._generate_fallback(text, seed)
            
        except Exception as e:
            logger.exception(f"Embedding error: {e}")
            self.error_count += 1
            return self._generate_fallback(text, seed)
    
    def embed_batch(self, texts: List[str], seed: int = 42) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with error handling."""
        embeddings = []
        failed_indices = []
        
        for i, text in enumerate(texts):
            try:
                # Use different seed for each text in batch
                text_seed = seed + i
                embedding = self.embed(text, seed=text_seed)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to embed text {i}: {e}")
                failed_indices.append(i)
                embeddings.append(self._generate_fallback(text, seed + i))
        
        if failed_indices:
            logger.warning(f"Failed to embed {len(failed_indices)} texts: {failed_indices}")
        
        return embeddings
    
    def find_similar_cached(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find similar embeddings from cache."""
        if not self.embedding_cache:
            return []
        
        similarities = []
        for cache_key, cached_embedding in self.embedding_cache.items():
            similarity = np.dot(embedding, cached_embedding)
            if similarity >= self.similarity_threshold:
                text = cache_key.split(":")[0]  # Extract text from cache key
                similarities.append((text, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _validate_embedding(self, embedding: np.ndarray) -> bool:
        """Validate embedding array."""
        if not isinstance(embedding, np.ndarray):
            return False
        
        if embedding.shape != (1024,):
            return False
        
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return False
        
        # Check if embedding is normalized
        norm = np.linalg.norm(embedding)
        if not np.isclose(norm, 1.0, atol=1e-6):
            logger.warning(f"Embedding not normalized: norm={norm}")
        
        return True
    
    def _generate_fallback(self, text: str, seed: int) -> np.ndarray:
        """Generate fallback embedding based on method."""
        if self.fallback_method == "zero":
            return np.zeros(1024, dtype=np.float32)
        
        elif self.fallback_method == "random":
            # Deterministic random based on text and seed
            np.random.seed(hash(f"{text}:{seed}") % (2**32))
            embedding = np.random.randn(1024).astype(np.float32)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        
        elif self.fallback_method == "hash":
            # Hash-based deterministic embedding
            hash_input = f"{text}:{seed}".encode()
            hash_bytes = hashlib.sha256(hash_input).digest()
            
            # Convert hash to embedding
            embedding = np.frombuffer(hash_bytes * 32, dtype=np.float32)[:1024]
            # Normalize to [-1, 1] range
            embedding = 2 * (embedding / 255.0) - 1
            # L2 normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        
        else:
            return np.zeros(1024, dtype=np.float32)
    
    def _get_cache_key(self, text: str, seed: int) -> str:
        """Generate cache key."""
        return f"{text}:{seed}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedder statistics."""
        total = self.success_count + self.error_count
        return {
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / total if total > 0 else 0,
            "cache_size": len(self.embedding_cache),
            "fallback_method": self.fallback_method
        }

# Usage example
embedder = RobustEmbedder(fallback_method="hash")

# Single embedding
embedding = embedder.embed("test text", seed=42)
print(f"Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")

# Batch embedding
texts = ["text 1", "text 2", "text 3"]
embeddings = embedder.embed_batch(texts, seed=100)
print(f"Generated {len(embeddings)} embeddings")

# Find similar
similar = embedder.find_similar_cached(embedding, top_k=3)
print(f"Similar embeddings: {similar}")

# Stats
print(f"Embedder stats: {embedder.get_stats()}")
```

## Streaming Error Handling

### Basic Streaming Error Handling

```python
import steadytext
from typing import Iterator, Optional

def safe_generate_iter(prompt: str, seed: int = 42) -> Iterator[str]:
    """Safely generate streaming text with error handling."""
    try:
        stream = steadytext.generate_iter(prompt, seed=seed)
        
        # Check if stream is empty (indicates error)
        first_token = None
        try:
            first_token = next(stream)
        except StopIteration:
            logger.error("Empty stream returned")
            yield "[Error: No content generated]"
            return
        
        # Yield first token
        if first_token:
            yield first_token
        
        # Yield remaining tokens
        for token in stream:
            yield token
            
    except Exception as e:
        logger.exception(f"Streaming error: {e}")
        yield f"[Error: {str(e)}]"

# Usage with timeout
def generate_with_timeout(prompt: str, seed: int = 42, timeout: float = 30.0) -> str:
    """Generate with streaming and timeout."""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Generation timed out")
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))
    
    try:
        result = []
        for token in safe_generate_iter(prompt, seed=seed):
            result.append(token)
        
        # Cancel timeout
        signal.alarm(0)
        return "".join(result)
        
    except TimeoutError:
        logger.error(f"Generation timed out after {timeout}s")
        return "[Error: Generation timed out]"
    finally:
        # Ensure timeout is cancelled
        signal.alarm(0)
```

### Advanced Streaming Error Handling

```python
import steadytext
from typing import Iterator, Optional, Callable
import time
import threading
from queue import Queue, Empty

class RobustStreamer:
    """Robust streaming generation with comprehensive error handling."""
    
    def __init__(self,
                 timeout: float = 30.0,
                 max_tokens: int = 512,
                 heartbeat_interval: float = 1.0):
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.heartbeat_interval = heartbeat_interval
        self.error_handlers = []
        self.token_validators = []
    
    def on_error(self, handler: Callable):
        """Register error handler."""
        self.error_handlers.append(handler)
    
    def add_token_validator(self, validator: Callable[[str], bool]):
        """Add token validator."""
        self.token_validators.append(validator)
    
    def generate_stream(self, prompt: str, seed: int = 42) -> Iterator[str]:
        """Generate streaming text with comprehensive error handling."""
        start_time = time.time()
        tokens_generated = 0
        last_token_time = start_time
        
        try:
            stream = steadytext.generate_iter(prompt, seed=seed)
            
            for token in stream:
                current_time = time.time()
                
                # Check timeout
                if current_time - start_time > self.timeout:
                    self._handle_error("Timeout", prompt, tokens_generated)
                    yield "[Timeout]"
                    break
                
                # Check token count
                if tokens_generated >= self.max_tokens:
                    logger.warning(f"Max tokens ({self.max_tokens}) reached")
                    break
                
                # Validate token
                if not self._validate_token(token):
                    logger.warning(f"Invalid token detected: {repr(token)}")
                    continue
                
                # Check for stalled generation
                if current_time - last_token_time > self.heartbeat_interval * 10:
                    self._handle_error("Stalled generation", prompt, tokens_generated)
                    yield "[Stalled]"
                    break
                
                # Yield valid token
                yield token
                tokens_generated += 1
                last_token_time = current_time
            
            # Check if generation completed successfully
            if tokens_generated == 0:
                self._handle_error("No tokens generated", prompt, 0)
                yield "[No content]"
                
        except Exception as e:
            self._handle_error(str(e), prompt, tokens_generated)
            yield f"[Error: {str(e)}]"
    
    def generate_async(self, prompt: str, seed: int = 42, 
                      callback: Optional[Callable] = None) -> threading.Thread:
        """Generate asynchronously with error handling."""
        result_queue = Queue()
        
        def worker():
            try:
                tokens = []
                for token in self.generate_stream(prompt, seed):
                    tokens.append(token)
                    if callback:
                        callback(token)
                
                result_queue.put(("success", "".join(tokens)))
                
            except Exception as e:
                result_queue.put(("error", str(e)))
        
        thread = threading.Thread(target=worker)
        thread.start()
        
        # Return thread and queue for monitoring
        thread.result_queue = result_queue
        return thread
    
    def generate_with_fallback(self, prompt: str, seed: int = 42,
                              fallback_prompts: Optional[List[str]] = None) -> Iterator[str]:
        """Generate with fallback prompts on error."""
        fallback_prompts = fallback_prompts or [
            f"Please provide a response about: {prompt[:50]}...",
            "Generate a helpful response.",
            "Provide relevant information."
        ]
        
        # Try main prompt
        tokens = list(self.generate_stream(prompt, seed))
        if self._is_valid_generation(tokens):
            for token in tokens:
                yield token
            return
        
        # Try fallback prompts
        for i, fallback in enumerate(fallback_prompts):
            logger.info(f"Trying fallback prompt {i+1}")
            tokens = list(self.generate_stream(fallback, seed + i + 1))
            if self._is_valid_generation(tokens):
                for token in tokens:
                    yield token
                return
        
        # All attempts failed
        yield "[All generation attempts failed]"
    
    def _validate_token(self, token: str) -> bool:
        """Validate a token."""
        # Basic validation
        if not isinstance(token, str):
            return False
        
        # Custom validators
        for validator in self.token_validators:
            if not validator(token):
                return False
        
        return True
    
    def _is_valid_generation(self, tokens: List[str]) -> bool:
        """Check if generation is valid."""
        if not tokens:
            return False
        
        content = "".join(tokens)
        
        # Check for error markers
        if any(marker in content for marker in ["[Error", "[Timeout", "[Stalled", "[No content"]):
            return False
        
        # Check minimum length
        if len(content.strip()) < 10:
            return False
        
        return True
    
    def _handle_error(self, error: str, prompt: str, tokens_generated: int):
        """Handle streaming error."""
        error_info = {
            "error": error,
            "prompt": prompt,
            "tokens_generated": tokens_generated,
            "timestamp": time.time()
        }
        
        logger.error(f"Streaming error: {error_info}")
        
        for handler in self.error_handlers:
            try:
                handler(error_info)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")

# Usage example
streamer = RobustStreamer(timeout=20.0, max_tokens=300)

# Add custom token validator
streamer.add_token_validator(lambda token: len(token) < 100)

# Add error handler
streamer.on_error(lambda info: print(f"Error handled: {info['error']}"))

# Stream with error handling
print("Streaming with error handling:")
for token in streamer.generate_stream("Write a story", seed=42):
    print(token, end="", flush=True)
print()

# Async generation
print("\nAsync generation:")
thread = streamer.generate_async(
    "Explain quantum computing",
    seed=123,
    callback=lambda token: print(token, end="", flush=True)
)

# Wait for completion
thread.join()
try:
    status, result = thread.result_queue.get(timeout=1)
    print(f"\nAsync result: {status}")
except Empty:
    print("\nAsync generation did not complete")

# Generation with fallbacks
print("\nGeneration with fallbacks:")
for token in streamer.generate_with_fallback("Complex prompt that might fail", seed=456):
    print(token, end="", flush=True)
print()
```

## Daemon Error Handling

### Basic Daemon Error Handling

```python
import steadytext
from steadytext.daemon import use_daemon
from steadytext.daemon.client import is_daemon_running
import subprocess
import time

def ensure_daemon_running(max_retries: int = 3) -> bool:
    """Ensure daemon is running with retries."""
    for attempt in range(max_retries):
        if is_daemon_running():
            return True
        
        logger.info(f"Daemon not running, attempting to start (attempt {attempt + 1}/{max_retries})")
        
        try:
            # Start daemon
            result = subprocess.run(
                ["st", "daemon", "start"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Wait for daemon to be ready
                time.sleep(2)
                if is_daemon_running():
                    logger.info("Daemon started successfully")
                    return True
            else:
                logger.error(f"Failed to start daemon: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("Daemon start timed out")
        except Exception as e:
            logger.error(f"Error starting daemon: {e}")
        
        time.sleep(1)
    
    return False

def generate_with_daemon_fallback(prompt: str, seed: int = 42) -> Optional[str]:
    """Generate with automatic daemon fallback."""
    try:
        # Try with daemon first
        with use_daemon():
            return steadytext.generate(prompt, seed=seed)
    except Exception as e:
        logger.warning(f"Daemon generation failed: {e}, falling back to direct mode")
        
        # Fall back to direct generation
        try:
            return steadytext.generate(prompt, seed=seed)
        except Exception as e2:
            logger.error(f"Direct generation also failed: {e2}")
            return None

# Usage
if ensure_daemon_running():
    result = generate_with_daemon_fallback("Hello world", seed=42)
    if result:
        print(result)
    else:
        print("Generation failed")
else:
    print("Could not start daemon, using direct mode")
    result = steadytext.generate("Hello world", seed=42)
```

### Advanced Daemon Error Handling

```python
import steadytext
from steadytext.daemon import use_daemon
from steadytext.daemon.client import DaemonClient, is_daemon_running
import time
import threading
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class DaemonHealth:
    """Daemon health status."""
    is_running: bool
    response_time: Optional[float]
    last_check: datetime
    consecutive_failures: int
    error_message: Optional[str] = None

class ResilientDaemonClient:
    """Resilient daemon client with comprehensive error handling."""
    
    def __init__(self,
                 health_check_interval: int = 60,
                 max_consecutive_failures: int = 5,
                 auto_restart: bool = True):
        self.health_check_interval = health_check_interval
        self.max_consecutive_failures = max_consecutive_failures
        self.auto_restart = auto_restart
        self.health = DaemonHealth(
            is_running=False,
            response_time=None,
            last_check=datetime.now(),
            consecutive_failures=0
        )
        self._health_check_thread = None
        self._stop_health_check = threading.Event()
        self._fallback_mode = False
        self._callbacks = {
            "daemon_down": [],
            "daemon_up": [],
            "daemon_slow": [],
            "fallback_activated": []
        }
    
    def on(self, event: str, callback: Callable):
        """Register event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def start_monitoring(self):
        """Start health monitoring thread."""
        if self._health_check_thread is None or not self._health_check_thread.is_alive():
            self._stop_health_check.clear()
            self._health_check_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True
            )
            self._health_check_thread.start()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._stop_health_check.set()
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
    
    def generate(self, prompt: str, seed: int = 42, **kwargs) -> Optional[str]:
        """Generate with comprehensive error handling."""
        # Check if we should use fallback mode
        if self._fallback_mode or not self.health.is_running:
            return self._fallback_generate(prompt, seed, **kwargs)
        
        try:
            # Attempt daemon generation
            start_time = time.time()
            
            with use_daemon():
                result = steadytext.generate(prompt, seed=seed, **kwargs)
            
            # Update response time
            response_time = time.time() - start_time
            self._update_health(True, response_time)
            
            # Check for slow responses
            if response_time > 5.0:
                self._trigger_event("daemon_slow", {"response_time": response_time})
            
            return result
            
        except Exception as e:
            logger.error(f"Daemon generation error: {e}")
            self._update_health(False, error=str(e))
            
            # Check if we should switch to fallback mode
            if self.health.consecutive_failures >= self.max_consecutive_failures:
                self._activate_fallback_mode()
            
            # Try direct generation
            return self._fallback_generate(prompt, seed, **kwargs)
    
    def embed(self, text: str, seed: int = 42) -> Optional[np.ndarray]:
        """Embed with comprehensive error handling."""
        # Similar implementation to generate
        pass
    
    def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while not self._stop_health_check.is_set():
            try:
                self._perform_health_check()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            # Wait for next check
            self._stop_health_check.wait(self.health_check_interval)
    
    def _perform_health_check(self):
        """Perform daemon health check."""
        try:
            start_time = time.time()
            
            # Check if daemon is running
            if not is_daemon_running():
                self._update_health(False, error="Daemon not running")
                
                if self.auto_restart and self.health.consecutive_failures > 0:
                    self._attempt_restart()
                return
            
            # Test daemon responsiveness
            client = DaemonClient()
            response = client._send_request({"type": "ping"})
            
            if response and response.get("success"):
                response_time = time.time() - start_time
                self._update_health(True, response_time)
                
                # Check if we can deactivate fallback mode
                if self._fallback_mode and self.health.consecutive_failures == 0:
                    self._deactivate_fallback_mode()
            else:
                self._update_health(False, error="Ping failed")
                
        except Exception as e:
            self._update_health(False, error=str(e))
    
    def _update_health(self, success: bool, response_time: Optional[float] = None,
                      error: Optional[str] = None):
        """Update health status."""
        self.health.last_check = datetime.now()
        
        if success:
            self.health.is_running = True
            self.health.response_time = response_time
            self.health.error_message = None
            
            if self.health.consecutive_failures > 0:
                # Daemon recovered
                self.health.consecutive_failures = 0
                self._trigger_event("daemon_up", {"response_time": response_time})
        else:
            self.health.consecutive_failures += 1
            self.health.error_message = error
            
            if self.health.consecutive_failures == 1:
                # Daemon just went down
                self._trigger_event("daemon_down", {"error": error})
            
            if self.health.consecutive_failures >= self.max_consecutive_failures:
                self.health.is_running = False
    
    def _activate_fallback_mode(self):
        """Activate fallback mode."""
        if not self._fallback_mode:
            self._fallback_mode = True
            logger.warning("Activating fallback mode due to daemon failures")
            self._trigger_event("fallback_activated", {
                "consecutive_failures": self.health.consecutive_failures
            })
    
    def _deactivate_fallback_mode(self):
        """Deactivate fallback mode."""
        if self._fallback_mode:
            self._fallback_mode = False
            logger.info("Deactivating fallback mode - daemon recovered")
    
    def _fallback_generate(self, prompt: str, seed: int = 42, **kwargs) -> Optional[str]:
        """Fallback generation without daemon."""
        try:
            logger.info("Using direct generation (fallback mode)")
            return steadytext.generate(prompt, seed=seed, **kwargs)
        except Exception as e:
            logger.error(f"Fallback generation error: {e}")
            return None
    
    def _attempt_restart(self):
        """Attempt to restart daemon."""
        logger.info("Attempting to restart daemon")
        
        try:
            # Stop daemon if running
            subprocess.run(["st", "daemon", "stop"], timeout=5)
            time.sleep(1)
            
            # Start daemon
            result = subprocess.run(
                ["st", "daemon", "start"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("Daemon restart successful")
                time.sleep(2)  # Wait for startup
            else:
                logger.error(f"Daemon restart failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Error restarting daemon: {e}")
    
    def _trigger_event(self, event: str, data: Dict[str, Any]):
        """Trigger event callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in {event} callback: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "health": {
                "is_running": self.health.is_running,
                "response_time": self.health.response_time,
                "last_check": self.health.last_check.isoformat(),
                "consecutive_failures": self.health.consecutive_failures,
                "error_message": self.health.error_message
            },
            "fallback_mode": self._fallback_mode,
            "monitoring": self._health_check_thread.is_alive() if self._health_check_thread else False
        }

# Usage example
client = ResilientDaemonClient(auto_restart=True)

# Register event handlers
client.on("daemon_down", lambda data: print(f"Daemon down: {data}"))
client.on("daemon_up", lambda data: print(f"Daemon recovered: {data}"))
client.on("daemon_slow", lambda data: print(f"Slow response: {data}"))
client.on("fallback_activated", lambda data: print(f"Fallback mode: {data}"))

# Start monitoring
client.start_monitoring()

# Use with automatic error handling
result = client.generate("Write a poem", seed=42)
if result:
    print(result)
else:
    print("Generation failed")

# Check status
status = client.get_status()
print(f"Client status: {status}")

# Stop monitoring when done
client.stop_monitoring()
```

## CLI Error Handling

### Shell Script Error Handling

```bash
#!/bin/bash
# robust_cli.sh - Robust CLI usage with error handling

set -euo pipefail  # Exit on error, undefined variable, pipe failure

# Error handling function
handle_error() {
    local exit_code=$?
    local line_number=$1
    echo "Error on line $line_number: Command exited with status $exit_code" >&2
    
    # Log error
    echo "[$(date)] Error on line $line_number, exit code $exit_code" >> steadytext_errors.log
    
    # Cleanup if needed
    cleanup
    
    exit $exit_code
}

# Set error trap
trap 'handle_error ${LINENO}' ERR

# Cleanup function
cleanup() {
    # Remove temporary files
    rm -f /tmp/steadytext_temp_*
}

# Function to safely generate text
safe_generate() {
    local prompt="$1"
    local seed="${2:-42}"
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if result=$(st generate "$prompt" --seed "$seed" --json 2>/dev/null); then
            # Extract text from JSON
            if text=$(echo "$result" | jq -r '.text' 2>/dev/null); then
                echo "$text"
                return 0
            else
                echo "Error: Invalid JSON response" >&2
            fi
        else
            echo "Error: Generation failed (attempt $((retry_count + 1))/$max_retries)" >&2
        fi
        
        retry_count=$((retry_count + 1))
        sleep 1
    done
    
    return 1
}

# Function to check daemon status
check_daemon() {
    if st daemon status >/dev/null 2>&1; then
        echo "Daemon is running"
        return 0
    else
        echo "Daemon is not running"
        return 1
    fi
}

# Function to ensure daemon is running
ensure_daemon() {
    if ! check_daemon; then
        echo "Starting daemon..."
        if st daemon start; then
            sleep 2
            if check_daemon; then
                echo "Daemon started successfully"
                return 0
            fi
        fi
        echo "Failed to start daemon" >&2
        return 1
    fi
    return 0
}

# Main script
main() {
    echo "SteadyText Robust CLI Example"
    echo "============================="
    
    # Ensure daemon is running (optional)
    if ensure_daemon; then
        echo "Using daemon mode"
    else
        echo "Using direct mode"
    fi
    
    # Generate with error handling
    echo -e "\nGenerating text..."
    if text=$(safe_generate "Write a haiku about error handling" 123); then
        echo "Generated text:"
        echo "$text"
    else
        echo "Failed to generate text"
        exit 1
    fi
    
    # Batch processing with error handling
    echo -e "\nBatch processing..."
    prompts=("Task 1" "Task 2" "Task 3")
    
    for i in "${!prompts[@]}"; do
        prompt="${prompts[$i]}"
        echo -n "Processing '$prompt': "
        
        if result=$(safe_generate "$prompt" $((100 + i))); then
            echo "Success"
            echo "$result" > "output_$i.txt"
        else
            echo "Failed"
            # Continue with next prompt instead of exiting
        fi
    done
    
    echo -e "\nCompleted successfully"
}

# Run main function
main "$@"
```

### Python CLI Wrapper

```python
#!/usr/bin/env python3
"""
robust_cli_wrapper.py - Robust wrapper for SteadyText CLI
"""

import subprocess
import json
import time
import sys
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SteadyTextCLI:
    """Robust wrapper for SteadyText CLI with error handling."""
    
    def __init__(self, 
                 timeout: int = 30,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.daemon_checked = False
        self.daemon_available = False
    
    def _run_command(self, cmd: List[str], input_text: Optional[str] = None) -> subprocess.CompletedProcess:
        """Run CLI command with timeout and error handling."""
        try:
            result = subprocess.run(
                cmd,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return result
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            raise
        except Exception as e:
            logger.error(f"Command error: {e}")
            raise
    
    def check_daemon(self) -> bool:
        """Check if daemon is running."""
        if not self.daemon_checked:
            try:
                result = self._run_command(["st", "daemon", "status", "--json"])
                if result.returncode == 0:
                    status = json.loads(result.stdout)
                    self.daemon_available = status.get("running", False)
            except:
                self.daemon_available = False
            
            self.daemon_checked = True
            logger.info(f"Daemon available: {self.daemon_available}")
        
        return self.daemon_available
    
    def generate(self, prompt: str, seed: int = 42, **kwargs) -> Optional[str]:
        """Generate text with error handling and retries."""
        cmd = ["st", "generate", prompt, "--seed", str(seed), "--json", "--wait"]
        
        # Add additional options
        if "max_new_tokens" in kwargs:
            cmd.extend(["--max-new-tokens", str(kwargs["max_new_tokens"])])
        
        for attempt in range(self.max_retries):
            try:
                result = self._run_command(cmd)
                
                if result.returncode == 0:
                    # Parse JSON response
                    try:
                        data = json.loads(result.stdout)
                        return data.get("text")
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON response: {result.stdout}")
                else:
                    logger.error(f"Generation failed: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                logger.error(f"Generation timed out (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"Generation error: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        return None
    
    def generate_stream(self, prompt: str, seed: int = 42, callback=None) -> bool:
        """Stream generation with error handling."""
        cmd = ["st", "generate", prompt, "--seed", str(seed)]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Read output character by character
            while True:
                char = process.stdout.read(1)
                if not char:
                    break
                
                if callback:
                    callback(char)
                else:
                    print(char, end="", flush=True)
            
            # Wait for process to complete
            process.wait(timeout=self.timeout)
            
            return process.returncode == 0
            
        except subprocess.TimeoutExpired:
            logger.error("Streaming generation timed out")
            process.kill()
            return False
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            return False
    
    def embed(self, text: str, seed: int = 42) -> Optional[List[float]]:
        """Generate embedding with error handling."""
        cmd = ["st", "embed", text, "--seed", str(seed), "--format", "json"]
        
        for attempt in range(self.max_retries):
            try:
                result = self._run_command(cmd)
                
                if result.returncode == 0:
                    # Parse JSON array
                    try:
                        embedding = json.loads(result.stdout)
                        if isinstance(embedding, list) and len(embedding) == 1024:
                            return embedding
                        else:
                            logger.error("Invalid embedding format")
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON embedding")
                else:
                    logger.error(f"Embedding failed: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Embedding error: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        return None
    
    def batch_generate(self, prompts: List[str], seeds: Optional[List[int]] = None) -> List[Optional[str]]:
        """Batch generate with parallel processing."""
        import concurrent.futures
        
        if seeds is None:
            seeds = [42 + i for i in range(len(prompts))]
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.generate, prompt, seed)
                for prompt, seed in zip(prompts, seeds)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch generation error: {e}")
                    results.append(None)
        
        return results

def main():
    """Example usage of robust CLI wrapper."""
    cli = SteadyTextCLI()
    
    # Check daemon
    if cli.check_daemon():
        print("✓ Daemon is running")
    else:
        print("✗ Daemon not available, using direct mode")
    
    # Single generation
    print("\nGenerating text...")
    text = cli.generate("Write a short poem", seed=123)
    if text:
        print(f"Generated: {text}")
    else:
        print("Generation failed")
    
    # Streaming generation
    print("\nStreaming generation...")
    success = cli.generate_stream("Tell me a story", seed=456)
    print(f"\nStreaming {'succeeded' if success else 'failed'}")
    
    # Batch generation
    print("\nBatch generation...")
    prompts = ["Task 1", "Task 2", "Task 3"]
    results = cli.batch_generate(prompts)
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        status = "✓" if result else "✗"
        print(f"{status} {prompt}: {result[:50] if result else 'Failed'}...")

if __name__ == "__main__":
    main()
```

## Production Patterns

### Circuit Breaker Pattern

```python
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
import threading

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker for SteadyText operations."""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset circuit."""
        return (self.last_failure_time and
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout))
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
            }

# Usage with SteadyText
circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

def protected_generate(prompt: str, seed: int = 42) -> Optional[str]:
    """Generate with circuit breaker protection."""
    try:
        return circuit_breaker.call(steadytext.generate, prompt, seed=seed)
    except Exception as e:
        logger.error(f"Circuit breaker triggered: {e}")
        return None
```

### Retry with Exponential Backoff

```python
import time
import random
from typing import TypeVar, Callable, Optional, Any

T = TypeVar('T')

def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> Callable[..., Optional[T]]:
    """Decorator for retry with exponential backoff."""
    
    def wrapper(*args, **kwargs) -> Optional[T]:
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed: {e}")
                    break
                
                # Calculate delay with exponential backoff
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                
                # Add jitter
                if jitter:
                    delay = delay * (0.5 + random.random())
                
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
        
        return None
    
    return wrapper

# Apply to SteadyText functions
@retry_with_backoff
def robust_generate(prompt: str, seed: int = 42) -> Optional[str]:
    """Generate with automatic retry."""
    result = steadytext.generate(prompt, seed=seed)
    if result is None:
        raise Exception("Generation returned None")
    return result

@retry_with_backoff
def robust_embed(text: str, seed: int = 42) -> Optional[np.ndarray]:
    """Embed with automatic retry."""
    result = steadytext.embed(text, seed=seed)
    if result is None:
        raise Exception("Embedding returned None")
    return result
```

## Monitoring and Alerting

### Error Monitoring System

```python
import time
import json
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional
import smtplib
from email.mime.text import MIMEText

class ErrorMonitor:
    """Comprehensive error monitoring for SteadyText."""
    
    def __init__(self,
                 window_size: int = 1000,
                 alert_threshold: int = 10,
                 alert_window: int = 300):  # 5 minutes
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.alert_window = alert_window
        self.errors = deque(maxlen=window_size)
        self.error_counts = defaultdict(int)
        self.alert_sent = {}
        self.metrics = {
            "total_errors": 0,
            "errors_by_type": defaultdict(int),
            "errors_by_hour": defaultdict(int),
            "recent_error_rate": 0.0
        }
    
    def record_error(self, error_type: str, error_message: str,
                    context: Optional[Dict[str, Any]] = None):
        """Record an error occurrence."""
        error = {
            "timestamp": datetime.now(),
            "type": error_type,
            "message": error_message,
            "context": context or {}
        }
        
        self.errors.append(error)
        self.error_counts[error_type] += 1
        self.metrics["total_errors"] += 1
        self.metrics["errors_by_type"][error_type] += 1
        
        # Update hourly metrics
        hour = datetime.now().strftime("%Y-%m-%d %H:00")
        self.metrics["errors_by_hour"][hour] += 1
        
        # Check if alert needed
        self._check_alert_condition(error_type)
    
    def _check_alert_condition(self, error_type: str):
        """Check if we should send an alert."""
        # Count recent errors of this type
        cutoff_time = datetime.now() - timedelta(seconds=self.alert_window)
        recent_errors = sum(
            1 for error in self.errors
            if error["type"] == error_type and error["timestamp"] > cutoff_time
        )
        
        # Check threshold
        if recent_errors >= self.alert_threshold:
            last_alert = self.alert_sent.get(error_type)
            if not last_alert or datetime.now() - last_alert > timedelta(seconds=self.alert_window):
                self._send_alert(error_type, recent_errors)
                self.alert_sent[error_type] = datetime.now()
    
    def _send_alert(self, error_type: str, error_count: int):
        """Send alert notification."""
        message = f"""
        SteadyText Error Alert
        
        Error Type: {error_type}
        Count: {error_count} errors in last {self.alert_window} seconds
        Threshold: {self.alert_threshold}
        Time: {datetime.now().isoformat()}
        
        Recent errors:
        """
        
        # Add recent errors
        recent = [e for e in self.errors if e["type"] == error_type][-5:]
        for error in recent:
            message += f"\n- {error['timestamp']}: {error['message']}"
        
        logger.critical(f"ALERT: {message}")
        
        # Implement your alert mechanism here
        # e.g., send email, Slack, PagerDuty, etc.
    
    def get_error_rate(self, window_seconds: int = 60) -> float:
        """Calculate error rate in errors per second."""
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_errors = sum(
            1 for error in self.errors
            if error["timestamp"] > cutoff_time
        )
        return recent_errors / window_seconds
    
    def get_report(self) -> Dict[str, Any]:
        """Generate error report."""
        return {
            "summary": {
                "total_errors": self.metrics["total_errors"],
                "unique_error_types": len(self.error_counts),
                "error_rate_per_minute": self.get_error_rate(60) * 60,
                "most_common_error": max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
            },
            "errors_by_type": dict(self.metrics["errors_by_type"]),
            "recent_errors": [
                {
                    "timestamp": e["timestamp"].isoformat(),
                    "type": e["type"],
                    "message": e["message"]
                }
                for e in list(self.errors)[-10:]
            ],
            "alerts_sent": {
                error_type: timestamp.isoformat()
                for error_type, timestamp in self.alert_sent.items()
            }
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_report(), f, indent=2)

# Global error monitor
error_monitor = ErrorMonitor(alert_threshold=5, alert_window=300)

# Integration with SteadyText operations
def monitored_generate(prompt: str, seed: int = 42) -> Optional[str]:
    """Generate with error monitoring."""
    try:
        result = steadytext.generate(prompt, seed=seed)
        
        if result is None:
            error_monitor.record_error(
                "generation_failed",
                "Generation returned None",
                {"prompt": prompt[:50], "seed": seed}
            )
        
        return result
        
    except Exception as e:
        error_monitor.record_error(
            "generation_exception",
            str(e),
            {"prompt": prompt[:50], "seed": seed}
        )
        return None
```

## Recovery Strategies

### Graceful Degradation

```python
class GracefulDegradationManager:
    """Manage graceful degradation strategies."""
    
    def __init__(self):
        self.degradation_levels = {
            0: "full_service",
            1: "reduced_quality",
            2: "cached_only",
            3: "static_responses",
            4: "maintenance_mode"
        }
        self.current_level = 0
        self.level_thresholds = {
            "error_rate": [0.1, 0.3, 0.5, 0.7, 0.9],
            "response_time": [2.0, 5.0, 10.0, 20.0, 30.0]
        }
    
    def evaluate_service_health(self, metrics: Dict[str, float]) -> int:
        """Evaluate service health and determine degradation level."""
        error_rate = metrics.get("error_rate", 0.0)
        response_time = metrics.get("response_time", 0.0)
        
        # Determine level based on metrics
        level = 0
        for i, (error_threshold, time_threshold) in enumerate(
            zip(self.level_thresholds["error_rate"], 
                self.level_thresholds["response_time"])
        ):
            if error_rate > error_threshold or response_time > time_threshold:
                level = i + 1
        
        return min(level, 4)
    
    def apply_degradation_strategy(self, level: int, operation: str, **kwargs) -> Any:
        """Apply appropriate degradation strategy."""
        self.current_level = level
        strategy = self.degradation_levels[level]
        
        logger.info(f"Applying degradation strategy: {strategy}")
        
        if strategy == "full_service":
            return self._full_service(operation, **kwargs)
        elif strategy == "reduced_quality":
            return self._reduced_quality(operation, **kwargs)
        elif strategy == "cached_only":
            return self._cached_only(operation, **kwargs)
        elif strategy == "static_responses":
            return self._static_responses(operation, **kwargs)
        else:  # maintenance_mode
            return self._maintenance_mode(operation, **kwargs)
    
    def _full_service(self, operation: str, **kwargs):
        """Normal operation."""
        if operation == "generate":
            return steadytext.generate(**kwargs)
        elif operation == "embed":
            return steadytext.embed(**kwargs)
    
    def _reduced_quality(self, operation: str, **kwargs):
        """Reduced quality but faster."""
        if operation == "generate":
            # Reduce token count
            kwargs["max_new_tokens"] = min(kwargs.get("max_new_tokens", 512), 100)
            return steadytext.generate(**kwargs)
    
    def _cached_only(self, operation: str, **kwargs):
        """Return only cached responses."""
        # Check cache directly
        cache_manager = get_cache_manager()
        # Implement cache-only logic
        return None
    
    def _static_responses(self, operation: str, **kwargs):
        """Return static pre-defined responses."""
        static_responses = {
            "generate": "Service is currently limited. Please try again later.",
            "embed": np.zeros(1024, dtype=np.float32)
        }
        return static_responses.get(operation)
    
    def _maintenance_mode(self, operation: str, **kwargs):
        """System in maintenance mode."""
        return None
```

## Best Practices

### 1. Comprehensive Error Handler

```python
class SteadyTextErrorHandler:
    """Comprehensive error handler for all SteadyText operations."""
    
    def __init__(self):
        self.handlers = {
            "generation": self._handle_generation_error,
            "embedding": self._handle_embedding_error,
            "streaming": self._handle_streaming_error,
            "daemon": self._handle_daemon_error
        }
        self.fallback_strategies = {
            "generation": self._generation_fallback,
            "embedding": self._embedding_fallback
        }
    
    def handle(self, operation: str, error: Any, context: Dict[str, Any]) -> Any:
        """Central error handling."""
        handler = self.handlers.get(operation, self._default_handler)
        return handler(error, context)
    
    def _handle_generation_error(self, error: Any, context: Dict[str, Any]):
        """Handle generation errors."""
        logger.error(f"Generation error: {error}", extra=context)
        
        # Try fallback
        fallback = self.fallback_strategies["generation"]
        return fallback(context)
    
    def _generation_fallback(self, context: Dict[str, Any]) -> str:
        """Generation fallback strategy."""
        prompt = context.get("prompt", "")
        seed = context.get("seed", 42)
        
        # Try different approaches
        approaches = [
            lambda: f"[Unable to generate response for: {prompt[:50]}...]",
            lambda: "[Service temporarily unavailable]",
            lambda: ""
        ]
        
        for approach in approaches:
            try:
                return approach()
            except:
                continue
        
        return "[Critical error]"
```

### 2. Error Context Manager

```python
from contextlib import contextmanager

@contextmanager
def error_handling(operation: str, **context):
    """Context manager for consistent error handling."""
    start_time = time.time()
    try:
        yield
    except Exception as e:
        duration = time.time() - start_time
        
        # Log error with context
        logger.error(f"{operation} failed after {duration:.2f}s", extra={
            "operation": operation,
            "error": str(e),
            "error_type": type(e).__name__,
            "duration": duration,
            **context
        })
        
        # Record in monitoring
        error_monitor.record_error(
            f"{operation}_error",
            str(e),
            context
        )
        
        # Re-raise or handle based on configuration
        if should_reraise(e):
            raise
        else:
            return handle_gracefully(operation, e, context)

# Usage
with error_handling("generation", prompt="test", seed=42):
    result = steadytext.generate("test", seed=42)
```

### 3. Testing Error Scenarios

```python
import unittest
from unittest.mock import patch, MagicMock

class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""
    
    def test_generation_returns_none(self):
        """Test handling when generation returns None."""
        handler = SteadyTextErrorHandler()
        
        with patch('steadytext.generate', return_value=None):
            result = monitored_generate("test prompt", seed=42)
            self.assertIsNone(result)
            
            # Check error was recorded
            report = error_monitor.get_report()
            self.assertGreater(report["summary"]["total_errors"], 0)
    
    def test_daemon_failure_fallback(self):
        """Test daemon failure with fallback."""
        with patch('steadytext.daemon.client.is_daemon_running', return_value=False):
            result = generate_with_daemon_fallback("test", seed=42)
            self.assertIsNotNone(result)  # Should use direct mode
    
    def test_circuit_breaker_opens(self):
        """Test circuit breaker opening after failures."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        def failing_func():
            raise Exception("Test failure")
        
        # First failures
        for _ in range(2):
            with self.assertRaises(Exception):
                breaker.call(failing_func)
        
        # Circuit should be open
        self.assertEqual(breaker.state, CircuitState.OPEN)
        
        # Further calls should fail immediately
        with self.assertRaises(Exception) as ctx:
            breaker.call(failing_func)
        self.assertIn("Circuit breaker is OPEN", str(ctx.exception))
```

This comprehensive guide covers all aspects of error handling in SteadyText, from basic None checks to advanced production patterns like circuit breakers and graceful degradation. The key principle is that SteadyText's "never fail" philosophy requires careful handling of None returns and proper fallback strategies.
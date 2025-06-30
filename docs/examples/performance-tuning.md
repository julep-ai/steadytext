# Performance Tuning Guide

Optimize SteadyText for maximum performance, reduced latency, and efficient resource usage.

## Overview

SteadyText performance optimization focuses on:

- **Daemon mode**: 160x faster first response
- **Cache optimization**: Hit rates up to 95%+
- **Batch processing**: Amortize model loading costs
- **Resource management**: Memory and CPU optimization
- **Concurrent operations**: Thread-safe parallel processing

## Table of Contents

- [Performance Metrics](#performance-metrics)
- [Daemon Optimization](#daemon-optimization)
- [Cache Tuning](#cache-tuning)
- [Model Performance](#model-performance)
- [Batch Processing](#batch-processing)
- [Memory Management](#memory-management)
- [Concurrent Operations](#concurrent-operations)
- [Monitoring and Profiling](#monitoring-and-profiling)
- [Production Optimization](#production-optimization)
- [Benchmarking](#benchmarking)

## Performance Metrics

### Key Metrics to Monitor

```python
import time
import psutil
import steadytext
from steadytext import get_cache_manager
from dataclasses import dataclass
from typing import List, Dict, Any
import statistics

@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    operation: str
    latency_ms: float
    throughput_tps: float
    memory_mb: float
    cache_hit: bool
    cpu_percent: float

class PerformanceMonitor:
    """Monitor SteadyText performance metrics."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.cache_manager = get_cache_manager()
        self.process = psutil.Process()
    
    def measure_operation(self, func, *args, **kwargs):
        """Measure performance of a single operation."""
        # Get initial state
        initial_mem = self.process.memory_info().rss / 1024 / 1024
        initial_cache_stats = self.cache_manager.get_cache_stats()
        
        # Measure CPU
        self.process.cpu_percent()  # Initialize
        
        # Time the operation
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Calculate metrics
        latency_ms = (end_time - start_time) * 1000
        throughput_tps = 1000 / latency_ms
        final_mem = self.process.memory_info().rss / 1024 / 1024
        memory_delta = final_mem - initial_mem
        cpu_percent = self.process.cpu_percent()
        
        # Check cache hit
        final_cache_stats = self.cache_manager.get_cache_stats()
        cache_hit = self._detect_cache_hit(initial_cache_stats, final_cache_stats)
        
        metric = PerformanceMetrics(
            operation=func.__name__,
            latency_ms=latency_ms,
            throughput_tps=throughput_tps,
            memory_mb=memory_delta,
            cache_hit=cache_hit,
            cpu_percent=cpu_percent
        )
        
        self.metrics.append(metric)
        return result, metric
    
    def _detect_cache_hit(self, initial: dict, final: dict) -> bool:
        """Detect if a cache hit occurred."""
        for cache_type in ['generation', 'embedding']:
            initial_hits = initial.get(cache_type, {}).get('hits', 0)
            final_hits = final.get(cache_type, {}).get('hits', 0)
            if final_hits > initial_hits:
                return True
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics:
            return {}
        
        latencies = [m.latency_ms for m in self.metrics]
        throughputs = [m.throughput_tps for m in self.metrics]
        memory_deltas = [m.memory_mb for m in self.metrics]
        cpu_percents = [m.cpu_percent for m in self.metrics]
        cache_hits = sum(1 for m in self.metrics if m.cache_hit)
        
        return {
            'operations': len(self.metrics),
            'cache_hit_rate': cache_hits / len(self.metrics),
            'latency': {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'p95': sorted(latencies)[int(len(latencies) * 0.95)],
                'p99': sorted(latencies)[int(len(latencies) * 0.99)],
            },
            'throughput': {
                'mean': statistics.mean(throughputs),
                'total': sum(throughputs),
            },
            'memory': {
                'total_mb': sum(memory_deltas),
                'mean_mb': statistics.mean(memory_deltas),
            },
            'cpu': {
                'mean_percent': statistics.mean(cpu_percents),
                'max_percent': max(cpu_percents),
            }
        }

# Usage example
monitor = PerformanceMonitor()

# Measure generation performance
for i in range(100):
    prompt = f"Test prompt {i}"
    result, metric = monitor.measure_operation(
        steadytext.generate, 
        prompt, 
        seed=42
    )
    print(f"Latency: {metric.latency_ms:.2f}ms, Cache: {metric.cache_hit}")

# Get summary
summary = monitor.get_summary()
print(f"\nPerformance Summary:")
print(f"Cache hit rate: {summary['cache_hit_rate']:.2%}")
print(f"Mean latency: {summary['latency']['mean']:.2f}ms")
print(f"P95 latency: {summary['latency']['p95']:.2f}ms")
```

## Daemon Optimization

### Startup Performance

```python
import subprocess
import time
from typing import Optional

class DaemonOptimizer:
    """Optimize daemon startup and performance."""
    
    @staticmethod
    def start_daemon_with_profiling():
        """Start daemon with performance profiling."""
        start_time = time.time()
        
        # Start daemon
        result = subprocess.run([
            'st', 'daemon', 'start',
            '--seed', '42'
        ], capture_output=True, text=True)
        
        # Wait for daemon to be ready
        ready = False
        for _ in range(30):  # 30 second timeout
            status = subprocess.run([
                'st', 'daemon', 'status', '--json'
            ], capture_output=True, text=True)
            
            if status.returncode == 0:
                import json
                data = json.loads(status.stdout)
                if data.get('running'):
                    ready = True
                    break
            
            time.sleep(0.1)
        
        startup_time = time.time() - start_time
        print(f"Daemon startup time: {startup_time:.2f}s")
        
        return ready

    @staticmethod
    def benchmark_daemon_vs_direct():
        """Compare daemon vs direct performance."""
        import steadytext
        from steadytext.daemon import use_daemon
        
        prompt = "Benchmark test prompt"
        iterations = 50
        
        # Benchmark direct access
        print("Benchmarking direct access...")
        direct_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = steadytext.generate(prompt, seed=42)
            direct_times.append(time.perf_counter() - start)
        
        # Benchmark daemon access
        print("Benchmarking daemon access...")
        daemon_times = []
        with use_daemon():
            for _ in range(iterations):
                start = time.perf_counter()
                _ = steadytext.generate(prompt, seed=42)
                daemon_times.append(time.perf_counter() - start)
        
        # Calculate statistics
        direct_avg = sum(direct_times) / len(direct_times) * 1000
        daemon_avg = sum(daemon_times) / len(daemon_times) * 1000
        speedup = direct_avg / daemon_avg
        
        print(f"\nResults:")
        print(f"Direct access: {direct_avg:.2f}ms average")
        print(f"Daemon access: {daemon_avg:.2f}ms average")
        print(f"Speedup: {speedup:.1f}x")
        
        # First response comparison
        print(f"\nFirst response:")
        print(f"Direct: {direct_times[0]*1000:.2f}ms")
        print(f"Daemon: {daemon_times[0]*1000:.2f}ms")
        print(f"First response speedup: {direct_times[0]/daemon_times[0]:.1f}x")
```

### Connection Pooling

```python
import zmq
from contextlib import contextmanager
from threading import Lock
from typing import Dict, Any

class DaemonConnectionPool:
    """Connection pool for daemon clients."""
    
    def __init__(self, host='127.0.0.1', port=5557, pool_size=10):
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self.connections = []
        self.available = []
        self.lock = Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool."""
        context = zmq.Context()
        for _ in range(self.pool_size):
            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://{self.host}:{self.port}")
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
            self.connections.append(socket)
            self.available.append(socket)
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        socket = None
        try:
            with self.lock:
                if self.available:
                    socket = self.available.pop()
            
            if socket is None:
                raise RuntimeError("No connections available")
            
            yield socket
            
        finally:
            if socket:
                with self.lock:
                    self.available.append(socket)
    
    def execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute request using pooled connection."""
        import json
        
        with self.get_connection() as socket:
            socket.send_json(request)
            response = socket.recv_json()
            return response

# Usage
pool = DaemonConnectionPool(pool_size=20)

# Concurrent requests
import concurrent.futures

def make_request(i):
    request = {
        'type': 'generate',
        'prompt': f'Test {i}',
        'seed': 42
    }
    return pool.execute_request(request)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(make_request, i) for i in range(100)]
    results = [f.result() for f in futures]
```

## Cache Tuning

### Optimal Cache Configuration

```python
import os
from typing import Dict, Tuple

class CacheTuner:
    """Tune cache settings for optimal performance."""
    
    @staticmethod
    def calculate_optimal_settings(
        available_memory_mb: float,
        expected_qps: float,
        avg_prompt_length: int,
        cache_for_hours: float = 24
    ) -> Dict[str, Dict[str, float]]:
        """Calculate optimal cache settings based on workload."""
        
        # Estimate cache entry sizes
        gen_entry_size_kb = 2 + (avg_prompt_length * 0.001)  # Rough estimate
        embed_entry_size_kb = 4.2  # 1024 floats + metadata
        
        # Calculate expected entries
        expected_requests = expected_qps * 3600 * cache_for_hours
        unique_ratio = 0.3  # Assume 30% unique requests
        expected_unique = expected_requests * unique_ratio
        
        # Allocate memory (70% for generation, 30% for embedding)
        gen_memory_mb = available_memory_mb * 0.7
        embed_memory_mb = available_memory_mb * 0.3
        
        # Calculate capacities
        gen_capacity = min(
            int(gen_memory_mb * 1024 / gen_entry_size_kb),
            int(expected_unique * 0.8)  # 80% of expected unique
        )
        
        embed_capacity = min(
            int(embed_memory_mb * 1024 / embed_entry_size_kb),
            int(expected_unique * 0.5)  # 50% of expected unique
        )
        
        return {
            'generation': {
                'capacity': gen_capacity,
                'max_size_mb': gen_memory_mb
            },
            'embedding': {
                'capacity': embed_capacity,
                'max_size_mb': embed_memory_mb
            }
        }
    
    @staticmethod
    def apply_settings(settings: Dict[str, Dict[str, float]]):
        """Apply cache settings via environment variables."""
        os.environ['STEADYTEXT_GENERATION_CACHE_CAPACITY'] = str(
            int(settings['generation']['capacity'])
        )
        os.environ['STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB'] = str(
            settings['generation']['max_size_mb']
        )
        os.environ['STEADYTEXT_EMBEDDING_CACHE_CAPACITY'] = str(
            int(settings['embedding']['capacity'])
        )
        os.environ['STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB'] = str(
            settings['embedding']['max_size_mb']
        )
        
        print("Applied cache settings:")
        print(f"Generation: {settings['generation']['capacity']} entries, "
              f"{settings['generation']['max_size_mb']:.1f}MB")
        print(f"Embedding: {settings['embedding']['capacity']} entries, "
              f"{settings['embedding']['max_size_mb']:.1f}MB")

# Example usage
tuner = CacheTuner()

# For a server with 1GB available for caching, expecting 10 QPS
settings = tuner.calculate_optimal_settings(
    available_memory_mb=1024,
    expected_qps=10,
    avg_prompt_length=100,
    cache_for_hours=24
)

tuner.apply_settings(settings)
```

### Cache Warming

```python
import asyncio
from typing import List, Tuple
import steadytext

class CacheWarmer:
    """Warm up caches with common queries."""
    
    def __init__(self, prompts: List[str], seeds: List[int] = None):
        self.prompts = prompts
        self.seeds = seeds or [42]
    
    async def warm_generation_cache(self):
        """Warm generation cache asynchronously."""
        tasks = []
        
        for prompt in self.prompts:
            for seed in self.seeds:
                task = asyncio.create_task(
                    self._generate_async(prompt, seed)
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        successful = sum(1 for r in results if r is not None)
        print(f"Warmed generation cache: {successful}/{len(tasks)} entries")
    
    async def _generate_async(self, prompt: str, seed: int):
        """Generate text asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            steadytext.generate, 
            prompt, 
            seed
        )
    
    def warm_embedding_cache(self):
        """Warm embedding cache."""
        successful = 0
        
        for text in self.prompts:
            for seed in self.seeds:
                try:
                    _ = steadytext.embed(text, seed=seed)
                    successful += 1
                except Exception as e:
                    print(f"Failed to warm embed cache: {e}")
        
        print(f"Warmed embedding cache: {successful}/{len(self.prompts) * len(self.seeds)} entries")

# Common prompts for warming
COMMON_PROMPTS = [
    "Explain machine learning",
    "Write a Python function",
    "What is artificial intelligence?",
    "How does deep learning work?",
    "Summarize this text:",
    "Translate to Spanish:",
    "Generate documentation for",
    "Create a test case",
    "Explain the error:",
    "Optimize this code:"
]

# Warm caches on startup
async def warm_caches():
    warmer = CacheWarmer(COMMON_PROMPTS, seeds=[42, 123, 456])
    await warmer.warm_generation_cache()
    warmer.warm_embedding_cache()

# Run warming
asyncio.run(warm_caches())
```

## Model Performance

### Model Size Selection

```python
from typing import Dict, Any
import time

class ModelBenchmark:
    """Benchmark different model configurations."""
    
    @staticmethod
    def compare_model_sizes():
        """Compare performance of different model sizes."""
        import subprocess
        import json
        
        test_prompts = [
            "Write a short function",
            "Explain quantum computing in simple terms",
            "Generate a creative story about AI"
        ]
        
        results = {}
        
        for size in ['small', 'large']:
            print(f"\nBenchmarking {size} model...")
            size_results = {
                'latencies': [],
                'quality_scores': [],
                'memory_usage': []
            }
            
            for prompt in test_prompts:
                # Measure latency
                start = time.perf_counter()
                result = subprocess.run([
                    'st', 'generate', prompt,
                    '--size', size,
                    '--json',
                    '--wait'
                ], capture_output=True, text=True)
                latency = time.perf_counter() - start
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    size_results['latencies'].append(latency)
                    
                    # Simple quality metric (length and vocabulary)
                    text = data['text']
                    quality = len(set(text.split())) / len(text.split())
                    size_results['quality_scores'].append(quality)
            
            results[size] = size_results
        
        # Print comparison
        print("\nModel Size Comparison:")
        print("-" * 50)
        for size, data in results.items():
            avg_latency = sum(data['latencies']) / len(data['latencies'])
            avg_quality = sum(data['quality_scores']) / len(data['quality_scores'])
            
            print(f"{size.capitalize()} Model:")
            print(f"  Average latency: {avg_latency:.2f}s")
            print(f"  Quality score: {avg_quality:.3f}")
            print(f"  Latency range: {min(data['latencies']):.2f}s - {max(data['latencies']):.2f}s")
        
        return results
```

### Custom Model Configuration

```python
class ModelOptimizer:
    """Optimize model loading and configuration."""
    
    @staticmethod
    def get_optimal_config(use_case: str) -> Dict[str, Any]:
        """Get optimal model configuration for use case."""
        
        configs = {
            'realtime': {
                'model': 'small',
                'n_threads': 4,
                'n_batch': 8,
                'context_length': 512,
                'use_mlock': True,
                'use_mmap': True
            },
            'quality': {
                'model': 'large',
                'n_threads': 8,
                'n_batch': 16,
                'context_length': 2048,
                'use_mlock': True,
                'use_mmap': True
            },
            'batch': {
                'model': 'large',
                'n_threads': 16,
                'n_batch': 32,
                'context_length': 1024,
                'use_mlock': False,
                'use_mmap': True
            }
        }
        
        return configs.get(use_case, configs['realtime'])
    
    @staticmethod
    def optimize_for_hardware():
        """Detect hardware and optimize configuration."""
        import psutil
        
        # Get system info
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Determine optimal settings
        if memory_gb >= 32 and cpu_count >= 16:
            config = {
                'profile': 'high-performance',
                'model': 'large',
                'n_threads': min(cpu_count - 2, 24),
                'cache_size_mb': 2048
            }
        elif memory_gb >= 16 and cpu_count >= 8:
            config = {
                'profile': 'balanced',
                'model': 'large',
                'n_threads': min(cpu_count - 1, 12),
                'cache_size_mb': 1024
            }
        else:
            config = {
                'profile': 'low-resource',
                'model': 'small',
                'n_threads': min(cpu_count, 4),
                'cache_size_mb': 256
            }
        
        print(f"Hardware profile: {config['profile']}")
        print(f"Detected: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
        print(f"Recommended: {config['model']} model, {config['n_threads']} threads")
        
        return config
```

## Batch Processing

### Efficient Batch Operations

```python
from typing import List, Dict, Any
import concurrent.futures
import asyncio

class BatchProcessor:
    """Process multiple requests efficiently."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def process_batch_sync(
        self, 
        prompts: List[str], 
        seeds: List[int] = None
    ) -> List[str]:
        """Process batch synchronously with thread pool."""
        if seeds is None:
            seeds = [42] * len(prompts)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for prompt, seed in zip(prompts, seeds):
                future = executor.submit(steadytext.generate, prompt, seed)
                futures.append(future)
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(f"Error: {e}")
            
            return results
    
    async def process_batch_async(
        self, 
        prompts: List[str],
        seeds: List[int] = None
    ) -> List[str]:
        """Process batch asynchronously."""
        if seeds is None:
            seeds = [42] * len(prompts)
        
        tasks = []
        for prompt, seed in zip(prompts, seeds):
            task = asyncio.create_task(
                self._generate_async(prompt, seed)
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def _generate_async(self, prompt: str, seed: int) -> str:
        """Generate text asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            steadytext.generate,
            prompt,
            seed
        )
    
    def process_streaming_batch(
        self,
        prompts: List[str],
        callback: callable
    ):
        """Process batch with streaming results."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_prompt = {
                executor.submit(steadytext.generate, prompt, 42): prompt
                for prompt in prompts
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    callback(prompt, result, None)
                except Exception as e:
                    callback(prompt, None, e)

# Usage example
processor = BatchProcessor(max_workers=8)

# Sync batch processing
prompts = ["Explain " + topic for topic in ["AI", "ML", "DL", "NLP"]]
results = processor.process_batch_sync(prompts)

# Async batch processing
async def async_example():
    results = await processor.process_batch_async(prompts)
    for prompt, result in zip(prompts, results):
        print(f"{prompt}: {len(result)} chars")

# Streaming results
def handle_result(prompt, result, error):
    if error:
        print(f"Error for '{prompt}': {error}")
    else:
        print(f"Completed '{prompt}': {len(result)} chars")

processor.process_streaming_batch(prompts, handle_result)
```

### Pipeline Optimization

```python
class Pipeline:
    """Optimized processing pipeline."""
    
    def __init__(self):
        self.stages = []
    
    def add_stage(self, func, name=None):
        """Add processing stage."""
        self.stages.append({
            'func': func,
            'name': name or func.__name__
        })
        return self
    
    async def process(self, items: List[Any]) -> List[Any]:
        """Process items through pipeline."""
        current = items
        
        for stage in self.stages:
            print(f"Processing stage: {stage['name']}")
            
            # Process stage in parallel
            tasks = []
            for item in current:
                task = asyncio.create_task(
                    self._process_item(stage['func'], item)
                )
                tasks.append(task)
            
            current = await asyncio.gather(*tasks)
        
        return current
    
    async def _process_item(self, func, item):
        """Process single item."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, item)

# Example: Text processing pipeline
async def text_pipeline_example():
    # Define stages
    def clean_text(text):
        return text.strip().lower()
    
    def generate_summary(text):
        prompt = f"Summarize in one sentence: {text}"
        return steadytext.generate(prompt, seed=42)
    
    def extract_keywords(summary):
        prompt = f"Extract 3 keywords from: {summary}"
        return steadytext.generate(prompt, seed=123)
    
    # Build pipeline
    pipeline = Pipeline()
    pipeline.add_stage(clean_text, "Clean")
    pipeline.add_stage(generate_summary, "Summarize")
    pipeline.add_stage(extract_keywords, "Keywords")
    
    # Process texts
    texts = [
        "Machine learning is transforming industries...",
        "Artificial intelligence enables computers...",
        "Deep learning uses neural networks..."
    ]
    
    results = await pipeline.process(texts)
    return results
```

## Memory Management

### Memory Optimization Strategies

```python
import gc
import tracemalloc
from typing import List, Dict, Any

class MemoryOptimizer:
    """Optimize memory usage for SteadyText operations."""
    
    def __init__(self):
        self.snapshots = []
    
    def start_profiling(self):
        """Start memory profiling."""
        tracemalloc.start()
        self.snapshots = []
    
    def take_snapshot(self, label: str):
        """Take memory snapshot."""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((label, snapshot))
    
    def get_memory_report(self) -> str:
        """Generate memory usage report."""
        if len(self.snapshots) < 2:
            return "Not enough snapshots for comparison"
        
        report = []
        
        for i in range(1, len(self.snapshots)):
            label1, snap1 = self.snapshots[i-1]
            label2, snap2 = self.snapshots[i]
            
            diff = snap2.compare_to(snap1, 'lineno')
            report.append(f"\n{label1} -> {label2}:")
            
            for stat in diff[:10]:  # Top 10 differences
                report.append(f"  {stat}")
        
        return "\n".join(report)
    
    @staticmethod
    def optimize_batch_memory(items: List[Any], batch_size: int = 100):
        """Process items in batches to control memory."""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Process batch
            batch_results = [
                steadytext.generate(item, seed=42)
                for item in batch
            ]
            
            results.extend(batch_results)
            
            # Force garbage collection after each batch
            gc.collect()
        
        return results
    
    @staticmethod
    def memory_efficient_streaming(prompts: List[str]):
        """Memory-efficient streaming generation."""
        for prompt in prompts:
            # Generate and yield immediately
            result = steadytext.generate(prompt, seed=42)
            yield result
            
            # Clear any references
            del result
            
            # Periodic garbage collection
            if prompts.index(prompt) % 100 == 0:
                gc.collect()

# Example usage
optimizer = MemoryOptimizer()
optimizer.start_profiling()

# Take initial snapshot
optimizer.take_snapshot("Initial")

# Generate some text
texts = []
for i in range(1000):
    text = steadytext.generate(f"Test {i}", seed=42)
    texts.append(text)

optimizer.take_snapshot("After 1000 generations")

# Clear and collect
texts.clear()
gc.collect()

optimizer.take_snapshot("After cleanup")

# Get report
print(optimizer.get_memory_report())
```

### Resource Limits

```python
import resource
import signal
from contextlib import contextmanager

class ResourceLimiter:
    """Set resource limits for operations."""
    
    @staticmethod
    @contextmanager
    def limit_memory(max_memory_mb: int):
        """Limit memory usage."""
        # Convert MB to bytes
        max_memory = max_memory_mb * 1024 * 1024
        
        # Set soft and hard limits
        resource.setrlimit(
            resource.RLIMIT_AS,
            (max_memory, max_memory)
        )
        
        try:
            yield
        finally:
            # Reset to unlimited
            resource.setrlimit(
                resource.RLIMIT_AS,
                (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
            )
    
    @staticmethod
    @contextmanager
    def timeout(seconds: int):
        """Set operation timeout."""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        # Set handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

# Usage
limiter = ResourceLimiter()

# Limit memory usage
try:
    with limiter.limit_memory(1024):  # 1GB limit
        # Memory-intensive operation
        results = [
            steadytext.generate(f"Prompt {i}", seed=42)
            for i in range(10000)
        ]
except MemoryError:
    print("Memory limit exceeded")

# Set timeout
try:
    with limiter.timeout(5):  # 5 second timeout
        result = steadytext.generate("Complex prompt", seed=42)
except TimeoutError:
    print("Operation timed out")
```

## Concurrent Operations

### Thread-Safe Operations

```python
import threading
from queue import Queue
from typing import List, Tuple, Any

class ConcurrentProcessor:
    """Thread-safe concurrent processing."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.workers = []
        self.running = False
    
    def start(self):
        """Start worker threads."""
        self.running = True
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker,
                name=f"Worker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """Stop all workers."""
        self.running = False
        
        # Add stop signals
        for _ in range(self.num_workers):
            self.input_queue.put(None)
        
        # Wait for workers
        for worker in self.workers:
            worker.join()
    
    def _worker(self):
        """Worker thread function."""
        while self.running:
            item = self.input_queue.get()
            
            if item is None:
                break
            
            prompt, seed, request_id = item
            
            try:
                result = steadytext.generate(prompt, seed=seed)
                self.output_queue.put((request_id, result, None))
            except Exception as e:
                self.output_queue.put((request_id, None, e))
            
            self.input_queue.task_done()
    
    def process_concurrent(
        self, 
        prompts: List[str], 
        seeds: List[int] = None
    ) -> List[Tuple[int, Any, Any]]:
        """Process prompts concurrently."""
        if seeds is None:
            seeds = [42] * len(prompts)
        
        # Add all items to queue
        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            self.input_queue.put((prompt, seed, i))
        
        # Collect results
        results = []
        for _ in range(len(prompts)):
            result = self.output_queue.get()
            results.append(result)
        
        # Sort by request ID
        results.sort(key=lambda x: x[0])
        
        return results

# Usage
processor = ConcurrentProcessor(num_workers=8)
processor.start()

# Process requests
prompts = [f"Generate text about topic {i}" for i in range(100)]
results = processor.process_concurrent(prompts)

# Check results
successful = sum(1 for _, result, error in results if error is None)
print(f"Processed {successful}/{len(prompts)} successfully")

processor.stop()
```

### Async Concurrency

```python
import asyncio
from typing import List, Dict, Any

class AsyncConcurrentProcessor:
    """Asynchronous concurrent processing."""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.results = {}
    
    async def process_with_limit(
        self,
        prompt: str,
        seed: int,
        request_id: int
    ):
        """Process with concurrency limit."""
        async with self.semaphore:
            result = await self._generate_async(prompt, seed)
            self.results[request_id] = result
            return result
    
    async def _generate_async(self, prompt: str, seed: int) -> str:
        """Async generation wrapper."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            steadytext.generate,
            prompt,
            seed
        )
    
    async def process_batch_limited(
        self,
        prompts: List[str],
        seeds: List[int] = None
    ) -> List[str]:
        """Process batch with concurrency limit."""
        if seeds is None:
            seeds = [42] * len(prompts)
        
        tasks = []
        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            task = self.process_with_limit(prompt, seed, i)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Return results in order
        return [self.results[i] for i in range(len(prompts))]

# Usage
async def concurrent_example():
    processor = AsyncConcurrentProcessor(max_concurrent=20)
    
    # Generate 100 prompts
    prompts = [f"Explain concept {i}" for i in range(100)]
    
    start_time = asyncio.get_event_loop().time()
    results = await processor.process_batch_limited(prompts)
    end_time = asyncio.get_event_loop().time()
    
    print(f"Processed {len(results)} prompts in {end_time - start_time:.2f}s")
    print(f"Average: {(end_time - start_time) / len(results):.3f}s per prompt")

# Run
asyncio.run(concurrent_example())
```

## Monitoring and Profiling

### Performance Dashboard

```python
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Any
import threading

@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str]

class PerformanceDashboard:
    """Real-time performance monitoring dashboard."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, Deque[MetricPoint]] = {}
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value."""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.window_size)
            
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            )
            
            self.metrics[name].append(point)
    
    def get_stats(self, name: str, window_seconds: float = 60) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self.lock:
            if name not in self.metrics:
                return {}
            
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            # Filter points within window
            points = [
                p.value for p in self.metrics[name]
                if p.timestamp >= cutoff_time
            ]
            
            if not points:
                return {}
            
            return {
                'count': len(points),
                'mean': sum(points) / len(points),
                'min': min(points),
                'max': max(points),
                'rate': len(points) / window_seconds
            }
    
    def print_dashboard(self):
        """Print performance dashboard."""
        print("\n" + "="*60)
        print("SteadyText Performance Dashboard")
        print("="*60)
        
        for metric_name in sorted(self.metrics.keys()):
            stats = self.get_stats(metric_name)
            if stats:
                print(f"\n{metric_name}:")
                print(f"  Rate: {stats['rate']:.2f}/s")
                print(f"  Mean: {stats['mean']:.2f}")
                print(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")

# Global dashboard instance
dashboard = PerformanceDashboard()

# Instrumented generation function
def monitored_generate(prompt: str, seed: int = 42) -> str:
    """Generate with monitoring."""
    start_time = time.perf_counter()
    
    try:
        result = steadytext.generate(prompt, seed=seed)
        latency = (time.perf_counter() - start_time) * 1000
        
        dashboard.record_metric('generation_latency_ms', latency)
        dashboard.record_metric('generation_success', 1)
        
        return result
    except Exception as e:
        dashboard.record_metric('generation_error', 1)
        raise

# Background monitoring thread
def monitor_system():
    """Monitor system metrics."""
    import psutil
    
    while True:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        dashboard.record_metric('cpu_percent', cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        dashboard.record_metric('memory_percent', memory.percent)
        dashboard.record_metric('memory_mb', memory.used / 1024 / 1024)
        
        # Print dashboard every 10 seconds
        time.sleep(10)
        dashboard.print_dashboard()

# Start monitoring
monitor_thread = threading.Thread(target=monitor_system, daemon=True)
monitor_thread.start()
```

## Production Optimization

### Production Configuration

```python
from typing import Dict, Any
import yaml

class ProductionConfig:
    """Production-optimized configuration."""
    
    @staticmethod
    def generate_config(environment: str = 'production') -> Dict[str, Any]:
        """Generate environment-specific configuration."""
        
        configs = {
            'development': {
                'daemon': {
                    'enabled': False,
                    'host': '127.0.0.1',
                    'port': 5557
                },
                'cache': {
                    'generation_capacity': 256,
                    'generation_max_size_mb': 50,
                    'embedding_capacity': 512,
                    'embedding_max_size_mb': 100
                },
                'models': {
                    'default_size': 'small',
                    'preload': False
                },
                'monitoring': {
                    'enabled': True,
                    'verbose': True
                }
            },
            'staging': {
                'daemon': {
                    'enabled': True,
                    'host': '0.0.0.0',
                    'port': 5557,
                    'workers': 4
                },
                'cache': {
                    'generation_capacity': 1024,
                    'generation_max_size_mb': 200,
                    'embedding_capacity': 2048,
                    'embedding_max_size_mb': 400
                },
                'models': {
                    'default_size': 'large',
                    'preload': True
                },
                'monitoring': {
                    'enabled': True,
                    'verbose': False
                }
            },
            'production': {
                'daemon': {
                    'enabled': True,
                    'host': '0.0.0.0',
                    'port': 5557,
                    'workers': 16,
                    'max_connections': 1000
                },
                'cache': {
                    'generation_capacity': 4096,
                    'generation_max_size_mb': 1024,
                    'embedding_capacity': 8192,
                    'embedding_max_size_mb': 2048
                },
                'models': {
                    'default_size': 'large',
                    'preload': True,
                    'mlock': True
                },
                'monitoring': {
                    'enabled': True,
                    'verbose': False,
                    'metrics_endpoint': '/metrics'
                },
                'security': {
                    'rate_limiting': True,
                    'max_requests_per_minute': 600,
                    'require_auth': True
                }
            }
        }
        
        return configs.get(environment, configs['production'])
    
    @staticmethod
    def save_config(config: Dict[str, Any], filename: str):
        """Save configuration to file."""
        with open(filename, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    @staticmethod
    def apply_config(config: Dict[str, Any]):
        """Apply configuration to environment."""
        import os
        
        # Apply cache settings
        cache = config.get('cache', {})
        os.environ['STEADYTEXT_GENERATION_CACHE_CAPACITY'] = str(
            cache.get('generation_capacity', 256)
        )
        os.environ['STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB'] = str(
            cache.get('generation_max_size_mb', 50)
        )
        
        # Apply model settings
        models = config.get('models', {})
        if models.get('preload'):
            import subprocess
            subprocess.run(['st', 'models', 'preload'], check=True)
        
        print(f"Applied configuration for environment")

# Generate and apply production config
config = ProductionConfig.generate_config('production')
ProductionConfig.save_config(config, 'steadytext-prod.yaml')
ProductionConfig.apply_config(config)
```

### Health Checks

```python
import asyncio
from enum import Enum
from typing import Dict, Any, List

class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthChecker:
    """Production health checking."""
    
    def __init__(self):
        self.checks = {}
    
    async def check_daemon_health(self) -> Dict[str, Any]:
        """Check daemon health."""
        import subprocess
        import json
        
        try:
            result = subprocess.run([
                'st', 'daemon', 'status', '--json'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return {
                    'status': HealthStatus.HEALTHY if data.get('running') else HealthStatus.UNHEALTHY,
                    'details': data
                }
            else:
                return {
                    'status': HealthStatus.UNHEALTHY,
                    'error': 'Daemon not responding'
                }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'error': str(e)
            }
    
    async def check_model_health(self) -> Dict[str, Any]:
        """Check model availability."""
        try:
            # Quick generation test
            start = time.time()
            result = steadytext.generate("health check", seed=42)
            latency = time.time() - start
            
            if result and latency < 5.0:
                status = HealthStatus.HEALTHY
            elif result and latency < 10.0:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
            
            return {
                'status': status,
                'latency': latency,
                'model_loaded': result is not None
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'error': str(e)
            }
    
    async def check_cache_health(self) -> Dict[str, Any]:
        """Check cache health."""
        try:
            cache_manager = get_cache_manager()
            stats = cache_manager.get_cache_stats()
            
            # Check if caches are responsive
            gen_size = stats.get('generation', {}).get('size', 0)
            embed_size = stats.get('embedding', {}).get('size', 0)
            
            return {
                'status': HealthStatus.HEALTHY,
                'generation_cache_size': gen_size,
                'embedding_cache_size': embed_size
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'error': str(e)
            }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        checks = {
            'daemon': self.check_daemon_health(),
            'model': self.check_model_health(),
            'cache': self.check_cache_health()
        }
        
        results = {}
        for name, check in checks.items():
            results[name] = await check
        
        # Overall status
        statuses = [r['status'] for r in results.values()]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED
        
        return {
            'status': overall.value,
            'checks': results,
            'timestamp': time.time()
        }

# Health check endpoint
async def health_endpoint():
    """Health check endpoint for monitoring."""
    checker = HealthChecker()
    result = await checker.run_all_checks()
    
    # Return appropriate HTTP status
    if result['status'] == 'healthy':
        return result, 200
    elif result['status'] == 'degraded':
        return result, 200
    else:
        return result, 503
```

## Benchmarking

### Comprehensive Benchmark Suite

```python
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any

class BenchmarkSuite:
    """Comprehensive performance benchmarking."""
    
    def __init__(self, output_dir: str = "./benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def run_latency_benchmark(self) -> Dict[str, Any]:
        """Benchmark operation latencies."""
        results = {
            'generation': [],
            'embedding': []
        }
        
        # Test different prompt lengths
        prompt_lengths = [10, 50, 100, 500, 1000]
        
        for length in prompt_lengths:
            prompt = " ".join(["word"] * length)
            
            # Generation latency
            start = time.perf_counter()
            _ = steadytext.generate(prompt, seed=42)
            gen_latency = (time.perf_counter() - start) * 1000
            
            # Embedding latency
            start = time.perf_counter()
            _ = steadytext.embed(prompt, seed=42)
            embed_latency = (time.perf_counter() - start) * 1000
            
            results['generation'].append({
                'prompt_length': length,
                'latency_ms': gen_latency
            })
            
            results['embedding'].append({
                'text_length': length,
                'latency_ms': embed_latency
            })
        
        return results
    
    def run_throughput_benchmark(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Benchmark throughput over time."""
        results = {
            'generation': {'requests': 0, 'duration': duration_seconds},
            'embedding': {'requests': 0, 'duration': duration_seconds}
        }
        
        # Generation throughput
        start_time = time.time()
        gen_count = 0
        while time.time() - start_time < duration_seconds / 2:
            _ = steadytext.generate(f"Test {gen_count}", seed=42)
            gen_count += 1
        results['generation']['requests'] = gen_count
        results['generation']['rps'] = gen_count / (duration_seconds / 2)
        
        # Embedding throughput
        start_time = time.time()
        embed_count = 0
        while time.time() - start_time < duration_seconds / 2:
            _ = steadytext.embed(f"Test {embed_count}", seed=42)
            embed_count += 1
        results['embedding']['requests'] = embed_count
        results['embedding']['rps'] = embed_count / (duration_seconds / 2)
        
        return results
    
    def run_cache_benchmark(self) -> Dict[str, Any]:
        """Benchmark cache performance."""
        from steadytext import get_cache_manager
        
        cache_manager = get_cache_manager()
        results = {'before': {}, 'after': {}}
        
        # Clear caches
        cache_manager.clear_all_caches()
        
        # Get initial stats
        results['before'] = cache_manager.get_cache_stats()
        
        # Generate cache misses
        miss_times = []
        for i in range(100):
            start = time.perf_counter()
            _ = steadytext.generate(f"Unique prompt {i}", seed=42)
            miss_times.append((time.perf_counter() - start) * 1000)
        
        # Generate cache hits
        hit_times = []
        for i in range(100):
            start = time.perf_counter()
            _ = steadytext.generate(f"Unique prompt {i}", seed=42)
            hit_times.append((time.perf_counter() - start) * 1000)
        
        # Get final stats
        results['after'] = cache_manager.get_cache_stats()
        
        results['performance'] = {
            'miss_latency_avg': sum(miss_times) / len(miss_times),
            'hit_latency_avg': sum(hit_times) / len(hit_times),
            'speedup': sum(miss_times) / sum(hit_times)
        }
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("Running SteadyText Performance Benchmark Suite...")
        
        results = {
            'timestamp': time.time(),
            'latency': self.run_latency_benchmark(),
            'throughput': self.run_throughput_benchmark(30),
            'cache': self.run_cache_benchmark()
        }
        
        # Save results
        output_file = self.output_dir / f"benchmark_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Benchmark complete. Results saved to {output_file}")
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("Benchmark Summary")
        print("="*60)
        
        # Latency summary
        gen_latencies = [r['latency_ms'] for r in results['latency']['generation']]
        print(f"\nGeneration Latency:")
        print(f"  Min: {min(gen_latencies):.2f}ms")
        print(f"  Max: {max(gen_latencies):.2f}ms")
        print(f"  Avg: {sum(gen_latencies)/len(gen_latencies):.2f}ms")
        
        # Throughput summary
        print(f"\nThroughput:")
        print(f"  Generation: {results['throughput']['generation']['rps']:.2f} req/s")
        print(f"  Embedding: {results['throughput']['embedding']['rps']:.2f} req/s")
        
        # Cache summary
        cache_perf = results['cache']['performance']
        print(f"\nCache Performance:")
        print(f"  Miss latency: {cache_perf['miss_latency_avg']:.2f}ms")
        print(f"  Hit latency: {cache_perf['hit_latency_avg']:.2f}ms")
        print(f"  Speedup: {cache_perf['speedup']:.1f}x")

# Run benchmarks
if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_full_benchmark()
```

## Best Practices

### Performance Checklist

1. **Always use daemon mode** for production deployments
2. **Configure caches** based on workload and available memory
3. **Use appropriate model sizes** - small for real-time, large for quality
4. **Batch operations** when processing multiple items
5. **Monitor performance** continuously in production
6. **Set resource limits** to prevent runaway processes
7. **Use connection pooling** for high-concurrency scenarios
8. **Implement health checks** for production monitoring
9. **Profile regularly** to identify bottlenecks
10. **Optimize for your hardware** - use all available cores

### Quick Optimization Guide

```bash
# 1. Start daemon for 160x faster responses
st daemon start

# 2. Preload models to avoid first-request delay
st models preload

# 3. Configure optimal cache sizes
export STEADYTEXT_GENERATION_CACHE_CAPACITY=2048
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=500

# 4. Use batch processing in your code
# 5. Monitor with built-in tools
st cache --status

# 6. Run benchmarks to validate
python benchmarks/run_all_benchmarks.py --quick
```

### Common Pitfalls

!!! warning "Performance Pitfalls to Avoid"
    - **Not using daemon mode** - 160x slower first requests
    - **Cache thrashing** - Set appropriate capacity limits
    - **Memory leaks** - Use batch processing with cleanup
    - **Thread contention** - Limit concurrent operations
    - **Inefficient prompts** - Keep prompts concise
    - **Ignoring monitoring** - Always track performance metrics
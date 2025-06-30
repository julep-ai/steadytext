# Caching Guide

Learn how to configure and optimize SteadyText's caching system for maximum performance.

## Overview

SteadyText uses a sophisticated frecency cache (frequency + recency) that combines:
- **LRU (Least Recently Used)**: Recent items stay cached
- **Frequency counting**: Popular items are retained longer
- **Disk persistence**: Cache survives restarts
- **Thread safety**: Safe for concurrent access

## Cache Architecture

### Two-Tier Cache System

```
┌─────────────────────────────────────┐
│         Application Layer           │
├─────────────────────────────────────┤
│     Generation Cache    │ Embedding │
│    (256 entries, 50MB) │   Cache   │
│                        │(512, 100MB)│
├─────────────────────────────────────┤
│      SQLite Backend (Thread-Safe)   │
└─────────────────────────────────────┘
```

### Cache Files Location

```python
import steadytext
from pathlib import Path

# Get cache directory
cache_dir = Path.home() / ".cache" / "steadytext" / "caches"
print(f"Cache location: {cache_dir}")

# Cache files
generation_cache = cache_dir / "generation_cache.db"
embedding_cache = cache_dir / "embedding_cache.db"
```

## Configuration

### Environment Variables

```bash
# Generation cache settings
export STEADYTEXT_GENERATION_CACHE_CAPACITY=256      # Max entries
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=50.0  # Max file size

# Embedding cache settings  
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=512       # Max entries
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=100.0  # Max file size

# Disable cache entirely (not recommended)
export STEADYTEXT_DISABLE_CACHE=1
```

### Python Configuration

```python
import os
import steadytext

# Configure before importing/using steadytext
os.environ['STEADYTEXT_GENERATION_CACHE_CAPACITY'] = '1024'
os.environ['STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB'] = '200.0'

# Verify configuration
from steadytext import get_cache_manager
cache_manager = get_cache_manager()
stats = cache_manager.get_cache_stats()
print(f"Generation cache capacity: {stats['generation']['capacity']}")
```

## Cache Management

### Monitoring Cache Performance

```python
from steadytext import get_cache_manager
import time

class CacheMonitor:
    """Monitor cache performance and hit rates."""
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
        self.initial_stats = self.cache_manager.get_cache_stats()
    
    def get_hit_rate(self, cache_type='generation'):
        """Calculate cache hit rate."""
        stats = self.cache_manager.get_cache_stats()[cache_type]
        hits = stats.get('hits', 0)
        misses = stats.get('misses', 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return hits / total * 100
    
    def monitor_operation(self, operation, *args, **kwargs):
        """Monitor a single operation's cache behavior."""
        stats_before = self.cache_manager.get_cache_stats()
        start_time = time.time()
        
        result = operation(*args, **kwargs)
        
        duration = time.time() - start_time
        stats_after = self.cache_manager.get_cache_stats()
        
        # Determine if it was a cache hit
        gen_hits_diff = stats_after['generation']['hits'] - stats_before['generation']['hits']
        emb_hits_diff = stats_after['embedding']['hits'] - stats_before['embedding']['hits']
        
        cache_hit = gen_hits_diff > 0 or emb_hits_diff > 0
        
        return {
            'result': result,
            'duration': duration,
            'cache_hit': cache_hit,
            'stats_delta': {
                'generation_hits': gen_hits_diff,
                'embedding_hits': emb_hits_diff
            }
        }
    
    def print_summary(self):
        """Print cache performance summary."""
        stats = self.cache_manager.get_cache_stats()
        
        print("=== Cache Performance Summary ===")
        for cache_type in ['generation', 'embedding']:
            cache_stats = stats[cache_type]
            hit_rate = self.get_hit_rate(cache_type)
            
            print(f"\n{cache_type.title()} Cache:")
            print(f"  Size: {cache_stats['size']} entries")
            print(f"  Hit Rate: {hit_rate:.1f}%")
            print(f"  Hits: {cache_stats.get('hits', 0)}")
            print(f"  Misses: {cache_stats.get('misses', 0)}")

# Usage example
monitor = CacheMonitor()

# Monitor text generation
result1 = monitor.monitor_operation(
    steadytext.generate, 
    "Write a haiku about caching"
)
print(f"First call: {result1['duration']:.3f}s (cache hit: {result1['cache_hit']})")

# Same prompt - should be cached
result2 = monitor.monitor_operation(
    steadytext.generate, 
    "Write a haiku about caching"
)
print(f"Second call: {result2['duration']:.3f}s (cache hit: {result2['cache_hit']})")

monitor.print_summary()
```

### Cache Warming

```python
import steadytext
from typing import List
import concurrent.futures

def warm_cache_sequential(prompts: List[str], seeds: List[int] = None):
    """Warm cache with common prompts sequentially."""
    if seeds is None:
        seeds = [42]  # Default seed only
    
    warmed = 0
    for prompt in prompts:
        for seed in seeds:
            _ = steadytext.generate(prompt, seed=seed, max_new_tokens=100)
            warmed += 1
    
    return warmed

def warm_cache_parallel(prompts: List[str], seeds: List[int] = None, max_workers: int = 4):
    """Warm cache with parallel generation."""
    if seeds is None:
        seeds = [42]
    
    tasks = [(prompt, seed) for prompt in prompts for seed in seeds]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(steadytext.generate, prompt, seed=seed, max_new_tokens=100)
            for prompt, seed in tasks
        ]
        
        # Wait for all to complete
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Get result to ensure completion
            completed += 1
    
    return completed

# Common prompts to cache
common_prompts = [
    "Write a Python function",
    "Explain this error",
    "Generate test data",
    "Create documentation",
    "Write unit tests",
    "Optimize this code",
    "Review this pull request",
    "Suggest improvements"
]

# Common seeds if using multiple
common_seeds = [42, 100, 200]  # Add your common seeds

# Warm cache
print("Warming cache...")
warmed = warm_cache_parallel(common_prompts, common_seeds)
print(f"Cache warmed with {warmed} entries")

# Verify cache is warm
from steadytext import get_cache_manager
stats = get_cache_manager().get_cache_stats()
print(f"Generation cache size: {stats['generation']['size']}")
```

### Cache Optimization Strategies

```python
import steadytext
from collections import defaultdict
from datetime import datetime, timedelta

class CacheOptimizer:
    """Optimize cache usage patterns."""
    
    def __init__(self):
        self.usage_patterns = defaultdict(lambda: {
            'count': 0,
            'last_used': None,
            'avg_generation_time': 0
        })
    
    def track_usage(self, prompt: str, seed: int, generation_time: float):
        """Track prompt usage patterns."""
        key = f"{prompt}:{seed}"
        pattern = self.usage_patterns[key]
        
        pattern['count'] += 1
        pattern['last_used'] = datetime.now()
        
        # Update average generation time
        avg = pattern['avg_generation_time']
        count = pattern['count']
        pattern['avg_generation_time'] = (avg * (count - 1) + generation_time) / count
    
    def get_cache_priorities(self, top_n: int = 20):
        """Get prompts that should be prioritized for caching."""
        # Score based on frequency and recency
        now = datetime.now()
        scores = []
        
        for key, pattern in self.usage_patterns.items():
            # Frequency score
            freq_score = pattern['count']
            
            # Recency score (higher for more recent)
            if pattern['last_used']:
                age = (now - pattern['last_used']).total_seconds()
                recency_score = 1 / (1 + age / 3600)  # Decay over hours
            else:
                recency_score = 0
            
            # Generation time score (prioritize slow generations)
            time_score = pattern['avg_generation_time']
            
            # Combined score
            score = freq_score * 0.5 + recency_score * 0.3 + time_score * 0.2
            
            scores.append((score, key, pattern))
        
        # Sort by score
        scores.sort(reverse=True)
        
        return scores[:top_n]
    
    def recommend_cache_size(self):
        """Recommend optimal cache size based on usage."""
        total_unique = len(self.usage_patterns)
        frequently_used = sum(1 for p in self.usage_patterns.values() if p['count'] > 5)
        
        # Recommend 1.5x frequently used items + buffer
        recommended = int(frequently_used * 1.5 + 50)
        
        return {
            'total_unique_prompts': total_unique,
            'frequently_used': frequently_used,
            'recommended_size': recommended,
            'current_default': 256
        }

# Example usage
optimizer = CacheOptimizer()

# Simulate usage tracking
import time
test_prompts = [
    ("Write a function to sort a list", 42),
    ("Explain machine learning", 42),
    ("Write a function to sort a list", 42),  # Repeated
    ("Generate test cases", 100),
    ("Write a function to sort a list", 42),  # Popular
]

for prompt, seed in test_prompts:
    start = time.time()
    _ = steadytext.generate(prompt, seed=seed)
    duration = time.time() - start
    optimizer.track_usage(prompt, seed, duration)

# Get optimization recommendations
print("=== Cache Optimization Report ===")

priorities = optimizer.get_cache_priorities(5)
print("\nTop prompts to keep cached:")
for score, key, pattern in priorities:
    prompt, seed = key.rsplit(':', 1)
    print(f"  Score: {score:.2f} - {prompt[:50]}... (seed: {seed})")
    print(f"    Used: {pattern['count']}x, Avg time: {pattern['avg_generation_time']:.3f}s")

recommendations = optimizer.recommend_cache_size()
print(f"\nCache size recommendations:")
print(f"  Total unique: {recommendations['total_unique_prompts']}")
print(f"  Frequently used: {recommendations['frequently_used']}")
print(f"  Recommended size: {recommendations['recommended_size']}")
```

## Advanced Cache Patterns

### Hierarchical Caching

```python
import steadytext
from typing import Dict, Any, Optional
import json
import hashlib

class HierarchicalCache:
    """Implement hierarchical caching for complex workflows."""
    
    def __init__(self):
        self.memory_cache = {}  # Fast in-memory cache
        self.cache_manager = steadytext.get_cache_manager()
    
    def _generate_cache_key(self, category: str, subcategory: str, 
                          prompt: str, seed: int) -> str:
        """Generate hierarchical cache key."""
        components = [category, subcategory, prompt, str(seed)]
        combined = ":".join(components)
        
        # Create hash for consistent key length
        key_hash = hashlib.md5(combined.encode()).hexdigest()
        
        return f"{category}:{subcategory}:{key_hash}"
    
    def get_or_generate(self, category: str, subcategory: str, 
                       prompt: str, seed: int = 42, **kwargs) -> str:
        """Get from cache or generate with hierarchical key."""
        cache_key = self._generate_cache_key(category, subcategory, prompt, seed)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Generate and cache
        result = steadytext.generate(prompt, seed=seed, **kwargs)
        
        # Store in memory cache
        self.memory_cache[cache_key] = result
        
        return result
    
    def preload_category(self, category: str, items: List[Dict[str, Any]]):
        """Preload entire category into cache."""
        loaded = 0
        
        for item in items:
            result = self.get_or_generate(
                category,
                item.get('subcategory', 'default'),
                item['prompt'],
                item.get('seed', 42),
                **item.get('kwargs', {})
            )
            loaded += 1
        
        return loaded
    
    def clear_category(self, category: str):
        """Clear all cache entries for a category."""
        keys_to_remove = [k for k in self.memory_cache if k.startswith(f"{category}:")]
        
        for key in keys_to_remove:
            del self.memory_cache[key]
        
        return len(keys_to_remove)

# Usage example
h_cache = HierarchicalCache()

# Generate with hierarchy
email_subject = h_cache.get_or_generate(
    "emails", 
    "marketing",
    "Write a subject line for Black Friday sale",
    seed=100
)

email_body = h_cache.get_or_generate(
    "emails",
    "marketing", 
    "Write email body for Black Friday sale",
    seed=100
)

# Preload documentation category
docs_to_cache = [
    {
        'subcategory': 'api',
        'prompt': 'Document a REST API endpoint',
        'seed': 42,
        'kwargs': {'max_new_tokens': 200}
    },
    {
        'subcategory': 'functions',
        'prompt': 'Document a Python function',
        'seed': 42,
        'kwargs': {'max_new_tokens': 150}
    }
]

loaded = h_cache.preload_category('documentation', docs_to_cache)
print(f"Preloaded {loaded} documentation templates")
```

### Cache-Aware Generation

```python
import steadytext
from typing import Optional, Tuple
import time

class CacheAwareGenerator:
    """Generator that adapts based on cache state."""
    
    def __init__(self):
        self.cache_manager = steadytext.get_cache_manager()
        self.performance_threshold = 0.1  # 100ms
    
    def is_likely_cached(self, prompt: str, seed: int = 42) -> bool:
        """Check if a prompt is likely cached without generating."""
        # This is a heuristic - actual implementation would need
        # to check cache internals
        stats = self.cache_manager.get_cache_stats()
        
        # Simple heuristic: if we have items in cache and
        # this is a common prompt pattern
        if stats['generation']['size'] > 0:
            common_patterns = ['Write a', 'Explain', 'Create', 'Generate']
            return any(prompt.startswith(p) for p in common_patterns)
        
        return False
    
    def generate_with_fallback(self, primary_prompt: str, 
                             fallback_prompt: Optional[str] = None,
                             seed: int = 42, **kwargs) -> Tuple[str, bool]:
        """Generate with fallback if primary isn't cached."""
        start_time = time.time()
        
        # Try primary prompt
        result = steadytext.generate(primary_prompt, seed=seed, **kwargs)
        duration = time.time() - start_time
        
        # If slow (not cached) and we have fallback
        if duration > self.performance_threshold and fallback_prompt:
            # Check if fallback might be cached
            if self.is_likely_cached(fallback_prompt, seed):
                fallback_result = steadytext.generate(fallback_prompt, seed=seed, **kwargs)
                return fallback_result, True
        
        return result, False
    
    def batch_generate_optimized(self, prompts: List[str], seed: int = 42, **kwargs):
        """Generate batch with cache-aware ordering."""
        results = {}
        timings = {}
        
        # First pass: try all prompts and measure timing
        for prompt in prompts:
            start = time.time()
            result = steadytext.generate(prompt, seed=seed, **kwargs)
            duration = time.time() - start
            
            results[prompt] = result
            timings[prompt] = duration
        
        # Analyze cache performance
        cached_prompts = [p for p, t in timings.items() if t < self.performance_threshold]
        uncached_prompts = [p for p, t in timings.items() if t >= self.performance_threshold]
        
        stats = {
            'total': len(prompts),
            'cached': len(cached_prompts),
            'uncached': len(uncached_prompts),
            'cache_rate': len(cached_prompts) / len(prompts) * 100,
            'avg_cached_time': sum(timings[p] for p in cached_prompts) / len(cached_prompts) if cached_prompts else 0,
            'avg_uncached_time': sum(timings[p] for p in uncached_prompts) / len(uncached_prompts) if uncached_prompts else 0
        }
        
        return results, stats

# Usage
cache_gen = CacheAwareGenerator()

# Single generation with fallback
primary = "Generate a complex analysis of quantum computing applications in cryptography"
fallback = "Explain quantum computing"  # Likely cached

result, used_fallback = cache_gen.generate_with_fallback(
    primary, 
    fallback,
    max_new_tokens=200
)

print(f"Used fallback: {used_fallback}")

# Batch generation with analysis
test_prompts = [
    "Write a Python function",  # Likely cached
    "Explain machine learning",  # Likely cached
    "Analyze the socioeconomic impact of automation on rural communities",  # Unlikely
    "Generate test data",  # Possibly cached
    "Describe the philosophical implications of consciousness in AI systems"  # Unlikely
]

results, stats = cache_gen.batch_generate_optimized(test_prompts, max_new_tokens=100)

print("\n=== Batch Generation Cache Stats ===")
print(f"Total prompts: {stats['total']}")
print(f"Cached: {stats['cached']} ({stats['cache_rate']:.1f}%)")
print(f"Average cached time: {stats['avg_cached_time']:.3f}s")
print(f"Average uncached time: {stats['avg_uncached_time']:.3f}s")
print(f"Speed improvement: {stats['avg_uncached_time'] / stats['avg_cached_time']:.1f}x")
```

### Cache Persistence Patterns

```python
import steadytext
import json
from pathlib import Path
from typing import Dict, List
import pickle

class CachePersistenceManager:
    """Manage cache persistence and restoration."""
    
    def __init__(self, backup_dir: str = "./cache_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.cache_manager = steadytext.get_cache_manager()
    
    def export_cache_metadata(self) -> Dict:
        """Export cache metadata for analysis."""
        stats = self.cache_manager.get_cache_stats()
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'generation_cache': {
                'size': stats['generation']['size'],
                'capacity': stats['generation'].get('capacity', 256),
                'hit_rate': self._calculate_hit_rate(stats['generation'])
            },
            'embedding_cache': {
                'size': stats['embedding']['size'],
                'capacity': stats['embedding'].get('capacity', 512),
                'hit_rate': self._calculate_hit_rate(stats['embedding'])
            }
        }
        
        return metadata
    
    def _calculate_hit_rate(self, cache_stats: Dict) -> float:
        """Calculate cache hit rate."""
        hits = cache_stats.get('hits', 0)
        misses = cache_stats.get('misses', 0)
        total = hits + misses
        
        return (hits / total * 100) if total > 0 else 0.0
    
    def save_cache_state(self, name: str):
        """Save current cache state metadata."""
        metadata = self.export_cache_metadata()
        
        filename = self.backup_dir / f"cache_state_{name}.json"
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Cache state saved to {filename}")
        return filename
    
    def analyze_cache_history(self) -> Dict:
        """Analyze cache performance over time."""
        history_files = list(self.backup_dir.glob("cache_state_*.json"))
        
        if not history_files:
            return {"error": "No cache history found"}
        
        history = []
        for file in sorted(history_files):
            with open(file, 'r') as f:
                data = json.load(f)
                data['filename'] = file.name
                history.append(data)
        
        # Analyze trends
        analysis = {
            'total_snapshots': len(history),
            'date_range': {
                'start': history[0]['timestamp'],
                'end': history[-1]['timestamp']
            },
            'generation_cache_trend': {
                'min_size': min(h['generation_cache']['size'] for h in history),
                'max_size': max(h['generation_cache']['size'] for h in history),
                'avg_hit_rate': sum(h['generation_cache']['hit_rate'] for h in history) / len(history)
            },
            'embedding_cache_trend': {
                'min_size': min(h['embedding_cache']['size'] for h in history),
                'max_size': max(h['embedding_cache']['size'] for h in history),
                'avg_hit_rate': sum(h['embedding_cache']['hit_rate'] for h in history) / len(history)
            }
        }
        
        return analysis

# Usage
persistence = CachePersistenceManager()

# Save current state
persistence.save_cache_state("before_optimization")

# Do some work...
for i in range(10):
    steadytext.generate(f"Test prompt {i}", seed=42)

# Save after work
persistence.save_cache_state("after_batch_generation")

# Analyze history
analysis = persistence.analyze_cache_history()
print("\n=== Cache History Analysis ===")
print(json.dumps(analysis, indent=2))
```

## Cache Performance Tuning

### Benchmark Cache Impact

```python
import steadytext
import time
import statistics
from typing import List, Dict

class CacheBenchmark:
    """Benchmark cache performance impact."""
    
    def __init__(self):
        self.cache_manager = steadytext.get_cache_manager()
    
    def benchmark_single_prompt(self, prompt: str, seed: int = 42, 
                              iterations: int = 10) -> Dict:
        """Benchmark a single prompt with cold and warm cache."""
        # Clear cache for cold start
        self.cache_manager.clear_all_caches()
        
        timings = {
            'cold': [],
            'warm': []
        }
        
        # Cold cache timing (first call)
        start = time.time()
        _ = steadytext.generate(prompt, seed=seed)
        timings['cold'].append(time.time() - start)
        
        # Warm cache timings
        for _ in range(iterations - 1):
            start = time.time()
            _ = steadytext.generate(prompt, seed=seed)
            timings['warm'].append(time.time() - start)
        
        return {
            'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
            'cold_time': timings['cold'][0],
            'warm_avg': statistics.mean(timings['warm']),
            'warm_std': statistics.stdev(timings['warm']) if len(timings['warm']) > 1 else 0,
            'speedup': timings['cold'][0] / statistics.mean(timings['warm'])
        }
    
    def benchmark_cache_sizes(self, test_prompts: List[str], 
                            cache_sizes: List[int]) -> Dict:
        """Benchmark performance with different cache sizes."""
        results = {}
        original_capacity = os.environ.get('STEADYTEXT_GENERATION_CACHE_CAPACITY', '256')
        
        try:
            for size in cache_sizes:
                # Set cache size
                os.environ['STEADYTEXT_GENERATION_CACHE_CAPACITY'] = str(size)
                
                # Restart cache with new size
                # Note: In practice, this would require restarting the process
                self.cache_manager.clear_all_caches()
                
                # Benchmark with this cache size
                hit_count = 0
                total_time = 0
                
                for i, prompt in enumerate(test_prompts):
                    start = time.time()
                    _ = steadytext.generate(prompt, seed=42)
                    duration = time.time() - start
                    total_time += duration
                    
                    # Simple hit detection (fast = hit)
                    if duration < 0.1:
                        hit_count += 1
                
                results[size] = {
                    'hit_rate': hit_count / len(test_prompts) * 100,
                    'avg_time': total_time / len(test_prompts),
                    'total_time': total_time
                }
        
        finally:
            # Restore original capacity
            os.environ['STEADYTEXT_GENERATION_CACHE_CAPACITY'] = original_capacity
        
        return results
    
    def find_optimal_cache_size(self, typical_prompts: List[str]) -> int:
        """Find optimal cache size for typical usage."""
        unique_prompts = len(set(typical_prompts))
        prompt_frequency = {}
        
        for prompt in typical_prompts:
            prompt_frequency[prompt] = prompt_frequency.get(prompt, 0) + 1
        
        # Prompts that appear more than once
        repeated_prompts = sum(1 for count in prompt_frequency.values() if count > 1)
        
        # Recommend size based on usage pattern
        if repeated_prompts / unique_prompts > 0.5:
            # High repetition - smaller cache OK
            optimal = int(unique_prompts * 0.7)
        else:
            # Low repetition - need larger cache
            optimal = int(unique_prompts * 1.2)
        
        # Ensure reasonable bounds
        optimal = max(64, min(optimal, 1024))
        
        return optimal

# Run benchmarks
benchmark = CacheBenchmark()

# Single prompt benchmark
prompt = "Write a comprehensive guide to Python decorators"
result = benchmark.benchmark_single_prompt(prompt, iterations=20)

print("=== Single Prompt Benchmark ===")
print(f"Prompt: {result['prompt']}")
print(f"Cold cache: {result['cold_time']:.3f}s")
print(f"Warm cache: {result['warm_avg']:.3f}s ± {result['warm_std']:.3f}s")
print(f"Speedup: {result['speedup']:.1f}x")

# Typical usage pattern
typical_prompts = [
    "Write a function",
    "Explain this error",
    "Write a function",  # Repeated
    "Generate test data",
    "Write a function",  # Popular
    "Create documentation",
    "Explain this error",  # Repeated
    "Optimize code",
    "Write unit tests",
    "Write a function"   # Very popular
]

optimal = benchmark.find_optimal_cache_size(typical_prompts)
print(f"\nRecommended cache size for your usage: {optimal}")
```

## Cache Debugging

### Cache Inspector

```python
import steadytext
from typing import Optional
import json

class CacheInspector:
    """Debug and inspect cache behavior."""
    
    def __init__(self):
        self.cache_manager = steadytext.get_cache_manager()
        self.generation_log = []
    
    def trace_generation(self, prompt: str, seed: int = 42, **kwargs):
        """Trace a generation through the cache system."""
        # Get initial stats
        stats_before = self.cache_manager.get_cache_stats()
        
        # Time the generation
        import time
        start_time = time.time()
        result = steadytext.generate(prompt, seed=seed, **kwargs)
        duration = time.time() - start_time
        
        # Get final stats
        stats_after = self.cache_manager.get_cache_stats()
        
        # Analyze what happened
        gen_cache_before = stats_before['generation']
        gen_cache_after = stats_after['generation']
        
        cache_hit = gen_cache_after.get('hits', 0) > gen_cache_before.get('hits', 0)
        
        trace = {
            'prompt': prompt,
            'seed': seed,
            'duration': duration,
            'cache_hit': cache_hit,
            'cache_size_before': gen_cache_before['size'],
            'cache_size_after': gen_cache_after['size'],
            'result_preview': result[:100] + '...' if len(result) > 100 else result
        }
        
        self.generation_log.append(trace)
        
        return trace
    
    def analyze_cache_behavior(self):
        """Analyze patterns in cache behavior."""
        if not self.generation_log:
            return "No generation logs to analyze"
        
        total = len(self.generation_log)
        hits = sum(1 for log in self.generation_log if log['cache_hit'])
        
        hit_timings = [log['duration'] for log in self.generation_log if log['cache_hit']]
        miss_timings = [log['duration'] for log in self.generation_log if not log['cache_hit']]
        
        analysis = {
            'total_generations': total,
            'cache_hits': hits,
            'cache_misses': total - hits,
            'hit_rate': hits / total * 100 if total > 0 else 0,
            'avg_hit_time': sum(hit_timings) / len(hit_timings) if hit_timings else 0,
            'avg_miss_time': sum(miss_timings) / len(miss_timings) if miss_timings else 0,
            'time_saved': sum(miss_timings) - sum(hit_timings) if hit_timings else 0
        }
        
        return analysis
    
    def export_trace_log(self, filename: str):
        """Export trace log for analysis."""
        with open(filename, 'w') as f:
            json.dump(self.generation_log, f, indent=2)
        
        print(f"Trace log exported to {filename}")

# Debug cache behavior
inspector = CacheInspector()

# Trace various generations
test_cases = [
    ("Write a hello world program", 42),
    ("Write a hello world program", 42),  # Should hit
    ("Explain recursion", 42),
    ("Write a hello world program", 100),  # Different seed
    ("Explain recursion", 42),  # Should hit
]

print("=== Cache Trace Log ===")
for prompt, seed in test_cases:
    trace = inspector.trace_generation(prompt, seed)
    print(f"Prompt: {prompt[:30]}... | Seed: {seed}")
    print(f"  Hit: {trace['cache_hit']} | Time: {trace['duration']:.3f}s")
    print(f"  Cache size: {trace['cache_size_before']} -> {trace['cache_size_after']}")
    print()

# Analyze behavior
analysis = inspector.analyze_cache_behavior()
print("\n=== Cache Behavior Analysis ===")
print(f"Hit rate: {analysis['hit_rate']:.1f}%")
print(f"Average hit time: {analysis['avg_hit_time']:.3f}s")
print(f"Average miss time: {analysis['avg_miss_time']:.3f}s")
print(f"Time saved by cache: {analysis['time_saved']:.3f}s")

# Export for further analysis
inspector.export_trace_log("cache_trace.json")
```

## Best Practices

### 1. Cache Configuration

```python
# optimal_config.py - Optimal cache configuration

import os

def configure_cache_for_production():
    """Configure cache for production use."""
    # Larger cache for production
    os.environ['STEADYTEXT_GENERATION_CACHE_CAPACITY'] = '1024'
    os.environ['STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB'] = '200.0'
    
    # Even larger for embeddings (they're smaller)
    os.environ['STEADYTEXT_EMBEDDING_CACHE_CAPACITY'] = '2048'
    os.environ['STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB'] = '500.0'

def configure_cache_for_development():
    """Configure cache for development."""
    # Smaller cache for development
    os.environ['STEADYTEXT_GENERATION_CACHE_CAPACITY'] = '128'
    os.environ['STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB'] = '25.0'
    
    os.environ['STEADYTEXT_EMBEDDING_CACHE_CAPACITY'] = '256'
    os.environ['STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB'] = '50.0'

def configure_cache_for_testing():
    """Configure cache for testing."""
    # Minimal cache for testing
    os.environ['STEADYTEXT_GENERATION_CACHE_CAPACITY'] = '32'
    os.environ['STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB'] = '10.0'
    
    os.environ['STEADYTEXT_EMBEDDING_CACHE_CAPACITY'] = '64'
    os.environ['STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB'] = '20.0'
```

### 2. Cache Warming Strategy

```python
# cache_warmer.py - Strategic cache warming

import steadytext
from typing import List, Dict

class StrategicCacheWarmer:
    """Warm cache based on usage patterns."""
    
    def __init__(self):
        self.priority_prompts = {
            'high': [],      # Always cache
            'medium': [],    # Cache if space
            'low': []        # Cache opportunistically
        }
    
    def add_prompts(self, prompts: List[str], priority: str = 'medium'):
        """Add prompts to warming queue."""
        self.priority_prompts[priority].extend(prompts)
    
    def warm_cache(self, available_time: float = 30.0):
        """Warm cache within time budget."""
        import time
        start_time = time.time()
        warmed = {'high': 0, 'medium': 0, 'low': 0}
        
        # Process by priority
        for priority in ['high', 'medium', 'low']:
            for prompt in self.priority_prompts[priority]:
                if time.time() - start_time > available_time:
                    break
                
                _ = steadytext.generate(prompt, max_new_tokens=100)
                warmed[priority] += 1
        
        return warmed

# Configure warming
warmer = StrategicCacheWarmer()

# High priority - critical paths
warmer.add_prompts([
    "Generate error message",
    "Create validation response",
    "Format API response"
], priority='high')

# Medium priority - common operations
warmer.add_prompts([
    "Write documentation",
    "Generate test data",
    "Create example"
], priority='medium')

# Low priority - nice to have
warmer.add_prompts([
    "Explain concept",
    "Generate tutorial"
], priority='low')

# Warm with 10 second budget
warmed = warmer.warm_cache(available_time=10.0)
print(f"Cache warmed: {warmed}")
```

### 3. Cache Monitoring

```python
# monitor_cache.py - Production cache monitoring

import steadytext
import time
import logging
from datetime import datetime

class ProductionCacheMonitor:
    """Monitor cache in production."""
    
    def __init__(self, alert_threshold: float = 50.0):
        self.alert_threshold = alert_threshold
        self.logger = logging.getLogger(__name__)
    
    def check_cache_health(self) -> Dict:
        """Check cache health metrics."""
        cache_manager = steadytext.get_cache_manager()
        stats = cache_manager.get_cache_stats()
        
        health = {
            'timestamp': datetime.now().isoformat(),
            'healthy': True,
            'warnings': []
        }
        
        # Check generation cache
        gen_stats = stats['generation']
        gen_hit_rate = self._calculate_hit_rate(gen_stats)
        
        if gen_hit_rate < self.alert_threshold:
            health['warnings'].append(
                f"Low generation cache hit rate: {gen_hit_rate:.1f}%"
            )
            health['healthy'] = False
        
        # Check embedding cache
        emb_stats = stats['embedding']
        emb_hit_rate = self._calculate_hit_rate(emb_stats)
        
        if emb_hit_rate < self.alert_threshold:
            health['warnings'].append(
                f"Low embedding cache hit rate: {emb_hit_rate:.1f}%"
            )
            health['healthy'] = False
        
        # Check cache size
        if gen_stats['size'] >= gen_stats.get('capacity', 256) * 0.95:
            health['warnings'].append("Generation cache near capacity")
        
        if emb_stats['size'] >= emb_stats.get('capacity', 512) * 0.95:
            health['warnings'].append("Embedding cache near capacity")
        
        return health
    
    def _calculate_hit_rate(self, stats: Dict) -> float:
        """Calculate hit rate from stats."""
        hits = stats.get('hits', 0)
        misses = stats.get('misses', 0)
        total = hits + misses
        
        return (hits / total * 100) if total > 0 else 0.0
    
    def continuous_monitoring(self, interval: int = 300):
        """Monitor cache continuously."""
        while True:
            health = self.check_cache_health()
            
            if not health['healthy']:
                self.logger.warning(f"Cache health issues: {health['warnings']}")
            else:
                self.logger.info("Cache healthy")
            
            time.sleep(interval)

# Set up monitoring
monitor = ProductionCacheMonitor(alert_threshold=60.0)
health = monitor.check_cache_health()

print("=== Cache Health Check ===")
print(f"Status: {'Healthy' if health['healthy'] else 'Issues Detected'}")
if health['warnings']:
    print("Warnings:")
    for warning in health['warnings']:
        print(f"  - {warning}")
```

## Summary

Effective cache management in SteadyText involves:

1. **Configuration**: Size caches appropriately for your workload
2. **Warming**: Pre-populate cache with common prompts
3. **Monitoring**: Track hit rates and performance
4. **Optimization**: Adjust based on usage patterns
5. **Debugging**: Use tools to understand cache behavior

Remember: A well-tuned cache can provide 10-100x speedup for repeated operations!
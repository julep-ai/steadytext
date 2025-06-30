# Cache Backends

SteadyText now supports pluggable cache backends, allowing you to choose the best caching solution for your deployment scenario.

## Available Backends

### SQLite (Default)

The SQLite backend provides thread-safe, process-safe caching with automatic frecency-based eviction.

**Features:**
- Default backend, no configuration required
- Thread-safe and process-safe using WAL mode
- Automatic migration from legacy pickle format
- Configurable size limits with automatic eviction
- Persistent storage with atomic operations

**Configuration:**
```bash
# Optional: explicitly select SQLite backend
export STEADYTEXT_CACHE_BACKEND=sqlite

# Configure cache settings
export STEADYTEXT_GENERATION_CACHE_CAPACITY=256
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=50.0
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=512
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=100.0
```

### Cloudflare D1

The D1 backend enables distributed caching using Cloudflare's edge SQLite database.

**Features:**
- Global distribution across Cloudflare's edge network
- Automatic replication and disaster recovery
- Serverless with no infrastructure to manage
- Pay-per-use pricing model
- Frecency-based eviction algorithm

**Requirements:**
- Cloudflare account with Workers enabled
- Deployed D1 proxy Worker (see setup guide below)

**Configuration:**
```bash
# Select D1 backend
export STEADYTEXT_CACHE_BACKEND=d1

# Required: D1 proxy Worker configuration
export STEADYTEXT_D1_API_URL=https://your-worker.workers.dev
export STEADYTEXT_D1_API_KEY=your-secret-api-key

# Optional: Batch size for operations
export STEADYTEXT_D1_BATCH_SIZE=50
```

### Memory

The memory backend provides fast, in-memory caching for testing or ephemeral workloads.

**Features:**
- Fastest performance (no disk I/O)
- Simple FIFO eviction when capacity reached
- No persistence (data lost on restart)
- Minimal overhead

**Configuration:**
```bash
# Select memory backend
export STEADYTEXT_CACHE_BACKEND=memory

# Same capacity settings apply
export STEADYTEXT_GENERATION_CACHE_CAPACITY=256
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=512
```

## D1 Backend Setup Guide

### 1. Prerequisites

- Cloudflare account with Workers enabled
- Node.js 16.17.0 or later
- Wrangler CLI: `npm install -g wrangler`

### 2. Deploy the D1 Proxy Worker

```bash
# Navigate to the Worker directory
cd workers/d1-cache-proxy

# Install dependencies
npm install

# Login to Cloudflare
npx wrangler login

# Create D1 database
npx wrangler d1 create steadytext-cache

# Update wrangler.toml with the database ID from above

# Initialize database schema
npx wrangler d1 execute steadytext-cache --file=src/schema.sql

# Generate API key
openssl rand -base64 32

# Set API key as secret
npx wrangler secret put API_KEY
# Paste your generated API key when prompted

# Deploy the Worker
npm run deploy
```

### 3. Configure SteadyText

After deployment, configure SteadyText to use your D1 Worker:

```python
import os

# Configure D1 backend
os.environ["STEADYTEXT_CACHE_BACKEND"] = "d1"
os.environ["STEADYTEXT_D1_API_URL"] = "https://d1-cache-proxy.your-subdomain.workers.dev"
os.environ["STEADYTEXT_D1_API_KEY"] = "your-api-key-from-step-2"

# Now use SteadyText normally
from steadytext import generate, embed

text = generate("Hello world")  # Uses D1 cache
embedding = embed("Some text")   # Uses D1 cache
```

## Choosing a Backend

### Use SQLite (default) when:
- Running on a single machine or small cluster
- Need persistent cache that survives restarts
- Want zero configuration
- Have moderate traffic levels

### Use D1 when:
- Deploying globally distributed applications
- Need cache shared across multiple regions
- Want serverless, managed infrastructure
- Can tolerate slight network latency for cache operations
- Building on Cloudflare Workers platform

### Use Memory when:
- Running tests or development
- Cache persistence is not important
- Need maximum performance
- Have plenty of available RAM

## Performance Considerations

### Latency Comparison
- **Memory**: ~0.01ms per operation
- **SQLite**: ~0.1-1ms per operation
- **D1**: ~10-50ms per operation (depends on proximity to edge)

### Throughput
- **Memory**: Highest (limited by CPU)
- **SQLite**: High (limited by disk I/O)
- **D1**: Moderate (limited by network and API rate limits)

### Recommendations
1. For single-machine deployments: Use SQLite (default)
2. For global/edge deployments: Use D1
3. For testing: Use Memory
4. For high-throughput local apps: Consider Memory with periodic persistence

## Advanced Configuration

### Custom Backend Implementation

You can create your own cache backend by implementing the `CacheBackend` interface:

```python
from steadytext.cache.base import CacheBackend
from typing import Any, Dict, Optional

class MyCustomBackend(CacheBackend):
    def get(self, key: Any) -> Optional[Any]:
        # Implement get logic
        pass
    
    def set(self, key: Any, value: Any) -> None:
        # Implement set logic
        pass
    
    def clear(self) -> None:
        # Implement clear logic
        pass
    
    def sync(self) -> None:
        # Implement sync logic (if needed)
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        # Return statistics
        return {"backend": "custom", "entries": 0}
    
    def __len__(self) -> int:
        # Return number of entries
        return 0
```

### Programmatic Backend Selection

```python
from steadytext.disk_backed_frecency_cache import DiskBackedFrecencyCache

# Use specific backend programmatically
cache = DiskBackedFrecencyCache(
    backend_type="d1",
    api_url="https://your-worker.workers.dev",
    api_key="your-api-key",
    capacity=1000,
    max_size_mb=100.0
)

# Or with memory backend
cache = DiskBackedFrecencyCache(backend_type="memory")
```

## Monitoring and Debugging

### Cache Statistics

All backends provide statistics through the `get_stats()` method:

```python
from steadytext import get_cache_manager

cache_manager = get_cache_manager()
stats = cache_manager.get_cache_stats()

print(f"Generation cache: {stats['generation']}")
print(f"Embedding cache: {stats['embedding']}")
```

### D1 Worker Monitoring

Monitor your D1 Worker performance:

```bash
# View real-time logs
cd workers/d1-cache-proxy
npm run tail

# Check Worker analytics in Cloudflare dashboard
```

### Debug Environment Variables

```bash
# Enable debug logging
export STEADYTEXT_LOG_LEVEL=DEBUG

# Skip cache initialization (for testing)
export STEADYTEXT_SKIP_CACHE_INIT=1

# Disable specific cache
export STEADYTEXT_DISABLE_CACHE=1
```

## Troubleshooting

### D1 Backend Issues

**Connection Errors:**
- Verify Worker is deployed: `npx wrangler tail`
- Check API URL is correct (no trailing slash)
- Verify API key matches the secret set in Worker

**Authentication Errors:**
- Ensure Bearer token format in API_KEY
- Check secret was set correctly: `npx wrangler secret list`

**Performance Issues:**
- Monitor Worker CPU usage in Cloudflare dashboard
- Consider increasing batch size for bulk operations
- Check proximity to nearest Cloudflare edge location

### SQLite Backend Issues

**Database Corruption:**
- SteadyText automatically moves corrupted databases to `.corrupted.*` files
- Check logs for corruption warnings
- Delete corrupted files if disk space is an issue

**Lock Timeouts:**
- Usually indicates high concurrency
- Consider using D1 for distributed workloads
- Increase timeout values if needed

### Memory Backend Issues

**Out of Memory:**
- Reduce cache capacity settings
- Monitor memory usage of your application
- Consider using SQLite for overflow

## Migration Guide

### From Pickle to SQLite (Automatic)

The SQLite backend automatically migrates legacy pickle caches:

1. On first use, it detects `.pkl` files
2. Migrates all entries to SQLite format
3. Removes old pickle files after successful migration
4. No manual intervention required

### Switching Backends

To switch backends:

1. Export existing cache data (optional)
2. Change `STEADYTEXT_CACHE_BACKEND` environment variable
3. Restart application
4. Cache will be empty (unless migrating to same backend type)

Note: Cache data is not automatically transferred between different backend types.

## Best Practices

1. **Start with defaults**: SQLite backend works well for most use cases
2. **Monitor cache hit rates**: Use statistics to optimize capacity
3. **Set appropriate size limits**: Prevent unbounded cache growth
4. **Use batch operations**: Reduce round trips for D1 backend
5. **Test backend switching**: Ensure your app handles empty caches gracefully
6. **Secure your API keys**: Use environment variables, never commit keys
7. **Monitor Worker health**: Set up alerts for D1 Worker errors

## Future Backends

Planned backend support:
- Redis/Valkey (for traditional distributed caching)
- DynamoDB (for AWS deployments)
- Cloud Storage (for large value caching)

To request a backend, open an issue on GitHub.
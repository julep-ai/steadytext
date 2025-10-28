# Frequently Asked Questions

Curated answers grouped by theme. Use the tabs below to jump to the section that matches your role.

---

## Platform & Licensing

**What is SteadyText?**  
A deterministic AI platform with two first-class surfaces: the Python SDK and the `pg_steadytext` Postgres extension. Both share models, daemon, cache, and structured generation capabilities.

**Is it production-ready?**  
Yes. Deterministic outputs, `never fail` semantics, and daemon-backed caching are designed for CI, data pipelines, and transactional SQL workloads.

**What models ship with the platform?**  
Default bundles include Qwen3-based generators, Jina v4 embeddings (truncated to 1024 dims), and Qwen3 rerankers. You can swap or override via configuration (see [Model Switching](model-switching.md)).

**What about licensing?**  
SteadyText is MIT licensed. Bundled models inherit their upstream licenses (Qwen, Jina); review `LICENSE` and model-specific notices before redistribution.

---

## Python SDK

**Supported Python versions?**  
3.10+ (3.11 recommended). Earlier versions are not tested in current releases.

**Fastest way to install?**  
Use UV for reproducible environments:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv add steadytext
```

See the [Python Quick Start](quick-start.md) for daemon setup and first-generation snippets.

**How do I guarantee deterministic outputs?**  
Provide the same seed. Seeds propagate across `generate`, `generate_iter`, `embed`, and structured generation helpers.

```python
result_a = steadytext.generate("Summarize incidents", seed=7)
result_b = steadytext.generate("Summarize incidents", seed=7)
assert result_a == result_b
```

**Can I stream results?**  
Yes. `steadytext.generate_iter(prompt, seed=...)` yields deterministic token streams suitable for TUI/CLI output.

**How do I debug errors?**  
Use `steadytext.generate(..., raise_on_error=True)` during development, or inspect `steadytext.get_last_error()` when working with `None` results. For CI, log seeds whenever you capture outputs.

---

## Postgres Extension

**Which Postgres versions are supported?**  
14, 15, 16, and 17. Requires `plpython3u`, `pgvector`, and `omni_python`.

**How do I point the extension at the daemon?**  
Set GUCs and reload:

```sql
ALTER SYSTEM SET steadytext.daemon_host = '127.0.0.1';
ALTER SYSTEM SET steadytext.daemon_port = 5556;
SELECT pg_reload_conf();
```

**What functions are available?**  
`steadytext_generate`, `steadytext_generate_stream`, `steadytext_embed`, `steadytext_rerank`, prompt-registry helpers, async queue functions, and more. See the [Function Catalog](postgresql-extension-reference.md).

**How do I keep outputs deterministic inside transactions?**  
Pass explicit seeds and keep prompt text stable. The cache ensures identical SQL calls return identical results, even inside repeatable read transactions.

**Where do I start for real projects?**  
Follow the [Postgres Quick Start](postgresql-extension.md) and dive into the [Postgres journey tutorials](examples/index.md#postgres-extension).

---

## Operations & Performance

**Why is the first call slow?**  
Models download and load on demand. Start the daemon (`st daemon start`) or preload (`st models preload`) to warm caches.

**How do I monitor health?**  
Run `steadytext_healthcheck()` from SQL or `st daemon status` from the CLI. Export metrics via standard Postgres monitoring or wrap the daemon in systemd with health scripts.

**Can I change cache/backends?**  
Yes. Configure `STEADYTEXT_CACHE_BACKEND`, `STEADYTEXT_CACHE_CAPACITY`, and related env vars. Review [Cache Backends](cache-backends.md) and [Caching Recipes](examples/caching.md).

**What are typical response times?**  
Once warm: generation \<100 ms, embeddings \<15 ms, reranking \<50 ms. Performance depends on hardware and cache hit rate.

**How do upgrades work?**
Releases follow date-based versioning. Check [Version History](version_history.md) and [Changelog (GitHub)](https://github.com/julep-ai/steadytext/blob/main/CHANGELOG.md), then upgrade SDK/extension in lockstep to keep parity.

---

## Troubleshooting

**Generation returns NULL/None. What now?**  
Inspect the daemon logs, confirm connectivity (`steadytext_healthcheck()`), and ensure models are downloaded. In Postgres, review `steadytext.get_last_error()` via plpython3u helpers.

**Daemon can’t bind to a port.**  
Verify no other process is using the port. On shared hosts, bind with `--host 0.0.0.0 --port 5556` and restrict via firewall/security groups.

**Embeddings differ between Python and SQL.**  
Ensure both pillars target the same daemon and model version. Mixed versions or different seeds cause divergence.

**Link checker failures during docs build.**  
Run the command documented in the migration checklist (see project repo) and confirm any renamed pages have updated redirects.

Still stuck? Open an issue with reproduction details (prompt, seed, command) and include anchor references from the relevant docs section.
| Batch (100) | 3-5s | <100ms | <200ms |

## Model Questions

### Can I use different model sizes?

Yes, SteadyText supports multiple model sizes:

```bash
# CLI
st generate "Hello" --size small  # Fast, 2B parameters
st generate "Hello" --size large  # Better quality, 4B parameters

# Python
text = steadytext.generate("Hello", model_size="large")
```

### Can I use custom models?

Currently, SteadyText uses pre-selected models for consistency. Custom model support is planned for future releases.

### How much disk space do models use?

- **Small generation model**: ~1.3GB
- **Large generation model**: ~2.1GB  
- **Embedding model**: ~0.6GB
- **Total (all models)**: ~4GB

### Where are models stored?

Models are cached in platform-specific directories:

```python
# Linux/Mac
~/.cache/steadytext/models/

# Windows
%LOCALAPPDATA%\steadytext\steadytext\models\

# Check location
from steadytext.utils import get_model_cache_dir
print(get_model_cache_dir())
```

## Caching Questions

### How does caching work?

SteadyText uses a frecency cache (frequency + recency):

```python
# First call: generates and caches
result1 = steadytext.generate("Hello", seed=42)  # Slow

# Second call: returns from cache
result2 = steadytext.generate("Hello", seed=42)  # Instant

# Different seed: new generation
result3 = steadytext.generate("Hello", seed=123)  # Slow
```

### Can I disable caching?

```bash
# Disable via environment variable
export STEADYTEXT_DISABLE_CACHE=1

# Or in Python
import os
os.environ['STEADYTEXT_DISABLE_CACHE'] = '1'
```

### How do I clear the cache?

```bash
# CLI
st cache --clear

# Python
from steadytext import get_cache_manager
cache_manager = get_cache_manager()
cache_manager.clear_all_caches()
```

### How much cache space is used?

```python
# Check cache statistics
from steadytext import get_cache_manager
stats = get_cache_manager().get_cache_stats()
print(f"Generation cache: {stats['generation']['size']} entries")
print(f"Embedding cache: {stats['embedding']['size']} entries")
```

## Daemon Questions

### What is daemon mode?

The daemon is a background service that keeps models loaded in memory, providing 160x faster first responses.

### How do I start the daemon?

```bash
# Start in background
st daemon start

# Start in foreground (see logs)
st daemon start --foreground

# Check status
st daemon status
```

### Is the daemon used automatically?

Yes! When the daemon is running, all SteadyText operations automatically use it:

```python
# Automatically uses daemon if available
text = steadytext.generate("Hello")
```

### How do I stop the daemon?

```bash
# Graceful stop
st daemon stop

# Force stop
st daemon stop --force
```

### Can I run multiple daemons?

Currently, only one daemon instance is supported per machine. Multi-daemon support is planned for future releases.

## PostgreSQL Extension

### How do I install pg_steadytext?

```bash
# Using Docker (recommended)
cd pg_steadytext
docker build -t pg_steadytext .
docker run -d -p 5432:5432 pg_steadytext

# Manual installation
cd pg_steadytext
make && sudo make install
```

### How do I use it in SQL?

```sql
-- Enable extension
CREATE EXTENSION pg_steadytext;

-- Generate text
SELECT steadytext_generate('Write a SQL tutorial');

-- Create embeddings
SELECT steadytext_embed('PostgreSQL database');
```

### Is it production-ready?

The PostgreSQL extension is currently experimental. Use with caution in production environments.

## Troubleshooting

### "Model not found" error

```bash
# Download models manually
st models download --all

# Or set environment variable
export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true
```

### "None" returned instead of text

This is the expected behavior in v2.1.0+ when models can't be loaded:

```python
# Check if generation succeeded
result = steadytext.generate("Hello")
if result is None:
    print("Model not available")
else:
    print(f"Generated: {result}")
```

### Daemon won't start

```bash
# Check if port is in use
lsof -i :5557

# Try different port
st daemon start --port 5558

# Check logs
st daemon start --foreground
```

### High memory usage

```bash
# Use smaller model
st generate "Hello" --size small

# Limit cache size
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=50
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=100
```

### Slow generation

```bash
# Start daemon for faster responses
st daemon start

# Check cache is working
st cache --status

# Use smaller model
st generate "Hello" --size small
```

## Advanced Topics

### How do I use SteadyText in production?

1. **Use daemon mode**:
   ```bash
   # systemd service
   sudo systemctl enable steadytext
   sudo systemctl start steadytext
   ```

2. **Configure caching**:
   ```bash
   export STEADYTEXT_GENERATION_CACHE_CAPACITY=2048
   export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=500
   ```

3. **Monitor performance**:
   ```python
   from steadytext import get_cache_manager
   stats = get_cache_manager().get_cache_stats()
   # Log stats to monitoring system
   ```

### Can I use SteadyText with async code?

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def async_generate(prompt):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, 
        steadytext.generate, 
        prompt
    )

# Use in async function
result = await async_generate("Hello")
```

### How do I handle errors gracefully?

```python
def safe_generate(prompt, fallback="Unable to generate"):
    try:
        result = steadytext.generate(prompt)
        if result is None:
            return fallback
        return result
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return fallback
```

### Can I use SteadyText with langchain?

```python
from langchain.llms.base import LLM

class SteadyTextLLM(LLM):
    def _call(self, prompt: str, stop=None) -> str:
        result = steadytext.generate(prompt)
        return result if result else ""
    
    @property
    def _llm_type(self) -> str:
        return "steadytext"

# Use with langchain
llm = SteadyTextLLM()
```

### How do I benchmark performance?

```bash
# Run built-in benchmarks
cd benchmarks
python run_all_benchmarks.py

# Quick benchmark
python run_all_benchmarks.py --quick
```

### Can I contribute to SteadyText?

Yes! We welcome contributions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `uv run pytest`
5. Submit a pull request

See [CONTRIBUTING.md](https://github.com/diwank/steadytext/blob/main/CONTRIBUTING.md) for details.

## Still Have Questions?

- **GitHub Issues**: [Report bugs or request features](https://github.com/diwank/steadytext/issues)
- **Discussions**: [Join the community](https://github.com/diwank/steadytext/discussions)
- **Documentation**: [Read the full docs](https://steadytext.readthedocs.io)

## Quick Reference

### Common Commands

```bash
# Generation
echo "prompt" | st
st generate "prompt" --seed 42

# Embeddings
st embed "text"
st embed "text" --format numpy

# Daemon
st daemon start
st daemon status
st daemon stop

# Cache
st cache --status
st cache --clear

# Models
st models list
st models download --all
st models preload
```

### Common Patterns

```python
# Basic usage
import steadytext

# Generate text
text = steadytext.generate("Hello world")

# Create embedding
embedding = steadytext.embed("Hello world")

# With custom seed
text = steadytext.generate("Hello", seed=123)

# Streaming
for chunk in steadytext.generate_iter("Tell a story"):
    print(chunk, end='')

# Batch processing
prompts = ["One", "Two", "Three"]
results = [steadytext.generate(p) for p in prompts]
```

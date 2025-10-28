# Postgres Quick Start

Use this guide to install the `pg_steadytext` extension, connect it to the SteadyText daemon, and run your first deterministic SQL workflows. If you are building with the Python SDK, start with the [Python Quick Start](quick-start.md).

---

## 1. Prerequisites

- **PostgreSQL** 14–17 with superuser access for extension install
- **Extensions**: `plpython3u`, `pgvector`, `omni_python`
- **Python runtime** (for daemon + plpython3u) 3.10+
- **Daemon host**: access to a running SteadyText daemon (`st daemon start`)

Verify prerequisites inside psql:

```sql
SHOW server_version;
SELECT extname FROM pg_extension WHERE extname IN ('plpython3u', 'pgvector', 'omni_python');
```

---

## 2. Install the Extension

### Option A — Package / Source Install

```bash
# Install Python dependencies for plpython3u
pip install "steadytext>=2025.10.0" pyzmq numpy

# Build and install pg_steadytext
git clone https://github.com/julep-ai/steadytext.git
cd steadytext/pg_steadytext
make && sudo make install
```

### Option B — Docker Playground

```bash
cd steadytext/pg_steadytext
docker build -t pg_steadytext .
docker run -d -p 5432:5432 --name pg_steadytext pg_steadytext
```

---

## 3. Enable Required Extensions

Inside `psql` (or another SQL client):

```sql
CREATE EXTENSION IF NOT EXISTS plpython3u;
CREATE EXTENSION IF NOT EXISTS omni_python;
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS pg_steadytext;
```

Check the version:

```sql
SELECT steadytext_version();
```

---

## 4. Connect to the Daemon

The extension talks to the same daemon the Python SDK uses. On your app host (or another reachable machine) run:

```bash
st daemon start --host 0.0.0.0 --port 5556
```

Then configure connection details inside Postgres:

```sql
ALTER SYSTEM SET steadytext.daemon_host = '127.0.0.1';
ALTER SYSTEM SET steadytext.daemon_port = 5556;
SELECT pg_reload_conf();
```

Confirm connectivity:

```sql
SELECT steadytext_healthcheck();
```

---

## 5. First Deterministic Query

```sql
SELECT steadytext_generate(
  prompt      => 'Write a release announcement for v2025.10',
  max_tokens  => 200,
  seed        => 42
);
```

- Same prompt + seed ⇒ identical output across environments.
- Set `use_cache := false` to bypass the shared frecency cache.

Streaming mirrors the Python behavior via `steadytext_generate_stream` (see [Function Catalog](postgresql-extension-reference.md)).

---

## 6. Embeddings & Retrieval

```sql
SELECT steadytext_embed('Customer escalated about latency');
```

Store vectors with pgvector:

```sql
CREATE TABLE support_docs (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding VECTOR(1024)
);

INSERT INTO support_docs (content, embedding)
VALUES (
  'Investigate connection pooling saturation',
  steadytext_embed('Investigate connection pooling saturation')
);
```

Combine embeddings with reranking:

```sql
WITH candidates AS (
  SELECT id, content
  FROM support_docs
  ORDER BY content <-> steadytext_embed('timeouts on checkout')  -- pgvector distance
  LIMIT 10
)
SELECT * FROM steadytext_rerank(
  'timeouts on checkout',
  ARRAY(SELECT content FROM candidates)
);
```

---

## 7. Operational Flight Checks

- Monitor daemon status with `steadytext_healthcheck()`.
- Use `steadytext_cache_stats()` to view shared cache metrics.
- Toggle deterministic defaults via `ALTER SYSTEM SET steadytext.default_seed = 42;`.
- For async workflows see [Postgres Async Operations](postgresql-extension-async.md).

---

## 8. Next Steps

- Follow the [Postgres Journey](examples/postgresql-integration.md) for deeper tutorials.
- Browse the [Function Catalog](postgresql-extension-reference.md) for full SQL signatures.
- Configure production rollouts using the [Deployment guides](deployment.md) and [Cloudflare recipe](deployment/cloudflare.md).

Need parity with Python workflows? Visit the [Core Platform Hub](architecture.md) to see how the pillars share models, caching, and structured generation.
SELECT steadytext_embed(
    'artificial intelligence',
    seed := 123
);

-- Handle NULL embeddings from failed generation
SELECT 
    text,
    CASE 
        WHEN steadytext_embed(text, seed := 42) IS NOT NULL 
        THEN 'Embedding generated'
        ELSE 'Embedding failed'
    END AS status
FROM documents;

-- Semantic similarity using pgvector with NULL handling
WITH base_embedding AS (
    SELECT steadytext_embed('machine learning', seed := 42) AS vector
)
SELECT 
    text,
    embedding <-> (SELECT vector FROM base_embedding) AS distance
FROM documents
WHERE embedding IS NOT NULL 
    AND (SELECT vector FROM base_embedding) IS NOT NULL
ORDER BY distance
LIMIT 5;

-- Compare embeddings with different seeds (with NULL checks)
SELECT 
    variant,
    CASE 
        WHEN embedding IS NOT NULL THEN 'Generated'
        ELSE 'Failed'
    END AS status,
    embedding
FROM (
    SELECT 
        'Default seed' AS variant,
        steadytext_embed('AI technology') AS embedding
    UNION ALL
    SELECT 
        'Custom seed' AS variant,
        steadytext_embed('AI technology', seed := 789) AS embedding
) results;
```

## Additional Features

### Structured Generation (v2.4.1+)

The extension supports structured text generation using llama.cpp's native grammar support:

- **JSON Generation**: Generate JSON conforming to schemas
- **Regex Patterns**: Generate text matching regular expressions  
- **Choice Constraints**: Generate text from predefined choices

📖 **[Full Structured Generation Documentation →](postgresql-extension-structured.md)**

### Document Reranking

Available since v1.3.0+.

Rerank documents by relevance using the Qwen3-Reranker-4B model:

- **Query-based Reranking**: Reorder documents by relevance
- **Batch Operations**: Process multiple queries efficiently
- **Custom Task Descriptions**: Domain-specific reranking

📖 **[Full Reranking Documentation →](postgresql-extension-reranking.md)**

## Management Functions

### Daemon Management

#### `steadytext_daemon_start()`

Start the SteadyText daemon for improved performance.

```sql
SELECT steadytext_daemon_start();
SELECT steadytext_daemon_start('localhost', 5557); -- Custom host/port
```

#### `steadytext_daemon_status()`

Check daemon health and status.

```sql
SELECT * FROM steadytext_daemon_status();
-- Returns: running, pid, host, port, uptime, health
```

#### `steadytext_daemon_stop()`

Stop the daemon gracefully.

```sql
SELECT steadytext_daemon_stop();
SELECT steadytext_daemon_stop(true); -- Force stop
```

### Cache Management

#### `steadytext_cache_stats()`

View cache performance statistics.

```sql
SELECT * FROM steadytext_cache_stats();
-- Returns: entries, total_size_mb, hit_rate, evictions, oldest_entry
```

#### `steadytext_cache_clear()`

Clear the cache for fresh results.

```sql
SELECT steadytext_cache_clear();                    -- Clear all
SELECT steadytext_cache_clear('generation');        -- Clear generation cache only
SELECT steadytext_cache_clear('embedding');         -- Clear embedding cache only
```

#### Automatic Cache Eviction with pg_cron

The extension supports automatic cache eviction using pg_cron:

```sql
-- Basic setup with default settings
SELECT steadytext_setup_cache_eviction();

-- Custom eviction settings
SELECT steadytext_setup_cache_eviction(
    eviction_interval := '1 hour',
    max_age_days := 7,
    target_cache_size_mb := 100.0
);
```

📖 **[Full Cache Management Documentation →](postgresql-extension-advanced.md#automatic-cache-eviction-with-pg_cron)**

### Configuration

#### `steadytext_config_get()` / `steadytext_config_set()`

Manage extension configuration.

```sql
-- View all configuration
SELECT * FROM steadytext_config;

-- Get specific setting
SELECT steadytext_config_get('default_max_tokens');

-- Update settings
SELECT steadytext_config_set('default_max_tokens', '1024');
SELECT steadytext_config_set('cache_enabled', 'true');
SELECT steadytext_config_set('daemon_host', 'localhost');
SELECT steadytext_config_set('daemon_port', '5557');
SELECT steadytext_config_set('default_seed', '42');
```

## Database Schema

The extension creates several tables to manage caching, configuration, and monitoring:

### `steadytext_cache`

Stores cached generation and embedding results with frecency metadata.

```sql
\d steadytext_cache
```

| Column | Type | Description |
|--------|------|-------------|
| `key` | TEXT | Cache key (hash of input + parameters) |
| `prompt` | TEXT | Original prompt text |
| `result` | TEXT | Generated text result |
| `embedding` | vector(1024) | Generated embedding vector |
| `seed` | INTEGER | Seed used for generation |
| `frequency` | INTEGER | Access frequency counter |
| `last_access` | TIMESTAMP | Last access time |
| `created_at` | TIMESTAMP | Creation timestamp |

### `steadytext_config`

Extension configuration settings.

```sql
SELECT key, value, description FROM steadytext_config;
```

| Key | Default | Description |
|-----|---------|-------------|
| `default_max_tokens` | `512` | Default maximum tokens to generate |
| `cache_enabled` | `true` | Enable/disable caching |
| `daemon_host` | `localhost` | Daemon server host |
| `daemon_port` | `5557` | Daemon server port |
| `default_seed` | `42` | Default seed for operations |
| `use_fallback_model` | `false` | Use fallback model if primary fails |
| `rate_limit_enabled` | `false` | Enable rate limiting |
| `max_requests_per_minute` | `60` | Rate limit threshold |

### `steadytext_daemon_health`

Daemon health monitoring and diagnostics.

```sql
SELECT * FROM steadytext_daemon_health ORDER BY checked_at DESC LIMIT 5;
```

## Advanced Topics

### Performance Optimization

- **Cache Management**: Monitor and optimize cache performance
- **Memory Management**: Configure model memory usage
- **Connection Pooling**: Daemon connection optimization
- **Query Optimization**: Batch operations and indexing

### Security & Integration

- **Input Validation**: Safe text generation patterns
- **Rate Limiting**: Control resource usage
- **Access Control**: Role-based permissions
- **Integration Patterns**: pgvector, TimescaleDB, PostGIS

📖 **[Full Advanced Topics Documentation →](postgresql-extension-advanced.md)**

### AI Summarization (v1.1.0+)

Powerful AI summarization aggregate functions with TimescaleDB support:

- **Text Summarization**: Single and aggregate text summarization
- **Fact Extraction**: Extract and deduplicate key facts
- **Partial Aggregation**: Efficient time-series summarization
- **Metadata Support**: Context-aware summarization

📖 **[Full AI Summarization Documentation →](postgresql-extension-ai.md)**

### Async Functions (v1.1.0+)

Non-blocking AI operations for high-throughput applications:

- **Queue-based Processing**: Background worker architecture
- **Priority Support**: Control processing order
- **Batch Operations**: Efficient bulk processing
- **LISTEN/NOTIFY Integration**: Real-time notifications

📖 **[Full Async Functions Documentation →](postgresql-extension-async.md)**

## Troubleshooting

### Common Issues

#### 1. "No module named 'steadytext'" Error

This indicates PostgreSQL cannot find the SteadyText library:

```sql
-- Check Python environment
DO $$
BEGIN
    RAISE NOTICE 'Python version: %', (SELECT version());
END;
$$ LANGUAGE plpython3u;

-- Manually initialize (if needed)
SELECT _steadytext_init_python();

-- Verify installation
DO $$
import sys
import os
plpy.notice(f"Python path: {sys.path}")
plpy.notice(f"Current user: {os.getenv('USER', 'unknown')}")
try:
    import steadytext
    plpy.notice(f"SteadyText version: {steadytext.__version__}")
except ImportError as e:
    plpy.error(f"SteadyText not available: {e}")
$$ LANGUAGE plpython3u;
```

**Solution:**
```bash
# Install SteadyText for the PostgreSQL Python environment
sudo -u postgres pip3 install steadytext>=2.1.0

# Or reinstall the extension
make clean && make install
```

#### 2. Model Loading Errors

If functions return NULL due to model loading issues:

```sql
-- Check current model configuration
SELECT steadytext_config_get('use_fallback_model');

-- Enable fallback model
SELECT steadytext_config_set('use_fallback_model', 'true');

-- Test generation (will return NULL if still failing)
SELECT 
    CASE 
        WHEN steadytext_generate('Test model loading') IS NOT NULL 
        THEN 'Model working'
        ELSE 'Model still failing - check daemon status'
    END AS status;
```

**Environment Solution:**
```bash
# Set fallback model environment variable
export STEADYTEXT_USE_FALLBACK_MODEL=true

# Restart PostgreSQL
sudo systemctl restart postgresql
```

#### 3. Daemon Connection Issues

```sql
-- Check daemon status
SELECT * FROM steadytext_daemon_status();

-- Restart daemon with custom settings
SELECT steadytext_daemon_stop();
SELECT steadytext_config_set('daemon_host', 'localhost');
SELECT steadytext_config_set('daemon_port', '5557');
SELECT steadytext_daemon_start();

-- Test daemon connectivity
SELECT steadytext_generate('Test daemon connection');
```

#### 4. NULL Returns and Error Handling

```sql
-- Check if functions are returning NULL
SELECT 
    'Generation test' AS test_type,
    CASE 
        WHEN steadytext_generate('Test prompt') IS NOT NULL 
        THEN 'Working'
        ELSE 'Returning NULL - check daemon'
    END AS status
UNION ALL
SELECT 
    'Embedding test' AS test_type,
    CASE 
        WHEN steadytext_embed('Test text') IS NOT NULL 
        THEN 'Working'
        ELSE 'Returning NULL - check daemon'
    END AS status;

-- Application-level NULL handling pattern
CREATE OR REPLACE FUNCTION robust_generate(
    prompt TEXT,
    retry_count INTEGER DEFAULT 3
)
RETURNS TEXT AS $$
DECLARE
    result TEXT;
    i INTEGER;
BEGIN
    FOR i IN 1..retry_count LOOP
        result := steadytext_generate(prompt);
        IF result IS NOT NULL THEN
            RETURN result;
        END IF;
        
        -- Wait before retry
        PERFORM pg_sleep(1);
    END LOOP;
    
    -- All retries failed
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
```

#### 5. Cache Performance Issues

```sql
-- Monitor cache statistics
SELECT * FROM steadytext_cache_stats();

-- Clear cache if needed
SELECT steadytext_cache_clear();

-- Adjust cache settings
SELECT steadytext_config_set('cache_capacity', '1000');
SELECT steadytext_config_set('cache_max_size_mb', '200');
```

### Debugging Mode

Enable verbose logging for troubleshooting:

```sql
-- Enable PostgreSQL notices
SET client_min_messages TO NOTICE;

-- Test with debug output and NULL checking
SELECT 
    'Debug test' AS test_name,
    steadytext_generate('Debug test', max_tokens := 10) AS result,
    CASE 
        WHEN steadytext_generate('Debug test', max_tokens := 10) IS NULL 
        THEN 'Generation failed - check notices above'
        ELSE 'Generation successful'
    END AS status;

-- Check daemon health
SELECT * FROM steadytext_daemon_status();

-- Check recent health history
SELECT * FROM steadytext_daemon_health ORDER BY last_heartbeat DESC LIMIT 10;
```

## Version Compatibility

| PostgreSQL | Python | SteadyText | Status |
|------------|--------|------------|---------|
| 14+ | 3.8+ | 2.1.0+ | ✅ Fully Supported |
| 13 | 3.8+ | 2.1.0+ | ⚠️ Limited Testing |
| 12 | 3.7+ | 2.0.0+ | ❌ Not Recommended |

## Migration Guide

### Upgrading from v1.0.0

1. **Update Dependencies:**
```bash
pip3 install --upgrade steadytext>=2.1.0
```

2. **Update Extension:**
```sql
ALTER EXTENSION pg_steadytext UPDATE TO '1.1.0';
```

3. **Update Function Calls and Error Handling:**
```sql
-- Old (v1.0.0) - returned fallback text on errors
SELECT steadytext_generate('prompt', 512, true);

-- New (v1.1.0+) - with seed support and NULL returns on errors
SELECT steadytext_generate('prompt', max_tokens := 512, seed := 42);

-- Application code should now handle NULL returns
SELECT 
    COALESCE(
        steadytext_generate('prompt', max_tokens := 512, seed := 42),
        'Error: Generation failed'
    ) AS result;
```

## Contributing

The pg_steadytext extension is part of the main SteadyText project. Contributions are welcome!

- **GitHub Repository**: https://github.com/julep-ai/steadytext
- **Issues**: https://github.com/julep-ai/steadytext/issues
- **Extension Directory**: `pg_steadytext/`

## License

This extension is released under the PostgreSQL License, consistent with the main SteadyText project.

---

## Documentation Index

### Core Documentation
- 📖 [Main Documentation](postgresql-extension.md) - This page
- 📖 [Structured Generation & Reranking](postgresql-extension-structured.md)
- 📖 [AI Summarization Features](postgresql-extension-ai.md)
- 📖 [Async Functions](postgresql-extension-async.md)
- 📖 [Advanced Topics & Performance](postgresql-extension-advanced.md)

### Additional Resources
- 🚀 [Main SteadyText Documentation](https://github.com/julep-ai/steadytext)
- 🐛 [Report Issues](https://github.com/julep-ai/steadytext/issues)
- 📦 [Extension Directory](https://github.com/julep-ai/steadytext/tree/main/pg_steadytext)

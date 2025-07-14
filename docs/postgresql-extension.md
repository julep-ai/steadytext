# PostgreSQL Extension (pg_steadytext)

The **pg_steadytext** PostgreSQL extension provides native SQL functions for deterministic text generation and embeddings by integrating with the SteadyText library. It brings the power of modern language models directly into your PostgreSQL database.

## Overview

pg_steadytext extends PostgreSQL with:

- **Deterministic Text Generation**: SQL functions that generate consistent text output with custom seeds
- **Vector Embeddings**: Create 1024-dimensional embeddings compatible with pgvector
- **Built-in Caching**: PostgreSQL-based frecency cache for optimal performance
- **Daemon Integration**: Seamless integration with SteadyText's ZeroMQ daemon
- **Custom Seed Support**: Full control over deterministic generation with custom seeds
- **Reliable Error Handling**: Functions return NULL on errors instead of fallback text
- **Security**: Input validation, rate limiting, and safe error handling

## Requirements

- **PostgreSQL**: 14+ (tested on 14, 15, 16, 17)
- **Python**: 3.8+ (matches plpython3u version)
- **SteadyText**: 2.3.0+ (for reranking support, daemon, and custom seeds)
- **Extensions**:
  - `plpython3u` (required for Python integration)
  - `pgvector` (required for embedding storage)
  - `omni_python` (required for enhanced Python integration, see https://docs.omnigres.org/quick_start/)

## Installation

### Quick Installation

```bash
# Install Python dependencies
pip3 install steadytext>=2.3.0 pyzmq numpy

# Install omni-python (if not available via package manager)
git clone https://github.com/omnigres/omnigres.git
cd omnigres/extensions/omni_python
make && sudo make install
cd ../../..

# Clone the SteadyText repository
git clone https://github.com/julep-ai/steadytext.git
cd steadytext/pg_steadytext

# Build and install the extension
make && sudo make install

# Enable in PostgreSQL
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;"
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS omni_python CASCADE;"
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS pgvector CASCADE;"
psql -U postgres -c "CREATE EXTENSION pg_steadytext CASCADE;"
```

### Docker Installation

For a complete containerized setup:

```bash
# Standard build
docker build -t pg_steadytext .

# Build with fallback model (recommended for compatibility)
docker build --build-arg STEADYTEXT_USE_FALLBACK_MODEL=true -t pg_steadytext .

# Run the container
docker run -d -p 5432:5432 --name pg_steadytext pg_steadytext

# Test the installation
docker exec -it pg_steadytext psql -U postgres -c "SELECT steadytext_version();"
```

## Core Functions

### Text Generation

#### `steadytext_generate()`

Generate deterministic text from a prompt with full customization options.

```sql
steadytext_generate(
    prompt TEXT,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TEXT
-- Returns NULL if generation fails
```

**Examples:**

```sql
-- Simple text generation (uses default seed 42)
SELECT steadytext_generate('Write a haiku about PostgreSQL');

-- Custom seed for reproducible results
SELECT steadytext_generate(
    'Tell me a story',
    max_tokens := 256,
    seed := 12345
);

-- Disable caching for fresh results
SELECT steadytext_generate(
    'Random joke',
    use_cache := false,
    seed := 999
);

-- Handle NULL results from failed generation
SELECT COALESCE(
    steadytext_generate('Generate text', seed := 100),
    'Generation failed - please check daemon status'
) AS result;

-- Compare outputs with different seeds
SELECT 
    'Seed 100' AS variant,
    steadytext_generate('Explain machine learning', seed := 100) AS output
UNION ALL
SELECT 
    'Seed 200' AS variant,
    steadytext_generate('Explain machine learning', seed := 200) AS output;
```

#### `steadytext_generate_stream()`

Stream text generation for real-time applications (future feature).

```sql
steadytext_generate_stream(
    prompt TEXT,
    max_tokens INTEGER DEFAULT 512,
    seed INTEGER DEFAULT 42
) RETURNS SETOF TEXT
```

### Embeddings

#### `steadytext_embed()`

Generate 1024-dimensional L2-normalized embeddings for text.

```sql
steadytext_embed(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS vector(1024)
-- Returns NULL vector if embedding fails
```

**Examples:**

```sql
-- Simple embedding (uses default seed 42)
SELECT steadytext_embed('PostgreSQL is a powerful database');

-- Custom seed for reproducible embeddings
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

### Document Reranking (v1.3.0+)

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
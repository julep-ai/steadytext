# pg_steadytext - PostgreSQL Extension for SteadyText

**pg_steadytext** is a PostgreSQL extension that provides deterministic text generation and embeddings by integrating with the [SteadyText](https://github.com/julep-ai/steadytext) library. It offers SQL functions for text generation, embedding creation, and intelligent caching with frecency-based eviction.

## Features

- **Deterministic Text Generation**: Always returns the same output for the same input
- **Vector Embeddings**: Generate 1024-dimensional embeddings compatible with pgvector
- **Built-in Caching**: PostgreSQL-based frecency cache that mirrors SteadyText's cache
- **Daemon Integration**: Seamlessly integrates with SteadyText's ZeroMQ daemon
- **Async Processing**: Queue-based asynchronous text generation (coming soon)
- **Security**: Input validation and rate limiting
- **Monitoring**: Health checks and performance statistics

## Requirements

- PostgreSQL 14+ 
- Python 3.10+
- Extensions:
  - `plpython3u` (required)
  - `pgvector` (required)
- Python packages:
  - `steadytext` (install with `pip3 install steadytext`)

## Installation

### Quick Install

```bash
# Install Python dependencies
pip3 install steadytext

# Clone and install the extension
git clone https://github.com/julep-ai/steadytext.git
cd steadytext/pg_steadytext
make && sudo make install

# In PostgreSQL
CREATE EXTENSION pg_steadytext CASCADE;
```

### Development Install

```bash
# Install with development setup
make dev-install

# Run tests
make dev-test
```

## Basic Usage

### Text Generation

```sql
-- Simple text generation
SELECT steadytext_generate('Write a haiku about PostgreSQL');

-- With parameters
SELECT steadytext_generate(
    'Explain quantum computing',
    max_tokens := 256,
    use_cache := true
);

-- Using a custom seed for reproducible results
SELECT steadytext_generate(
    'Create a short story',
    seed := 12345
);

-- Check cache statistics
SELECT * FROM steadytext_cache_stats();
```

### Embeddings

```sql
-- Generate embedding for text
SELECT steadytext_embed('PostgreSQL is a powerful database');

-- Find similar texts using pgvector
SELECT prompt, embedding <-> steadytext_embed('database query') AS distance
FROM steadytext_cache
WHERE embedding IS NOT NULL
ORDER BY distance
LIMIT 5;
```

### Daemon Management

```sql
-- Start the SteadyText daemon
SELECT steadytext_daemon_start();

-- Check daemon status
SELECT * FROM steadytext_daemon_status();

-- Stop daemon
SELECT steadytext_daemon_stop();
```

### Configuration

```sql
-- View current configuration
SELECT * FROM steadytext_config;

-- Update settings
SELECT steadytext_config_set('default_max_tokens', '1024');
SELECT steadytext_config_set('cache_enabled', 'false');

-- Get specific setting
SELECT steadytext_config_get('daemon_port');
```

## Architecture

pg_steadytext integrates with SteadyText's existing architecture:

```
PostgreSQL Client
       |
       v
  SQL Functions
       |
       v
 Python Bridge -----> SteadyText Daemon (ZeroMQ)
       |                    |
       v                    v
 PostgreSQL Cache <--- SteadyText Cache (SQLite)
```

## Tables

- `steadytext_cache` - Stores generated text and embeddings with frecency statistics
- `steadytext_queue` - Queue for async operations (future feature)
- `steadytext_config` - Extension configuration
- `steadytext_daemon_health` - Daemon health monitoring

## Functions

### Core Functions
- `steadytext_generate(prompt, max_tokens, use_cache, seed)` - Generate text
- `steadytext_embed(text, use_cache)` - Generate embedding
- `steadytext_generate_stream(prompt, max_tokens)` - Stream text generation

### Management Functions
- `steadytext_daemon_start()` - Start the daemon
- `steadytext_daemon_status()` - Check daemon health
- `steadytext_daemon_stop()` - Stop the daemon
- `steadytext_cache_stats()` - Get cache statistics
- `steadytext_cache_clear()` - Clear the cache
- `steadytext_version()` - Get extension version

### Configuration Functions
- `steadytext_config_get(key)` - Get configuration value
- `steadytext_config_set(key, value)` - Set configuration value

## Performance

The extension uses several optimizations:
- Prepared statements for repeated queries
- In-memory configuration caching
- Connection pooling to the daemon
- Frecency-based cache eviction
- Indexes on cache keys and frecency scores

## Security

- Input validation for all user inputs
- Protection against prompt injection
- Rate limiting support (configure in `steadytext_rate_limits` table)
- Configurable resource limits

## Troubleshooting

### Common Issues

#### "No module named 'daemon_connector'" Error
This is the most common issue, occurring when PostgreSQL's plpython3u cannot find the extension's Python modules.

**Solution:**
```sql
-- 1. Initialize Python environment manually
SELECT _steadytext_init_python();

-- 2. Check Python path configuration
SHOW plpython3.python_path;

-- 3. Verify modules are installed in the correct location
DO $$
DECLARE
    pg_lib_dir TEXT;
BEGIN
    SELECT setting INTO pg_lib_dir FROM pg_settings WHERE name = 'pkglibdir';
    RAISE NOTICE 'Modules should be in: %/pg_steadytext/python/', pg_lib_dir;
END;
$$;
```

**If the error persists:**
```bash
# Reinstall the extension
make clean && make install

# Verify installation
ls $(pg_config --pkglibdir)/pg_steadytext/python/
```

#### Docker-specific Issues
When running in Docker, additional steps may be needed:

```bash
# Test Docker installation
./test_docker.sh

# Debug module loading in Docker
docker exec <container> psql -U postgres -c "SELECT _steadytext_init_python();"

# Check Python modules in container
docker exec <container> ls -la $(pg_config --pkglibdir)/pg_steadytext/python/
```

#### Model Loading Errors: "Failed to load model from file"
If you see errors like "Failed to load model from file: /path/to/gemma-3n-*.gguf", this is a known compatibility issue between gemma-3n models and the inference-sh fork of llama-cpp-python.

**Quick Fix - Use Fallback Model:**
```bash
# For Docker build:
docker build --build-arg STEADYTEXT_USE_FALLBACK_MODEL=true -t pg_steadytext .

# For Docker run:
docker run -e STEADYTEXT_USE_FALLBACK_MODEL=true -p 5432:5432 pg_steadytext

# For direct usage:
export STEADYTEXT_USE_FALLBACK_MODEL=true
```

**Alternative - Specify Compatible Model:**
```bash
export STEADYTEXT_GENERATION_MODEL_REPO=lmstudio-community/Qwen2.5-3B-Instruct-GGUF
export STEADYTEXT_GENERATION_MODEL_FILENAME=Qwen2.5-3B-Instruct-Q8_0.gguf
```

**Diagnose the Issue:**
```bash
# Run diagnostic script in Docker
docker exec -it <container> /usr/local/bin/diagnose_pg_model

# Or run directly
python3 -m steadytext.diagnose_model
```

#### Daemon not starting
```sql
-- Check if SteadyText is installed
SELECT steadytext_daemon_status();

-- Manually start with specific settings
SELECT steadytext_config_set('daemon_host', 'localhost');
SELECT steadytext_config_set('daemon_port', '5555');
SELECT steadytext_daemon_start();

-- Check daemon logs
-- On host: st daemon status
```

#### Cache issues
```sql
-- View cache statistics
SELECT * FROM steadytext_cache_stats();

-- Clear cache if needed
SELECT steadytext_cache_clear();

-- Check cache eviction settings
SELECT * FROM steadytext_config WHERE key LIKE '%cache%';
```

#### Python module version mismatches
```bash
# Check Python version used by PostgreSQL
psql -c "DO $$ import sys; plpy.notice(f'Python {sys.version}') $$ LANGUAGE plpython3u;"

# Ensure SteadyText is installed for the correct Python version
python3 -m pip show steadytext

# If using system packages, ensure they're accessible
sudo python3 -m pip install --system steadytext
```

### Debug Mode
Enable verbose logging to diagnose issues:

```sql
-- Enable notices for debugging
SET client_min_messages TO NOTICE;

-- Re-initialize to see debug output
SELECT _steadytext_init_python();

-- Test with verbose output
SELECT steadytext_generate('test', 10);
```

## Contributing

Contributions are welcome! Please see the main [SteadyText repository](https://github.com/julep-ai/steadytext) for contribution guidelines.

## License

This extension is released under the PostgreSQL License. See LICENSE file for details.

## Support

- GitHub Issues: https://github.com/julep-ai/steadytext/issues
- Documentation: https://github.com/julep-ai/steadytext/tree/main/pg_steadytext

---

**AIDEV-NOTE**: This extension is designed to be a thin PostgreSQL wrapper around SteadyText, leveraging its existing daemon architecture and caching system rather than reimplementing functionality.
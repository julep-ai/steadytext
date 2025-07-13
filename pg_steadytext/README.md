# pg_steadytext - PostgreSQL Extension for SteadyText

**pg_steadytext** is a PostgreSQL extension that provides deterministic text generation and embeddings by integrating with the [SteadyText](https://github.com/julep-ai/steadytext) library. It offers SQL functions for text generation, embedding creation, and intelligent caching with frecency-based eviction.

## Features

- **Deterministic Text Generation**: Always returns the same output for the same input
- **Vector Embeddings**: Generate 1024-dimensional embeddings compatible with pgvector
- **Built-in Caching**: PostgreSQL-based frecency cache that mirrors SteadyText's cache
- **Daemon Integration**: Seamlessly integrates with SteadyText's ZeroMQ daemon
- **Async Processing**: Queue-based asynchronous generation and embedding with background workers
- **Security**: Input validation and rate limiting
- **Monitoring**: Health checks and performance statistics
- **AI Summarization**: Aggregate functions for intelligent text summarization with TimescaleDB support (v1.1.0+)

## Requirements

- PostgreSQL 14+ 
- Python 3.10+
- Extensions:
  - `plpython3u` (required)
  - `pgvector` (required)
- Python packages:
  - `steadytext` (install with `pip3 install steadytext`)

## Installation

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

### Quick Install

```bash
# Install Python dependencies
pip3 install steadytext pyzmq numpy

# Clone and install the extension
git clone https://github.com/julep-ai/steadytext.git
cd steadytext/pg_steadytext
make && sudo make install

# In PostgreSQL
CREATE EXTENSION pg_steadytext CASCADE;
```

### Docker Install (Recommended)

```bash
# Build Docker image with pg_steadytext
docker build -t pg_steadytext .
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pg_steadytext
```

See [INSTALL.md](INSTALL.md) for complete instructions including troubleshooting.

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

### Structured Generation (v2.4.1+)

```sql
-- Generate JSON
SELECT steadytext_generate_json(
    'Create a person named Alice, age 30',
    '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'::jsonb
);

-- Generate text matching regex
SELECT steadytext_generate_regex(
    'Phone: ',
    '\d{3}-\d{3}-\d{4}'
);

-- Generate from choices
SELECT steadytext_generate_choice(
    'The sentiment is',
    ARRAY['positive', 'negative', 'neutral']
);
```

### AI Summarization (v1.1.0+)

```sql
-- Summarize a single text
SELECT ai_summarize_text(
    'PostgreSQL provides ACID compliance and supports complex queries with JSON.',
    '{"source": "documentation"}'::jsonb
);

-- Aggregate summarization
SELECT 
    category,
    ai_summarize(content, jsonb_build_object('importance', importance)) as summary,
    count(*) as doc_count
FROM documents
GROUP BY category;

-- Use with TimescaleDB continuous aggregates
CREATE MATERIALIZED VIEW hourly_log_summaries
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS hour,
    log_level,
    ai_summarize_partial(
        message,
        jsonb_build_object('severity', severity, 'service', service_name)
    ) AS partial_summary,
    count(*) as log_count
FROM application_logs
GROUP BY hour, log_level;

-- Query continuous aggregate with final summarization
SELECT 
    time_bucket('1 day', hour) as day,
    log_level,
    ai_summarize_final(partial_summary) as daily_summary,
    sum(log_count) as total_logs
FROM hourly_log_summaries
WHERE hour >= NOW() - INTERVAL '7 days'
GROUP BY day, log_level;

-- Extract facts from text
SELECT ai_extract_facts(
    'PostgreSQL supports JSON, arrays, full-text search, and has built-in replication.',
    5  -- max facts
);
```

### Document Reranking (v1.3.0+)

```sql
-- Rerank documents by relevance to query
SELECT * FROM steadytext_rerank(
    'Python programming',
    ARRAY[
        'Python is a programming language',
        'Cats are cute animals',
        'Python snakes are found in Asia'
    ]
);

-- Get only reranked documents (no scores)
SELECT * FROM steadytext_rerank_docs_only(
    'machine learning',
    ARRAY(SELECT content FROM documents WHERE category = 'tech')
);

-- Get top 5 most relevant documents
SELECT * FROM steadytext_rerank_top_k(
    'customer complaint',
    ARRAY(SELECT ticket_text FROM support_tickets),
    5
);

-- Batch reranking for multiple queries
SELECT * FROM steadytext_rerank_batch(
    ARRAY['query1', 'query2', 'query3'],
    ARRAY['doc1', 'doc2', 'doc3']
);
```

### Async Functions (v1.1.0+)

```sql
-- Queue async generation
SELECT request_id FROM steadytext_generate_async('Write a story about space');

-- Queue async structured generation
SELECT steadytext_generate_json_async(
    'Create a product',
    '{"type": "object", "properties": {"name": {"type": "string"}, "price": {"type": "number"}}}'::jsonb
);

-- Check async status
SELECT * FROM steadytext_check_async('your-request-id'::uuid);

-- Get result (blocks until ready)
SELECT steadytext_get_async_result('your-request-id'::uuid, timeout_seconds := 30);

-- Batch operations
SELECT unnest(steadytext_generate_batch_async(
    ARRAY['Prompt 1', 'Prompt 2', 'Prompt 3']
));
```

See [docs/ASYNC_FUNCTIONS.md](docs/ASYNC_FUNCTIONS.md) for complete async documentation.


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

### Structured Generation Functions (v2.4.1+)
- `steadytext_generate_json(prompt, schema, max_tokens, use_cache, seed)` - Generate JSON conforming to schema
- `steadytext_generate_regex(prompt, pattern, max_tokens, use_cache, seed)` - Generate text matching regex
- `steadytext_generate_choice(prompt, choices, max_tokens, use_cache, seed)` - Generate one of the choices

### AI Summarization Functions (v1.1.0+)
- `ai_summarize(text, metadata)` - Aggregate function for text summarization
- `ai_summarize_partial(text, metadata)` - Partial aggregate for TimescaleDB continuous aggregates
- `ai_summarize_final(jsonb)` - Final aggregate for completing partial summaries
- `ai_summarize_text(text, metadata)` - Convenience function for single-value summarization
- `ai_extract_facts(text, max_facts)` - Extract structured facts from text
- `ai_deduplicate_facts(jsonb, similarity_threshold)` - Deduplicate facts using semantic similarity

### Document Reranking Functions (v1.3.0+)
- `steadytext_rerank(query, documents)` - Rerank documents with relevance scores
- `steadytext_rerank_docs_only(query, documents)` - Rerank and return only documents
- `steadytext_rerank_top_k(query, documents, k)` - Return top K reranked documents
- `steadytext_rerank_async(query, documents)` - Async reranking operation
- `steadytext_rerank_batch(queries, documents)` - Batch reranking for multiple queries
- `steadytext_rerank_batch_async(queries, documents)` - Async batch reranking

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

## Testing

### Running Tests

The extension includes comprehensive test suites using both PostgreSQL regression tests and pgTAP.

#### pgTAP Tests (Recommended)

pgTAP provides a TAP (Test Anything Protocol) testing framework for PostgreSQL.

```bash
# Install pgTAP (if not already installed)
sudo apt-get install postgresql-17-pgtap  # Adjust version as needed

# Run all pgTAP tests
make test-pgtap

# Run with verbose output
make test-pgtap-verbose

# Run with TAP output for CI integration
make test-pgtap-tap

# Run specific test file
./run_pgtap_tests.sh test/pgtap/01_basic.sql
```

#### Legacy Regression Tests

```bash
# Run traditional regression tests
make test

# Run all test suites
make test-all
```

#### Test Coverage

The test suite covers:
- Basic extension functionality
- Text generation and determinism
- Embedding generation and normalization  
- Async queue operations
- Structured generation (JSON, regex, choice)
- Cache management and eviction
- Daemon integration
- Streaming text generation
- Input validation and error handling

### Writing New Tests

Create new pgTAP tests in `test/pgtap/` following the naming convention `NN_description.sql`:

```sql
-- test/pgtap/99_custom.sql
BEGIN;
SELECT plan(3);  -- Number of tests

-- Test function exists
SELECT has_function('my_function');

-- Test behavior
SELECT is(my_function('input'), 'expected', 'Description');

-- Test error handling
SELECT throws_ok(
    $$ SELECT my_function(NULL) $$,
    'P0001',
    'Error message',
    'Should fail with null input'
);

SELECT * FROM finish();
ROLLBACK;
```

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
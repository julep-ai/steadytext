
# PostgreSQL Extension (pg_steadytext)

> **Warning: Experimental Feature**
>
> The `pg_steadytext` extension is currently experimental. Its APIs and behavior may change in future releases. Use in production environments with caution.

`pg_steadytext` brings the power of SteadyText's deterministic AI and embedding models directly into your PostgreSQL database. This allows you to generate text, create embeddings, and perform vector similarity searches seamlessly within your SQL queries.

The extension is designed for performance and safety, running the intensive model inference in a separate daemon process to avoid blocking the database.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Docker Installation](#docker-installation)
  - [Manual Installation](#manual-installation)
  - [Cloud Deployments](#cloud-deployments)
- [Usage Examples](#usage-examples)
  - [Text Generation](#text-generation)
  - [Embeddings](#embeddings)
  - [Daemon Management](#daemon-management)
  - [Configuration](#configuration)
- [Advanced Features](#advanced-features)
  - [Batch Operations](#batch-operations)
  - [Custom Seeds](#custom-seeds)
  - [Vector Search](#vector-search)
  - [Performance Optimization](#performance-optimization)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Migration Guide](#migration-guide)
- [Best Practices](#best-practices)

## Features

- **In-Database AI:** Access text generation and embedding functions directly from SQL.
- **Vector Support:** Integrates with the `pgvector` extension for storing and querying embeddings.
- **Daemon-Based:** Model inference is handled by a background daemon for non-blocking performance.
- **Configurable:** All settings are managed within a standard PostgreSQL table.
- **Easy Deployment:** Includes a Docker setup for a ready-to-use environment.

## Installation

There are two primary ways to install and use `pg_steadytext`:

1.  **Docker (Recommended):** Use the provided Docker setup for a self-contained environment.
2.  **Manual Installation:** Build and install the extension on your own PostgreSQL server.

### Docker Installation

The included `Dockerfile` builds a PostgreSQL image with `pg_steadytext` and all its dependencies pre-installed.

1.  **Build the Docker Image:**

    ```bash
    cd pg_steadytext
    docker build -t pg_steadytext .
    ```

2.  **Run the Docker Container:**

    ```bash
    docker run -d -p 5432:5432 --name pg_steadytext_container pg_steadytext
    ```

3.  **Connect to the Database:**

    You can now connect to the PostgreSQL instance using any standard client, such as `psql`:

    ```bash
    psql -h localhost -p 5432 -U postgres
    ```

    The password is `postgres`, as defined in the `Dockerfile`. The extension is automatically created and ready to use.

### Manual Installation

For manual installation, you'll need a PostgreSQL server with `pgvector` and `plpython3u` already installed.

1.  **Install Dependencies:**

    Ensure you have the necessary development packages and Python libraries:

    ```bash
    # System packages (example for Debian/Ubuntu)
    sudo apt-get update
    sudo apt-get install postgresql-server-dev-all postgresql-plpython3-17 python3-pip

    # Python packages
    pip3 install steadytext pyzmq numpy
    ```

2.  **Build and Install the Extension:**

    Run the installation script from the `pg_steadytext` directory:

    ```bash
    cd pg_steadytext
    ./install.sh
    ```

    The script will guide you through the process and may prompt for `sudo` access to install files into the PostgreSQL directories.

3.  **Enable the Extension in Your Database:**

    Connect to your database as a superuser and run the following commands:

    ```sql
    CREATE EXTENSION IF NOT EXISTS plpython3u;
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION pg_steadytext;
    ```

## Usage Examples

Once installed, you can use the `steadytext` functions in your SQL queries.

### Text Generation

The `steadytext_generate` function creates text based on a prompt.

```sql
-- Basic text generation
SELECT steadytext_generate('The capital of France is');

-- With custom parameters
SELECT steadytext_generate(
    'Write a short poem about a robot:',
    max_tokens := 100,
    seed := 123
);
```

### Embeddings

The `steadytext_embed` function creates a vector embedding for a given text. This requires the `pgvector` extension.

```sql
-- Create a table with a vector column
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    name TEXT,
    embedding VECTOR(1024)
);

-- Generate and insert an embedding
INSERT INTO items (name, embedding)
VALUES (
    'A futuristic city',
    steadytext_embed('A bustling metropolis with flying cars and neon lights.')
);

-- Find similar items using vector similarity search
SELECT name, 1 - (embedding <=> (SELECT embedding FROM items WHERE name = 'A futuristic city')) AS similarity
FROM items
ORDER BY similarity DESC
LIMIT 5;
```

### Daemon Management

The extension provides functions to manage the background daemon process.

```sql
-- Check the status of the daemon
SELECT * FROM steadytext_daemon_status();

-- Start the daemon (if not already running)
SELECT steadytext_daemon_start();

-- Stop the daemon
SELECT steadytext_daemon_stop();
```

### Configuration

You can view and modify the extension's configuration using the `steadytext_config` table.

```sql
-- View current configuration
SELECT * FROM steadytext_config;

-- Change the default number of tokens for generation
SELECT steadytext_config_set('default_max_tokens', '1024');

-- Change the default seed for deterministic output
SELECT steadytext_config_set('default_seed', '42');

-- Configure daemon connection settings
SELECT steadytext_config_set('daemon_host', 'localhost');
SELECT steadytext_config_set('daemon_port', '5557');
```

## Advanced Features

### Batch Operations

Process multiple texts efficiently in a single query.

```sql
-- Create a batch processing function
CREATE OR REPLACE FUNCTION batch_generate_summaries(
    texts TEXT[],
    max_tokens INT DEFAULT 100
) RETURNS TABLE(original TEXT, summary TEXT) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        unnest(texts) AS original,
        steadytext_generate(
            'Summarize: ' || unnest(texts),
            max_tokens := max_tokens
        ) AS summary;
END;
$$ LANGUAGE plpgsql;

-- Use the batch function
SELECT * FROM batch_generate_summaries(
    ARRAY[
        'PostgreSQL is a powerful, open source object-relational database system...',
        'Machine learning is a subset of artificial intelligence...',
        'Docker containers package software and its dependencies...'
    ]
);

-- Batch embedding generation
CREATE OR REPLACE FUNCTION batch_embed_texts(
    texts TEXT[]
) RETURNS TABLE(text TEXT, embedding VECTOR(1024)) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        unnest(texts) AS text,
        steadytext_embed(unnest(texts)) AS embedding;
END;
$$ LANGUAGE plpgsql;
```

### Custom Seeds

Ensure reproducible results across queries with custom seeds.

```sql
-- Create a function that always generates the same output for the same input
CREATE OR REPLACE FUNCTION deterministic_expand(
    topic TEXT,
    seed INT DEFAULT 42
) RETURNS TEXT AS $$
BEGIN
    RETURN steadytext_generate(
        'Write a detailed explanation about ' || topic,
        max_tokens := 200,
        seed := seed
    );
END;
$$ LANGUAGE plpgsql;

-- Same input + same seed = same output
SELECT deterministic_expand('quantum computing', 123);
SELECT deterministic_expand('quantum computing', 123); -- Identical result

-- Different seeds produce different outputs
SELECT deterministic_expand('quantum computing', 456); -- Different result

-- Create deterministic embeddings
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1024),
    seed INT DEFAULT 42
);

-- Insert with custom seed for reproducibility
INSERT INTO documents (content, embedding, seed)
VALUES (
    'Introduction to machine learning',
    steadytext_embed('Introduction to machine learning', seed := 100),
    100
);
```

### Vector Search

Advanced vector similarity search with pgvector integration.

```sql
-- Create a products table with embeddings
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT,
    price DECIMAL(10,2),
    embedding VECTOR(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient similarity search
CREATE INDEX ON products USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Function to add product with auto-generated embedding
CREATE OR REPLACE FUNCTION add_product(
    p_name TEXT,
    p_description TEXT,
    p_category TEXT,
    p_price DECIMAL
) RETURNS INTEGER AS $$
DECLARE
    v_id INTEGER;
    v_combined_text TEXT;
BEGIN
    -- Combine fields for richer embedding
    v_combined_text := p_name || ' ' || p_description || ' Category: ' || p_category;
    
    INSERT INTO products (name, description, category, price, embedding)
    VALUES (
        p_name,
        p_description,
        p_category,
        p_price,
        steadytext_embed(v_combined_text)
    )
    RETURNING id INTO v_id;
    
    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Semantic product search function
CREATE OR REPLACE FUNCTION search_products(
    query_text TEXT,
    limit_results INT DEFAULT 10,
    category_filter TEXT DEFAULT NULL
) RETURNS TABLE(
    product_id INT,
    product_name TEXT,
    product_description TEXT,
    similarity FLOAT
) AS $$
DECLARE
    query_embedding VECTOR(1024);
BEGIN
    -- Generate embedding for search query
    query_embedding := steadytext_embed(query_text);
    
    RETURN QUERY
    SELECT 
        p.id AS product_id,
        p.name AS product_name,
        p.description AS product_description,
        1 - (p.embedding <=> query_embedding) AS similarity
    FROM products p
    WHERE (category_filter IS NULL OR p.category = category_filter)
    ORDER BY p.embedding <=> query_embedding
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;

-- Example usage
SELECT * FROM add_product(
    'Eco-Friendly Water Bottle',
    'Stainless steel water bottle with vacuum insulation',
    'Kitchenware',
    24.99
);

-- Search for similar products
SELECT * FROM search_products('sustainable drinking container', 5);
```

### Performance Optimization

Optimize pg_steadytext for production workloads.

```sql
-- Connection pooling for daemon requests
CREATE OR REPLACE FUNCTION steadytext_warmup() RETURNS VOID AS $$
BEGIN
    -- Pre-warm the daemon connection
    PERFORM steadytext_generate('warmup', max_tokens := 1);
    PERFORM steadytext_embed('warmup');
END;
$$ LANGUAGE plpgsql;

-- Call on database startup
SELECT steadytext_warmup();

-- Caching frequently used embeddings
CREATE TABLE embedding_cache (
    text_hash BYTEA PRIMARY KEY,
    text_content TEXT,
    embedding VECTOR(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hit_count INT DEFAULT 0
);

-- Cached embedding function
CREATE OR REPLACE FUNCTION cached_embed(input_text TEXT) 
RETURNS VECTOR(1024) AS $$
DECLARE
    text_hash BYTEA;
    cached_embedding VECTOR(1024);
BEGIN
    -- Generate hash of input text
    text_hash := digest(input_text, 'sha256');
    
    -- Check cache
    SELECT embedding INTO cached_embedding
    FROM embedding_cache
    WHERE embedding_cache.text_hash = cached_embed.text_hash;
    
    IF FOUND THEN
        -- Update hit count
        UPDATE embedding_cache 
        SET hit_count = hit_count + 1
        WHERE embedding_cache.text_hash = cached_embed.text_hash;
        
        RETURN cached_embedding;
    ELSE
        -- Generate new embedding
        cached_embedding := steadytext_embed(input_text);
        
        -- Store in cache
        INSERT INTO embedding_cache (text_hash, text_content, embedding)
        VALUES (text_hash, input_text, cached_embedding)
        ON CONFLICT (text_hash) DO NOTHING;
        
        RETURN cached_embedding;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Parallel processing for large datasets
CREATE OR REPLACE FUNCTION parallel_embed_documents(
    table_name TEXT,
    text_column TEXT,
    embedding_column TEXT
) RETURNS VOID AS $$
BEGIN
    EXECUTE format(
        'UPDATE %I SET %I = steadytext_embed(%I) WHERE %I IS NULL',
        table_name, embedding_column, text_column, embedding_column
    );
END;
$$ LANGUAGE plpgsql;
```

## API Reference

### Core Functions

#### steadytext_generate

Generate text based on a prompt.

```sql
steadytext_generate(
    prompt TEXT,
    max_tokens INT DEFAULT NULL,
    seed INT DEFAULT NULL,
    temperature FLOAT DEFAULT NULL,
    model_size TEXT DEFAULT NULL
) RETURNS TEXT
```

**Parameters:**
- `prompt`: The input text to generate from
- `max_tokens`: Maximum number of tokens to generate (default: from config)
- `seed`: Random seed for deterministic output (default: from config)
- `temperature`: Controls randomness (0.0 = deterministic, default: 0.0)
- `model_size`: Model size to use ('small' or 'large', default: from config)

**Returns:** Generated text string

**Example:**
```sql
SELECT steadytext_generate(
    'Explain database normalization',
    max_tokens := 150,
    seed := 42
);
```

#### steadytext_embed

Create a vector embedding for text.

```sql
steadytext_embed(
    text TEXT,
    seed INT DEFAULT NULL,
    model_size TEXT DEFAULT NULL
) RETURNS VECTOR(1024)
```

**Parameters:**
- `text`: The input text to embed
- `seed`: Random seed for deterministic embeddings (default: from config)
- `model_size`: Model size to use (default: from config)

**Returns:** 1024-dimensional embedding vector

**Example:**
```sql
SELECT steadytext_embed('PostgreSQL database', seed := 42);
```

#### steadytext_generate_stream

Generate text with streaming support (returns results incrementally).

```sql
steadytext_generate_stream(
    prompt TEXT,
    max_tokens INT DEFAULT NULL,
    seed INT DEFAULT NULL
) RETURNS SETOF TEXT
```

**Returns:** Set of text chunks as they are generated

**Example:**
```sql
-- Get streamed output
SELECT * FROM steadytext_generate_stream('Write a story about AI');
```

### Daemon Management Functions

#### steadytext_daemon_status

Get the current status of the daemon.

```sql
steadytext_daemon_status() RETURNS TABLE(
    running BOOLEAN,
    pid INT,
    host TEXT,
    port INT,
    uptime_seconds INT,
    requests_processed BIGINT,
    errors_count BIGINT,
    avg_latency_ms FLOAT
)
```

#### steadytext_daemon_start

Start the daemon if not running.

```sql
steadytext_daemon_start() RETURNS BOOLEAN
```

**Returns:** TRUE if daemon started successfully

#### steadytext_daemon_stop

Stop the daemon.

```sql
steadytext_daemon_stop() RETURNS BOOLEAN
```

**Returns:** TRUE if daemon stopped successfully

#### steadytext_daemon_restart

Restart the daemon.

```sql
steadytext_daemon_restart() RETURNS BOOLEAN
```

### Configuration Functions

#### steadytext_config_get

Get a configuration value.

```sql
steadytext_config_get(key TEXT) RETURNS TEXT
```

#### steadytext_config_set

Set a configuration value.

```sql
steadytext_config_set(key TEXT, value TEXT) RETURNS VOID
```

**Available configuration keys:**
- `daemon_host`: Daemon host address (default: 'localhost')
- `daemon_port`: Daemon port number (default: '5557')
- `daemon_timeout_ms`: Request timeout in milliseconds (default: '30000')
- `default_max_tokens`: Default max tokens for generation (default: '512')
- `default_seed`: Default seed for deterministic output (default: '42')
- `default_model_size`: Default model size (default: 'small')
- `cache_enabled`: Enable result caching (default: 'true')
- `cache_ttl_seconds`: Cache time-to-live (default: '3600')

### Utility Functions

#### steadytext_version

Get the extension version.

```sql
steadytext_version() RETURNS TEXT
```

#### steadytext_model_info

Get information about loaded models.

```sql
steadytext_model_info() RETURNS TABLE(
    model_type TEXT,
    model_size TEXT,
    loaded BOOLEAN,
    memory_mb INT
)
```

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────┐
│              PostgreSQL Server                   │
├─────────────────────────────────────────────────┤
│         pg_steadytext Extension                  │
│  ┌─────────────────────────────────────────┐   │
│  │  SQL Functions (PL/Python)               │   │
│  │  - steadytext_generate()                 │   │
│  │  - steadytext_embed()                    │   │
│  │  - Daemon management                     │   │
│  └─────────────────────────────────────────┘   │
│                      │                           │
│                      │ ZeroMQ                    │
│                      ▼                           │
├─────────────────────────────────────────────────┤
│         SteadyText Daemon Process               │
│  ┌─────────────────────────────────────────┐   │
│  │  Model Manager                           │   │
│  │  - Gemma-3n (generation)                 │   │
│  │  - Qwen3 (embeddings)                    │   │
│  └─────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────┐   │
│  │  Request Handler                         │   │
│  │  - ZeroMQ REP server                     │   │
│  │  - Request queuing                       │   │
│  │  - Response caching                      │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Communication Flow

1. **SQL Query**: User executes a SQL function like `steadytext_generate()`
2. **Extension Layer**: PL/Python function prepares the request
3. **ZeroMQ Request**: Request sent to daemon via ZeroMQ REQ socket
4. **Daemon Processing**: Daemon processes request using loaded models
5. **Response**: Result returned through ZeroMQ to PostgreSQL
6. **SQL Result**: Function returns the result to the SQL query

### Security Model

- **Process Isolation**: Models run in separate daemon process
- **Communication**: Secure local ZeroMQ sockets
- **Access Control**: PostgreSQL's standard role-based access
- **Resource Limits**: Configurable timeouts and limits

## Troubleshooting

### Common Issues

#### Extension Creation Fails

```sql
ERROR: could not load library "/usr/lib/postgresql/17/lib/pg_steadytext.so"
```

**Solution:**
- Verify Python packages are installed: `pip3 show steadytext pyzmq`
- Check PostgreSQL Python support: `CREATE EXTENSION plpython3u;`
- Ensure proper permissions on extension files

#### Daemon Not Responding

```sql
ERROR: SteadyText daemon is not responding
```

**Solution:**
```sql
-- Check daemon status
SELECT * FROM steadytext_daemon_status();

-- Restart daemon
SELECT steadytext_daemon_restart();

-- Check logs
SELECT * FROM steadytext_logs ORDER BY timestamp DESC LIMIT 10;
```

#### Model Loading Errors

```sql
ERROR: Failed to load generation model
```

**Solution:**
```bash
# Ensure models are downloaded
st models download --all

# Check model directory permissions
ls -la ~/.cache/steadytext/models/
```

#### Performance Issues

```sql
-- Check query performance
EXPLAIN ANALYZE SELECT steadytext_generate('test prompt');

-- Monitor daemon performance
SELECT * FROM steadytext_daemon_status();

-- Check cache hit rate
SELECT 
    cache_hits::float / (cache_hits + cache_misses) AS hit_rate
FROM steadytext_stats;
```

### Debug Mode

Enable debug logging for troubleshooting:

```sql
-- Enable debug mode
SELECT steadytext_config_set('debug_mode', 'true');

-- View debug logs
SELECT * FROM steadytext_debug_logs 
ORDER BY timestamp DESC 
LIMIT 50;

-- Disable debug mode
SELECT steadytext_config_set('debug_mode', 'false');
```

## Migration Guide

### From Python to PostgreSQL

If you're currently using SteadyText in Python and want to migrate to pg_steadytext:

```python
# Python code
import steadytext

def generate_product_description(name, features):
    prompt = f"Write a product description for {name} with features: {features}"
    return steadytext.generate(prompt, seed=42)

def create_product_embedding(description):
    return steadytext.embed(description, seed=42)
```

Equivalent PostgreSQL:

```sql
-- PostgreSQL equivalent
CREATE OR REPLACE FUNCTION generate_product_description(
    name TEXT,
    features TEXT
) RETURNS TEXT AS $$
DECLARE
    prompt TEXT;
BEGIN
    prompt := format('Write a product description for %s with features: %s', name, features);
    RETURN steadytext_generate(prompt, seed := 42);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION create_product_embedding(
    description TEXT
) RETURNS VECTOR(1024) AS $$
BEGIN
    RETURN steadytext_embed(description, seed := 42);
END;
$$ LANGUAGE plpgsql;
```

### Batch Migration

Migrate existing data to use embeddings:

```sql
-- Add embedding column to existing table
ALTER TABLE products ADD COLUMN embedding VECTOR(1024);

-- Create migration function
CREATE OR REPLACE FUNCTION migrate_embeddings(
    batch_size INT DEFAULT 100
) RETURNS VOID AS $$
DECLARE
    batch_count INT := 0;
    total_count INT;
BEGIN
    SELECT COUNT(*) INTO total_count 
    FROM products 
    WHERE embedding IS NULL;
    
    RAISE NOTICE 'Migrating % products', total_count;
    
    WHILE EXISTS (SELECT 1 FROM products WHERE embedding IS NULL) LOOP
        UPDATE products
        SET embedding = steadytext_embed(name || ' ' || COALESCE(description, ''))
        WHERE id IN (
            SELECT id FROM products
            WHERE embedding IS NULL
            LIMIT batch_size
        );
        
        batch_count := batch_count + 1;
        RAISE NOTICE 'Processed batch %', batch_count;
        
        -- Optional: Add delay to avoid overloading
        PERFORM pg_sleep(0.1);
    END LOOP;
    
    RAISE NOTICE 'Migration complete';
END;
$$ LANGUAGE plpgsql;

-- Run migration
SELECT migrate_embeddings(100);
```

## Best Practices

### 1. Use Prepared Statements

For repeated operations, use prepared statements:

```sql
-- Prepare statements for better performance
PREPARE generate_stmt (TEXT, INT) AS
    SELECT steadytext_generate($1, max_tokens := $2);

PREPARE embed_stmt (TEXT) AS
    SELECT steadytext_embed($1);

-- Execute prepared statements
EXECUTE generate_stmt('Write about AI', 100);
EXECUTE embed_stmt('Machine learning concepts');
```

### 2. Index Strategies

Optimize vector searches with appropriate indexes:

```sql
-- For small datasets (< 1M rows)
CREATE INDEX idx_embedding ON products 
USING ivfflat (embedding vector_cosine_ops);

-- For large datasets
CREATE INDEX idx_embedding ON products 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1000);

-- Partial indexes for filtered searches
CREATE INDEX idx_embedding_electronics ON products 
USING ivfflat (embedding vector_cosine_ops)
WHERE category = 'Electronics';
```

### 3. Connection Pooling

Use connection pooling for high-concurrency applications:

```sql
-- Configure pgBouncer or similar with:
-- - pool_mode = transaction
-- - default_pool_size = 25
-- - max_client_conn = 100
```

### 4. Monitoring

Set up monitoring for production deployments:

```sql
-- Create monitoring view
CREATE VIEW steadytext_metrics AS
SELECT 
    (SELECT COUNT(*) FROM pg_stat_activity WHERE query LIKE '%steadytext_%') as active_queries,
    (SELECT running FROM steadytext_daemon_status()) as daemon_running,
    (SELECT requests_processed FROM steadytext_daemon_status()) as total_requests,
    (SELECT avg_latency_ms FROM steadytext_daemon_status()) as avg_latency,
    (SELECT value FROM steadytext_config WHERE key = 'cache_hit_rate') as cache_hit_rate;

-- Query metrics
SELECT * FROM steadytext_metrics;
```

### 5. Security Considerations

```sql
-- Create dedicated role for AI operations
CREATE ROLE ai_user;
GRANT USAGE ON SCHEMA public TO ai_user;
GRANT EXECUTE ON FUNCTION steadytext_generate TO ai_user;
GRANT EXECUTE ON FUNCTION steadytext_embed TO ai_user;

-- Revoke dangerous functions
REVOKE EXECUTE ON FUNCTION steadytext_daemon_stop FROM ai_user;
REVOKE EXECUTE ON FUNCTION steadytext_config_set FROM ai_user;

-- Row-level security for embeddings
ALTER TABLE products ENABLE ROW LEVEL SECURITY;

CREATE POLICY embedding_access ON products
    FOR SELECT
    USING (embedding IS NOT NULL);
```

### 6. Production Checklist

- [ ] Configure appropriate resource limits
- [ ] Set up connection pooling
- [ ] Create indexes on vector columns
- [ ] Enable query logging for AI operations
- [ ] Set up monitoring and alerting
- [ ] Configure backup strategy for embeddings
- [ ] Test failover scenarios
- [ ] Document custom functions and workflows
- [ ] Set up regular embedding regeneration jobs
- [ ] Monitor disk space for vector storage

## Performance Tuning

### Query Optimization

```sql
-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM search_products('laptop', 10);

-- Optimize daemon connection
SELECT steadytext_config_set('daemon_timeout_ms', '5000');
SELECT steadytext_config_set('connection_pool_size', '10');

-- Batch processing for better throughput
WITH batch AS (
    SELECT id, description 
    FROM products 
    WHERE embedding IS NULL 
    LIMIT 1000
)
UPDATE products p
SET embedding = steadytext_embed(b.description)
FROM batch b
WHERE p.id = b.id;
```

### Resource Management

```sql
-- Set work_mem for vector operations
SET work_mem = '256MB';

-- Configure shared_buffers for vector data
-- In postgresql.conf:
-- shared_buffers = 4GB
-- effective_cache_size = 12GB

-- Monitor resource usage
SELECT 
    pid,
    usename,
    application_name,
    state,
    query_start,
    state_change,
    query
FROM pg_stat_activity
WHERE query LIKE '%steadytext_%'
ORDER BY query_start DESC;
```

## Future Enhancements

Planned features for future releases:

- **Streaming Generation**: Real-time text generation with partial results
- **Multi-Model Support**: Switch between different model sizes dynamically
- **Distributed Processing**: Scale across multiple daemon instances
- **GPU Acceleration**: Optional GPU support for faster inference
- **Custom Models**: Support for user-provided GGUF models
- **Async Operations**: Non-blocking function variants
- **Compression**: Automatic embedding compression for storage efficiency
- **Incremental Indexing**: Real-time index updates for new embeddings

## Support and Contributing

- **Issues**: Report bugs at [GitHub Issues](https://github.com/diwank/steadytext/issues)
- **Documentation**: Full docs at [steadytext.readthedocs.io](https://steadytext.readthedocs.io)
- **Community**: Join discussions on Discord
- **Contributing**: See CONTRIBUTING.md for guidelines

## License

pg_steadytext is released under the same license as SteadyText (MIT License).

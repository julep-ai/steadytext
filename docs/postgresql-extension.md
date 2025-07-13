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

### Structured Generation (v2.4.1+)

New in v2.4.1, the PostgreSQL extension now supports structured text generation using llama.cpp's native grammar support.

#### `steadytext_generate_json()`

Generate JSON that conforms to a JSON schema.

```sql
steadytext_generate_json(
    prompt TEXT,
    schema JSONB,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TEXT
-- Returns NULL if generation fails
```

**Examples:**

```sql
-- Simple JSON generation
SELECT steadytext_generate_json(
    'Create a user named John, age 30',
    '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'::jsonb
);

-- Generate product information
SELECT steadytext_generate_json(
    'Create a product listing for a laptop',
    '{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "specs": {
                "type": "object",
                "properties": {
                    "cpu": {"type": "string"},
                    "ram": {"type": "string"},
                    "storage": {"type": "string"}
                }
            }
        }
    }'::jsonb,
    seed := 999
);
```

#### `steadytext_generate_regex()`

Generate text that matches a regular expression pattern.

```sql
steadytext_generate_regex(
    prompt TEXT,
    pattern TEXT,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TEXT
-- Returns NULL if generation fails
```

**Examples:**

```sql
-- Generate a phone number
SELECT steadytext_generate_regex(
    'Contact number: ',
    '\d{3}-\d{3}-\d{4}'
);

-- Generate a date
SELECT steadytext_generate_regex(
    'Event date: ',
    '\d{4}-\d{2}-\d{2}'
);

-- Generate an email
SELECT steadytext_generate_regex(
    'Email: ',
    '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
);
```

#### `steadytext_generate_choice()`

Generate text that is one of the provided choices.

```sql
steadytext_generate_choice(
    prompt TEXT,
    choices TEXT[],
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TEXT
-- Returns NULL if generation fails
```

**Examples:**

```sql
-- Simple choice
SELECT steadytext_generate_choice(
    'The weather today is',
    ARRAY['sunny', 'cloudy', 'rainy']
);

-- Sentiment analysis
SELECT 
    review,
    steadytext_generate_choice(
        'Sentiment of this review: ' || review,
        ARRAY['positive', 'negative', 'neutral']
    ) AS sentiment
FROM product_reviews
LIMIT 5;

-- Classification with custom seed
SELECT steadytext_generate_choice(
    'This document is about',
    ARRAY['technology', 'business', 'health', 'sports', 'entertainment'],
    seed := 456
);
```

### Document Reranking (v1.3.0+)

PostgreSQL extension v1.3.0+ includes document reranking functionality powered by the Qwen3-Reranker-4B model.

#### `steadytext_rerank()`

Rerank documents by relevance to a query.

```sql
steadytext_rerank(
    query TEXT,
    documents TEXT[],
    task TEXT DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TABLE(document TEXT, score FLOAT)
```

**Examples:**

```sql
-- Basic reranking
SELECT * FROM steadytext_rerank(
    'Python programming',
    ARRAY[
        'Python is a programming language',
        'Cats are cute animals',
        'Python snakes are found in Asia'
    ]
);

-- Custom task description
SELECT * FROM steadytext_rerank(
    'customer complaint about delivery',
    ARRAY(SELECT ticket_text FROM support_tickets WHERE created_at > NOW() - INTERVAL '7 days'),
    task := 'support ticket prioritization'
);

-- Integration with search results
WITH search_results AS (
    SELECT content, ts_rank(search_vector, query) AS text_score
    FROM documents, plainto_tsquery('english', 'machine learning') query
    WHERE search_vector @@ query
    LIMIT 20
)
SELECT r.document, r.score as ai_score, s.text_score
FROM search_results s,
     LATERAL steadytext_rerank(
         'machine learning',
         ARRAY_AGG(s.content),
         seed := 456
     ) r
WHERE s.content = r.document
ORDER BY r.score DESC
LIMIT 5;
```

#### `steadytext_rerank_docs_only()`

Get reranked documents without scores.

```sql
steadytext_rerank_docs_only(
    query TEXT,
    documents TEXT[],
    task TEXT DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TABLE(document TEXT)
```

**Example:**

```sql
-- Get reranked documents for display
SELECT * FROM steadytext_rerank_docs_only(
    'machine learning',
    ARRAY(SELECT content FROM documents WHERE category = 'tech')
);
```

#### `steadytext_rerank_top_k()`

Get top K most relevant documents.

```sql
steadytext_rerank_top_k(
    query TEXT,
    documents TEXT[],
    k INTEGER,
    task TEXT DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TABLE(document TEXT, score FLOAT)
```

**Example:**

```sql
-- Get top 5 most relevant support tickets
SELECT * FROM steadytext_rerank_top_k(
    'refund request',
    ARRAY(SELECT ticket_text FROM support_tickets),
    5
);
```

#### `steadytext_rerank_batch()`

Batch reranking for multiple queries.

```sql
steadytext_rerank_batch(
    queries TEXT[],
    documents TEXT[],
    task TEXT DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TABLE(query_idx INTEGER, doc_idx INTEGER, score FLOAT)
```

**Example:**

```sql
-- Rerank documents for multiple queries
SELECT * FROM steadytext_rerank_batch(
    ARRAY['Python programming', 'machine learning', 'data science'],
    ARRAY['Python is great', 'ML algorithms', 'Data analysis with Python']
);
```

#### Async Reranking Functions

All reranking functions have async counterparts:

```sql
-- Queue async reranking
SELECT request_id FROM steadytext_rerank_async(
    'search query',
    ARRAY(SELECT content FROM documents)
);

-- Check status and get results
SELECT * FROM steadytext_check_async(request_id);
SELECT * FROM steadytext_get_async_result(request_id, timeout_seconds := 30);
```

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

## Advanced Usage

### Batch Operations

Process multiple prompts efficiently:

```sql
-- Batch generation with different seeds
WITH prompts AS (
    SELECT unnest(ARRAY[
        'Explain quantum computing',
        'Describe machine learning',
        'What is artificial intelligence'
    ]) AS prompt,
    unnest(ARRAY[100, 200, 300]) AS seed
)
SELECT 
    prompt,
    seed,
    steadytext_generate(prompt, seed := seed) AS response
FROM prompts;

-- Batch embeddings for similarity analysis
WITH texts AS (
    SELECT unnest(ARRAY[
        'artificial intelligence',
        'machine learning',
        'deep learning',
        'neural networks'
    ]) AS text
)
SELECT 
    text,
    steadytext_embed(text, seed := 42) AS embedding
FROM texts;
```

### Semantic Search Implementation

Build a semantic search system using pg_steadytext:

```sql
-- Create a documents table with embeddings
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index for fast similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Populate with embeddings using consistent seed (skip failed embeddings)
INSERT INTO documents (title, content, embedding)
SELECT 
    title,
    content,
    embedding
FROM (
    SELECT 
        title,
        content,
        steadytext_embed(content, seed := 42) AS embedding
    FROM source_documents
) WITH_EMBEDDINGS
WHERE embedding IS NOT NULL;

-- Semantic search function
CREATE OR REPLACE FUNCTION semantic_search(
    query_text TEXT,
    max_results INTEGER DEFAULT 5,
    search_seed INTEGER DEFAULT 42
)
RETURNS TABLE(id INTEGER, title TEXT, content TEXT, similarity REAL) AS $$
DECLARE
    query_embedding vector(1024);
BEGIN
    -- Generate query embedding with error handling
    query_embedding := steadytext_embed(query_text, seed := search_seed);
    
    -- Return empty result if embedding generation failed
    IF query_embedding IS NULL THEN
        RAISE WARNING 'Failed to generate embedding for query: %', query_text;
        RETURN;
    END IF;
    
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        d.content,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE d.embedding IS NOT NULL
    ORDER BY d.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT * FROM semantic_search('machine learning algorithms', 10);
```

### Content Generation Pipeline

Create a content generation workflow:

```sql
-- Content generation pipeline with different styles
CREATE OR REPLACE FUNCTION generate_content_variants(
    base_prompt TEXT,
    num_variants INTEGER DEFAULT 3
)
RETURNS TABLE(variant_id INTEGER, style TEXT, content TEXT) AS $$
DECLARE
    styles TEXT[] := ARRAY['formal', 'casual', 'technical'];
    i INTEGER;
    current_style TEXT;
    enhanced_prompt TEXT;
BEGIN
    FOR i IN 1..LEAST(num_variants, array_length(styles, 1)) LOOP
        current_style := styles[i];
        enhanced_prompt := format('Write in a %s style: %s', current_style, base_prompt);
        
        -- Generate content with error handling
        DECLARE
            generated_content TEXT;
        BEGIN
            generated_content := steadytext_generate(
                enhanced_prompt,
                max_tokens := 200,
                seed := 100 + i  -- Different seed for each variant
            );
            
            -- Skip variants that failed to generate
            IF generated_content IS NOT NULL THEN
                RETURN QUERY
                SELECT 
                    i AS variant_id,
                    current_style AS style,
                    generated_content AS content;
            ELSE
                RAISE WARNING 'Failed to generate content for style: %', current_style;
            END IF;
        END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT * FROM generate_content_variants('Explain the benefits of PostgreSQL');
```

## Performance Optimization

### Cache Strategy

```sql
-- Monitor cache performance
SELECT 
    'Generation Cache' AS cache_type,
    entries,
    total_size_mb,
    hit_rate,
    evictions
FROM steadytext_cache_stats()
WHERE cache_type = 'generation'
UNION ALL
SELECT 
    'Embedding Cache' AS cache_type,
    entries,
    total_size_mb,
    hit_rate,
    evictions
FROM steadytext_cache_stats()
WHERE cache_type = 'embedding';

-- Optimize cache by pre-warming common queries
WITH common_prompts AS (
    SELECT unnest(ARRAY[
        'Summarize the key points',
        'Explain this concept',
        'Generate a brief description'
    ]) AS prompt
)
SELECT 
    prompt,
    'Pre-warmed: ' || length(steadytext_generate(prompt)) || ' chars' AS status
FROM common_prompts;
```

### Daemon Performance

```sql
-- Check daemon performance metrics
SELECT 
    running,
    uptime,
    health_score,
    last_response_time_ms,
    total_requests,
    error_rate
FROM steadytext_daemon_status();

-- Restart daemon if performance degrades
DO $$
BEGIN
    IF (SELECT health_score FROM steadytext_daemon_status()) < 0.8 THEN
        PERFORM steadytext_daemon_stop();
        PERFORM pg_sleep(2);
        PERFORM steadytext_daemon_start();
        RAISE NOTICE 'Daemon restarted due to poor health score';
    END IF;
END;
$$;
```

## Security Considerations

### Input Validation

```sql
-- Safe text generation with input validation and NULL handling
CREATE OR REPLACE FUNCTION safe_generate(
    user_prompt TEXT,
    max_length INTEGER DEFAULT 512,
    custom_seed INTEGER DEFAULT 42
)
RETURNS TEXT AS $$
DECLARE
    result TEXT;
BEGIN
    -- Validate input length
    IF length(user_prompt) > 1000 THEN
        RAISE EXCEPTION 'Prompt too long (max 1000 characters)';
    END IF;
    
    -- Validate max_length
    IF max_length > 2048 OR max_length < 1 THEN
        RAISE EXCEPTION 'Invalid max_length (must be 1-2048)';
    END IF;
    
    -- Validate seed
    IF custom_seed < 0 THEN
        RAISE EXCEPTION 'Seed must be non-negative';
    END IF;
    
    -- Sanitize prompt (basic example)
    user_prompt := regexp_replace(user_prompt, '[<>]', '', 'g');
    
    -- Generate with error handling
    result := steadytext_generate(user_prompt, max_length, true, custom_seed);
    
    -- Return error message if generation failed
    IF result IS NULL THEN
        RETURN '[Error: Text generation failed. Please check system status.]';
    END IF;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant limited access
GRANT EXECUTE ON FUNCTION safe_generate TO app_user;
```

### Rate Limiting

```sql
-- Enable rate limiting
SELECT steadytext_config_set('rate_limit_enabled', 'true');
SELECT steadytext_config_set('max_requests_per_minute', '30');

-- Custom rate limiting per user
CREATE TABLE user_rate_limits (
    user_id INTEGER PRIMARY KEY,
    requests_made INTEGER DEFAULT 0,
    window_start TIMESTAMP DEFAULT NOW(),
    max_requests INTEGER DEFAULT 10
);

-- Rate-limited generation function
CREATE OR REPLACE FUNCTION rate_limited_generate(
    user_id INTEGER,
    prompt TEXT
)
RETURNS TEXT AS $$
DECLARE
    current_requests INTEGER;
    window_start TIMESTAMP;
    max_allowed INTEGER;
BEGIN
    -- Get or create rate limit record
    INSERT INTO user_rate_limits (user_id)
    VALUES (user_id)
    ON CONFLICT (user_id) DO NOTHING;
    
    -- Check current usage
    SELECT requests_made, window_start, max_requests
    INTO current_requests, window_start, max_allowed
    FROM user_rate_limits
    WHERE user_id = rate_limited_generate.user_id;
    
    -- Reset window if expired (1 hour)
    IF window_start < NOW() - INTERVAL '1 hour' THEN
        UPDATE user_rate_limits
        SET requests_made = 0, window_start = NOW()
        WHERE user_id = rate_limited_generate.user_id;
        current_requests := 0;
    END IF;
    
    -- Check rate limit
    IF current_requests >= max_allowed THEN
        RAISE EXCEPTION 'Rate limit exceeded. Try again later.';
    END IF;
    
    -- Increment counter
    UPDATE user_rate_limits
    SET requests_made = requests_made + 1
    WHERE user_id = rate_limited_generate.user_id;
    
    -- Generate text with error handling
    DECLARE
        result TEXT;
    BEGIN
        result := steadytext_generate(prompt, 512, true, 42);
        
        IF result IS NULL THEN
            RAISE EXCEPTION 'Text generation failed. Please try again later.';
        END IF;
        
        RETURN result;
    END;
END;
$$ LANGUAGE plpgsql;
```

## AI Summarization (v1.1.0+)

The PostgreSQL extension includes powerful AI summarization aggregate functions that work seamlessly with TimescaleDB continuous aggregates.

### Core Summarization Functions

#### `ai_summarize_text()`

Summarize a single text with optional metadata.

```sql
ai_summarize_text(
    text_input TEXT,
    metadata JSONB DEFAULT NULL,
    max_tokens INTEGER DEFAULT 150,
    seed INTEGER DEFAULT 42
) RETURNS TEXT
```

**Examples:**

```sql
-- Simple text summarization
SELECT ai_summarize_text(
    'PostgreSQL is an advanced open-source relational database with ACID compliance, 
     JSON support, and extensibility through custom functions and types.',
    '{"source": "documentation"}'::jsonb
);

-- Summarize with custom parameters
SELECT ai_summarize_text(
    content,
    jsonb_build_object('importance', importance, 'category', category),
    max_tokens := 200,
    seed := 123
) AS summary
FROM documents
WHERE length(content) > 1000;
```

#### `ai_summarize()` Aggregate Function

Intelligently summarize multiple texts into a coherent summary.

```sql
-- Basic aggregate summarization
SELECT 
    category,
    ai_summarize(content) AS category_summary,
    count(*) AS doc_count
FROM documents
GROUP BY category;

-- With metadata
SELECT 
    department,
    ai_summarize(
        report_text,
        jsonb_build_object('priority', priority, 'date', report_date)
    ) AS department_summary
FROM reports
WHERE report_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY department;
```

### Partial Aggregation for TimescaleDB

The extension supports partial aggregation for use with TimescaleDB continuous aggregates:

#### `ai_summarize_partial()` and `ai_summarize_final()`

```sql
-- Create continuous aggregate with partial summarization
CREATE MATERIALIZED VIEW hourly_log_summaries
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS hour,
    log_level,
    service_name,
    ai_summarize_partial(
        log_message,
        jsonb_build_object(
            'severity', severity,
            'service', service_name,
            'error_code', error_code
        )
    ) AS partial_summary,
    count(*) AS log_count
FROM application_logs
GROUP BY hour, log_level, service_name;

-- Query with final summarization
SELECT 
    time_bucket('1 day', hour) as day,
    log_level,
    ai_summarize_final(partial_summary) as daily_summary,
    sum(log_count) as total_logs
FROM hourly_log_summaries
WHERE hour >= NOW() - INTERVAL '7 days'
GROUP BY day, log_level
ORDER BY day DESC;
```

### Fact Extraction

#### `ai_extract_facts()`

Extract key facts from text content.

```sql
ai_extract_facts(
    text_input TEXT,
    max_facts INTEGER DEFAULT 5,
    seed INTEGER DEFAULT 42
) RETURNS TEXT[]
```

**Examples:**

```sql
-- Extract facts from a document
SELECT ai_extract_facts(
    'PostgreSQL supports JSON, arrays, full-text search, window functions, 
     CTEs, and has built-in replication. It also offers ACID compliance 
     and supports multiple programming languages for stored procedures.',
    max_facts := 7
);
-- Returns: {
--   "PostgreSQL supports JSON",
--   "PostgreSQL supports arrays",
--   "PostgreSQL has full-text search",
--   "PostgreSQL has window functions",
--   "PostgreSQL supports CTEs",
--   "PostgreSQL has built-in replication",
--   "PostgreSQL offers ACID compliance"
-- }

-- Extract facts from multiple documents
SELECT 
    doc_id,
    title,
    ai_extract_facts(content, 3) AS key_facts
FROM technical_docs
WHERE category = 'database'
LIMIT 10;
```

### Real-World Use Cases

#### Log Analysis Dashboard

```sql
-- Real-time error summarization
CREATE OR REPLACE VIEW error_summaries AS
SELECT 
    date_trunc('hour', timestamp) AS error_hour,
    service_name,
    ai_summarize(
        error_message,
        jsonb_build_object(
            'count', count(*),
            'unique_errors', count(DISTINCT error_code)
        )
    ) AS error_summary,
    array_agg(DISTINCT error_code) AS error_codes
FROM error_logs
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY error_hour, service_name
ORDER BY error_hour DESC;
```

#### Document Intelligence

```sql
-- Automatic document categorization and summarization
WITH doc_summaries AS (
    SELECT 
        document_id,
        ai_summarize_text(content) AS summary,
        ai_extract_facts(content, 5) AS key_facts
    FROM documents
    WHERE created_at >= CURRENT_DATE
)
SELECT 
    d.document_id,
    d.title,
    ds.summary,
    ds.key_facts,
    steadytext_generate_choice(
        'Category for document: ' || ds.summary,
        ARRAY['technical', 'business', 'legal', 'marketing', 'other']
    ) AS suggested_category
FROM documents d
JOIN doc_summaries ds ON d.document_id = ds.document_id;
```

## Async Functions (v1.1.0+)

The PostgreSQL extension includes asynchronous functions for non-blocking AI operations, perfect for high-throughput applications.

### Overview

Async functions return a UUID immediately and process requests in the background using a queue-based architecture with priority support.

### Core Async Functions

#### Generation Functions

```sql
-- Async text generation
steadytext_generate_async(
    prompt TEXT,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 5,
    seed INTEGER DEFAULT 42
) RETURNS UUID

-- Async JSON generation
steadytext_generate_json_async(
    prompt TEXT,
    schema JSONB,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS UUID

-- Async regex generation
steadytext_generate_regex_async(
    prompt TEXT,
    pattern TEXT,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS UUID

-- Async choice generation
steadytext_generate_choice_async(
    prompt TEXT,
    choices TEXT[],
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS UUID
```

#### Embedding Functions

```sql
-- Async embedding generation
steadytext_embed_async(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 5
) RETURNS UUID
```

#### Batch Operations

```sql
-- Batch text generation
steadytext_generate_batch_async(
    prompts TEXT[],
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true
) RETURNS UUID[]

-- Batch embedding generation
steadytext_embed_batch_async(
    texts TEXT[],
    use_cache BOOLEAN DEFAULT true
) RETURNS UUID[]
```

### Result Management

#### `steadytext_check_async()`

Check the status of an async request.

```sql
steadytext_check_async(request_id UUID)
RETURNS TABLE(
    status TEXT,           -- 'pending', 'processing', 'completed', 'failed'
    result TEXT,          -- Generated text (NULL if not completed)
    error TEXT,           -- Error message (NULL if successful)
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    processing_time_ms INTEGER
)
```

#### `steadytext_get_async_result()`

Wait for and retrieve the result of an async request.

```sql
steadytext_get_async_result(
    request_id UUID,
    timeout_seconds INTEGER DEFAULT 30
) RETURNS TEXT
-- Returns NULL on timeout or error
```

#### `steadytext_cancel_async()`

Cancel a pending async request.

```sql
steadytext_cancel_async(request_id UUID) RETURNS BOOLEAN
```

#### `steadytext_check_async_batch()`

Check multiple async requests at once.

```sql
steadytext_check_async_batch(request_ids UUID[])
RETURNS TABLE(
    request_id UUID,
    status TEXT,
    result TEXT,
    error TEXT
)
```

### Usage Examples

#### Basic Async Generation

```sql
-- Start async generation
SELECT request_id FROM steadytext_generate_async(
    'Write a comprehensive guide to PostgreSQL performance tuning',
    max_tokens := 1024
);

-- Check status
SELECT * FROM steadytext_check_async('your-request-id'::uuid);

-- Wait for result
SELECT steadytext_get_async_result('your-request-id'::uuid, 60);
```

#### Batch Processing Pattern

```sql
-- Process multiple documents asynchronously
WITH async_requests AS (
    SELECT 
        doc_id,
        steadytext_generate_async(
            'Summarize: ' || content,
            max_tokens := 200
        ) AS request_id
    FROM documents
    WHERE needs_summary = true
    LIMIT 100
)
-- Store request mappings
INSERT INTO document_summary_requests (doc_id, request_id, requested_at)
SELECT doc_id, request_id, NOW()
FROM async_requests;

-- Later: Collect results
UPDATE documents d
SET summary = r.result,
    summarized_at = NOW()
FROM (
    SELECT 
        dsr.doc_id,
        steadytext_get_async_result(dsr.request_id, 30) AS result
    FROM document_summary_requests dsr
    WHERE dsr.completed_at IS NULL
) r
WHERE d.doc_id = r.doc_id
  AND r.result IS NOT NULL;
```

#### Priority Queue Example

```sql
-- High priority requests
SELECT steadytext_generate_async(
    'URGENT: ' || request_text,
    priority := 10  -- Higher priority
) FROM urgent_requests;

-- Normal priority requests
SELECT steadytext_generate_async(
    request_text,
    priority := 5   -- Default priority
) FROM normal_requests;
```

#### LISTEN/NOTIFY Integration

```sql
-- Set up notification channel
LISTEN steadytext_async_complete;

-- Process completed requests
CREATE OR REPLACE FUNCTION process_completed_requests()
RETURNS void AS $$
DECLARE
    notification RECORD;
BEGIN
    -- Get completed notifications
    FOR notification IN 
        SELECT request_id::uuid 
        FROM pg_notification 
        WHERE channel = 'steadytext_async_complete'
    LOOP
        -- Process completed request
        UPDATE processed_texts
        SET result = (
            SELECT result 
            FROM steadytext_check_async(notification.request_id)
            WHERE status = 'completed'
        ),
        processed_at = NOW()
        WHERE request_id = notification.request_id;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

### Background Worker Configuration

The async functions use a background worker that can be configured:

```sql
-- Start the background worker
SELECT steadytext_worker_start();

-- Check worker status
SELECT * FROM steadytext_worker_status();

-- Stop the worker
SELECT steadytext_worker_stop();

-- Configure worker settings
SELECT steadytext_config_set('worker_batch_size', '10');
SELECT steadytext_config_set('worker_poll_interval_ms', '1000');
```

### Performance Considerations

1. **Queue Management**: The queue table (`steadytext_queue`) should be regularly monitored and old completed requests cleaned up.

2. **Priority Levels**: Use priority levels (1-10) wisely. Higher numbers get processed first.

3. **Batch Operations**: Batch operations are more efficient than individual async calls for large datasets.

4. **Result Polling**: Use LISTEN/NOTIFY instead of polling for better performance in real-time applications.

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

**Need Help?** Check the [main SteadyText documentation](https://github.com/julep-ai/steadytext) or open an issue on GitHub.
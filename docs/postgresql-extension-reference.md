# PostgreSQL Extension: Function Reference

Complete reference for all SteadyText PostgreSQL extension functions.

## Table of Contents

- [Core Functions](#core-functions)
- [Generation Functions](#generation-functions)
- [Embedding Functions](#embedding-functions)
- [Reranking Functions](#reranking-functions)
- [Structured Generation Functions](#structured-generation-functions)
- [Async Functions](#async-functions)
- [Utility Functions](#utility-functions)
- [Administrative Functions](#administrative-functions)
- [Deprecated Functions](#deprecated-functions)

## Core Functions

### steadytext_version()

Returns the current version of the SteadyText extension.

```sql
SELECT steadytext_version();
-- Returns: '1.1.0'
```

**Returns**: `TEXT` - Version string

### steadytext_health_check()

Performs a comprehensive health check of the extension.

```sql
SELECT * FROM steadytext_health_check();
-- Returns table with component status
```

**Returns**: `TABLE(component TEXT, status TEXT, details TEXT)`

## Generation Functions

### steadytext_generate()

Generate deterministic text from a prompt.

```sql
-- Basic usage
SELECT steadytext_generate('Write a function to sort an array');

-- With token limit
SELECT steadytext_generate('Explain quantum computing', 100);

-- With all parameters
SELECT steadytext_generate(
    prompt := 'Create a Python class',
    max_tokens := 200,
    use_cache := true,
    seed := 42
);
```

**Parameters**:
- `prompt` (TEXT, required): Input text prompt
- `max_tokens` (INTEGER, optional): Maximum tokens to generate (default: 512)
- `use_cache` (BOOLEAN, optional): Whether to use cache (default: true)
- `seed` (INTEGER, optional): Random seed for generation (default: 42)

**Returns**: `TEXT` - Generated text or NULL on error

### steadytext_generate_batch()

Generate text for multiple prompts efficiently.

```sql
-- Generate for array of prompts
SELECT * FROM steadytext_generate_batch(
    ARRAY['prompt1', 'prompt2', 'prompt3'],
    100
);

-- From table column
SELECT * FROM steadytext_generate_batch(
    ARRAY(SELECT title FROM articles),
    200
);
```

**Parameters**:
- `prompts` (TEXT[], required): Array of prompts
- `max_tokens` (INTEGER, optional): Maximum tokens per generation (default: 512)
- `use_cache` (BOOLEAN, optional): Whether to use cache (default: true)

**Returns**: `TABLE(idx INTEGER, prompt TEXT, generated_text TEXT)`

### steadytext_generate_table()

Generate text for each row in a query result.

```sql
-- Generate summaries for articles
SELECT * FROM steadytext_generate_table(
    'SELECT id, title, content FROM articles WHERE published = true',
    'Summarize: {content}',
    100
);
```

**Parameters**:
- `query` (TEXT, required): SQL query to execute
- `prompt_template` (TEXT, required): Template with {column} placeholders
- `max_tokens` (INTEGER, optional): Maximum tokens (default: 512)

**Returns**: `TABLE(row_data JSONB, generated_text TEXT)`

## Embedding Functions

### steadytext_embed()

Generate embedding vector for text.

```sql
-- Single text embedding
SELECT steadytext_embed('machine learning concepts');

-- With custom seed
SELECT steadytext_embed('data science', true, 123);
```

**Parameters**:
- `text` (TEXT, required): Text to embed
- `use_cache` (BOOLEAN, optional): Whether to use cache (default: true)
- `seed` (INTEGER, optional): Random seed (default: 42)

**Returns**: `vector(1024)` - 1024-dimensional embedding vector

### steadytext_embed_batch()

Generate embeddings for multiple texts.

```sql
-- Embed multiple texts
SELECT * FROM steadytext_embed_batch(
    ARRAY['text1', 'text2', 'text3']
);

-- From table
UPDATE documents 
SET embedding = batch.embedding
FROM (
    SELECT * FROM steadytext_embed_batch(
        ARRAY(SELECT content FROM documents WHERE embedding IS NULL)
    )
) batch
WHERE documents.content = batch.text;
```

**Parameters**:
- `texts` (TEXT[], required): Array of texts to embed
- `use_cache` (BOOLEAN, optional): Whether to use cache (default: true)

**Returns**: `TABLE(idx INTEGER, text TEXT, embedding vector(1024))`

### steadytext_embed_distance()

Calculate distance between two embeddings.

```sql
-- Cosine distance (default)
SELECT steadytext_embed_distance(
    steadytext_embed('cat'),
    steadytext_embed('dog')
);

-- Euclidean distance
SELECT steadytext_embed_distance(
    steadytext_embed('cat'),
    steadytext_embed('dog'),
    'euclidean'
);
```

**Parameters**:
- `embedding1` (vector(1024), required): First embedding
- `embedding2` (vector(1024), required): Second embedding
- `metric` (TEXT, optional): Distance metric ('cosine', 'euclidean', 'manhattan')

**Returns**: `FLOAT` - Distance value

## Reranking Functions

### steadytext_rerank()

Rerank documents by relevance to a query.

```sql
-- Basic reranking
SELECT * FROM steadytext_rerank(
    'machine learning tutorials',
    ARRAY['Intro to ML', 'Cat pictures', 'Deep learning guide']
);

-- With custom task description
SELECT * FROM steadytext_rerank(
    'patient symptoms: fever, headache',
    ARRAY['COVID guide', 'Common cold', 'Migraine info'],
    'Rank medical articles by relevance to symptoms'
);
```

**Parameters**:
- `query` (TEXT, required): Search query
- `documents` (TEXT[], required): Array of documents to rank
- `task_description` (TEXT, optional): Custom ranking instructions
- `use_cache` (BOOLEAN, optional): Whether to use cache (default: true)

**Returns**: `TABLE(position INTEGER, document TEXT, score FLOAT)`

### steadytext_rerank_batch()

Rerank documents for multiple queries.

```sql
SELECT * FROM steadytext_rerank_batch(
    ARRAY['query1', 'query2'],
    ARRAY[
        ARRAY['doc1', 'doc2'],
        ARRAY['doc3', 'doc4']
    ]
);
```

**Parameters**:
- `queries` (TEXT[], required): Array of queries
- `document_sets` (TEXT[][], required): 2D array of document sets
- `use_cache` (BOOLEAN, optional): Whether to use cache (default: true)

**Returns**: `TABLE(query_idx INTEGER, position INTEGER, document TEXT, score FLOAT)`

## Structured Generation Functions

### steadytext_generate_json()

Generate JSON output with schema validation.

```sql
-- Generate with JSON schema
SELECT steadytext_generate_json(
    'Create a person named Alice age 30',
    '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'
);

-- Extract JSON from result
SELECT result::json->>'name' as name
FROM (
    SELECT steadytext_generate_json(
        'Create user data',
        '{"type": "object", "properties": {"name": {"type": "string"}}}'
    ) as result
) t;
```

**Parameters**:
- `prompt` (TEXT, required): Generation prompt
- `schema` (JSON/TEXT, required): JSON schema for validation
- `max_tokens` (INTEGER, optional): Maximum tokens (default: 512)
- `use_cache` (BOOLEAN, optional): Whether to use cache (default: true)
- `seed` (INTEGER, optional): Random seed (default: 42)

**Returns**: `JSON` - Generated JSON object

### steadytext_generate_regex()

Generate text matching a regex pattern.

```sql
-- Generate phone number
SELECT steadytext_generate_regex(
    'Contact number:',
    '\d{3}-\d{3}-\d{4}'
);

-- Generate email
SELECT steadytext_generate_regex(
    'Email address:',
    '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
);
```

**Parameters**:
- `prompt` (TEXT, required): Generation prompt
- `pattern` (TEXT, required): Regular expression pattern
- `max_tokens` (INTEGER, optional): Maximum tokens (default: 512)
- `use_cache` (BOOLEAN, optional): Whether to use cache (default: true)
- `seed` (INTEGER, optional): Random seed (default: 42)

**Returns**: `TEXT` - Generated text matching pattern

### steadytext_generate_choice()

Generate text constrained to specific choices.

```sql
-- Single choice
SELECT steadytext_generate_choice(
    'Is Python a good language for data science?',
    ARRAY['yes', 'no', 'maybe']
);

-- Multiple choice analysis
SELECT 
    question,
    steadytext_generate_choice(
        question,
        ARRAY['strongly agree', 'agree', 'neutral', 'disagree', 'strongly disagree']
    ) as response
FROM survey_questions;
```

**Parameters**:
- `prompt` (TEXT, required): Generation prompt
- `choices` (TEXT[], required): Array of valid choices
- `use_cache` (BOOLEAN, optional): Whether to use cache (default: true)
- `seed` (INTEGER, optional): Random seed (default: 42)

**Returns**: `TEXT` - One of the provided choices

## Async Functions

### steadytext_generate_async()

Start async text generation job.

```sql
-- Start generation
SELECT steadytext_generate_async('Write a long essay', 1000);

-- With priority
SELECT steadytext_generate_async(
    'Urgent request',
    500,
    priority := 10
);
```

**Parameters**:
- `prompt` (TEXT, required): Generation prompt
- `max_tokens` (INTEGER, optional): Maximum tokens (default: 512)
- `priority` (INTEGER, optional): Job priority 1-10 (default: 5)
- `use_cache` (BOOLEAN, optional): Whether to use cache (default: true)
- `seed` (INTEGER, optional): Random seed (default: 42)

**Returns**: `UUID` - Job request ID

### steadytext_check_async()

Check status of async job.

```sql
-- Check job status
SELECT * FROM steadytext_check_async('550e8400-e29b-41d4-a716-446655440000');

-- Wait for completion
SELECT * FROM steadytext_get_async_result(
    '550e8400-e29b-41d4-a716-446655440000',
    timeout_seconds := 30
);
```

**Parameters**:
- `request_id` (UUID, required): Job request ID

**Returns**: `TABLE(status TEXT, result TEXT, error TEXT, created_at TIMESTAMP, completed_at TIMESTAMP)`

### steadytext_cancel_async()

Cancel pending async job.

```sql
-- Cancel job
SELECT steadytext_cancel_async('550e8400-e29b-41d4-a716-446655440000');
```

**Parameters**:
- `request_id` (UUID, required): Job request ID

**Returns**: `BOOLEAN` - True if cancelled, false if not found or already completed

## Utility Functions

### steadytext_cache_stats()

Get cache statistics.

```sql
-- All cache stats
SELECT * FROM steadytext_cache_stats();

-- Specific cache type
SELECT * FROM steadytext_cache_stats()
WHERE cache_type = 'generation';
```

**Returns**: `TABLE(cache_type TEXT, entries BIGINT, size_bytes BIGINT, hit_rate FLOAT, oldest_entry TIMESTAMP)`

### steadytext_clear_cache()

Clear cache entries.

```sql
-- Clear all caches
SELECT steadytext_clear_cache();

-- Clear specific cache type
SELECT steadytext_clear_cache('embedding');

-- Clear entries older than date
SELECT steadytext_clear_cache_older_than('2024-01-01'::timestamp);
```

**Parameters**:
- `cache_type` (TEXT, optional): Cache type to clear ('generation', 'embedding', 'reranking')

**Returns**: `INTEGER` - Number of entries cleared

### steadytext_model_status()

Check model loading status.

```sql
SELECT * FROM steadytext_model_status();
```

**Returns**: `TABLE(model_type TEXT, model_name TEXT, loaded BOOLEAN, size_mb FLOAT, path TEXT)`

### steadytext_estimate_tokens()

Estimate token count for text.

```sql
-- Single text
SELECT steadytext_estimate_tokens('This is a sample text');

-- Multiple texts
SELECT 
    title,
    steadytext_estimate_tokens(content) as token_count
FROM articles;
```

**Parameters**:
- `text` (TEXT, required): Text to analyze

**Returns**: `INTEGER` - Estimated token count

## Administrative Functions

### steadytext_daemon_start()

Start the SteadyText daemon.

```sql
-- Start with defaults
SELECT steadytext_daemon_start();

-- Start with custom settings
SELECT steadytext_daemon_start(
    host := '127.0.0.1',
    port := 5557,
    workers := 4
);
```

**Parameters**:
- `host` (TEXT, optional): Daemon host (default: '127.0.0.1')
- `port` (INTEGER, optional): Daemon port (default: 5557)
- `workers` (INTEGER, optional): Number of workers (default: 2)

**Returns**: `BOOLEAN` - True if started successfully

### steadytext_daemon_stop()

Stop the SteadyText daemon.

```sql
-- Graceful stop
SELECT steadytext_daemon_stop();

-- Force stop
SELECT steadytext_daemon_stop(force := true);
```

**Parameters**:
- `force` (BOOLEAN, optional): Force immediate stop (default: false)

**Returns**: `BOOLEAN` - True if stopped successfully

### steadytext_daemon_status()

Get daemon status information.

```sql
SELECT * FROM steadytext_daemon_status();
```

**Returns**: `TABLE(running BOOLEAN, pid INTEGER, host TEXT, port INTEGER, workers INTEGER, uptime INTERVAL)`

### steadytext_download_models()

Download required model files.

```sql
-- Download all models
SELECT steadytext_download_models();

-- Force re-download
SELECT steadytext_download_models(force := true);

-- Download specific model
SELECT steadytext_download_models(
    model_type := 'generation',
    force := false
);
```

**Parameters**:
- `model_type` (TEXT, optional): Specific model to download
- `force` (BOOLEAN, optional): Force re-download (default: false)

**Returns**: `BOOLEAN` - True if successful

### steadytext_init_cache()

Initialize cache tables.

```sql
-- Initialize with defaults
SELECT steadytext_init_cache();

-- Initialize with custom settings
SELECT steadytext_init_cache(
    generation_capacity := 1000,
    embedding_capacity := 2000
);
```

**Parameters**:
- `generation_capacity` (INTEGER, optional): Max generation cache entries
- `embedding_capacity` (INTEGER, optional): Max embedding cache entries

**Returns**: `VOID`

## Deprecated Functions

### steadytext_embed_distance_cosine() [DEPRECATED]

Use `steadytext_embed_distance()` with metric parameter instead.

```sql
-- Old way (deprecated)
SELECT steadytext_embed_distance_cosine(embed1, embed2);

-- New way
SELECT steadytext_embed_distance(embed1, embed2, 'cosine');
```

### steadytext_generate_with_model() [DEPRECATED]

Use `steadytext_generate()` with environment variables for model selection.

```sql
-- Old way (deprecated)
SELECT steadytext_generate_with_model('prompt', 'model-name');

-- New way
SET LOCAL steadytext.model = 'model-name';
SELECT steadytext_generate('prompt');
```

## Function Patterns

### Error Handling Pattern

All functions follow consistent error handling:

```sql
-- Functions return NULL on error
SELECT COALESCE(
    steadytext_generate('prompt'),
    'Generation failed'
) as result;

-- Check for NULL and handle
DO $$
DECLARE
    result TEXT;
BEGIN
    result := steadytext_generate('prompt');
    IF result IS NULL THEN
        RAISE NOTICE 'Generation failed, using fallback';
        result := 'Fallback text';
    END IF;
END $$;
```

### Batch Processing Pattern

Process large datasets efficiently:

```sql
-- Process in batches
DO $$
DECLARE
    batch_size INTEGER := 100;
    offset_val INTEGER := 0;
    total_rows INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_rows FROM documents;
    
    WHILE offset_val < total_rows LOOP
        UPDATE documents d
        SET embedding = e.embedding
        FROM (
            SELECT * FROM steadytext_embed_batch(
                ARRAY(
                    SELECT content 
                    FROM documents 
                    ORDER BY id 
                    LIMIT batch_size 
                    OFFSET offset_val
                )
            )
        ) e
        WHERE d.content = e.text;
        
        offset_val := offset_val + batch_size;
        RAISE NOTICE 'Processed % of % rows', offset_val, total_rows;
    END LOOP;
END $$;
```

### Caching Pattern

Optimize cache usage:

```sql
-- Pre-warm cache for common queries
INSERT INTO cache_warmup_queries (query)
SELECT DISTINCT query 
FROM search_logs 
WHERE created_at > NOW() - INTERVAL '7 days'
ORDER BY COUNT(*) DESC
LIMIT 100;

-- Warm cache
SELECT steadytext_generate(query, 100)
FROM cache_warmup_queries;
```

## Performance Considerations

1. **Use batch functions** for multiple operations
2. **Enable caching** for repeated operations
3. **Use async functions** for long-running tasks
4. **Monitor cache hit rates** with `steadytext_cache_stats()`
5. **Pre-load models** with `steadytext_download_models()`

## Related Documentation

- [PostgreSQL Extension Overview](postgresql-extension.md)
- [Async Operations Guide](postgresql-extension-async.md)
- [Structured Generation](postgresql-extension-structured.md)
- [Troubleshooting Guide](postgresql-extension-troubleshooting.md)
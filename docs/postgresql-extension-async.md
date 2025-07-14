# PostgreSQL Extension - Async Functions

This document covers asynchronous text generation and processing features in the pg_steadytext PostgreSQL extension.

**Navigation**: [Main Documentation](postgresql-extension.md) | [Structured Generation](postgresql-extension-structured.md) | [AI Features](postgresql-extension-ai.md) | [Advanced Topics](postgresql-extension-advanced.md)

---

## Async Functions Overview (v1.1.0+)

The PostgreSQL extension includes asynchronous counterparts for all generation and embedding functions, enabling non-blocking AI operations at scale.

### Key Features

- **Non-blocking Execution**: Functions return UUID immediately
- **Queue-based Processing**: Background worker handles AI operations
- **Priority Support**: Control processing order with priority levels
- **Batch Operations**: Process multiple items efficiently
- **LISTEN/NOTIFY Integration**: Real-time notifications when results are ready
- **Result Persistence**: Results stored until explicitly retrieved

## Core Async Functions

### Text Generation

#### `steadytext_generate_async()`

Queue text generation for background processing.

```sql
steadytext_generate_async(
    prompt TEXT,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42,
    priority INTEGER DEFAULT 0
) RETURNS UUID
```

**Examples:**

```sql
-- Queue simple generation
SELECT steadytext_generate_async('Write a poem about databases');

-- Queue with custom parameters
SELECT steadytext_generate_async(
    prompt := 'Explain PostgreSQL indexing',
    max_tokens := 1024,
    seed := 123,
    priority := 1  -- Higher priority
);

-- Queue multiple generations
WITH prompts AS (
    SELECT 
        id,
        'Summarize: ' || content AS prompt
    FROM articles
    WHERE needs_summary = true
)
SELECT 
    id,
    steadytext_generate_async(prompt, max_tokens := 200) AS request_id
FROM prompts;
```

### Embeddings

#### `steadytext_embed_async()`

Queue embedding generation for background processing.

```sql
steadytext_embed_async(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 0
) RETURNS UUID
```

**Examples:**

```sql
-- Queue embedding generation
SELECT steadytext_embed_async('PostgreSQL is amazing');

-- Process documents for embedding
INSERT INTO embedding_queue (doc_id, request_id)
SELECT 
    id,
    steadytext_embed_async(content, priority := 2)
FROM documents
WHERE embedding IS NULL;

-- High-priority embedding for real-time search
SELECT steadytext_embed_async(
    user_query,
    use_cache := false,  -- Skip cache for real-time
    priority := 10       -- Highest priority
) AS query_embedding_id;
```

### Structured Generation

All structured generation functions have async counterparts:

#### `steadytext_generate_json_async()`

```sql
steadytext_generate_json_async(
    prompt TEXT,
    schema JSONB,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42,
    priority INTEGER DEFAULT 0
) RETURNS UUID
```

#### `steadytext_generate_regex_async()`

```sql
steadytext_generate_regex_async(
    prompt TEXT,
    pattern TEXT,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42,
    priority INTEGER DEFAULT 0
) RETURNS UUID
```

#### `steadytext_generate_choice_async()`

```sql
steadytext_generate_choice_async(
    prompt TEXT,
    choices TEXT[],
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42,
    priority INTEGER DEFAULT 0
) RETURNS UUID
```

**Example - Async Structured Processing:**

```sql
-- Queue JSON extraction from multiple documents
WITH extraction_jobs AS (
    SELECT 
        doc_id,
        steadytext_generate_json_async(
            'Extract entities from: ' || content,
            '{"type": "object", "properties": {"entities": {"type": "array"}}}'::jsonb,
            priority := CASE 
                WHEN doc_type = 'urgent' THEN 5
                ELSE 0
            END
        ) AS request_id
    FROM documents
    WHERE processed = false
)
INSERT INTO processing_queue (doc_id, request_id, created_at)
SELECT doc_id, request_id, NOW()
FROM extraction_jobs;

-- Queue sentiment analysis
SELECT 
    review_id,
    steadytext_generate_choice_async(
        'Sentiment: ' || review_text,
        ARRAY['positive', 'negative', 'neutral'],
        priority := 1
    ) AS sentiment_request_id
FROM reviews
WHERE sentiment IS NULL;
```

### Batch Operations

#### `steadytext_generate_batch_async()`

Process multiple prompts in a single batch.

```sql
steadytext_generate_batch_async(
    prompts TEXT[],
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42,
    priority INTEGER DEFAULT 0
) RETURNS UUID[]
```

#### `steadytext_embed_batch_async()`

Generate embeddings for multiple texts.

```sql
steadytext_embed_batch_async(
    texts TEXT[],
    use_cache BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 0
) RETURNS UUID[]
```

**Example - Batch Processing:**

```sql
-- Batch generate summaries
WITH batch_job AS (
    SELECT steadytext_generate_batch_async(
        ARRAY_AGG('Summarize: ' || content),
        max_tokens := 150,
        priority := 3
    ) AS request_ids
    FROM articles
    WHERE date_published = CURRENT_DATE
)
INSERT INTO batch_results (request_id, article_id)
SELECT 
    unnest(request_ids),
    article_id
FROM batch_job,
     generate_series(1, array_length(request_ids, 1)) AS article_id;

-- Batch embeddings for similarity search
SELECT steadytext_embed_batch_async(
    ARRAY(
        SELECT description 
        FROM products 
        WHERE category = 'electronics'
    ),
    priority := 2
) AS embedding_requests;
```

### Reranking

#### `steadytext_rerank_async()`

Asynchronously rerank documents by relevance.

```sql
steadytext_rerank_async(
    query TEXT,
    documents TEXT[],
    task TEXT DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42,
    priority INTEGER DEFAULT 0
) RETURNS UUID
```

**Example:**

```sql
-- Queue reranking for search results
WITH search_job AS (
    SELECT steadytext_rerank_async(
        user_query,
        ARRAY(
            SELECT content 
            FROM documents 
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', user_query)
            LIMIT 100
        ),
        priority := 5
    ) AS rerank_request_id
    FROM user_searches
    WHERE id = 12345
)
INSERT INTO search_jobs (search_id, rerank_request_id)
SELECT 12345, rerank_request_id
FROM search_job;
```

## Result Management

### Checking Status

#### `steadytext_check_async()`

Check the status of an async request.

```sql
steadytext_check_async(request_id UUID)
RETURNS TABLE(
    status TEXT,
    result TEXT,
    error TEXT,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    priority INTEGER
)
```

**Examples:**

```sql
-- Check single request
SELECT * FROM steadytext_check_async('550e8400-e29b-41d4-a716-446655440000');

-- Monitor all pending requests
SELECT 
    request_id,
    status,
    EXTRACT(EPOCH FROM (NOW() - created_at)) AS waiting_seconds
FROM steadytext_queue
WHERE status IN ('pending', 'processing')
ORDER BY priority DESC, created_at;

-- Find stuck requests
SELECT * FROM steadytext_check_async(request_id)
FROM steadytext_queue
WHERE status = 'processing'
AND started_at < NOW() - INTERVAL '5 minutes';
```

### Retrieving Results

#### `steadytext_get_async_result()`

Wait for and retrieve async results with timeout.

```sql
steadytext_get_async_result(
    request_id UUID,
    timeout_seconds INTEGER DEFAULT 30
) RETURNS TEXT
```

**Examples:**

```sql
-- Get result with default timeout
SELECT steadytext_get_async_result('550e8400-e29b-41d4-a716-446655440000');

-- Get result with custom timeout
SELECT steadytext_get_async_result(request_id, timeout_seconds := 60)
FROM processing_queue
WHERE doc_id = 123;

-- Process results as they complete
CREATE OR REPLACE FUNCTION process_completed_embeddings()
RETURNS void AS $$
DECLARE
    rec RECORD;
BEGIN
    FOR rec IN 
        SELECT eq.doc_id, eq.request_id
        FROM embedding_queue eq
        JOIN steadytext_queue sq ON eq.request_id = sq.request_id
        WHERE sq.status = 'completed'
        AND eq.processed = false
    LOOP
        UPDATE documents
        SET embedding = steadytext_get_async_result(rec.request_id)::vector
        WHERE id = rec.doc_id;
        
        UPDATE embedding_queue
        SET processed = true
        WHERE request_id = rec.request_id;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

### Batch Result Checking

#### `steadytext_check_async_batch()`

Check status of multiple async requests.

```sql
steadytext_check_async_batch(request_ids UUID[])
RETURNS TABLE(
    request_id UUID,
    status TEXT,
    result TEXT,
    error TEXT,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
)
```

**Example:**

```sql
-- Check batch status
WITH batch_status AS (
    SELECT * FROM steadytext_check_async_batch(
        ARRAY[
            '550e8400-e29b-41d4-a716-446655440000',
            '550e8400-e29b-41d4-a716-446655440001',
            '550e8400-e29b-41d4-a716-446655440002'
        ]
    )
)
SELECT 
    request_id,
    status,
    CASE 
        WHEN status = 'completed' THEN result
        WHEN status = 'failed' THEN error
        ELSE 'Processing...'
    END AS outcome
FROM batch_status;
```

### Canceling Requests

#### `steadytext_cancel_async()`

Cancel a pending async request.

```sql
steadytext_cancel_async(request_id UUID) RETURNS BOOLEAN
```

**Examples:**

```sql
-- Cancel single request
SELECT steadytext_cancel_async('550e8400-e29b-41d4-a716-446655440000');

-- Cancel old pending requests
SELECT 
    request_id,
    steadytext_cancel_async(request_id) AS cancelled
FROM steadytext_queue
WHERE status = 'pending'
AND created_at < NOW() - INTERVAL '1 hour';

-- Cancel low-priority requests during high load
CREATE OR REPLACE FUNCTION manage_queue_load()
RETURNS void AS $$
BEGIN
    IF (SELECT COUNT(*) FROM steadytext_queue WHERE status = 'pending') > 1000 THEN
        -- Cancel low-priority old requests
        PERFORM steadytext_cancel_async(request_id)
        FROM steadytext_queue
        WHERE status = 'pending'
        AND priority < 5
        AND created_at < NOW() - INTERVAL '30 minutes'
        LIMIT 100;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

## Background Worker Configuration

### Starting the Worker

The async worker can be started as a system service or manually:

```bash
# System service (recommended)
sudo systemctl start steadytext-worker
sudo systemctl enable steadytext-worker

# Manual start
python -m pg_steadytext.python.worker \
    --db-host localhost \
    --db-port 5432 \
    --db-name mydb \
    --db-user postgres
```

### Worker Configuration

Configure worker behavior through environment variables:

```bash
# Number of concurrent workers
export STEADYTEXT_WORKER_CONCURRENCY=4

# Polling interval (seconds)
export STEADYTEXT_WORKER_POLL_INTERVAL=1

# Maximum retries for failed jobs
export STEADYTEXT_WORKER_MAX_RETRIES=3

# Worker timeout (seconds)
export STEADYTEXT_WORKER_TIMEOUT=300
```

### PostgreSQL Configuration

```sql
-- Set worker parameters
ALTER SYSTEM SET steadytext.worker_concurrency = 4;
ALTER SYSTEM SET steadytext.worker_poll_interval = '1s';
ALTER SYSTEM SET steadytext.max_queue_size = 10000;

-- Reload configuration
SELECT pg_reload_conf();
```

## LISTEN/NOTIFY Integration

Use PostgreSQL's LISTEN/NOTIFY for real-time updates:

```sql
-- Listen for completion notifications
LISTEN steadytext_completed;

-- Process completed requests in real-time
CREATE OR REPLACE FUNCTION handle_completion_notification()
RETURNS event_trigger AS $$
DECLARE
    payload JSONB;
    request_id UUID;
BEGIN
    -- Parse notification payload
    payload := current_setting('steadytext.notify_payload')::JSONB;
    request_id := (payload->>'request_id')::UUID;
    
    -- Process based on request type
    CASE payload->>'function'
        WHEN 'generate' THEN
            PERFORM process_completed_generation(request_id);
        WHEN 'embed' THEN
            PERFORM process_completed_embedding(request_id);
        WHEN 'rerank' THEN
            PERFORM process_completed_reranking(request_id);
    END CASE;
END;
$$ LANGUAGE plpgsql;

-- Client-side listening (in application code)
-- Python example:
/*
import psycopg2
import select

conn = psycopg2.connect(...)
conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()
cur.execute("LISTEN steadytext_completed;")

while True:
    if select.select([conn], [], [], 5) == ([], [], []):
        print("Timeout")
    else:
        conn.poll()
        while conn.notifies:
            notify = conn.notifies.pop(0)
            print(f"Got NOTIFY: {notify.channel} {notify.payload}")
*/
```

## Usage Patterns

### Fire-and-Forget Pattern

```sql
-- Queue work without waiting for results
CREATE OR REPLACE FUNCTION queue_daily_summaries()
RETURNS void AS $$
BEGIN
    INSERT INTO summary_jobs (article_id, request_id)
    SELECT 
        id,
        steadytext_generate_async(
            'Summarize: ' || title || ' - ' || content,
            max_tokens := 200
        )
    FROM articles
    WHERE published_date = CURRENT_DATE
    AND summary IS NULL;
END;
$$ LANGUAGE plpgsql;

-- Run via cron
SELECT cron.schedule('daily-summaries', '0 2 * * *', 'SELECT queue_daily_summaries()');
```

### Request-Response Pattern

```sql
-- Submit and wait for result
CREATE OR REPLACE FUNCTION generate_and_wait(
    prompt TEXT,
    timeout INTEGER DEFAULT 30
)
RETURNS TEXT AS $$
DECLARE
    request_id UUID;
BEGIN
    -- Queue request
    request_id := steadytext_generate_async(prompt);
    
    -- Wait for result
    RETURN steadytext_get_async_result(request_id, timeout);
END;
$$ LANGUAGE plpgsql;
```

### Batch Pipeline Pattern

```sql
-- Complex pipeline with multiple stages
CREATE OR REPLACE FUNCTION process_document_pipeline(
    doc_id INTEGER
)
RETURNS TABLE(
    stage TEXT,
    request_id UUID,
    status TEXT
) AS $$
BEGIN
    -- Stage 1: Generate summary
    INSERT INTO pipeline_stages (doc_id, stage, request_id)
    VALUES (
        doc_id,
        'summary',
        steadytext_generate_async(
            'Summarize: ' || (SELECT content FROM documents WHERE id = doc_id),
            max_tokens := 150,
            priority := 5
        )
    );
    
    -- Stage 2: Extract entities
    INSERT INTO pipeline_stages (doc_id, stage, request_id)
    VALUES (
        doc_id,
        'entities',
        steadytext_generate_json_async(
            'Extract entities: ' || (SELECT content FROM documents WHERE id = doc_id),
            '{"type": "object", "properties": {"entities": {"type": "array"}}}'::jsonb,
            priority := 4
        )
    );
    
    -- Stage 3: Generate embedding
    INSERT INTO pipeline_stages (doc_id, stage, request_id)
    VALUES (
        doc_id,
        'embedding',
        steadytext_embed_async(
            (SELECT content FROM documents WHERE id = doc_id),
            priority := 3
        )
    );
    
    -- Return pipeline status
    RETURN QUERY
    SELECT 
        ps.stage,
        ps.request_id,
        sq.status
    FROM pipeline_stages ps
    JOIN steadytext_queue sq ON ps.request_id = sq.request_id
    WHERE ps.doc_id = doc_id;
END;
$$ LANGUAGE plpgsql;
```

## Performance Optimization

### Priority Management

```sql
-- Dynamic priority based on user tier
CREATE OR REPLACE FUNCTION get_user_priority(user_id INTEGER)
RETURNS INTEGER AS $$
BEGIN
    RETURN CASE
        WHEN EXISTS (SELECT 1 FROM users WHERE id = user_id AND tier = 'premium') THEN 10
        WHEN EXISTS (SELECT 1 FROM users WHERE id = user_id AND tier = 'standard') THEN 5
        ELSE 0
    END;
END;
$$ LANGUAGE plpgsql;

-- Use dynamic priority
SELECT steadytext_generate_async(
    prompt,
    priority := get_user_priority(current_user_id)
);
```

### Queue Monitoring

```sql
-- Monitor queue health
CREATE OR REPLACE VIEW queue_health AS
SELECT 
    status,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (NOW() - created_at))) as avg_age_seconds,
    MIN(priority) as min_priority,
    MAX(priority) as max_priority
FROM steadytext_queue
GROUP BY status;

-- Alert on queue backup
CREATE OR REPLACE FUNCTION check_queue_health()
RETURNS TABLE(alert_level TEXT, message TEXT) AS $$
BEGIN
    -- Check for too many pending
    IF (SELECT count FROM queue_health WHERE status = 'pending') > 1000 THEN
        RETURN QUERY SELECT 'WARNING', 'Queue backup: >1000 pending requests';
    END IF;
    
    -- Check for old pending requests
    IF EXISTS (
        SELECT 1 FROM steadytext_queue 
        WHERE status = 'pending' 
        AND created_at < NOW() - INTERVAL '10 minutes'
    ) THEN
        RETURN QUERY SELECT 'WARNING', 'Old pending requests detected';
    END IF;
    
    -- Check for stuck processing
    IF EXISTS (
        SELECT 1 FROM steadytext_queue 
        WHERE status = 'processing' 
        AND started_at < NOW() - INTERVAL '5 minutes'
    ) THEN
        RETURN QUERY SELECT 'ERROR', 'Stuck processing requests detected';
    END IF;
    
    RETURN QUERY SELECT 'OK', 'Queue healthy';
END;
$$ LANGUAGE plpgsql;
```

### Batch Optimization

```sql
-- Optimize batch sizes based on queue load
CREATE OR REPLACE FUNCTION optimal_batch_size()
RETURNS INTEGER AS $$
DECLARE
    pending_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO pending_count
    FROM steadytext_queue
    WHERE status = 'pending';
    
    RETURN CASE
        WHEN pending_count < 100 THEN 50    -- Low load: larger batches
        WHEN pending_count < 500 THEN 20    -- Medium load: medium batches
        ELSE 10                             -- High load: smaller batches
    END;
END;
$$ LANGUAGE plpgsql;

-- Use optimal batch size
WITH batch AS (
    SELECT array_agg(content) as contents
    FROM (
        SELECT content
        FROM documents
        WHERE needs_embedding = true
        LIMIT optimal_batch_size()
    ) t
)
SELECT steadytext_embed_batch_async(contents, priority := 5)
FROM batch;
```

## Error Handling

### Retry Logic

```sql
-- Automatic retry for failed requests
CREATE OR REPLACE FUNCTION retry_failed_requests()
RETURNS TABLE(original_id UUID, new_id UUID) AS $$
BEGIN
    RETURN QUERY
    WITH failed AS (
        SELECT 
            request_id,
            function_name,
            parameters,
            priority
        FROM steadytext_queue
        WHERE status = 'failed'
        AND retry_count < 3
        AND failed_at > NOW() - INTERVAL '1 hour'
    )
    SELECT 
        f.request_id as original_id,
        CASE f.function_name
            WHEN 'generate' THEN 
                steadytext_generate_async(
                    (f.parameters->>'prompt')::TEXT,
                    (f.parameters->>'max_tokens')::INTEGER,
                    (f.parameters->>'use_cache')::BOOLEAN,
                    (f.parameters->>'seed')::INTEGER,
                    f.priority
                )
            WHEN 'embed' THEN
                steadytext_embed_async(
                    (f.parameters->>'text_input')::TEXT,
                    (f.parameters->>'use_cache')::BOOLEAN,
                    f.priority
                )
        END as new_id
    FROM failed f;
    
    -- Update retry count
    UPDATE steadytext_queue
    SET retry_count = retry_count + 1
    WHERE request_id IN (SELECT original_id FROM failed);
END;
$$ LANGUAGE plpgsql;
```

### Dead Letter Queue

```sql
-- Move permanently failed requests to dead letter queue
CREATE TABLE steadytext_dead_letter_queue (
    LIKE steadytext_queue INCLUDING ALL,
    moved_at TIMESTAMP DEFAULT NOW(),
    failure_reason TEXT
);

CREATE OR REPLACE FUNCTION process_dead_letters()
RETURNS void AS $$
BEGIN
    INSERT INTO steadytext_dead_letter_queue (
        request_id, function_name, parameters, status, 
        error, created_at, failed_at, retry_count, failure_reason
    )
    SELECT 
        request_id, function_name, parameters, status,
        error, created_at, failed_at, retry_count,
        CASE
            WHEN retry_count >= 3 THEN 'Max retries exceeded'
            WHEN failed_at < NOW() - INTERVAL '24 hours' THEN 'Expired'
            ELSE 'Unknown'
        END
    FROM steadytext_queue
    WHERE status = 'failed'
    AND (retry_count >= 3 OR failed_at < NOW() - INTERVAL '24 hours');
    
    -- Clean up moved requests
    DELETE FROM steadytext_queue
    WHERE request_id IN (
        SELECT request_id FROM steadytext_dead_letter_queue
    );
END;
$$ LANGUAGE plpgsql;
```

## Best Practices

1. **Priority Usage**: Reserve high priorities (8-10) for real-time user requests
2. **Batch Size**: Keep batch sizes between 10-100 items for optimal throughput
3. **Timeout Selection**: Set timeouts based on expected processing time + buffer
4. **Queue Monitoring**: Implement monitoring and alerting for queue health
5. **Error Handling**: Always check for NULL results and handle failures gracefully
6. **Worker Scaling**: Scale workers based on queue depth and processing time
7. **Result Cleanup**: Implement periodic cleanup of old completed results

## Troubleshooting

### Common Issues

```sql
-- Check worker status
SELECT * FROM steadytext_worker_status();

-- View queue statistics
SELECT * FROM steadytext_queue_stats();

-- Find slow requests
SELECT 
    request_id,
    function_name,
    EXTRACT(EPOCH FROM (NOW() - started_at)) as processing_seconds
FROM steadytext_queue
WHERE status = 'processing'
ORDER BY started_at;

-- Debug specific request
SELECT 
    request_id,
    function_name,
    parameters,
    status,
    error,
    created_at,
    started_at,
    completed_at
FROM steadytext_queue
WHERE request_id = '550e8400-e29b-41d4-a716-446655440000';
```

### Performance Diagnostics

```sql
-- Analyze processing times
WITH stats AS (
    SELECT 
        function_name,
        COUNT(*) as total_requests,
        AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_processing_time,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (completed_at - started_at))
        ) as p95_processing_time
    FROM steadytext_queue
    WHERE status = 'completed'
    AND completed_at > NOW() - INTERVAL '1 hour'
    GROUP BY function_name
)
SELECT * FROM stats ORDER BY avg_processing_time DESC;

-- Queue depth over time
CREATE OR REPLACE VIEW queue_depth_history AS
SELECT 
    date_trunc('minute', created_at) as minute,
    COUNT(*) FILTER (WHERE status = 'pending') as pending_count,
    COUNT(*) FILTER (WHERE status = 'processing') as processing_count,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_count,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_count
FROM steadytext_queue
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY minute
ORDER BY minute DESC;
```

---

**Next**: [Advanced Topics](postgresql-extension-advanced.md) | [Main Documentation](postgresql-extension.md)
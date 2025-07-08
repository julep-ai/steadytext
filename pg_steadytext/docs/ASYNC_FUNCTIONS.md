# Async Functions in pg_steadytext

AIDEV-NOTE: This document describes the async counterparts of pg_steadytext functions added in v1.1.0

## Overview

Version 1.1.0 of pg_steadytext adds asynchronous counterparts for all generation and embedding functions. These functions queue requests for background processing, allowing PostgreSQL to continue execution without blocking on model inference.

## Architecture

### Queue-Based Processing

All async functions insert requests into the `steadytext_queue` table with:
- Unique request ID (UUID) for tracking
- Request type and parameters
- Priority support (1-10, default 5)
- Status tracking (pending → processing → completed/failed)
- Retry logic with configurable limits

### Background Worker

The Python worker (`pg_steadytext/python/worker.py`) polls the queue and processes requests:
- Uses `FOR UPDATE SKIP LOCKED` for concurrent worker support
- Processes requests in priority order
- Updates status and stores results in the queue table
- Handles failures with automatic retry

### LISTEN/NOTIFY Integration

Async functions use `pg_notify('steadytext_queue', request_id)` to alert workers of new requests, enabling more responsive processing than pure polling.

## Async Functions

### Text Generation

```sql
-- Queue async text generation
SELECT steadytext_generate_async('Your prompt here', max_tokens := 512);
-- Returns: UUID

-- Queue async JSON generation with schema
SELECT steadytext_generate_json_async(
    'Create a person',
    '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'::jsonb,
    max_tokens := 100
);
-- Returns: UUID

-- Queue async regex-constrained generation
SELECT steadytext_generate_regex_async(
    'My phone number is',
    '\d{3}-\d{3}-\d{4}',
    max_tokens := 50
);
-- Returns: UUID

-- Queue async choice generation
SELECT steadytext_generate_choice_async(
    'Is Python good?',
    ARRAY['yes', 'no', 'maybe']
);
-- Returns: UUID
```

### Embeddings

```sql
-- Queue async embedding generation
SELECT steadytext_embed_async('Text to embed');
-- Returns: UUID
```

### Batch Operations

```sql
-- Queue multiple generation requests
SELECT steadytext_generate_batch_async(
    ARRAY['Prompt 1', 'Prompt 2', 'Prompt 3'],
    max_tokens := 100
);
-- Returns: UUID[]

-- Queue multiple embedding requests
SELECT steadytext_embed_batch_async(
    ARRAY['Text 1', 'Text 2', 'Text 3']
);
-- Returns: UUID[]
```

## Result Retrieval

### Check Request Status

```sql
-- Check single request status
SELECT * FROM steadytext_check_async('your-request-id'::uuid);

-- Returns:
-- status: pending/processing/completed/failed/cancelled
-- result: Generated text (for generation requests)
-- embedding: Vector result (for embedding requests)
-- error: Error message if failed
-- created_at, completed_at, processing_time_ms

-- Check multiple requests
SELECT * FROM steadytext_check_async_batch(
    ARRAY['id1'::uuid, 'id2'::uuid, 'id3'::uuid]
);
```

### Get Result (Blocking)

```sql
-- Wait for result with timeout
SELECT steadytext_get_async_result('your-request-id'::uuid, timeout_seconds := 30);
-- Returns: Generated text or raises exception on timeout/error
```

### Cancel Request

```sql
-- Cancel a pending or processing request
SELECT steadytext_cancel_async('your-request-id'::uuid);
-- Returns: TRUE if cancelled, FALSE if not found or already completed
```

## Usage Examples

### Basic Async Generation

```sql
-- Start generation
WITH request AS (
    SELECT steadytext_generate_async('Write a haiku about databases') AS id
)
-- Poll for completion
SELECT 
    status,
    result,
    processing_time_ms
FROM request r, steadytext_check_async(r.id)
WHERE status = 'completed';
```

### Parallel Processing

```sql
-- Queue multiple requests
WITH requests AS (
    SELECT unnest(steadytext_generate_batch_async(
        ARRAY[
            'Explain SQL',
            'Explain NoSQL', 
            'Explain NewSQL'
        ],
        max_tokens := 200
    )) AS request_id
)
-- Check all statuses
SELECT 
    request_id,
    status,
    substring(result, 1, 50) || '...' AS result_preview
FROM requests r, 
     LATERAL steadytext_check_async(r.request_id);
```

### Async with Callback Pattern

```sql
-- Create a table for results
CREATE TABLE generation_results (
    request_id UUID PRIMARY KEY,
    prompt TEXT,
    result TEXT,
    completed_at TIMESTAMPTZ
);

-- Function to process completed requests
CREATE OR REPLACE FUNCTION process_completed_requests()
RETURNS void AS $$
DECLARE
    rec RECORD;
BEGIN
    -- Find completed requests not yet processed
    FOR rec IN 
        SELECT q.request_id, q.prompt, q.result
        FROM steadytext_queue q
        LEFT JOIN generation_results r ON q.request_id = r.request_id
        WHERE q.status = 'completed' 
        AND r.request_id IS NULL
    LOOP
        INSERT INTO generation_results (request_id, prompt, result, completed_at)
        VALUES (rec.request_id, rec.prompt, rec.result, NOW());
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Schedule periodic processing (requires pg_cron or similar)
```

## Configuration

### Queue Processing

Configure via `steadytext_config` table:

```sql
-- Set maximum queue size
SELECT steadytext_config_set('max_queue_size', '1000');

-- Set worker poll interval (seconds)
SELECT steadytext_config_set('worker_poll_interval', '0.5');

-- Set default retry limit
SELECT steadytext_config_set('max_retries', '3');
```

### Priority Management

```sql
-- High priority request
WITH request AS (
    INSERT INTO steadytext_queue (
        request_type, prompt, params, priority
    ) VALUES (
        'generate', 'Urgent request', '{"max_tokens": 100}'::jsonb, 9
    ) RETURNING request_id
)
SELECT request_id FROM request;
```

## Performance Considerations

1. **Non-Blocking**: Async functions return immediately, allowing PostgreSQL to continue processing
2. **Scalability**: Multiple workers can process the queue concurrently
3. **Resource Management**: Queue size limits prevent unbounded growth
4. **Caching**: Results are still cached for deduplication

## Monitoring

```sql
-- Queue statistics
SELECT 
    status,
    COUNT(*) as count,
    AVG(processing_time_ms) as avg_time_ms,
    MIN(created_at) as oldest_request
FROM steadytext_queue
GROUP BY status;

-- User request rates
SELECT 
    user_id,
    COUNT(*) as requests_last_hour,
    AVG(processing_time_ms) as avg_time_ms
FROM steadytext_queue
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY user_id
ORDER BY requests_last_hour DESC;
```

## Error Handling

Failed requests are automatically retried up to `max_retries` times. After exhausting retries, the request remains in 'failed' status with the error message stored.

```sql
-- View failed requests
SELECT 
    request_id,
    prompt,
    error,
    retry_count,
    created_at
FROM steadytext_queue
WHERE status = 'failed'
ORDER BY created_at DESC;

-- Manually retry failed request
UPDATE steadytext_queue
SET status = 'pending', retry_count = 0
WHERE request_id = 'your-request-id'::uuid
AND status = 'failed';
```

## Best Practices

1. **Use async for long-running operations**: Especially beneficial for large max_tokens or complex schemas
2. **Monitor queue depth**: Set up alerts for queue backlog
3. **Implement result callbacks**: Process completed results asynchronously
4. **Set appropriate timeouts**: Balance between waiting and system resources
5. **Use batch operations**: More efficient than individual async calls

AIDEV-NOTE: The async implementation provides a robust foundation for non-blocking AI operations in PostgreSQL while maintaining the deterministic guarantees of SteadyText.
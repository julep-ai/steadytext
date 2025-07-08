-- 03_async.sql - pgTAP tests for async queue functionality
-- AIDEV-NOTE: Tests for asynchronous generation and embedding queues

BEGIN;
SELECT plan(20);

-- Test 1: Async generation function exists
SELECT has_function(
    'public',
    'steadytext_generate_async',
    ARRAY['text', 'integer'],
    'Function steadytext_generate_async(text, integer) should exist'
);

SELECT has_function(
    'public',
    'steadytext_generate_async',
    ARRAY['text', 'integer', 'boolean'],
    'Function steadytext_generate_async(text, integer, boolean) should exist with cache parameter'
);

-- Test 2: Async generation returns UUID
SELECT function_returns(
    'public',
    'steadytext_generate_async',
    ARRAY['text', 'integer'],
    'uuid',
    'Function steadytext_generate_async should return UUID'
);

-- Test 3: Queue table exists
SELECT has_table(
    'public',
    'steadytext_queue',
    'Table steadytext_queue should exist'
);

-- Test 4: Create async request and verify queue entry
SELECT ok(
    steadytext_generate_async('pgTAP async test', 100) IS NOT NULL,
    'Async generation should return a request ID'
);

-- Test 5: Queue entry has correct initial state
WITH request AS (
    SELECT steadytext_generate_async('pgTAP queue test', 50, true) AS request_id
)
SELECT is(
    q.status,
    'pending',
    'New queue entry should have pending status'
)
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test 6: Queue entry has correct request type
WITH request AS (
    SELECT steadytext_generate_async('pgTAP type test', 50) AS request_id
)
SELECT is(
    q.request_type,
    'generate',
    'Queue entry should have generate request type'
)
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test 7: Queue entry parameters are stored correctly
WITH request AS (
    SELECT steadytext_generate_async('pgTAP params test', 75) AS request_id
)
SELECT is(
    (q.params->>'max_tokens')::int,
    75,
    'Queue entry should store max_tokens parameter correctly'
)
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test 8: Status check function exists
SELECT has_function(
    'public',
    'steadytext_check_async',
    ARRAY['uuid'],
    'Function steadytext_check_async(uuid) should exist'
);

-- Test 9: Status check returns correct columns
SELECT has_column('steadytext_queue', 'status', 'Queue table should have status column');
SELECT has_column('steadytext_queue', 'result', 'Queue table should have result column');
SELECT has_column('steadytext_queue', 'error', 'Queue table should have error column');
SELECT has_column('steadytext_queue', 'created_at', 'Queue table should have created_at column');

-- Test 10: Status check works for pending request
WITH request AS (
    SELECT steadytext_generate_async('pgTAP status test', 25) AS request_id
)
SELECT ok(
    (SELECT status = 'pending' FROM steadytext_check_async(r.request_id)),
    'Status check should show pending for new request'
)
FROM request r;

-- Test 11: Empty prompt validation
SELECT throws_ok(
    $$ SELECT steadytext_generate_async('', 10) $$,
    'P0001',
    'Prompt cannot be empty',
    'Empty prompt should raise an error'
);

-- Test 12: Max tokens validation
SELECT throws_ok(
    $$ SELECT steadytext_generate_async('Test', 5000) $$,
    'P0001',
    'max_tokens cannot exceed 4096',
    'Max tokens over 4096 should raise an error'
);

-- Test 13: Async embed function exists
SELECT has_function(
    'public',
    'steadytext_embed_async',
    'Function steadytext_embed_async should exist'
);

-- Test 14: Batch async functions exist
SELECT has_function(
    'public',
    'steadytext_generate_batch_async',
    'Function steadytext_generate_batch_async should exist'
);

SELECT has_function(
    'public',
    'steadytext_embed_batch_async',
    'Function steadytext_embed_batch_async should exist'
);

-- Test 15: Cancel function exists
SELECT has_function(
    'public',
    'steadytext_cancel_async',
    ARRAY['uuid'],
    'Function steadytext_cancel_async(uuid) should exist'
);

-- Test 16: Get result with timeout function exists
SELECT has_function(
    'public',
    'steadytext_get_async_result',
    ARRAY['uuid', 'integer'],
    'Function steadytext_get_async_result(uuid, integer) should exist'
);

-- Clean up test queue entries
DELETE FROM steadytext_queue WHERE prompt LIKE 'pgTAP%';

SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: Async tests cover:
-- - Queue creation and management
-- - Request status tracking
-- - Input validation
-- - Error handling
-- - All async function variants
-- Tests don't wait for actual processing since that requires the worker
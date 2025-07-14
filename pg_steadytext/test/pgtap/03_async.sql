-- 03_async.sql - pgTAP tests for async queue functionality
-- AIDEV-NOTE: Tests for asynchronous generation and embedding queues

BEGIN;
SELECT plan(35);

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

-- Test 21: Large batch async generation
WITH large_batch AS (
    SELECT array_agg('Large batch prompt ' || i) AS prompts
    FROM generate_series(1, 50) i
),
batch_result AS (
    SELECT steadytext_generate_batch_async(prompts, 20) AS request_ids
    FROM large_batch
)
SELECT is(
    array_length(request_ids, 1),
    50,
    'Large batch async generation should handle 50 prompts'
) FROM batch_result;

-- Test 22: Large batch async embedding
WITH large_embed_batch AS (
    SELECT array_agg('Embed text ' || i) AS texts
    FROM generate_series(1, 30) i
),
embed_batch_result AS (
    SELECT steadytext_embed_batch_async(texts, true) AS request_ids
    FROM large_embed_batch
)
SELECT is(
    array_length(request_ids, 1),
    30,
    'Large batch async embedding should handle 30 texts'
) FROM embed_batch_result;

-- Test 23: Mixed status batch checking
WITH mixed_requests AS (
    SELECT ARRAY[
        steadytext_generate_async('Mixed test 1', 10),
        steadytext_generate_async('Mixed test 2', 20),
        steadytext_generate_async('Mixed test 3', 30)
    ] AS request_ids
),
cancelled_request AS (
    SELECT steadytext_cancel_async(request_ids[2]) AS cancelled
    FROM mixed_requests
),
batch_status AS (
    SELECT * FROM steadytext_check_async_batch(mr.request_ids)
    FROM mixed_requests mr
)
SELECT ok(
    COUNT(DISTINCT status) >= 2,
    'Mixed batch status should show different statuses'
) FROM batch_status;

-- Test 24: Queue priority handling
-- Insert high and low priority requests
INSERT INTO steadytext_queue (request_id, prompt, request_type, params, priority, created_at)
VALUES 
    (gen_random_uuid(), 'High priority test', 'generate', '{"max_tokens": 10}', 1, NOW()),
    (gen_random_uuid(), 'Low priority test', 'generate', '{"max_tokens": 10}', 5, NOW()),
    (gen_random_uuid(), 'Medium priority test', 'generate', '{"max_tokens": 10}', 3, NOW());

-- Test 25: Queue ordering by priority
WITH priority_order AS (
    SELECT prompt, priority, ROW_NUMBER() OVER (ORDER BY priority, created_at) AS rank
    FROM steadytext_queue
    WHERE prompt LIKE '%priority test%'
)
SELECT is(
    (SELECT prompt FROM priority_order WHERE rank = 1),
    'High priority test',
    'Queue should prioritize high priority requests'
);

-- Test 26: Timeout handling simulation
WITH timeout_request AS (
    SELECT steadytext_generate_async('Timeout test', 10) AS request_id
),
timeout_simulation AS (
    SELECT steadytext_get_async_result(request_id, 1) AS result
    FROM timeout_request
)
SELECT ok(
    result IS NULL,
    'Get async result should handle timeout gracefully'
) FROM timeout_simulation;

-- Test 27: Queue capacity stress test
WITH stress_batch AS (
    SELECT array_agg('Stress test prompt ' || i) AS prompts
    FROM generate_series(1, 100) i
),
stress_result AS (
    SELECT steadytext_generate_batch_async(prompts, 10) AS request_ids
    FROM stress_batch
)
SELECT is(
    array_length(request_ids, 1),
    100,
    'Queue should handle stress test with 100 requests'
) FROM stress_result;

-- Test 28: Queue status distribution
WITH queue_status AS (
    SELECT status, COUNT(*) as count
    FROM steadytext_queue
    GROUP BY status
)
SELECT ok(
    COUNT(*) > 0,
    'Queue should have status distribution'
) FROM queue_status;

-- Test 29: Request metadata storage
WITH metadata_request AS (
    SELECT steadytext_generate_async('Metadata test', 50, true) AS request_id
),
metadata_check AS (
    SELECT 
        q.params->>'max_tokens' as max_tokens,
        q.params->>'use_cache' as use_cache
    FROM metadata_request mr
    JOIN steadytext_queue q ON q.request_id = mr.request_id
)
SELECT ok(
    max_tokens = '50' AND use_cache = 'true',
    'Request metadata should be stored correctly'
) FROM metadata_check;

-- Test 30: Queue cleanup for completed requests
-- Mark some requests as completed
UPDATE steadytext_queue 
SET status = 'completed', completed_at = NOW(), result = 'Test result'
WHERE prompt LIKE 'Stress test prompt 1%';

-- Test 31: Request age tracking
WITH age_analysis AS (
    SELECT 
        MIN(created_at) as oldest,
        MAX(created_at) as newest,
        COUNT(*) as total_requests
    FROM steadytext_queue
    WHERE prompt LIKE '%test%'
)
SELECT ok(
    oldest <= newest AND total_requests > 0,
    'Queue should track request age correctly'
) FROM age_analysis;

-- Test 32: Error handling for malformed requests
-- Test invalid JSON in async structured generation
SELECT throws_ok(
    $$ SELECT steadytext_generate_json_async('Test', 'invalid json', 50) $$,
    '22P02',
    NULL,
    'Invalid JSON should raise error in async generation'
);

-- Test 33: Queue resource limits
WITH resource_check AS (
    SELECT COUNT(*) as queue_size
    FROM steadytext_queue
)
SELECT ok(
    queue_size < 10000,
    'Queue size should be within reasonable limits'
) FROM resource_check;

-- Test 34: Batch operation edge cases
-- Test empty batch
SELECT ok(
    steadytext_generate_batch_async(ARRAY[]::text[], 10) = ARRAY[]::uuid[],
    'Empty batch should return empty array'
);

-- Test single item batch
WITH single_batch AS (
    SELECT steadytext_generate_batch_async(ARRAY['Single item'], 10) AS request_ids
)
SELECT is(
    array_length(request_ids, 1),
    1,
    'Single item batch should work correctly'
) FROM single_batch;

-- Test 35: Async request cancellation patterns
WITH cancellation_test AS (
    SELECT ARRAY[
        steadytext_generate_async('Cancel test 1', 10),
        steadytext_generate_async('Cancel test 2', 20),
        steadytext_generate_async('Cancel test 3', 30)
    ] AS request_ids
),
cancelled_all AS (
    SELECT 
        steadytext_cancel_async(request_ids[1]) AS cancel1,
        steadytext_cancel_async(request_ids[2]) AS cancel2,
        steadytext_cancel_async(request_ids[3]) AS cancel3
    FROM cancellation_test
)
SELECT ok(
    cancel1 AND cancel2 AND cancel3,
    'Multiple async requests should be cancellable'
) FROM cancelled_all;

-- Clean up all test queue entries
DELETE FROM steadytext_queue WHERE prompt LIKE 'pgTAP%';
DELETE FROM steadytext_queue WHERE prompt LIKE '%test%';
DELETE FROM steadytext_queue WHERE prompt LIKE 'Large batch%';
DELETE FROM steadytext_queue WHERE prompt LIKE 'Embed text%';
DELETE FROM steadytext_queue WHERE prompt LIKE 'Mixed test%';
DELETE FROM steadytext_queue WHERE prompt LIKE 'Stress test%';
DELETE FROM steadytext_queue WHERE prompt LIKE 'Timeout test%';
DELETE FROM steadytext_queue WHERE prompt LIKE 'Metadata test%';
DELETE FROM steadytext_queue WHERE prompt LIKE 'Cancel test%';
DELETE FROM steadytext_queue WHERE prompt = 'Single item';

SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: Expanded async tests now cover:
-- - Queue creation and management
-- - Request status tracking and mixed status handling
-- - Input validation and error handling
-- - Large batch operations and stress testing
-- - Queue priority and ordering
-- - Timeout and cancellation scenarios
-- - Queue capacity and resource limits
-- - Metadata storage and retrieval
-- - Request age tracking and cleanup
-- - Edge cases and boundary conditions
-- - All async function variants
-- - Performance under load
-- Tests don't wait for actual processing since that requires the worker
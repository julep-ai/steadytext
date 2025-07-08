-- 04_structured.sql - pgTAP tests for structured generation functionality
-- AIDEV-NOTE: Tests for JSON, regex, and choice-based generation

BEGIN;
SELECT plan(30);

-- Test 1: Synchronous structured generation functions exist
SELECT has_function(
    'public',
    'steadytext_generate_json',
    'Function steadytext_generate_json should exist'
);

SELECT has_function(
    'public',
    'steadytext_generate_regex',
    'Function steadytext_generate_regex should exist'
);

SELECT has_function(
    'public',
    'steadytext_generate_choice',
    'Function steadytext_generate_choice should exist'
);

-- Test 2: Async structured generation functions exist
SELECT has_function(
    'public',
    'steadytext_generate_json_async',
    'Function steadytext_generate_json_async should exist'
);

SELECT has_function(
    'public',
    'steadytext_generate_regex_async',
    'Function steadytext_generate_regex_async should exist'
);

SELECT has_function(
    'public',
    'steadytext_generate_choice_async',
    'Function steadytext_generate_choice_async should exist'
);

-- Test 3: JSON async generation creates correct queue entry
WITH request AS (
    SELECT steadytext_generate_json_async(
        'Create a person', 
        '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'::jsonb,
        100
    ) AS request_id
)
SELECT is(
    q.request_type,
    'generate_json',
    'JSON async should have generate_json request type'
)
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test 4: JSON async stores schema correctly
WITH request AS (
    SELECT steadytext_generate_json_async(
        'Create object', 
        '{"type": "object", "properties": {"id": {"type": "integer"}}}'::jsonb,
        50
    ) AS request_id
)
SELECT ok(
    q.params->>'schema' IS NOT NULL,
    'JSON async should store schema in parameters'
)
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test 5: Regex async generation creates correct queue entry
WITH request AS (
    SELECT steadytext_generate_regex_async(
        'My phone number is',
        '\d{3}-\d{3}-\d{4}',
        50
    ) AS request_id
)
SELECT is(
    q.request_type,
    'generate_regex',
    'Regex async should have generate_regex request type'
)
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test 6: Regex async stores pattern correctly
WITH request AS (
    SELECT steadytext_generate_regex_async(
        'Generate pattern',
        '[A-Z]{3}-\d{4}',
        30
    ) AS request_id
)
SELECT is(
    q.params->>'pattern',
    '[A-Z]{3}-\d{4}',
    'Regex async should store pattern correctly'
)
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test 7: Choice async generation creates correct queue entry
WITH request AS (
    SELECT steadytext_generate_choice_async(
        'Is Python good?',
        ARRAY['yes', 'no', 'maybe']
    ) AS request_id
)
SELECT is(
    q.request_type,
    'generate_choice',
    'Choice async should have generate_choice request type'
)
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test 8: Choice async stores choices correctly
WITH request AS (
    SELECT steadytext_generate_choice_async(
        'Pick one',
        ARRAY['option1', 'option2', 'option3']
    ) AS request_id
)
SELECT ok(
    (q.params->>'choices')::jsonb = '["option1", "option2", "option3"]'::jsonb,
    'Choice async should store choices array correctly'
)
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test 9: Batch generation functions exist
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

-- Test 10: Batch generate creates multiple queue entries
WITH request_ids AS (
    SELECT steadytext_generate_batch_async(
        ARRAY['First prompt', 'Second prompt', 'Third prompt'],
        64
    ) AS ids
)
SELECT is(
    array_length(ids, 1),
    3,
    'Batch generate should return 3 request IDs'
)
FROM request_ids;

-- Test 11: Batch generate creates correct queue entries
WITH request_ids AS (
    SELECT steadytext_generate_batch_async(
        ARRAY['Batch test A', 'Batch test B'],
        50
    ) AS ids
)
SELECT is(
    COUNT(*)::integer,
    2,
    'Batch generate should create 2 queue entries'
)
FROM request_ids, 
     LATERAL unnest(ids) AS request_id
JOIN steadytext_queue q ON q.request_id = request_id
WHERE q.request_type = 'generate';

-- Test 12: Batch embed creates multiple queue entries
WITH request_ids AS (
    SELECT steadytext_embed_batch_async(
        ARRAY['Text one', 'Text two', 'Text three']
    ) AS ids
)
SELECT is(
    array_length(ids, 1),
    3,
    'Batch embed should return 3 request IDs'
)
FROM request_ids;

-- Test 13: Cancel function works
WITH request AS (
    SELECT steadytext_generate_async('Test prompt to cancel', 10) AS request_id
),
cancelled AS (
    SELECT steadytext_cancel_async(request_id) AS was_cancelled
    FROM request
)
SELECT ok(
    was_cancelled,
    'Cancel should return true for pending request'
)
FROM cancelled;

-- Test 14: Cancelled request has correct status
WITH request AS (
    SELECT steadytext_generate_async('Another test to cancel', 10) AS request_id
),
cancelled AS (
    SELECT steadytext_cancel_async(r.request_id) AS was_cancelled
    FROM request r
)
SELECT is(
    q.status,
    'cancelled',
    'Cancelled request should have cancelled status'
)
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test 15: Check batch function exists
SELECT has_function(
    'public',
    'steadytext_check_async_batch',
    'Function steadytext_check_async_batch should exist'
);

-- Test 16: Check batch returns correct results
WITH requests AS (
    SELECT ARRAY[
        steadytext_generate_async('pgTAP batch 1', 10),
        steadytext_generate_async('pgTAP batch 2', 20)
    ] AS request_ids
),
batch_check AS (
    SELECT * FROM steadytext_check_async_batch(request_ids)
    FROM requests
)
SELECT is(
    COUNT(*)::integer,
    2,
    'Check batch should return 2 results'
)
FROM batch_check;

-- Test 17: Input validation - empty prompt for JSON
SELECT throws_ok(
    $$ SELECT steadytext_generate_json_async('', '{"type": "string"}'::jsonb) $$,
    'P0001',
    NULL,
    'Empty prompt should raise an error for JSON generation'
);

-- Test 18: Input validation - null schema
SELECT throws_ok(
    $$ SELECT steadytext_generate_json_async('Test', NULL) $$,
    'P0001',
    NULL,
    'Null schema should raise an error'
);

-- Test 19: Input validation - empty regex pattern
SELECT throws_ok(
    $$ SELECT steadytext_generate_regex_async('Test', '') $$,
    'P0001',
    NULL,
    'Empty pattern should raise an error'
);

-- Test 20: Input validation - insufficient choices
SELECT throws_ok(
    $$ SELECT steadytext_generate_choice_async('Test', ARRAY['only_one']) $$,
    'P0001',
    NULL,
    'Single choice should raise an error'
);

-- Test 21: Get result with timeout function
SELECT has_function(
    'public',
    'steadytext_get_async_result',
    ARRAY['uuid', 'integer'],
    'Function steadytext_get_async_result should exist'
);

-- Test 22: Timeout behavior
DO $$
DECLARE
    req_id UUID;
    timed_out BOOLEAN := FALSE;
BEGIN
    -- Create a request
    req_id := steadytext_generate_async('pgTAP timeout test', 10);
    
    -- Try to get result with very short timeout
    BEGIN
        PERFORM steadytext_get_async_result(req_id, 1);
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM LIKE '%Timeout%' OR SQLERRM LIKE '%timeout%' THEN
                timed_out := TRUE;
            END IF;
    END;
    
    -- Pass test if timeout occurred
    PERFORM ok(timed_out, 'Get result should timeout with short wait');
END $$;

-- Clean up test queue entries
-- Use specific prefixes to avoid deleting legitimate data
DELETE FROM steadytext_queue 
WHERE prompt LIKE 'pgTAP%' 
   OR prompt LIKE '__PGTAP_TEST__%'
   OR prompt LIKE 'Batch test prompt%'
   OR prompt = 'First prompt'
   OR prompt = 'Second prompt'
   OR prompt = 'Third prompt';

SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: Structured generation tests cover:
-- - JSON schema-based generation
-- - Regex pattern matching
-- - Choice constraints
-- - Batch operations
-- - Queue management
-- - Input validation
-- - Timeout handling
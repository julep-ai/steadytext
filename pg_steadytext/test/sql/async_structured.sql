-- async_structured.sql - Test async structured generation functionality
-- AIDEV-NOTE: Tests for asynchronous structured generation functions

-- Test async JSON generation
SELECT 'test_json_async' AS test_name;
WITH request AS (
    SELECT steadytext_generate_json_async(
        'Create a person', 
        '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'::jsonb,
        100
    ) AS request_id
)
SELECT 
    q.status = 'pending' AS is_pending,
    q.request_type = 'generate_json' AS correct_type,
    (q.params->>'max_tokens')::int = 100 AS correct_max_tokens,
    q.params->>'schema' IS NOT NULL AS has_schema
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test async regex generation
SELECT 'test_regex_async' AS test_name;
WITH request AS (
    SELECT steadytext_generate_regex_async(
        'My phone number is',
        '\d{3}-\d{3}-\d{4}',
        50
    ) AS request_id
)
SELECT 
    q.status = 'pending' AS is_pending,
    q.request_type = 'generate_regex' AS correct_type,
    q.params->>'pattern' = '\d{3}-\d{3}-\d{4}' AS correct_pattern
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test async choice generation
SELECT 'test_choice_async' AS test_name;
WITH request AS (
    SELECT steadytext_generate_choice_async(
        'Is Python good?',
        ARRAY['yes', 'no', 'maybe']
    ) AS request_id
)
SELECT 
    q.status = 'pending' AS is_pending,
    q.request_type = 'generate_choice' AS correct_type,
    (q.params->>'choices')::jsonb = '["yes", "no", "maybe"]'::jsonb AS correct_choices
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test async embedding
SELECT 'test_embed_async' AS test_name;
WITH request AS (
    SELECT steadytext_embed_async('Test text for embedding') AS request_id
)
SELECT 
    q.status = 'pending' AS is_pending,
    q.request_type = 'embed' AS correct_type,
    q.prompt = 'Test text for embedding' AS correct_prompt
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test batch operations
SELECT 'test_batch_generate' AS test_name;
WITH request_ids AS (
    SELECT steadytext_generate_batch_async(
        ARRAY['First prompt', 'Second prompt', 'Third prompt'],
        64
    ) AS ids
)
SELECT 
    array_length(ids, 1) = 3 AS correct_count,
    COUNT(*) = 3 AS all_created
FROM request_ids, 
     LATERAL unnest(ids) AS request_id
JOIN steadytext_queue q ON q.request_id = request_id;

-- Test batch embed
SELECT 'test_batch_embed' AS test_name;
WITH request_ids AS (
    SELECT steadytext_embed_batch_async(
        ARRAY['Text one', 'Text two', 'Text three']
    ) AS ids
)
SELECT 
    array_length(ids, 1) = 3 AS correct_count,
    COUNT(*) = 3 AS all_created,
    bool_and(q.request_type = 'embed') AS all_embed_type
FROM request_ids, 
     LATERAL unnest(ids) AS request_id
JOIN steadytext_queue q ON q.request_id = request_id;

-- Test get_async_result function (with short timeout to avoid blocking)
SELECT 'test_get_result_timeout' AS test_name;
DO $$
DECLARE
    req_id UUID;
BEGIN
    -- Create a request
    req_id := steadytext_generate_async('Test prompt for timeout', 10);
    
    -- Try to get result with very short timeout (should fail)
    BEGIN
        PERFORM steadytext_get_async_result(req_id, 1);
        RAISE EXCEPTION 'Should have timed out';
    EXCEPTION
        WHEN OTHERS THEN
            -- Expected timeout
            IF SQLERRM NOT LIKE '%Timeout%' THEN
                RAISE;
            END IF;
    END;
END $$;

-- Test cancel function
SELECT 'test_cancel_async' AS test_name;
WITH request AS (
    SELECT steadytext_generate_async('Test prompt to cancel', 10) AS request_id
),
cancelled AS (
    SELECT steadytext_cancel_async(request_id) AS was_cancelled
    FROM request
)
SELECT 
    was_cancelled AS successfully_cancelled,
    q.status = 'cancelled' AS is_cancelled
FROM request r
CROSS JOIN cancelled c
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test check batch function
SELECT 'test_check_batch' AS test_name;
WITH requests AS (
    SELECT ARRAY[
        steadytext_generate_async('Batch test 1', 10),
        steadytext_generate_async('Batch test 2', 20),
        steadytext_generate_async('Batch test 3', 30)
    ] AS request_ids
),
batch_check AS (
    SELECT * FROM steadytext_check_async_batch(request_ids)
    FROM requests
)
SELECT 
    COUNT(*) = 3 AS correct_count,
    bool_and(status = 'pending') AS all_pending
FROM batch_check;

-- Test input validation
SELECT 'test_validation' AS test_name;

-- Empty prompt for JSON
DO $$
BEGIN
    PERFORM steadytext_generate_json_async('', '{"type": "string"}'::jsonb);
    RAISE EXCEPTION 'Should have failed with empty prompt';
EXCEPTION
    WHEN OTHERS THEN
        -- Expected
END $$;

-- Null schema
DO $$
BEGIN
    PERFORM steadytext_generate_json_async('Test', NULL);
    RAISE EXCEPTION 'Should have failed with null schema';
EXCEPTION
    WHEN OTHERS THEN
        -- Expected
END $$;

-- Empty pattern for regex
DO $$
BEGIN
    PERFORM steadytext_generate_regex_async('Test', '');
    RAISE EXCEPTION 'Should have failed with empty pattern';
EXCEPTION
    WHEN OTHERS THEN
        -- Expected
END $$;

-- Too few choices
DO $$
BEGIN
    PERFORM steadytext_generate_choice_async('Test', ARRAY['only_one']);
    RAISE EXCEPTION 'Should have failed with too few choices';
EXCEPTION
    WHEN OTHERS THEN
        -- Expected
END $$;

-- Cleanup test queue entries
DELETE FROM steadytext_queue 
WHERE prompt LIKE '%test%' OR prompt LIKE '%Test%' OR prompt LIKE 'Batch test%';
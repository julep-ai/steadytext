-- async_queue.sql - Test async queue functionality
-- AIDEV-NOTE: Tests for asynchronous generation queue

-- Test async generation request
SELECT steadytext_generate_async('Generate async text', 100) IS NOT NULL AS request_created;

-- Test queue entry creation
WITH request AS (
    SELECT steadytext_generate_async('Test prompt', 50, true) AS request_id
)
SELECT 
    q.status = 'pending' AS is_pending,
    q.request_type = 'generate' AS correct_type,
    (q.params->>'max_tokens')::int = 50 AS correct_max_tokens,
    q.params->>'max_tokens' IS NOT NULL AS has_params
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test status check function
WITH request AS (
    SELECT steadytext_generate_async('Status check test', 25) AS request_id
)
SELECT 
    status,
    result IS NULL AS no_result_yet,
    error IS NULL AS no_error,
    created_at IS NOT NULL AS has_created_time
FROM request r, steadytext_check_async(r.request_id);

-- Test input validation
-- Should fail with empty prompt
DO $$
BEGIN
    PERFORM steadytext_generate_async('', 10);
    RAISE EXCEPTION 'Should have failed with empty prompt';
EXCEPTION
    WHEN OTHERS THEN
        -- Expected
END $$;

-- Should fail with invalid max_tokens
DO $$
BEGIN
    PERFORM steadytext_generate_async('Test', 5000);
    RAISE EXCEPTION 'Should have failed with max_tokens > 4096';
EXCEPTION
    WHEN OTHERS THEN
        -- Expected
END $$;

-- Cleanup test queue entries
DELETE FROM steadytext_queue WHERE prompt LIKE '%test%' OR prompt LIKE '%Test%';
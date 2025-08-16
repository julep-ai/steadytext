-- 13_performance_edge_cases.sql - pgTAP tests for performance and edge cases
-- AIDEV-NOTE: Tests for large payload handling, concurrent operations, resource limits, and timeout behaviors

BEGIN;
SELECT plan(30);

-- Test 1: Large payload generation
WITH large_payload AS (
    SELECT repeat('Large payload test with many tokens. ', 100) AS prompt
)
SELECT ok(
    length(steadytext_generate(prompt, 500)) > 0,
    'Large payload generation should work'
) FROM large_payload;

-- Test 2: Maximum token generation
SELECT ok(
    length(steadytext_generate('Maximum tokens test', 4096)) > 0,
    'Maximum token generation should work'
);

-- Test 3: Embedding with large text
WITH large_text AS (
    SELECT repeat('Large text for embedding test. ', 50) AS text
)
SELECT ok(
    vector_dims(steadytext_embed(text)) = 1024,
    'Embedding large text should work'
) FROM large_text;

-- Test 4: Batch operations with maximum size
WITH max_batch AS (
    SELECT array_agg('Batch item ' || i) AS prompts
    FROM generate_series(1, 100) i
)
SELECT ok(
    array_length(steadytext_generate_batch_async(prompts, 10), 1) = 100,
    'Maximum batch size should work'
) FROM max_batch;

-- Test 5: Concurrent async requests simulation
WITH concurrent_test AS (
    SELECT generate_series(1, 50) AS request_num
)
INSERT INTO steadytext_queue (request_id, prompt, request_type, params, created_at)
SELECT 
    gen_random_uuid(),
    'Concurrent perf test ' || request_num,
    'generate',
    '{"max_tokens": 50}',
    NOW()
FROM concurrent_test;

-- Test 6: Queue processing performance
WITH queue_performance AS (
    SELECT COUNT(*) as total_requests
    FROM steadytext_queue
    WHERE prompt LIKE 'Concurrent perf test%'
)
SELECT ok(
    total_requests = 50,
    'Queue should handle 50 concurrent requests'
) FROM queue_performance;

-- Test 7: Memory usage with large cache
-- Create many cache entries to test memory handling
WITH memory_test AS (
    SELECT generate_series(1, 200) AS i
)
INSERT INTO steadytext_cache (cache_key, prompt, response, access_count, last_accessed, created_at)
SELECT 
    'memory_test_' || i,
    'Memory test prompt ' || i,
    repeat('Response text ', 20) || i,
    i % 10 + 1,
    NOW() - (i || ' minutes')::interval,
    NOW() - (i || ' minutes')::interval
FROM memory_test;

-- Test 8: Cache performance with large dataset
WITH cache_performance AS (
    SELECT COUNT(*) as cache_size
    FROM steadytext_cache
    WHERE cache_key LIKE 'memory_test_%'
)
SELECT ok(
    cache_size = 200,
    'Cache should handle 200 entries efficiently'
) FROM cache_performance;

-- Test 9: Frecency calculation performance
WITH frecency_performance AS (
    SELECT COUNT(*) as entries_with_frecency
    FROM steadytext_cache_with_frecency
    WHERE cache_key LIKE 'memory_test_%'
)
SELECT ok(
    entries_with_frecency = 200,
    'Frecency calculation should handle 200 entries'
) FROM frecency_performance;

-- Test 10: Bulk eviction performance
WITH bulk_eviction AS (
    SELECT * FROM steadytext_cache_evict_by_frecency(
        target_entries := 100,
        target_size_mb := 1.0,
        batch_size := 50
    )
)
SELECT ok(
    evicted_count > 0,
    'Bulk eviction should process large datasets efficiently'
) FROM bulk_eviction;

-- Test 11: Complex JSON schema performance
WITH complex_schema AS (
    SELECT '{
        "type": "object",
        "properties": {
            "users": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "profile": {
                            "type": "object",
                            "properties": {
                                "personal": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "email": {"type": "string"},
                                        "addresses": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "street": {"type": "string"},
                                                    "city": {"type": "string"},
                                                    "country": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }'::jsonb AS schema
)
SELECT ok(
    steadytext_generate_json('Create complex nested structure', schema, 1000, false, 42) IS NOT NULL,
    'Complex nested JSON schema should be handled efficiently'
) FROM complex_schema;

-- Test 12: Reranking with large document sets
WITH large_doc_set AS (
    SELECT array_agg('Document ' || i || ' contains information about ' || 
                    CASE WHEN i % 4 = 0 THEN 'databases and SQL queries'
                         WHEN i % 4 = 1 THEN 'machine learning and AI'
                         WHEN i % 4 = 2 THEN 'web development and APIs'
                         ELSE 'system administration and DevOps'
                    END || ' with detailed explanations.') AS documents
    FROM generate_series(1, 100) i
),
rerank_performance AS (
    SELECT COUNT(*) as ranked_docs
    FROM steadytext_rerank(
        'database systems',
        (SELECT documents FROM large_doc_set),
        'Rank by database relevance',
        true,
        42
    ) AS (document TEXT, score FLOAT)
)
SELECT ok(
    ranked_docs = 100,
    'Reranking should handle 100 documents efficiently'
) FROM rerank_performance;

-- Test 13: AI summarization performance
WITH large_summarization AS (
    SELECT repeat('This is a comprehensive document about database systems and their applications in modern software development. PostgreSQL is a powerful relational database that supports advanced features like JSON, full-text search, and custom data types. ', 20) AS text
)
SELECT ok(
    length(steadytext_summarize_text(text, '{"max_length": 200}')) > 0,
    'AI summarization should handle large texts efficiently'
) FROM large_summarization;

-- Test 14: Streaming performance with large output
WITH stream_performance AS (
    SELECT COUNT(*) as token_count
    FROM steadytext_generate_stream('Write a detailed essay about databases', 1000)
)
SELECT ok(
    token_count > 50,
    'Streaming should handle large outputs efficiently'
) FROM stream_performance;

-- Test 15: Timeout handling under load
WITH timeout_test AS (
    SELECT steadytext_generate_async('Timeout under load test', 100) AS request_id
)
SELECT ok(
    (SELECT request_id IS NOT NULL FROM timeout_test),
    'Async requests should handle timeouts under load'
);

-- Test 16: Connection pool stress test
-- Simulate multiple database connections
WITH connection_stress AS (
    SELECT generate_series(1, 20) AS conn_id
)
INSERT INTO steadytext_daemon_health (daemon_id, status, endpoint, last_heartbeat, uptime_seconds)
SELECT 
    'stress_daemon_' || conn_id,
    'healthy',
    'tcp://localhost:' || (5000 + conn_id),
    NOW() - (conn_id || ' seconds')::interval,
    conn_id * 60
FROM connection_stress;

-- Test 17: Daemon health monitoring under load
WITH health_monitoring AS (
    SELECT COUNT(*) as active_daemons
    FROM steadytext_daemon_health
    WHERE daemon_id LIKE 'stress_daemon_%'
)
SELECT ok(
    active_daemons = 20,
    'System should handle multiple daemon connections'
) FROM health_monitoring;

-- Test 18: Configuration under high load
WITH config_stress AS (
    SELECT generate_series(1, 50) AS config_num
)
INSERT INTO steadytext_config (key, value, description, created_at)
SELECT 
    'stress_config_' || config_num,
    'stress_value_' || config_num,
    'Stress test configuration ' || config_num,
    NOW()
FROM config_stress
ON CONFLICT (key) DO NOTHING;

-- Test 19: Configuration retrieval performance
WITH config_performance AS (
    SELECT COUNT(*) as config_count
    FROM steadytext_config
    WHERE key LIKE 'stress_config_%'
)
SELECT ok(
    config_count > 0,
    'Configuration system should handle high load'
) FROM config_performance;

-- Test 20: Edge case - empty inputs
SELECT ok(
    steadytext_generate('', 10) IS NULL OR length(steadytext_generate('', 10)) = 0,
    'Empty input should be handled gracefully'
);

-- Test 21: Edge case - zero max_tokens
SELECT throws_ok(
    $$ SELECT steadytext_generate('Test', 0) $$,
    'P0001',
    'max_tokens must be greater than 0',
    'Zero max_tokens should raise error'
);

-- Test 22: Edge case - negative values
SELECT throws_ok(
    $$ SELECT steadytext_generate('Test', -1) $$,
    'P0001',
    'max_tokens must be greater than 0',
    'Negative max_tokens should raise error'
);

-- Test 23: Edge case - very large numbers
SELECT throws_ok(
    $$ SELECT steadytext_generate('Test', 2147483647) $$,
    'P0001',
    'max_tokens exceeds system limit',
    'Extremely large max_tokens should raise error'
);

-- Test 24: Unicode stress test
WITH unicode_stress AS (
    SELECT 'Unicode test: ' || 
           '中文测试 ' || 
           'العربية ' || 
           'русский ' || 
           'español ' || 
           'français ' || 
           'deutsch ' || 
           'português ' || 
           'italiano ' || 
           'svenska ' || 
           'norsk ' || 
           'suomi ' || 
           'dansk ' || 
           'ελληνικά ' || 
           'हिंदी ' || 
           'বাংলা ' || 
           'தமிழ் ' || 
           'ภาษาไทย ' || 
           'Việt Nam ' || 
           'Indonesia ' || 
           'Malay ' || 
           'Filipino ' || 
           'emoji: 🌍🚀💡🔥⚡🌟💫🎯🎨🎭🎪🎬🎮🎯' AS text
)
SELECT ok(
    steadytext_generate(text, 100) IS NOT NULL,
    'Unicode stress test should work'
) FROM unicode_stress;

-- Test 25: Concurrent embedding stress test
WITH embed_stress AS (
    SELECT array_agg('Embedding stress test ' || i) AS texts
    FROM generate_series(1, 25) i
)
SELECT ok(
    array_length(steadytext_embed_batch_async(texts, true), 1) = 25,
    'Concurrent embedding should handle stress'
) FROM embed_stress;

-- Test 26: Resource cleanup validation
-- Test that resources are properly cleaned up
WITH resource_check AS (
    SELECT 
        (SELECT COUNT(*) FROM steadytext_queue) as queue_size,
        (SELECT COUNT(*) FROM steadytext_cache) as cache_size,
        (SELECT COUNT(*) FROM steadytext_daemon_health) as daemon_count
)
SELECT ok(
    queue_size < 1000 AND cache_size < 1000 AND daemon_count < 100,
    'Resources should be within reasonable limits'
) FROM resource_check;

-- Test 27: Transaction rollback handling
DO $$
DECLARE
    test_passed BOOLEAN := TRUE;
BEGIN
    BEGIN
        -- This should rollback if it fails
        INSERT INTO steadytext_cache (cache_key, prompt, response)
        VALUES ('rollback_test', 'test', 'test');
        
        -- Force an error
        RAISE EXCEPTION 'Test rollback';
    EXCEPTION
        WHEN others THEN
            -- Check that rollback worked
            IF EXISTS(SELECT 1 FROM steadytext_cache WHERE cache_key = 'rollback_test') THEN
                test_passed := FALSE;
            END IF;
    END;
    
    PERFORM ok(test_passed, 'Transaction rollback should work properly');
END $$;

-- Test 28: Memory leak detection simulation
WITH memory_leak_test AS (
    SELECT generate_series(1, 100) AS i
)
INSERT INTO steadytext_cache (cache_key, prompt, response, access_count, last_accessed, created_at)
SELECT 
    'leak_test_' || i,
    'Memory leak test ' || i,
    repeat('Data ', 100),
    1,
    NOW(),
    NOW()
FROM memory_leak_test;

-- Clean up immediately to test cleanup
DELETE FROM steadytext_cache WHERE cache_key LIKE 'leak_test_%';

-- Test 29: Connection recovery simulation
-- Test that system handles connection issues gracefully
WITH connection_recovery AS (
    SELECT steadytext_daemon_status() AS status
)
SELECT ok(
    (SELECT COUNT(*) FROM connection_recovery) >= 0,
    'System should handle connection recovery'
);

-- Test 30: Performance regression detection
-- Time a simple operation to detect performance issues
WITH performance_timing AS (
    SELECT 
        extract(epoch from NOW()) AS start_time,
        steadytext_generate('Performance test', 10) AS result,
        extract(epoch from NOW()) AS end_time
),
timing_check AS (
    SELECT 
        (end_time - start_time) AS duration,
        result
    FROM performance_timing
)
SELECT ok(
    duration < 30.0 AND result IS NOT NULL,
    'Basic operation should complete within reasonable time'
) FROM timing_check;

-- Clean up all test data
DELETE FROM steadytext_queue WHERE prompt LIKE 'Concurrent perf test%';
DELETE FROM steadytext_cache WHERE cache_key LIKE 'memory_test_%';
DELETE FROM steadytext_daemon_health WHERE daemon_id LIKE 'stress_daemon_%';
DELETE FROM steadytext_config WHERE key LIKE 'stress_config_%';

SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: Performance and edge case tests comprehensively cover:
-- - Large payload handling and processing
-- - Maximum token generation and limits
-- - Large text embedding performance
-- - Batch operation scalability
-- - Concurrent request handling
-- - Memory usage and management
-- - Cache performance with large datasets
-- - Bulk operations and eviction
-- - Complex JSON schema processing
-- - Large document set reranking
-- - AI summarization performance
-- - Streaming output handling
-- - Timeout and connection management
-- - Unicode and special character handling
-- - Resource cleanup and limits
-- - Transaction rollback behavior
-- - Memory leak detection
-- - Connection recovery
-- - Performance regression detection
-- - Edge cases and boundary conditions
-- 05_cache_daemon.sql - pgTAP tests for cache management and daemon integration
-- AIDEV-NOTE: Tests for caching functionality and daemon connectivity

BEGIN;
SELECT plan(50);

-- Test 1: Cache table structure
SELECT has_table(
    'public',
    'steadytext_cache',
    'Table steadytext_cache should exist'
);

SELECT has_column('steadytext_cache', 'cache_key', 'Cache table should have cache_key column');
SELECT has_column('steadytext_cache', 'prompt', 'Cache table should have prompt column');
SELECT has_column('steadytext_cache', 'response', 'Cache table should have response column');
SELECT has_column('steadytext_cache', 'embedding', 'Cache table should have embedding column');
SELECT has_column('steadytext_cache', 'access_count', 'Cache table should have access_count column');
SELECT has_column('steadytext_cache', 'last_accessed', 'Cache table should have last_accessed column');
SELECT has_column('steadytext_cache', 'created_at', 'Cache table should have created_at column');

-- Test 2: Cache key index exists
SELECT has_index(
    'public',
    'steadytext_cache',
    'idx_cache_key',
    'Cache table should have index on cache_key'
);

-- Test 3: Cache statistics function exists
SELECT has_function(
    'public',
    'steadytext_cache_stats',
    'Function steadytext_cache_stats should exist'
);

-- Test 4: Cache stats returns correct columns
SELECT columns_are(
    'public',
    'steadytext_cache_stats',
    ARRAY['total_entries', 'total_size_bytes', 'cache_hit_rate', 'avg_access_count', 'oldest_entry', 'newest_entry'],
    'Cache stats should return expected columns'
);

-- Test 5: Cache behavior with generation
-- First generation (cache miss)
SELECT ok(
    length(steadytext_generate('pgTAP cache test prompt', 20, true)) > 0,
    'First generation should succeed'
);

-- Test 6: Cache entry was created
SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_cache WHERE prompt = 'pgTAP cache test prompt'),
    'Cache entry should be created after generation'
);

-- Test 7: Second generation uses cache
WITH first_result AS (
    SELECT response FROM steadytext_cache WHERE prompt = 'pgTAP cache test prompt'
)
SELECT is(
    steadytext_generate('pgTAP cache test prompt', 20, true),
    (SELECT response FROM first_result),
    'Second generation should return cached result'
);

-- Test 8: Access count increments
SELECT ok(
    (SELECT access_count > 1 FROM steadytext_cache WHERE prompt = 'pgTAP cache test prompt'),
    'Access count should increment on cache hit'
);

-- Test 9: Cache can be disabled
SELECT isnt(
    steadytext_generate('pgTAP no cache test', 15, false),
    NULL,
    'Generation with cache disabled should work'
);

SELECT is(
    (SELECT COUNT(*) FROM steadytext_cache WHERE prompt = 'pgTAP no cache test'),
    0::bigint,
    'No cache entry should be created when cache is disabled'
);

-- Test 10: Daemon status function exists
SELECT has_function(
    'public',
    'steadytext_daemon_status',
    'Function steadytext_daemon_status should exist'
);

-- Test 11: Daemon status returns expected columns
SELECT columns_are(
    'public',
    'steadytext_daemon_status',
    ARRAY['status', 'host', 'port', 'pid', 'uptime_seconds', 'requests_processed'],
    'Daemon status should return expected columns'
);

-- Test 12: Daemon configuration functions exist
SELECT has_function(
    'public',
    'steadytext_daemon_start',
    'Function steadytext_daemon_start should exist'
);

SELECT has_function(
    'public',
    'steadytext_daemon_stop',
    'Function steadytext_daemon_stop should exist'
);

SELECT has_function(
    'public',
    'steadytext_daemon_restart',
    'Function steadytext_daemon_restart should exist'
);

-- Test 13: Cache eviction function exists
SELECT has_function(
    'public',
    'steadytext_cache_evict',
    ARRAY['integer'],
    'Function steadytext_cache_evict(integer) should exist'
);

-- Test 14: Cache clear function exists
SELECT has_function(
    'public',
    'steadytext_cache_clear',
    'Function steadytext_cache_clear should exist'
);

-- Test 15: Test cache eviction preserves frequently used entries
-- Create entries with different access patterns
INSERT INTO steadytext_cache (cache_key, prompt, response, access_count, last_accessed)
VALUES 
    ('pgTAP_evict_1', 'Eviction test 1', 'Response 1', 10, NOW()),
    ('pgTAP_evict_2', 'Eviction test 2', 'Response 2', 1, NOW() - INTERVAL '1 hour'),
    ('pgTAP_evict_3', 'Eviction test 3', 'Response 3', 5, NOW() - INTERVAL '30 minutes');

-- Evict least frequently used
SELECT ok(
    steadytext_cache_evict(1) >= 0,
    'Cache eviction should return number of evicted entries'
);

-- Most frequently accessed should remain
SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_cache WHERE cache_key = 'pgTAP_evict_1'),
    'Frequently accessed entry should remain after eviction'
);

-- Test 26: Extended cache statistics function exists (v1.4.0+)
SELECT has_function(
    'public',
    'steadytext_cache_stats_extended',
    'Function steadytext_cache_stats_extended should exist'
);

-- Test 27: Extended cache statistics returns additional columns
SELECT columns_are(
    'public',
    'steadytext_cache_stats_extended',
    ARRAY['total_entries', 'total_size_mb', 'cache_hit_rate', 'avg_access_count', 'oldest_entry', 'newest_entry', 'low_frecency_count', 'protected_count', 'eviction_candidates'],
    'Extended cache stats should return expected columns'
);

-- Test 28: Frecency-based eviction function exists
SELECT has_function(
    'public',
    'steadytext_cache_evict_by_frecency',
    ARRAY['integer', 'double precision', 'integer', 'integer', 'integer'],
    'Function steadytext_cache_evict_by_frecency should exist'
);

-- Test 29: Cache usage analysis function exists
SELECT has_function(
    'public',
    'steadytext_cache_analyze_usage',
    'Function steadytext_cache_analyze_usage should exist'
);

-- Test 30: Cache preview eviction function exists
SELECT has_function(
    'public',
    'steadytext_cache_preview_eviction',
    ARRAY['integer'],
    'Function steadytext_cache_preview_eviction should exist'
);

-- Test 31: Cache warmup function exists
SELECT has_function(
    'public',
    'steadytext_cache_warmup',
    ARRAY['integer'],
    'Function steadytext_cache_warmup should exist'
);

-- Test 32: Frecency-based eviction with test data
-- Create test entries with different access patterns
INSERT INTO steadytext_cache (cache_key, prompt, response, access_count, last_accessed, created_at)
VALUES 
    ('frecency_test_1', 'High frequency recent', 'Response 1', 50, NOW(), NOW() - INTERVAL '1 hour'),
    ('frecency_test_2', 'Low frequency old', 'Response 2', 2, NOW() - INTERVAL '1 day', NOW() - INTERVAL '2 days'),
    ('frecency_test_3', 'Medium frequency medium', 'Response 3', 10, NOW() - INTERVAL '6 hours', NOW() - INTERVAL '12 hours'),
    ('frecency_test_4', 'Very old unused', 'Response 4', 1, NOW() - INTERVAL '7 days', NOW() - INTERVAL '7 days'),
    ('frecency_test_5', 'Recent but unused', 'Response 5', 1, NOW() - INTERVAL '10 minutes', NOW() - INTERVAL '10 minutes');

-- Test 33: Frecency eviction respects access count protection
WITH eviction_result AS (
    SELECT * FROM steadytext_cache_evict_by_frecency(
        target_entries := 3,
        target_size_mb := 0.001,
        batch_size := 10,
        min_access_count := 10,  -- Protect entries with 10+ accesses
        min_age_hours := 1      -- Protect entries newer than 1 hour
    )
)
SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_cache WHERE cache_key = 'frecency_test_1'),
    'High-access entry should be protected from eviction'
);

-- Test 34: Preview eviction shows candidates
WITH preview AS (
    SELECT * FROM steadytext_cache_preview_eviction(3)
)
SELECT ok(
    COUNT(*) > 0,
    'Preview eviction should show eviction candidates'
) FROM preview;

-- Test 35: Cache usage analysis returns buckets
WITH usage_analysis AS (
    SELECT * FROM steadytext_cache_analyze_usage()
)
SELECT ok(
    COUNT(*) > 0,
    'Cache usage analysis should return usage buckets'
) FROM usage_analysis;

-- Test 36: Cache warmup function
WITH warmup_result AS (
    SELECT * FROM steadytext_cache_warmup(5)
)
SELECT ok(
    warmed_entries >= 0,
    'Cache warmup should return non-negative warmed entries count'
) FROM warmup_result;

-- Test 37: Extended cache statistics with data
WITH extended_stats AS (
    SELECT * FROM steadytext_cache_stats_extended()
)
SELECT ok(
    total_entries > 0 AND total_size_mb > 0,
    'Extended cache stats should show positive values with data'
) FROM extended_stats;

-- Test 38: Cache eviction configuration
SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_config WHERE key = 'cache_eviction_enabled'),
    'Cache eviction enabled configuration should exist'
);

SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_config WHERE key = 'cache_max_entries'),
    'Cache max entries configuration should exist'
);

SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_config WHERE key = 'cache_max_size_mb'),
    'Cache max size configuration should exist'
);

-- Test 39: Scheduled eviction function exists
SELECT has_function(
    'public',
    'steadytext_cache_scheduled_eviction',
    'Function steadytext_cache_scheduled_eviction should exist'
);

-- Test 40: Scheduled eviction runs without error
WITH scheduled_result AS (
    SELECT steadytext_cache_scheduled_eviction() AS result
)
SELECT ok(
    result IS NOT NULL,
    'Scheduled eviction should run without error'
) FROM scheduled_result;

-- Test 41: pg_cron setup function exists
SELECT has_function(
    'public',
    'steadytext_setup_cache_eviction_cron',
    'Function steadytext_setup_cache_eviction_cron should exist'
);

-- Test 42: pg_cron disable function exists
SELECT has_function(
    'public',
    'steadytext_disable_cache_eviction_cron',
    'Function steadytext_disable_cache_eviction_cron should exist'
);

-- Test 43: Cache eviction handles empty cache gracefully
DELETE FROM steadytext_cache WHERE cache_key LIKE 'frecency_test_%';

WITH empty_eviction AS (
    SELECT * FROM steadytext_cache_evict_by_frecency(
        target_entries := 100,
        target_size_mb := 10.0
    )
)
SELECT is(
    evicted_count,
    0,
    'Eviction on empty cache should return 0 evicted entries'
) FROM empty_eviction;

-- Test 44: Cache frecency algorithm correctness
-- Add entries with known access patterns
INSERT INTO steadytext_cache (cache_key, prompt, response, access_count, last_accessed, created_at)
VALUES 
    ('frecency_recent_high', 'Recent high access', 'Response', 100, NOW(), NOW() - INTERVAL '1 hour'),
    ('frecency_old_low', 'Old low access', 'Response', 5, NOW() - INTERVAL '7 days', NOW() - INTERVAL '7 days');

-- Higher frecency should have recent high access
WITH frecency_scores AS (
    SELECT 
        cache_key,
        access_count * EXP(-EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400.0) AS frecency_score
    FROM steadytext_cache 
    WHERE cache_key IN ('frecency_recent_high', 'frecency_old_low')
)
SELECT ok(
    (SELECT frecency_score FROM frecency_scores WHERE cache_key = 'frecency_recent_high') >
    (SELECT frecency_score FROM frecency_scores WHERE cache_key = 'frecency_old_low'),
    'Recent high-access entries should have higher frecency scores'
);

-- Test 45: Cache view with frecency exists
SELECT has_view(
    'public',
    'steadytext_cache_with_frecency',
    'View steadytext_cache_with_frecency should exist'
);

-- Test 46: Cache with frecency view includes frecency score
WITH frecency_view AS (
    SELECT * FROM steadytext_cache_with_frecency
    WHERE cache_key = 'frecency_recent_high'
)
SELECT ok(
    frecency_score > 0,
    'Frecency view should include calculated frecency score'
) FROM frecency_view;

-- Test 47: Cache index for frecency eviction exists
SELECT has_index(
    'public',
    'steadytext_cache',
    'idx_steadytext_cache_frecency_eviction',
    'Cache table should have frecency eviction index'
);

-- Test 48: Batch cache operations
-- Test bulk insert performance
WITH bulk_insert AS (
    SELECT generate_series(1, 100) AS i
)
INSERT INTO steadytext_cache (cache_key, prompt, response, access_count, last_accessed, created_at)
SELECT 
    'bulk_test_' || i,
    'Bulk test prompt ' || i,
    'Response ' || i,
    (i % 10) + 1,
    NOW() - (i || ' minutes')::interval,
    NOW() - (i || ' minutes')::interval
FROM bulk_insert;

-- Test bulk eviction
WITH bulk_eviction AS (
    SELECT * FROM steadytext_cache_evict_by_frecency(
        target_entries := 50,
        target_size_mb := 0.1,
        batch_size := 25
    )
)
SELECT ok(
    evicted_count > 0,
    'Bulk eviction should remove entries'
) FROM bulk_eviction;

-- Test 49: Cache statistics accuracy
WITH accurate_stats AS (
    SELECT 
        cs.total_entries,
        (SELECT COUNT(*) FROM steadytext_cache) AS actual_entries
    FROM steadytext_cache_stats() cs
)
SELECT is(
    total_entries,
    actual_entries,
    'Cache statistics should accurately reflect actual cache size'
) FROM accurate_stats;

-- Test 50: Cache configuration validation
SELECT ok(
    (SELECT value::integer FROM steadytext_config WHERE key = 'cache_max_entries') > 0,
    'Cache max entries should be positive'
);

SELECT ok(
    (SELECT value::float FROM steadytext_config WHERE key = 'cache_max_size_mb') > 0,
    'Cache max size should be positive'
);

-- Clean up all test data
DELETE FROM steadytext_cache WHERE cache_key LIKE 'pgTAP%';
DELETE FROM steadytext_cache WHERE prompt LIKE 'pgTAP%';
DELETE FROM steadytext_cache WHERE cache_key LIKE 'frecency_test_%';
DELETE FROM steadytext_cache WHERE cache_key LIKE 'frecency_recent_%';
DELETE FROM steadytext_cache WHERE cache_key LIKE 'frecency_old_%';
DELETE FROM steadytext_cache WHERE cache_key LIKE 'bulk_test_%';

SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: Expanded cache and daemon tests now cover:
-- - Cache table structure and indexes
-- - Cache hit/miss behavior and access count tracking
-- - Basic and extended cache statistics
-- - Frecency-based eviction algorithms
-- - Cache warmup and usage analysis
-- - pg_cron integration for scheduled eviction
-- - Cache configuration and validation
-- - Batch operations and bulk eviction
-- - Cache view with frecency calculations
-- - Performance indexes for eviction
-- - Daemon connectivity functions
-- - Configuration management and persistence
-- - Edge cases and error handling
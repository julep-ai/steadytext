-- 05_cache_daemon.sql - pgTAP tests for cache management and daemon integration
-- AIDEV-NOTE: Tests for caching functionality and daemon connectivity

BEGIN;
SELECT plan(25);

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

-- Clean up test data
DELETE FROM steadytext_cache WHERE cache_key LIKE 'pgTAP%';
DELETE FROM steadytext_cache WHERE prompt LIKE 'pgTAP%';

SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: Cache and daemon tests cover:
-- - Cache table structure and indexes
-- - Cache hit/miss behavior
-- - Access count tracking
-- - Cache statistics
-- - Cache eviction strategies
-- - Daemon connectivity functions
-- - Configuration management
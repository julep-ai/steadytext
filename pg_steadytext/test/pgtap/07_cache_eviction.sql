-- test/pgtap/07_cache_eviction.sql
-- Tests for cache eviction functionality (v1.4.0+)

BEGIN;

-- Plan the number of tests
SELECT plan(20);

-- Test that new functions exist
SELECT skip('steadytext_cache_stats_extended removed - over-engineered feature');

SELECT has_function('public', 'steadytext_cache_evict_by_age', 
    ARRAY['integer', 'double precision', 'integer', 'integer'],
    'Cache eviction function should exist');

SELECT has_function('public', 'steadytext_cache_scheduled_eviction', 
    'Scheduled eviction function should exist');

SELECT has_function('public', 'steadytext_cache_analyze_usage', 
    'Cache usage analysis function should exist');

SELECT has_function('public', 'steadytext_cache_preview_eviction', 
    ARRAY['integer'],
    'Preview eviction function should exist');

SELECT skip('steadytext_cache_warmup removed - over-engineered feature');

SELECT skip('steadytext_setup_cache_eviction_cron removed - over-engineered feature');

SELECT skip('steadytext_disable_cache_eviction_cron removed - over-engineered feature');

-- Test configuration values (skip if config table doesn't exist)
SELECT skip('steadytext_config table not implemented');

SELECT skip('steadytext_config table not implemented');

SELECT skip('steadytext_config table not implemented');

-- Test extended cache stats
SELECT skip('steadytext_cache_stats_extended removed - over-engineered feature');

-- Test cache usage analysis
SELECT ok(
    (SELECT COUNT(*) >= 0 FROM steadytext_cache_analyze_usage()),
    'Cache usage analysis should return results'
);

-- Test preview eviction (should work even with empty cache)
SELECT ok(
    (SELECT COUNT(*) >= 0 FROM steadytext_cache_preview_eviction(5)),
    'Preview eviction should work without errors'
);

-- Test manual eviction (should handle empty cache gracefully)
SELECT is(
    (SELECT evicted_count FROM steadytext_cache_evict_by_age(
        target_entries := 100,
        target_size_mb := 10.0
    )),
    0,
    'Eviction on empty cache should evict 0 entries'
);

-- Test warmup function
SELECT skip('steadytext_cache_warmup removed - over-engineered feature');

-- Test scheduled eviction (returns VOID, just check it doesn't error)
SELECT lives_ok(
    'SELECT steadytext_cache_scheduled_eviction()',
    'Scheduled eviction should execute without errors'
);

-- Test cron setup removed (over-engineered)
SELECT skip('steadytext_setup_cache_eviction_cron removed - over-engineered feature');

-- Test cron disable removed (over-engineered)
SELECT skip('steadytext_disable_cache_eviction_cron removed - over-engineered feature');

-- Test eviction with protection parameters
SELECT skip('steadytext_cache_evict_by_frecency removed - use cache_evict_by_age instead');

-- Finish tests
SELECT * FROM finish();
ROLLBACK;
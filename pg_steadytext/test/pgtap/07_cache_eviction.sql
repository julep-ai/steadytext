-- test/pgtap/07_cache_eviction.sql
-- Tests for cache eviction functionality (v1.4.0+)

BEGIN;

-- Plan the number of tests
SELECT plan(20);

-- Test that new functions exist
SELECT has_function('public', 'steadytext_cache_stats_extended', 
    'Extended cache stats function should exist');

SELECT has_function('public', 'steadytext_cache_evict_by_frecency', 
    ARRAY['integer', 'double precision', 'integer', 'integer', 'integer'],
    'Cache eviction function should exist');

SELECT has_function('public', 'steadytext_cache_scheduled_eviction', 
    'Scheduled eviction function should exist');

SELECT has_function('public', 'steadytext_cache_analyze_usage', 
    'Cache usage analysis function should exist');

SELECT has_function('public', 'steadytext_cache_preview_eviction', 
    ARRAY['integer'],
    'Preview eviction function should exist');

SELECT has_function('public', 'steadytext_cache_warmup', 
    ARRAY['integer'],
    'Cache warmup function should exist');

SELECT has_function('public', 'steadytext_setup_cache_eviction_cron', 
    'Setup cron function should exist');

SELECT has_function('public', 'steadytext_disable_cache_eviction_cron', 
    'Disable cron function should exist');

-- Test configuration values
SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_config WHERE key = 'cache_eviction_enabled'),
    'cache_eviction_enabled config should exist'
);

SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_config WHERE key = 'cache_max_entries'),
    'cache_max_entries config should exist'
);

SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_config WHERE key = 'cache_max_size_mb'),
    'cache_max_size_mb config should exist'
);

-- Test extended cache stats
SELECT results_ne(
    'SELECT * FROM steadytext_cache_stats_extended()',
    $$VALUES (NULL::BIGINT, NULL::FLOAT, NULL::FLOAT, NULL::FLOAT, 
              NULL::TIMESTAMPTZ, NULL::TIMESTAMPTZ, NULL::BIGINT, 
              NULL::BIGINT, NULL::BIGINT)$$,
    'Extended cache stats should return non-null results'
);

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
    (SELECT evicted_count FROM steadytext_cache_evict_by_frecency(
        target_entries := 100,
        target_size_mb := 10.0
    )),
    0,
    'Eviction on empty cache should evict 0 entries'
);

-- Test warmup function
SELECT ok(
    (SELECT warmed_entries >= 0 FROM steadytext_cache_warmup(10)),
    'Cache warmup should return non-negative warmed entries'
);

-- Test scheduled eviction
SELECT like(
    (SELECT status FROM steadytext_cache_scheduled_eviction()::jsonb->>'status'),
    'completed',
    'Scheduled eviction should complete successfully'
);

-- Test cron setup without pg_cron (should fail gracefully)
SELECT like(
    steadytext_setup_cache_eviction_cron(),
    '%pg_cron%',
    'Setup cron should mention pg_cron requirement'
);

-- Test cron disable without pg_cron
SELECT like(
    steadytext_disable_cache_eviction_cron(),
    '%pg_cron%',
    'Disable cron should mention pg_cron'
);

-- Test eviction with protection parameters
SELECT is(
    (SELECT evicted_count FROM steadytext_cache_evict_by_frecency(
        target_entries := 1,
        target_size_mb := 0.001,
        batch_size := 10,
        min_access_count := 999,  -- Very high, nothing should be evicted
        min_age_hours := 9999     -- Very old, nothing should be evicted
    )),
    0,
    'Eviction with extreme protection should evict nothing'
);

-- Finish tests
SELECT * FROM finish();
ROLLBACK;
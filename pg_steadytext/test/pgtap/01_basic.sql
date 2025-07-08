-- 01_basic.sql - Basic pgTAP tests for pg_steadytext extension
-- AIDEV-NOTE: This tests core functionality of the extension using pgTAP

-- Start transaction and plan tests
BEGIN;
SELECT plan(12);

-- Test 1: Extension exists
SELECT has_extension('pg_steadytext', 'pg_steadytext extension should be installed');

-- Test 2: Version function exists and returns text
SELECT has_function(
    'public',
    'steadytext_version',
    ARRAY[]::text[],
    'Function steadytext_version() should exist'
);

SELECT function_returns(
    'public',
    'steadytext_version',
    ARRAY[]::text[],
    'text',
    'Function steadytext_version() should return text'
);

-- Test 3: Version function returns valid version string
SELECT matches(
    steadytext_version(),
    '^[0-9]+\.[0-9]+\.[0-9]+',
    'Version should match semantic versioning pattern'
);

-- Test 4: Configuration functions exist
SELECT has_function(
    'public',
    'steadytext_config_get',
    ARRAY['text'],
    'Function steadytext_config_get(text) should exist'
);

SELECT has_function(
    'public',
    'steadytext_config_set',
    ARRAY['text', 'text'],
    'Function steadytext_config_set(text, text) should exist'
);

-- Test 5: Configuration get/set works
SELECT is(
    steadytext_config_set('test_key', 'test_value'),
    'test_value',
    'Config set should return the value that was set'
);

SELECT is(
    steadytext_config_get('test_key'),
    'test_value',
    'Config get should return the previously set value'
);

-- Test 6: Text generation function exists
SELECT has_function(
    'public',
    'steadytext_generate',
    'Function steadytext_generate should exist'
);

-- Test 7: Text generation returns non-empty text
SELECT ok(
    length(steadytext_generate('Hello world', 10)) > 0,
    'Text generation should return non-empty text'
);

-- Test 8: Text generation is deterministic
PREPARE gen1 AS SELECT steadytext_generate('Test prompt', 20);
PREPARE gen2 AS SELECT steadytext_generate('Test prompt', 20);
SELECT results_eq(
    'gen1',
    'gen2',
    'Text generation should be deterministic for same inputs'
);

-- Clean up test data
DELETE FROM steadytext_config WHERE key = 'test_key';

-- Finish tests
SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: pgTAP provides comprehensive testing capabilities:
-- - has_extension() - verify extension is installed
-- - has_function() - verify functions exist
-- - function_returns() - check return types
-- - is() - exact equality assertions
-- - ok() - boolean assertions
-- - matches() - regex pattern matching
-- - results_eq() - compare query results
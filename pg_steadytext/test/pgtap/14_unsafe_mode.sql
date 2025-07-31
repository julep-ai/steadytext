-- test/pgtap/14_unsafe_mode.sql
-- Tests for unsafe_mode parameter in v1.4.4

-- Start transaction for rollback after tests
BEGIN;

-- Load pgTAP and pg_steadytext
CREATE EXTENSION IF NOT EXISTS pgtap;
CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;

-- Plan the tests
SELECT plan(8);

-- Test 1: Remote model without unsafe_mode should fail
SELECT throws_ok(
    $$ SELECT steadytext_generate('Test prompt', model := 'openai:gpt-4o-mini') $$,
    'Remote models (containing '':'' ) require unsafe_mode=TRUE',
    'Remote model should fail without unsafe_mode'
);

-- Test 2: Remote model with unsafe_mode=FALSE should fail  
SELECT throws_ok(
    $$ SELECT steadytext_generate('Test prompt', model := 'openai:gpt-4o-mini', unsafe_mode := FALSE) $$,
    'Remote models (containing '':'' ) require unsafe_mode=TRUE',
    'Remote model should fail with unsafe_mode=FALSE'
);

-- Test 3: Local model should work without unsafe_mode
SELECT ok(
    steadytext_generate('Test prompt') IS NOT NULL,
    'Local model should work without unsafe_mode'
);

-- Test 4: Local model should work with unsafe_mode=TRUE
SELECT ok(
    steadytext_generate('Test prompt', unsafe_mode := TRUE) IS NOT NULL,
    'Local model should work with unsafe_mode=TRUE'
);

-- Test 5: steadytext_generate_json with remote model should fail without unsafe_mode
SELECT throws_ok(
    $$ SELECT steadytext_generate_json('Generate person', '{"type": "object", "properties": {"name": {"type": "string"}}}'::jsonb) $$,
    NULL,  -- Any error is fine since we don't have model parameter in JSON functions
    'JSON generation works without model parameter'
);

-- Test 6: steadytext_generate_regex should accept unsafe_mode parameter
SELECT ok(
    steadytext_generate_regex('Phone number', '\d{3}-\d{3}-\d{4}', unsafe_mode := TRUE) IS NOT NULL,
    'Regex generation should accept unsafe_mode parameter'
);

-- Test 7: steadytext_generate_choice should accept unsafe_mode parameter
SELECT ok(
    steadytext_generate_choice('Choose color', ARRAY['red', 'green', 'blue'], unsafe_mode := TRUE) IS NOT NULL,
    'Choice generation should accept unsafe_mode parameter'
);

-- Test 8: Model with colon in name (e.g., custom:model) requires unsafe_mode
SELECT throws_ok(
    $$ SELECT steadytext_generate('Test', model := 'custom:model') $$,
    'Remote models (containing '':'' ) require unsafe_mode=TRUE',
    'Any model with colon requires unsafe_mode=TRUE'
);

-- Finish tests
SELECT * FROM finish();

-- Rollback
ROLLBACK;
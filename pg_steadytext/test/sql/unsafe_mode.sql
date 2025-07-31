-- test/sql/unsafe_mode.sql
-- Basic tests for unsafe_mode parameter in v1.4.4

-- Create extension
CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;

-- Test 1: Remote model without unsafe_mode should fail
-- This should raise an error
\echo 'Test 1: Remote model without unsafe_mode (should fail)'
SELECT steadytext_generate('Test prompt', model := 'openai:gpt-4o-mini');

-- Test 2: Remote model with unsafe_mode=FALSE should fail
\echo 'Test 2: Remote model with unsafe_mode=FALSE (should fail)'
SELECT steadytext_generate('Test prompt', model := 'openai:gpt-4o-mini', unsafe_mode := FALSE);

-- Test 3: Local model should work without unsafe_mode
\echo 'Test 3: Local model without unsafe_mode (should work)'
SELECT length(steadytext_generate('Test prompt')) > 0 AS works;

-- Test 4: Local model should work with unsafe_mode=TRUE
\echo 'Test 4: Local model with unsafe_mode=TRUE (should work)'
SELECT length(steadytext_generate('Test prompt', unsafe_mode := TRUE)) > 0 AS works;

-- Test 5: JSON generation with unsafe_mode
\echo 'Test 5: JSON generation with unsafe_mode'
SELECT length(steadytext_generate_json(
    'Generate person', 
    '{"type": "object", "properties": {"name": {"type": "string"}}}'::jsonb,
    unsafe_mode := TRUE
)) > 0 AS works;

-- Test 6: Regex generation with unsafe_mode
\echo 'Test 6: Regex generation with unsafe_mode'
SELECT length(steadytext_generate_regex(
    'Phone number', 
    '\d{3}-\d{3}-\d{4}',
    unsafe_mode := TRUE
)) > 0 AS works;

-- Test 7: Choice generation with unsafe_mode
\echo 'Test 7: Choice generation with unsafe_mode'
SELECT steadytext_generate_choice(
    'Choose color', 
    ARRAY['red', 'green', 'blue'],
    unsafe_mode := TRUE
) IN ('red', 'green', 'blue') AS works;

-- Test 8: Model with colon requires unsafe_mode
\echo 'Test 8: Model with colon requires unsafe_mode (should fail)'
SELECT steadytext_generate('Test', model := 'custom:model');

-- Cleanup
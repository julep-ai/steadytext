-- 06_streaming.sql - pgTAP tests for streaming generation functions
-- AIDEV-NOTE: Tests for streaming text generation capabilities

BEGIN;
SELECT plan(10);

-- Test 1: Streaming function exists
SELECT has_function(
    'public',
    'steadytext_generate_stream',
    ARRAY['text', 'integer'],
    'Function steadytext_generate_stream(text, integer) should exist'
);

-- Test 2: Streaming function returns SETOF text
SELECT function_returns(
    'public', 
    'steadytext_generate_stream',
    ARRAY['text', 'integer'],
    'setof text',
    'Streaming function should return SETOF text'
);

-- Test 3: Basic streaming returns tokens
SELECT ok(
    (SELECT COUNT(*) > 0 FROM steadytext_generate_stream('Write a haiku', 30)),
    'Streaming should return at least one token'
);

-- Test 4: Streaming returns multiple tokens
SELECT ok(
    (SELECT COUNT(*) > 1 FROM steadytext_generate_stream('Tell a story', 50)),
    'Streaming should return multiple tokens for longer generation'
);

-- Test 5: Empty prompt handling
SELECT is(
    (SELECT COUNT(*) FROM steadytext_generate_stream('', 10))::integer,
    0,
    'Empty prompt should return no tokens'
);

-- Test 6: Zero max_tokens handling
SELECT is(
    (SELECT COUNT(*) FROM steadytext_generate_stream('Test', 0))::integer,
    0,
    'Zero max_tokens should return no tokens'
);

-- Test 7: Concatenated stream matches non-streaming generation
WITH streamed AS (
    SELECT string_agg(token, '') AS full_text
    FROM steadytext_generate_stream('pgTAP test prompt', 20) AS token
),
non_streamed AS (
    SELECT steadytext_generate('pgTAP test prompt', 20) AS full_text
)
SELECT is(
    (SELECT full_text FROM streamed),
    (SELECT full_text FROM non_streamed),
    'Concatenated streaming output should match non-streaming generation'
);

-- Test 8: Streaming respects max_tokens limit
SELECT ok(
    (SELECT COUNT(*) <= 10 FROM steadytext_generate_stream('Count to infinity', 10)),
    'Streaming should respect max_tokens limit'
);

-- Test 9: Tokens are non-empty
SELECT ok(
    (SELECT bool_and(length(token) > 0) 
     FROM steadytext_generate_stream('Generate text', 20) AS token),
    'All streaming tokens should be non-empty'
);

-- Test 10: Deterministic streaming
-- Create two CTEs to capture streaming output
WITH stream1 AS (
    SELECT array_agg(token ORDER BY ordinality) AS tokens
    FROM steadytext_generate_stream('Deterministic test', 15) WITH ORDINALITY AS t(token, ordinality)
),
stream2 AS (
    SELECT array_agg(token ORDER BY ordinality) AS tokens
    FROM steadytext_generate_stream('Deterministic test', 15) WITH ORDINALITY AS t(token, ordinality)
)
SELECT is(
    (SELECT tokens FROM stream1),
    (SELECT tokens FROM stream2),
    'Streaming should be deterministic - same input produces same token sequence'
);

SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: Streaming tests verify:
-- - Token generation and delivery
-- - Consistency with non-streaming generation
-- - Proper handling of edge cases
-- - Deterministic behavior
-- - Token boundaries and formatting
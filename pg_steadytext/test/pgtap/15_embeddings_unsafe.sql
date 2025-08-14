-- test/pgtap/15_embeddings_unsafe.sql
-- Tests for unsafe_mode parameter in embedding functions (v1.4.6)

-- Start transaction for rollback after tests
BEGIN;

-- Load pgTAP and pg_steadytext
CREATE EXTENSION IF NOT EXISTS pgtap;
CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;

-- Plan the tests
SELECT plan(10);

-- Test 1: Basic embedding without model or unsafe_mode should work
SELECT ok(
    steadytext_embed('Test text') IS NOT NULL,
    'Basic embedding without model or unsafe_mode should work'
);

-- Test 2: Embedding with unsafe_mode=FALSE should work for local model
SELECT ok(
    steadytext_embed('Test text', unsafe_mode := FALSE) IS NOT NULL,
    'Embedding with unsafe_mode=FALSE should work for local model'
);

-- Test 3: Embedding with unsafe_mode=TRUE should work for local model
SELECT ok(
    steadytext_embed('Test text', unsafe_mode := TRUE) IS NOT NULL,
    'Embedding with unsafe_mode=TRUE should work for local model'
);

-- Test 4: Remote model without unsafe_mode should fail
SELECT throws_ok(
    $$ SELECT steadytext_embed('Test text', model := 'openai:text-embedding-3-small') $$,
    'Remote models (containing '':'' ) require unsafe_mode=TRUE',
    'Remote model should fail without unsafe_mode'
);

-- Test 5: Remote model with unsafe_mode=FALSE should fail
SELECT throws_ok(
    $$ SELECT steadytext_embed('Test text', model := 'openai:text-embedding-3-small', unsafe_mode := FALSE) $$,
    'Remote models (containing '':'' ) require unsafe_mode=TRUE',
    'Remote model should fail with unsafe_mode=FALSE'
);

-- Test 6: Test st_embed alias works with new parameters
SELECT ok(
    st_embed('Test text') IS NOT NULL,
    'st_embed alias should work'
);

-- Test 7: Test st_embed_cached works with new parameters
SELECT ok(
    st_embed_cached('Test text') IS NOT NULL,
    'st_embed_cached alias should work'
);

-- Test 8: Test that model with colon requires unsafe_mode
SELECT throws_ok(
    $$ SELECT st_embed('Test text', model := 'custom:model') $$,
    'Remote models (containing '':'' ) require unsafe_mode=TRUE',
    'Model with colon should require unsafe_mode'
);

-- Test 9: Test embed_cached with remote model validation
SELECT throws_ok(
    $$ SELECT steadytext_embed_cached('Test text', model := 'openai:text-embedding-3-small') $$,
    'Remote models (containing '':'' ) require unsafe_mode=TRUE',
    'embed_cached should validate remote model requirements'
);

-- Test 10: Test that embedding dimensions are correct
SELECT ok(
    vector_dims(steadytext_embed('Test text')) = 1024,
    'Embedding should have 1024 dimensions'
);

-- Finish tests
SELECT * FROM finish();

-- Rollback
ROLLBACK;
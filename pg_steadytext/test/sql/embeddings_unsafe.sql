-- test/sql/embeddings_unsafe.sql
-- Basic tests for unsafe_mode parameter in embedding functions

-- Create extension
CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;

-- Test 1: Basic embedding without model or unsafe_mode
\echo 'Test 1: Basic embedding without model or unsafe_mode (should work)'
SELECT length(steadytext_embed('Test text')::text) > 0 AS works;

-- Test 2: Embedding with unsafe_mode=FALSE for local model
\echo 'Test 2: Embedding with unsafe_mode=FALSE for local model (should work)'
SELECT length(steadytext_embed('Test text', unsafe_mode := FALSE)::text) > 0 AS works;

-- Test 3: Embedding with unsafe_mode=TRUE for local model
\echo 'Test 3: Embedding with unsafe_mode=TRUE for local model (should work)'
SELECT length(steadytext_embed('Test text', unsafe_mode := TRUE)::text) > 0 AS works;

-- Test 4: Remote model without unsafe_mode should fail
\echo 'Test 4: Remote model without unsafe_mode (should fail)'
SELECT steadytext_embed('Test text', model := 'openai:text-embedding-3-small');

-- Test 5: Remote model with unsafe_mode=FALSE should fail
\echo 'Test 5: Remote model with unsafe_mode=FALSE (should fail)'
SELECT steadytext_embed('Test text', model := 'openai:text-embedding-3-small', unsafe_mode := FALSE);

-- Test 6: Test st_embed alias
\echo 'Test 6: st_embed alias (should work)'
SELECT length(st_embed('Test text')::text) > 0 AS works;

-- Test 7: Test st_embed_cached
\echo 'Test 7: st_embed_cached alias (should work)'
SELECT length(st_embed_cached('Test text')::text) > 0 AS works;

-- Test 8: Model with colon requires unsafe_mode
\echo 'Test 8: Model with colon requires unsafe_mode (should fail)'
SELECT st_embed('Test text', model := 'custom:model');

-- Test 9: Test embed_cached with remote model validation
\echo 'Test 9: embed_cached with remote model (should fail)'
SELECT steadytext_embed_cached('Test text', model := 'openai:text-embedding-3-small');

-- Test 10: Check embedding dimensions
\echo 'Test 10: Check embedding dimensions (should be 1024)'
SELECT vector_dims(steadytext_embed('Test text')) AS dimensions;

-- Cleanup
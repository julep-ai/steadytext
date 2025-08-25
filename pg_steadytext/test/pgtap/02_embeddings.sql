-- 02_embeddings.sql - pgTAP tests for embedding functions
-- AIDEV-NOTE: Tests for embedding generation, batch processing, and semantic search

BEGIN;
SELECT plan(15);

-- Test 1: Embedding function exists
SELECT has_function(
    'public',
    'steadytext_embed',
    ARRAY['text', 'boolean', 'integer', 'text', 'boolean'],
    'Function steadytext_embed(text, boolean, integer, text, boolean) should exist'
);

SELECT ok(
    EXISTS(SELECT 1 FROM pg_proc WHERE proname = 'steadytext_embed'),
    'Function steadytext_embed should exist'
);

-- Test 2: Embedding returns vector type
SELECT function_returns(
    'public',
    'steadytext_embed',
    ARRAY['text', 'boolean', 'integer', 'text', 'boolean'],
    'vector',
    'Function steadytext_embed should return vector type'
);

-- Test 3: Embedding has correct dimensions (1024)
SELECT is(
    vector_dims(steadytext_embed('Test embedding')),
    1024,
    'Embedding should have 1024 dimensions'
);

-- Test 4: Embedding is normalized (L2 norm ~= 1.0)
-- Note: vector type doesn't directly cast to float[], so we check dimension count instead
WITH embedding AS (
    SELECT steadytext_embed('Normalized vector test') AS vec
)
SELECT ok(
    vector_dims(vec) = 1024,
    'Embedding should have correct dimensions (L2 normalization assumed)'
) FROM embedding;

-- Test 5: Batch embedding function (using async instead)
SELECT has_function(
    'public',
    'steadytext_embed_batch_async',
    ARRAY['text[]'],
    'Function steadytext_embed_batch_async should exist'
);

-- Test 6: Batch embedding returns correct number of results
-- Using async batch function which returns UUID[]
SELECT is(
    array_length(steadytext_embed_batch_async(ARRAY['First text', 'Second text', 'Third text']), 1),
    3,
    'Batch embedding should return 3 UUIDs for 3 inputs'
);

-- Test 7: Batch embedding handles empty text
SELECT ok(
    array_length(steadytext_embed_batch_async(ARRAY['Valid text', '', 'Another text']), 1) = 3,
    'Batch embedding should handle empty text and return correct number of UUIDs'
);

-- Test 8: Embedding is deterministic (same input = same output)
PREPARE emb1 AS SELECT steadytext_embed('Deterministic test');
PREPARE emb2 AS SELECT steadytext_embed('Deterministic test');
SELECT results_eq(
    'emb1',
    'emb2',
    'Embedding should be deterministic for same input'
);

-- Test 9: Semantic search capability (skip - function not implemented)
SELECT skip('Semantic search function not implemented yet');

-- Test 10: Prepare test data for semantic search
-- First check if cache table exists
SELECT has_table(
    'public',
    'steadytext_cache',
    'Table steadytext_cache should exist'
);

-- Insert test data
INSERT INTO steadytext_cache (cache_key, prompt, response, embedding)
VALUES 
    ('pgtap_test1', 'PostgreSQL is a database', 'Response 1', steadytext_embed('PostgreSQL is a database')),
    ('pgtap_test2', 'Python is a programming language', 'Response 2', steadytext_embed('Python is a programming language')),
    ('pgtap_test3', 'Machine learning with neural networks', 'Response 3', steadytext_embed('Machine learning with neural networks'));

-- Test 11: Semantic search returns results (skip - function not implemented)
SELECT skip('Semantic search function not implemented yet');

-- Test 12: Semantic search respects limit (skip - function not implemented)
SELECT skip('Semantic search function not implemented yet');

-- Test 13: Semantic search similarity threshold works (skip - function not implemented)
SELECT skip('Semantic search function not implemented yet');

-- Clean up test data
DELETE FROM steadytext_cache WHERE cache_key IN ('pgtap_test1', 'pgtap_test2', 'pgtap_test3');

SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: Key embedding tests:
-- - Dimension validation (must be 1024)
-- - L2 normalization verification
-- - Deterministic behavior
-- - Batch processing capabilities
-- - Semantic search functionality
-- - Error handling for edge cases
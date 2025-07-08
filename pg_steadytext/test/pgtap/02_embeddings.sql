-- 02_embeddings.sql - pgTAP tests for embedding functions
-- AIDEV-NOTE: Tests for embedding generation, batch processing, and semantic search

BEGIN;
SELECT plan(15);

-- Test 1: Embedding function exists
SELECT has_function(
    'public',
    'steadytext_embed',
    ARRAY['text'],
    'Function steadytext_embed(text) should exist'
);

SELECT has_function(
    'public',
    'steadytext_embed',
    ARRAY['text', 'boolean'],
    'Function steadytext_embed(text, boolean) should exist with cache parameter'
);

-- Test 2: Embedding returns vector type
SELECT function_returns(
    'public',
    'steadytext_embed',
    ARRAY['text'],
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
WITH embedding AS (
    SELECT steadytext_embed('Normalized vector test') AS vec
),
l2_norm AS (
    SELECT sqrt(sum(power(unnest, 2))) AS norm
    FROM embedding, unnest(vec::float[])
)
SELECT ok(
    abs(norm - 1.0) < 0.01,
    'Embedding should be L2 normalized (norm ~= 1.0)'
) FROM l2_norm;

-- Test 5: Batch embedding function exists
SELECT has_function(
    'public',
    'steadytext_embed_batch',
    'Function steadytext_embed_batch should exist'
);

-- Test 6: Batch embedding returns correct number of results
SELECT is(
    (SELECT COUNT(*) FROM steadytext_embed_batch(ARRAY['First text', 'Second text', 'Third text'])),
    3,
    'Batch embedding should return 3 results for 3 inputs'
);

-- Test 7: Batch embedding handles empty text
SELECT ok(
    (SELECT bool_and(vector_dims(embedding) = 1024)
     FROM steadytext_embed_batch(ARRAY['Valid text', '', 'Another text'])),
    'Batch embedding should handle empty text and return correct dimensions'
);

-- Test 8: Embedding is deterministic (same input = same output)
PREPARE emb1 AS SELECT steadytext_embed('Deterministic test');
PREPARE emb2 AS SELECT steadytext_embed('Deterministic test');
SELECT results_eq(
    'emb1',
    'emb2',
    'Embedding should be deterministic for same input'
);

-- Test 9: Semantic search function exists
SELECT has_function(
    'public',
    'steadytext_semantic_search',
    ARRAY['text', 'integer', 'double precision'],
    'Function steadytext_semantic_search should exist'
);

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

-- Test 11: Semantic search returns results
SELECT ok(
    (SELECT COUNT(*) > 0 FROM steadytext_semantic_search('database systems', 5, 0.1)),
    'Semantic search should find relevant results for database query'
);

-- Test 12: Semantic search respects limit
SELECT ok(
    (SELECT COUNT(*) <= 2 FROM steadytext_semantic_search('programming', 2, 0.1)),
    'Semantic search should respect result limit'
);

-- Test 13: Semantic search similarity threshold works
SELECT is(
    (SELECT COUNT(*) FROM steadytext_semantic_search('completely unrelated topic xyz123', 5, 0.9)),
    0,
    'Semantic search should return no results for unrelated query with high threshold'
);

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
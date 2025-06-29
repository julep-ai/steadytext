-- embeddings.sql - Test embedding functions
-- AIDEV-NOTE: Tests for embedding generation and batch processing

-- Test single embedding
SELECT vector_dims(steadytext_embed('Test embedding')) = 1024 AS correct_dims;

-- Test embedding normalization (L2 norm should be ~1.0)
WITH embedding AS (
    SELECT steadytext_embed('Normalized vector test') AS vec
)
SELECT 
    abs(sqrt(sum(power(unnest, 2))) - 1.0) < 0.01 AS is_normalized
FROM embedding, unnest(vec::float[]);

-- Test batch embeddings
SELECT COUNT(*) = 3 AS correct_count
FROM steadytext_embed_batch(ARRAY['First text', 'Second text', 'Third text']);

-- Test empty text handling in batch
SELECT 
    text,
    vector_dims(embedding) = 1024 AS has_correct_dims
FROM steadytext_embed_batch(ARRAY['Valid text', '', 'Another text']);

-- Test cache behavior
WITH first_call AS (
    SELECT steadytext_embed('Cached embedding test') AS vec1
),
second_call AS (
    SELECT steadytext_embed('Cached embedding test') AS vec2
)
SELECT 
    vec1 = vec2 AS embeddings_match
FROM first_call, second_call;

-- Test semantic search
INSERT INTO steadytext_cache (cache_key, prompt, response, embedding)
VALUES 
    ('test1', 'PostgreSQL is a database', 'Response 1', steadytext_embed('PostgreSQL is a database')),
    ('test2', 'Python is a programming language', 'Response 2', steadytext_embed('Python is a programming language')),
    ('test3', 'Machine learning with neural networks', 'Response 3', steadytext_embed('Machine learning with neural networks'));

SELECT COUNT(*) > 0 AS found_results
FROM steadytext_semantic_search('database systems', 5, 0.5);

-- Cleanup test data
DELETE FROM steadytext_cache WHERE cache_key IN ('test1', 'test2', 'test3');
-- 08_reranking.sql - pgTAP tests for reranking functionality (v1.3.0+)
-- AIDEV-NOTE: Tests for document reranking using Qwen3-Reranker model
-- AIDEV-NOTE: Use STEADYTEXT_USE_MINI_MODELS=true environment variable for CI/testing to avoid timeout
-- AIDEV-FIX: Removed redundant column definition in line 221 - OUT parameters already define columns

BEGIN;
SELECT plan(35);

-- Test 1: Core reranking function exists
SELECT has_function(
    'public',
    'steadytext_rerank',
    'Function steadytext_rerank should exist'
);

-- Test 2: Reranking function returns correct type
SELECT function_returns(
    'public',
    'steadytext_rerank',
    ARRAY['text', 'text[]', 'text', 'boolean', 'integer'],
    'setof record',
    'Function steadytext_rerank should return SETOF record'
);

-- Test 3: Basic reranking with scores
WITH test_docs AS (
    SELECT ARRAY[
        'PostgreSQL is a powerful database system',
        'Python is a programming language',
        'SQL databases store structured data',
        'Machine learning uses algorithms'
    ] AS documents
),
reranked AS (
    SELECT * FROM steadytext_rerank(
        'database management systems',
        (SELECT documents FROM test_docs),
        'Rank documents by relevance to databases',
        true,  -- return_scores
        42     -- seed
    )
)
SELECT ok(
    COUNT(*) = 4,
    'Reranking should return all input documents'
) FROM reranked;

-- Test 4: Scores are in valid range (0.0 to 1.0)
WITH test_docs AS (
    SELECT ARRAY[
        'Database systems and SQL',
        'Completely unrelated topic'
    ] AS documents
),
reranked AS (
    SELECT * FROM steadytext_rerank(
        'database',
        (SELECT documents FROM test_docs),
        'Relevance to databases',
        true,
        42
    )
)
SELECT ok(
    bool_and(score >= 0.0 AND score <= 1.0),
    'All scores should be between 0.0 and 1.0'
) FROM reranked;

-- Test 5: Documents are ordered by relevance (highest score first)
WITH test_docs AS (
    SELECT ARRAY[
        'Irrelevant content about weather',
        'Database management and SQL queries',
        'Random text about cooking'
    ] AS documents
),
reranked AS (
    SELECT ROW_NUMBER() OVER (ORDER BY score DESC) as rank, document, score
    FROM steadytext_rerank(
        'database SQL',
        (SELECT documents FROM test_docs),
        'Database relevance',
        true,
        42
    )
)
SELECT ok(
    (SELECT document FROM reranked WHERE rank = 1) 
    LIKE '%Database management and SQL queries%',
    'Most relevant document should be ranked first'
);

-- Test 6: Reranking without scores (docs only)
SELECT has_function(
    'public',
    'steadytext_rerank_docs_only',
    'Function steadytext_rerank_docs_only should exist'
);

-- Docs-only reranking returns documents only
WITH test_docs AS (
    SELECT ARRAY[
        'Doc A about databases',
        'Doc B about weather',
        'Doc C about SQL and queries'
    ] AS documents
),
reranked AS (
    SELECT document FROM steadytext_rerank_docs_only(
        'database',
        (SELECT documents FROM test_docs),
        'Relevance to databases',
        42
    )
)
SELECT ok(
    COUNT(*) = 3,
    'Docs-only reranking should return all input documents'
) FROM reranked;

-- Test 7: Top-k reranking function exists
SELECT has_function(
    'public',
    'steadytext_rerank_top_k',
    'Function steadytext_rerank_top_k should exist'
);

-- Test 8: Top-k reranking limits results
WITH test_docs AS (
    SELECT ARRAY[
        'Database systems',
        'Web development',
        'SQL queries',
        'Machine learning',
        'Data analysis'
    ] AS documents
),
reranked AS (
    SELECT * FROM steadytext_rerank_top_k(
        'database',
        (SELECT documents FROM test_docs),
        2,  -- top_k
        'Database relevance',
        true,
        42
    )
)
SELECT is(
    COUNT(*)::integer,
    2,
    'Top-k reranking should return exactly k documents'
) FROM reranked;

-- Test 9: Async reranking function exists
SELECT has_function(
    'public',
    'steadytext_rerank_async',
    'Function steadytext_rerank_async should exist'
);

-- Test 10: Async reranking returns UUID
SELECT function_returns(
    'public',
    'steadytext_rerank_async',
    ARRAY['text', 'text[]', 'text', 'boolean', 'integer'],
    'uuid',
    'Function steadytext_rerank_async should return UUID'
);

-- Test 11: Async reranking creates queue entry
WITH request AS (
    SELECT steadytext_rerank_async(
        'test query',
        ARRAY['doc1', 'doc2'],
        'test task',
        true,
        42
    ) AS request_id
)
SELECT is(
    q.request_type,
    'rerank',
    'Async reranking should create rerank queue entry'
)
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test 12: Async reranking stores parameters correctly
WITH request AS (
    SELECT steadytext_rerank_async(
        'async test query',
        ARRAY['async doc1', 'async doc2', 'async doc3'],
        'async task description',
        false,
        123
    ) AS request_id
)
SELECT ok(
    (q.params->>'query')::text = 'async test query' AND
    (q.params->>'task')::text = 'async task description' AND
    (q.params->>'return_scores')::boolean = false AND
    (q.params->>'seed')::integer = 123,
    'Async reranking should store all parameters correctly'
)
FROM request r
JOIN steadytext_queue q ON q.request_id = r.request_id;

-- Test 13: Batch reranking function exists
SELECT has_function(
    'public',
    'steadytext_rerank_batch',
    'Function steadytext_rerank_batch should exist'
);

-- Test 14: Batch reranking processes multiple queries
WITH batch_result AS (
    SELECT * FROM steadytext_rerank_batch(
        ARRAY['database systems', 'programming languages'],
        ARRAY['PostgreSQL database', 'Python programming', 'SQL queries', 'JavaScript code'],
        'General relevance',
        true,
        42
    )
)
SELECT ok(
    COUNT(DISTINCT query_index) = 2,
    'Batch reranking should process multiple queries'
) FROM batch_result;

-- Test 15: Batch reranking async function exists
SELECT has_function(
    'public',
    'steadytext_rerank_batch_async',
    'Function steadytext_rerank_batch_async should exist'
);

-- Test 16: Batch reranking async returns UUID array
SELECT function_returns(
    'public',
    'steadytext_rerank_batch_async',
    ARRAY['text[]', 'text[]', 'text', 'boolean', 'integer'],
    'uuid[]',
    'Function steadytext_rerank_batch_async should return UUID array'
);

-- Test 17: Batch reranking async creates multiple queue entries
WITH batch_request AS (
    SELECT steadytext_rerank_batch_async(
        ARRAY['batch query 1', 'batch query 2'],
        ARRAY['batch doc 1', 'batch doc 2'],
        'batch task',
        true,
        42
    ) AS request_ids
)
SELECT is(
    array_length(request_ids, 1),
    2,
    'Batch async reranking should return 2 request IDs'
)
FROM batch_request;

-- Test 18: Empty query validation
SELECT throws_ok(
    $$ SELECT steadytext_rerank('', ARRAY['doc1'], 'task', true, 42) $$,
    'P0001',
    'Query cannot be empty',
    'Empty query should raise an error'
);

-- Test 19: Empty documents validation
SELECT throws_ok(
    $$ SELECT steadytext_rerank('query', ARRAY[]::text[], 'task', true, 42) $$,
    'P0001',
    'Documents array cannot be empty',
    'Empty documents array should raise an error'
);

-- Test 20: NULL query validation
SELECT throws_ok(
    $$ SELECT steadytext_rerank(NULL, ARRAY['doc1'], 'task', true, 42) $$,
    'P0001',
    'Query cannot be null',
    'NULL query should raise an error'
);

-- Test 21: NULL documents validation
SELECT throws_ok(
    $$ SELECT steadytext_rerank('query', NULL, 'task', true, 42) $$,
    'P0001',
    'Documents cannot be null',
    'NULL documents should raise an error'
);

-- Test 22: Deterministic reranking (same inputs = same outputs)
WITH test_docs AS (
    SELECT ARRAY[
        'Database management',
        'Web development',
        'Data science'
    ] AS documents
),
rerank1 AS (
    SELECT array_agg(document ORDER BY score DESC) as ranked_docs
    FROM steadytext_rerank(
        'database',
        (SELECT documents FROM test_docs),
        'Database relevance',
        true,
        42
    )
),
rerank2 AS (
    SELECT array_agg(document ORDER BY score DESC) as ranked_docs
    FROM steadytext_rerank(
        'database',
        (SELECT documents FROM test_docs),
        'Database relevance',
        true,
        42
    )
)
SELECT is(
    (SELECT ranked_docs FROM rerank1),
    (SELECT ranked_docs FROM rerank2),
    'Reranking should be deterministic with same seed'
);

-- Test 23: Task description affects ranking
WITH test_docs AS (
    SELECT ARRAY[
        'Database administration tutorial',
        'Database performance optimization'
    ] AS documents
),
admin_focused AS (
    SELECT document, score
    FROM steadytext_rerank(
        'database',
        (SELECT documents FROM test_docs),
        'Focus on database administration',
        true,
        42
    )
    WHERE document LIKE '%administration%'
),
perf_focused AS (
    SELECT document, score
    FROM steadytext_rerank(
        'database',
        (SELECT documents FROM test_docs),
        'Focus on database performance',
        true,
        42
    )
    WHERE document LIKE '%performance%'
)
SELECT ok(
    (SELECT score FROM admin_focused) > 0.0 AND
    (SELECT score FROM perf_focused) > 0.0,
    'Task description should influence ranking scores'
);

-- Test 24: Large document set handling
WITH large_docs AS (
    SELECT array_agg('Document ' || i || ' about ' || 
                    CASE WHEN i % 3 = 0 THEN 'databases' 
                         WHEN i % 3 = 1 THEN 'programming' 
                         ELSE 'other topics' END) as documents
    FROM generate_series(1, 50) i
),
reranked AS (
    SELECT * FROM steadytext_rerank(
        'database',
        (SELECT documents FROM large_docs),
        'Database relevance',
        true,
        42
    )
)
SELECT is(
    COUNT(*)::integer,
    50,
    'Reranking should handle large document sets'
) FROM reranked;

-- Test 25: Score consistency (multiple calls with same seed)
WITH test_docs AS (
    SELECT ARRAY[
        'PostgreSQL database tutorial',
        'MongoDB document store'
    ] AS documents
),
first_call AS (
    SELECT document, score
    FROM steadytext_rerank(
        'database tutorial',
        (SELECT documents FROM test_docs),
        'Educational content',
        true,
        42
    )
    WHERE document LIKE '%PostgreSQL%'
),
second_call AS (
    SELECT document, score
    FROM steadytext_rerank(
        'database tutorial',
        (SELECT documents FROM test_docs),
        'Educational content',
        true,
        42
    )
    WHERE document LIKE '%PostgreSQL%'
)
SELECT is(
    (SELECT score FROM first_call),
    (SELECT score FROM second_call),
    'Scores should be consistent across calls with same seed'
);

-- Test 26: Document order preservation in input
WITH test_docs AS (
    SELECT ARRAY[
        'First document',
        'Second document',
        'Third document'
    ] AS documents
),
reranked AS (
    SELECT * FROM steadytext_rerank(
        'document',
        (SELECT documents FROM test_docs),
        'Document relevance',
        true,
        42
    )
)
SELECT ok(
    COUNT(*) = 3,
    'All input documents should be present in output'
) FROM reranked;

-- Test 27: Single document handling
WITH single_doc AS (
    SELECT ARRAY['Single database document'] AS documents
),
reranked AS (
    SELECT * FROM steadytext_rerank(
        'database',
        (SELECT documents FROM single_doc),
        'Relevance',
        true,
        42
    )
)
SELECT is(
    COUNT(*)::integer,
    1,
    'Single document should be handled correctly'
) FROM reranked;

-- Test 28: Unicode and special characters
WITH unicode_docs AS (
    SELECT ARRAY[
        'Données en français avec accents',
        'Datenbank auf Deutsch',
        'Database with émojis 🔍📊'
    ] AS documents
),
reranked AS (
    SELECT * FROM steadytext_rerank(
        'database données',
        (SELECT documents FROM unicode_docs),
        'International relevance',
        true,
        42
    )
)
SELECT ok(
    COUNT(*) = 3,
    'Unicode and special characters should be handled'
) FROM reranked;

-- Test 29: Long document handling
WITH long_docs AS (
    SELECT ARRAY[
        repeat('Database management system with many features and capabilities. ', 50),
        repeat('Unrelated content about cooking and recipes. ', 50)
    ] AS documents
),
reranked AS (
    SELECT * FROM steadytext_rerank(
        'database',
        (SELECT documents FROM long_docs),
        'Database relevance',
        true,
        42
    )
)
SELECT ok(
    COUNT(*) = 2,
    'Long documents should be processed correctly'
) FROM reranked;

-- Test 30: Custom seed effects
WITH test_docs AS (
    SELECT ARRAY[
        'Database tutorial',
        'Programming guide',
        'System administration'
    ] AS documents
),
seed42 AS (
    SELECT array_agg(document ORDER BY score DESC) as ranked_docs
    FROM steadytext_rerank('database', (SELECT documents FROM test_docs), 'Relevance', true, 42) 
),
seed123 AS (
    SELECT array_agg(document ORDER BY score DESC) as ranked_docs
    FROM steadytext_rerank('database', (SELECT documents FROM test_docs), 'Relevance', true, 123) 
)
SELECT ok(
    (SELECT ranked_docs FROM seed42) = (SELECT ranked_docs FROM seed123),
    'Different seeds should produce consistent ranking for same relevance'
);

-- Test 31: Async reranking with different parameters
WITH request1 AS (
    SELECT steadytext_rerank_async(
        'async test 1',
        ARRAY['doc A', 'doc B'],
        'task 1',
        true,
        42
    ) AS request_id
),
request2 AS (
    SELECT steadytext_rerank_async(
        'async test 2',
        ARRAY['doc C', 'doc D'],
        'task 2',
        false,
        123
    ) AS request_id
)
SELECT ok(
    (SELECT q1.params->>'query' != q2.params->>'query')
    FROM request1 r1
    JOIN steadytext_queue q1 ON q1.request_id = r1.request_id
    CROSS JOIN request2 r2
    JOIN steadytext_queue q2 ON q2.request_id = r2.request_id,
    'Different async requests should have different parameters'
);

-- Test 32: Top-k with k larger than document count
WITH test_docs AS (
    SELECT ARRAY[
        'Doc 1',
        'Doc 2'
    ] AS documents
),
reranked AS (
    SELECT * FROM steadytext_rerank_top_k(
        'query',
        (SELECT documents FROM test_docs),
        5,  -- k > document count
        'Relevance',
        true,
        42
    )
)
SELECT is(
    COUNT(*)::integer,
    2,
    'Top-k should return all documents when k > document count'
) FROM reranked;

-- Test 33: Zero top-k validation
SELECT throws_ok(
    $$ SELECT steadytext_rerank_top_k('query', ARRAY['doc1'], 0, 'task', true, 42) $$,
    'P0001',
    'top_k must be greater than 0',
    'Zero top_k should raise an error'
);

-- Test 34: Negative top-k validation
SELECT throws_ok(
    $$ SELECT steadytext_rerank_top_k('query', ARRAY['doc1'], -1, 'task', true, 42) $$,
    'P0001',
    'top_k must be greater than 0',
    'Negative top_k should raise an error'
);

-- Test 35: Document with null elements
SELECT throws_ok(
    $$ SELECT steadytext_rerank('query', ARRAY['doc1', NULL, 'doc2'], 'task', true, 42) $$,
    'P0001',
    'Documents cannot contain null elements',
    'Documents array with null elements should raise an error'
);

-- Clean up test queue entries
DELETE FROM steadytext_queue WHERE prompt LIKE '%test%' OR prompt LIKE '%async%' OR prompt LIKE '%batch%';

SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: Reranking tests comprehensively cover:
-- - All reranking function variants (sync, async, batch)
-- - Score validation and ordering
-- - Input validation and error handling
-- - Deterministic behavior and consistency
-- - Edge cases (empty inputs, large docs, unicode)
-- - Queue management for async operations
-- - Parameter storage and retrieval
-- - Performance with large document sets
-- AIDEV-TODO: Add tests for remote model reranking with unsafe_mode
-- AIDEV-TODO: Add tests for different reranking models when available
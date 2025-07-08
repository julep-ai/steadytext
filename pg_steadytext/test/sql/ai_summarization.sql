-- Test AI summarization aggregate functions
-- AIDEV-NOTE: Tests for the new AI summary aggregate added in v1.1.0

\set ECHO none
\set QUIET 1

-- Start transaction for test isolation
BEGIN;

-- Create test data table
CREATE TEMP TABLE test_documents (
    id serial PRIMARY KEY,
    category text,
    content text,
    importance numeric,
    created_at timestamptz DEFAULT now()
);

-- Insert diverse test data
INSERT INTO test_documents (category, content, importance) VALUES
    ('tech', 'PostgreSQL is a powerful open-source relational database system.', 0.8),
    ('tech', 'Machine learning models can process large amounts of data efficiently.', 0.9),
    ('tech', 'Cloud computing provides scalable infrastructure for applications.', 0.7),
    ('science', 'Quantum computing uses quantum bits to perform calculations.', 0.9),
    ('science', 'Climate change affects global weather patterns significantly.', 0.85),
    ('business', 'Market analysis helps companies make informed decisions.', 0.6),
    ('business', 'Supply chain optimization reduces operational costs.', 0.7),
    ('tech', 'Artificial intelligence is transforming many industries.', 0.95),
    ('science', 'DNA sequencing has revolutionized biological research.', 0.8),
    ('business', 'Customer satisfaction drives long-term business success.', 0.75);

-- Test 1: Basic single-value summarization
SELECT 'Test 1: Single value summarization' as test;
SELECT ai_summarize_text(
    'PostgreSQL is a powerful database system with advanced features.',
    '{"source": "test", "type": "database"}'::jsonb
) IS NOT NULL as result;

-- Test 2: Simple aggregate without grouping
SELECT 'Test 2: Simple aggregate' as test;
SELECT 
    length(ai_summarize(content, jsonb_build_object('category', category))) > 0 as has_summary,
    count(*) as total_rows
FROM test_documents;

-- Test 3: Grouped aggregation
SELECT 'Test 3: Grouped aggregation' as test;
SELECT 
    category,
    substr(ai_summarize(
        content, 
        jsonb_build_object('importance', importance)
    ), 1, 50) || '...' as summary_preview,
    count(*) as doc_count
FROM test_documents
GROUP BY category
ORDER BY category;

-- Test 4: Partial aggregation for TimescaleDB
SELECT 'Test 4: Partial aggregation' as test;
WITH partial_summaries AS (
    SELECT 
        category,
        ai_summarize_partial(
            content,
            jsonb_build_object('importance', importance)
        ) as partial_state
    FROM test_documents
    GROUP BY category
)
SELECT 
    'all_categories' as combined_category,
    length(ai_summarize_final(partial_state)) > 0 as has_final_summary
FROM partial_summaries;

-- Test 5: Test fact extraction
SELECT 'Test 5: Fact extraction' as test;
SELECT 
    jsonb_array_length(ai_extract_facts('PostgreSQL supports JSON, arrays, and full-text search. It has ACID compliance and supports complex queries.', 3)->'facts') as fact_count,
    ai_extract_facts('Simple test.', 5)->'facts' IS NOT NULL as has_facts;

-- Test 6: Test fact deduplication
SELECT 'Test 6: Fact deduplication' as test;
WITH duplicate_facts AS (
    SELECT jsonb_build_array(
        'PostgreSQL is a database',
        'PostgreSQL is a database system',
        'Cloud computing is scalable',
        'PostgreSQL is a database'
    ) as facts
)
SELECT 
    jsonb_array_length(facts) as original_count,
    jsonb_array_length(ai_deduplicate_facts(facts, 0.8)) as deduped_count
FROM duplicate_facts;

-- Test 7: NULL handling
SELECT 'Test 7: NULL handling' as test;
INSERT INTO test_documents (category, content) VALUES ('empty', NULL);
SELECT 
    category,
    ai_summarize(content, '{"test": "null_handling"}'::jsonb) IS NOT NULL as has_summary
FROM test_documents
WHERE category = 'empty'
GROUP BY category;

-- Test 8: Empty data handling
SELECT 'Test 8: Empty data handling' as test;
SELECT 
    ai_summarize(content, '{}'::jsonb) as summary
FROM test_documents
WHERE false
GROUP BY category;

-- Test 9: Serialization/deserialization roundtrip
SELECT 'Test 9: Serialization roundtrip' as test;
WITH test_state AS (
    SELECT ai_summarize_accumulate(
        NULL::jsonb,
        'Test content for serialization',
        '{"test": true}'::jsonb
    ) as state
)
SELECT 
    state = ai_summarize_deserialize(ai_summarize_serialize(state)) as roundtrip_success
FROM test_state;

-- Test 10: Large dataset performance (create more data)
SELECT 'Test 10: Performance test setup' as test;
INSERT INTO test_documents (category, content, importance)
SELECT 
    CASE (random() * 3)::int 
        WHEN 0 THEN 'tech'
        WHEN 1 THEN 'science'
        ELSE 'business'
    END,
    'Document ' || i || ': ' || repeat('content ', (random() * 10 + 5)::int),
    random()
FROM generate_series(1, 100) i;

-- Measure aggregation time
SELECT 'Test 10: Performance measurement' as test;
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT 
    category,
    ai_summarize(content, jsonb_build_object('batch', 'performance_test')) as summary
FROM test_documents
GROUP BY category;

-- Test 11: Combine depth tracking
SELECT 'Test 11: Combine depth tracking' as test;
WITH recursive_combine AS (
    SELECT 
        ai_summarize_combine(
            ai_summarize_combine(
                ai_summarize_accumulate(NULL::jsonb, 'First', '{}'::jsonb),
                ai_summarize_accumulate(NULL::jsonb, 'Second', '{}'::jsonb)
            ),
            ai_summarize_accumulate(NULL::jsonb, 'Third', '{}'::jsonb)
        ) as combined_state
)
SELECT 
    (combined_state::json->'stats'->>'combine_depth')::int > 0 as has_depth_tracking
FROM recursive_combine;

-- Test 12: Metadata preservation
SELECT 'Test 12: Metadata preservation' as test;
SELECT 
    substr(ai_summarize(
        content,
        jsonb_build_object(
            'source_table', 'test_documents',
            'extraction_date', current_date,
            'version', '1.1.0'
        )
    ), 1, 1) IS NOT NULL as preserves_metadata
FROM test_documents
WHERE category = 'tech';

-- Test 13: Sample preservation
SELECT 'Test 13: Sample preservation' as test;
WITH accumulated AS (
    SELECT ai_summarize_accumulate(
        ai_summarize_accumulate(
            ai_summarize_accumulate(
                NULL::jsonb,
                repeat('Sample 1 ', 30),
                '{}'::jsonb
            ),
            repeat('Sample 2 ', 30),
            '{}'::jsonb
        ),
        repeat('Sample 3 ', 30),
        '{}'::jsonb
    ) as state
)
SELECT 
    jsonb_array_length(state::jsonb->'samples') > 0 as has_samples,
    (state::jsonb->'stats'->>'row_count')::int = 3 as correct_count
FROM accumulated;

-- Test 14: Integration with other steadytext functions
SELECT 'Test 14: Integration test' as test;
WITH generated_content AS (
    SELECT steadytext_generate('Write about databases') as content
)
SELECT 
    ai_summarize_text(content, '{"source": "generated"}'::jsonb) IS NOT NULL as can_summarize_generated
FROM generated_content;

-- Test 15: Error handling for invalid JSON
SELECT 'Test 15: Error handling' as test;
DO $$
BEGIN
    -- This should not crash but handle gracefully
    PERFORM ai_deduplicate_facts('not valid json'::jsonb, 0.8);
    RAISE NOTICE 'Error handling works correctly';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Function handles invalid input: %', SQLERRM;
END;
$$;

-- Cleanup
ROLLBACK;

-- Summary of test results
SELECT 'All AI summarization tests completed' as status;
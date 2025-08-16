-- Final test of unsafe_mode and remote model support

-- Test 1: Local model with explicit parameters
SELECT 'Test 1: Local model (explicit NULL parameters)' as test;
SELECT steadytext_summarize_text(
    'PostgreSQL is a powerful relational database management system. It supports SQL queries and has many advanced features.',
    '{}'::jsonb,
    NULL::text,  -- model
    false        -- unsafe_mode
) as local_summary;

-- Test 2: OpenAI GPT-4o-mini model  
SELECT 'Test 2: OpenAI GPT-4o-mini' as test;
SELECT steadytext_summarize_text(
    'PostgreSQL is a powerful open-source relational database management system. It provides robust transaction support with ACID compliance, advanced indexing capabilities, and extensive SQL standards compliance. PostgreSQL supports both SQL and NoSQL data models, making it versatile for various application needs.',
    '{}'::jsonb,
    'openai:gpt-4o-mini'::text,
    true
) as openai_summary;

-- Test 3: Using metadata to pass model info (aggregate style)
SELECT 'Test 3: Model via metadata' as test;
SELECT steadytext_summarize_text(
    'Machine learning models can be integrated with PostgreSQL using extensions like pgvector for similarity search and embeddings storage.',
    '{"model": "openai:gpt-4o-mini", "unsafe_mode": true}'::jsonb,
    NULL::text,
    false
) as metadata_summary;

-- Test 4: Error handling - remote model without unsafe_mode
SELECT 'Test 4: Error test - remote without unsafe_mode' as test;
DO $$
BEGIN
    PERFORM steadytext_summarize_text(
        'Test text',
        '{}'::jsonb,
        'openai:gpt-4o-mini'::text,
        false
    );
    RAISE NOTICE 'ERROR: Should have failed but did not';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Expected error caught: %', SQLERRM;
END $$;

-- Test 5: Using st_ alias
SELECT 'Test 5: st_ alias with OpenAI' as test;
SELECT st_summarize_text(
    'Vector databases are becoming increasingly important for AI applications, enabling semantic search and recommendation systems.',
    '{}'::jsonb,
    'openai:gpt-4o-mini'::text,
    true
) as st_openai_summary;

-- Test 6: Aggregate summarization with model in metadata
SELECT 'Test 6: Aggregate with OpenAI' as test;
WITH test_data AS (
    SELECT unnest(ARRAY[
        'PostgreSQL supports JSON data types.',
        'It has full-text search capabilities.',
        'PostgreSQL provides ACID compliance.',
        'It supports multiple programming languages for stored procedures.',
        'PostgreSQL has excellent performance and scalability.'
    ]) AS text_content
)
SELECT steadytext_summarize(
    text_content,
    '{"model": "openai:gpt-4o-mini", "unsafe_mode": true}'::jsonb
) as aggregate_summary
FROM test_data;

-- Test 7: Check API key configuration
SELECT 'Test 7: Environment check' as test;
DO $$
DECLARE
    has_key boolean;
BEGIN
    -- Check if OPENAI_API_KEY environment variable is set
    -- This is just informational
    RAISE NOTICE 'Note: OpenAI API calls require OPENAI_API_KEY environment variable to be set';
    RAISE NOTICE 'If the OpenAI tests fail, please set: export OPENAI_API_KEY=your-key-here';
END $$;
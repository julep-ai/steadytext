-- Test script for unsafe_mode and remote model support in summarization functions

-- Test 1: Basic summarization with local model (default)
SELECT 'Test 1: Local model summarization' as test;
SELECT steadytext_summarize_text(
    'PostgreSQL is a powerful relational database management system. It supports SQL queries and has many advanced features like JSON support, full-text search, and custom data types.',
    '{"max_length": 50}'::jsonb
) as local_summary;

-- Test 2: Fact extraction with local model
SELECT 'Test 2: Local model fact extraction' as test;
SELECT steadytext_extract_facts(
    'PostgreSQL was created by Michael Stonebraker in 1986. It is open source and supports ACID transactions. The latest version includes better performance.',
    5
) as local_facts;

-- Test 3: Summarization with OpenAI model (requires OPENAI_API_KEY)
SELECT 'Test 3: OpenAI model summarization' as test;
SELECT steadytext_summarize_text(
    'PostgreSQL is a powerful relational database management system. It supports SQL queries and has many advanced features like JSON support, full-text search, and custom data types. PostgreSQL provides robust transaction support with ACID compliance, making it suitable for enterprise applications.',
    '{"max_length": 100}'::jsonb,
    'openai:gpt-4o-mini',
    true
) as openai_summary;

-- Test 4: Fact extraction with OpenAI model
SELECT 'Test 4: OpenAI model fact extraction' as test;
SELECT steadytext_extract_facts(
    'PostgreSQL was created by Michael Stonebraker in 1986. It started as the POSTGRES project at UC Berkeley. It is open source and supports ACID transactions. The latest version includes better performance. PostgreSQL supports multiple programming languages for stored procedures.',
    10,
    'openai:gpt-4o-mini',
    true
) as openai_facts;

-- Test 5: Using st_* aliases with OpenAI
SELECT 'Test 5: st_* aliases with OpenAI' as test;
SELECT st_summarize_text(
    'Database systems are critical for modern applications. They provide data persistence, transaction support, and query capabilities.',
    '{"style": "brief"}'::jsonb,
    'openai:gpt-4o-mini',
    true
) as st_openai_summary;

-- Test 6: Using ai_* backward compatibility aliases
SELECT 'Test 6: ai_* aliases with OpenAI' as test;
SELECT ai_extract_facts(
    'Machine learning models can be stored in databases. Vector databases are optimized for similarity search. Graph databases excel at relationship queries.',
    5,
    'openai:gpt-4o-mini',
    true
) as ai_openai_facts;

-- Test 7: Aggregate summarization with metadata containing model info
SELECT 'Test 7: Aggregate with model in metadata' as test;
WITH test_data AS (
    SELECT unnest(ARRAY[
        'PostgreSQL is a relational database.',
        'It supports ACID transactions.',
        'PostgreSQL has JSON support.',
        'It includes full-text search.',
        'PostgreSQL is open source.'
    ]) AS text_content
)
SELECT steadytext_summarize(
    text_content, 
    '{"model": "openai:gpt-4o-mini", "unsafe_mode": true, "topic": "database features"}'::jsonb
) as aggregate_summary
FROM test_data;

-- Test 8: Error handling - remote model without unsafe_mode should fail
SELECT 'Test 8: Error handling - remote model without unsafe_mode' as test;
DO $$
BEGIN
    PERFORM steadytext_summarize_text(
        'Test text',
        '{}'::jsonb,
        'openai:gpt-4o-mini',
        false
    );
    RAISE NOTICE 'ERROR: Should have failed but did not';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Expected error: %', SQLERRM;
END $$;

-- Test 9: Check if OPENAI_API_KEY is set
SELECT 'Test 9: Check API key configuration' as test;
DO $$
DECLARE
    api_key text;
BEGIN
    -- Check if the environment variable is accessible (this depends on your setup)
    -- You may need to set it in postgresql.conf or as a system environment variable
    SELECT current_setting('custom.openai_api_key', true) INTO api_key;
    
    IF api_key IS NULL OR api_key = '' THEN
        RAISE NOTICE 'OPENAI_API_KEY is not set. You need to set it to use OpenAI models.';
        RAISE NOTICE 'Set it in postgresql.conf: custom.openai_api_key = ''your-api-key''';
        RAISE NOTICE 'Or as environment variable: OPENAI_API_KEY=your-api-key';
    ELSE
        RAISE NOTICE 'OPENAI_API_KEY is configured (first 10 chars): %...', substring(api_key, 1, 10);
    END IF;
END $$;
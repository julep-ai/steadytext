-- Test summarization with local model
SELECT 'Test 1: Local model summarization' as test;
SELECT steadytext_summarize_text(
    'PostgreSQL is a powerful relational database management system. It supports SQL queries and has many advanced features.',
    '{}'::jsonb
) as local_summary;

-- Test with OpenAI (will need API key)
SELECT 'Test 2: Check if OpenAI works' as test;
SELECT steadytext_summarize_text(
    'PostgreSQL is amazing for building web applications.',
    '{}'::jsonb,
    'openai:gpt-4o-mini',
    true
) as openai_summary;

-- Test st_ alias
SELECT 'Test 3: st_ alias' as test;
SELECT st_summarize_text(
    'Testing the alias function',
    '{}'::jsonb
) as st_summary;
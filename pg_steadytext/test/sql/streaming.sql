-- streaming.sql - Test streaming generation functions
-- AIDEV-NOTE: Tests for streaming text generation

-- Test basic streaming
SELECT COUNT(*) > 0 AS has_tokens FROM steadytext_generate_stream('Write a poem about databases', 50);

-- Test streaming with thinking mode
SELECT COUNT(*) > 0 AS has_tokens FROM steadytext_generate_stream('Explain recursion', 30, true);

-- Test empty prompt handling
SELECT COUNT(*) AS token_count FROM steadytext_generate_stream('', 10);

-- Test max_tokens validation
SELECT COUNT(*) AS token_count FROM steadytext_generate_stream('Test', 0);

-- Test concatenation of streamed tokens
WITH streamed AS (
    SELECT string_agg(token, '') AS full_text
    FROM steadytext_generate_stream('Hello world', 20) AS token
)
SELECT length(full_text) > 0 AS has_content FROM streamed;
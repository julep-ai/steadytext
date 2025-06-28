-- basic.sql - Basic tests for pg_steadytext extension
-- AIDEV-NOTE: This tests core functionality of the extension

-- Test extension creation
CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;

-- Test version function
SELECT steadytext_version();

-- Test configuration
SELECT steadytext_config_get('daemon_host');
SELECT steadytext_config_set('test_key', 'test_value');
SELECT steadytext_config_get('test_key');

-- Test daemon status (may fail if daemon not running)
SELECT * FROM steadytext_daemon_status();

-- Test text generation
SELECT length(steadytext_generate('Hello world', 10)) > 0 AS has_output;

-- Test embedding generation  
SELECT vector_dims(steadytext_embed('Test text')) = 1024 AS correct_dims;

-- Test cache stats
SELECT * FROM steadytext_cache_stats();

-- Cleanup test config
DELETE FROM steadytext_config WHERE key = 'test_key';
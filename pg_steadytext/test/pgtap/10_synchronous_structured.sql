-- 10_synchronous_structured.sql - pgTAP tests for synchronous structured generation
-- AIDEV-NOTE: Tests for JSON, regex, and choice-based synchronous generation functions

BEGIN;
SELECT plan(45);

-- Test 1: Synchronous JSON generation function exists
SELECT has_function(
    'public',
    'steadytext_generate_json',
    'Function steadytext_generate_json should exist'
);

-- Test 2: JSON generation function returns text
SELECT function_returns(
    'public',
    'steadytext_generate_json',
    ARRAY['text', 'jsonb', 'integer', 'boolean', 'integer'],
    'text',
    'Function steadytext_generate_json should return text'
);

-- Test 3: Basic JSON generation with simple schema
WITH json_result AS (
    SELECT steadytext_generate_json(
        'Create a person',
        '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'::jsonb,
        100,
        true,
        42
    ) AS result
)
SELECT ok(
    result IS NOT NULL AND length(result) > 0,
    'Basic JSON generation should return non-empty result'
) FROM json_result;

-- Test 4: JSON generation produces valid JSON
WITH json_result AS (
    SELECT steadytext_generate_json(
        'Create a simple object',
        '{"type": "object", "properties": {"id": {"type": "integer"}, "name": {"type": "string"}}}'::jsonb,
        50,
        false,
        42
    ) AS result
)
SELECT ok(
    result::jsonb IS NOT NULL,
    'JSON generation should produce valid JSON'
) FROM json_result;

-- Test 5: JSON generation with complex schema
WITH complex_schema AS (
    SELECT '{
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "profile": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            }
        }
    }'::jsonb AS schema
),
json_result AS (
    SELECT steadytext_generate_json(
        'Create a user profile',
        schema,
        200,
        true,
        42
    ) AS result
    FROM complex_schema
)
SELECT ok(
    result IS NOT NULL AND length(result) > 10,
    'Complex JSON schema should work'
) FROM json_result;

-- Test 6: Synchronous regex generation function exists
SELECT has_function(
    'public',
    'steadytext_generate_regex',
    'Function steadytext_generate_regex should exist'
);

-- Test 7: Regex generation function returns text
SELECT function_returns(
    'public',
    'steadytext_generate_regex',
    ARRAY['text', 'text', 'integer', 'boolean', 'integer'],
    'text',
    'Function steadytext_generate_regex should return text'
);

-- Test 8: Basic regex generation - phone number
WITH regex_result AS (
    SELECT steadytext_generate_regex(
        'My phone number is',
        '\d{3}-\d{3}-\d{4}',
        50,
        true,
        42
    ) AS result
)
SELECT ok(
    result ~ '^\d{3}-\d{3}-\d{4}$',
    'Regex generation should match the specified pattern'
) FROM regex_result;

-- Test 9: Regex generation - email pattern
WITH regex_result AS (
    SELECT steadytext_generate_regex(
        'Contact email:',
        '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        100,
        false,
        42
    ) AS result
)
SELECT ok(
    result ~ '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    'Email regex generation should match email pattern'
) FROM regex_result;

-- Test 10: Regex generation - alphanumeric code
WITH regex_result AS (
    SELECT steadytext_generate_regex(
        'Product code:',
        '[A-Z]{3}-\d{4}',
        30,
        true,
        42
    ) AS result
)
SELECT ok(
    result ~ '^[A-Z]{3}-\d{4}$',
    'Alphanumeric code regex should match pattern'
) FROM regex_result;

-- Test 11: Synchronous choice generation function exists
SELECT has_function(
    'public',
    'steadytext_generate_choice',
    'Function steadytext_generate_choice should exist'
);

-- Test 12: Choice generation function returns text
SELECT function_returns(
    'public',
    'steadytext_generate_choice',
    ARRAY['text', 'text[]', 'integer', 'boolean', 'integer'],
    'text',
    'Function steadytext_generate_choice should return text'
);

-- Test 13: Basic choice generation
WITH choice_result AS (
    SELECT steadytext_generate_choice(
        'Is PostgreSQL good?',
        ARRAY['yes', 'no', 'maybe'],
        50,
        true,
        42
    ) AS result
)
SELECT ok(
    result = ANY(ARRAY['yes', 'no', 'maybe']),
    'Choice generation should return one of the provided choices'
) FROM choice_result;

-- Test 14: Choice generation with many options
WITH choice_result AS (
    SELECT steadytext_generate_choice(
        'Pick a color',
        ARRAY['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown'],
        30,
        false,
        42
    ) AS result
)
SELECT ok(
    result = ANY(ARRAY['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']),
    'Choice generation should work with many options'
) FROM choice_result;

-- Test 15: Choice generation with complex choices
WITH choice_result AS (
    SELECT steadytext_generate_choice(
        'Choose a programming language',
        ARRAY['Python for data science', 'JavaScript for web development', 'Rust for systems programming'],
        100,
        true,
        42
    ) AS result
)
SELECT ok(
    result = ANY(ARRAY['Python for data science', 'JavaScript for web development', 'Rust for systems programming']),
    'Choice generation should work with complex choice strings'
) FROM choice_result;

-- Test 16: JSON generation input validation - empty prompt
SELECT throws_ok(
    $$ SELECT steadytext_generate_json('', '{"type": "string"}'::jsonb, 50, true, 42) $$,
    'P0001',
    'Prompt cannot be empty',
    'Empty prompt should raise error for JSON generation'
);

-- Test 17: JSON generation input validation - null schema
SELECT throws_ok(
    $$ SELECT steadytext_generate_json('Test', NULL, 50, true, 42) $$,
    'P0001',
    'Schema cannot be null',
    'Null schema should raise error'
);

-- Test 18: JSON generation input validation - invalid max_tokens
SELECT throws_ok(
    $$ SELECT steadytext_generate_json('Test', '{"type": "string"}'::jsonb, 0, true, 42) $$,
    'P0001',
    'max_tokens must be greater than 0',
    'Zero max_tokens should raise error'
);

-- Test 19: Regex generation input validation - empty pattern
SELECT throws_ok(
    $$ SELECT steadytext_generate_regex('Test', '', 50, true, 42) $$,
    'P0001',
    'Pattern cannot be empty',
    'Empty regex pattern should raise error'
);

-- Test 20: Regex generation input validation - null pattern
SELECT throws_ok(
    $$ SELECT steadytext_generate_regex('Test', NULL, 50, true, 42) $$,
    'P0001',
    'Pattern cannot be null',
    'Null regex pattern should raise error'
);

-- Test 21: Choice generation input validation - empty choices
SELECT throws_ok(
    $$ SELECT steadytext_generate_choice('Test', ARRAY[]::text[], 50, true, 42) $$,
    'P0001',
    'Choices array cannot be empty',
    'Empty choices array should raise error'
);

-- Test 22: Choice generation input validation - single choice
SELECT throws_ok(
    $$ SELECT steadytext_generate_choice('Test', ARRAY['only_one'], 50, true, 42) $$,
    'P0001',
    'Choices array must contain at least 2 options',
    'Single choice should raise error'
);

-- Test 23: Choice generation input validation - null choices
SELECT throws_ok(
    $$ SELECT steadytext_generate_choice('Test', NULL, 50, true, 42) $$,
    'P0001',
    'Choices cannot be null',
    'Null choices should raise error'
);

-- Test 24: JSON generation determinism
WITH json_result1 AS (
    SELECT steadytext_generate_json(
        'Deterministic test',
        '{"type": "object", "properties": {"test": {"type": "string"}}}'::jsonb,
        50,
        true,
        42
    ) AS result
),
json_result2 AS (
    SELECT steadytext_generate_json(
        'Deterministic test',
        '{"type": "object", "properties": {"test": {"type": "string"}}}'::jsonb,
        50,
        true,
        42
    ) AS result
)
SELECT is(
    (SELECT result FROM json_result1),
    (SELECT result FROM json_result2),
    'JSON generation should be deterministic with same seed'
);

-- Test 25: Regex generation determinism
WITH regex_result1 AS (
    SELECT steadytext_generate_regex(
        'Deterministic regex test',
        '[A-Z]{2}\d{2}',
        30,
        true,
        42
    ) AS result
),
regex_result2 AS (
    SELECT steadytext_generate_regex(
        'Deterministic regex test',
        '[A-Z]{2}\d{2}',
        30,
        true,
        42
    ) AS result
)
SELECT is(
    (SELECT result FROM regex_result1),
    (SELECT result FROM regex_result2),
    'Regex generation should be deterministic with same seed'
);

-- Test 26: Choice generation determinism
WITH choice_result1 AS (
    SELECT steadytext_generate_choice(
        'Deterministic choice test',
        ARRAY['option1', 'option2', 'option3'],
        30,
        true,
        42
    ) AS result
),
choice_result2 AS (
    SELECT steadytext_generate_choice(
        'Deterministic choice test',
        ARRAY['option1', 'option2', 'option3'],
        30,
        true,
        42
    ) AS result
)
SELECT is(
    (SELECT result FROM choice_result1),
    (SELECT result FROM choice_result2),
    'Choice generation should be deterministic with same seed'
);

-- Test 27: JSON generation with different seeds
WITH json_seed42 AS (
    SELECT steadytext_generate_json(
        'Seed test',
        '{"type": "object", "properties": {"value": {"type": "integer"}}}'::jsonb,
        50,
        false,
        42
    ) AS result
),
json_seed123 AS (
    SELECT steadytext_generate_json(
        'Seed test',
        '{"type": "object", "properties": {"value": {"type": "integer"}}}'::jsonb,
        50,
        false,
        123
    ) AS result
)
SELECT ok(
    (SELECT result FROM json_seed42) IS NOT NULL AND
    (SELECT result FROM json_seed123) IS NOT NULL,
    'Different seeds should both produce valid results'
);

-- Test 28: Caching behavior - JSON generation
WITH cached_result AS (
    SELECT steadytext_generate_json(
        'Cache test JSON',
        '{"type": "object", "properties": {"cached": {"type": "boolean"}}}'::jsonb,
        50,
        true,  -- use_cache = true
        42
    ) AS result
)
SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_cache WHERE prompt = 'Cache test JSON'),
    'JSON generation should create cache entry when caching enabled'
);

-- Test 29: Caching behavior - regex generation
WITH cached_result AS (
    SELECT steadytext_generate_regex(
        'Cache test regex',
        '\d{4}-\d{2}-\d{2}',
        50,
        true,  -- use_cache = true
        42
    ) AS result
)
SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_cache WHERE prompt = 'Cache test regex'),
    'Regex generation should create cache entry when caching enabled'
);

-- Test 30: Caching behavior - choice generation
WITH cached_result AS (
    SELECT steadytext_generate_choice(
        'Cache test choice',
        ARRAY['cached', 'not_cached'],
        50,
        true,  -- use_cache = true
        42
    ) AS result
)
SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_cache WHERE prompt = 'Cache test choice'),
    'Choice generation should create cache entry when caching enabled'
);

-- Test 31: No caching when disabled
WITH uncached_result AS (
    SELECT steadytext_generate_json(
        'No cache test',
        '{"type": "string"}'::jsonb,
        50,
        false,  -- use_cache = false
        42
    ) AS result
)
SELECT ok(
    NOT EXISTS(SELECT 1 FROM steadytext_cache WHERE prompt = 'No cache test'),
    'No cache entry should be created when caching disabled'
);

-- Test 32: JSON schema with arrays
WITH array_result AS (
    SELECT steadytext_generate_json(
        'Create a list',
        '{"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}}}}'::jsonb,
        100,
        false,
        42
    ) AS result
)
SELECT ok(
    (result::jsonb->'items') IS NOT NULL,
    'JSON generation should handle array schemas'
) FROM array_result;

-- Test 33: JSON schema with required fields
WITH required_result AS (
    SELECT steadytext_generate_json(
        'Create required fields',
        '{"type": "object", "properties": {"name": {"type": "string"}, "id": {"type": "integer"}}, "required": ["name", "id"]}'::jsonb,
        100,
        false,
        42
    ) AS result
)
SELECT ok(
    (result::jsonb->'name') IS NOT NULL AND (result::jsonb->'id') IS NOT NULL,
    'JSON generation should respect required fields'
) FROM required_result;

-- Test 34: Complex regex with quantifiers
WITH complex_regex AS (
    SELECT steadytext_generate_regex(
        'Version number',
        '\d+\.\d+\.\d+',
        50,
        false,
        42
    ) AS result
)
SELECT ok(
    result ~ '^\d+\.\d+\.\d+$',
    'Complex regex with quantifiers should work'
) FROM complex_regex;

-- Test 35: Regex with character classes
WITH char_class_result AS (
    SELECT steadytext_generate_regex(
        'Hex color',
        '#[0-9a-fA-F]{6}',
        50,
        false,
        42
    ) AS result
)
SELECT ok(
    result ~ '^#[0-9a-fA-F]{6}$',
    'Regex with character classes should work'
) FROM char_class_result;

-- Test 36: Choice generation with boolean-like choices
WITH bool_choice AS (
    SELECT steadytext_generate_choice(
        'True or false?',
        ARRAY['true', 'false'],
        30,
        false,
        42
    ) AS result
)
SELECT ok(
    result = ANY(ARRAY['true', 'false']),
    'Boolean-like choices should work'
) FROM bool_choice;

-- Test 37: JSON generation with null values allowed
WITH null_schema AS (
    SELECT steadytext_generate_json(
        'Object with nullable field',
        '{"type": "object", "properties": {"nullable_field": {"type": ["string", "null"]}}}'::jsonb,
        50,
        false,
        42
    ) AS result
)
SELECT ok(
    result IS NOT NULL,
    'JSON schema allowing null should work'
) FROM null_schema;

-- Test 38: Large max_tokens handling
WITH large_tokens AS (
    SELECT steadytext_generate_json(
        'Large response',
        '{"type": "object", "properties": {"description": {"type": "string"}}}'::jsonb,
        1000,
        false,
        42
    ) AS result
)
SELECT ok(
    result IS NOT NULL,
    'Large max_tokens should be handled'
) FROM large_tokens;

-- Test 39: Regex generation with optional groups
WITH optional_regex AS (
    SELECT steadytext_generate_regex(
        'Optional area code',
        '(\d{3}-)??\d{3}-\d{4}',
        50,
        false,
        42
    ) AS result
)
SELECT ok(
    result ~ '^(\d{3}-)??\d{3}-\d{4}$',
    'Regex with optional groups should work'
) FROM optional_regex;

-- Test 40: Choice generation with duplicates
SELECT throws_ok(
    $$ SELECT steadytext_generate_choice('Test', ARRAY['option1', 'option1', 'option2'], 50, true, 42) $$,
    'P0001',
    'Choices array cannot contain duplicates',
    'Duplicate choices should raise error'
);

-- Test 41: JSON generation with invalid JSON schema
SELECT throws_ok(
    $$ SELECT steadytext_generate_json('Test', '{"type": "invalid_type"}'::jsonb, 50, true, 42) $$,
    'P0001',
    'Invalid JSON schema',
    'Invalid JSON schema should raise error'
);

-- Test 42: Unicode handling in all functions
WITH unicode_json AS (
    SELECT steadytext_generate_json(
        'Créer un objet avec émojis 🎉',
        '{"type": "object", "properties": {"message": {"type": "string"}}}'::jsonb,
        100,
        false,
        42
    ) AS result
)
SELECT ok(
    result IS NOT NULL,
    'Unicode should be handled in JSON generation'
) FROM unicode_json;

-- Test 43: Large choice array
WITH large_choices AS (
    SELECT array_agg('choice_' || i) AS choices
    FROM generate_series(1, 100) i
),
large_choice_result AS (
    SELECT steadytext_generate_choice(
        'Pick from many',
        choices,
        50,
        false,
        42
    ) AS result
    FROM large_choices
)
SELECT ok(
    result LIKE 'choice_%',
    'Large choice array should work'
) FROM large_choice_result;

-- Test 44: Negative seed handling
WITH negative_seed AS (
    SELECT steadytext_generate_json(
        'Negative seed test',
        '{"type": "string"}'::jsonb,
        50,
        false,
        -42
    ) AS result
)
SELECT ok(
    result IS NOT NULL,
    'Negative seed should be handled'
) FROM negative_seed;

-- Test 45: Schema validation with nested objects
WITH nested_schema AS (
    SELECT '{
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": {
                    "database": {
                        "type": "object",
                        "properties": {
                            "host": {"type": "string"},
                            "port": {"type": "integer"}
                        },
                        "required": ["host", "port"]
                    }
                }
            }
        }
    }'::jsonb AS schema
),
nested_result AS (
    SELECT steadytext_generate_json(
        'Create database config',
        schema,
        200,
        false,
        42
    ) AS result
    FROM nested_schema
)
SELECT ok(
    (result::jsonb->'config'->'database'->'host') IS NOT NULL AND
    (result::jsonb->'config'->'database'->'port') IS NOT NULL,
    'Nested object schema should work'
) FROM nested_result;

-- Clean up test cache entries
DELETE FROM steadytext_cache WHERE prompt LIKE '%test%' OR prompt LIKE '%Cache%';

SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: Synchronous structured generation tests comprehensively cover:
-- - JSON generation with various schema types
-- - Regex generation with different pattern complexities
-- - Choice generation with various option sets
-- - Input validation and error handling
-- - Deterministic behavior with seeds
-- - Caching behavior (enabled/disabled)
-- - Unicode and special character handling
-- - Performance with large inputs
-- - Edge cases and boundary conditions
-- - Schema validation and complex nested structures
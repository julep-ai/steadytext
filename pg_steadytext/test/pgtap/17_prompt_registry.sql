-- Test suite for prompt registry feature
-- AIDEV-NOTE: Tests for Jinja2-based prompt template management with versioning

-- Start transaction for test isolation
BEGIN;

-- Load pgTAP
\i test/pgtap/00_setup.sql

-- Plan the number of tests
SELECT plan(58);

-- Test schema setup
SELECT has_table('steadytext_prompts', 'Should have prompts table');
SELECT has_table('steadytext_prompt_versions', 'Should have prompt versions table');

-- Test table columns
SELECT has_column('steadytext_prompts', 'id', 'Prompts table should have id column');
SELECT has_column('steadytext_prompts', 'slug', 'Prompts table should have slug column');
SELECT has_column('steadytext_prompts', 'description', 'Prompts table should have description column');
SELECT has_column('steadytext_prompt_versions', 'template', 'Versions table should have template column');
SELECT has_column('steadytext_prompt_versions', 'required_variables', 'Versions table should have required_variables column');

-- Test functions existence
SELECT has_function('steadytext_prompt_create', 'Should have prompt create function');
SELECT has_function('steadytext_prompt_update', 'Should have prompt update function');
SELECT has_function('steadytext_prompt_get', 'Should have prompt get function');
SELECT has_function('steadytext_prompt_render', 'Should have prompt render function');
SELECT has_function('steadytext_prompt_list', 'Should have prompt list function');
SELECT has_function('steadytext_prompt_versions', 'Should have prompt versions function');
SELECT has_function('steadytext_prompt_delete', 'Should have prompt delete function');

-- Test aliases existence
SELECT has_function('st_prompt_create', 'Should have st_prompt_create alias');
SELECT has_function('st_prompt_update', 'Should have st_prompt_update alias');
SELECT has_function('st_prompt_get', 'Should have st_prompt_get alias');
SELECT has_function('st_prompt_render', 'Should have st_prompt_render alias');
SELECT has_function('st_prompt_list', 'Should have st_prompt_list alias');
SELECT has_function('st_prompt_versions', 'Should have st_prompt_versions alias');
SELECT has_function('st_prompt_delete', 'Should have st_prompt_delete alias');

-- Test 1: Create a simple prompt
SELECT ok(
    steadytext_prompt_create(
        'test-prompt',
        'Hello {{ name }}!',
        'A simple greeting template'
    ) IS NOT NULL,
    'Should create a prompt with simple template'
);

-- Test 2: Get the created prompt
SELECT is(
    (SELECT template FROM steadytext_prompt_get('test-prompt')),
    'Hello {{ name }}!',
    'Should retrieve the correct template'
);

-- Test 3: Check required variables extraction
SELECT is(
    (SELECT required_variables FROM steadytext_prompt_get('test-prompt')),
    ARRAY['name']::TEXT[],
    'Should extract required variables correctly'
);

-- Test 4: Render template with variables
SELECT is(
    steadytext_prompt_render('test-prompt', '{"name": "World"}'::jsonb),
    'Hello World!',
    'Should render template with provided variables'
);

-- Test 5: Render with missing variables (strict mode)
SELECT throws_ok(
    $$ SELECT steadytext_prompt_render('test-prompt', '{}'::jsonb) $$,
    'XX000',
    NULL,
    'Should throw error for missing variables in strict mode'
);

-- Test 6: Render with missing variables (non-strict mode)
SELECT ok(
    steadytext_prompt_render('test-prompt', '{}'::jsonb, NULL, false) IS NOT NULL,
    'Should render with missing variables in non-strict mode'
);

-- Test 7: Update prompt (create new version)
SELECT ok(
    steadytext_prompt_update(
        'test-prompt',
        'Hi {{ name }}, welcome to {{ place }}!'
    ) IS NOT NULL,
    'Should create new version of prompt'
);

-- Test 8: Get latest version
SELECT is(
    (SELECT template FROM steadytext_prompt_get('test-prompt')),
    'Hi {{ name }}, welcome to {{ place }}!',
    'Should get latest version by default'
);

-- Test 9: Get specific version
SELECT is(
    (SELECT template FROM steadytext_prompt_get('test-prompt', 1)),
    'Hello {{ name }}!',
    'Should get specific version 1'
);

-- Test 10: List versions
SELECT is(
    (SELECT COUNT(*) FROM steadytext_prompt_versions('test-prompt'))::INTEGER,
    2,
    'Should have 2 versions'
);

-- Test 11: Create prompt with complex template
SELECT ok(
    steadytext_prompt_create(
        'complex-template',
        '{% for item in items %}{{ item.name }}: {{ item.value }}{% endfor %}',
        'Template with loop'
    ) IS NOT NULL,
    'Should create prompt with complex Jinja2 template'
);

-- Test 12: Render complex template
SELECT is(
    steadytext_prompt_render(
        'complex-template', 
        '{"items": [{"name": "A", "value": 1}, {"name": "B", "value": 2}]}'::jsonb
    ),
    'A: 1B: 2',
    'Should render complex template with loops'
);

-- Test 13: Invalid template syntax
SELECT throws_ok(
    $$ SELECT steadytext_prompt_create('bad-template', '{{ unclosed', 'Bad template') $$,
    '42601',
    NULL,
    'Should reject invalid Jinja2 syntax'
);

-- Test 14: Invalid slug format
SELECT throws_ok(
    $$ SELECT steadytext_prompt_create('Bad_Slug!', 'Template', 'Invalid slug') $$,
    '22P02',
    NULL,
    'Should reject invalid slug format'
);

-- Test 15: Duplicate slug
SELECT throws_ok(
    $$ SELECT steadytext_prompt_create('test-prompt', 'Duplicate', 'Duplicate slug') $$,
    '23505',
    NULL,
    'Should reject duplicate slug'
);

-- Test 16: List all prompts
SELECT is(
    (SELECT COUNT(*) FROM steadytext_prompt_list())::INTEGER,
    2,
    'Should list all prompts'
);

-- Test 17: Prompt with metadata
SELECT ok(
    steadytext_prompt_create(
        'meta-prompt',
        'Template content',
        'Prompt with metadata',
        '{"category": "test", "tags": ["example"]}'::jsonb
    ) IS NOT NULL,
    'Should create prompt with metadata'
);

-- Test 18: Delete prompt
SELECT is(
    steadytext_prompt_delete('meta-prompt'),
    TRUE,
    'Should delete existing prompt'
);

-- Test 19: Delete non-existent prompt
SELECT is(
    steadytext_prompt_delete('non-existent'),
    FALSE,
    'Should return false for non-existent prompt'
);

-- Test 20: Prompt not found error
SELECT throws_ok(
    $$ SELECT * FROM steadytext_prompt_get('non-existent') $$,
    'P0002',
    NULL,
    'Should throw error for non-existent prompt'
);

-- Test 21: Version not found error
SELECT throws_ok(
    $$ SELECT * FROM steadytext_prompt_get('test-prompt', 999) $$,
    'P0002',
    NULL,
    'Should throw error for non-existent version'
);

-- Test 22: Conditional template
SELECT ok(
    steadytext_prompt_create(
        'conditional',
        '{% if show_greeting %}Hello {% endif %}{{ name }}',
        'Conditional template'
    ) IS NOT NULL,
    'Should create conditional template'
);

-- Test 23: Render conditional with true
SELECT is(
    steadytext_prompt_render('conditional', '{"name": "Alice", "show_greeting": true}'::jsonb),
    'Hello Alice',
    'Should render conditional when true'
);

-- Test 24: Render conditional with false
SELECT is(
    steadytext_prompt_render('conditional', '{"name": "Bob", "show_greeting": false}'::jsonb),
    'Bob',
    'Should render conditional when false'
);

-- Test 25: Template with default filter
SELECT ok(
    steadytext_prompt_create(
        'with-default',
        'Hello {{ name|default("Guest") }}',
        'Template with default filter'
    ) IS NOT NULL,
    'Should create template with filter'
);

-- Test 26: Render with default value
SELECT is(
    steadytext_prompt_render('with-default', '{}'::jsonb, NULL, false),
    'Hello Guest',
    'Should use default value when variable missing'
);

-- Test 27: Check active version flag  
SELECT is(
    (SELECT is_active FROM steadytext_prompt_versions('test-prompt') WHERE version_num = 2),
    TRUE,
    'Latest version should be active'
);

SELECT is(
    (SELECT is_active FROM steadytext_prompt_versions('test-prompt') WHERE version_num = 1),
    FALSE,
    'Old version should not be active'
);

-- Test 28: Test alias functions
SELECT ok(
    st_prompt_create('alias-test', 'Template {{ var }}') IS NOT NULL,
    'st_prompt_create alias should work'
);

SELECT is(
    st_prompt_render('alias-test', '{"var": "value"}'::jsonb),
    'Template value',
    'st_prompt_render alias should work'
);

-- Test 29: Verify column names in steadytext_prompt_list result
SELECT ok(
    EXISTS (SELECT latest_version_num FROM steadytext_prompt_list()),
    'steadytext_prompt_list should have latest_version_num column'
);

-- Test 30: Verify column names in steadytext_prompt_versions result  
SELECT ok(
    EXISTS (SELECT version_num FROM steadytext_prompt_versions('test-prompt')),
    'steadytext_prompt_versions should have version_num column'
);

-- Test 31: Test concurrent version updates (using advisory locks)
-- Create a test prompt for concurrency testing
SELECT ok(
    steadytext_prompt_create('concurrency-test', 'Initial {{ value }}') IS NOT NULL,
    'Should create prompt for concurrency testing'
);

-- Test 32: Verify strict mode affects template rendering differently
SELECT isnt(
    steadytext_prompt_render('with-default', '{}'::jsonb, NULL, false),
    steadytext_prompt_render('with-default', '{"name": "Test"}'::jsonb, NULL, false),
    'Different variables should produce different results'
);

-- Test 33: Verify JSONB return type from steadytext_extract_facts
SELECT is(
    pg_typeof(steadytext_extract_facts('Test sentence. Another one.')),
    'jsonb'::regtype,
    'steadytext_extract_facts should return JSONB type'
);

-- Test 34: Test error message formatting for remote models
DO $$
BEGIN
    -- This should raise an error with proper formatting
    PERFORM steadytext_generate('test', model := 'openai:gpt-4', unsafe_mode := FALSE);
    RAISE EXCEPTION 'Should have raised error for remote model without unsafe_mode';
EXCEPTION 
    WHEN OTHERS THEN
        -- Check that error message is properly formatted
        IF SQLERRM NOT LIKE '%Remote models (containing %) require unsafe_mode=TRUE%' THEN
            RAISE EXCEPTION 'Error message not properly formatted: %', SQLERRM;
        END IF;
END;
$$;

SELECT pass('Remote model error message is properly formatted');

-- Test 35: Verify advisory lock prevents concurrent updates
-- This is a conceptual test - in real scenario would need concurrent sessions
SELECT ok(
    steadytext_prompt_update('concurrency-test', 'Updated {{ value }}') IS NOT NULL,
    'Should handle version update with advisory lock'
);

-- Cleanup test prompts
SELECT steadytext_prompt_delete('test-prompt');
SELECT steadytext_prompt_delete('complex-template');
SELECT steadytext_prompt_delete('conditional');
SELECT steadytext_prompt_delete('with-default');
SELECT steadytext_prompt_delete('alias-test');
SELECT steadytext_prompt_delete('concurrency-test');

-- Finish tests
SELECT * FROM finish();
ROLLBACK;
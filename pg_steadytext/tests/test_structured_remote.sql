-- Test structured generation functions with remote models
-- AIDEV-NOTE: Tests for v1.4.5 remote model support in structured functions

-- Start transaction
BEGIN;

-- Test that structured functions require model parameter when unsafe_mode is true
DO $$
BEGIN
    -- st_generate_json should fail without model when unsafe_mode=true
    BEGIN
        PERFORM st_generate_json('Create person', '{"type": "object"}'::jsonb, unsafe_mode => true);
        RAISE EXCEPTION 'Expected error for unsafe_mode without model';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM NOT LIKE '%unsafe_mode=TRUE requires a model parameter%' THEN
                RAISE EXCEPTION 'Unexpected error: %', SQLERRM;
            END IF;
    END;

    -- st_generate_regex should fail without model when unsafe_mode=true
    BEGIN
        PERFORM st_generate_regex('Phone', '\d{3}-\d{3}-\d{4}', unsafe_mode => true);
        RAISE EXCEPTION 'Expected error for unsafe_mode without model';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM NOT LIKE '%unsafe_mode=TRUE requires a model parameter%' THEN
                RAISE EXCEPTION 'Unexpected error: %', SQLERRM;
            END IF;
    END;

    -- st_generate_choice should fail without model when unsafe_mode=true
    BEGIN
        PERFORM st_generate_choice('Pick', ARRAY['yes', 'no'], unsafe_mode => true);
        RAISE EXCEPTION 'Expected error for unsafe_mode without model';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM NOT LIKE '%unsafe_mode=TRUE requires a model parameter%' THEN
                RAISE EXCEPTION 'Unexpected error: %', SQLERRM;
            END IF;
    END;
END $$;

-- Test that remote models require unsafe_mode=true
DO $$
BEGIN
    -- st_generate_json should fail with remote model without unsafe_mode
    BEGIN
        PERFORM st_generate_json('Create person', '{"type": "object"}'::jsonb, model => 'openai:gpt-4o-mini');
        RAISE EXCEPTION 'Expected error for remote model without unsafe_mode';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM NOT LIKE '%Remote models%require unsafe_mode=TRUE%' THEN
                RAISE EXCEPTION 'Unexpected error: %', SQLERRM;
            END IF;
    END;

    -- st_generate_regex should fail with remote model without unsafe_mode
    BEGIN
        PERFORM st_generate_regex('Phone', '\d{3}-\d{3}-\d{4}', model => 'openai:gpt-4o-mini');
        RAISE EXCEPTION 'Expected error for remote model without unsafe_mode';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM NOT LIKE '%Remote models%require unsafe_mode=TRUE%' THEN
                RAISE EXCEPTION 'Unexpected error: %', SQLERRM;
            END IF;
    END;

    -- st_generate_choice should fail with remote model without unsafe_mode
    BEGIN
        PERFORM st_generate_choice('Pick', ARRAY['yes', 'no'], model => 'openai:gpt-4o-mini');
        RAISE EXCEPTION 'Expected error for remote model without unsafe_mode';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM NOT LIKE '%Remote models%require unsafe_mode=TRUE%' THEN
                RAISE EXCEPTION 'Unexpected error: %', SQLERRM;
            END IF;
    END;
END $$;

-- Test that structured functions work with local models (backward compatibility)
SELECT length(st_generate_json('Test', '{"type": "string"}'::jsonb)) > 0 AS json_works;
SELECT length(st_generate_regex('Test', '[a-z]+')) > 0 AS regex_works;
SELECT st_generate_choice('Test', ARRAY['a', 'b', 'c']) IN ('a', 'b', 'c') AS choice_works;

-- Test that function signatures accept all parameters
SELECT 
    proname,
    pronargs,
    proargtypes::regtype[]
FROM pg_proc 
WHERE proname IN ('st_generate_json', 'st_generate_regex', 'st_generate_choice')
ORDER BY proname;

-- Verify new parameters are in the correct position
DO $$
DECLARE
    func_info record;
BEGIN
    -- Check st_generate_json has 7 parameters
    SELECT pronargs INTO func_info
    FROM pg_proc 
    WHERE proname = 'st_generate_json' 
    AND pronargs = 7;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'st_generate_json should have 7 parameters (including new model parameter)';
    END IF;

    -- Check st_generate_regex has 7 parameters
    SELECT pronargs INTO func_info
    FROM pg_proc 
    WHERE proname = 'st_generate_regex' 
    AND pronargs = 7;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'st_generate_regex should have 7 parameters (including new model parameter)';
    END IF;

    -- Check st_generate_choice has 7 parameters
    SELECT pronargs INTO func_info
    FROM pg_proc 
    WHERE proname = 'st_generate_choice' 
    AND pronargs = 7;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'st_generate_choice should have 7 parameters (including new model parameter)';
    END IF;
END $$;

-- Test comments are updated
SELECT 
    p.proname,
    d.description
FROM pg_proc p
JOIN pg_description d ON d.objoid = p.oid
WHERE p.proname IN ('steadytext_generate_json', 'steadytext_generate_regex', 'steadytext_generate_choice')
AND d.description LIKE '%remote models%'
ORDER BY p.proname;

-- Rollback (tests only)
ROLLBACK;
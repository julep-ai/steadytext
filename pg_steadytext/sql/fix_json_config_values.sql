-- Migration script to fix JSON parsing errors in existing pg_steadytext installations
-- This fixes the "invalid input syntax for type json" error by ensuring string values
-- in the config table are properly JSON-encoded

-- AIDEV-NOTE: This script fixes existing installations where string config values
-- were stored without JSON encoding (e.g., 'localhost' instead of '"localhost"')

BEGIN;

-- Fix daemon_host if it's not properly quoted
UPDATE steadytext_config 
SET value = '"' || value || '"'
WHERE key = 'daemon_host'
AND value NOT LIKE '"%"'
AND value NOT LIKE '{%}'  -- Don't quote if it's already a JSON object
AND value NOT LIKE '[%]'; -- Don't quote if it's already a JSON array

-- Fix cron_host if it exists and is not properly quoted
UPDATE steadytext_config 
SET value = '"' || value || '"'
WHERE key = 'cron_host'
AND value NOT LIKE '"%"'
AND value NOT LIKE '{%}'  -- Don't quote if it's already a JSON object
AND value NOT LIKE '[%]'; -- Don't quote if it's already a JSON array

-- Log what was fixed
DO $$
DECLARE
    v_fixed_count INT;
BEGIN
    SELECT COUNT(*) INTO v_fixed_count
    FROM steadytext_config
    WHERE key IN ('daemon_host', 'cron_host')
    AND value LIKE '"%"';
    
    IF v_fixed_count > 0 THEN
        RAISE NOTICE 'Fixed % config values to use proper JSON encoding', v_fixed_count;
    ELSE
        RAISE NOTICE 'No config values needed fixing';
    END IF;
END
$$;

-- Verify the fix by attempting to parse the values
DO $$
DECLARE
    v_host TEXT;
    v_config RECORD;
BEGIN
    FOR v_config IN 
        SELECT key, value 
        FROM steadytext_config 
        WHERE key IN ('daemon_host', 'cron_host')
    LOOP
        BEGIN
            -- This will throw an error if the value is not valid JSON
            v_host := v_config.value::json#>>'{}';
            RAISE NOTICE '✓ % is valid JSON: %', v_config.key, v_host;
        EXCEPTION WHEN OTHERS THEN
            RAISE EXCEPTION 'Failed to parse % as JSON: %', v_config.key, v_config.value;
        END;
    END LOOP;
END
$$;

COMMIT;

-- Usage:
-- psql -d your_database -f fix_json_config_values.sql
-- Or run directly in psql:
-- \i fix_json_config_values.sql
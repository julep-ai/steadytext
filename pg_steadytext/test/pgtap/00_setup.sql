-- 00_setup.sql - pgTAP setup and verification
-- AIDEV-NOTE: This ensures pgTAP is properly installed before running tests

BEGIN;
SELECT plan(3);

-- Test 1: pgTAP extension is installed
SELECT has_extension('pgtap', 'pgTAP extension should be installed');

-- Test 2: pgTAP version is available
SELECT ok(
    pgtap_version() IS NOT NULL,
    'pgTAP version should be accessible'
);

-- Test 3: pg_steadytext extension is installed
SELECT has_extension('pg_steadytext', 'pg_steadytext extension should be installed');

SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: This is the first test that runs to ensure the test environment
-- is properly set up. If this fails, other tests won't work correctly.
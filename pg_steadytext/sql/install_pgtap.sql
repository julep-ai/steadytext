-- install_pgtap.sql - Install pgTAP extension for testing
-- AIDEV-NOTE: This script installs pgTAP which is PostgreSQL's testing framework

-- Create pgTAP extension if not exists
CREATE EXTENSION IF NOT EXISTS pgtap;

-- Verify pgTAP is installed correctly
SELECT pgtap_version();

-- Create a test schema for pgTAP tests
CREATE SCHEMA IF NOT EXISTS tap_tests;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA tap_tests TO PUBLIC;
GRANT CREATE ON SCHEMA tap_tests TO PUBLIC;

-- Set search path to include tap_tests
ALTER DATABASE current_database() SET search_path TO public, tap_tests;

-- AIDEV-NOTE: pgTAP provides TAP (Test Anything Protocol) output for PostgreSQL
-- Key functions include:
-- - plan(n) - declare how many tests
-- - ok(boolean, description) - basic assertion
-- - is(actual, expected, description) - equality assertion
-- - has_function(name) - check function exists
-- - function_returns(name, args, type) - check return type
-- - throws_ok() - test for expected errors
-- - finish() - complete the test suite
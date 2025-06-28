-- uninstall_pg_steadytext.sql
-- Script to cleanly uninstall pg_steadytext extension

-- AIDEV-NOTE: This script removes all pg_steadytext objects
-- It should be run before DROP EXTENSION to ensure clean removal

-- Drop all functions
DROP FUNCTION IF EXISTS steadytext_generate(TEXT, INT, BOOLEAN, BOOLEAN);
DROP FUNCTION IF EXISTS steadytext_embed(TEXT, BOOLEAN);
DROP FUNCTION IF EXISTS steadytext_daemon_start();
DROP FUNCTION IF EXISTS steadytext_daemon_status();
DROP FUNCTION IF EXISTS steadytext_daemon_stop();
DROP FUNCTION IF EXISTS steadytext_cache_stats();
DROP FUNCTION IF EXISTS steadytext_cache_clear();
DROP FUNCTION IF EXISTS steadytext_version();
DROP FUNCTION IF EXISTS steadytext_config_set(TEXT, TEXT);
DROP FUNCTION IF EXISTS steadytext_config_get(TEXT);
DROP FUNCTION IF EXISTS _steadytext_init_python();

-- Drop all tables
DROP TABLE IF EXISTS steadytext_cache CASCADE;
DROP TABLE IF EXISTS steadytext_queue CASCADE;
DROP TABLE IF EXISTS steadytext_config CASCADE;
DROP TABLE IF EXISTS steadytext_daemon_health CASCADE;

-- Reset Python path (optional, database-specific)
-- ALTER DATABASE current_database() RESET plpython3.python_path;
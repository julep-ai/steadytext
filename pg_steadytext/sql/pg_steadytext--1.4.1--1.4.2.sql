-- pg_steadytext--1.4.1--1.4.2.sql
-- Migration from version 1.4.1 to 1.4.2
-- 
-- AIDEV-NOTE: Version 1.4.2 fixes AttributeError by adding public start_daemon() method
-- to SteadyTextConnector class in daemon_connector.py
--
-- No SQL changes required - this is a Python module fix only

-- Update version in config table
UPDATE steadytext_config 
SET value = '"1.4.2"' 
WHERE key = 'extension_version';

-- Add version history entry
INSERT INTO steadytext_version_history (version, upgraded_at, notes)
VALUES ('1.4.2', NOW(), 'Fixed AttributeError: Added public start_daemon() method to SteadyTextConnector');

-- AIDEV-NOTE: The fix was implemented in daemon_connector.py by adding a public
-- start_daemon() method that wraps the existing private _start_daemon() method.
-- This maintains compatibility with SQL files that call connector.start_daemon()
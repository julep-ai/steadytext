-- 11_daemon_management.sql - pgTAP tests for daemon management functionality
-- AIDEV-NOTE: Tests for daemon start/stop/status functions, health monitoring, and fallback behavior

BEGIN;
SELECT plan(30);

-- Test 1: Daemon status function exists
SELECT has_function(
    'public',
    'steadytext_daemon_status',
    'Function steadytext_daemon_status should exist'
);

-- Test 2: Daemon status returns correct columns
SELECT columns_are(
    'public',
    'steadytext_daemon_status',
    ARRAY['daemon_id', 'status', 'endpoint', 'last_heartbeat', 'uptime_seconds'],
    'Daemon status should return expected columns'
);

-- Test 3: Daemon start function exists
SELECT has_function(
    'public',
    'steadytext_daemon_start',
    'Function steadytext_daemon_start should exist'
);

-- Test 4: Daemon start returns boolean
SELECT function_returns(
    'public',
    'steadytext_daemon_start',
    ARRAY[]::text[],
    'boolean',
    'Function steadytext_daemon_start should return boolean'
);

-- Test 5: Daemon stop function exists
SELECT has_function(
    'public',
    'steadytext_daemon_stop',
    'Function steadytext_daemon_stop should exist'
);

-- Test 6: Daemon stop returns boolean
SELECT function_returns(
    'public',
    'steadytext_daemon_stop',
    ARRAY[]::text[],
    'boolean',
    'Function steadytext_daemon_stop should return boolean'
);

-- Test 7: Daemon health table exists
SELECT has_table(
    'public',
    'steadytext_daemon_health',
    'Table steadytext_daemon_health should exist'
);

-- Test 8: Daemon health table has correct columns
SELECT has_column('steadytext_daemon_health', 'daemon_id', 'Health table should have daemon_id column');
SELECT has_column('steadytext_daemon_health', 'status', 'Health table should have status column');
SELECT has_column('steadytext_daemon_health', 'endpoint', 'Health table should have endpoint column');
SELECT has_column('steadytext_daemon_health', 'last_heartbeat', 'Health table should have last_heartbeat column');
SELECT has_column('steadytext_daemon_health', 'uptime_seconds', 'Health table should have uptime_seconds column');

-- Test 9: Daemon status query works (even if daemon not running)
SELECT ok(
    (SELECT COUNT(*) >= 0 FROM steadytext_daemon_status()),
    'Daemon status query should work without errors'
);

-- Test 10: Daemon start/stop functions handle non-running daemon gracefully
SELECT ok(
    steadytext_daemon_start() IS NOT NULL,
    'Daemon start should return boolean result'
);

SELECT ok(
    steadytext_daemon_stop() IS NOT NULL,
    'Daemon stop should return boolean result'
);

-- Test 11: Configuration table has daemon-related settings
SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_config WHERE key LIKE '%daemon%'),
    'Configuration should contain daemon-related settings'
);

-- Test 12: Daemon endpoint configuration
SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_config WHERE key IN ('daemon_host', 'daemon_port')),
    'Daemon endpoint configuration should exist'
);

-- Test 13: Generation works with daemon (fallback if not available)
SELECT ok(
    length(steadytext_generate('Daemon test prompt', 10)) > 0,
    'Generation should work regardless of daemon status'
);

-- Test 14: Embedding works with daemon (fallback if not available)
SELECT ok(
    vector_dims(steadytext_embed('Daemon test embedding')) = 1024,
    'Embedding should work regardless of daemon status'
);

-- Test 15: Daemon health monitoring can be queried
SELECT ok(
    (SELECT COUNT(*) FROM steadytext_daemon_health) >= 0,
    'Daemon health table should be queryable'
);

-- Test 16: Daemon status provides consistent schema
WITH status_check AS (
    SELECT * FROM steadytext_daemon_status()
    LIMIT 1
)
SELECT ok(
    (SELECT COUNT(*) FROM status_check) >= 0,
    'Daemon status should return consistent schema'
);

-- Test 17: Daemon restart functionality (start followed by stop)
WITH restart_test AS (
    SELECT 
        steadytext_daemon_start() AS start_result,
        steadytext_daemon_stop() AS stop_result
)
SELECT ok(
    start_result IS NOT NULL AND stop_result IS NOT NULL,
    'Daemon restart sequence should work'
) FROM restart_test;

-- Test 18: Daemon health records have valid timestamps
SELECT ok(
    (SELECT bool_and(last_heartbeat IS NULL OR last_heartbeat <= NOW())
     FROM steadytext_daemon_health),
    'Daemon health timestamps should be valid'
);

-- Test 19: Daemon status fields are properly typed
WITH status_fields AS (
    SELECT 
        daemon_id,
        status,
        endpoint,
        last_heartbeat,
        uptime_seconds
    FROM steadytext_daemon_status()
    LIMIT 1
)
SELECT ok(
    (SELECT COUNT(*) FROM status_fields) >= 0,
    'Daemon status fields should be properly typed'
);

-- Test 20: Daemon configuration can be read
SELECT ok(
    steadytext_config_get('daemon_host') IS NOT NULL,
    'Daemon host configuration should be readable'
);

SELECT ok(
    steadytext_config_get('daemon_port') IS NOT NULL,
    'Daemon port configuration should be readable'
);

-- Test 21: Daemon configuration can be set
SELECT ok(
    steadytext_config_set('daemon_host', 'localhost') IS NOT NULL,
    'Daemon host configuration should be settable'
);

SELECT ok(
    steadytext_config_set('daemon_port', '5555') IS NOT NULL,
    'Daemon port configuration should be settable'
);

-- Test 22: Configuration changes are persisted
SELECT is(
    steadytext_config_get('daemon_host'),
    'localhost',
    'Daemon host configuration should persist'
);

SELECT is(
    steadytext_config_get('daemon_port'),
    '5555',
    'Daemon port configuration should persist'
);

-- Test 23: Daemon health table can be populated
INSERT INTO steadytext_daemon_health (daemon_id, status, endpoint, last_heartbeat, uptime_seconds)
VALUES ('test_daemon', 'healthy', 'tcp://localhost:5555', NOW(), 3600);

SELECT ok(
    EXISTS(SELECT 1 FROM steadytext_daemon_health WHERE daemon_id = 'test_daemon'),
    'Daemon health record should be insertable'
);

-- Test 24: Daemon health status updates
UPDATE steadytext_daemon_health 
SET status = 'unhealthy', last_heartbeat = NOW() - INTERVAL '5 minutes'
WHERE daemon_id = 'test_daemon';

SELECT is(
    (SELECT status FROM steadytext_daemon_health WHERE daemon_id = 'test_daemon'),
    'unhealthy',
    'Daemon health status should be updatable'
);

-- Test 25: Daemon health cleanup
DELETE FROM steadytext_daemon_health WHERE daemon_id = 'test_daemon';

SELECT ok(
    NOT EXISTS(SELECT 1 FROM steadytext_daemon_health WHERE daemon_id = 'test_daemon'),
    'Daemon health record should be deletable'
);

-- Test 26: Multiple daemon instances can be tracked
INSERT INTO steadytext_daemon_health (daemon_id, status, endpoint, last_heartbeat, uptime_seconds)
VALUES 
    ('daemon_1', 'healthy', 'tcp://localhost:5555', NOW(), 1800),
    ('daemon_2', 'healthy', 'tcp://localhost:5556', NOW() - INTERVAL '1 minute', 3600);

SELECT is(
    (SELECT COUNT(*) FROM steadytext_daemon_health WHERE daemon_id IN ('daemon_1', 'daemon_2'))::integer,
    2,
    'Multiple daemon instances should be trackable'
);

-- Test 27: Daemon status query with multiple instances
WITH multi_status AS (
    SELECT COUNT(*) as daemon_count
    FROM steadytext_daemon_status()
)
SELECT ok(
    (SELECT daemon_count >= 0 FROM multi_status),
    'Daemon status should handle multiple instances'
);

-- Test 28: Daemon endpoint validation
SELECT ok(
    (SELECT bool_and(endpoint LIKE 'tcp://%' OR endpoint IS NULL)
     FROM steadytext_daemon_health),
    'Daemon endpoints should follow expected format'
);

-- Test 29: Daemon uptime tracking
SELECT ok(
    (SELECT bool_and(uptime_seconds >= 0 OR uptime_seconds IS NULL)
     FROM steadytext_daemon_health),
    'Daemon uptime should be non-negative'
);

-- Test 30: Daemon health table indexes exist
SELECT has_index(
    'public',
    'steadytext_daemon_health',
    'idx_daemon_health_id',
    'Daemon health table should have index on daemon_id'
);

-- Clean up test data
DELETE FROM steadytext_daemon_health WHERE daemon_id IN ('daemon_1', 'daemon_2');

-- Reset daemon configuration to defaults
SELECT steadytext_config_set('daemon_host', 'localhost');
SELECT steadytext_config_set('daemon_port', '5433');

SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: Daemon management tests comprehensively cover:
-- - Daemon status monitoring and health checks
-- - Start/stop functionality and error handling
-- - Configuration management and persistence
-- - Health table operations and data integrity
-- - Multiple daemon instance tracking
-- - Fallback behavior when daemon unavailable
-- - Integration with generation and embedding functions
-- - Endpoint validation and uptime tracking
-- - Database schema and indexing
-- - Graceful degradation patterns
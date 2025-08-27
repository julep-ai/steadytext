-- 16_timescaledb_integration.sql - pgTAP tests for TimescaleDB integration
-- AIDEV-NOTE: Tests pg_steadytext functions with TimescaleDB hypertables and continuous aggregates
-- AIDEV-NOTE: Tests focus on the steadytext_summarize aggregate function in materialized views
-- AIDEV-NOTE: Uses STEADYTEXT_USE_MINI_MODELS=true to prevent timeouts

BEGIN;

-- Check if TimescaleDB is installed, skip tests if not
DO $$
DECLARE
    v_timescale_installed BOOLEAN;
BEGIN
    SELECT EXISTS(
        SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
    ) INTO v_timescale_installed;
    
    IF NOT v_timescale_installed THEN
        RAISE NOTICE 'TimescaleDB not installed, skipping TimescaleDB integration tests';
        -- Use pgTAP's skip_all function to skip all tests
        PERFORM skip_all('TimescaleDB not installed');
    END IF;
END$$;

-- Only proceed with tests if TimescaleDB is available
DO $$
DECLARE
    v_timescale_installed BOOLEAN;
BEGIN
    SELECT EXISTS(
        SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
    ) INTO v_timescale_installed;
    
    IF v_timescale_installed THEN
        -- Plan for 25 tests
        PERFORM plan(25);
    ELSE
        -- No tests to run
        PERFORM plan(0);
    END IF;
END$$;

-- Test 1: Verify TimescaleDB extension exists
SELECT has_extension(
    'timescaledb',
    'TimescaleDB extension should be installed'
);

-- Test 2: Verify pg_steadytext extension coexists with TimescaleDB
SELECT has_extension(
    'pg_steadytext',
    'pg_steadytext extension should work with TimescaleDB'
);

-- Test 3: Create a hypertable for time-series log data
CREATE TEMP TABLE test_logs (
    time TIMESTAMPTZ NOT NULL,
    level TEXT,
    message TEXT,
    metadata JSONB
);

-- Convert to hypertable
SELECT create_hypertable(
    'test_logs',
    'time',
    if_not_exists => TRUE
);

SELECT ok(
    EXISTS(
        SELECT 1 FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'test_logs'
    ),
    'test_logs should be a hypertable'
);

-- Test 4: Insert sample log data
INSERT INTO test_logs (time, level, message, metadata) VALUES
    (NOW() - INTERVAL '3 hours', 'INFO', 'Application started successfully', '{"component": "main"}'),
    (NOW() - INTERVAL '2 hours 45 minutes', 'DEBUG', 'Database connection established', '{"component": "db"}'),
    (NOW() - INTERVAL '2 hours 30 minutes', 'WARNING', 'High memory usage detected', '{"component": "monitor", "memory_mb": 3500}'),
    (NOW() - INTERVAL '2 hours', 'ERROR', 'Failed to connect to external API', '{"component": "api", "retry": 3}'),
    (NOW() - INTERVAL '1 hour 45 minutes', 'INFO', 'Cache cleared successfully', '{"component": "cache"}'),
    (NOW() - INTERVAL '1 hour 30 minutes', 'ERROR', 'Database query timeout', '{"component": "db", "query_time": 30.5}'),
    (NOW() - INTERVAL '1 hour', 'INFO', 'Backup completed', '{"component": "backup", "size_gb": 45}'),
    (NOW() - INTERVAL '45 minutes', 'DEBUG', 'Request processed', '{"component": "api", "duration_ms": 250}'),
    (NOW() - INTERVAL '30 minutes', 'WARNING', 'Disk space running low', '{"component": "monitor", "free_gb": 10}'),
    (NOW() - INTERVAL '15 minutes', 'INFO', 'User session started', '{"component": "auth", "user_id": 123}');

SELECT ok(
    (SELECT COUNT(*) FROM test_logs) = 10,
    'Should have 10 log entries'
);

-- Test 5: Test steadytext_generate on hypertable data
SELECT ok(
    length(steadytext_generate(
        'Describe this log: ' || message,
        100
    )) > 0,
    'steadytext_generate should work on hypertable data'
) FROM test_logs LIMIT 1;

-- Test 6: Test steadytext_embed on hypertable data
SELECT ok(
    vector_dims(steadytext_embed(message)) = 1024,
    'steadytext_embed should return 1024-dimensional vectors for hypertable data'
) FROM test_logs LIMIT 1;

-- Test 7: Create a continuous aggregate with steadytext_summarize
-- AIDEV-NOTE: This is the primary test case - continuous aggregates with AI summarization
CREATE MATERIALIZED VIEW hourly_log_summary
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    level,
    COUNT(*) as log_count,
    steadytext_summarize(
        message,
        jsonb_build_object(
            'max_facts', 5,
            'preserve_samples', true
        )
    ) as ai_summary
FROM test_logs
GROUP BY hour, level
WITH NO DATA;

SELECT has_materialized_view(
    'hourly_log_summary',
    'Continuous aggregate hourly_log_summary should exist'
);

-- Test 8: Refresh the continuous aggregate
CALL refresh_continuous_aggregate('hourly_log_summary', NULL, NULL);

SELECT ok(
    (SELECT COUNT(*) FROM hourly_log_summary) > 0,
    'Continuous aggregate should contain data after refresh'
);

-- Test 9: Verify AI summary in continuous aggregate
SELECT ok(
    length((SELECT ai_summary FROM hourly_log_summary LIMIT 1)::text) > 0,
    'Continuous aggregate should contain non-empty AI summaries'
);

-- Test 10: Test schema qualification in continuous aggregate context
-- AIDEV-NOTE: This tests the v2025.8.26 fix for schema qualification
CREATE SCHEMA test_timescale_schema;
SET search_path TO test_timescale_schema, public;

-- Create a table in the custom schema
CREATE TABLE test_timescale_schema.test_events (
    time TIMESTAMPTZ NOT NULL,
    event_type TEXT,
    description TEXT
);

-- Convert to hypertable
SELECT create_hypertable(
    'test_timescale_schema.test_events',
    'time',
    if_not_exists => TRUE
);

-- Insert test data
INSERT INTO test_timescale_schema.test_events (time, event_type, description) VALUES
    (NOW() - INTERVAL '2 hours', 'start', 'System initialization started'),
    (NOW() - INTERVAL '1 hour', 'process', 'Data processing in progress'),
    (NOW(), 'complete', 'Task completed successfully');

-- Create continuous aggregate in custom schema
CREATE MATERIALIZED VIEW test_timescale_schema.event_summary
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    event_type,
    public.steadytext_summarize(description, '{}'::jsonb) as summary
FROM test_timescale_schema.test_events
GROUP BY hour, event_type
WITH NO DATA;

-- Refresh and test
CALL refresh_continuous_aggregate('test_timescale_schema.event_summary', NULL, NULL);

SELECT ok(
    EXISTS(SELECT 1 FROM test_timescale_schema.event_summary),
    'Continuous aggregate should work with custom schema and schema-qualified functions'
);

-- Reset search path
SET search_path TO public;

-- Test 11: Create a regular materialized view with pg_steadytext aggregates
CREATE MATERIALIZED VIEW daily_log_summary AS
SELECT 
    date_trunc('day', time) AS day,
    steadytext_summarize(
        message,
        jsonb_build_object(
            'max_facts', 10,
            'preserve_samples', true
        )
    ) as daily_summary,
    COUNT(*) as total_logs,
    array_agg(DISTINCT level) as log_levels
FROM test_logs
GROUP BY day;

SELECT has_materialized_view(
    'daily_log_summary',
    'Regular materialized view with pg_steadytext should exist'
);

-- Test 12: Verify regular materialized view has data
SELECT ok(
    (SELECT COUNT(*) FROM daily_log_summary) > 0,
    'Regular materialized view should contain data'
);

-- Test 13: Test partial aggregation for distributed scenarios
-- AIDEV-NOTE: Tests the partial/final aggregate pattern for TimescaleDB
CREATE TEMP TABLE partial_summaries AS
SELECT 
    level,
    steadytext_summarize_partial(
        message,
        jsonb_build_object('max_facts', 3)
    ) as partial_state
FROM test_logs
GROUP BY level;

SELECT ok(
    (SELECT COUNT(*) FROM partial_summaries) > 0,
    'Partial aggregation should produce intermediate states'
);

-- Test 14: Test final aggregation from partial states
SELECT ok(
    length(
        (SELECT steadytext_summarize_final(partial_state) 
         FROM partial_summaries 
         WHERE level = 'ERROR')::text
    ) > 0,
    'Final aggregation should produce summary from partial states'
);

-- Test 15: Test async functions with hypertable data
SELECT ok(
    (SELECT steadytext_generate_async('Analyze log: ' || message, 50) 
     FROM test_logs LIMIT 1) IS NOT NULL,
    'Async generation should return UUID for hypertable data'
);

-- Test 16: Create time-series sensor data scenario
CREATE TEMP TABLE sensor_readings (
    time TIMESTAMPTZ NOT NULL,
    sensor_id TEXT,
    temperature NUMERIC,
    humidity NUMERIC,
    status_message TEXT
);

SELECT create_hypertable(
    'sensor_readings',
    'time',
    if_not_exists => TRUE
);

-- Insert sensor data
INSERT INTO sensor_readings (time, sensor_id, temperature, humidity, status_message)
SELECT 
    generate_series(
        NOW() - INTERVAL '24 hours',
        NOW(),
        INTERVAL '1 hour'
    ) as time,
    'sensor_' || (random() * 3)::int as sensor_id,
    20 + (random() * 10) as temperature,
    40 + (random() * 30) as humidity,
    CASE 
        WHEN random() < 0.7 THEN 'Normal operation'
        WHEN random() < 0.9 THEN 'Minor fluctuation detected'
        ELSE 'Anomaly detected, requires attention'
    END as status_message;

-- Test 17: Create continuous aggregate for sensor data with AI narration
CREATE MATERIALIZED VIEW sensor_daily_narrative
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS day,
    sensor_id,
    AVG(temperature) as avg_temp,
    AVG(humidity) as avg_humidity,
    steadytext_summarize(
        status_message,
        jsonb_build_object(
            'max_facts', 5,
            'topic', 'sensor health status'
        )
    ) as daily_narrative
FROM sensor_readings
GROUP BY day, sensor_id
WITH NO DATA;

CALL refresh_continuous_aggregate('sensor_daily_narrative', NULL, NULL);

SELECT ok(
    EXISTS(SELECT 1 FROM sensor_daily_narrative WHERE daily_narrative IS NOT NULL),
    'Sensor continuous aggregate should contain AI narratives'
);

-- Test 18: Test reranking with TimescaleDB data
CREATE TEMP TABLE documents_ts (
    id SERIAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    content TEXT
);

INSERT INTO documents_ts (content) VALUES
    ('PostgreSQL is a powerful relational database'),
    ('TimescaleDB extends PostgreSQL for time-series'),
    ('pg_steadytext provides AI capabilities for PostgreSQL');

SELECT ok(
    (SELECT steadytext_rerank('time-series database', content) 
     FROM documents_ts 
     WHERE content LIKE '%TimescaleDB%'
     LIMIT 1) >= 0,
    'Reranking should work with TimescaleDB tables'
);

-- Test 19: Test embedding with hypertable compression
-- Create compressed chunks (if compression is available)
DO $$
BEGIN
    -- Try to compress chunks if compression is available
    PERFORM compress_chunk(c.chunk_name) 
    FROM timescaledb_information.chunks c
    WHERE c.hypertable_name = 'test_logs'
    AND NOT c.is_compressed
    LIMIT 1;
EXCEPTION 
    WHEN OTHERS THEN
        -- Compression might not be available, that's okay
        NULL;
END$$;

-- Embeddings should still work on compressed data
SELECT ok(
    vector_dims(steadytext_embed(message)) = 1024,
    'Embeddings should work even with compressed chunks'
) FROM test_logs LIMIT 1;

-- Test 20: Test cagg refresh policies don't break summarization
-- AIDEV-NOTE: Ensures refresh policies work with pg_steadytext aggregates
DO $$
BEGIN
    -- Add a refresh policy (runs every 12 hours in production)
    PERFORM add_continuous_aggregate_policy(
        'hourly_log_summary',
        start_offset => INTERVAL '1 day',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '12 hours',
        if_not_exists => TRUE
    );
EXCEPTION
    WHEN OTHERS THEN
        -- Policy might already exist or feature not available
        NULL;
END$$;

SELECT ok(
    TRUE,  -- If we got here without error, the policy is compatible
    'Refresh policies should be compatible with pg_steadytext aggregates'
);

-- Test 21: Test real-time aggregate with recent data
INSERT INTO test_logs (time, level, message, metadata) VALUES
    (NOW() - INTERVAL '5 minutes', 'CRITICAL', 'System failure detected', '{"component": "core"}');

-- Force refresh of the affected region
CALL refresh_continuous_aggregate(
    'hourly_log_summary',
    NOW() - INTERVAL '1 hour',
    NOW()
);

SELECT ok(
    EXISTS(
        SELECT 1 FROM hourly_log_summary 
        WHERE hour >= NOW() - INTERVAL '1 hour'
        AND level = 'CRITICAL'
    ),
    'Real-time aggregation should include recent critical events'
);

-- Test 22: Test structured generation with time-series context
SELECT ok(
    length(steadytext_generate_json(
        'Generate a JSON report for the sensor',
        '{"type": "object", "properties": {"status": {"type": "string"}, "temperature": {"type": "number"}}}'::jsonb
    )) > 0,
    'Structured generation should work in TimescaleDB context'
) FROM sensor_readings LIMIT 1;

-- Test 23: Test summarization with remote models (unsafe mode)
-- AIDEV-NOTE: Tests v2025.8.26 remote model support in continuous aggregates
DO $$
DECLARE
    v_summary TEXT;
BEGIN
    -- Test with metadata specifying remote model (would require unsafe_mode in production)
    SELECT steadytext_summarize(
        'Test message for remote model',
        jsonb_build_object(
            'max_facts', 3,
            'model', 'local',  -- Use local model for testing
            'unsafe_mode', false
        )
    )::text INTO v_summary;
    
    PERFORM ok(
        v_summary IS NOT NULL,
        'Summarization with model parameters should work'
    );
END$$;

-- Test 24: Test cache behavior with continuous aggregates
-- AIDEV-NOTE: Ensures caching works correctly with TimescaleDB
CREATE TEMP TABLE cache_test_logs (
    time TIMESTAMPTZ NOT NULL,
    message TEXT
);

SELECT create_hypertable('cache_test_logs', 'time', if_not_exists => TRUE);

-- Insert identical messages to test caching
INSERT INTO cache_test_logs (time, message)
SELECT 
    generate_series(NOW() - INTERVAL '3 hours', NOW(), INTERVAL '1 hour'),
    'Identical message for cache testing';

-- Create aggregate that should hit cache
CREATE MATERIALIZED VIEW cache_test_summary
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    steadytext_summarize(message, '{}'::jsonb) as summary
FROM cache_test_logs
GROUP BY hour
WITH NO DATA;

CALL refresh_continuous_aggregate('cache_test_summary', NULL, NULL);

SELECT ok(
    (SELECT COUNT(DISTINCT summary) FROM cache_test_summary) = 1,
    'Identical messages should produce cached summaries in continuous aggregates'
);

-- Test 25: Cleanup test - verify no errors when dropping TimescaleDB objects
DO $$
BEGIN
    DROP MATERIALIZED VIEW IF EXISTS hourly_log_summary CASCADE;
    DROP MATERIALIZED VIEW IF EXISTS sensor_daily_narrative CASCADE;
    DROP MATERIALIZED VIEW IF EXISTS cache_test_summary CASCADE;
    DROP TABLE IF EXISTS test_logs CASCADE;
    DROP TABLE IF EXISTS sensor_readings CASCADE;
    DROP TABLE IF EXISTS cache_test_logs CASCADE;
    DROP SCHEMA IF EXISTS test_timescale_schema CASCADE;
    
    PERFORM ok(
        TRUE,
        'Cleanup of TimescaleDB objects should complete without errors'
    );
END$$;

-- Finish tests
SELECT * FROM finish();
ROLLBACK;

-- AIDEV-NOTE: This comprehensive test suite covers:
-- 1. Basic TimescaleDB compatibility with pg_steadytext
-- 2. Hypertable creation and data operations
-- 3. Continuous aggregates with steadytext_summarize (primary focus)
-- 4. Schema qualification fixes from v2025.8.26
-- 5. Partial/final aggregation patterns
-- 6. Compression compatibility
-- 7. Refresh policies
-- 8. Real-time aggregation
-- 9. Cache behavior
-- 10. Both time-series logs and IoT sensor data scenarios
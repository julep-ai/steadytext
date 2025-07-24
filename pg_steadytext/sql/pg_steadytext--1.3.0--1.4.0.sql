-- pg_steadytext extension migration from 1.3.0 to 1.4.0
-- Adds pg_cron-based automatic cache eviction with frecency algorithm

-- AIDEV-NOTE: This migration adds automatic cache eviction using pg_cron
-- to maintain cache size and capacity limits based on frecency scores

-- Update version
CREATE OR REPLACE FUNCTION steadytext_version()
RETURNS text AS $$
BEGIN
    RETURN '1.4.0';
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- AIDEV-SECTION: CACHE_EVICTION_CONFIGURATION
-- Add cache eviction configuration to steadytext_config
INSERT INTO steadytext_config (key, value, description) VALUES
    ('cache_eviction_enabled', 'true', 'Enable automatic cache eviction'),
    ('cache_eviction_interval', '300', 'Cache eviction interval in seconds (default: 5 minutes)'),
    ('cache_max_entries', '10000', 'Maximum number of cache entries before eviction'),
    ('cache_max_size_mb', '1000', 'Maximum cache size in MB before eviction'),
    ('cache_eviction_batch_size', '100', 'Number of entries to evict in each batch'),
    ('cache_min_access_count', '2', 'Minimum access count to protect from eviction'),
    ('cache_min_age_hours', '1', 'Minimum age in hours to protect from eviction'),
    ('cron_host', '"localhost"', 'Database host for pg_cron connections'),
    ('cron_port', '5432', 'Database port for pg_cron connections')
ON CONFLICT (key) DO NOTHING;

-- AIDEV-SECTION: CACHE_PERFORMANCE_INDEXES
-- Add indexes for optimal frecency-based eviction performance
-- This index supports the ORDER BY frecency_score query in eviction
-- AIDEV-NOTE: WHERE clause removed due to NOW() not being immutable
CREATE INDEX IF NOT EXISTS idx_steadytext_cache_frecency_eviction 
ON steadytext_cache (access_count, last_accessed);

-- AIDEV-SECTION: CACHE_STATISTICS_FUNCTIONS
-- Enhanced cache statistics function with size calculations
CREATE OR REPLACE FUNCTION steadytext_cache_stats_extended()
RETURNS TABLE(
    total_entries BIGINT,
    total_size_mb FLOAT,
    cache_hit_rate FLOAT,
    avg_access_count FLOAT,
    oldest_entry TIMESTAMPTZ,
    newest_entry TIMESTAMPTZ,
    low_frecency_count BIGINT,
    protected_count BIGINT,
    eviction_candidates BIGINT
)
LANGUAGE sql
STABLE PARALLEL SAFE
AS $$
    WITH cache_analysis AS (
        SELECT 
            COUNT(*)::BIGINT as total_entries,
            COALESCE(SUM(pg_column_size(response) + pg_column_size(embedding)) / 1024.0 / 1024.0, 0)::FLOAT as total_size_mb,
            COALESCE(SUM(CASE WHEN access_count > 1 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0), 0)::FLOAT as cache_hit_rate,
            COALESCE(AVG(access_count), 0)::FLOAT as avg_access_count,
            MIN(created_at) as oldest_entry,
            MAX(created_at) as newest_entry,
            -- Count entries with low frecency scores
            SUM(CASE 
                WHEN access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0) < 1 
                THEN 1 ELSE 0 
            END)::BIGINT as low_frecency_count,
            -- Count protected entries (high access count or recently created)
            SUM(CASE 
                WHEN access_count >= 2 OR created_at > NOW() - INTERVAL '1 hour' 
                THEN 1 ELSE 0 
            END)::BIGINT as protected_count,
            -- Count eviction candidates
            SUM(CASE 
                WHEN access_count < 2 
                    AND created_at < NOW() - INTERVAL '1 hour'
                    AND access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0) < 1
                THEN 1 ELSE 0 
            END)::BIGINT as eviction_candidates
        FROM steadytext_cache
    )
    SELECT * FROM cache_analysis;
$$;

-- AIDEV-SECTION: CACHE_EVICTION_FUNCTIONS
-- Function to perform cache eviction based on frecency
CREATE OR REPLACE FUNCTION steadytext_cache_evict_by_frecency(
    target_entries INT DEFAULT NULL,
    target_size_mb FLOAT DEFAULT NULL,
    batch_size INT DEFAULT 100,
    min_access_count INT DEFAULT 2,
    min_age_hours INT DEFAULT 1
)
RETURNS TABLE(
    evicted_count INT,
    freed_size_mb FLOAT,
    remaining_entries BIGINT,
    remaining_size_mb FLOAT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_evicted_count INT := 0;
    v_freed_size BIGINT := 0;
    v_current_stats RECORD;
    v_max_entries INT;
    v_max_size_mb FLOAT;
    v_should_evict BOOLEAN := FALSE;
    v_loop_count INT := 0;
    v_max_loop_count INT := 1000; -- Safety limit to prevent infinite loops
BEGIN
    -- Get current cache stats
    SELECT 
        COUNT(*)::BIGINT as total_entries,
        COALESCE(SUM(pg_column_size(response) + pg_column_size(embedding)), 0)::BIGINT as total_size
    INTO v_current_stats
    FROM steadytext_cache;
    
    -- Use provided targets or get from config
    IF target_entries IS NULL THEN
        SELECT value::INT INTO v_max_entries 
        FROM steadytext_config 
        WHERE key = 'cache_max_entries';
        v_max_entries := COALESCE(v_max_entries, 10000);
    ELSE
        v_max_entries := target_entries;
    END IF;
    
    IF target_size_mb IS NULL THEN
        SELECT value::FLOAT INTO v_max_size_mb 
        FROM steadytext_config 
        WHERE key = 'cache_max_size_mb';
        v_max_size_mb := COALESCE(v_max_size_mb, 1000);
    ELSE
        v_max_size_mb := target_size_mb;
    END IF;
    
    -- Check if eviction is needed
    IF v_current_stats.total_entries > v_max_entries OR 
       (v_current_stats.total_size / 1024.0 / 1024.0) > v_max_size_mb THEN
        v_should_evict := TRUE;
    END IF;
    
    -- Perform eviction if needed
    WHILE v_should_evict LOOP
        -- Safety check to prevent infinite loops
        v_loop_count := v_loop_count + 1;
        IF v_loop_count > v_max_loop_count THEN
            RAISE WARNING 'Cache eviction loop exceeded maximum iterations (%), breaking to prevent infinite loop', v_max_loop_count;
            EXIT;
        END IF;
        WITH eviction_batch AS (
            DELETE FROM steadytext_cache
            WHERE id IN (
                SELECT id 
                FROM steadytext_cache_with_frecency
                WHERE 
                    -- Don't evict entries with high access count
                    access_count < min_access_count
                    -- Don't evict very recent entries
                    AND created_at < NOW() - INTERVAL '1 hour' * min_age_hours
                ORDER BY frecency_score ASC
                LIMIT batch_size
            )
            RETURNING pg_column_size(response) + pg_column_size(embedding) as entry_size
        )
        SELECT 
            COUNT(*)::INT,
            COALESCE(SUM(entry_size), 0)::BIGINT
        INTO 
            v_evicted_count,
            v_freed_size
        FROM eviction_batch;
        
        -- Break if nothing was evicted
        IF v_evicted_count = 0 THEN
            EXIT;
        END IF;
        
        -- Update running totals
        v_current_stats.total_entries := v_current_stats.total_entries - v_evicted_count;
        v_current_stats.total_size := v_current_stats.total_size - v_freed_size;
        
        -- Check if we've reached targets
        IF v_current_stats.total_entries <= v_max_entries AND 
           (v_current_stats.total_size / 1024.0 / 1024.0) <= v_max_size_mb THEN
            v_should_evict := FALSE;
        END IF;
    END LOOP;
    
    -- Return results
    RETURN QUERY
    SELECT 
        v_evicted_count,
        (v_freed_size / 1024.0 / 1024.0)::FLOAT,
        v_current_stats.total_entries,
        (v_current_stats.total_size / 1024.0 / 1024.0)::FLOAT;
END;
$$;

-- AIDEV-SECTION: SCHEDULED_EVICTION_FUNCTION
-- Function to be called by pg_cron for scheduled eviction
CREATE OR REPLACE FUNCTION steadytext_cache_scheduled_eviction()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    v_eviction_enabled BOOLEAN;
    v_result RECORD;
    v_start_time TIMESTAMPTZ;
    v_end_time TIMESTAMPTZ;
    v_duration_ms INT;
BEGIN
    -- Check if eviction is enabled
    SELECT value::BOOLEAN INTO v_eviction_enabled
    FROM steadytext_config
    WHERE key = 'cache_eviction_enabled';
    
    IF NOT COALESCE(v_eviction_enabled, TRUE) THEN
        RETURN jsonb_build_object(
            'status', 'skipped',
            'reason', 'Cache eviction disabled',
            'timestamp', NOW()
        );
    END IF;
    
    v_start_time := clock_timestamp();
    
    -- Perform eviction using configured parameters
    SELECT * INTO v_result
    FROM steadytext_cache_evict_by_frecency();
    
    v_end_time := clock_timestamp();
    v_duration_ms := EXTRACT(MILLISECONDS FROM (v_end_time - v_start_time))::INT;
    
    -- Log to audit table if significant eviction occurred
    IF v_result.evicted_count > 0 THEN
        INSERT INTO steadytext_audit_log (
            action, 
            details,
            success
        ) VALUES (
            'cache_eviction',
            jsonb_build_object(
                'evicted_count', v_result.evicted_count,
                'freed_size_mb', v_result.freed_size_mb,
                'remaining_entries', v_result.remaining_entries,
                'remaining_size_mb', v_result.remaining_size_mb,
                'duration_ms', v_duration_ms
            ),
            TRUE
        );
    END IF;
    
    -- Return detailed result
    RETURN jsonb_build_object(
        'status', 'completed',
        'timestamp', NOW(),
        'evicted_count', v_result.evicted_count,
        'freed_size_mb', v_result.freed_size_mb,
        'remaining_entries', v_result.remaining_entries,
        'remaining_size_mb', v_result.remaining_size_mb,
        'duration_ms', v_duration_ms
    );
END;
$$;

-- AIDEV-SECTION: CACHE_MAINTENANCE_FUNCTIONS
-- Function to analyze cache usage patterns
CREATE OR REPLACE FUNCTION steadytext_cache_analyze_usage()
RETURNS TABLE(
    access_bucket TEXT,
    entry_count BIGINT,
    avg_frecency_score FLOAT,
    total_size_mb FLOAT,
    percentage_of_cache FLOAT
)
LANGUAGE sql
STABLE PARALLEL SAFE
AS $$
    WITH cache_buckets AS (
        SELECT 
            CASE 
                WHEN access_count = 1 THEN '1_single_access'
                WHEN access_count BETWEEN 2 AND 5 THEN '2_low_access'
                WHEN access_count BETWEEN 6 AND 20 THEN '3_medium_access'
                WHEN access_count BETWEEN 21 AND 100 THEN '4_high_access'
                ELSE '5_very_high_access'
            END as access_bucket,
            COUNT(*) as entry_count,
            AVG(access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0)) as avg_frecency_score,
            SUM(pg_column_size(response) + pg_column_size(embedding)) / 1024.0 / 1024.0 as total_size_mb
        FROM steadytext_cache
        GROUP BY access_bucket
    ),
    totals AS (
        SELECT 
            SUM(entry_count) as total_entries
        FROM cache_buckets
    )
    SELECT 
        cb.access_bucket,
        cb.entry_count,
        cb.avg_frecency_score,
        cb.total_size_mb,
        (cb.entry_count::FLOAT / NULLIF(t.total_entries, 0) * 100)::FLOAT as percentage_of_cache
    FROM cache_buckets cb, totals t
    ORDER BY cb.access_bucket;
$$;

-- Function to get cache entries that would be evicted
CREATE OR REPLACE FUNCTION steadytext_cache_preview_eviction(
    preview_count INT DEFAULT 10
)
RETURNS TABLE(
    cache_key TEXT,
    prompt TEXT,
    access_count INT,
    last_accessed TIMESTAMPTZ,
    created_at TIMESTAMPTZ,
    frecency_score FLOAT,
    size_bytes BIGINT
)
LANGUAGE sql
STABLE PARALLEL SAFE
AS $$
    SELECT 
        cache_key,
        LEFT(prompt, 100) as prompt,  -- Truncate for display
        access_count,
        last_accessed,
        created_at,
        access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0) as frecency_score,
        pg_column_size(response) + pg_column_size(embedding) as size_bytes
    FROM steadytext_cache
    WHERE 
        -- Same criteria as eviction function
        access_count < 2
        AND created_at < NOW() - INTERVAL '1 hour'
    ORDER BY 
        access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0) ASC
    LIMIT preview_count;
$$;

-- AIDEV-SECTION: PG_CRON_SETUP
-- Note: pg_cron must be installed and configured separately
-- This creates a helper function to set up the cron job

CREATE OR REPLACE FUNCTION steadytext_setup_cache_eviction_cron()
RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    v_interval INT;
    v_cron_expression TEXT;
    v_job_id BIGINT;
    v_host TEXT;
    v_port INT;
BEGIN
    -- Check if pg_cron is available
    IF NOT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'pg_cron'
    ) THEN
        RETURN 'Error: pg_cron extension is not installed. Please install pg_cron first.';
    END IF;
    
    -- Get configuration values
    SELECT value::INT INTO v_interval
    FROM steadytext_config
    WHERE key = 'cache_eviction_interval';
    
    SELECT value INTO v_host
    FROM steadytext_config
    WHERE key = 'cron_host';
    
    SELECT value::INT INTO v_port
    FROM steadytext_config
    WHERE key = 'cron_port';
    
    -- Set defaults if not configured
    v_interval := COALESCE(v_interval, 300); -- Default 5 minutes
    v_host := COALESCE(v_host, '"localhost"');
    v_port := COALESCE(v_port, 5432);
    
    -- Validate configuration values
    IF v_interval < 60 THEN
        RETURN 'Error: cache_eviction_interval must be at least 60 seconds';
    END IF;
    
    IF v_host = '' OR v_host IS NULL THEN
        RETURN 'Error: cron_host cannot be empty';
    END IF;
    
    IF v_port < 1 OR v_port > 65535 THEN
        RETURN 'Error: cron_port must be between 1 and 65535';
    END IF;
    
    -- Convert seconds to cron expression
    -- For intervals less than an hour, use minute-based scheduling
    IF v_interval < 3600 THEN
        -- Ensure minimum 1-minute interval to avoid invalid cron expressions
        v_cron_expression := '*/' || GREATEST(1, v_interval / 60)::TEXT || ' * * * *';
    ELSE
        -- For longer intervals, use hourly scheduling
        -- Ensure minimum 1-hour interval to avoid invalid cron expressions
        v_cron_expression := '0 */' || GREATEST(1, v_interval / 3600)::TEXT || ' * * *';
    END IF;
    
    -- Remove existing job if any
    DELETE FROM cron.job 
    WHERE jobname = 'steadytext_cache_eviction';
    
    -- Schedule the job with error handling
    BEGIN
        INSERT INTO cron.job (
            schedule, 
            command, 
            nodename, 
            nodeport, 
            database, 
            username,
            jobname
        ) VALUES (
            v_cron_expression,
            'SELECT steadytext_cache_scheduled_eviction();',
            v_host,
            v_port,
            current_database(),
            current_user,
            'steadytext_cache_eviction'
        ) RETURNING jobid INTO v_job_id;
        
        RETURN 'Cache eviction cron job scheduled with ID ' || v_job_id || 
               ' using schedule: ' || v_cron_expression ||
               ' on host: ' || v_host || ':' || v_port;
    EXCEPTION
        WHEN OTHERS THEN
            RETURN 'Error scheduling cron job: ' || SQLERRM || 
                   ' (schedule: ' || v_cron_expression || 
                   ', host: ' || v_host || ':' || v_port || ')';
    END;
END;
$$;

-- Helper function to disable cache eviction cron
CREATE OR REPLACE FUNCTION steadytext_disable_cache_eviction_cron()
RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    v_deleted_count INT;
BEGIN
    -- Check if pg_cron is available
    IF NOT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'pg_cron'
    ) THEN
        RETURN 'Error: pg_cron extension is not installed.';
    END IF;
    
    -- Remove the job
    DELETE FROM cron.job 
    WHERE jobname = 'steadytext_cache_eviction';
    
    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
    
    IF v_deleted_count > 0 THEN
        RETURN 'Cache eviction cron job disabled successfully.';
    ELSE
        RETURN 'No cache eviction cron job found.';
    END IF;
END;
$$;

-- AIDEV-SECTION: CACHE_WARMUP_FUNCTIONS
-- Function to warm up cache with frequently accessed entries
CREATE OR REPLACE FUNCTION steadytext_cache_warmup(
    warmup_count INT DEFAULT 100
)
RETURNS TABLE(
    warmed_entries INT,
    total_time_ms INT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_start_time TIMESTAMPTZ;
    v_end_time TIMESTAMPTZ;
    v_warmed_count INT := 0;
    v_cache_entry RECORD;
BEGIN
    v_start_time := clock_timestamp();
    
    -- Find entries that are frequently accessed but not in memory
    FOR v_cache_entry IN 
        SELECT cache_key, prompt, generation_params
        FROM steadytext_cache
        WHERE access_count > 5
        ORDER BY access_count DESC
        LIMIT warmup_count
    LOOP
        -- Touch the cache entry to warm it up
        UPDATE steadytext_cache
        SET last_accessed = NOW()
        WHERE cache_key = v_cache_entry.cache_key;
        
        v_warmed_count := v_warmed_count + 1;
    END LOOP;
    
    v_end_time := clock_timestamp();
    
    RETURN QUERY
    SELECT 
        v_warmed_count,
        EXTRACT(MILLISECONDS FROM (v_end_time - v_start_time))::INT;
END;
$$;

-- Add helpful comments
COMMENT ON FUNCTION steadytext_cache_stats_extended() IS 
'Extended cache statistics including eviction candidate analysis';

COMMENT ON FUNCTION steadytext_cache_evict_by_frecency(INT, FLOAT, INT, INT, INT) IS 
'Evict cache entries based on frecency score to maintain size/capacity limits';

COMMENT ON FUNCTION steadytext_cache_scheduled_eviction() IS 
'Scheduled cache eviction function to be called by pg_cron';

COMMENT ON FUNCTION steadytext_cache_analyze_usage() IS 
'Analyze cache usage patterns by access frequency buckets';

COMMENT ON FUNCTION steadytext_cache_preview_eviction(INT) IS 
'Preview which cache entries would be evicted next';

COMMENT ON FUNCTION steadytext_setup_cache_eviction_cron() IS 
'Set up pg_cron job for automatic cache eviction (requires pg_cron extension)';

COMMENT ON FUNCTION steadytext_disable_cache_eviction_cron() IS 
'Disable the pg_cron job for automatic cache eviction';

COMMENT ON FUNCTION steadytext_cache_warmup(INT) IS 
'Warm up cache by touching frequently accessed entries';

-- AIDEV-NOTE: This completes the cache eviction implementation for pg_steadytext v1.4.0
-- 
-- To enable automatic cache eviction:
-- 1. Install pg_cron extension: CREATE EXTENSION pg_cron;
-- 2. Run: SELECT steadytext_setup_cache_eviction_cron();
-- 3. Configure parameters in steadytext_config table
-- 
-- The eviction algorithm uses frecency (frequency + recency) to determine
-- which entries to evict, protecting high-access and recent entries.
--
-- AIDEV-TODO: Future enhancements could include:
-- - Adaptive eviction thresholds based on cache hit rates
-- - Different eviction strategies (LRU, LFU, ARC)
-- - Cache partitioning by model or use case
-- - Integration with PostgreSQL's shared buffer cache
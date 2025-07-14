# PostgreSQL Extension - Advanced Topics

This document covers advanced configuration, performance tuning, security, and integration patterns for the pg_steadytext PostgreSQL extension.

**Navigation**: [Main Documentation](postgresql-extension.md) | [Structured Generation](postgresql-extension-structured.md) | [AI Features](postgresql-extension-ai.md) | [Async Functions](postgresql-extension-async.md)

---

## Performance Tuning

### Cache Configuration

The extension uses PostgreSQL-based caching for optimal performance:

```sql
-- View cache statistics
SELECT * FROM steadytext_cache_stats();

-- Clear specific cache types
SELECT steadytext_clear_cache('generation');
SELECT steadytext_clear_cache('embedding');
SELECT steadytext_clear_cache('reranking');
SELECT steadytext_clear_cache('all');

-- Configure cache settings
ALTER SYSTEM SET steadytext.generation_cache_size = '512MB';
ALTER SYSTEM SET steadytext.embedding_cache_size = '1GB';
ALTER SYSTEM SET steadytext.cache_ttl = '7 days';
SELECT pg_reload_conf();

-- Monitor cache hit rates
CREATE OR REPLACE VIEW cache_performance AS
SELECT 
    cache_type,
    hit_count,
    miss_count,
    ROUND(hit_count::numeric / NULLIF(hit_count + miss_count, 0) * 100, 2) as hit_rate,
    pg_size_pretty(cache_size_bytes) as cache_size,
    entry_count
FROM steadytext_cache_stats();
```

### Automatic Cache Eviction with pg_cron

The extension supports automatic cache eviction using pg_cron:

```sql
-- Enable pg_cron extension
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Schedule automatic cache eviction
SELECT cron.schedule(
    'steadytext-cache-eviction',
    '0 3 * * *',  -- Daily at 3 AM
    $$SELECT steadytext_evict_cache(
        target_memory_mb := 500,
        eviction_strategy := 'frecency'
    )$$
);

-- Custom eviction for specific cache types
SELECT cron.schedule(
    'steadytext-embedding-cache-cleanup',
    '0 */6 * * *',  -- Every 6 hours
    $$SELECT steadytext_evict_cache(
        cache_type := 'embedding',
        target_memory_mb := 200,
        min_age_hours := 24
    )$$
);

-- Monitor eviction effectiveness
CREATE OR REPLACE VIEW eviction_history AS
SELECT 
    eviction_time,
    cache_type,
    entries_before,
    entries_after,
    bytes_freed,
    duration_ms
FROM steadytext_eviction_log
ORDER BY eviction_time DESC
LIMIT 100;
```

### Memory Management

```sql
-- Monitor model memory usage
SELECT * FROM steadytext_model_memory_usage();

-- Configure memory limits
ALTER SYSTEM SET steadytext.max_model_memory = '4GB';
ALTER SYSTEM SET steadytext.model_cache_mode = 'mmap';  -- or 'ram'
ALTER SYSTEM SET steadytext.enable_model_sharing = true;

-- Preload models for better performance
SELECT steadytext_preload_models();

-- Unload models to free memory
SELECT steadytext_unload_models();

-- Dynamic memory management based on system load
CREATE OR REPLACE FUNCTION manage_model_memory()
RETURNS void AS $$
DECLARE
    free_memory_mb INTEGER;
BEGIN
    -- Get free memory
    SELECT (memory_free_mb + memory_cached_mb) INTO free_memory_mb
    FROM pg_stat_memory;
    
    IF free_memory_mb < 1000 THEN
        -- Low memory: unload models
        PERFORM steadytext_unload_models();
    ELSIF free_memory_mb > 4000 THEN
        -- Plenty of memory: preload models
        PERFORM steadytext_preload_models();
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Schedule memory management
SELECT cron.schedule('memory-management', '*/5 * * * *', 'SELECT manage_model_memory()');
```

### Connection Pooling

For high-concurrency scenarios with the daemon:

```sql
-- Configure connection pooling
ALTER SYSTEM SET steadytext.daemon_pool_size = 10;
ALTER SYSTEM SET steadytext.daemon_pool_timeout = '5s';
ALTER SYSTEM SET steadytext.daemon_reconnect_interval = '1s';

-- Monitor daemon connections
CREATE OR REPLACE VIEW daemon_pool_status AS
SELECT 
    connection_id,
    state,
    last_used,
    request_count,
    error_count,
    avg_response_time_ms
FROM steadytext_daemon_connections();

-- Health check for daemon connections
CREATE OR REPLACE FUNCTION check_daemon_health()
RETURNS TABLE(status TEXT, details JSONB) AS $$
BEGIN
    -- Test daemon connectivity
    IF NOT EXISTS (
        SELECT 1 FROM steadytext_daemon_status() 
        WHERE daemon_running = true
    ) THEN
        RETURN QUERY SELECT 'ERROR', 
            jsonb_build_object('message', 'Daemon not running');
    END IF;
    
    -- Check connection pool health
    IF EXISTS (
        SELECT 1 FROM daemon_pool_status 
        WHERE error_count > 10
    ) THEN
        RETURN QUERY SELECT 'WARNING', 
            jsonb_build_object('message', 'High error rate in connection pool');
    END IF;
    
    RETURN QUERY SELECT 'OK', 
        jsonb_build_object('message', 'Daemon healthy');
END;
$$ LANGUAGE plpgsql;
```

## Security Configuration

### Input Validation

```sql
-- Enable input validation
ALTER SYSTEM SET steadytext.enable_input_validation = true;
ALTER SYSTEM SET steadytext.max_input_length = 10000;
ALTER SYSTEM SET steadytext.max_tokens_limit = 2048;

-- Custom validation rules
CREATE OR REPLACE FUNCTION validate_generation_input(
    prompt TEXT,
    max_tokens INTEGER
)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check prompt length
    IF length(prompt) > 10000 THEN
        RAISE EXCEPTION 'Prompt too long: % characters', length(prompt);
    END IF;
    
    -- Check for injection attempts
    IF prompt ~* '(DROP|DELETE|TRUNCATE|INSERT|UPDATE)\s+(TABLE|DATABASE)' THEN
        RAISE EXCEPTION 'Potentially malicious prompt detected';
    END IF;
    
    -- Validate token limit
    IF max_tokens > 2048 THEN
        RAISE EXCEPTION 'Token limit too high: %', max_tokens;
    END IF;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Apply validation
CREATE OR REPLACE FUNCTION secure_generate(
    prompt TEXT,
    max_tokens INTEGER DEFAULT 512
)
RETURNS TEXT AS $$
BEGIN
    PERFORM validate_generation_input(prompt, max_tokens);
    RETURN steadytext_generate(prompt, max_tokens);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

### Rate Limiting

```sql
-- Enable rate limiting
ALTER SYSTEM SET steadytext.enable_rate_limiting = true;
ALTER SYSTEM SET steadytext.rate_limit_requests_per_minute = 60;
ALTER SYSTEM SET steadytext.rate_limit_tokens_per_hour = 100000;

-- Per-user rate limiting
CREATE TABLE user_rate_limits (
    user_id INTEGER PRIMARY KEY,
    requests_per_minute INTEGER DEFAULT 30,
    tokens_per_hour INTEGER DEFAULT 50000,
    last_reset TIMESTAMP DEFAULT NOW()
);

-- Rate limiting function
CREATE OR REPLACE FUNCTION check_rate_limit(
    p_user_id INTEGER,
    p_tokens INTEGER DEFAULT 512
)
RETURNS BOOLEAN AS $$
DECLARE
    v_requests_count INTEGER;
    v_tokens_count INTEGER;
    v_limit RECORD;
BEGIN
    -- Get user limits
    SELECT * INTO v_limit
    FROM user_rate_limits
    WHERE user_id = p_user_id;
    
    IF NOT FOUND THEN
        INSERT INTO user_rate_limits (user_id) 
        VALUES (p_user_id)
        RETURNING * INTO v_limit;
    END IF;
    
    -- Check requests per minute
    SELECT COUNT(*) INTO v_requests_count
    FROM steadytext_request_log
    WHERE user_id = p_user_id
    AND requested_at > NOW() - INTERVAL '1 minute';
    
    IF v_requests_count >= v_limit.requests_per_minute THEN
        RAISE EXCEPTION 'Rate limit exceeded: too many requests';
    END IF;
    
    -- Check tokens per hour
    SELECT COALESCE(SUM(tokens_used), 0) INTO v_tokens_count
    FROM steadytext_request_log
    WHERE user_id = p_user_id
    AND requested_at > NOW() - INTERVAL '1 hour';
    
    IF v_tokens_count + p_tokens > v_limit.tokens_per_hour THEN
        RAISE EXCEPTION 'Rate limit exceeded: token limit reached';
    END IF;
    
    RETURN true;
END;
$$ LANGUAGE plpgsql;
```

### Access Control

```sql
-- Create roles for different access levels
CREATE ROLE steadytext_reader;
CREATE ROLE steadytext_writer;
CREATE ROLE steadytext_admin;

-- Grant permissions
GRANT EXECUTE ON FUNCTION steadytext_generate(TEXT, INTEGER, BOOLEAN, INTEGER) 
TO steadytext_reader, steadytext_writer;

GRANT EXECUTE ON FUNCTION steadytext_embed(TEXT, BOOLEAN) 
TO steadytext_reader, steadytext_writer;

GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public 
TO steadytext_admin;

-- Row-level security for async queue
ALTER TABLE steadytext_queue ENABLE ROW LEVEL SECURITY;

CREATE POLICY queue_user_policy ON steadytext_queue
    FOR ALL
    USING (user_id = current_user_id())
    WITH CHECK (user_id = current_user_id());

-- Audit logging
CREATE TABLE steadytext_audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    user_id INTEGER,
    function_name TEXT,
    parameters JSONB,
    result_size INTEGER,
    duration_ms INTEGER,
    ip_address INET
);

-- Audit trigger
CREATE OR REPLACE FUNCTION audit_steadytext_usage()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO steadytext_audit_log (
        user_id, function_name, parameters, 
        result_size, duration_ms, ip_address
    )
    VALUES (
        current_user_id(),
        TG_ARGV[0],
        to_jsonb(NEW),
        length(NEW.result),
        EXTRACT(EPOCH FROM (NOW() - NEW.created_at)) * 1000,
        inet_client_addr()
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

## Integration Patterns

### With pgvector

```sql
-- Optimized similarity search with reranking
CREATE OR REPLACE FUNCTION semantic_search_with_rerank(
    query_text TEXT,
    limit_results INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.7
)
RETURNS TABLE(
    doc_id INTEGER,
    content TEXT,
    vector_similarity FLOAT,
    rerank_score FLOAT,
    final_score FLOAT
) AS $$
DECLARE
    query_embedding vector;
BEGIN
    -- Generate query embedding
    query_embedding := steadytext_embed(query_text)::vector;
    
    RETURN QUERY
    WITH candidates AS (
        -- Vector similarity search
        SELECT 
            d.id,
            d.content,
            1 - (d.embedding <=> query_embedding) AS similarity
        FROM documents d
        WHERE 1 - (d.embedding <=> query_embedding) > similarity_threshold
        ORDER BY d.embedding <=> query_embedding
        LIMIT limit_results * 3  -- Get more candidates for reranking
    ),
    reranked AS (
        -- Rerank candidates
        SELECT 
            c.id,
            c.content,
            c.similarity,
            r.score as rerank_score
        FROM candidates c,
        LATERAL steadytext_rerank(
            query_text,
            ARRAY_AGG(c.content) OVER (),
            'semantic search reranking'
        ) r
        WHERE c.content = r.document
    )
    SELECT 
        r.id,
        r.content,
        r.similarity,
        r.rerank_score,
        (0.6 * r.rerank_score + 0.4 * r.similarity) as final_score
    FROM reranked r
    ORDER BY final_score DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;

-- Create specialized indexes
CREATE INDEX idx_documents_embedding_cosine 
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_documents_embedding_l2 
ON documents USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);
```

### With TimescaleDB

```sql
-- Time-series text analysis
CREATE TABLE sensor_logs (
    time TIMESTAMPTZ NOT NULL,
    sensor_id INTEGER,
    log_message TEXT,
    severity TEXT,
    embedding vector(1024)
);

SELECT create_hypertable('sensor_logs', 'time');

-- Continuous aggregate for log summarization
CREATE MATERIALIZED VIEW hourly_log_analysis
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    sensor_id,
    ai_summarize(log_message) AS hourly_summary,
    array_agg(DISTINCT severity) AS severity_levels,
    count(*) AS log_count
FROM sensor_logs
GROUP BY hour, sensor_id
WITH NO DATA;

-- Refresh policy
SELECT add_continuous_aggregate_policy(
    'hourly_log_analysis',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '10 minutes'
);

-- Anomaly detection with embeddings
CREATE OR REPLACE FUNCTION detect_log_anomalies(
    time_window INTERVAL DEFAULT '1 hour'
)
RETURNS TABLE(
    sensor_id INTEGER,
    anomaly_time TIMESTAMPTZ,
    log_message TEXT,
    anomaly_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH recent_logs AS (
        SELECT 
            l.*,
            avg(embedding) OVER (
                PARTITION BY sensor_id 
                ORDER BY time 
                ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING
            ) AS baseline_embedding
        FROM sensor_logs l
        WHERE time > NOW() - time_window
    )
    SELECT 
        rl.sensor_id,
        rl.time,
        rl.log_message,
        (rl.embedding <=> rl.baseline_embedding) AS anomaly_score
    FROM recent_logs rl
    WHERE (rl.embedding <=> rl.baseline_embedding) > 0.5
    ORDER BY anomaly_score DESC;
END;
$$ LANGUAGE plpgsql;
```

### With PostGIS

```sql
-- Location-aware text generation
CREATE OR REPLACE FUNCTION generate_location_description(
    location geometry,
    style TEXT DEFAULT 'descriptive'
)
RETURNS TEXT AS $$
DECLARE
    lat FLOAT;
    lon FLOAT;
    nearby_places TEXT[];
    place_types TEXT[];
BEGIN
    -- Extract coordinates
    lat := ST_Y(location);
    lon := ST_X(location);
    
    -- Find nearby places
    SELECT array_agg(name ORDER BY ST_Distance(geom, location) LIMIT 5)
    INTO nearby_places
    FROM places
    WHERE ST_DWithin(geom, location, 1000);  -- Within 1km
    
    -- Generate description
    RETURN steadytext_generate(
        format('Describe a location at latitude %s, longitude %s. Nearby places: %s. Style: %s',
            lat, lon, array_to_string(nearby_places, ', '), style),
        max_tokens := 200
    );
END;
$$ LANGUAGE plpgsql;

-- Geo-tagged content search
CREATE OR REPLACE FUNCTION search_geo_content(
    query_text TEXT,
    center_location geometry,
    radius_meters FLOAT
)
RETURNS TABLE(
    content_id INTEGER,
    content TEXT,
    location geometry,
    distance FLOAT,
    relevance_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH geo_filtered AS (
        SELECT 
            c.id,
            c.content,
            c.location,
            ST_Distance(c.location, center_location) AS distance
        FROM content c
        WHERE ST_DWithin(c.location, center_location, radius_meters)
    ),
    reranked AS (
        SELECT 
            gf.*,
            r.score
        FROM geo_filtered gf,
        LATERAL steadytext_rerank(
            query_text,
            ARRAY_AGG(gf.content) OVER (),
            'location-based search'
        ) r
        WHERE gf.content = r.document
    )
    SELECT 
        r.id,
        r.content,
        r.location,
        r.distance,
        r.score
    FROM reranked r
    ORDER BY r.score DESC, r.distance ASC;
END;
$$ LANGUAGE plpgsql;
```

## Monitoring and Observability

### Performance Metrics

```sql
-- Comprehensive performance view
CREATE OR REPLACE VIEW steadytext_performance_metrics AS
SELECT 
    -- Function metrics
    f.function_name,
    f.call_count,
    f.total_duration_ms,
    f.avg_duration_ms,
    f.p95_duration_ms,
    f.p99_duration_ms,
    
    -- Cache metrics
    c.cache_hit_rate,
    c.cache_size_mb,
    
    -- Queue metrics
    q.pending_requests,
    q.processing_requests,
    q.failed_requests,
    q.avg_queue_time_seconds,
    
    -- Resource metrics
    r.model_memory_mb,
    r.daemon_connections,
    r.active_workers
FROM (
    SELECT 
        function_name,
        COUNT(*) as call_count,
        SUM(duration_ms) as total_duration_ms,
        AVG(duration_ms) as avg_duration_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_duration_ms,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99_duration_ms
    FROM steadytext_function_stats
    WHERE called_at > NOW() - INTERVAL '1 hour'
    GROUP BY function_name
) f
CROSS JOIN LATERAL (
    SELECT 
        AVG(hit_rate) as cache_hit_rate,
        SUM(cache_size_bytes) / 1024 / 1024 as cache_size_mb
    FROM cache_performance
) c
CROSS JOIN LATERAL (
    SELECT 
        COUNT(*) FILTER (WHERE status = 'pending') as pending_requests,
        COUNT(*) FILTER (WHERE status = 'processing') as processing_requests,
        COUNT(*) FILTER (WHERE status = 'failed') as failed_requests,
        AVG(EXTRACT(EPOCH FROM (started_at - created_at))) as avg_queue_time_seconds
    FROM steadytext_queue
    WHERE created_at > NOW() - INTERVAL '1 hour'
) q
CROSS JOIN LATERAL (
    SELECT 
        SUM(model_size_mb) as model_memory_mb,
        COUNT(*) as daemon_connections,
        COUNT(*) FILTER (WHERE state = 'active') as active_workers
    FROM steadytext_system_status()
) r;

-- Export metrics for monitoring systems
CREATE OR REPLACE FUNCTION export_prometheus_metrics()
RETURNS TEXT AS $$
DECLARE
    metrics TEXT := '';
    rec RECORD;
BEGIN
    -- Function metrics
    FOR rec IN 
        SELECT * FROM steadytext_performance_metrics
    LOOP
        metrics := metrics || format(
            '# HELP steadytext_function_calls_total Total function calls
# TYPE steadytext_function_calls_total counter
steadytext_function_calls_total{function="%s"} %s

# HELP steadytext_function_duration_milliseconds Function duration
# TYPE steadytext_function_duration_milliseconds histogram
steadytext_function_duration_milliseconds{function="%s",quantile="0.95"} %s
steadytext_function_duration_milliseconds{function="%s",quantile="0.99"} %s

# HELP steadytext_cache_hit_rate Cache hit rate
# TYPE steadytext_cache_hit_rate gauge
steadytext_cache_hit_rate %s

# HELP steadytext_queue_depth Current queue depth
# TYPE steadytext_queue_depth gauge
steadytext_queue_depth{status="pending"} %s
steadytext_queue_depth{status="processing"} %s
steadytext_queue_depth{status="failed"} %s
',
            rec.function_name, rec.call_count,
            rec.function_name, rec.p95_duration_ms,
            rec.function_name, rec.p99_duration_ms,
            rec.cache_hit_rate,
            rec.pending_requests,
            rec.processing_requests,
            rec.failed_requests
        );
    END LOOP;
    
    RETURN metrics;
END;
$$ LANGUAGE plpgsql;
```

### Logging and Debugging

```sql
-- Enable detailed logging
ALTER SYSTEM SET steadytext.log_level = 'debug';
ALTER SYSTEM SET steadytext.log_queries = true;
ALTER SYSTEM SET steadytext.log_cache_operations = true;
ALTER SYSTEM SET steadytext.log_daemon_communication = true;

-- Debug function execution
CREATE OR REPLACE FUNCTION debug_generation(
    prompt TEXT,
    max_tokens INTEGER DEFAULT 512
)
RETURNS TABLE(
    step TEXT,
    duration_ms FLOAT,
    details JSONB
) AS $$
DECLARE
    start_time TIMESTAMP;
    step_start TIMESTAMP;
    cache_key TEXT;
    cached_result TEXT;
    model_loaded BOOLEAN;
BEGIN
    start_time := clock_timestamp();
    
    -- Step 1: Validate input
    step_start := clock_timestamp();
    PERFORM validate_generation_input(prompt, max_tokens);
    RETURN QUERY SELECT 
        'Input validation',
        EXTRACT(EPOCH FROM (clock_timestamp() - step_start)) * 1000,
        jsonb_build_object('prompt_length', length(prompt), 'max_tokens', max_tokens);
    
    -- Step 2: Check cache
    step_start := clock_timestamp();
    cache_key := steadytext_cache_key('generation', prompt, max_tokens);
    cached_result := steadytext_cache_get(cache_key);
    RETURN QUERY SELECT 
        'Cache check',
        EXTRACT(EPOCH FROM (clock_timestamp() - step_start)) * 1000,
        jsonb_build_object('cache_hit', cached_result IS NOT NULL, 'cache_key', cache_key);
    
    -- Step 3: Check model status
    step_start := clock_timestamp();
    SELECT model_loaded INTO model_loaded FROM steadytext_model_status();
    RETURN QUERY SELECT 
        'Model check',
        EXTRACT(EPOCH FROM (clock_timestamp() - step_start)) * 1000,
        jsonb_build_object('model_loaded', model_loaded);
    
    -- Step 4: Generation (if needed)
    IF cached_result IS NULL THEN
        step_start := clock_timestamp();
        cached_result := steadytext_generate(prompt, max_tokens);
        RETURN QUERY SELECT 
            'Generation',
            EXTRACT(EPOCH FROM (clock_timestamp() - step_start)) * 1000,
            jsonb_build_object('result_length', length(cached_result));
    END IF;
    
    -- Total time
    RETURN QUERY SELECT 
        'Total',
        EXTRACT(EPOCH FROM (clock_timestamp() - start_time)) * 1000,
        jsonb_build_object('success', true);
END;
$$ LANGUAGE plpgsql;
```

## Deployment Best Practices

### Production Configuration

```sql
-- Production settings
ALTER SYSTEM SET steadytext.enable_daemon = true;
ALTER SYSTEM SET steadytext.daemon_host = 'steadytext-daemon.internal';
ALTER SYSTEM SET steadytext.daemon_port = 5555;
ALTER SYSTEM SET steadytext.daemon_timeout = '10s';
ALTER SYSTEM SET steadytext.enable_fallback = false;  -- No fallback in production
ALTER SYSTEM SET steadytext.enable_monitoring = true;
ALTER SYSTEM SET steadytext.enable_rate_limiting = true;

-- Connection limits
ALTER SYSTEM SET steadytext.max_concurrent_requests = 100;
ALTER SYSTEM SET steadytext.queue_max_size = 10000;
ALTER SYSTEM SET steadytext.worker_pool_size = 8;

-- Memory limits
ALTER SYSTEM SET steadytext.max_memory_per_request = '256MB';
ALTER SYSTEM SET steadytext.cache_memory_target = '2GB';
ALTER SYSTEM SET steadytext.model_memory_limit = '8GB';

-- Apply configuration
SELECT pg_reload_conf();
```

### High Availability Setup

```sql
-- Primary server configuration
ALTER SYSTEM SET steadytext.ha_mode = 'primary';
ALTER SYSTEM SET steadytext.ha_sync_cache = true;
ALTER SYSTEM SET steadytext.ha_sync_interval = '1s';

-- Standby server configuration
ALTER SYSTEM SET steadytext.ha_mode = 'standby';
ALTER SYSTEM SET steadytext.ha_primary_host = 'primary.db.internal';
ALTER SYSTEM SET steadytext.ha_readonly_cache = true;

-- Failover function
CREATE OR REPLACE FUNCTION steadytext_promote_to_primary()
RETURNS void AS $$
BEGIN
    -- Update HA mode
    ALTER SYSTEM SET steadytext.ha_mode = 'primary';
    
    -- Start daemon if not running
    PERFORM steadytext_daemon_start();
    
    -- Warm up cache
    PERFORM steadytext_preload_models();
    
    -- Notify applications
    PERFORM pg_notify('steadytext_failover', 'promoted_to_primary');
    
    -- Reload configuration
    PERFORM pg_reload_conf();
END;
$$ LANGUAGE plpgsql;
```

### Backup and Recovery

```sql
-- Backup cache and queue state
CREATE OR REPLACE FUNCTION backup_steadytext_state(
    backup_path TEXT
)
RETURNS void AS $$
BEGIN
    -- Export cache
    COPY (
        SELECT * FROM steadytext_cache_entries
    ) TO format('%s/cache_backup.csv', backup_path) WITH CSV HEADER;
    
    -- Export queue
    COPY (
        SELECT * FROM steadytext_queue
        WHERE status IN ('pending', 'processing')
    ) TO format('%s/queue_backup.csv', backup_path) WITH CSV HEADER;
    
    -- Export configuration
    COPY (
        SELECT name, setting 
        FROM pg_settings 
        WHERE name LIKE 'steadytext.%'
    ) TO format('%s/config_backup.csv', backup_path) WITH CSV HEADER;
END;
$$ LANGUAGE plpgsql;

-- Restore state
CREATE OR REPLACE FUNCTION restore_steadytext_state(
    backup_path TEXT
)
RETURNS void AS $$
BEGIN
    -- Clear existing state
    TRUNCATE steadytext_cache_entries, steadytext_queue;
    
    -- Restore cache
    EXECUTE format(
        'COPY steadytext_cache_entries FROM %L WITH CSV HEADER',
        format('%s/cache_backup.csv', backup_path)
    );
    
    -- Restore queue
    EXECUTE format(
        'COPY steadytext_queue FROM %L WITH CSV HEADER',
        format('%s/queue_backup.csv', backup_path)
    );
    
    -- Restore configuration
    -- (Applied through ALTER SYSTEM commands)
END;
$$ LANGUAGE plpgsql;
```

## Troubleshooting Guide

### Common Issues and Solutions

```sql
-- Diagnostic function
CREATE OR REPLACE FUNCTION diagnose_steadytext()
RETURNS TABLE(
    check_name TEXT,
    status TEXT,
    details TEXT,
    recommendation TEXT
) AS $$
BEGIN
    -- Check 1: Extension version
    RETURN QUERY
    SELECT 
        'Extension Version',
        'INFO',
        (SELECT extversion FROM pg_extension WHERE extname = 'pg_steadytext'),
        'Keep extension updated';
    
    -- Check 2: Daemon status
    RETURN QUERY
    SELECT 
        'Daemon Status',
        CASE WHEN daemon_running THEN 'OK' ELSE 'ERROR' END,
        CASE WHEN daemon_running 
            THEN 'Daemon running on ' || daemon_host || ':' || daemon_port
            ELSE 'Daemon not running'
        END,
        CASE WHEN daemon_running 
            THEN 'No action needed'
            ELSE 'Start daemon: steadytext daemon start'
        END
    FROM steadytext_daemon_status();
    
    -- Check 3: Model status
    RETURN QUERY
    SELECT 
        'Model Status',
        CASE WHEN model_loaded THEN 'OK' ELSE 'WARNING' END,
        'Models loaded: ' || model_loaded::text,
        CASE WHEN model_loaded 
            THEN 'No action needed'
            ELSE 'Preload models: SELECT steadytext_preload_models()'
        END
    FROM steadytext_model_status();
    
    -- Check 4: Cache health
    RETURN QUERY
    WITH cache_stats AS (
        SELECT 
            SUM(hit_count + miss_count) as total_requests,
            AVG(hit_rate) as avg_hit_rate
        FROM cache_performance
    )
    SELECT 
        'Cache Health',
        CASE 
            WHEN avg_hit_rate > 0.8 THEN 'OK'
            WHEN avg_hit_rate > 0.5 THEN 'WARNING'
            ELSE 'ERROR'
        END,
        format('Hit rate: %.2f%%, Total requests: %s', 
            avg_hit_rate * 100, total_requests),
        CASE 
            WHEN avg_hit_rate < 0.5 
            THEN 'Consider increasing cache size'
            ELSE 'Cache performing well'
        END
    FROM cache_stats;
    
    -- Check 5: Queue health
    RETURN QUERY
    WITH queue_stats AS (
        SELECT 
            COUNT(*) FILTER (WHERE status = 'pending') as pending,
            COUNT(*) FILTER (WHERE status = 'failed') as failed,
            MAX(EXTRACT(EPOCH FROM (NOW() - created_at))) as oldest_pending_seconds
        FROM steadytext_queue
    )
    SELECT 
        'Queue Health',
        CASE 
            WHEN pending > 1000 OR failed > 100 THEN 'ERROR'
            WHEN pending > 500 OR failed > 50 THEN 'WARNING'
            ELSE 'OK'
        END,
        format('Pending: %s, Failed: %s, Oldest: %s seconds', 
            pending, failed, oldest_pending_seconds),
        CASE 
            WHEN pending > 1000 
            THEN 'Scale up workers or reduce load'
            WHEN failed > 100
            THEN 'Check failed requests and retry'
            ELSE 'Queue healthy'
        END
    FROM queue_stats;
    
    -- Check 6: Memory usage
    RETURN QUERY
    SELECT 
        'Memory Usage',
        CASE 
            WHEN used_memory_mb > total_memory_mb * 0.9 THEN 'ERROR'
            WHEN used_memory_mb > total_memory_mb * 0.7 THEN 'WARNING'
            ELSE 'OK'
        END,
        format('Using %.1f GB of %.1f GB', 
            used_memory_mb / 1024.0, total_memory_mb / 1024.0),
        CASE 
            WHEN used_memory_mb > total_memory_mb * 0.9
            THEN 'Reduce cache size or unload models'
            ELSE 'Memory usage acceptable'
        END
    FROM steadytext_memory_usage();
END;
$$ LANGUAGE plpgsql;

-- Run diagnostics
SELECT * FROM diagnose_steadytext();
```

---

**Navigation**: [Main Documentation](postgresql-extension.md) | [Structured Generation](postgresql-extension-structured.md) | [AI Features](postgresql-extension-ai.md) | [Async Functions](postgresql-extension-async.md)
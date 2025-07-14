# PostgreSQL Extension: Troubleshooting Guide

This comprehensive guide helps you diagnose and resolve common issues with the SteadyText PostgreSQL extension.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Problems](#performance-problems)
- [Connection Issues](#connection-issues)
- [Model Loading Problems](#model-loading-problems)
- [Cache Issues](#cache-issues)
- [Async Operation Problems](#async-operation-problems)
- [Upgrade Issues](#upgrade-issues)
- [Debugging Tools](#debugging-tools)

## Installation Issues

### Extension Creation Failed

**Error**: `ERROR: could not access file "$libdir/pg_steadytext": No such file or directory`

**Solution**:
```sql
-- Check if extension files are installed
SELECT * FROM pg_available_extensions WHERE name = 'pg_steadytext';

-- Verify installation path
SHOW dynamic_library_path;

-- For manual installation
sudo cp pg_steadytext.so $(pg_config --pkglibdir)/
sudo cp pg_steadytext--*.sql $(pg_config --sharedir)/extension/
sudo cp pg_steadytext.control $(pg_config --sharedir)/extension/
```

### Python Path Issues

**Error**: `ERROR: Python module steadytext not found`

**Solution**:
```sql
-- Check Python path configuration
SHOW plpython3.python_path;

-- Update Python path if needed
ALTER DATABASE your_db SET plpython3.python_path TO '/opt/steadytext/venv/lib/python3.11/site-packages:$libdir';

-- Restart connection and retry
\c
CREATE EXTENSION pg_steadytext;
```

### Permission Denied

**Error**: `ERROR: permission denied to create extension "pg_steadytext"`

**Solution**:
```sql
-- Grant necessary permissions
GRANT CREATE ON DATABASE your_db TO your_user;

-- Or use superuser
\c - postgres
CREATE EXTENSION pg_steadytext;
GRANT USAGE ON SCHEMA public TO your_user;
```

## Runtime Errors

### Model Not Found

**Error**: `ERROR: Model files not found`

**Symptoms**:
- Functions return NULL
- Error messages about missing GGUF files

**Solution**:
```sql
-- Check model status
SELECT * FROM steadytext_model_status();

-- Force model download
SELECT steadytext_download_models();

-- Verify model cache
SELECT * FROM steadytext_model_cache_info();

-- Check file permissions
-- From shell:
ls -la /opt/steadytext/models/
chmod -R 755 /opt/steadytext/models/
```

### Out of Memory

**Error**: `ERROR: out of memory` or `Cannot allocate memory`

**Solution**:
```sql
-- Check current memory usage
SELECT 
    pg_size_pretty(pg_database_size(current_database())) as db_size,
    pg_size_pretty(sum(pg_total_relation_size(oid))) as total_size
FROM pg_class WHERE relkind = 'r';

-- Adjust PostgreSQL memory settings
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';

-- Reload configuration
SELECT pg_reload_conf();

-- For model memory issues, use environment variables
-- In postgresql.conf:
shared_preload_libraries = 'pg_steadytext'
pg_steadytext.max_model_memory = '4GB'
```

### Function Returns NULL

**Common Causes**:
1. Model not loaded
2. Invalid input
3. Cache corruption
4. Daemon not running

**Diagnostic Steps**:
```sql
-- Step 1: Check basic functionality
SELECT steadytext_version();
SELECT steadytext_health_check();

-- Step 2: Test with simple input
SELECT steadytext_generate('test', 10);

-- Step 3: Check daemon status
SELECT * FROM steadytext_daemon_status();

-- Step 4: Clear cache and retry
SELECT steadytext_clear_cache();
SELECT steadytext_generate('test', 10);

-- Step 5: Check logs
-- From shell:
tail -f /var/log/postgresql/postgresql-*.log
```

## Performance Problems

### Slow Generation

**Symptoms**: Generation takes > 5 seconds

**Solutions**:
```sql
-- 1. Check if daemon is running
SELECT * FROM steadytext_daemon_status();

-- Start daemon if not running
SELECT steadytext_daemon_start();

-- 2. Analyze query performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT steadytext_generate('your prompt', 100);

-- 3. Check cache hit rate
SELECT * FROM steadytext_cache_stats();

-- 4. Optimize batch operations
-- Instead of:
SELECT steadytext_generate(prompt, 100) FROM prompts;

-- Use:
SELECT * FROM steadytext_generate_batch(
    ARRAY(SELECT prompt FROM prompts),
    100
);
```

### High Memory Usage

**Monitor Memory**:
```sql
-- Create monitoring function
CREATE OR REPLACE FUNCTION monitor_steadytext_memory()
RETURNS TABLE(
    metric TEXT,
    value BIGINT,
    human_readable TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'model_memory'::TEXT,
        pg_size_bytes(current_setting('pg_steadytext.model_memory_usage', true))::BIGINT,
        pg_size_pretty(pg_size_bytes(current_setting('pg_steadytext.model_memory_usage', true)))::TEXT
    UNION ALL
    SELECT 
        'cache_memory'::TEXT,
        (SELECT SUM(size_bytes) FROM steadytext_cache_entries)::BIGINT,
        pg_size_pretty((SELECT SUM(size_bytes) FROM steadytext_cache_entries))::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Set memory limits
ALTER SYSTEM SET pg_steadytext.generation_cache_max_size = '100MB';
ALTER SYSTEM SET pg_steadytext.embedding_cache_max_size = '200MB';
```

## Connection Issues

### Daemon Connection Failed

**Error**: `ERROR: Could not connect to SteadyText daemon`

**Solution**:
```sql
-- Check daemon process
-- From shell:
ps aux | grep steadytext-daemon
systemctl status steadytext-daemon

-- Restart daemon
systemctl restart steadytext-daemon

-- Check daemon logs
journalctl -u steadytext-daemon -f

-- Test daemon connectivity
-- From SQL:
SELECT * FROM steadytext_daemon_ping();

-- Check firewall/ports
-- From shell:
sudo ss -tlnp | grep 5557
```

### Connection Pool Exhausted

**Error**: `ERROR: connection pool exhausted`

**Solution**:
```sql
-- Increase connection pool size
ALTER SYSTEM SET pg_steadytext.daemon_pool_size = 20;
SELECT pg_reload_conf();

-- Monitor active connections
CREATE OR REPLACE VIEW steadytext_active_connections AS
SELECT 
    pid,
    usename,
    application_name,
    state,
    query_start,
    state_change,
    query
FROM pg_stat_activity
WHERE query LIKE '%steadytext%'
AND state != 'idle';
```

## Model Loading Problems

### Model Download Failures

**Error**: `ERROR: Failed to download model`

**Debugging**:
```sql
-- Enable verbose logging
SET client_min_messages = DEBUG1;
SELECT steadytext_download_models();

-- Check network connectivity
-- From shell:
curl -I https://huggingface.co/

-- Manual download
cd /opt/steadytext/models
wget https://huggingface.co/ggml-org/gemma-3n-E2B-it-GGUF/resolve/main/gemma-3n-E2B-it-Q8_0.gguf

-- Verify checksums
sha256sum *.gguf
```

### Model Corruption

**Symptoms**: Garbled output, crashes

**Solution**:
```sql
-- Verify model integrity
SELECT * FROM steadytext_verify_models();

-- Clear corrupted models
-- From shell:
rm -f /opt/steadytext/models/*.gguf
rm -f /opt/steadytext/models/*.gguf.*

-- Re-download
SELECT steadytext_download_models(force => true);
```

## Cache Issues

### Cache Corruption

**Symptoms**: Inconsistent results, errors

**Solution**:
```sql
-- Diagnose cache issues
SELECT * FROM steadytext_diagnose_cache();

-- Clear specific cache
SELECT steadytext_clear_cache('generation');
SELECT steadytext_clear_cache('embedding');
SELECT steadytext_clear_cache('reranking');

-- Rebuild cache tables
DROP TABLE IF EXISTS steadytext_cache_entries CASCADE;
SELECT steadytext_init_cache();

-- Monitor cache health
CREATE OR REPLACE FUNCTION cache_health_check()
RETURNS TABLE(
    cache_type TEXT,
    total_entries BIGINT,
    total_size TEXT,
    hit_rate NUMERIC,
    status TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.cache_type,
        COUNT(*)::BIGINT as total_entries,
        pg_size_pretty(SUM(c.size_bytes))::TEXT as total_size,
        COALESCE(
            (c.hits::NUMERIC / NULLIF(c.hits + c.misses, 0) * 100), 
            0
        )::NUMERIC(5,2) as hit_rate,
        CASE 
            WHEN COUNT(*) = 0 THEN 'EMPTY'
            WHEN COALESCE((c.hits::NUMERIC / NULLIF(c.hits + c.misses, 0)), 0) < 0.1 THEN 'POOR'
            WHEN COALESCE((c.hits::NUMERIC / NULLIF(c.hits + c.misses, 0)), 0) < 0.5 THEN 'FAIR'
            ELSE 'GOOD'
        END as status
    FROM steadytext_cache_entries c
    GROUP BY c.cache_type, c.hits, c.misses;
END;
$$ LANGUAGE plpgsql;
```

## Async Operation Problems

### Stuck Async Jobs

**Symptoms**: Jobs remain in 'processing' state

**Solution**:
```sql
-- Find stuck jobs
SELECT * FROM steadytext_queue 
WHERE status = 'processing' 
AND updated_at < NOW() - INTERVAL '5 minutes';

-- Reset stuck jobs
UPDATE steadytext_queue 
SET status = 'pending', 
    worker_id = NULL,
    error_message = 'Reset due to timeout'
WHERE status = 'processing' 
AND updated_at < NOW() - INTERVAL '5 minutes';

-- Check worker status
SELECT * FROM steadytext_workers;

-- Restart workers
SELECT steadytext_restart_workers();
```

### Async Result Not Found

**Error**: `ERROR: Async result not found`

**Solution**:
```sql
-- Check if job exists
SELECT * FROM steadytext_queue WHERE request_id = 'your-uuid';

-- Check retention policy
SHOW pg_steadytext.async_result_retention;

-- Increase retention if needed
ALTER SYSTEM SET pg_steadytext.async_result_retention = '7 days';

-- Create job tracking
CREATE TABLE async_job_log (
    request_id UUID PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status TEXT,
    result_size BIGINT
);
```

## Upgrade Issues

### Extension Upgrade Failed

**Error**: `ERROR: cannot update extension "pg_steadytext"`

**Solution**:
```sql
-- Check current version
SELECT * FROM pg_extension WHERE extname = 'pg_steadytext';

-- List available versions
SELECT * FROM pg_available_extension_versions 
WHERE name = 'pg_steadytext';

-- Backup before upgrade
pg_dump -d your_db -t 'steadytext_*' > steadytext_backup.sql

-- Try update
ALTER EXTENSION pg_steadytext UPDATE TO '1.1.0';

-- If fails, drop and recreate
DROP EXTENSION pg_steadytext CASCADE;
CREATE EXTENSION pg_steadytext VERSION '1.1.0';

-- Restore data if needed
psql -d your_db < steadytext_backup.sql
```

### Post-Upgrade Issues

**Common Problems**:
1. Missing functions
2. Changed signatures
3. Performance regression

**Solution**:
```sql
-- Verify all functions exist
SELECT proname, pg_get_function_identity_arguments(oid) 
FROM pg_proc 
WHERE proname LIKE 'steadytext_%'
ORDER BY proname;

-- Recompile dependent functions
SELECT pg_catalog.pg_recompile_function(oid)
FROM pg_proc
WHERE prosrc LIKE '%steadytext_%';

-- Reset statistics
SELECT pg_stat_reset();
```

## Debugging Tools

### Enable Debug Logging

```sql
-- Session level
SET log_min_messages = 'DEBUG1';
SET log_statement = 'all';

-- Database level
ALTER DATABASE your_db SET log_min_messages = 'DEBUG1';

-- Extension specific
SET pg_steadytext.debug = on;
```

### Performance Profiling

```sql
-- Create profiling function
CREATE OR REPLACE FUNCTION profile_steadytext_operation(
    operation TEXT,
    input_text TEXT
) RETURNS TABLE(
    step TEXT,
    duration INTERVAL,
    memory_used BIGINT
) AS $$
DECLARE
    start_time TIMESTAMP;
    step_time TIMESTAMP;
    start_mem BIGINT;
    step_mem BIGINT;
BEGIN
    start_time := clock_timestamp();
    SELECT pg_backend_memory_contexts_total_bytes() INTO start_mem;
    
    -- Profile each step
    step_time := clock_timestamp();
    PERFORM steadytext_generate(input_text, 10);
    
    RETURN QUERY
    SELECT 
        'total_time'::TEXT,
        clock_timestamp() - start_time,
        pg_backend_memory_contexts_total_bytes() - start_mem;
END;
$$ LANGUAGE plpgsql;
```

### Health Check Dashboard

```sql
-- Comprehensive health check
CREATE OR REPLACE VIEW steadytext_health_dashboard AS
SELECT 
    'Models' as component,
    CASE 
        WHEN EXISTS (SELECT 1 FROM steadytext_model_status() WHERE loaded = true)
        THEN 'OK' ELSE 'ERROR' 
    END as status,
    (SELECT COUNT(*) FROM steadytext_model_status() WHERE loaded = true)::TEXT || ' loaded' as details
UNION ALL
SELECT 
    'Daemon',
    CASE 
        WHEN (SELECT running FROM steadytext_daemon_status())
        THEN 'OK' ELSE 'ERROR'
    END,
    COALESCE((SELECT status FROM steadytext_daemon_status()), 'Not running')
UNION ALL
SELECT 
    'Cache',
    'OK',
    (SELECT COUNT(*)::TEXT || ' entries' FROM steadytext_cache_entries)
UNION ALL
SELECT 
    'Async Queue',
    CASE 
        WHEN EXISTS (SELECT 1 FROM steadytext_queue WHERE status = 'failed')
        THEN 'WARNING' ELSE 'OK'
    END,
    (SELECT COUNT(*)::TEXT || ' pending' FROM steadytext_queue WHERE status = 'pending');
```

## Getting Help

### Collect Diagnostic Information

```sql
-- Run comprehensive diagnostics
CREATE OR REPLACE FUNCTION steadytext_diagnostics()
RETURNS TEXT AS $$
DECLARE
    report TEXT;
BEGIN
    report := E'SteadyText Diagnostics Report\n';
    report := report || E'========================\n\n';
    
    -- Version info
    report := report || 'Version: ' || steadytext_version() || E'\n';
    report := report || 'PostgreSQL: ' || version() || E'\n\n';
    
    -- Add more diagnostic queries...
    
    RETURN report;
END;
$$ LANGUAGE plpgsql;

-- Generate report
\o steadytext_diagnostics.txt
SELECT steadytext_diagnostics();
\o
```

### Contact Support

When reporting issues, include:
1. Diagnostic report
2. PostgreSQL logs
3. Extension version
4. Error messages
5. Steps to reproduce

## Related Documentation

- [PostgreSQL Extension Overview](postgresql-extension.md)
- [Advanced Features](postgresql-extension-advanced.md)
- [Performance Tuning](examples/performance-tuning.md)
- [Installation Guide](deployment/production.md)
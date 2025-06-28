-- pg_steadytext--1.0.0.sql
-- Initial schema for pg_steadytext extension

-- AIDEV-NOTE: This SQL script creates the core schema for the pg_steadytext extension
-- It mirrors SteadyText's cache structure and adds PostgreSQL-specific features

-- Complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION pg_steadytext" to load this file. \quit

-- Create schema for internal objects (optional, can use public)
-- CREATE SCHEMA IF NOT EXISTS _steadytext;

-- AIDEV-SECTION: CORE_TABLE_DEFINITIONS
-- Cache table that mirrors and extends SteadyText's SQLite cache
CREATE TABLE steadytext_cache (
    id SERIAL PRIMARY KEY,
    cache_key TEXT UNIQUE NOT NULL,  -- Matches SteadyText's cache key generation
    prompt TEXT NOT NULL,
    response TEXT,
    embedding vector(1024),  -- For embedding cache using pgvector
    
    -- Frecency statistics (synced with SteadyText's cache)
    access_count INT DEFAULT 1,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- SteadyText integration metadata
    steadytext_cache_hit BOOLEAN DEFAULT FALSE,  -- Whether this came from ST's cache
    model_name TEXT NOT NULL DEFAULT 'qwen3-1.7b',  -- Model used (supports switching)
    model_size TEXT CHECK (model_size IN ('small', 'medium', 'large')),
    thinking_mode BOOLEAN DEFAULT FALSE,  -- Whether thinking mode was enabled
    eos_string TEXT,  -- Custom end-of-sequence string if used
    
    -- Generation parameters
    generation_params JSONB,  -- temperature, max_tokens, seed, etc.
    response_size INT,
    generation_time_ms INT  -- Time taken to generate (if not cached)
    
    -- AIDEV-NOTE: frecency_score removed - calculated via view instead
    -- Previously used GENERATED column with NOW() which is not immutable
);

-- Create indexes for performance
CREATE INDEX idx_steadytext_cache_key ON steadytext_cache USING hash(cache_key);
CREATE INDEX idx_steadytext_cache_last_accessed ON steadytext_cache(last_accessed);
CREATE INDEX idx_steadytext_cache_access_count ON steadytext_cache(access_count);

-- Request queue for async operations with priority and resource management
CREATE TABLE steadytext_queue (
    id SERIAL PRIMARY KEY,
    request_id UUID DEFAULT gen_random_uuid(),
    request_type TEXT CHECK (request_type IN ('generate', 'embed', 'batch_embed')),
    
    -- Request data
    prompt TEXT,  -- For single requests
    prompts TEXT[],  -- For batch requests
    params JSONB,  -- Model params, thinking_mode, etc.
    
    -- Priority and scheduling
    priority INT DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    user_id TEXT,  -- For rate limiting per user
    session_id TEXT,  -- For request grouping
    
    -- Status tracking
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    result TEXT,
    results TEXT[],  -- For batch results
    embedding vector(1024),
    embeddings vector(1024)[],  -- For batch embeddings
    error TEXT,
    
    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    processing_time_ms INT,
    
    -- Resource tracking
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    daemon_endpoint TEXT  -- Which daemon instance handled this
);

CREATE INDEX idx_steadytext_queue_status_priority_created ON steadytext_queue(status, priority DESC, created_at);
CREATE INDEX idx_steadytext_queue_request_id ON steadytext_queue(request_id);
CREATE INDEX idx_steadytext_queue_user_created ON steadytext_queue(user_id, created_at DESC);
CREATE INDEX idx_steadytext_queue_session ON steadytext_queue(session_id);

-- Configuration storage
CREATE TABLE steadytext_config (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by TEXT DEFAULT current_user
);

-- Insert default configuration
INSERT INTO steadytext_config (key, value, description) VALUES
    ('daemon_host', '"localhost"', 'SteadyText daemon host'),
    ('daemon_port', '5555', 'SteadyText daemon port'),
    ('cache_enabled', 'true', 'Enable caching'),
    ('max_cache_entries', '1000', 'Maximum cache entries'),
    ('max_cache_size_mb', '500', 'Maximum cache size in MB'),
    ('default_max_tokens', '512', 'Default max tokens for generation'),
    ('daemon_auto_start', 'true', 'Auto-start daemon if not running');

-- Daemon health monitoring
CREATE TABLE steadytext_daemon_health (
    daemon_id TEXT PRIMARY KEY DEFAULT 'default',
    endpoint TEXT NOT NULL,
    last_heartbeat TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'unknown' CHECK (status IN ('healthy', 'unhealthy', 'starting', 'stopping', 'unknown')),
    version TEXT,
    models_loaded TEXT[],
    memory_usage_mb INT,
    active_connections INT DEFAULT 0,
    total_requests BIGINT DEFAULT 0,
    error_count INT DEFAULT 0,
    avg_response_time_ms INT
);

-- Insert default daemon entry
INSERT INTO steadytext_daemon_health (daemon_id, endpoint, status)
VALUES ('default', 'tcp://localhost:5555', 'unknown');

-- Rate limiting per user
CREATE TABLE steadytext_rate_limits (
    user_id TEXT PRIMARY KEY,
    requests_per_minute INT DEFAULT 60,
    requests_per_hour INT DEFAULT 1000,
    requests_per_day INT DEFAULT 10000,
    current_minute_count INT DEFAULT 0,
    current_hour_count INT DEFAULT 0,
    current_day_count INT DEFAULT 0,
    last_reset_minute TIMESTAMPTZ DEFAULT NOW(),
    last_reset_hour TIMESTAMPTZ DEFAULT NOW(),
    last_reset_day TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit log for security and debugging
CREATE TABLE steadytext_audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id TEXT DEFAULT current_user,
    action TEXT NOT NULL,
    request_id UUID,
    details JSONB,
    ip_address INET,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

CREATE INDEX idx_steadytext_audit_timestamp ON steadytext_audit_log(timestamp DESC);
CREATE INDEX idx_steadytext_audit_user ON steadytext_audit_log(user_id, timestamp DESC);

-- AIDEV-SECTION: VIEWS
-- View for calculating frecency scores dynamically
CREATE VIEW steadytext_cache_with_frecency AS
SELECT *,
    -- Calculate frecency score dynamically
    access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0) AS frecency_score
FROM steadytext_cache;

-- AIDEV-NOTE: This view replaces the GENERATED column which couldn't use NOW()
-- The frecency score decays exponentially based on time since last access

-- AIDEV-SECTION: PYTHON_INTEGRATION
-- AIDEV-NOTE: Python integration layer path setup
-- This allows PostgreSQL to find our Python modules
DO $$
DECLARE
    current_path TEXT;
    new_path TEXT;
BEGIN
    -- Get current Python path if it exists
    BEGIN
        current_path := current_setting('plpython3.python_path', true);
    EXCEPTION
        WHEN undefined_object THEN
            current_path := NULL;
        WHEN OTHERS THEN
            current_path := NULL;
    END;
    
    -- Build new path
    IF current_path IS NOT NULL AND current_path != '' THEN
        new_path := '$libdir/pg_steadytext/python:' || current_path;
    ELSE
        new_path := '$libdir/pg_steadytext/python';
    END IF;
    
    -- Set the Python path
    EXECUTE format('ALTER DATABASE %I SET plpython3.python_path TO %L',
        current_database(),
        new_path
    );
END;
$$;

-- Create Python function container
CREATE OR REPLACE FUNCTION _steadytext_init_python()
RETURNS void
LANGUAGE plpython3u
AS $$
# AIDEV-NOTE: Initialize Python environment for pg_steadytext
import sys
import os

# Try to import our modules
try:
    import daemon_connector
    import cache_manager
    GD['steadytext_initialized'] = True
    plpy.notice("pg_steadytext Python environment initialized successfully")
except ImportError as e:
    GD['steadytext_initialized'] = False
    plpy.warning(f"Failed to initialize pg_steadytext Python environment: {e}")
$$;

-- Initialize Python environment
SELECT _steadytext_init_python();

-- AIDEV-SECTION: CORE_FUNCTIONS
-- Core function: Synchronous text generation
CREATE OR REPLACE FUNCTION steadytext_generate(
    prompt TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    thinking_mode BOOLEAN DEFAULT FALSE
)
RETURNS TEXT
LANGUAGE plpython3u
AS $$
# AIDEV-NOTE: Main text generation function that integrates with SteadyText daemon
import json
import hashlib

# Get configuration
plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])

# Get default max_tokens if not provided
if max_tokens is None:
    rv = plpy.execute(plan, ["default_max_tokens"])
    max_tokens = json.loads(rv[0]["value"]) if rv else 512

# Check if we should use cache
if use_cache:
    # Generate cache key consistent with SteadyText format
    # AIDEV-NOTE: Use SHA256 to match SteadyText's cache key format
    params = {"max_new_tokens": max_tokens, "thinking_mode": thinking_mode}
    cache_key_input = f"{prompt}|{json.dumps(params, sort_keys=True)}"
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()
    
    # Try to get from cache first
    cache_plan = plpy.prepare("""
        UPDATE steadytext_cache 
        SET access_count = access_count + 1,
            last_accessed = NOW()
        WHERE cache_key = $1
        RETURNING response
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        return cache_result[0]["response"]

# If not in cache or cache disabled, generate new response
try:
    # Import here to avoid issues if not initialized
    from daemon_connector import SteadyTextConnector
    
    # Get daemon configuration
    host_rv = plpy.execute(plan, ["daemon_host"])
    port_rv = plpy.execute(plan, ["daemon_port"])
    
    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Connect to daemon and generate
    connector = SteadyTextConnector(host, port)
    response = connector.generate(prompt, max_tokens=max_tokens, thinking_mode=thinking_mode)
    
    # Store in cache if enabled
    if use_cache and response:
        insert_plan = plpy.prepare("""
            INSERT INTO steadytext_cache 
            (cache_key, prompt, response, generation_params, thinking_mode)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (cache_key) DO UPDATE
            SET response = EXCLUDED.response,
                access_count = steadytext_cache.access_count + 1,
                last_accessed = NOW()
        """, ["text", "text", "text", "jsonb", "bool"])
        
        params = {"max_tokens": max_tokens}
        plpy.execute(insert_plan, [cache_key, prompt, response, json.dumps(params), thinking_mode])
    
    return response
    
except Exception as e:
    plpy.warning(f"Failed to generate text: {e}")
    # Return a deterministic fallback
    return f"[SteadyText Error: {str(e)}]"
$$;

-- Core function: Synchronous embedding generation
CREATE OR REPLACE FUNCTION steadytext_embed(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT TRUE
)
RETURNS vector(1024)
LANGUAGE plpython3u
AS $$
# AIDEV-NOTE: Embedding generation function that integrates with SteadyText daemon
import json
import numpy as np
import hashlib

# Check cache first if enabled
if use_cache:
    # Generate cache key for embedding
    # AIDEV-NOTE: Use SHA256 to match SteadyText's cache key format
    cache_key_input = f"embed:{text_input}|{{}}"  # Empty params for embeddings
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()
    
    cache_plan = plpy.prepare("""
        UPDATE steadytext_cache 
        SET access_count = access_count + 1,
            last_accessed = NOW()
        WHERE cache_key = $1
        RETURNING embedding
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["embedding"]:
        return cache_result[0]["embedding"]

# Generate new embedding
try:
    from daemon_connector import SteadyTextConnector
    
    # Get daemon configuration
    plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])
    host_rv = plpy.execute(plan, ["daemon_host"])
    port_rv = plpy.execute(plan, ["daemon_port"])
    
    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Connect and generate embedding
    connector = SteadyTextConnector(host, port)
    embedding = connector.embed(text_input)
    
    # Convert numpy array to list for storage
    embedding_list = embedding.tolist()
    
    # Store in cache if enabled
    if use_cache and embedding_list:
        insert_plan = plpy.prepare("""
            INSERT INTO steadytext_cache 
            (cache_key, prompt, embedding)
            VALUES ($1, $2, $3::vector)
            ON CONFLICT (cache_key) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                access_count = steadytext_cache.access_count + 1,
                last_accessed = NOW()
        """, ["text", "text", "text"])
        
        # Convert to PostgreSQL vector format
        vector_str = '[' + ','.join(map(str, embedding_list)) + ']'
        plpy.execute(insert_plan, [cache_key, text_input, vector_str])
    
    return embedding_list
    
except Exception as e:
    plpy.warning(f"Failed to generate embedding: {e}")
    # Return zero vector as fallback
    return [0.0] * 1024
$$;

-- AIDEV-SECTION: DAEMON_MANAGEMENT_FUNCTIONS
-- Daemon management functions
CREATE OR REPLACE FUNCTION steadytext_daemon_start()
RETURNS BOOLEAN
LANGUAGE plpython3u
AS $$
# AIDEV-NOTE: Start the SteadyText daemon if not already running
import subprocess
import time
import json

try:
    # Get daemon configuration
    plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])
    host_rv = plpy.execute(plan, ["daemon_host"])
    port_rv = plpy.execute(plan, ["daemon_port"])
    
    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Check if daemon is already running by trying to start it
    # SteadyText daemon start command is idempotent
    try:
        result = subprocess.run(['st', 'daemon', 'start'], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # Update health status
            health_plan = plpy.prepare("""
                UPDATE steadytext_daemon_health 
                SET status = 'healthy',
                    last_heartbeat = NOW()
                WHERE daemon_id = 'default'
            """)
            plpy.execute(health_plan)
            return True
        else:
            plpy.warning(f"Failed to start daemon: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        plpy.warning("Timeout starting daemon")
        return False
        
except Exception as e:
    plpy.error(f"Error in daemon start: {e}")
    return False
$$;

-- Get daemon status
CREATE OR REPLACE FUNCTION steadytext_daemon_status()
RETURNS TABLE(
    daemon_id TEXT,
    status TEXT,
    endpoint TEXT,
    last_heartbeat TIMESTAMPTZ,
    uptime_seconds INT
)
LANGUAGE plpython3u
AS $$
# AIDEV-NOTE: Check SteadyText daemon health status
try:
    from daemon_connector import SteadyTextConnector
    import json
    
    # Get configuration
    plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])
    host_rv = plpy.execute(plan, ["daemon_host"])
    port_rv = plpy.execute(plan, ["daemon_port"])
    
    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Try to connect
    try:
        connector = SteadyTextConnector(host, port)
        # If we can create connector, daemon is healthy
        status = 'healthy'
    except:
        status = 'unhealthy'
    
    # Update and return health status
    update_plan = plpy.prepare("""
        UPDATE steadytext_daemon_health 
        SET status = $1,
            last_heartbeat = CASE WHEN $1 = 'healthy' THEN NOW() ELSE last_heartbeat END
        WHERE daemon_id = 'default'
        RETURNING daemon_id, status, endpoint, last_heartbeat,
                  EXTRACT(EPOCH FROM (NOW() - last_heartbeat))::INT as uptime_seconds
    """, ["text"])
    
    result = plpy.execute(update_plan, [status])
    return result
    
except Exception as e:
    plpy.warning(f"Error checking daemon status: {e}")
    # Return current status from table
    select_plan = plpy.prepare("""
        SELECT daemon_id, status, endpoint, last_heartbeat,
               EXTRACT(EPOCH FROM (NOW() - last_heartbeat))::INT as uptime_seconds
        FROM steadytext_daemon_health
        WHERE daemon_id = 'default'
    """)
    return plpy.execute(select_plan)
$$;

-- Stop daemon
CREATE OR REPLACE FUNCTION steadytext_daemon_stop()
RETURNS BOOLEAN
LANGUAGE plpython3u
AS $$
# AIDEV-NOTE: Stop the SteadyText daemon gracefully
import subprocess
import json

try:
    # Stop daemon using CLI
    result = subprocess.run(['st', 'daemon', 'stop'], capture_output=True, text=True)
    
    if result.returncode == 0:
        # Update health status
        health_plan = plpy.prepare("""
            UPDATE steadytext_daemon_health 
            SET status = 'stopping',
                last_heartbeat = NOW()
            WHERE daemon_id = 'default'
        """)
        plpy.execute(health_plan)
        
        return True
    else:
        plpy.warning(f"Failed to stop daemon: {result.stderr}")
        return False
        
except Exception as e:
    plpy.error(f"Error stopping daemon: {e}")
    return False
$$;

-- AIDEV-SECTION: CACHE_MANAGEMENT_FUNCTIONS
-- Cache management functions
CREATE OR REPLACE FUNCTION steadytext_cache_stats()
RETURNS TABLE(
    total_entries BIGINT,
    total_size_mb FLOAT,
    cache_hit_rate FLOAT,
    avg_access_count FLOAT,
    oldest_entry TIMESTAMPTZ,
    newest_entry TIMESTAMPTZ
)
LANGUAGE sql
AS $$
    SELECT 
        COUNT(*)::BIGINT as total_entries,
        COALESCE(SUM(pg_column_size(response) + pg_column_size(embedding)) / 1024.0 / 1024.0, 0)::FLOAT as total_size_mb,
        COALESCE(SUM(CASE WHEN access_count > 1 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0), 0)::FLOAT as cache_hit_rate,
        COALESCE(AVG(access_count), 0)::FLOAT as avg_access_count,
        MIN(created_at) as oldest_entry,
        MAX(created_at) as newest_entry
    FROM steadytext_cache;
$$;

-- Clear cache
CREATE OR REPLACE FUNCTION steadytext_cache_clear()
RETURNS BIGINT
LANGUAGE sql
AS $$
    WITH deleted AS (
        DELETE FROM steadytext_cache
        RETURNING *
    )
    SELECT COUNT(*) FROM deleted;
$$;

-- Get extension version
CREATE OR REPLACE FUNCTION steadytext_version()
RETURNS TEXT
LANGUAGE sql
AS $$
    SELECT '1.0.0'::TEXT;
$$;

-- AIDEV-SECTION: CONFIGURATION_FUNCTIONS
-- Configuration helper functions
CREATE OR REPLACE FUNCTION steadytext_config_set(key TEXT, value TEXT)
RETURNS VOID
LANGUAGE sql
AS $$
    INSERT INTO steadytext_config (key, value)
    VALUES (key, to_jsonb(value))
    ON CONFLICT (key) DO UPDATE
    SET value = to_jsonb(value),
        updated_at = NOW(),
        updated_by = current_user;
$$;

CREATE OR REPLACE FUNCTION steadytext_config_get(key TEXT)
RETURNS TEXT
LANGUAGE sql
AS $$
    SELECT value::text FROM steadytext_config WHERE key = $1;
$$;

-- Grant appropriate permissions
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO PUBLIC;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO PUBLIC;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO PUBLIC;

-- AIDEV-NOTE: This completes the base schema for pg_steadytext v1.0.0
-- 
-- AIDEV-SECTION: CHANGES_MADE_IN_REVIEW
-- The following issues were identified and fixed during review:
-- 1. Added missing columns: model_size, eos_string, response_size, daemon_endpoint
-- 2. Enhanced queue table with priority, user_id, session_id, batch support
-- 3. Added rate limiting and audit logging tables  
-- 4. Fixed cache key generation to use SHA256 and match SteadyText format
-- 5. Fixed daemon integration to use proper SteadyText API methods
-- 6. Added proper indexes for performance
--
-- AIDEV-TODO: Future versions should add:
-- - Async processing functions (steadytext_generate_async, steadytext_get_result)
-- - Streaming generation function (steadytext_generate_stream) 
-- - Batch operations (steadytext_embed_batch)
-- - FAISS index operations (steadytext_index_create, steadytext_index_search)
-- - Worker management functions
-- - Enhanced security and rate limiting functions
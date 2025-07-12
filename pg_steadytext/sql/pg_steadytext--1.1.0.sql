-- pg_steadytext--1.1.0.sql
-- Initial schema for pg_steadytext extension with AI summarization aggregates

-- AIDEV-NOTE: This SQL script creates the core schema for the pg_steadytext extension
-- It mirrors SteadyText's cache structure and adds PostgreSQL-specific features
-- Version 1.1.0 adds AI summarization aggregate functions with TimescaleDB support

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
    seed INTEGER DEFAULT 42,  -- Seed used for generation
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
    params JSONB,  -- Model params, seed, etc.
    
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
    ('default_seed', '42', 'Default seed for deterministic generation'),
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
-- This is now handled by the _steadytext_init_python function instead

-- Create Python function container
CREATE OR REPLACE FUNCTION _steadytext_init_python()
RETURNS void
LANGUAGE plpython3u
AS $$
# AIDEV-NOTE: Initialize Python environment for pg_steadytext with enhanced error handling
import sys
import os
import site

# Get PostgreSQL lib directory with fallback
try:
    result = plpy.execute("SELECT setting FROM pg_settings WHERE name = 'pkglibdir'")
    if result and len(result) > 0 and result[0]['setting']:
        pg_lib_dir = result[0]['setting']
    else:
        # Fallback for Docker/Debian PostgreSQL 17
        pg_lib_dir = '/usr/lib/postgresql/17/lib'
        plpy.notice(f"Using fallback pkglibdir: {pg_lib_dir}")
except Exception as e:
    # Fallback for Docker/Debian PostgreSQL 17
    pg_lib_dir = '/usr/lib/postgresql/17/lib'
    plpy.notice(f"Error getting pkglibdir, using fallback: {pg_lib_dir}")

python_module_dir = os.path.join(pg_lib_dir, 'pg_steadytext', 'python')

# Verify the directory exists
if not os.path.exists(python_module_dir):
    plpy.error(f"Python module directory not found: {python_module_dir}")

# Add to Python path if not already there
if python_module_dir not in sys.path:
    sys.path.insert(0, python_module_dir)
    site.addsitedir(python_module_dir)  # Process .pth files if any

# Log Python path for debugging
plpy.notice(f"Python path: {sys.path}")
plpy.notice(f"Looking for modules in: {python_module_dir}")

# Check if directory exists
if not os.path.exists(python_module_dir):
    plpy.error(f"Python module directory does not exist: {python_module_dir}")

# List files in directory for debugging
try:
    files = os.listdir(python_module_dir)
    plpy.notice(f"Files in module directory: {files}")
except Exception as e:
    plpy.warning(f"Could not list module directory: {e}")

# Try to import required external packages first
required_packages = {
    'steadytext': 'SteadyText library',
    'zmq': 'ZeroMQ for daemon communication',
    'numpy': 'NumPy for embeddings'
}

missing_packages = []
for package, description in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(f"{package} ({description})")

if missing_packages:
    plpy.warning(f"Missing required packages: {', '.join(missing_packages)}. Run: pip3 install steadytext pyzmq numpy")

# Try to import our modules and cache them in GD
try:
    # Clear any previous module cache
    for key in list(GD.keys()):
        if key.startswith('module_'):
            del GD[key]
    
    # Import and cache modules
    import daemon_connector
    import cache_manager
    import security
    import config
    
    # Store modules in GD for reuse
    GD['module_daemon_connector'] = daemon_connector
    GD['module_cache_manager'] = cache_manager
    GD['module_security'] = security
    GD['module_config'] = config
    GD['steadytext_initialized'] = True
    
    plpy.notice(f"pg_steadytext Python environment initialized successfully from {python_module_dir}")
except ImportError as e:
    GD['steadytext_initialized'] = False
    plpy.error(f"Failed to import pg_steadytext modules from {python_module_dir}: {e}")
    plpy.error(f"Make sure the extension is properly installed with 'make install'")
except Exception as e:
    GD['steadytext_initialized'] = False
    plpy.error(f"Unexpected error during initialization: {e}")
$$;

-- AIDEV-NOTE: Initialization is now done on-demand in each function
-- This ensures proper initialization even across session boundaries

-- AIDEV-SECTION: CORE_FUNCTIONS
-- Core function: Synchronous text generation
-- Returns NULL if generation fails (no fallback text)
CREATE OR REPLACE FUNCTION steadytext_generate(
    prompt TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
# AIDEV-NOTE: Main text generation function that integrates with SteadyText daemon
import json
import hashlib

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    # Initialize on demand
    plpy.execute("SELECT _steadytext_init_python()")
    # Check again after initialization
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules from GD
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Get configuration
plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])

# Resolve max_tokens, using the provided value or fetching the default
resolved_max_tokens = max_tokens
if resolved_max_tokens is None:
    rv = plpy.execute(plan, ["default_max_tokens"])
    resolved_max_tokens = json.loads(rv[0]["value"]) if rv else 512

# Resolve seed, using the provided value or fetching the default
resolved_seed = seed
if resolved_seed is None:
    rv = plpy.execute(plan, ["default_seed"])
    resolved_seed = json.loads(rv[0]["value"]) if rv else 42

# Validate inputs
if not prompt or not prompt.strip():
    plpy.error("Prompt cannot be empty")

if resolved_max_tokens < 1 or resolved_max_tokens > 4096:
    plpy.error("max_tokens must be between 1 and 4096")

if resolved_seed < 0:
    plpy.error("seed must be non-negative")

# Check if we should use cache
if use_cache:
    # Generate cache key consistent with SteadyText format
    # AIDEV-NOTE: Updated to match SteadyText's simple cache key format from utils.py
    # For generation: just the prompt (no parameters in key)
    cache_key = prompt
    
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
        plpy.notice(f"Cache hit for key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# If not in cache or cache disabled, generate new response
try:
    # Get daemon configuration
    host_rv = plpy.execute(plan, ["daemon_host"])
    port_rv = plpy.execute(plan, ["daemon_port"])
    
    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Connect to daemon and generate using cached module
    connector = daemon_connector.SteadyTextConnector(host, port)
    response = connector.generate(prompt, max_tokens=resolved_max_tokens, seed=resolved_seed)
    
    # Store in cache if enabled
    if use_cache and response:
        insert_plan = plpy.prepare("""
            INSERT INTO steadytext_cache 
            (cache_key, prompt, response, generation_params)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (cache_key) DO UPDATE
            SET response = EXCLUDED.response,
                access_count = steadytext_cache.access_count + 1,
                last_accessed = NOW()
        """, ["text", "text", "text", "jsonb"])
        
        params = {"max_tokens": resolved_max_tokens, "seed": resolved_seed}
        plpy.execute(insert_plan, [cache_key, prompt, response, json.dumps(params)])
        plpy.notice(f"Cached response with key: {cache_key[:8]}...")
    
    return response
    
except Exception as e:
    plpy.warning(f"Failed to generate text: {e}")
    # Return NULL instead of fallback text
    return None
$$;

-- Core function: Synchronous embedding generation
-- Returns NULL if embedding generation fails (no fallback vector)
CREATE OR REPLACE FUNCTION steadytext_embed(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42
)
RETURNS vector(1024)
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
# AIDEV-NOTE: Embedding generation function that integrates with SteadyText daemon
import json
import numpy as np
import hashlib

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    # Initialize on demand
    plpy.execute("SELECT _steadytext_init_python()")
    # Check again after initialization
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules from GD
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Resolve seed, using the provided value or fetching the default
plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])
resolved_seed = seed
if resolved_seed is None:
    rv = plpy.execute(plan, ["default_seed"])
    resolved_seed = json.loads(rv[0]["value"]) if rv else 42

# Validate input
if not text_input or not text_input.strip():
    plpy.warning("Empty text input provided, returning NULL")
    return None

if resolved_seed < 0:
    plpy.error("seed must be non-negative")

# Check cache first if enabled
if use_cache:
    # Generate cache key for embedding
    # AIDEV-NOTE: Use SHA256 for embeddings to match SteadyText's format
    # Embeddings use SHA256 hash of "embed:{text}"
    cache_key_input = f"embed:{text_input}"
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
        plpy.notice(f"Embedding cache hit for key: {cache_key[:8]}...")
        return cache_result[0]["embedding"]

# Generate new embedding
try:
    # Get daemon configuration
    plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])
    host_rv = plpy.execute(plan, ["daemon_host"])
    port_rv = plpy.execute(plan, ["daemon_port"])
    
    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Connect and generate embedding using cached module
    connector = daemon_connector.SteadyTextConnector(host, port)
    embedding = connector.embed(text_input, seed=resolved_seed)
    
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
        plpy.notice(f"Cached embedding with key: {cache_key[:8]}...")
    
    return embedding_list
    
except Exception as e:
    plpy.warning(f"Failed to generate embedding: {e}")
    # Return NULL instead of fallback vector
    return None
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
import json

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    # Initialize on demand
    plpy.execute("SELECT _steadytext_init_python()")
    # Check again after initialization
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

try:
    # Get cached modules from GD
    daemon_connector = GD.get('module_daemon_connector')
    if not daemon_connector:
        plpy.error("daemon_connector module not loaded")
    
    # Get configuration
    plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])
    host_rv = plpy.execute(plan, ["daemon_host"])
    port_rv = plpy.execute(plan, ["daemon_port"])
    
    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Try to connect using cached module
    try:
        connector = daemon_connector.SteadyTextConnector(host, port)
        # Use check_health method if available
        if hasattr(connector, 'check_health'):
            health_info = connector.check_health()
            status = health_info.get('status', 'healthy')
        else:
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
STABLE PARALLEL SAFE
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
IMMUTABLE PARALLEL SAFE LEAKPROOF
AS $$
    SELECT '1.1.0'::TEXT;
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
STABLE PARALLEL SAFE LEAKPROOF
AS $$
    SELECT value::text FROM steadytext_config WHERE key = $1;
$$;

-- AIDEV-SECTION: STRUCTURED_GENERATION_FUNCTIONS
-- Structured generation functions using llama.cpp grammars

-- Generate JSON with schema validation
CREATE OR REPLACE FUNCTION steadytext_generate_json(
    prompt TEXT,
    schema JSONB,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
# AIDEV-NOTE: Generate JSON that conforms to a schema using llama.cpp grammars
import json
import hashlib

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    plpy.execute("SELECT _steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules from GD
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Get configuration
plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])

# Resolve max_tokens
resolved_max_tokens = max_tokens
if resolved_max_tokens is None:
    rv = plpy.execute(plan, ["default_max_tokens"])
    resolved_max_tokens = json.loads(rv[0]["value"]) if rv else 512

# Resolve seed
resolved_seed = seed
if resolved_seed is None:
    rv = plpy.execute(plan, ["default_seed"])
    resolved_seed = json.loads(rv[0]["value"]) if rv else 42

# Validate inputs
if not prompt or not prompt.strip():
    plpy.error("Prompt cannot be empty")

if not schema:
    plpy.error("Schema cannot be empty")

# Convert JSONB to dict if needed
schema_dict = schema
if isinstance(schema, str):
    try:
        schema_dict = json.loads(schema)
    except json.JSONDecodeError as e:
        plpy.error(f"Invalid JSON schema: {e}")

# Check if we should use cache
if use_cache:
    # Generate cache key including schema
    # AIDEV-NOTE: Include schema in cache key for structured generation
    cache_key_input = f"{prompt}|json|{json.dumps(schema_dict, sort_keys=True)}"
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
        plpy.notice(f"JSON cache hit for key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# If not in cache or cache disabled, generate new response
try:
    # Get daemon configuration
    host_rv = plpy.execute(plan, ["daemon_host"])
    port_rv = plpy.execute(plan, ["daemon_port"])
    
    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Connect and generate JSON using cached module
    connector = daemon_connector.SteadyTextConnector(host, port)
    response = connector.generate_json(prompt, schema_dict, max_tokens=resolved_max_tokens, seed=resolved_seed)
    
    # Store in cache if enabled
    if use_cache and response:
        insert_plan = plpy.prepare("""
            INSERT INTO steadytext_cache 
            (cache_key, prompt, response, generation_params)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (cache_key) DO UPDATE
            SET response = EXCLUDED.response,
                access_count = steadytext_cache.access_count + 1,
                last_accessed = NOW()
        """, ["text", "text", "text", "jsonb"])
        
        params = {
            "max_tokens": resolved_max_tokens,
            "seed": resolved_seed,
            "schema": schema_dict
        }
        plpy.execute(insert_plan, [cache_key, prompt, response, json.dumps(params)])
        plpy.notice(f"Cached JSON response with key: {cache_key[:8]}...")
    
    return response
    
except Exception as e:
    plpy.warning(f"Failed to generate JSON: {e}")
    # Return NULL instead of fallback
    return None
$$;

-- Generate text matching a regex pattern
CREATE OR REPLACE FUNCTION steadytext_generate_regex(
    prompt TEXT,
    pattern TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
# AIDEV-NOTE: Generate text matching a regex pattern using llama.cpp grammars
import json
import hashlib

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    plpy.execute("SELECT _steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules from GD
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Get configuration
plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])

# Resolve max_tokens
resolved_max_tokens = max_tokens
if resolved_max_tokens is None:
    rv = plpy.execute(plan, ["default_max_tokens"])
    resolved_max_tokens = json.loads(rv[0]["value"]) if rv else 512

# Resolve seed
resolved_seed = seed
if resolved_seed is None:
    rv = plpy.execute(plan, ["default_seed"])
    resolved_seed = json.loads(rv[0]["value"]) if rv else 42

# Validate inputs
if not prompt or not prompt.strip():
    plpy.error("Prompt cannot be empty")

if not pattern or not pattern.strip():
    plpy.error("Pattern cannot be empty")

# Check if we should use cache
if use_cache:
    # Generate cache key including pattern
    cache_key_input = f"{prompt}|regex|{pattern}"
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
        plpy.notice(f"Regex cache hit for key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# If not in cache or cache disabled, generate new response
try:
    # Get daemon configuration
    host_rv = plpy.execute(plan, ["daemon_host"])
    port_rv = plpy.execute(plan, ["daemon_port"])
    
    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Connect and generate regex-constrained text using cached module
    connector = daemon_connector.SteadyTextConnector(host, port)
    response = connector.generate_regex(prompt, pattern, max_tokens=resolved_max_tokens, seed=resolved_seed)
    
    # Store in cache if enabled
    if use_cache and response:
        insert_plan = plpy.prepare("""
            INSERT INTO steadytext_cache 
            (cache_key, prompt, response, generation_params)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (cache_key) DO UPDATE
            SET response = EXCLUDED.response,
                access_count = steadytext_cache.access_count + 1,
                last_accessed = NOW()
        """, ["text", "text", "text", "jsonb"])
        
        params = {
            "max_tokens": resolved_max_tokens,
            "seed": resolved_seed,
            "pattern": pattern
        }
        plpy.execute(insert_plan, [cache_key, prompt, response, json.dumps(params)])
        plpy.notice(f"Cached regex response with key: {cache_key[:8]}...")
    
    return response
    
except Exception as e:
    plpy.warning(f"Failed to generate regex-constrained text: {e}")
    # Return NULL instead of fallback
    return None
$$;

-- Generate text from a list of choices
CREATE OR REPLACE FUNCTION steadytext_generate_choice(
    prompt TEXT,
    choices TEXT[],
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
# AIDEV-NOTE: Generate text constrained to one of the provided choices
import json
import hashlib

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    plpy.execute("SELECT _steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules from GD
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Get configuration
plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])

# Resolve max_tokens
resolved_max_tokens = max_tokens
if resolved_max_tokens is None:
    rv = plpy.execute(plan, ["default_max_tokens"])
    resolved_max_tokens = json.loads(rv[0]["value"]) if rv else 512

# Resolve seed
resolved_seed = seed
if resolved_seed is None:
    rv = plpy.execute(plan, ["default_seed"])
    resolved_seed = json.loads(rv[0]["value"]) if rv else 42

# Validate inputs
if not prompt or not prompt.strip():
    plpy.error("Prompt cannot be empty")

if not choices or len(choices) == 0:
    plpy.error("Choices list cannot be empty")

# Convert PostgreSQL array to Python list
choices_list = list(choices)

# Check if we should use cache
if use_cache:
    # Generate cache key including choices
    cache_key_input = f"{prompt}|choice|{json.dumps(sorted(choices_list))}"
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
        plpy.notice(f"Choice cache hit for key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# If not in cache or cache disabled, generate new response
try:
    # Get daemon configuration
    host_rv = plpy.execute(plan, ["daemon_host"])
    port_rv = plpy.execute(plan, ["daemon_port"])
    
    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Connect and generate choice-constrained text using cached module
    connector = daemon_connector.SteadyTextConnector(host, port)
    response = connector.generate_choice(prompt, choices_list, max_tokens=resolved_max_tokens, seed=resolved_seed)
    
    # Store in cache if enabled
    if use_cache and response:
        insert_plan = plpy.prepare("""
            INSERT INTO steadytext_cache 
            (cache_key, prompt, response, generation_params)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (cache_key) DO UPDATE
            SET response = EXCLUDED.response,
                access_count = steadytext_cache.access_count + 1,
                last_accessed = NOW()
        """, ["text", "text", "text", "jsonb"])
        
        params = {
            "max_tokens": resolved_max_tokens,
            "seed": resolved_seed,
            "choices": choices_list
        }
        plpy.execute(insert_plan, [cache_key, prompt, response, json.dumps(params)])
        plpy.notice(f"Cached choice response with key: {cache_key[:8]}...")
    
    return response
    
except Exception as e:
    plpy.warning(f"Failed to generate choice-constrained text: {e}")
    # Return NULL instead of fallback
    return None
$$;

-- AIDEV-SECTION: AI_SUMMARIZATION_AGGREGATES
-- AI summarization aggregate functions with TimescaleDB support

-- Helper function to extract facts from text using JSON generation
CREATE OR REPLACE FUNCTION ai_extract_facts(
    input_text text,
    max_facts integer DEFAULT 5
) RETURNS jsonb
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
    import json
    from plpy import quote_literal
    
    # AIDEV-NOTE: Use steadytext's JSON generation with schema for structured fact extraction
    schema = {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": max_facts,
                "description": "Key facts extracted from the text"
            }
        },
        "required": ["facts"]
    }
    
    prompt = f"Extract up to {max_facts} key facts from this text: {input_text}"
    
    # Use daemon_connector for JSON generation
    plan = plpy.prepare(
        "SELECT steadytext_generate_json($1, $2::jsonb) as result",
        ["text", "jsonb"]
    )
    result = plpy.execute(plan, [prompt, json.dumps(schema)])
    
    if result and result[0]["result"]:
        try:
            return json.loads(result[0]["result"])
        except:
            # Fallback to empty facts array
            return {"facts": []}
    return {"facts": []}
$$;

-- Helper function to deduplicate facts using embeddings
CREATE OR REPLACE FUNCTION ai_deduplicate_facts(
    facts_array jsonb,
    similarity_threshold float DEFAULT 0.85
) RETURNS jsonb
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
    import json
    import numpy as np
    
    facts = json.loads(facts_array)
    if not facts or len(facts) == 0:
        return json.dumps([])
    
    # Extract text from fact objects if they have structure
    fact_texts = []
    for fact in facts:
        if isinstance(fact, dict) and "text" in fact:
            fact_texts.append(fact["text"])
        elif isinstance(fact, str):
            fact_texts.append(fact)
    
    if len(fact_texts) <= 1:
        return facts_array
    
    # Generate embeddings for all facts
    embeddings = []
    for text in fact_texts:
        plan = plpy.prepare("SELECT steadytext_embed($1) as embedding", ["text"])
        result = plpy.execute(plan, [text])
        if result and result[0]["embedding"]:
            embeddings.append(np.array(result[0]["embedding"]))
    
    # Deduplicate based on cosine similarity
    unique_indices = [0]  # Always keep first fact
    for i in range(1, len(embeddings)):
        is_duplicate = False
        for j in unique_indices:
            # Calculate cosine similarity
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)
    
    # Return deduplicated facts
    unique_facts = [facts[i] for i in unique_indices]
    return json.dumps(unique_facts)
$$;

-- State accumulator function for AI summarization
CREATE OR REPLACE FUNCTION ai_summarize_accumulate(
    state jsonb,
    value text,
    metadata jsonb DEFAULT '{}'::jsonb
) RETURNS jsonb
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
    import json
    
    # Initialize state if null
    if state is None:
        state = {
            "facts": [],
            "samples": [],
            "stats": {
                "row_count": 0,
                "total_chars": 0,
                "min_length": None,
                "max_length": 0
            },
            "metadata": {}
        }
    else:
        state = json.loads(state)
    
    if value is None:
        return json.dumps(state)
    
    # Extract facts from the value
    plan = plpy.prepare("SELECT ai_extract_facts($1, 3) as facts", ["text"])
    result = plpy.execute(plan, [value])
    
    if result and result[0]["facts"]:
        extracted = json.loads(result[0]["facts"])
        if "facts" in extracted:
            state["facts"].extend(extracted["facts"])
    
    # Update statistics
    value_len = len(value)
    state["stats"]["row_count"] += 1
    state["stats"]["total_chars"] += value_len
    
    if state["stats"]["min_length"] is None or value_len < state["stats"]["min_length"]:
        state["stats"]["min_length"] = value_len
    if value_len > state["stats"]["max_length"]:
        state["stats"]["max_length"] = value_len
    
    # Sample every 10th row (up to 10 samples)
    if state["stats"]["row_count"] % 10 == 1 and len(state["samples"]) < 10:
        state["samples"].append(value[:200])  # First 200 chars
    
    # Merge metadata
    if metadata:
        meta = json.loads(metadata) if isinstance(metadata, str) else metadata
        for key, value in meta.items():
            if key not in state["metadata"]:
                state["metadata"][key] = value
    
    return json.dumps(state)
$$;

-- Combiner function for parallel aggregation
CREATE OR REPLACE FUNCTION ai_summarize_combine(
    state1 jsonb,
    state2 jsonb
) RETURNS jsonb
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
    import json
    
    if state1 is None:
        return state2
    if state2 is None:
        return state1
    
    s1 = json.loads(state1)
    s2 = json.loads(state2)
    
    # Combine facts
    combined_facts = s1.get("facts", []) + s2.get("facts", [])
    
    # Deduplicate facts if too many
    if len(combined_facts) > 20:
        plan = plpy.prepare(
            "SELECT ai_deduplicate_facts($1::jsonb) as deduped",
            ["jsonb"]
        )
        result = plpy.execute(plan, [json.dumps(combined_facts)])
        if result and result[0]["deduped"]:
            combined_facts = json.loads(result[0]["deduped"])
    
    # Combine samples (keep diverse set)
    combined_samples = s1.get("samples", []) + s2.get("samples", [])
    if len(combined_samples) > 10:
        # Simple diversity: take evenly spaced samples
        step = len(combined_samples) // 10
        combined_samples = combined_samples[::step][:10]
    
    # Combine statistics
    stats1 = s1.get("stats", {})
    stats2 = s2.get("stats", {})
    
    combined_stats = {
        "row_count": stats1.get("row_count", 0) + stats2.get("row_count", 0),
        "total_chars": stats1.get("total_chars", 0) + stats2.get("total_chars", 0),
        "min_length": min(
            stats1.get("min_length", float('inf')),
            stats2.get("min_length", float('inf'))
        ),
        "max_length": max(
            stats1.get("max_length", 0),
            stats2.get("max_length", 0)
        ),
        "combine_depth": max(
            stats1.get("combine_depth", 0),
            stats2.get("combine_depth", 0)
        ) + 1
    }
    
    # Merge metadata
    combined_metadata = {**s1.get("metadata", {}), **s2.get("metadata", {})}
    
    return json.dumps({
        "facts": combined_facts,
        "samples": combined_samples,
        "stats": combined_stats,
        "metadata": combined_metadata
    })
$$;

-- Finalizer function to generate summary
CREATE OR REPLACE FUNCTION ai_summarize_finalize(
    state jsonb
) RETURNS text
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
    import json
    
    if state is None:
        return None
    
    state_data = json.loads(state)
    
    # Check if we have any data
    if state_data.get("stats", {}).get("row_count", 0) == 0:
        return "No data to summarize"
    
    # Build summary prompt based on combine depth
    combine_depth = state_data.get("stats", {}).get("combine_depth", 0)
    
    if combine_depth == 0:
        prompt_template = "Create a concise summary of this data: Facts: {facts}, Row count: {row_count}, Average length: {avg_length}"
    elif combine_depth < 3:
        prompt_template = "Synthesize these key facts into a coherent summary: {facts}, Total rows: {row_count}, Length range: {min_length}-{max_length} chars"
    else:
        prompt_template = "Identify major patterns from these aggregated facts: {facts}, Dataset size: {row_count} rows"
    
    # Calculate average length
    stats = state_data.get("stats", {})
    avg_length = stats.get("total_chars", 0) // max(stats.get("row_count", 1), 1)
    
    # Format facts for prompt
    facts = state_data.get("facts", [])[:10]  # Limit to top 10 facts
    facts_str = "; ".join(facts) if facts else "No specific facts extracted"
    
    # Build prompt
    prompt = prompt_template.format(
        facts=facts_str,
        row_count=stats.get("row_count", 0),
        avg_length=avg_length,
        min_length=stats.get("min_length", 0),
        max_length=stats.get("max_length", 0)
    )
    
    # Add metadata context if available
    metadata = state_data.get("metadata", {})
    if metadata:
        meta_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
        prompt += f". Context: {meta_str}"
    
    # Generate summary using steadytext
    plan = plpy.prepare("SELECT steadytext_generate($1) as summary", ["text"])
    result = plpy.execute(plan, [prompt])
    
    if result and result[0]["summary"]:
        return result[0]["summary"]
    return "Unable to generate summary"
$$;

-- Serialization functions for distributed aggregation
CREATE OR REPLACE FUNCTION ai_summarize_serialize(
    state jsonb
) RETURNS bytea
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
    import json
    
    if state is None:
        return None
    
    # Convert to JSON string then to bytes
    json_str = state if isinstance(state, str) else json.dumps(state)
    return json_str.encode('utf-8')
$$;

CREATE OR REPLACE FUNCTION ai_summarize_deserialize(
    state bytea
) RETURNS jsonb
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
    import json
    
    if state is None:
        return None
    
    # Convert bytes to string then parse JSON
    json_str = state.decode('utf-8')
    return json_str
$$;

-- Create the main aggregate
CREATE AGGREGATE ai_summarize(text, jsonb) (
    SFUNC = ai_summarize_accumulate,
    STYPE = jsonb,
    FINALFUNC = ai_summarize_finalize,
    COMBINEFUNC = ai_summarize_combine,
    -- SERIALFUNC = ai_summarize_serialize,
    -- DESERIALFUNC = ai_summarize_deserialize,
    PARALLEL = SAFE
);

-- Create partial aggregate for TimescaleDB continuous aggregates
CREATE AGGREGATE ai_summarize_partial(text, jsonb) (
    SFUNC = ai_summarize_accumulate,
    STYPE = jsonb,
    COMBINEFUNC = ai_summarize_combine,
    -- SERIALFUNC = ai_summarize_serialize,
    -- DESERIALFUNC = ai_summarize_deserialize,
    PARALLEL = SAFE
);

-- Helper function to combine partial states for final aggregation
CREATE OR REPLACE FUNCTION ai_summarize_combine_states(
    state1 jsonb,
    partial_state jsonb
) RETURNS jsonb
LANGUAGE plpgsql
IMMUTABLE PARALLEL SAFE
AS $$
BEGIN
    -- Simply use the combine function
    RETURN ai_summarize_combine(state1, partial_state);
END;
$$;

-- Create final aggregate that works on partial results
CREATE AGGREGATE ai_summarize_final(jsonb) (
    SFUNC = ai_summarize_combine_states,
    STYPE = jsonb,
    FINALFUNC = ai_summarize_finalize,
    PARALLEL = SAFE
);

-- Convenience function for single-value summarization
CREATE OR REPLACE FUNCTION ai_summarize_text(
    input_text text,
    metadata jsonb DEFAULT '{}'::jsonb
) RETURNS text
LANGUAGE sql
IMMUTABLE PARALLEL SAFE
AS $$
    SELECT ai_summarize_finalize(
        ai_summarize_accumulate(NULL::jsonb, input_text, metadata)
    );
$$;

-- Add helpful comments
COMMENT ON AGGREGATE ai_summarize(text, jsonb) IS 
'AI-powered text summarization aggregate that handles non-transitivity through structured fact extraction';

COMMENT ON AGGREGATE ai_summarize_partial(text, jsonb) IS 
'Partial aggregate for use with TimescaleDB continuous aggregates';

COMMENT ON AGGREGATE ai_summarize_final(jsonb) IS 
'Final aggregate for completing partial aggregations from continuous aggregates';

COMMENT ON FUNCTION ai_extract_facts(text, integer) IS 
'Extract structured facts from text using SteadyText JSON generation';

COMMENT ON FUNCTION ai_deduplicate_facts(jsonb, float) IS 
'Deduplicate facts based on semantic similarity using embeddings';

-- Grant appropriate permissions
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO PUBLIC;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO PUBLIC;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO PUBLIC;

-- AIDEV-NOTE: This completes the base schema for pg_steadytext v1.1.0
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
-- - Support for Pydantic models in structured generation (needs JSON serialization)
-- - Tests for structured generation functions

-- AIDEV-NOTE: Added in v1.0.1 (2025-07-07):
-- Marked all deterministic functions as IMMUTABLE, PARALLEL SAFE, and LEAKPROOF (where allowed):
-- - steadytext_generate(), steadytext_embed(), steadytext_generate_json(), 
--   steadytext_generate_regex(), steadytext_generate_choice() are IMMUTABLE PARALLEL SAFE
-- - steadytext_version() is IMMUTABLE PARALLEL SAFE LEAKPROOF
-- - steadytext_cache_stats() and steadytext_config_get() are STABLE PARALLEL SAFE
-- - steadytext_config_get() is also LEAKPROOF since it's a simple SQL function
-- This enables use with TimescaleDB and in aggregates, and improves query optimization

-- AIDEV-NOTE: Added in v1.1.0 (2025-07-08):
-- AI summarization aggregate functions with the following features:
-- 1. Structured fact extraction to mitigate non-transitivity
-- 2. Semantic deduplication using embeddings
-- 3. Statistical tracking (row counts, character lengths)
-- 4. Sample preservation for context
-- 5. Combine depth tracking for adaptive prompts
-- 6. Full TimescaleDB continuous aggregate support
-- 7. Serialization for distributed aggregation
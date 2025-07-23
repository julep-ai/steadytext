-- pg_steadytext--1.2.0.sql
-- Initial schema for pg_steadytext extension with AI summarization aggregates

-- AIDEV-NOTE: This SQL script creates the core schema for the pg_steadytext extension
-- It mirrors SteadyText's cache structure and adds PostgreSQL-specific features
-- Version 1.2.0 includes improved AI summarization aggregate functions with better error handling

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

# Save pg_lib_dir in GD for use in error messages
GD['pg_lib_dir'] = pg_lib_dir

# Verify the directory exists
if not os.path.exists(python_module_dir):
    plpy.error(f"Python module directory not found: {python_module_dir}")

# Add to Python path if not already there
if python_module_dir not in sys.path:
    sys.path.insert(0, python_module_dir)
    site.addsitedir(python_module_dir)  # Process .pth files if any

# AIDEV-NOTE: Add site-packages directory for locally installed Python packages
site_packages_dir = os.path.join(pg_lib_dir, 'pg_steadytext', 'site-packages')
if os.path.exists(site_packages_dir) and site_packages_dir not in sys.path:
    sys.path.insert(0, site_packages_dir)
    site.addsitedir(site_packages_dir)
    plpy.notice(f"Added site-packages to path: {site_packages_dir}")

# AIDEV-NOTE: Add common Python package locations
# These are common locations where pip might install packages
common_paths = [
    # User site-packages
    site.getusersitepackages(),
    # System-wide site-packages
    '/usr/local/lib/python3.10/dist-packages',
    '/usr/local/lib/python3.11/dist-packages',
    '/usr/local/lib/python3.12/dist-packages',
    '/usr/lib/python3/dist-packages',
    # Virtual environment (if activated)
    os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'),
]

for path in common_paths:
    if path and os.path.exists(path) and path not in sys.path:
        sys.path.append(path)

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
    pg_lib_dir = GD.get('pg_lib_dir', '/usr/lib/postgresql/17/lib')
    site_packages_dir = os.path.join(pg_lib_dir, 'pg_steadytext', 'site-packages')
    
    error_msg = f"""
=================================================================
Missing required Python packages: {', '.join(missing_packages)}

The pg_steadytext extension requires these packages to function.

To fix this, run ONE of the following commands:

1. Install via make (recommended):
   cd /path/to/pg_steadytext
   sudo make install

2. Install to PostgreSQL's site-packages:
   sudo pip3 install --target={site_packages_dir} steadytext pyzmq numpy

3. Install system-wide:
   sudo pip3 install steadytext pyzmq numpy

4. Install to user directory:
   pip3 install --user steadytext pyzmq numpy

After installation, restart PostgreSQL and try again.
=================================================================
"""
    plpy.error(error_msg)

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
    pg_lib_dir = GD.get('pg_lib_dir', '/usr/lib/postgresql/17/lib')
    site_packages_dir = os.path.join(pg_lib_dir, 'pg_steadytext', 'site-packages')
    
    error_msg = f"""
=================================================================
Failed to import pg_steadytext modules from {python_module_dir}

Error: {str(e)}

This usually means the extension files are installed but Python
module files are missing or there's an import error.

To fix this:

1. Ensure all Python module files are present in:
   {python_module_dir}

2. Check that required packages are installed:
   sudo pip3 install --target={site_packages_dir} steadytext pyzmq numpy

3. Or reinstall the extension:
   cd /path/to/pg_steadytext
   sudo make install

After fixing, restart PostgreSQL and try again.
=================================================================
"""
    plpy.error(error_msg)
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
    SELECT '1.4.0'::TEXT;
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
    
    # Validate inputs
    if not input_text or not input_text.strip():
        return json.dumps({"facts": []})
    
    if max_facts <= 0 or max_facts > 50:
        plpy.error("max_facts must be between 1 and 50")
    
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
        except json.JSONDecodeError as e:
            plpy.warning(f"Failed to parse JSON response: {e}")
            return json.dumps({"facts": []})
        except Exception as e:
            plpy.warning(f"Unexpected error parsing response: {e}")
            return json.dumps({"facts": []})
    return json.dumps({"facts": []})
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
    
    # Validate similarity threshold
    if similarity_threshold < 0.0 or similarity_threshold > 1.0:
        plpy.error("similarity_threshold must be between 0.0 and 1.0")
    
    try:
        facts = json.loads(facts_array)
    except (json.JSONDecodeError, TypeError) as e:
        plpy.warning(f"Invalid JSON input: {e}")
        return json.dumps([])
    
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
    # AIDEV-NOTE: Consider batching embedding generation for better performance
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
            # Calculate cosine similarity with zero-norm protection
            norm_i = np.linalg.norm(embeddings[i])
            norm_j = np.linalg.norm(embeddings[j])
            
            if norm_i == 0 or norm_j == 0:
                # Treat zero-norm vectors as non-duplicate
                similarity = 0.0
            else:
                similarity = np.dot(embeddings[i], embeddings[j]) / (norm_i * norm_j)
            
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
        try:
            state = json.loads(state)
        except (json.JSONDecodeError, TypeError) as e:
            plpy.error(f"Invalid state JSON: {e}")
    
    if value is None:
        return json.dumps(state)
    
    # Extract facts from the value
    plan = plpy.prepare("SELECT ai_extract_facts($1, 3) as facts", ["text"])
    result = plpy.execute(plan, [value])
    
    if result and result[0]["facts"]:
        try:
            extracted = json.loads(result[0]["facts"])
            if "facts" in extracted:
                state["facts"].extend(extracted["facts"])
        except (json.JSONDecodeError, TypeError):
            # Skip if fact extraction failed
            pass
    
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
        try:
            meta = json.loads(metadata) if isinstance(metadata, str) else metadata
            for key, value in meta.items():
                if key not in state["metadata"]:
                    state["metadata"][key] = value
        except (json.JSONDecodeError, TypeError):
            # Skip invalid metadata
            pass
    
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
    
    try:
        s1 = json.loads(state1)
    except (json.JSONDecodeError, TypeError):
        return state2
    
    try:
        s2 = json.loads(state2)
    except (json.JSONDecodeError, TypeError):
        return state1
    
    # Combine facts
    combined_facts = s1.get("facts", []) + s2.get("facts", [])
    
    # Deduplicate facts if too many
    # AIDEV-NOTE: Threshold of 20 may need tuning based on usage patterns
    if len(combined_facts) > 20:
        plan = plpy.prepare(
            "SELECT ai_deduplicate_facts($1::jsonb) as deduped",
            ["jsonb"]
        )
        result = plpy.execute(plan, [json.dumps(combined_facts)])
        if result and result[0]["deduped"]:
            try:
                combined_facts = json.loads(result[0]["deduped"])
            except (json.JSONDecodeError, TypeError):
                # Keep original if deduplication failed
                pass
    
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
    
    try:
        state_data = json.loads(state)
    except (json.JSONDecodeError, TypeError):
        return "Unable to parse aggregation state"
    
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
    
    # Calculate average length with division by zero protection
    stats = state_data.get("stats", {})
    row_count = stats.get("row_count", 0)
    if row_count > 0:
        avg_length = stats.get("total_chars", 0) // row_count
    else:
        avg_length = 0
    
    # Format facts for prompt
    facts = state_data.get("facts", [])[:10]  # Limit to top 10 facts
    facts_str = "; ".join(facts) if facts else "No specific facts extracted"
    
    # Build prompt
    prompt = prompt_template.format(
        facts=facts_str,
        row_count=row_count,
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

-- AIDEV-NOTE: Serialization functions removed because they are only allowed for STYPE = internal.
-- Since we use STYPE = jsonb, PostgreSQL handles serialization automatically for parallel processing.

-- Create the main aggregate
CREATE AGGREGATE ai_summarize(text, jsonb) (
    SFUNC = ai_summarize_accumulate,
    STYPE = jsonb,
    FINALFUNC = ai_summarize_finalize,
    COMBINEFUNC = ai_summarize_combine,
    PARALLEL = SAFE
);

-- Create partial aggregate for TimescaleDB continuous aggregates
CREATE AGGREGATE ai_summarize_partial(text, jsonb) (
    SFUNC = ai_summarize_accumulate,
    STYPE = jsonb,
    COMBINEFUNC = ai_summarize_combine,
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

-- AIDEV-NOTE: This completes the base schema for pg_steadytext v1.2.0
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

-- AIDEV-NOTE: Updated in v1.2.0 (2025-07-08):
-- Improved AI summarization aggregate functions with:
-- 1. Better error handling throughout all functions
-- 2. Input validation for all parameters
-- 3. Protection against division by zero in cosine similarity calculations
-- 4. Specific exception handling instead of bare except clauses
-- 5. Proper handling of invalid JSON inputs
-- 6. Zero-norm vector protection in similarity calculations
-- 7. Graceful fallback when parsing fails
-- 8. FIXED: Removed serialization functions to resolve "serialization functions may be 
--    specified only when the aggregate transition data type is internal" error[38;5;238m───────┬────────────────────────────────────────────────────────────────────────[0m
       [38;5;238m│ [0mFile: [1mpg_steadytext--1.4.0.sql.append[0m
[38;5;238m───────┼────────────────────────────────────────────────────────────────────────[0m
[38;5;238m   1[0m   [38;5;238m│[0m 
[38;5;238m   2[0m   [38;5;238m│[0m [38;5;246m-- =====================================================[0m
[38;5;238m   3[0m   [38;5;238m│[0m [38;5;246m-- Migration from 1.2.0 to 1.3.0: Reranking Functions[0m
[38;5;238m   4[0m   [38;5;238m│[0m [38;5;246m-- =====================================================[0m
[38;5;238m   5[0m   [38;5;238m│[0m 
[38;5;238m   6[0m   [38;5;238m│[0m [38;5;246m-- Basic rerank function returning documents with scores[0m
[38;5;238m   7[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_rerank([0m
[38;5;238m   8[0m   [38;5;238m│[0m [38;5;246m    query text,[0m
[38;5;238m   9[0m   [38;5;238m│[0m [38;5;246m    documents text[],[0m
[38;5;238m  10[0m   [38;5;238m│[0m [38;5;246m    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',[0m
[38;5;238m  11[0m   [38;5;238m│[0m [38;5;246m    return_scores boolean DEFAULT true,[0m
[38;5;238m  12[0m   [38;5;238m│[0m [38;5;246m    seed integer DEFAULT 42[0m
[38;5;238m  13[0m   [38;5;238m│[0m [38;5;246m) RETURNS TABLE(document text, score float)[0m
[38;5;238m  14[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m  15[0m   [38;5;238m│[0m [38;5;246m    import json[0m
[38;5;238m  16[0m   [38;5;238m│[0m [38;5;246m    import logging[0m
[38;5;238m  17[0m   [38;5;238m│[0m [38;5;246m    from typing import List, Tuple[0m
[38;5;238m  18[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m  19[0m   [38;5;238m│[0m [38;5;246m    # Configure logging[0m
[38;5;238m  20[0m   [38;5;238m│[0m [38;5;246m    logging.basicConfig(level=logging.WARNING)[0m
[38;5;238m  21[0m   [38;5;238m│[0m [38;5;246m    logger = logging.getLogger(__name__)[0m
[38;5;238m  22[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m  23[0m   [38;5;238m│[0m [38;5;246m    # Validate inputs[0m
[38;5;238m  24[0m   [38;5;238m│[0m [38;5;246m    if not query:[0m
[38;5;238m  25[0m   [38;5;238m│[0m [38;5;246m        plpy.error("Query cannot be empty")[0m
[38;5;238m  26[0m   [38;5;238m│[0m [38;5;246m        [0m
[38;5;238m  27[0m   [38;5;238m│[0m [38;5;246m    if not documents or len(documents) == 0:[0m
[38;5;238m  28[0m   [38;5;238m│[0m [38;5;246m        return [][0m
[38;5;238m  29[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m  30[0m   [38;5;238m│[0m [38;5;246m    # Import daemon connector[0m
[38;5;238m  31[0m   [38;5;238m│[0m [38;5;246m    try:[0m
[38;5;238m  32[0m   [38;5;238m│[0m [38;5;246m        from daemon_connector import SteadyTextConnector[0m
[38;5;238m  33[0m   [38;5;238m│[0m [38;5;246m        connector = SteadyTextConnector()[0m
[38;5;238m  34[0m   [38;5;238m│[0m [38;5;246m    except Exception as e:[0m
[38;5;238m  35[0m   [38;5;238m│[0m [38;5;246m        logger.error(f"Failed to initialize SteadyText connector: {e}")[0m
[38;5;238m  36[0m   [38;5;238m│[0m [38;5;246m        # Return empty result on error[0m
[38;5;238m  37[0m   [38;5;238m│[0m [38;5;246m        return [][0m
[38;5;238m  38[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m  39[0m   [38;5;238m│[0m [38;5;246m    try:[0m
[38;5;238m  40[0m   [38;5;238m│[0m [38;5;246m        # Call rerank with scores always enabled for PostgreSQL[0m
[38;5;238m  41[0m   [38;5;238m│[0m [38;5;246m        results = connector.rerank([0m
[38;5;238m  42[0m   [38;5;238m│[0m [38;5;246m            query=query,[0m
[38;5;238m  43[0m   [38;5;238m│[0m [38;5;246m            documents=list(documents),  # Convert from PostgreSQL array[0m
[38;5;238m  44[0m   [38;5;238m│[0m [38;5;246m            task=task,[0m
[38;5;238m  45[0m   [38;5;238m│[0m [38;5;246m            return_scores=True,  # Always get scores for PostgreSQL[0m
[38;5;238m  46[0m   [38;5;238m│[0m [38;5;246m            seed=seed[0m
[38;5;238m  47[0m   [38;5;238m│[0m [38;5;246m        )[0m
[38;5;238m  48[0m   [38;5;238m│[0m [38;5;246m        [0m
[38;5;238m  49[0m   [38;5;238m│[0m [38;5;246m        # Return results as tuples[0m
[38;5;238m  50[0m   [38;5;238m│[0m [38;5;246m        return results[0m
[38;5;238m  51[0m   [38;5;238m│[0m [38;5;246m        [0m
[38;5;238m  52[0m   [38;5;238m│[0m [38;5;246m    except Exception as e:[0m
[38;5;238m  53[0m   [38;5;238m│[0m [38;5;246m        logger.error(f"Reranking failed: {e}")[0m
[38;5;238m  54[0m   [38;5;238m│[0m [38;5;246m        # Return empty result on error[0m
[38;5;238m  55[0m   [38;5;238m│[0m [38;5;246m        return [][0m
[38;5;238m  56[0m   [38;5;238m│[0m [38;5;246m$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;[0m
[38;5;238m  57[0m   [38;5;238m│[0m 
[38;5;238m  58[0m   [38;5;238m│[0m [38;5;246m-- Rerank function returning only documents (no scores)[0m
[38;5;238m  59[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_rerank_docs_only([0m
[38;5;238m  60[0m   [38;5;238m│[0m [38;5;246m    query text,[0m
[38;5;238m  61[0m   [38;5;238m│[0m [38;5;246m    documents text[],[0m
[38;5;238m  62[0m   [38;5;238m│[0m [38;5;246m    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',[0m
[38;5;238m  63[0m   [38;5;238m│[0m [38;5;246m    seed integer DEFAULT 42[0m
[38;5;238m  64[0m   [38;5;238m│[0m [38;5;246m) RETURNS TABLE(document text)[0m
[38;5;238m  65[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m  66[0m   [38;5;238m│[0m [38;5;246m    # Call the main rerank function and extract just documents[0m
[38;5;238m  67[0m   [38;5;238m│[0m [38;5;246m    results = plpy.execute([0m
[38;5;238m  68[0m   [38;5;238m│[0m [38;5;246m        "SELECT document FROM steadytext_rerank($1, $2, $3, true, $4)",[0m
[38;5;238m  69[0m   [38;5;238m│[0m [38;5;246m        [query, documents, task, seed][0m
[38;5;238m  70[0m   [38;5;238m│[0m [38;5;246m    )[0m
[38;5;238m  71[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m  72[0m   [38;5;238m│[0m [38;5;246m    return [{"document": row["document"]} for row in results][0m
[38;5;238m  73[0m   [38;5;238m│[0m [38;5;246m$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;[0m
[38;5;238m  74[0m   [38;5;238m│[0m 
[38;5;238m  75[0m   [38;5;238m│[0m [38;5;246m-- Rerank function with top-k filtering[0m
[38;5;238m  76[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_rerank_top_k([0m
[38;5;238m  77[0m   [38;5;238m│[0m [38;5;246m    query text,[0m
[38;5;238m  78[0m   [38;5;238m│[0m [38;5;246m    documents text[],[0m
[38;5;238m  79[0m   [38;5;238m│[0m [38;5;246m    top_k integer,[0m
[38;5;238m  80[0m   [38;5;238m│[0m [38;5;246m    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',[0m
[38;5;238m  81[0m   [38;5;238m│[0m [38;5;246m    return_scores boolean DEFAULT true,[0m
[38;5;238m  82[0m   [38;5;238m│[0m [38;5;246m    seed integer DEFAULT 42[0m
[38;5;238m  83[0m   [38;5;238m│[0m [38;5;246m) RETURNS TABLE(document text, score float)[0m
[38;5;238m  84[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m  85[0m   [38;5;238m│[0m [38;5;246m    # Validate top_k[0m
[38;5;238m  86[0m   [38;5;238m│[0m [38;5;246m    if top_k <= 0:[0m
[38;5;238m  87[0m   [38;5;238m│[0m [38;5;246m        plpy.error("top_k must be positive")[0m
[38;5;238m  88[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m  89[0m   [38;5;238m│[0m [38;5;246m    # Call the main rerank function[0m
[38;5;238m  90[0m   [38;5;238m│[0m [38;5;246m    results = plpy.execute([0m
[38;5;238m  91[0m   [38;5;238m│[0m [38;5;246m        "SELECT document, score FROM steadytext_rerank($1, $2, $3, true, $4) LIMIT $5",[0m
[38;5;238m  92[0m   [38;5;238m│[0m [38;5;246m        [query, documents, task, seed, top_k][0m
[38;5;238m  93[0m   [38;5;238m│[0m [38;5;246m    )[0m
[38;5;238m  94[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m  95[0m   [38;5;238m│[0m [38;5;246m    if return_scores:[0m
[38;5;238m  96[0m   [38;5;238m│[0m [38;5;246m        return results[0m
[38;5;238m  97[0m   [38;5;238m│[0m [38;5;246m    else:[0m
[38;5;238m  98[0m   [38;5;238m│[0m [38;5;246m        # Return without scores[0m
[38;5;238m  99[0m   [38;5;238m│[0m [38;5;246m        return [{"document": row["document"], "score": None} for row in results][0m
[38;5;238m 100[0m   [38;5;238m│[0m [38;5;246m$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;[0m
[38;5;238m 101[0m   [38;5;238m│[0m 
[38;5;238m 102[0m   [38;5;238m│[0m [38;5;246m-- Async rerank function[0m
[38;5;238m 103[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_rerank_async([0m
[38;5;238m 104[0m   [38;5;238m│[0m [38;5;246m    query text,[0m
[38;5;238m 105[0m   [38;5;238m│[0m [38;5;246m    documents text[],[0m
[38;5;238m 106[0m   [38;5;238m│[0m [38;5;246m    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',[0m
[38;5;238m 107[0m   [38;5;238m│[0m [38;5;246m    return_scores boolean DEFAULT true,[0m
[38;5;238m 108[0m   [38;5;238m│[0m [38;5;246m    seed integer DEFAULT 42[0m
[38;5;238m 109[0m   [38;5;238m│[0m [38;5;246m) RETURNS uuid[0m
[38;5;238m 110[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m 111[0m   [38;5;238m│[0m [38;5;246m    import uuid[0m
[38;5;238m 112[0m   [38;5;238m│[0m [38;5;246m    import json[0m
[38;5;238m 113[0m   [38;5;238m│[0m [38;5;246m    import logging[0m
[38;5;238m 114[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 115[0m   [38;5;238m│[0m [38;5;246m    # Generate request ID[0m
[38;5;238m 116[0m   [38;5;238m│[0m [38;5;246m    request_id = str(uuid.uuid4())[0m
[38;5;238m 117[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 118[0m   [38;5;238m│[0m [38;5;246m    # Prepare parameters[0m
[38;5;238m 119[0m   [38;5;238m│[0m [38;5;246m    params = {[0m
[38;5;238m 120[0m   [38;5;238m│[0m [38;5;246m        'query': query,[0m
[38;5;238m 121[0m   [38;5;238m│[0m [38;5;246m        'documents': documents,[0m
[38;5;238m 122[0m   [38;5;238m│[0m [38;5;246m        'task': task,[0m
[38;5;238m 123[0m   [38;5;238m│[0m [38;5;246m        'return_scores': return_scores,[0m
[38;5;238m 124[0m   [38;5;238m│[0m [38;5;246m        'seed': seed[0m
[38;5;238m 125[0m   [38;5;238m│[0m [38;5;246m    }[0m
[38;5;238m 126[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 127[0m   [38;5;238m│[0m [38;5;246m    # Insert into queue[0m
[38;5;238m 128[0m   [38;5;238m│[0m [38;5;246m    plpy.execute("""[0m
[38;5;238m 129[0m   [38;5;238m│[0m [38;5;246m        INSERT INTO steadytext_queue [0m
[38;5;238m 130[0m   [38;5;238m│[0m [38;5;246m        (request_id, function_name, parameters, status, created_at, priority)[0m
[38;5;238m 131[0m   [38;5;238m│[0m [38;5;246m        VALUES ($1, 'rerank', $2::jsonb, 'pending', CURRENT_TIMESTAMP, 5)[0m
[38;5;238m 132[0m   [38;5;238m│[0m [38;5;246m    """, [request_id, json.dumps(params)])[0m
[38;5;238m 133[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 134[0m   [38;5;238m│[0m [38;5;246m    # Send notification to worker[0m
[38;5;238m 135[0m   [38;5;238m│[0m [38;5;246m    plpy.execute("NOTIFY steadytext_queue_notify")[0m
[38;5;238m 136[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 137[0m   [38;5;238m│[0m [38;5;246m    return request_id[0m
[38;5;238m 138[0m   [38;5;238m│[0m [38;5;246m$$ LANGUAGE plpython3u VOLATILE;[0m
[38;5;238m 139[0m   [38;5;238m│[0m 
[38;5;238m 140[0m   [38;5;238m│[0m [38;5;246m-- Batch rerank function for multiple queries[0m
[38;5;238m 141[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_rerank_batch([0m
[38;5;238m 142[0m   [38;5;238m│[0m [38;5;246m    queries text[],[0m
[38;5;238m 143[0m   [38;5;238m│[0m [38;5;246m    documents text[],[0m
[38;5;238m 144[0m   [38;5;238m│[0m [38;5;246m    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',[0m
[38;5;238m 145[0m   [38;5;238m│[0m [38;5;246m    return_scores boolean DEFAULT true,[0m
[38;5;238m 146[0m   [38;5;238m│[0m [38;5;246m    seed integer DEFAULT 42[0m
[38;5;238m 147[0m   [38;5;238m│[0m [38;5;246m) RETURNS TABLE(query_index integer, document text, score float)[0m
[38;5;238m 148[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m 149[0m   [38;5;238m│[0m [38;5;246m    import json[0m
[38;5;238m 150[0m   [38;5;238m│[0m [38;5;246m    import logging[0m
[38;5;238m 151[0m   [38;5;238m│[0m [38;5;246m    from typing import List, Tuple[0m
[38;5;238m 152[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 153[0m   [38;5;238m│[0m [38;5;246m    # Configure logging[0m
[38;5;238m 154[0m   [38;5;238m│[0m [38;5;246m    logging.basicConfig(level=logging.WARNING)[0m
[38;5;238m 155[0m   [38;5;238m│[0m [38;5;246m    logger = logging.getLogger(__name__)[0m
[38;5;238m 156[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 157[0m   [38;5;238m│[0m [38;5;246m    # Validate inputs[0m
[38;5;238m 158[0m   [38;5;238m│[0m [38;5;246m    if not queries or len(queries) == 0:[0m
[38;5;238m 159[0m   [38;5;238m│[0m [38;5;246m        plpy.error("Queries cannot be empty")[0m
[38;5;238m 160[0m   [38;5;238m│[0m [38;5;246m        [0m
[38;5;238m 161[0m   [38;5;238m│[0m [38;5;246m    if not documents or len(documents) == 0:[0m
[38;5;238m 162[0m   [38;5;238m│[0m [38;5;246m        return [][0m
[38;5;238m 163[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 164[0m   [38;5;238m│[0m [38;5;246m    # Import daemon connector[0m
[38;5;238m 165[0m   [38;5;238m│[0m [38;5;246m    try:[0m
[38;5;238m 166[0m   [38;5;238m│[0m [38;5;246m        from daemon_connector import SteadyTextConnector[0m
[38;5;238m 167[0m   [38;5;238m│[0m [38;5;246m        connector = SteadyTextConnector()[0m
[38;5;238m 168[0m   [38;5;238m│[0m [38;5;246m    except Exception as e:[0m
[38;5;238m 169[0m   [38;5;238m│[0m [38;5;246m        logger.error(f"Failed to initialize SteadyText connector: {e}")[0m
[38;5;238m 170[0m   [38;5;238m│[0m [38;5;246m        return [][0m
[38;5;238m 171[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 172[0m   [38;5;238m│[0m [38;5;246m    all_results = [][0m
[38;5;238m 173[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 174[0m   [38;5;238m│[0m [38;5;246m    # Process each query[0m
[38;5;238m 175[0m   [38;5;238m│[0m [38;5;246m    for idx, query in enumerate(queries):[0m
[38;5;238m 176[0m   [38;5;238m│[0m [38;5;246m        try:[0m
[38;5;238m 177[0m   [38;5;238m│[0m [38;5;246m            # Call rerank for this query[0m
[38;5;238m 178[0m   [38;5;238m│[0m [38;5;246m            results = connector.rerank([0m
[38;5;238m 179[0m   [38;5;238m│[0m [38;5;246m                query=query,[0m
[38;5;238m 180[0m   [38;5;238m│[0m [38;5;246m                documents=list(documents),[0m
[38;5;238m 181[0m   [38;5;238m│[0m [38;5;246m                task=task,[0m
[38;5;238m 182[0m   [38;5;238m│[0m [38;5;246m                return_scores=True,[0m
[38;5;238m 183[0m   [38;5;238m│[0m [38;5;246m                seed=seed[0m
[38;5;238m 184[0m   [38;5;238m│[0m [38;5;246m            )[0m
[38;5;238m 185[0m   [38;5;238m│[0m [38;5;246m            [0m
[38;5;238m 186[0m   [38;5;238m│[0m [38;5;246m            # Add query index to results[0m
[38;5;238m 187[0m   [38;5;238m│[0m [38;5;246m            for doc, score in results:[0m
[38;5;238m 188[0m   [38;5;238m│[0m [38;5;246m                all_results.append({[0m
[38;5;238m 189[0m   [38;5;238m│[0m [38;5;246m                    "query_index": idx,[0m
[38;5;238m 190[0m   [38;5;238m│[0m [38;5;246m                    "document": doc,[0m
[38;5;238m 191[0m   [38;5;238m│[0m [38;5;246m                    "score": score[0m
[38;5;238m 192[0m   [38;5;238m│[0m [38;5;246m                })[0m
[38;5;238m 193[0m   [38;5;238m│[0m [38;5;246m                [0m
[38;5;238m 194[0m   [38;5;238m│[0m [38;5;246m        except Exception as e:[0m
[38;5;238m 195[0m   [38;5;238m│[0m [38;5;246m            logger.error(f"Reranking failed for query {idx}: {e}")[0m
[38;5;238m 196[0m   [38;5;238m│[0m [38;5;246m            # Continue with next query[0m
[38;5;238m 197[0m   [38;5;238m│[0m [38;5;246m            continue[0m
[38;5;238m 198[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 199[0m   [38;5;238m│[0m [38;5;246m    return all_results[0m
[38;5;238m 200[0m   [38;5;238m│[0m [38;5;246m$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;[0m
[38;5;238m 201[0m   [38;5;238m│[0m 
[38;5;238m 202[0m   [38;5;238m│[0m [38;5;246m-- Batch async rerank function[0m
[38;5;238m 203[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_rerank_batch_async([0m
[38;5;238m 204[0m   [38;5;238m│[0m [38;5;246m    queries text[],[0m
[38;5;238m 205[0m   [38;5;238m│[0m [38;5;246m    documents text[],[0m
[38;5;238m 206[0m   [38;5;238m│[0m [38;5;246m    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',[0m
[38;5;238m 207[0m   [38;5;238m│[0m [38;5;246m    return_scores boolean DEFAULT true,[0m
[38;5;238m 208[0m   [38;5;238m│[0m [38;5;246m    seed integer DEFAULT 42[0m
[38;5;238m 209[0m   [38;5;238m│[0m [38;5;246m) RETURNS uuid[][0m
[38;5;238m 210[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m 211[0m   [38;5;238m│[0m [38;5;246m    import uuid[0m
[38;5;238m 212[0m   [38;5;238m│[0m [38;5;246m    import json[0m
[38;5;238m 213[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 214[0m   [38;5;238m│[0m [38;5;246m    request_ids = [][0m
[38;5;238m 215[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 216[0m   [38;5;238m│[0m [38;5;246m    # Create separate async request for each query[0m
[38;5;238m 217[0m   [38;5;238m│[0m [38;5;246m    for query in queries:[0m
[38;5;238m 218[0m   [38;5;238m│[0m [38;5;246m        request_id = str(uuid.uuid4())[0m
[38;5;238m 219[0m   [38;5;238m│[0m [38;5;246m        request_ids.append(request_id)[0m
[38;5;238m 220[0m   [38;5;238m│[0m [38;5;246m        [0m
[38;5;238m 221[0m   [38;5;238m│[0m [38;5;246m        params = {[0m
[38;5;238m 222[0m   [38;5;238m│[0m [38;5;246m            'query': query,[0m
[38;5;238m 223[0m   [38;5;238m│[0m [38;5;246m            'documents': documents,[0m
[38;5;238m 224[0m   [38;5;238m│[0m [38;5;246m            'task': task,[0m
[38;5;238m 225[0m   [38;5;238m│[0m [38;5;246m            'return_scores': return_scores,[0m
[38;5;238m 226[0m   [38;5;238m│[0m [38;5;246m            'seed': seed[0m
[38;5;238m 227[0m   [38;5;238m│[0m [38;5;246m        }[0m
[38;5;238m 228[0m   [38;5;238m│[0m [38;5;246m        [0m
[38;5;238m 229[0m   [38;5;238m│[0m [38;5;246m        plpy.execute("""[0m
[38;5;238m 230[0m   [38;5;238m│[0m [38;5;246m            INSERT INTO steadytext_queue [0m
[38;5;238m 231[0m   [38;5;238m│[0m [38;5;246m            (request_id, function_name, parameters, status, created_at, priority)[0m
[38;5;238m 232[0m   [38;5;238m│[0m [38;5;246m            VALUES ($1, 'rerank', $2::jsonb, 'pending', CURRENT_TIMESTAMP, 5)[0m
[38;5;238m 233[0m   [38;5;238m│[0m [38;5;246m        """, [request_id, json.dumps(params)])[0m
[38;5;238m 234[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 235[0m   [38;5;238m│[0m [38;5;246m    # Send notification to worker[0m
[38;5;238m 236[0m   [38;5;238m│[0m [38;5;246m    plpy.execute("NOTIFY steadytext_queue_notify")[0m
[38;5;238m 237[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 238[0m   [38;5;238m│[0m [38;5;246m    return request_ids[0m
[38;5;238m 239[0m   [38;5;238m│[0m [38;5;246m$$ LANGUAGE plpython3u VOLATILE;[0m
[38;5;238m 240[0m   [38;5;238m│[0m 
[38;5;238m 241[0m   [38;5;238m│[0m [38;5;246m-- Add rerank support to worker processing[0m
[38;5;238m 242[0m   [38;5;238m│[0m [38;5;246m-- This updates the worker to handle 'rerank' function_name in the queue[0m
[38;5;238m 243[0m   [38;5;238m│[0m [38;5;246m-- AIDEV-NOTE: The actual worker.py file handles this, but we document it here[0m
[38;5;238m 244[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_rerank IS 'Rerank documents by relevance to a query using AI model';[0m
[38;5;238m 245[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_rerank_docs_only IS 'Rerank documents returning only sorted documents without scores';[0m
[38;5;238m 246[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_rerank_top_k IS 'Rerank documents and return only top K results';[0m
[38;5;238m 247[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_rerank_async IS 'Asynchronously rerank documents (returns request UUID)';[0m
[38;5;238m 248[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_rerank_batch IS 'Rerank documents for multiple queries in batch';[0m
[38;5;238m 249[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_rerank_batch_async IS 'Asynchronously rerank documents for multiple queries';[0m
[38;5;238m 250[0m   [38;5;238m│[0m 
[38;5;238m 251[0m   [38;5;238m│[0m [38;5;246m-- =====================================================[0m
[38;5;238m 252[0m   [38;5;238m│[0m [38;5;246m-- Migration from 1.3.0 to 1.4.0: Cache Eviction[0m
[38;5;238m 253[0m   [38;5;238m│[0m [38;5;246m-- =====================================================[0m
[38;5;238m 254[0m   [38;5;238m│[0m 
[38;5;238m 255[0m   [38;5;238m│[0m [38;5;246m-- AIDEV-SECTION: CACHE_EVICTION_CONFIGURATION[0m
[38;5;238m 256[0m   [38;5;238m│[0m [38;5;246m-- Add cache eviction configuration to steadytext_config[0m
[38;5;238m 257[0m   [38;5;238m│[0m [38;5;246mINSERT INTO steadytext_config (key, value, description) VALUES[0m
[38;5;238m 258[0m   [38;5;238m│[0m [38;5;246m    ('cache_eviction_enabled', 'true', 'Enable automatic cache eviction'),[0m
[38;5;238m 259[0m   [38;5;238m│[0m [38;5;246m    ('cache_eviction_interval', '300', 'Cache eviction interval in seconds (default: 5 minutes)'),[0m
[38;5;238m 260[0m   [38;5;238m│[0m [38;5;246m    ('cache_max_entries', '10000', 'Maximum number of cache entries before eviction'),[0m
[38;5;238m 261[0m   [38;5;238m│[0m [38;5;246m    ('cache_max_size_mb', '1000', 'Maximum cache size in MB before eviction'),[0m
[38;5;238m 262[0m   [38;5;238m│[0m [38;5;246m    ('cache_eviction_batch_size', '100', 'Number of entries to evict in each batch'),[0m
[38;5;238m 263[0m   [38;5;238m│[0m [38;5;246m    ('cache_min_access_count', '2', 'Minimum access count to protect from eviction'),[0m
[38;5;238m 264[0m   [38;5;238m│[0m [38;5;246m    ('cache_min_age_hours', '1', 'Minimum age in hours to protect from eviction'),[0m
[38;5;238m 265[0m   [38;5;238m│[0m [38;5;246m    ('cron_host', '"localhost"', 'Database host for pg_cron connections'),[0m
[38;5;238m 266[0m   [38;5;238m│[0m [38;5;246m    ('cron_port', '5432', 'Database port for pg_cron connections')[0m
[38;5;238m 267[0m   [38;5;238m│[0m [38;5;246mON CONFLICT (key) DO NOTHING;[0m
[38;5;238m 268[0m   [38;5;238m│[0m 
[38;5;238m 269[0m   [38;5;238m│[0m [38;5;246m-- Create view for cache with frecency scores[0m
[38;5;238m 270[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE VIEW steadytext_cache_with_frecency AS[0m
[38;5;238m 271[0m   [38;5;238m│[0m [38;5;246mSELECT [0m
[38;5;238m 272[0m   [38;5;238m│[0m [38;5;246m    *,[0m
[38;5;238m 273[0m   [38;5;238m│[0m [38;5;246m    access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0) as frecency_score[0m
[38;5;238m 274[0m   [38;5;238m│[0m [38;5;246mFROM steadytext_cache;[0m
[38;5;238m 275[0m   [38;5;238m│[0m 
[38;5;238m 276[0m   [38;5;238m│[0m [38;5;246m-- AIDEV-SECTION: CACHE_PERFORMANCE_INDEXES[0m
[38;5;238m 277[0m   [38;5;238m│[0m [38;5;246m-- Add indexes for optimal frecency-based eviction performance[0m
[38;5;238m 278[0m   [38;5;238m│[0m [38;5;246m-- This index supports the ORDER BY frecency_score query in eviction[0m
[38;5;238m 279[0m   [38;5;238m│[0m [38;5;246m-- AIDEV-NOTE: WHERE clause removed due to NOW() not being immutable[0m
[38;5;238m 280[0m   [38;5;238m│[0m [38;5;246mCREATE INDEX IF NOT EXISTS idx_steadytext_cache_frecency_eviction [0m
[38;5;238m 281[0m   [38;5;238m│[0m [38;5;246mON steadytext_cache (access_count, last_accessed);[0m
[38;5;238m 282[0m   [38;5;238m│[0m 
[38;5;238m 283[0m   [38;5;238m│[0m [38;5;246m-- AIDEV-SECTION: CACHE_STATISTICS_FUNCTIONS[0m
[38;5;238m 284[0m   [38;5;238m│[0m [38;5;246m-- Enhanced cache statistics function with size calculations[0m
[38;5;238m 285[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_cache_stats_extended()[0m
[38;5;238m 286[0m   [38;5;238m│[0m [38;5;246mRETURNS TABLE([0m
[38;5;238m 287[0m   [38;5;238m│[0m [38;5;246m    total_entries BIGINT,[0m
[38;5;238m 288[0m   [38;5;238m│[0m [38;5;246m    total_size_mb FLOAT,[0m
[38;5;238m 289[0m   [38;5;238m│[0m [38;5;246m    cache_hit_rate FLOAT,[0m
[38;5;238m 290[0m   [38;5;238m│[0m [38;5;246m    avg_access_count FLOAT,[0m
[38;5;238m 291[0m   [38;5;238m│[0m [38;5;246m    oldest_entry TIMESTAMPTZ,[0m
[38;5;238m 292[0m   [38;5;238m│[0m [38;5;246m    newest_entry TIMESTAMPTZ,[0m
[38;5;238m 293[0m   [38;5;238m│[0m [38;5;246m    low_frecency_count BIGINT,[0m
[38;5;238m 294[0m   [38;5;238m│[0m [38;5;246m    protected_count BIGINT,[0m
[38;5;238m 295[0m   [38;5;238m│[0m [38;5;246m    eviction_candidates BIGINT[0m
[38;5;238m 296[0m   [38;5;238m│[0m [38;5;246m)[0m
[38;5;238m 297[0m   [38;5;238m│[0m [38;5;246mLANGUAGE sql[0m
[38;5;238m 298[0m   [38;5;238m│[0m [38;5;246mSTABLE PARALLEL SAFE[0m
[38;5;238m 299[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m 300[0m   [38;5;238m│[0m [38;5;246m    WITH cache_analysis AS ([0m
[38;5;238m 301[0m   [38;5;238m│[0m [38;5;246m        SELECT [0m
[38;5;238m 302[0m   [38;5;238m│[0m [38;5;246m            COUNT(*)::BIGINT as total_entries,[0m
[38;5;238m 303[0m   [38;5;238m│[0m [38;5;246m            COALESCE(SUM(pg_column_size(response) + pg_column_size(embedding)) / 1024.0 / 1024.0, 0)::FLOAT as total_size_mb,[0m
[38;5;238m 304[0m   [38;5;238m│[0m [38;5;246m            COALESCE(SUM(CASE WHEN access_count > 1 THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0), 0)::FLOAT as cache_hit_rate,[0m
[38;5;238m 305[0m   [38;5;238m│[0m [38;5;246m            COALESCE(AVG(access_count), 0)::FLOAT as avg_access_count,[0m
[38;5;238m 306[0m   [38;5;238m│[0m [38;5;246m            MIN(created_at) as oldest_entry,[0m
[38;5;238m 307[0m   [38;5;238m│[0m [38;5;246m            MAX(created_at) as newest_entry,[0m
[38;5;238m 308[0m   [38;5;238m│[0m [38;5;246m            -- Count entries with low frecency scores[0m
[38;5;238m 309[0m   [38;5;238m│[0m [38;5;246m            SUM(CASE [0m
[38;5;238m 310[0m   [38;5;238m│[0m [38;5;246m                WHEN access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0) < 1 [0m
[38;5;238m 311[0m   [38;5;238m│[0m [38;5;246m                THEN 1 ELSE 0 [0m
[38;5;238m 312[0m   [38;5;238m│[0m [38;5;246m            END)::BIGINT as low_frecency_count,[0m
[38;5;238m 313[0m   [38;5;238m│[0m [38;5;246m            -- Count protected entries (high access count or recently created)[0m
[38;5;238m 314[0m   [38;5;238m│[0m [38;5;246m            SUM(CASE [0m
[38;5;238m 315[0m   [38;5;238m│[0m [38;5;246m                WHEN access_count >= 2 OR created_at > NOW() - INTERVAL '1 hour' [0m
[38;5;238m 316[0m   [38;5;238m│[0m [38;5;246m                THEN 1 ELSE 0 [0m
[38;5;238m 317[0m   [38;5;238m│[0m [38;5;246m            END)::BIGINT as protected_count,[0m
[38;5;238m 318[0m   [38;5;238m│[0m [38;5;246m            -- Count eviction candidates[0m
[38;5;238m 319[0m   [38;5;238m│[0m [38;5;246m            SUM(CASE [0m
[38;5;238m 320[0m   [38;5;238m│[0m [38;5;246m                WHEN access_count < 2 [0m
[38;5;238m 321[0m   [38;5;238m│[0m [38;5;246m                    AND created_at < NOW() - INTERVAL '1 hour'[0m
[38;5;238m 322[0m   [38;5;238m│[0m [38;5;246m                    AND access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0) < 1[0m
[38;5;238m 323[0m   [38;5;238m│[0m [38;5;246m                THEN 1 ELSE 0 [0m
[38;5;238m 324[0m   [38;5;238m│[0m [38;5;246m            END)::BIGINT as eviction_candidates[0m
[38;5;238m 325[0m   [38;5;238m│[0m [38;5;246m        FROM steadytext_cache[0m
[38;5;238m 326[0m   [38;5;238m│[0m [38;5;246m    )[0m
[38;5;238m 327[0m   [38;5;238m│[0m [38;5;246m    SELECT * FROM cache_analysis;[0m
[38;5;238m 328[0m   [38;5;238m│[0m [38;5;246m$$;[0m
[38;5;238m 329[0m   [38;5;238m│[0m 
[38;5;238m 330[0m   [38;5;238m│[0m [38;5;246m-- AIDEV-SECTION: CACHE_EVICTION_FUNCTIONS[0m
[38;5;238m 331[0m   [38;5;238m│[0m [38;5;246m-- Function to perform cache eviction based on frecency[0m
[38;5;238m 332[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_cache_evict_by_frecency([0m
[38;5;238m 333[0m   [38;5;238m│[0m [38;5;246m    target_entries INT DEFAULT NULL,[0m
[38;5;238m 334[0m   [38;5;238m│[0m [38;5;246m    target_size_mb FLOAT DEFAULT NULL,[0m
[38;5;238m 335[0m   [38;5;238m│[0m [38;5;246m    batch_size INT DEFAULT 100,[0m
[38;5;238m 336[0m   [38;5;238m│[0m [38;5;246m    min_access_count INT DEFAULT 2,[0m
[38;5;238m 337[0m   [38;5;238m│[0m [38;5;246m    min_age_hours INT DEFAULT 1[0m
[38;5;238m 338[0m   [38;5;238m│[0m [38;5;246m)[0m
[38;5;238m 339[0m   [38;5;238m│[0m [38;5;246mRETURNS TABLE([0m
[38;5;238m 340[0m   [38;5;238m│[0m [38;5;246m    evicted_count INT,[0m
[38;5;238m 341[0m   [38;5;238m│[0m [38;5;246m    freed_size_mb FLOAT,[0m
[38;5;238m 342[0m   [38;5;238m│[0m [38;5;246m    remaining_entries BIGINT,[0m
[38;5;238m 343[0m   [38;5;238m│[0m [38;5;246m    remaining_size_mb FLOAT[0m
[38;5;238m 344[0m   [38;5;238m│[0m [38;5;246m)[0m
[38;5;238m 345[0m   [38;5;238m│[0m [38;5;246mLANGUAGE plpgsql[0m
[38;5;238m 346[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m 347[0m   [38;5;238m│[0m [38;5;246mDECLARE[0m
[38;5;238m 348[0m   [38;5;238m│[0m [38;5;246m    v_evicted_count INT := 0;[0m
[38;5;238m 349[0m   [38;5;238m│[0m [38;5;246m    v_freed_size BIGINT := 0;[0m
[38;5;238m 350[0m   [38;5;238m│[0m [38;5;246m    v_current_stats RECORD;[0m
[38;5;238m 351[0m   [38;5;238m│[0m [38;5;246m    v_max_entries INT;[0m
[38;5;238m 352[0m   [38;5;238m│[0m [38;5;246m    v_max_size_mb FLOAT;[0m
[38;5;238m 353[0m   [38;5;238m│[0m [38;5;246m    v_should_evict BOOLEAN := FALSE;[0m
[38;5;238m 354[0m   [38;5;238m│[0m [38;5;246m    v_loop_count INT := 0;[0m
[38;5;238m 355[0m   [38;5;238m│[0m [38;5;246m    v_max_loop_count INT := 1000; -- Safety limit to prevent infinite loops[0m
[38;5;238m 356[0m   [38;5;238m│[0m [38;5;246mBEGIN[0m
[38;5;238m 357[0m   [38;5;238m│[0m [38;5;246m    -- Get current cache stats[0m
[38;5;238m 358[0m   [38;5;238m│[0m [38;5;246m    SELECT [0m
[38;5;238m 359[0m   [38;5;238m│[0m [38;5;246m        COUNT(*)::BIGINT as total_entries,[0m
[38;5;238m 360[0m   [38;5;238m│[0m [38;5;246m        COALESCE(SUM(pg_column_size(response) + pg_column_size(embedding)), 0)::BIGINT as total_size[0m
[38;5;238m 361[0m   [38;5;238m│[0m [38;5;246m    INTO v_current_stats[0m
[38;5;238m 362[0m   [38;5;238m│[0m [38;5;246m    FROM steadytext_cache;[0m
[38;5;238m 363[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 364[0m   [38;5;238m│[0m [38;5;246m    -- Use provided targets or get from config[0m
[38;5;238m 365[0m   [38;5;238m│[0m [38;5;246m    IF target_entries IS NULL THEN[0m
[38;5;238m 366[0m   [38;5;238m│[0m [38;5;246m        SELECT value::INT INTO v_max_entries [0m
[38;5;238m 367[0m   [38;5;238m│[0m [38;5;246m        FROM steadytext_config [0m
[38;5;238m 368[0m   [38;5;238m│[0m [38;5;246m        WHERE key = 'cache_max_entries';[0m
[38;5;238m 369[0m   [38;5;238m│[0m [38;5;246m        v_max_entries := COALESCE(v_max_entries, 10000);[0m
[38;5;238m 370[0m   [38;5;238m│[0m [38;5;246m    ELSE[0m
[38;5;238m 371[0m   [38;5;238m│[0m [38;5;246m        v_max_entries := target_entries;[0m
[38;5;238m 372[0m   [38;5;238m│[0m [38;5;246m    END IF;[0m
[38;5;238m 373[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 374[0m   [38;5;238m│[0m [38;5;246m    IF target_size_mb IS NULL THEN[0m
[38;5;238m 375[0m   [38;5;238m│[0m [38;5;246m        SELECT value::FLOAT INTO v_max_size_mb [0m
[38;5;238m 376[0m   [38;5;238m│[0m [38;5;246m        FROM steadytext_config [0m
[38;5;238m 377[0m   [38;5;238m│[0m [38;5;246m        WHERE key = 'cache_max_size_mb';[0m
[38;5;238m 378[0m   [38;5;238m│[0m [38;5;246m        v_max_size_mb := COALESCE(v_max_size_mb, 1000);[0m
[38;5;238m 379[0m   [38;5;238m│[0m [38;5;246m    ELSE[0m
[38;5;238m 380[0m   [38;5;238m│[0m [38;5;246m        v_max_size_mb := target_size_mb;[0m
[38;5;238m 381[0m   [38;5;238m│[0m [38;5;246m    END IF;[0m
[38;5;238m 382[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 383[0m   [38;5;238m│[0m [38;5;246m    -- Check if eviction is needed[0m
[38;5;238m 384[0m   [38;5;238m│[0m [38;5;246m    IF v_current_stats.total_entries > v_max_entries OR [0m
[38;5;238m 385[0m   [38;5;238m│[0m [38;5;246m       (v_current_stats.total_size / 1024.0 / 1024.0) > v_max_size_mb THEN[0m
[38;5;238m 386[0m   [38;5;238m│[0m [38;5;246m        v_should_evict := TRUE;[0m
[38;5;238m 387[0m   [38;5;238m│[0m [38;5;246m    END IF;[0m
[38;5;238m 388[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 389[0m   [38;5;238m│[0m [38;5;246m    -- Perform eviction if needed[0m
[38;5;238m 390[0m   [38;5;238m│[0m [38;5;246m    WHILE v_should_evict LOOP[0m
[38;5;238m 391[0m   [38;5;238m│[0m [38;5;246m        -- Safety check to prevent infinite loops[0m
[38;5;238m 392[0m   [38;5;238m│[0m [38;5;246m        v_loop_count := v_loop_count + 1;[0m
[38;5;238m 393[0m   [38;5;238m│[0m [38;5;246m        IF v_loop_count > v_max_loop_count THEN[0m
[38;5;238m 394[0m   [38;5;238m│[0m [38;5;246m            RAISE WARNING 'Cache eviction loop exceeded maximum iterations (%), breaking to prevent infinite loop', v_max_loop_count;[0m
[38;5;238m 395[0m   [38;5;238m│[0m [38;5;246m            EXIT;[0m
[38;5;238m 396[0m   [38;5;238m│[0m [38;5;246m        END IF;[0m
[38;5;238m 397[0m   [38;5;238m│[0m [38;5;246m        WITH eviction_batch AS ([0m
[38;5;238m 398[0m   [38;5;238m│[0m [38;5;246m            DELETE FROM steadytext_cache[0m
[38;5;238m 399[0m   [38;5;238m│[0m [38;5;246m            WHERE id IN ([0m
[38;5;238m 400[0m   [38;5;238m│[0m [38;5;246m                SELECT id [0m
[38;5;238m 401[0m   [38;5;238m│[0m [38;5;246m                FROM steadytext_cache_with_frecency[0m
[38;5;238m 402[0m   [38;5;238m│[0m [38;5;246m                WHERE [0m
[38;5;238m 403[0m   [38;5;238m│[0m [38;5;246m                    -- Don't evict entries with high access count[0m
[38;5;238m 404[0m   [38;5;238m│[0m [38;5;246m                    access_count < min_access_count[0m
[38;5;238m 405[0m   [38;5;238m│[0m [38;5;246m                    -- Don't evict very recent entries[0m
[38;5;238m 406[0m   [38;5;238m│[0m [38;5;246m                    AND created_at < NOW() - INTERVAL '1 hour' * min_age_hours[0m
[38;5;238m 407[0m   [38;5;238m│[0m [38;5;246m                ORDER BY frecency_score ASC[0m
[38;5;238m 408[0m   [38;5;238m│[0m [38;5;246m                LIMIT batch_size[0m
[38;5;238m 409[0m   [38;5;238m│[0m [38;5;246m            )[0m
[38;5;238m 410[0m   [38;5;238m│[0m [38;5;246m            RETURNING pg_column_size(response) + pg_column_size(embedding) as entry_size[0m
[38;5;238m 411[0m   [38;5;238m│[0m [38;5;246m        )[0m
[38;5;238m 412[0m   [38;5;238m│[0m [38;5;246m        SELECT [0m
[38;5;238m 413[0m   [38;5;238m│[0m [38;5;246m            COUNT(*)::INT,[0m
[38;5;238m 414[0m   [38;5;238m│[0m [38;5;246m            COALESCE(SUM(entry_size), 0)::BIGINT[0m
[38;5;238m 415[0m   [38;5;238m│[0m [38;5;246m        INTO [0m
[38;5;238m 416[0m   [38;5;238m│[0m [38;5;246m            v_evicted_count,[0m
[38;5;238m 417[0m   [38;5;238m│[0m [38;5;246m            v_freed_size[0m
[38;5;238m 418[0m   [38;5;238m│[0m [38;5;246m        FROM eviction_batch;[0m
[38;5;238m 419[0m   [38;5;238m│[0m [38;5;246m        [0m
[38;5;238m 420[0m   [38;5;238m│[0m [38;5;246m        -- Break if nothing was evicted[0m
[38;5;238m 421[0m   [38;5;238m│[0m [38;5;246m        IF v_evicted_count = 0 THEN[0m
[38;5;238m 422[0m   [38;5;238m│[0m [38;5;246m            EXIT;[0m
[38;5;238m 423[0m   [38;5;238m│[0m [38;5;246m        END IF;[0m
[38;5;238m 424[0m   [38;5;238m│[0m [38;5;246m        [0m
[38;5;238m 425[0m   [38;5;238m│[0m [38;5;246m        -- Update running totals[0m
[38;5;238m 426[0m   [38;5;238m│[0m [38;5;246m        v_current_stats.total_entries := v_current_stats.total_entries - v_evicted_count;[0m
[38;5;238m 427[0m   [38;5;238m│[0m [38;5;246m        v_current_stats.total_size := v_current_stats.total_size - v_freed_size;[0m
[38;5;238m 428[0m   [38;5;238m│[0m [38;5;246m        [0m
[38;5;238m 429[0m   [38;5;238m│[0m [38;5;246m        -- Check if we've reached targets[0m
[38;5;238m 430[0m   [38;5;238m│[0m [38;5;246m        IF v_current_stats.total_entries <= v_max_entries AND [0m
[38;5;238m 431[0m   [38;5;238m│[0m [38;5;246m           (v_current_stats.total_size / 1024.0 / 1024.0) <= v_max_size_mb THEN[0m
[38;5;238m 432[0m   [38;5;238m│[0m [38;5;246m            v_should_evict := FALSE;[0m
[38;5;238m 433[0m   [38;5;238m│[0m [38;5;246m        END IF;[0m
[38;5;238m 434[0m   [38;5;238m│[0m [38;5;246m    END LOOP;[0m
[38;5;238m 435[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 436[0m   [38;5;238m│[0m [38;5;246m    -- Return results[0m
[38;5;238m 437[0m   [38;5;238m│[0m [38;5;246m    RETURN QUERY[0m
[38;5;238m 438[0m   [38;5;238m│[0m [38;5;246m    SELECT [0m
[38;5;238m 439[0m   [38;5;238m│[0m [38;5;246m        v_evicted_count,[0m
[38;5;238m 440[0m   [38;5;238m│[0m [38;5;246m        (v_freed_size / 1024.0 / 1024.0)::FLOAT,[0m
[38;5;238m 441[0m   [38;5;238m│[0m [38;5;246m        v_current_stats.total_entries,[0m
[38;5;238m 442[0m   [38;5;238m│[0m [38;5;246m        (v_current_stats.total_size / 1024.0 / 1024.0)::FLOAT;[0m
[38;5;238m 443[0m   [38;5;238m│[0m [38;5;246mEND;[0m
[38;5;238m 444[0m   [38;5;238m│[0m [38;5;246m$$;[0m
[38;5;238m 445[0m   [38;5;238m│[0m 
[38;5;238m 446[0m   [38;5;238m│[0m [38;5;246m-- AIDEV-SECTION: SCHEDULED_EVICTION_FUNCTION[0m
[38;5;238m 447[0m   [38;5;238m│[0m [38;5;246m-- Function to be called by pg_cron for scheduled eviction[0m
[38;5;238m 448[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_cache_scheduled_eviction()[0m
[38;5;238m 449[0m   [38;5;238m│[0m [38;5;246mRETURNS JSONB[0m
[38;5;238m 450[0m   [38;5;238m│[0m [38;5;246mLANGUAGE plpgsql[0m
[38;5;238m 451[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m 452[0m   [38;5;238m│[0m [38;5;246mDECLARE[0m
[38;5;238m 453[0m   [38;5;238m│[0m [38;5;246m    v_eviction_enabled BOOLEAN;[0m
[38;5;238m 454[0m   [38;5;238m│[0m [38;5;246m    v_result RECORD;[0m
[38;5;238m 455[0m   [38;5;238m│[0m [38;5;246m    v_start_time TIMESTAMPTZ;[0m
[38;5;238m 456[0m   [38;5;238m│[0m [38;5;246m    v_end_time TIMESTAMPTZ;[0m
[38;5;238m 457[0m   [38;5;238m│[0m [38;5;246m    v_duration_ms INT;[0m
[38;5;238m 458[0m   [38;5;238m│[0m [38;5;246mBEGIN[0m
[38;5;238m 459[0m   [38;5;238m│[0m [38;5;246m    -- Check if eviction is enabled[0m
[38;5;238m 460[0m   [38;5;238m│[0m [38;5;246m    SELECT value::BOOLEAN INTO v_eviction_enabled[0m
[38;5;238m 461[0m   [38;5;238m│[0m [38;5;246m    FROM steadytext_config[0m
[38;5;238m 462[0m   [38;5;238m│[0m [38;5;246m    WHERE key = 'cache_eviction_enabled';[0m
[38;5;238m 463[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 464[0m   [38;5;238m│[0m [38;5;246m    IF NOT COALESCE(v_eviction_enabled, TRUE) THEN[0m
[38;5;238m 465[0m   [38;5;238m│[0m [38;5;246m        RETURN jsonb_build_object([0m
[38;5;238m 466[0m   [38;5;238m│[0m [38;5;246m            'status', 'skipped',[0m
[38;5;238m 467[0m   [38;5;238m│[0m [38;5;246m            'reason', 'Cache eviction disabled',[0m
[38;5;238m 468[0m   [38;5;238m│[0m [38;5;246m            'timestamp', NOW()[0m
[38;5;238m 469[0m   [38;5;238m│[0m [38;5;246m        );[0m
[38;5;238m 470[0m   [38;5;238m│[0m [38;5;246m    END IF;[0m
[38;5;238m 471[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 472[0m   [38;5;238m│[0m [38;5;246m    v_start_time := clock_timestamp();[0m
[38;5;238m 473[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 474[0m   [38;5;238m│[0m [38;5;246m    -- Perform eviction using configured parameters[0m
[38;5;238m 475[0m   [38;5;238m│[0m [38;5;246m    SELECT * INTO v_result[0m
[38;5;238m 476[0m   [38;5;238m│[0m [38;5;246m    FROM steadytext_cache_evict_by_frecency();[0m
[38;5;238m 477[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 478[0m   [38;5;238m│[0m [38;5;246m    v_end_time := clock_timestamp();[0m
[38;5;238m 479[0m   [38;5;238m│[0m [38;5;246m    v_duration_ms := EXTRACT(MILLISECONDS FROM (v_end_time - v_start_time))::INT;[0m
[38;5;238m 480[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 481[0m   [38;5;238m│[0m [38;5;246m    -- Log to audit table if significant eviction occurred[0m
[38;5;238m 482[0m   [38;5;238m│[0m [38;5;246m    IF v_result.evicted_count > 0 THEN[0m
[38;5;238m 483[0m   [38;5;238m│[0m [38;5;246m        INSERT INTO steadytext_audit_log ([0m
[38;5;238m 484[0m   [38;5;238m│[0m [38;5;246m            action, [0m
[38;5;238m 485[0m   [38;5;238m│[0m [38;5;246m            details,[0m
[38;5;238m 486[0m   [38;5;238m│[0m [38;5;246m            success[0m
[38;5;238m 487[0m   [38;5;238m│[0m [38;5;246m        ) VALUES ([0m
[38;5;238m 488[0m   [38;5;238m│[0m [38;5;246m            'cache_eviction',[0m
[38;5;238m 489[0m   [38;5;238m│[0m [38;5;246m            jsonb_build_object([0m
[38;5;238m 490[0m   [38;5;238m│[0m [38;5;246m                'evicted_count', v_result.evicted_count,[0m
[38;5;238m 491[0m   [38;5;238m│[0m [38;5;246m                'freed_size_mb', v_result.freed_size_mb,[0m
[38;5;238m 492[0m   [38;5;238m│[0m [38;5;246m                'remaining_entries', v_result.remaining_entries,[0m
[38;5;238m 493[0m   [38;5;238m│[0m [38;5;246m                'remaining_size_mb', v_result.remaining_size_mb,[0m
[38;5;238m 494[0m   [38;5;238m│[0m [38;5;246m                'duration_ms', v_duration_ms[0m
[38;5;238m 495[0m   [38;5;238m│[0m [38;5;246m            ),[0m
[38;5;238m 496[0m   [38;5;238m│[0m [38;5;246m            TRUE[0m
[38;5;238m 497[0m   [38;5;238m│[0m [38;5;246m        );[0m
[38;5;238m 498[0m   [38;5;238m│[0m [38;5;246m    END IF;[0m
[38;5;238m 499[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 500[0m   [38;5;238m│[0m [38;5;246m    -- Return detailed result[0m
[38;5;238m 501[0m   [38;5;238m│[0m [38;5;246m    RETURN jsonb_build_object([0m
[38;5;238m 502[0m   [38;5;238m│[0m [38;5;246m        'status', 'completed',[0m
[38;5;238m 503[0m   [38;5;238m│[0m [38;5;246m        'timestamp', NOW(),[0m
[38;5;238m 504[0m   [38;5;238m│[0m [38;5;246m        'evicted_count', v_result.evicted_count,[0m
[38;5;238m 505[0m   [38;5;238m│[0m [38;5;246m        'freed_size_mb', v_result.freed_size_mb,[0m
[38;5;238m 506[0m   [38;5;238m│[0m [38;5;246m        'remaining_entries', v_result.remaining_entries,[0m
[38;5;238m 507[0m   [38;5;238m│[0m [38;5;246m        'remaining_size_mb', v_result.remaining_size_mb,[0m
[38;5;238m 508[0m   [38;5;238m│[0m [38;5;246m        'duration_ms', v_duration_ms[0m
[38;5;238m 509[0m   [38;5;238m│[0m [38;5;246m    );[0m
[38;5;238m 510[0m   [38;5;238m│[0m [38;5;246mEND;[0m
[38;5;238m 511[0m   [38;5;238m│[0m [38;5;246m$$;[0m
[38;5;238m 512[0m   [38;5;238m│[0m 
[38;5;238m 513[0m   [38;5;238m│[0m [38;5;246m-- AIDEV-SECTION: CACHE_MAINTENANCE_FUNCTIONS[0m
[38;5;238m 514[0m   [38;5;238m│[0m [38;5;246m-- Function to analyze cache usage patterns[0m
[38;5;238m 515[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_cache_analyze_usage()[0m
[38;5;238m 516[0m   [38;5;238m│[0m [38;5;246mRETURNS TABLE([0m
[38;5;238m 517[0m   [38;5;238m│[0m [38;5;246m    access_bucket TEXT,[0m
[38;5;238m 518[0m   [38;5;238m│[0m [38;5;246m    entry_count BIGINT,[0m
[38;5;238m 519[0m   [38;5;238m│[0m [38;5;246m    avg_frecency_score FLOAT,[0m
[38;5;238m 520[0m   [38;5;238m│[0m [38;5;246m    total_size_mb FLOAT,[0m
[38;5;238m 521[0m   [38;5;238m│[0m [38;5;246m    percentage_of_cache FLOAT[0m
[38;5;238m 522[0m   [38;5;238m│[0m [38;5;246m)[0m
[38;5;238m 523[0m   [38;5;238m│[0m [38;5;246mLANGUAGE sql[0m
[38;5;238m 524[0m   [38;5;238m│[0m [38;5;246mSTABLE PARALLEL SAFE[0m
[38;5;238m 525[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m 526[0m   [38;5;238m│[0m [38;5;246m    WITH cache_buckets AS ([0m
[38;5;238m 527[0m   [38;5;238m│[0m [38;5;246m        SELECT [0m
[38;5;238m 528[0m   [38;5;238m│[0m [38;5;246m            CASE [0m
[38;5;238m 529[0m   [38;5;238m│[0m [38;5;246m                WHEN access_count = 1 THEN '1_single_access'[0m
[38;5;238m 530[0m   [38;5;238m│[0m [38;5;246m                WHEN access_count BETWEEN 2 AND 5 THEN '2_low_access'[0m
[38;5;238m 531[0m   [38;5;238m│[0m [38;5;246m                WHEN access_count BETWEEN 6 AND 20 THEN '3_medium_access'[0m
[38;5;238m 532[0m   [38;5;238m│[0m [38;5;246m                WHEN access_count BETWEEN 21 AND 100 THEN '4_high_access'[0m
[38;5;238m 533[0m   [38;5;238m│[0m [38;5;246m                ELSE '5_very_high_access'[0m
[38;5;238m 534[0m   [38;5;238m│[0m [38;5;246m            END as access_bucket,[0m
[38;5;238m 535[0m   [38;5;238m│[0m [38;5;246m            COUNT(*) as entry_count,[0m
[38;5;238m 536[0m   [38;5;238m│[0m [38;5;246m            AVG(access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0)) as avg_frecency_score,[0m
[38;5;238m 537[0m   [38;5;238m│[0m [38;5;246m            SUM(pg_column_size(response) + pg_column_size(embedding)) / 1024.0 / 1024.0 as total_size_mb[0m
[38;5;238m 538[0m   [38;5;238m│[0m [38;5;246m        FROM steadytext_cache[0m
[38;5;238m 539[0m   [38;5;238m│[0m [38;5;246m        GROUP BY access_bucket[0m
[38;5;238m 540[0m   [38;5;238m│[0m [38;5;246m    ),[0m
[38;5;238m 541[0m   [38;5;238m│[0m [38;5;246m    totals AS ([0m
[38;5;238m 542[0m   [38;5;238m│[0m [38;5;246m        SELECT [0m
[38;5;238m 543[0m   [38;5;238m│[0m [38;5;246m            SUM(entry_count) as total_entries[0m
[38;5;238m 544[0m   [38;5;238m│[0m [38;5;246m        FROM cache_buckets[0m
[38;5;238m 545[0m   [38;5;238m│[0m [38;5;246m    )[0m
[38;5;238m 546[0m   [38;5;238m│[0m [38;5;246m    SELECT [0m
[38;5;238m 547[0m   [38;5;238m│[0m [38;5;246m        cb.access_bucket,[0m
[38;5;238m 548[0m   [38;5;238m│[0m [38;5;246m        cb.entry_count,[0m
[38;5;238m 549[0m   [38;5;238m│[0m [38;5;246m        cb.avg_frecency_score,[0m
[38;5;238m 550[0m   [38;5;238m│[0m [38;5;246m        cb.total_size_mb,[0m
[38;5;238m 551[0m   [38;5;238m│[0m [38;5;246m        (cb.entry_count::FLOAT / NULLIF(t.total_entries, 0) * 100)::FLOAT as percentage_of_cache[0m
[38;5;238m 552[0m   [38;5;238m│[0m [38;5;246m    FROM cache_buckets cb, totals t[0m
[38;5;238m 553[0m   [38;5;238m│[0m [38;5;246m    ORDER BY cb.access_bucket;[0m
[38;5;238m 554[0m   [38;5;238m│[0m [38;5;246m$$;[0m
[38;5;238m 555[0m   [38;5;238m│[0m 
[38;5;238m 556[0m   [38;5;238m│[0m [38;5;246m-- Function to get cache entries that would be evicted[0m
[38;5;238m 557[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_cache_preview_eviction([0m
[38;5;238m 558[0m   [38;5;238m│[0m [38;5;246m    preview_count INT DEFAULT 10[0m
[38;5;238m 559[0m   [38;5;238m│[0m [38;5;246m)[0m
[38;5;238m 560[0m   [38;5;238m│[0m [38;5;246mRETURNS TABLE([0m
[38;5;238m 561[0m   [38;5;238m│[0m [38;5;246m    cache_key TEXT,[0m
[38;5;238m 562[0m   [38;5;238m│[0m [38;5;246m    prompt TEXT,[0m
[38;5;238m 563[0m   [38;5;238m│[0m [38;5;246m    access_count INT,[0m
[38;5;238m 564[0m   [38;5;238m│[0m [38;5;246m    last_accessed TIMESTAMPTZ,[0m
[38;5;238m 565[0m   [38;5;238m│[0m [38;5;246m    created_at TIMESTAMPTZ,[0m
[38;5;238m 566[0m   [38;5;238m│[0m [38;5;246m    frecency_score FLOAT,[0m
[38;5;238m 567[0m   [38;5;238m│[0m [38;5;246m    size_bytes BIGINT[0m
[38;5;238m 568[0m   [38;5;238m│[0m [38;5;246m)[0m
[38;5;238m 569[0m   [38;5;238m│[0m [38;5;246mLANGUAGE sql[0m
[38;5;238m 570[0m   [38;5;238m│[0m [38;5;246mSTABLE PARALLEL SAFE[0m
[38;5;238m 571[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m 572[0m   [38;5;238m│[0m [38;5;246m    SELECT [0m
[38;5;238m 573[0m   [38;5;238m│[0m [38;5;246m        cache_key,[0m
[38;5;238m 574[0m   [38;5;238m│[0m [38;5;246m        LEFT(prompt, 100) as prompt,  -- Truncate for display[0m
[38;5;238m 575[0m   [38;5;238m│[0m [38;5;246m        access_count,[0m
[38;5;238m 576[0m   [38;5;238m│[0m [38;5;246m        last_accessed,[0m
[38;5;238m 577[0m   [38;5;238m│[0m [38;5;246m        created_at,[0m
[38;5;238m 578[0m   [38;5;238m│[0m [38;5;246m        access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0) as frecency_score,[0m
[38;5;238m 579[0m   [38;5;238m│[0m [38;5;246m        pg_column_size(response) + pg_column_size(embedding) as size_bytes[0m
[38;5;238m 580[0m   [38;5;238m│[0m [38;5;246m    FROM steadytext_cache[0m
[38;5;238m 581[0m   [38;5;238m│[0m [38;5;246m    WHERE [0m
[38;5;238m 582[0m   [38;5;238m│[0m [38;5;246m        -- Same criteria as eviction function[0m
[38;5;238m 583[0m   [38;5;238m│[0m [38;5;246m        access_count < 2[0m
[38;5;238m 584[0m   [38;5;238m│[0m [38;5;246m        AND created_at < NOW() - INTERVAL '1 hour'[0m
[38;5;238m 585[0m   [38;5;238m│[0m [38;5;246m    ORDER BY [0m
[38;5;238m 586[0m   [38;5;238m│[0m [38;5;246m        access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0) ASC[0m
[38;5;238m 587[0m   [38;5;238m│[0m [38;5;246m    LIMIT preview_count;[0m
[38;5;238m 588[0m   [38;5;238m│[0m [38;5;246m$$;[0m
[38;5;238m 589[0m   [38;5;238m│[0m 
[38;5;238m 590[0m   [38;5;238m│[0m [38;5;246m-- AIDEV-SECTION: PG_CRON_SETUP[0m
[38;5;238m 591[0m   [38;5;238m│[0m [38;5;246m-- Note: pg_cron must be installed and configured separately[0m
[38;5;238m 592[0m   [38;5;238m│[0m [38;5;246m-- This creates a helper function to set up the cron job[0m
[38;5;238m 593[0m   [38;5;238m│[0m 
[38;5;238m 594[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_setup_cache_eviction_cron()[0m
[38;5;238m 595[0m   [38;5;238m│[0m [38;5;246mRETURNS TEXT[0m
[38;5;238m 596[0m   [38;5;238m│[0m [38;5;246mLANGUAGE plpgsql[0m
[38;5;238m 597[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m 598[0m   [38;5;238m│[0m [38;5;246mDECLARE[0m
[38;5;238m 599[0m   [38;5;238m│[0m [38;5;246m    v_interval INT;[0m
[38;5;238m 600[0m   [38;5;238m│[0m [38;5;246m    v_cron_expression TEXT;[0m
[38;5;238m 601[0m   [38;5;238m│[0m [38;5;246m    v_job_id BIGINT;[0m
[38;5;238m 602[0m   [38;5;238m│[0m [38;5;246m    v_host TEXT;[0m
[38;5;238m 603[0m   [38;5;238m│[0m [38;5;246m    v_port INT;[0m
[38;5;238m 604[0m   [38;5;238m│[0m [38;5;246mBEGIN[0m
[38;5;238m 605[0m   [38;5;238m│[0m [38;5;246m    -- Check if pg_cron is available[0m
[38;5;238m 606[0m   [38;5;238m│[0m [38;5;246m    IF NOT EXISTS ([0m
[38;5;238m 607[0m   [38;5;238m│[0m [38;5;246m        SELECT 1 FROM pg_extension WHERE extname = 'pg_cron'[0m
[38;5;238m 608[0m   [38;5;238m│[0m [38;5;246m    ) THEN[0m
[38;5;238m 609[0m   [38;5;238m│[0m [38;5;246m        RETURN 'Error: pg_cron extension is not installed. Please install pg_cron first.';[0m
[38;5;238m 610[0m   [38;5;238m│[0m [38;5;246m    END IF;[0m
[38;5;238m 611[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 612[0m   [38;5;238m│[0m [38;5;246m    -- Get configuration values[0m
[38;5;238m 613[0m   [38;5;238m│[0m [38;5;246m    SELECT value::INT INTO v_interval[0m
[38;5;238m 614[0m   [38;5;238m│[0m [38;5;246m    FROM steadytext_config[0m
[38;5;238m 615[0m   [38;5;238m│[0m [38;5;246m    WHERE key = 'cache_eviction_interval';[0m
[38;5;238m 616[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 617[0m   [38;5;238m│[0m [38;5;246m    SELECT value INTO v_host[0m
[38;5;238m 618[0m   [38;5;238m│[0m [38;5;246m    FROM steadytext_config[0m
[38;5;238m 619[0m   [38;5;238m│[0m [38;5;246m    WHERE key = 'cron_host';[0m
[38;5;238m 620[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 621[0m   [38;5;238m│[0m [38;5;246m    SELECT value::INT INTO v_port[0m
[38;5;238m 622[0m   [38;5;238m│[0m [38;5;246m    FROM steadytext_config[0m
[38;5;238m 623[0m   [38;5;238m│[0m [38;5;246m    WHERE key = 'cron_port';[0m
[38;5;238m 624[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 625[0m   [38;5;238m│[0m [38;5;246m    -- Set defaults if not configured[0m
[38;5;238m 626[0m   [38;5;238m│[0m [38;5;246m    v_interval := COALESCE(v_interval, 300); -- Default 5 minutes[0m
[38;5;238m 627[0m   [38;5;238m│[0m [38;5;246m    v_host := COALESCE(v_host, '"localhost"');[0m
[38;5;238m 628[0m   [38;5;238m│[0m [38;5;246m    v_port := COALESCE(v_port, 5432);[0m
[38;5;238m 629[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 630[0m   [38;5;238m│[0m [38;5;246m    -- Validate configuration values[0m
[38;5;238m 631[0m   [38;5;238m│[0m [38;5;246m    IF v_interval < 60 THEN[0m
[38;5;238m 632[0m   [38;5;238m│[0m [38;5;246m        RETURN 'Error: cache_eviction_interval must be at least 60 seconds';[0m
[38;5;238m 633[0m   [38;5;238m│[0m [38;5;246m    END IF;[0m
[38;5;238m 634[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 635[0m   [38;5;238m│[0m [38;5;246m    IF v_host = '' OR v_host IS NULL THEN[0m
[38;5;238m 636[0m   [38;5;238m│[0m [38;5;246m        RETURN 'Error: cron_host cannot be empty';[0m
[38;5;238m 637[0m   [38;5;238m│[0m [38;5;246m    END IF;[0m
[38;5;238m 638[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 639[0m   [38;5;238m│[0m [38;5;246m    IF v_port < 1 OR v_port > 65535 THEN[0m
[38;5;238m 640[0m   [38;5;238m│[0m [38;5;246m        RETURN 'Error: cron_port must be between 1 and 65535';[0m
[38;5;238m 641[0m   [38;5;238m│[0m [38;5;246m    END IF;[0m
[38;5;238m 642[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 643[0m   [38;5;238m│[0m [38;5;246m    -- Convert seconds to cron expression[0m
[38;5;238m 644[0m   [38;5;238m│[0m [38;5;246m    -- For intervals less than an hour, use minute-based scheduling[0m
[38;5;238m 645[0m   [38;5;238m│[0m [38;5;246m    IF v_interval < 3600 THEN[0m
[38;5;238m 646[0m   [38;5;238m│[0m [38;5;246m        -- Ensure minimum 1-minute interval to avoid invalid cron expressions[0m
[38;5;238m 647[0m   [38;5;238m│[0m [38;5;246m        v_cron_expression := '*/' || GREATEST(1, v_interval / 60)::TEXT || ' * * * *';[0m
[38;5;238m 648[0m   [38;5;238m│[0m [38;5;246m    ELSE[0m
[38;5;238m 649[0m   [38;5;238m│[0m [38;5;246m        -- For longer intervals, use hourly scheduling[0m
[38;5;238m 650[0m   [38;5;238m│[0m [38;5;246m        -- Ensure minimum 1-hour interval to avoid invalid cron expressions[0m
[38;5;238m 651[0m   [38;5;238m│[0m [38;5;246m        v_cron_expression := '0 */' || GREATEST(1, v_interval / 3600)::TEXT || ' * * *';[0m
[38;5;238m 652[0m   [38;5;238m│[0m [38;5;246m    END IF;[0m
[38;5;238m 653[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 654[0m   [38;5;238m│[0m [38;5;246m    -- Remove existing job if any[0m
[38;5;238m 655[0m   [38;5;238m│[0m [38;5;246m    DELETE FROM cron.job [0m
[38;5;238m 656[0m   [38;5;238m│[0m [38;5;246m    WHERE jobname = 'steadytext_cache_eviction';[0m
[38;5;238m 657[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 658[0m   [38;5;238m│[0m [38;5;246m    -- Schedule the job with error handling[0m
[38;5;238m 659[0m   [38;5;238m│[0m [38;5;246m    BEGIN[0m
[38;5;238m 660[0m   [38;5;238m│[0m [38;5;246m        INSERT INTO cron.job ([0m
[38;5;238m 661[0m   [38;5;238m│[0m [38;5;246m            schedule, [0m
[38;5;238m 662[0m   [38;5;238m│[0m [38;5;246m            command, [0m
[38;5;238m 663[0m   [38;5;238m│[0m [38;5;246m            nodename, [0m
[38;5;238m 664[0m   [38;5;238m│[0m [38;5;246m            nodeport, [0m
[38;5;238m 665[0m   [38;5;238m│[0m [38;5;246m            database, [0m
[38;5;238m 666[0m   [38;5;238m│[0m [38;5;246m            username,[0m
[38;5;238m 667[0m   [38;5;238m│[0m [38;5;246m            jobname[0m
[38;5;238m 668[0m   [38;5;238m│[0m [38;5;246m        ) VALUES ([0m
[38;5;238m 669[0m   [38;5;238m│[0m [38;5;246m            v_cron_expression,[0m
[38;5;238m 670[0m   [38;5;238m│[0m [38;5;246m            'SELECT steadytext_cache_scheduled_eviction();',[0m
[38;5;238m 671[0m   [38;5;238m│[0m [38;5;246m            v_host,[0m
[38;5;238m 672[0m   [38;5;238m│[0m [38;5;246m            v_port,[0m
[38;5;238m 673[0m   [38;5;238m│[0m [38;5;246m            current_database(),[0m
[38;5;238m 674[0m   [38;5;238m│[0m [38;5;246m            current_user,[0m
[38;5;238m 675[0m   [38;5;238m│[0m [38;5;246m            'steadytext_cache_eviction'[0m
[38;5;238m 676[0m   [38;5;238m│[0m [38;5;246m        ) RETURNING jobid INTO v_job_id;[0m
[38;5;238m 677[0m   [38;5;238m│[0m [38;5;246m        [0m
[38;5;238m 678[0m   [38;5;238m│[0m [38;5;246m        RETURN 'Cache eviction cron job scheduled with ID ' || v_job_id || [0m
[38;5;238m 679[0m   [38;5;238m│[0m [38;5;246m               ' using schedule: ' || v_cron_expression ||[0m
[38;5;238m 680[0m   [38;5;238m│[0m [38;5;246m               ' on host: ' || v_host || ':' || v_port;[0m
[38;5;238m 681[0m   [38;5;238m│[0m [38;5;246m    EXCEPTION[0m
[38;5;238m 682[0m   [38;5;238m│[0m [38;5;246m        WHEN OTHERS THEN[0m
[38;5;238m 683[0m   [38;5;238m│[0m [38;5;246m            RETURN 'Error scheduling cron job: ' || SQLERRM || [0m
[38;5;238m 684[0m   [38;5;238m│[0m [38;5;246m                   ' (schedule: ' || v_cron_expression || [0m
[38;5;238m 685[0m   [38;5;238m│[0m [38;5;246m                   ', host: ' || v_host || ':' || v_port || ')';[0m
[38;5;238m 686[0m   [38;5;238m│[0m [38;5;246m    END;[0m
[38;5;238m 687[0m   [38;5;238m│[0m [38;5;246mEND;[0m
[38;5;238m 688[0m   [38;5;238m│[0m [38;5;246m$$;[0m
[38;5;238m 689[0m   [38;5;238m│[0m 
[38;5;238m 690[0m   [38;5;238m│[0m [38;5;246m-- Helper function to disable cache eviction cron[0m
[38;5;238m 691[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_disable_cache_eviction_cron()[0m
[38;5;238m 692[0m   [38;5;238m│[0m [38;5;246mRETURNS TEXT[0m
[38;5;238m 693[0m   [38;5;238m│[0m [38;5;246mLANGUAGE plpgsql[0m
[38;5;238m 694[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m 695[0m   [38;5;238m│[0m [38;5;246mDECLARE[0m
[38;5;238m 696[0m   [38;5;238m│[0m [38;5;246m    v_deleted_count INT;[0m
[38;5;238m 697[0m   [38;5;238m│[0m [38;5;246mBEGIN[0m
[38;5;238m 698[0m   [38;5;238m│[0m [38;5;246m    -- Check if pg_cron is available[0m
[38;5;238m 699[0m   [38;5;238m│[0m [38;5;246m    IF NOT EXISTS ([0m
[38;5;238m 700[0m   [38;5;238m│[0m [38;5;246m        SELECT 1 FROM pg_extension WHERE extname = 'pg_cron'[0m
[38;5;238m 701[0m   [38;5;238m│[0m [38;5;246m    ) THEN[0m
[38;5;238m 702[0m   [38;5;238m│[0m [38;5;246m        RETURN 'Error: pg_cron extension is not installed.';[0m
[38;5;238m 703[0m   [38;5;238m│[0m [38;5;246m    END IF;[0m
[38;5;238m 704[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 705[0m   [38;5;238m│[0m [38;5;246m    -- Remove the job[0m
[38;5;238m 706[0m   [38;5;238m│[0m [38;5;246m    DELETE FROM cron.job [0m
[38;5;238m 707[0m   [38;5;238m│[0m [38;5;246m    WHERE jobname = 'steadytext_cache_eviction';[0m
[38;5;238m 708[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 709[0m   [38;5;238m│[0m [38;5;246m    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;[0m
[38;5;238m 710[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 711[0m   [38;5;238m│[0m [38;5;246m    IF v_deleted_count > 0 THEN[0m
[38;5;238m 712[0m   [38;5;238m│[0m [38;5;246m        RETURN 'Cache eviction cron job disabled successfully.';[0m
[38;5;238m 713[0m   [38;5;238m│[0m [38;5;246m    ELSE[0m
[38;5;238m 714[0m   [38;5;238m│[0m [38;5;246m        RETURN 'No cache eviction cron job found.';[0m
[38;5;238m 715[0m   [38;5;238m│[0m [38;5;246m    END IF;[0m
[38;5;238m 716[0m   [38;5;238m│[0m [38;5;246mEND;[0m
[38;5;238m 717[0m   [38;5;238m│[0m [38;5;246m$$;[0m
[38;5;238m 718[0m   [38;5;238m│[0m 
[38;5;238m 719[0m   [38;5;238m│[0m [38;5;246m-- AIDEV-SECTION: CACHE_WARMUP_FUNCTIONS[0m
[38;5;238m 720[0m   [38;5;238m│[0m [38;5;246m-- Function to warm up cache with frequently accessed entries[0m
[38;5;238m 721[0m   [38;5;238m│[0m [38;5;246mCREATE OR REPLACE FUNCTION steadytext_cache_warmup([0m
[38;5;238m 722[0m   [38;5;238m│[0m [38;5;246m    warmup_count INT DEFAULT 100[0m
[38;5;238m 723[0m   [38;5;238m│[0m [38;5;246m)[0m
[38;5;238m 724[0m   [38;5;238m│[0m [38;5;246mRETURNS TABLE([0m
[38;5;238m 725[0m   [38;5;238m│[0m [38;5;246m    warmed_entries INT,[0m
[38;5;238m 726[0m   [38;5;238m│[0m [38;5;246m    total_time_ms INT[0m
[38;5;238m 727[0m   [38;5;238m│[0m [38;5;246m)[0m
[38;5;238m 728[0m   [38;5;238m│[0m [38;5;246mLANGUAGE plpgsql[0m
[38;5;238m 729[0m   [38;5;238m│[0m [38;5;246mAS $$[0m
[38;5;238m 730[0m   [38;5;238m│[0m [38;5;246mDECLARE[0m
[38;5;238m 731[0m   [38;5;238m│[0m [38;5;246m    v_start_time TIMESTAMPTZ;[0m
[38;5;238m 732[0m   [38;5;238m│[0m [38;5;246m    v_end_time TIMESTAMPTZ;[0m
[38;5;238m 733[0m   [38;5;238m│[0m [38;5;246m    v_warmed_count INT := 0;[0m
[38;5;238m 734[0m   [38;5;238m│[0m [38;5;246m    v_cache_entry RECORD;[0m
[38;5;238m 735[0m   [38;5;238m│[0m [38;5;246mBEGIN[0m
[38;5;238m 736[0m   [38;5;238m│[0m [38;5;246m    v_start_time := clock_timestamp();[0m
[38;5;238m 737[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 738[0m   [38;5;238m│[0m [38;5;246m    -- Find entries that are frequently accessed but not in memory[0m
[38;5;238m 739[0m   [38;5;238m│[0m [38;5;246m    FOR v_cache_entry IN [0m
[38;5;238m 740[0m   [38;5;238m│[0m [38;5;246m        SELECT cache_key, prompt, generation_params[0m
[38;5;238m 741[0m   [38;5;238m│[0m [38;5;246m        FROM steadytext_cache[0m
[38;5;238m 742[0m   [38;5;238m│[0m [38;5;246m        WHERE access_count > 5[0m
[38;5;238m 743[0m   [38;5;238m│[0m [38;5;246m        ORDER BY access_count DESC[0m
[38;5;238m 744[0m   [38;5;238m│[0m [38;5;246m        LIMIT warmup_count[0m
[38;5;238m 745[0m   [38;5;238m│[0m [38;5;246m    LOOP[0m
[38;5;238m 746[0m   [38;5;238m│[0m [38;5;246m        -- Touch the cache entry to warm it up[0m
[38;5;238m 747[0m   [38;5;238m│[0m [38;5;246m        UPDATE steadytext_cache[0m
[38;5;238m 748[0m   [38;5;238m│[0m [38;5;246m        SET last_accessed = NOW()[0m
[38;5;238m 749[0m   [38;5;238m│[0m [38;5;246m        WHERE cache_key = v_cache_entry.cache_key;[0m
[38;5;238m 750[0m   [38;5;238m│[0m [38;5;246m        [0m
[38;5;238m 751[0m   [38;5;238m│[0m [38;5;246m        v_warmed_count := v_warmed_count + 1;[0m
[38;5;238m 752[0m   [38;5;238m│[0m [38;5;246m    END LOOP;[0m
[38;5;238m 753[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 754[0m   [38;5;238m│[0m [38;5;246m    v_end_time := clock_timestamp();[0m
[38;5;238m 755[0m   [38;5;238m│[0m [38;5;246m    [0m
[38;5;238m 756[0m   [38;5;238m│[0m [38;5;246m    RETURN QUERY[0m
[38;5;238m 757[0m   [38;5;238m│[0m [38;5;246m    SELECT [0m
[38;5;238m 758[0m   [38;5;238m│[0m [38;5;246m        v_warmed_count,[0m
[38;5;238m 759[0m   [38;5;238m│[0m [38;5;246m        EXTRACT(MILLISECONDS FROM (v_end_time - v_start_time))::INT;[0m
[38;5;238m 760[0m   [38;5;238m│[0m [38;5;246mEND;[0m
[38;5;238m 761[0m   [38;5;238m│[0m [38;5;246m$$;[0m
[38;5;238m 762[0m   [38;5;238m│[0m 
[38;5;238m 763[0m   [38;5;238m│[0m [38;5;246m-- Add helpful comments[0m
[38;5;238m 764[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_cache_stats_extended() IS [0m
[38;5;238m 765[0m   [38;5;238m│[0m [38;5;246m'Extended cache statistics including eviction candidate analysis';[0m
[38;5;238m 766[0m   [38;5;238m│[0m 
[38;5;238m 767[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_cache_evict_by_frecency(INT, FLOAT, INT, INT, INT) IS [0m
[38;5;238m 768[0m   [38;5;238m│[0m [38;5;246m'Evict cache entries based on frecency score to maintain size/capacity limits';[0m
[38;5;238m 769[0m   [38;5;238m│[0m 
[38;5;238m 770[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_cache_scheduled_eviction() IS [0m
[38;5;238m 771[0m   [38;5;238m│[0m [38;5;246m'Scheduled cache eviction function to be called by pg_cron';[0m
[38;5;238m 772[0m   [38;5;238m│[0m 
[38;5;238m 773[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_cache_analyze_usage() IS [0m
[38;5;238m 774[0m   [38;5;238m│[0m [38;5;246m'Analyze cache usage patterns by access frequency buckets';[0m
[38;5;238m 775[0m   [38;5;238m│[0m 
[38;5;238m 776[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_cache_preview_eviction(INT) IS [0m
[38;5;238m 777[0m   [38;5;238m│[0m [38;5;246m'Preview which cache entries would be evicted next';[0m
[38;5;238m 778[0m   [38;5;238m│[0m 
[38;5;238m 779[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_setup_cache_eviction_cron() IS [0m
[38;5;238m 780[0m   [38;5;238m│[0m [38;5;246m'Set up pg_cron job for automatic cache eviction (requires pg_cron extension)';[0m
[38;5;238m 781[0m   [38;5;238m│[0m 
[38;5;238m 782[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_disable_cache_eviction_cron() IS [0m
[38;5;238m 783[0m   [38;5;238m│[0m [38;5;246m'Disable the pg_cron job for automatic cache eviction';[0m
[38;5;238m 784[0m   [38;5;238m│[0m 
[38;5;238m 785[0m   [38;5;238m│[0m [38;5;246mCOMMENT ON FUNCTION steadytext_cache_warmup(INT) IS [0m
[38;5;238m 786[0m   [38;5;238m│[0m [38;5;246m'Warm up cache by touching frequently accessed entries';[0m
[38;5;238m 787[0m   [38;5;238m│[0m 
[38;5;238m 788[0m   [38;5;238m│[0m [38;5;246m-- AIDEV-NOTE: This completes the cache eviction implementation for pg_steadytext v1.4.0[0m
[38;5;238m 789[0m   [38;5;238m│[0m [38;5;246m-- [0m
[38;5;238m 790[0m   [38;5;238m│[0m [38;5;246m-- To enable automatic cache eviction:[0m
[38;5;238m 791[0m   [38;5;238m│[0m [38;5;246m-- 1. Install pg_cron extension: CREATE EXTENSION pg_cron;[0m
[38;5;238m 792[0m   [38;5;238m│[0m [38;5;246m-- 2. Run: SELECT steadytext_setup_cache_eviction_cron();[0m
[38;5;238m 793[0m   [38;5;238m│[0m [38;5;246m-- 3. Configure parameters in steadytext_config table[0m
[38;5;238m 794[0m   [38;5;238m│[0m [38;5;246m-- [0m
[38;5;238m 795[0m   [38;5;238m│[0m [38;5;246m-- The eviction algorithm uses frecency (frequency + recency) to determine[0m
[38;5;238m 796[0m   [38;5;238m│[0m [38;5;246m-- which entries to evict, protecting high-access and recent entries.[0m
[38;5;238m 797[0m   [38;5;238m│[0m [38;5;246m--[0m
[38;5;238m 798[0m   [38;5;238m│[0m [38;5;246m-- AIDEV-TODO: Future enhancements could include:[0m
[38;5;238m 799[0m   [38;5;238m│[0m [38;5;246m-- - Adaptive eviction thresholds based on cache hit rates[0m
[38;5;238m 800[0m   [38;5;238m│[0m [38;5;246m-- - Different eviction strategies (LRU, LFU, ARC)[0m
[38;5;238m 801[0m   [38;5;238m│[0m [38;5;246m-- - Cache partitioning by model or use case[0m
[38;5;238m 802[0m   [38;5;238m│[0m [38;5;246m-- - Integration with PostgreSQL's shared buffer cache[0m
[38;5;238m───────┴────────────────────────────────────────────────────────────────────────[0m

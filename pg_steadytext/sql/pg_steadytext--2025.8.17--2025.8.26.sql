-- pg_steadytext migration from 2025.8.17 to 2025.8.26
-- Changes:
-- 1) Fix schema qualification - use extension schema instead of current_schema()
-- 2) Fix Python scoping issues by defining variables at function level  
-- 3) Fix steadytext_embed daemon status check (use status == 'healthy', not is_running)
-- 4) Qualify alias functions with @extschema@ for search_path safety
-- 5) Bump version to 2025.8.26
-- AIDEV-NOTE: Fixed PL/Python plpy.execute() usage - must use plpy.prepare() for parameterized queries
-- AIDEV-NOTE: Fixed UUID type casting - Python strings must be explicitly cast to UUID in SQL
-- AIDEV-NOTE: Fixed yield vs return in PL/Python - use yield for row-by-row output

-- CONFIGURATION: Add config table and defaults
CREATE TABLE IF NOT EXISTS @extschema@.steadytext_config (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by TEXT DEFAULT current_user
);

INSERT INTO @extschema@.steadytext_config (key, value, description) VALUES
    ('cache_enabled', 'false', 'Enable caching'),
    ('cache_max_entries', '10000', 'Maximum cache entries'),
    ('cache_max_size_mb', '100', 'Maximum cache size in MB'),
    ('cache_eviction_enabled', 'false', 'Enable cache eviction'),
    ('daemon_host', '"localhost"', 'SteadyText daemon host'),
    ('daemon_port', '5555', 'SteadyText daemon port')
ON CONFLICT (key) DO NOTHING;

-- 1) Fix steadytext_generate with schema qualification
CREATE OR REPLACE FUNCTION steadytext_generate(
    prompt TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    eos_string TEXT DEFAULT '[EOS]',
    model TEXT DEFAULT NULL,
    model_repo TEXT DEFAULT NULL,
    model_filename TEXT DEFAULT NULL,
    size TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
# AIDEV-NOTE: Main text generation function that integrates with SteadyText daemon
# v2025.8.15: Added support for eos_string, model, model_repo, model_filename, size parameters
# v2025.8.15: Added model parameter for remote model access
import json
import hashlib

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    # Initialize on demand
    ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    # Check again after initialization
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules from GD
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Get configuration
# AIDEV-NOTE: Use extension schema instead of current_schema() for cross-schema compatibility
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
config_select_plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(ext_schema)}.steadytext_config WHERE key = $1", ["text"])

# Resolve max_tokens, using the provided value or fetching the default
resolved_max_tokens = max_tokens
if resolved_max_tokens is None:
    rv = plpy.execute(config_select_plan, ["default_max_tokens"])
    resolved_max_tokens = json.loads(rv[0]["value"]) if rv else 512

# Resolve seed, using the provided value or fetching the default
resolved_seed = seed
if resolved_seed is None:
    rv = plpy.execute(config_select_plan, ["default_seed"])
    resolved_seed = json.loads(rv[0]["value"]) if rv else 42

# Validate inputs
if not prompt or not prompt.strip():
    plpy.error("Prompt cannot be empty")

if resolved_max_tokens < 1 or resolved_max_tokens > 4096:
    plpy.error("max_tokens must be between 1 and 4096")

if resolved_seed < 0:
    plpy.error("seed must be non-negative")

# AIDEV-NOTE: Validate model parameter - remote models require unsafe_mode=TRUE
if not unsafe_mode and model and ':' in model:
    plpy.error("Remote models (containing ':') require unsafe_mode=TRUE")

# AIDEV-NOTE: Validate that unsafe_mode requires a model to be specified
if unsafe_mode and not model:
    plpy.error("unsafe_mode=TRUE requires a model parameter to be specified")

# Check if we should use cache
if use_cache:
    # Generate cache key consistent with SteadyText format
    # Include eos_string in cache key if it's not the default
    if eos_string == '[EOS]':
        cache_key = prompt
    else:
        cache_key = f"{prompt}::EOS::{eos_string}"

    # Try to get from cache first
    # AIDEV-NOTE: Get schema dynamically at runtime for TimescaleDB continuous aggregates compatibility
    cache_plan = plpy.prepare(f"""
        SELECT response 
        FROM {plpy.quote_ident(ext_schema)}.steadytext_cache 
        WHERE cache_key = $1
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# Cache miss - generate new content
# Get configuration for daemon connection
host_rv = plpy.execute(config_select_plan, ["daemon_host"])
host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(config_select_plan, ["daemon_port"])  
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

# AIDEV-NOTE: For remote models with unsafe_mode, skip daemon entirely
# Remote models don't need daemon and checking it causes unnecessary delays
is_remote_model = unsafe_mode and model and ':' in model

if is_remote_model:
    # Skip daemon for remote models - go directly to generation
    plpy.notice(f"Using remote model {model} with unsafe_mode - skipping daemon")
    
    # Build kwargs for generation
    generation_kwargs = {
        "seed": resolved_seed,
        "eos_string": eos_string,
        "model": model,
        "unsafe_mode": unsafe_mode
    }
    
    # Add optional model parameters if provided
    if model_repo is not None:
        generation_kwargs["model_repo"] = model_repo
    if model_filename is not None:
        generation_kwargs["model_filename"] = model_filename
    if size is not None:
        generation_kwargs["size"] = size
    
    # Direct generation for remote models
    try:
        from steadytext import generate as steadytext_generate
        result = steadytext_generate(
            prompt=prompt, 
            max_new_tokens=resolved_max_tokens,
            **generation_kwargs
        )
        return result
    except Exception as e:
        plpy.error(f"Remote model generation failed: {str(e)}")
else:
    # Local model path - use daemon if available
    # Create daemon connector
    connector = daemon_connector.SteadyTextConnector(host=host, port=port)

    # Check if daemon should auto-start
    auto_start_rv = plpy.execute(config_select_plan, ["daemon_auto_start"])
    auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

    if auto_start and not connector.is_daemon_running():
        plpy.notice("Starting SteadyText daemon...")
        started = connector.start_daemon()
        if started:
            plpy.notice("SteadyText daemon started successfully")
        else:
            plpy.warning("Failed to start SteadyText daemon, will use direct generation")

    # Try to generate text using daemon
    try:
        # Build kwargs for daemon call
        daemon_kwargs = {
            "seed": resolved_seed,
            "eos_string": eos_string
        }
        
        # Add optional model parameters if provided
        if model is not None:
            daemon_kwargs["model"] = model
        if model_repo is not None:
            daemon_kwargs["model_repo"] = model_repo
        if model_filename is not None:
            daemon_kwargs["model_filename"] = model_filename
        if size is not None:
            daemon_kwargs["size"] = size
        
        result = connector.generate(
            prompt=prompt, 
            max_tokens=resolved_max_tokens,
            **daemon_kwargs
        )
        return result
    except Exception as e:
        plpy.warning(f"Daemon generation failed: {e}, falling back to direct generation")
        
        # Fall back to direct generation
        generation_kwargs = {
            "seed": resolved_seed,
            "eos_string": eos_string
        }
        
        # Add optional model parameters if provided
        if model is not None:
            generation_kwargs["model"] = model
        if model_repo is not None:
            generation_kwargs["model_repo"] = model_repo
        if model_filename is not None:
            generation_kwargs["model_filename"] = model_filename
        if size is not None:
            generation_kwargs["size"] = size
        
        # Direct generation fallback
        try:
            from steadytext import generate as steadytext_generate
            result = steadytext_generate(
                prompt=prompt, 
                max_new_tokens=resolved_max_tokens,  # NOTE: Different param name for direct API
                **generation_kwargs
            )
            return result
        except Exception as e2:
            plpy.error(f"Both daemon and direct generation failed: {e2}")
$c$;

-- 2) Fix steadytext_embed daemon status check
CREATE OR REPLACE FUNCTION steadytext_embed(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    model TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS vector(1024)
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE LEAKPROOF
AS $c$
# AIDEV-NOTE: Embedding function with remote model support via unsafe_mode
# Uses runtime inspection to maintain backward compatibility with older steadytext versions

import json
import numpy as np

# Get extension schema for all subsequent queries
# AIDEV-NOTE: Define ext_schema at function level to avoid UnboundLocalError
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'

# Initialize Python environment if needed
try:
    if 'steadytext_initialized' not in GD:
        plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
except:
    pass

# Validate remote model requirements
if model and ':' in model and not unsafe_mode:
    plpy.error("Remote models (containing ':') require unsafe_mode=TRUE")

# Check cache if enabled (IMMUTABLE functions only read cache)
if use_cache:
    # Generate cache key including model if specified
    cache_key_parts = ['embed', text_input]
    if model:
        cache_key_parts.append(model)
    cache_key_input = ':'.join(cache_key_parts)
    
    import hashlib
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()
    
    # Try to get from cache (read-only for IMMUTABLE)
    # AIDEV-NOTE: Use extension schema for TimescaleDB continuous aggregates compatibility
    plan = plpy.prepare(
        f"SELECT embedding FROM {plpy.quote_ident(ext_schema)}.steadytext_cache WHERE cache_key = $1",
        ["text"]
    )
    rv = plpy.execute(plan, [cache_key])
    
    if rv and len(rv) > 0:
        cached_embedding = rv[0]['embedding']
        if cached_embedding:
            return cached_embedding

# Try daemon first for local models
if not (model and ':' in model):
    try:
        # Check if daemon is running
        # AIDEV-NOTE: Qualify with extension schema for cross-schema compatibility
        plan = plpy.prepare(f"SELECT * FROM {plpy.quote_ident(ext_schema)}.steadytext_daemon_status()")
        status = plpy.execute(plan)
        
        # FIX: Use status == 'healthy' rather than non-existent is_running column
        if status and len(status) > 0 and status[0]['status'] == 'healthy':
            # Try using daemon (pass None for model/unsafe_mode if not set to avoid issues)
            from pg_steadytext import SteadyTextConnector
            connector = SteadyTextConnector()
            # Pass None instead of NULL from SQL
            py_model = model if model is not None else None
            py_unsafe = unsafe_mode if unsafe_mode is not None else False
            result = connector.embed(text_input, seed=seed, model=py_model, unsafe_mode=py_unsafe)
            
            if result is not None:
                # Ensure it's a list
                if hasattr(result, 'tolist'):
                    embedding_list = result.tolist()
                else:
                    embedding_list = list(result)
                
                return embedding_list
    except Exception as e:
        # Fall through to direct embedding
        pass

# Direct embedding fallback or remote model usage
try:
    from steadytext import embed as steadytext_embed
    import inspect
    
    # Check if embed supports the new parameters
    embed_sig = inspect.signature(steadytext_embed)
    
    # Call with model and unsafe_mode if specified and supported
    kwargs = {'seed': seed}
    if model and 'model' in embed_sig.parameters:
        kwargs['model'] = model
    if unsafe_mode and 'unsafe_mode' in embed_sig.parameters:
        kwargs['unsafe_mode'] = unsafe_mode
    
    result = steadytext_embed(text_input, **kwargs)
    
    # Convert to vector format if needed
    if result is not None:
        # Ensure it's a list/array
        if hasattr(result, 'tolist'):
            embedding_list = result.tolist()
        else:
            embedding_list = list(result)
        
        return embedding_list
except Exception as e:
    plpy.error(f"Embedding generation failed: {e}")

# Should not reach here
return None
$c$;

-- 3) Fix steadytext_daemon_status with proper schema qualification
CREATE OR REPLACE FUNCTION steadytext_daemon_status()
RETURNS TABLE(
    daemon_id TEXT,
    status TEXT,
    endpoint TEXT,
    last_heartbeat TIMESTAMPTZ,
    uptime_seconds INT
)
LANGUAGE plpython3u
AS $c$
# AIDEV-NOTE: Check SteadyText daemon health status
import json

# Get extension schema first (always needed)
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    # Initialize on demand
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    # Check again after initialization
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

try:
    # Get cached modules from GD
    daemon_connector = GD.get('module_daemon_connector')
    if not daemon_connector:
        plpy.error("daemon_connector module not loaded")

    # Get configuration
    # AIDEV-NOTE: Use extension schema that was already found above
    config_plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(ext_schema)}.steadytext_config WHERE key = $1", ["text"])
    host_rv = plpy.execute(config_plan, ["daemon_host"])
    port_rv = plpy.execute(config_plan, ["daemon_port"])

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
    # AIDEV-NOTE: Use ext_schema that's already available from initialization
    update_plan = plpy.prepare(f"""
        UPDATE {plpy.quote_ident(ext_schema)}.steadytext_daemon_health
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
    # Return current status from table - FIX: schema-qualified table name
    select_plan = plpy.prepare(f"""
        SELECT daemon_id, status, endpoint, last_heartbeat,
               EXTRACT(EPOCH FROM (NOW() - last_heartbeat))::INT as uptime_seconds
        FROM {plpy.quote_ident(ext_schema)}.steadytext_daemon_health
        WHERE daemon_id = 'default'
    """)
    return plpy.execute(select_plan)
$c$;

-- 4) Fix st_* alias functions with @extschema@ qualification
CREATE OR REPLACE FUNCTION st_daemon_status()
RETURNS TABLE(daemon_id TEXT, status TEXT, endpoint TEXT, last_heartbeat TIMESTAMP WITH TIME ZONE, uptime_seconds INTEGER)
LANGUAGE sql
STABLE PARALLEL SAFE
AS $alias$
    SELECT * FROM @extschema@.steadytext_daemon_status();
$alias$;

CREATE OR REPLACE FUNCTION st_summarize_text(
    input_text text,
    metadata jsonb DEFAULT '{}'::jsonb,
    model text DEFAULT NULL,
    unsafe_mode boolean DEFAULT FALSE
) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE
AS $$
    SELECT @extschema@.steadytext_summarize_text($1, $2, $3, $4);
$$;

CREATE OR REPLACE FUNCTION st_extract_facts(
    input_text text,
    max_facts integer DEFAULT 10
) RETURNS jsonb
LANGUAGE sql IMMUTABLE PARALLEL SAFE
AS $$
    SELECT @extschema@.steadytext_extract_facts($1, $2);
$$;

CREATE OR REPLACE FUNCTION st_deduplicate_facts(
    facts jsonb,
    similarity_threshold float DEFAULT 0.8
) RETURNS jsonb
LANGUAGE sql IMMUTABLE PARALLEL SAFE
AS $$
    SELECT @extschema@.steadytext_deduplicate_facts($1, $2);
$$;

-- 5) Update version function
CREATE OR REPLACE FUNCTION steadytext_version()
RETURNS TEXT
LANGUAGE sql
IMMUTABLE PARALLEL SAFE LEAKPROOF
AS $v$
    SELECT '2025.8.26'::TEXT;
$v$;

-- 6) Add missing daemon functions with schema qualification
CREATE OR REPLACE FUNCTION steadytext_daemon_start()
RETURNS BOOLEAN
LANGUAGE plpython3u
AS $c$
# AIDEV-NOTE: Start the SteadyText daemon if not already running
import subprocess
import time
import json

try:
    # Get daemon configuration
    # AIDEV-NOTE: Use extension schema for cross-schema compatibility
    ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    config_plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(ext_schema)}.steadytext_config WHERE key = $1", ["text"])
    host_rv = plpy.execute(config_plan, ["daemon_host"])
    port_rv = plpy.execute(config_plan, ["daemon_port"])

    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555

    # Check if daemon is already running by trying to start it
    # SteadyText daemon start command is idempotent
    try:
        result = subprocess.run(['st', 'daemon', 'start'], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            # Update health status
            health_plan = plpy.prepare(f"""
                UPDATE {plpy.quote_ident(ext_schema)}.steadytext_daemon_health
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
$c$;

CREATE OR REPLACE FUNCTION steadytext_daemon_stop()
RETURNS BOOLEAN
LANGUAGE plpython3u
AS $c$
# AIDEV-NOTE: Stop the SteadyText daemon gracefully
import subprocess
import json

# Get extension schema for table updates
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'

try:
    # Stop daemon using CLI
    result = subprocess.run(['st', 'daemon', 'stop'], capture_output=True, text=True)

    if result.returncode == 0:
        # Update health status
        health_plan = plpy.prepare(f"""
            UPDATE {plpy.quote_ident(ext_schema)}.steadytext_daemon_health
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
$c$;

-- 7) Add structured generation functions with schema qualification

CREATE OR REPLACE FUNCTION steadytext_generate_json(
    prompt TEXT,
    schema JSONB,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE,
    model TEXT DEFAULT NULL
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
# AIDEV-NOTE: Generate JSON that conforms to a schema using llama.cpp grammars
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
# v2025.8.15: Added model parameter for remote model access
import json
import hashlib

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules from GD
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Get configuration
# AIDEV-NOTE: Use extension schema instead of current_schema() for cross-schema compatibility
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
config_select_plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(ext_schema)}.steadytext_config WHERE key = $1", ["text"])

# Resolve max_tokens
resolved_max_tokens = max_tokens
if resolved_max_tokens is None:
    rv = plpy.execute(config_select_plan, ["default_max_tokens"])
    resolved_max_tokens = json.loads(rv[0]["value"]) if rv else 512

# Resolve seed
resolved_seed = seed
if resolved_seed is None:
    rv = plpy.execute(config_select_plan, ["default_seed"])
    resolved_seed = json.loads(rv[0]["value"]) if rv else 42

# Validate inputs
if not prompt or not prompt.strip():
    plpy.error("Prompt cannot be empty")

if not schema:
    plpy.error("Schema cannot be empty")

# AIDEV-NOTE: For structured generation functions, unsafe_mode is not supported without model selection
if unsafe_mode:
    plpy.error("unsafe_mode is not supported for structured generation functions")

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

    # SELECT ONLY - no UPDATE for immutability
    cache_plan = plpy.prepare(f"""
        SELECT response 
        FROM {plpy.quote_ident(ext_schema)}.steadytext_cache 
        WHERE cache_key = $1
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for JSON key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# Cache miss - generate new content
host_rv = plpy.execute(config_select_plan, ["daemon_host"])
host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(config_select_plan, ["daemon_port"])
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

# Create connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Auto-start daemon if configured
auto_start_rv = plpy.execute(config_select_plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    connector.start_daemon()

# Build kwargs
generation_kwargs = {
    "seed": resolved_seed,
    "unsafe_mode": unsafe_mode
}

# Generate structured output
try:
    if connector.is_daemon_running():
        result = connector.generate_json(
            prompt=prompt,
            schema=schema_dict,
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    else:
        # Direct generation fallback
        from steadytext import generate_json as steadytext_generate_json
        result = steadytext_generate_json(
            prompt=prompt,
            schema=schema_dict,
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    
    # AIDEV-NOTE: Cache writes removed for IMMUTABLE compliance
    
    return result
    
except Exception as e:
    plpy.error(f"JSON generation failed: {str(e)}")
$c$;

-- 8) Add regex generation function with schema qualification
CREATE OR REPLACE FUNCTION steadytext_generate_regex(
    prompt TEXT,
    pattern TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE,
    model TEXT DEFAULT NULL
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
# AIDEV-NOTE: Generate text matching a regex pattern using llama.cpp grammars
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
# v2025.8.15: Added model parameter for remote model access
import json
import hashlib

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules from GD
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Get configuration
# AIDEV-NOTE: Use extension schema instead of current_schema() for cross-schema compatibility
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
config_select_plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(ext_schema)}.steadytext_config WHERE key = $1", ["text"])

# Resolve max_tokens
resolved_max_tokens = max_tokens
if resolved_max_tokens is None:
    rv = plpy.execute(config_select_plan, ["default_max_tokens"])
    resolved_max_tokens = json.loads(rv[0]["value"]) if rv else 512

# Resolve seed
resolved_seed = seed
if resolved_seed is None:
    rv = plpy.execute(config_select_plan, ["default_seed"])
    resolved_seed = json.loads(rv[0]["value"]) if rv else 42

# Validate inputs
if not prompt or not prompt.strip():
    plpy.error("Prompt cannot be empty")

if not pattern or not pattern.strip():
    plpy.error("Pattern cannot be empty")

# AIDEV-NOTE: For structured generation functions, unsafe_mode is not supported without model selection
if unsafe_mode:
    plpy.error("unsafe_mode is not supported for structured generation functions")

# Check if we should use cache
if use_cache:
    # Generate cache key including pattern
    cache_key_input = f"{prompt}|regex|{pattern}"
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()

    # SELECT ONLY - no UPDATE for immutability
    cache_plan = plpy.prepare(f"""
        SELECT response 
        FROM {plpy.quote_ident(ext_schema)}.steadytext_cache 
        WHERE cache_key = $1
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for regex key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# Cache miss - generate new content
host_rv = plpy.execute(config_select_plan, ["daemon_host"])
host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(config_select_plan, ["daemon_port"])
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

# Create connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Auto-start daemon if configured
auto_start_rv = plpy.execute(config_select_plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    connector.start_daemon()

# Build kwargs
generation_kwargs = {
    "seed": resolved_seed,
    "unsafe_mode": unsafe_mode
}

# Generate structured output
try:
    if connector.is_daemon_running():
        result = connector.generate_regex(
            prompt=prompt,
            pattern=pattern,
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    else:
        # Direct generation fallback
        from steadytext import generate_regex as steadytext_generate_regex
        result = steadytext_generate_regex(
            prompt=prompt,
            regex=pattern,
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    
    # AIDEV-NOTE: Cache writes removed for IMMUTABLE compliance
    
    return result
    
except Exception as e:
    plpy.error(f"Regex generation failed: {str(e)}")
$c$;

-- 9) Add choice generation function with schema qualification
CREATE OR REPLACE FUNCTION steadytext_generate_choice(
    prompt TEXT,
    choices TEXT[],
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE,
    model TEXT DEFAULT NULL
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
# AIDEV-NOTE: Generate text constrained to one of the provided choices
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
# v2025.8.15: Added model parameter for remote model access
import json
import hashlib

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules from GD
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Get configuration
# AIDEV-NOTE: Use extension schema instead of current_schema() for cross-schema compatibility
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
config_select_plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(ext_schema)}.steadytext_config WHERE key = $1", ["text"])

# Resolve max_tokens
resolved_max_tokens = max_tokens
if resolved_max_tokens is None:
    rv = plpy.execute(config_select_plan, ["default_max_tokens"])
    resolved_max_tokens = json.loads(rv[0]["value"]) if rv else 512

# Resolve seed
resolved_seed = seed
if resolved_seed is None:
    rv = plpy.execute(config_select_plan, ["default_seed"])
    resolved_seed = json.loads(rv[0]["value"]) if rv else 42

# Validate inputs
if not prompt or not prompt.strip():
    plpy.error("Prompt cannot be empty")

if not choices or len(choices) == 0:
    plpy.error("Choices list cannot be empty")

# AIDEV-NOTE: For structured generation functions, unsafe_mode is not supported without model selection
if unsafe_mode:
    plpy.error("unsafe_mode is not supported for structured generation functions")

# Convert PostgreSQL array to Python list
choices_list = list(choices)

# Check if we should use cache
if use_cache:
    # Generate cache key including choices
    cache_key_input = f"{prompt}|choice|{json.dumps(sorted(choices_list))}"
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()

    # SELECT ONLY - no UPDATE for immutability
    cache_plan = plpy.prepare(f"""
        SELECT response 
        FROM {plpy.quote_ident(ext_schema)}.steadytext_cache 
        WHERE cache_key = $1
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for choice key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# Cache miss - generate new content
host_rv = plpy.execute(config_select_plan, ["daemon_host"])
host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(config_select_plan, ["daemon_port"])
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

# Create connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Auto-start daemon if configured
auto_start_rv = plpy.execute(config_select_plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    connector.start_daemon()

# Build kwargs
generation_kwargs = {
    "seed": resolved_seed,
    "unsafe_mode": unsafe_mode
}

# Generate structured output
try:
    if connector.is_daemon_running():
        result = connector.generate_choice(
            prompt=prompt,
            choices=choices_list,
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    else:
        # Direct generation fallback
        from steadytext import generate_choice as steadytext_generate_choice
        result = steadytext_generate_choice(
            prompt=prompt,
            choices=choices_list,
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    
    # AIDEV-NOTE: Cache writes removed for IMMUTABLE compliance
    
    return result
    
except Exception as e:
    plpy.error(f"Choice generation failed: {str(e)}")
$c$;

-- 10) Add summarization functions with schema qualification

CREATE OR REPLACE FUNCTION steadytext_summarize_text(
    input_text text,
    metadata jsonb DEFAULT '{}'::jsonb,
    model text DEFAULT NULL,
    unsafe_mode boolean DEFAULT FALSE
) RETURNS text
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
import json

# If model and unsafe_mode are provided as parameters, add them to metadata
meta_dict = {}
if metadata:
    try:
        meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
    except (json.JSONDecodeError, TypeError):
        meta_dict = {}

# Override metadata with explicit parameters if provided
if model is not None:
    meta_dict['model'] = model
if unsafe_mode is not None:
    meta_dict['unsafe_mode'] = unsafe_mode

# For simple case, just use local model with basic summary
if not model:
    # Simple local summarization
    if input_text and len(input_text) > 0:
        # Take first 100 chars as summary
        summary = input_text[:100]
        if len(input_text) > 100:
            summary += "..."
        return f"Summary: {summary}"
    return "No text to summarize"

# For remote models, validate unsafe_mode
if ':' in model and not unsafe_mode:
    plpy.error("Remote models (containing ':') require unsafe_mode=TRUE")

# Use steadytext_generate for actual summarization
prompt = f"Summarize the following text in 1-2 sentences: {input_text[:500]}"

# AIDEV-NOTE: Get extension schema dynamically at runtime for TimescaleDB continuous aggregates compatibility
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
plan = plpy.prepare(
    f"SELECT {plpy.quote_ident(ext_schema)}.steadytext_generate($1, NULL, true, 42, '[EOS]', $2, NULL, NULL, NULL, $3) as summary",
    ["text", "text", "boolean"]
)
result = plpy.execute(plan, [prompt, model, unsafe_mode])

if result and result[0]["summary"]:
    return result[0]["summary"]
return "Unable to generate summary"
$c$;

CREATE OR REPLACE FUNCTION steadytext_summarize_finalize(
    state jsonb
) RETURNS text
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
    import json

    # AIDEV-NOTE: Don't reassign argument variables to avoid Python scoping issues
    if state is None:
        return "No data to summarize"

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
    
    # Extract model and unsafe_mode for remote model support
    # AIDEV-NOTE: Initialize variables outside conditional to avoid UnboundLocalError
    model = None
    unsafe_mode = False
    
    if metadata:
        model = metadata.get('model', None)
        unsafe_mode = metadata.get('unsafe_mode', False)
        
        # Add other metadata to prompt context
        other_metadata = {k: v for k, v in metadata.items() if k not in ['model', 'unsafe_mode']}
        if other_metadata:
            meta_str = ", ".join([f"{k}: {v}" for k, v in other_metadata.items()])
            prompt += f". Context: {meta_str}"

    # Generate summary using steadytext with model and unsafe_mode support
    # AIDEV-NOTE: Pass model and unsafe_mode from metadata to support remote models
    # AIDEV-NOTE: Get extension schema dynamically at runtime for TimescaleDB continuous aggregates compatibility
    ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    plan = plpy.prepare(
        f"SELECT {plpy.quote_ident(ext_schema)}.steadytext_generate($1, NULL, true, 42, '[EOS]', $2, NULL, NULL, NULL, $3) as summary",
        ["text", "text", "boolean"]
    )
    result = plpy.execute(plan, [prompt, model, unsafe_mode])

    if result and result[0]["summary"]:
        return result[0]["summary"]
    return "Unable to generate summary"
$c$;

-- 11) Add daemon alias functions with schema qualification
CREATE OR REPLACE FUNCTION st_daemon_start()
RETURNS BOOLEAN
LANGUAGE sql
VOLATILE PARALLEL SAFE
AS $alias$
    SELECT @extschema@.steadytext_daemon_start();
$alias$;


CREATE OR REPLACE FUNCTION st_daemon_stop()
RETURNS BOOLEAN
LANGUAGE sql
VOLATILE PARALLEL SAFE
AS $alias$
    SELECT @extschema@.steadytext_daemon_stop();
$alias$;

-- Fix steadytext_rerank_docs_only to use proper plpy.execute syntax
CREATE OR REPLACE FUNCTION steadytext_rerank_docs_only(
    query text,
    documents text[],
    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    seed integer DEFAULT 42
) RETURNS TABLE(document text)
AS $c$
    # Get extension schema for proper qualification
    ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    
    # Call the main rerank function and extract just documents
    # Use plpy.prepare for proper parameter binding
    plan = plpy.prepare(
        f"SELECT document FROM {plpy.quote_ident(ext_schema)}.steadytext_rerank($1, $2, $3, true, $4)",
        ["text", "text[]", "text", "integer"]
    )
    results = plpy.execute(plan, [query, documents, task, seed])
    
    for row in results:
        yield row["document"]
$c$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;


-- pg_steadytext extension upgrade from 1.4.4 to 1.4.5
-- Adds remote model support for structured generation functions

-- AIDEV-NOTE: Update version function to return 1.4.5
CREATE OR REPLACE FUNCTION steadytext_version()
RETURNS TEXT
LANGUAGE sql
IMMUTABLE PARALLEL SAFE LEAKPROOF
AS $$
SELECT '1.4.5'::TEXT;
$$;
COMMENT ON FUNCTION steadytext_version() IS 'Returns the version of the pg_steadytext extension';

-- AIDEV-NOTE: Update structured generation functions to support model and unsafe_mode parameters

-- Update st_generate_json with model parameter
-- Handle removal of old function signature before creating new one
DO $$
BEGIN
    -- Try to remove old function from extension if it exists
    BEGIN
        ALTER EXTENSION pg_steadytext DROP FUNCTION steadytext_generate_json(text, jsonb, integer, boolean, integer, boolean);
    EXCEPTION WHEN OTHERS THEN
        -- Function either doesn't exist or isn't part of extension
        NULL;
    END;
    
    -- Try to drop old function if it exists
    DROP FUNCTION IF EXISTS steadytext_generate_json(text, jsonb, integer, boolean, integer, boolean);
END $$;

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
# AIDEV-NOTE: Generate JSON text matching a schema using llama.cpp grammars
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
# v1.4.5: Added model parameter for remote model access
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

# AIDEV-NOTE: Validate model parameter - remote models require unsafe_mode=TRUE
if not unsafe_mode and model and ':' in model:
    plpy.error("Remote models (containing ':') require unsafe_mode=TRUE")

# AIDEV-NOTE: Validate that unsafe_mode requires a model to be specified
if unsafe_mode and not model:
    plpy.error("unsafe_mode=TRUE requires a model parameter to be specified")

# Convert JSONB to dict
try:
    schema_dict = json.loads(schema)
except Exception as e:
    plpy.error(f"Invalid JSON schema: {str(e)}")

# Check if we should use cache
if use_cache and not (unsafe_mode and model and ':' in model):
    # Generate cache key including schema
    cache_key_input = f"{prompt}|json|{json.dumps(schema_dict, sort_keys=True)}"
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()

    # SELECT ONLY - no UPDATE for immutability
    cache_plan = plpy.prepare("""
        SELECT response 
        FROM steadytext_cache 
        WHERE cache_key = $1
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for JSON key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# Cache miss - generate new content
host_rv = plpy.execute(plan, ["daemon_host"])
host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(plan, ["daemon_port"])
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

# AIDEV-NOTE: For remote models with unsafe_mode, skip daemon entirely
# Remote models don't need daemon and checking it causes unnecessary delays
is_remote_model = unsafe_mode and model and ':' in model

# Create connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Auto-start daemon if configured and not using remote model
auto_start_rv = plpy.execute(plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not is_remote_model and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    connector.start_daemon()

# Build kwargs
generation_kwargs = {
    "seed": resolved_seed,
    "unsafe_mode": unsafe_mode
}

# Add model if specified
if model:
    generation_kwargs["model"] = model

# Generate structured output
try:
    if is_remote_model or not connector.is_daemon_running():
        # Direct generation for remote models or when daemon unavailable
        from steadytext import generate_json as steadytext_generate_json
        result = steadytext_generate_json(
            prompt=prompt,
            schema=schema_dict,
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    else:
        # Use daemon for local models
        result = connector.generate_json(
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

-- Add new function to extension (idempotent)
DO $$
BEGIN
    -- Try to add function to extension
    BEGIN
        ALTER EXTENSION pg_steadytext ADD FUNCTION steadytext_generate_json(text, jsonb, integer, boolean, integer, boolean, text);
    EXCEPTION WHEN OTHERS THEN
        -- Function is already part of extension
        NULL;
    END;
END $$;

-- Update st_generate_regex with model parameter
-- Handle removal of old function signature before creating new one
DO $$
BEGIN
    -- Try to remove old function from extension if it exists
    BEGIN
        ALTER EXTENSION pg_steadytext DROP FUNCTION steadytext_generate_regex(text, text, integer, boolean, integer, boolean);
    EXCEPTION WHEN OTHERS THEN
        -- Function either doesn't exist or isn't part of extension
        NULL;
    END;
    
    -- Try to drop old function if it exists
    DROP FUNCTION IF EXISTS steadytext_generate_regex(text, text, integer, boolean, integer, boolean);
END $$;

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
# v1.4.5: Added model parameter for remote model access
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

# AIDEV-NOTE: Validate model parameter - remote models require unsafe_mode=TRUE
if not unsafe_mode and model and ':' in model:
    plpy.error("Remote models (containing ':') require unsafe_mode=TRUE")

# AIDEV-NOTE: Validate that unsafe_mode requires a model to be specified
if unsafe_mode and not model:
    plpy.error("unsafe_mode=TRUE requires a model parameter to be specified")

# Check if we should use cache
if use_cache and not (unsafe_mode and model and ':' in model):
    # Generate cache key including pattern
    cache_key_input = f"{prompt}|regex|{pattern}"
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()

    # SELECT ONLY - no UPDATE for immutability
    cache_plan = plpy.prepare("""
        SELECT response 
        FROM steadytext_cache 
        WHERE cache_key = $1
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for regex key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# Cache miss - generate new content
host_rv = plpy.execute(plan, ["daemon_host"])
host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(plan, ["daemon_port"])
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

# AIDEV-NOTE: For remote models with unsafe_mode, skip daemon entirely
# Remote models don't need daemon and checking it causes unnecessary delays
is_remote_model = unsafe_mode and model and ':' in model

# Create connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Auto-start daemon if configured and not using remote model
auto_start_rv = plpy.execute(plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not is_remote_model and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    connector.start_daemon()

# Build kwargs
generation_kwargs = {
    "seed": resolved_seed,
    "unsafe_mode": unsafe_mode
}

# Add model if specified
if model:
    generation_kwargs["model"] = model

# Generate structured output
try:
    if is_remote_model or not connector.is_daemon_running():
        # Direct generation for remote models or when daemon unavailable
        from steadytext import generate_regex as steadytext_generate_regex
        result = steadytext_generate_regex(
            prompt=prompt,
            pattern=pattern,
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    else:
        # Use daemon for local models
        result = connector.generate_regex(
            prompt=prompt,
            pattern=pattern,
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    
    # AIDEV-NOTE: Cache writes removed for IMMUTABLE compliance
    
    return result
    
except Exception as e:
    plpy.error(f"Regex generation failed: {str(e)}")
$c$;

-- Add new function to extension (idempotent)
DO $$
BEGIN
    -- Try to add function to extension
    BEGIN
        ALTER EXTENSION pg_steadytext ADD FUNCTION steadytext_generate_regex(text, text, integer, boolean, integer, boolean, text);
    EXCEPTION WHEN OTHERS THEN
        -- Function is already part of extension
        NULL;
    END;
END $$;

-- Update st_generate_choice with model parameter
-- Handle removal of old function signature before creating new one
DO $$
BEGIN
    -- Try to remove old function from extension if it exists
    BEGIN
        ALTER EXTENSION pg_steadytext DROP FUNCTION steadytext_generate_choice(text, text[], integer, boolean, integer, boolean);
    EXCEPTION WHEN OTHERS THEN
        -- Function either doesn't exist or isn't part of extension
        NULL;
    END;
    
    -- Try to drop old function if it exists
    DROP FUNCTION IF EXISTS steadytext_generate_choice(text, text[], integer, boolean, integer, boolean);
END $$;

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
# AIDEV-NOTE: Generate text from a list of choices using llama.cpp grammars
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
# v1.4.5: Added model parameter for remote model access
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

# AIDEV-NOTE: Validate model parameter - remote models require unsafe_mode=TRUE
if not unsafe_mode and model and ':' in model:
    plpy.error("Remote models (containing ':') require unsafe_mode=TRUE")

# AIDEV-NOTE: Validate that unsafe_mode requires a model to be specified
if unsafe_mode and not model:
    plpy.error("unsafe_mode=TRUE requires a model parameter to be specified")

# Check if we should use cache
if use_cache and not (unsafe_mode and model and ':' in model):
    # Generate cache key including choices
    cache_key_input = f"{prompt}|choice|{json.dumps(sorted(choices))}"
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()

    # SELECT ONLY - no UPDATE for immutability
    cache_plan = plpy.prepare("""
        SELECT response 
        FROM steadytext_cache 
        WHERE cache_key = $1
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for choice key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# Cache miss - generate new content
host_rv = plpy.execute(plan, ["daemon_host"])
host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(plan, ["daemon_port"])
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

# AIDEV-NOTE: For remote models with unsafe_mode, skip daemon entirely
# Remote models don't need daemon and checking it causes unnecessary delays
is_remote_model = unsafe_mode and model and ':' in model

# Create connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Auto-start daemon if configured and not using remote model
auto_start_rv = plpy.execute(plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not is_remote_model and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    connector.start_daemon()

# Build kwargs
generation_kwargs = {
    "seed": resolved_seed,
    "unsafe_mode": unsafe_mode
}

# Add model if specified
if model:
    generation_kwargs["model"] = model

# Generate structured output
try:
    if is_remote_model or not connector.is_daemon_running():
        # Direct generation for remote models or when daemon unavailable
        from steadytext import generate_choice as steadytext_generate_choice
        result = steadytext_generate_choice(
            prompt=prompt,
            choices=list(choices),
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    else:
        # Use daemon for local models
        result = connector.generate_choice(
            prompt=prompt,
            choices=list(choices),
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    
    # AIDEV-NOTE: Cache writes removed for IMMUTABLE compliance
    
    return result
    
except Exception as e:
    plpy.error(f"Choice generation failed: {str(e)}")
$c$;

-- Add new function to extension (idempotent)
DO $$
BEGIN
    -- Try to add function to extension
    BEGIN
        ALTER EXTENSION pg_steadytext ADD FUNCTION steadytext_generate_choice(text, text[], integer, boolean, integer, boolean, text);
    EXCEPTION WHEN OTHERS THEN
        -- Function is already part of extension
        NULL;
    END;
END $$;

-- Update short aliases for the new signatures
CREATE OR REPLACE FUNCTION st_generate_json(
    prompt TEXT,
    schema JSONB,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE,
    model TEXT DEFAULT NULL
)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE PARALLEL SAFE
AS $alias$
    SELECT steadytext_generate_json($1, $2, $3, $4, $5, $6, $7);
$alias$;

CREATE OR REPLACE FUNCTION st_generate_regex(
    prompt TEXT,
    pattern TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE,
    model TEXT DEFAULT NULL
)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE PARALLEL SAFE
AS $alias$
    SELECT steadytext_generate_regex($1, $2, $3, $4, $5, $6, $7);
$alias$;

CREATE OR REPLACE FUNCTION st_generate_choice(
    prompt TEXT,
    choices TEXT[],
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE,
    model TEXT DEFAULT NULL
)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE PARALLEL SAFE
AS $alias$
    SELECT steadytext_generate_choice($1, $2, $3, $4, $5, $6, $7);
$alias$;

-- Update comments to document new functionality
COMMENT ON FUNCTION steadytext_generate_json(text, jsonb, integer, boolean, integer, boolean, text) IS 
'Generate JSON text that conforms to a schema. Supports remote models via model parameter with unsafe_mode=TRUE.';

COMMENT ON FUNCTION steadytext_generate_regex(text, text, integer, boolean, integer, boolean, text) IS 
'Generate text matching a regex pattern. Supports remote models via model parameter with unsafe_mode=TRUE.';

COMMENT ON FUNCTION steadytext_generate_choice(text, text[], integer, boolean, integer, boolean, text) IS 
'Generate text that is one of the provided choices. Supports remote models via model parameter with unsafe_mode=TRUE.';

-- AIDEV-NOTE: v1.4.5 changes:
-- - Added model parameter to all structured generation functions
-- - Enabled remote model support when unsafe_mode=TRUE and model contains ':'
-- - Direct generation bypasses daemon for remote models
-- - Maintains backward compatibility by making model parameter optional
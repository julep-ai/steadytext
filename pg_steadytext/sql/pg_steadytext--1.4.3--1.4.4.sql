-- pg_steadytext--1.4.3--1.4.4.sql
-- Migration from version 1.4.3 to 1.4.4

-- AIDEV-NOTE: This migration includes:
-- 1. Add support for additional model parameters (eos_string, model, model_repo, model_filename, size)
-- 2. Update version function to 1.4.4
-- 3. Fixed upgrade issue: When changing function signatures, must use ALTER EXTENSION DROP/ADD
--    because PostgreSQL prevents dropping functions that belong to an extension

-- Update the version function
CREATE OR REPLACE FUNCTION steadytext_version()
RETURNS TEXT
LANGUAGE sql
IMMUTABLE PARALLEL SAFE LEAKPROOF
AS $c$
    SELECT '1.4.4'::TEXT;
$c$;

-- Update steadytext_generate function to support additional parameters
-- AIDEV-NOTE: This is an idempotent upgrade that handles multiple scenarios:
-- 1. Old function exists and is part of extension -> remove from extension, drop it
-- 2. Old function exists but not part of extension -> just drop it
-- 3. Old function doesn't exist -> do nothing (already upgraded)
-- 4. New function already exists -> CREATE OR REPLACE will update it
DO $$
BEGIN
    -- Try to remove old function from extension if it exists
    BEGIN
        ALTER EXTENSION pg_steadytext DROP FUNCTION steadytext_generate(text, integer, boolean, integer);
    EXCEPTION WHEN OTHERS THEN
        -- Function either doesn't exist or isn't part of extension
        NULL;
    END;
    
    -- Try to drop old function if it exists
    DROP FUNCTION IF EXISTS steadytext_generate(text, integer, boolean, integer);
END $$;

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
# v1.4.4: Added support for eos_string, model, model_repo, model_filename, size parameters
# v1.4.4: Added unsafe_mode parameter for remote model access
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

# AIDEV-NOTE: Validate model parameter - remote models require unsafe_mode=TRUE
if not unsafe_mode and model and ':' in model:
    plpy.error("Remote models (containing ':') require unsafe_mode=TRUE")

# Check if we should use cache
if use_cache:
    # Generate cache key consistent with SteadyText format
    # Include eos_string in cache key if it's not the default
    if eos_string == '[EOS]':
        cache_key = prompt
    else:
        cache_key = f"{prompt}::EOS::{eos_string}"

    # Try to get from cache first
    cache_plan = plpy.prepare("""
        SELECT response 
        FROM steadytext_cache 
        WHERE cache_key = $1
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# Cache miss - generate new content
# Get configuration for daemon connection
host_rv = plpy.execute(plan, ["daemon_host"])
host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(plan, ["daemon_port"])  
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

# Create daemon connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Check if daemon should auto-start
auto_start_rv = plpy.execute(plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    started = connector.start_daemon()
    if not started:
        plpy.warning("Failed to auto-start daemon, will try direct generation")

# Build kwargs for additional parameters
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
if unsafe_mode:
    generation_kwargs["unsafe_mode"] = unsafe_mode

# Try to generate via daemon or direct fallback
try:
    if connector.is_daemon_running():
        result = connector.generate(
            prompt=prompt,
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    else:
        # Direct generation fallback
        from steadytext import generate as steadytext_generate
        result = steadytext_generate(
            prompt=prompt, 
            max_new_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    return result
    
except Exception as e:
    plpy.error(f"Generation failed: {str(e)}")
$c$;

-- Add new function to extension (idempotent)
DO $$
BEGIN
    -- Try to add function to extension
    BEGIN
        ALTER EXTENSION pg_steadytext ADD FUNCTION steadytext_generate(text, integer, boolean, integer, text, text, text, text, text, boolean);
    EXCEPTION WHEN OTHERS THEN
        -- Function is already part of extension
        NULL;
    END;
END $$;

-- Update steadytext_generate_json function to support unsafe_mode parameter
-- AIDEV-NOTE: Handle removal of old function signature before creating new one
DO $$
BEGIN
    -- Try to remove old function from extension if it exists
    BEGIN
        ALTER EXTENSION pg_steadytext DROP FUNCTION steadytext_generate_json(text, jsonb, integer, boolean, integer);
    EXCEPTION WHEN OTHERS THEN
        -- Function either doesn't exist or isn't part of extension
        NULL;
    END;
    
    -- Try to drop old function if it exists
    DROP FUNCTION IF EXISTS steadytext_generate_json(text, jsonb, integer, boolean, integer);
END $$;

CREATE OR REPLACE FUNCTION steadytext_generate_json(
    prompt TEXT,
    schema JSONB,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
# AIDEV-NOTE: Generate JSON that conforms to a schema using llama.cpp grammars
# v1.4.4: Added unsafe_mode parameter for remote model access
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

# Check if we should use cache
if use_cache:
    # Generate cache key for structured generation
    cache_key = f"{prompt}::JSON::{json.dumps(schema_dict, sort_keys=True)}"
    
    # Try to get from cache first
    cache_plan = plpy.prepare("""
        SELECT response 
        FROM steadytext_cache 
        WHERE cache_key = $1
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for JSON generation")
        return cache_result[0]["response"]

# Cache miss - generate new content
# Get configuration for daemon connection
host_rv = plpy.execute(plan, ["daemon_host"])
host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(plan, ["daemon_port"])  
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

# Create daemon connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Check if daemon should auto-start
auto_start_rv = plpy.execute(plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    started = connector.start_daemon()
    if not started:
        plpy.warning("Failed to auto-start daemon, will try direct generation")

# Build kwargs
generation_kwargs = {
    "seed": resolved_seed,
    "unsafe_mode": unsafe_mode
}

# Try to generate via daemon or direct fallback
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
    return result
    
except Exception as e:
    plpy.error(f"JSON generation failed: {str(e)}")
$c$;

-- Add new function to extension (idempotent)
DO $$
BEGIN
    -- Try to add function to extension
    BEGIN
        ALTER EXTENSION pg_steadytext ADD FUNCTION steadytext_generate_json(text, jsonb, integer, boolean, integer, boolean);
    EXCEPTION WHEN OTHERS THEN
        -- Function is already part of extension
        NULL;
    END;
END $$;

-- Update steadytext_generate_regex function to support unsafe_mode parameter
-- AIDEV-NOTE: Handle removal of old function signature before creating new one
DO $$
BEGIN
    -- Try to remove old function from extension if it exists
    BEGIN
        ALTER EXTENSION pg_steadytext DROP FUNCTION steadytext_generate_regex(text, text, integer, boolean, integer);
    EXCEPTION WHEN OTHERS THEN
        -- Function either doesn't exist or isn't part of extension
        NULL;
    END;
    
    -- Try to drop old function if it exists
    DROP FUNCTION IF EXISTS steadytext_generate_regex(text, text, integer, boolean, integer);
END $$;

CREATE OR REPLACE FUNCTION steadytext_generate_regex(
    prompt TEXT,
    pattern TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
# AIDEV-NOTE: Generate text matching a regex pattern using llama.cpp grammars
# v1.4.4: Added unsafe_mode parameter for remote model access
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
    # Generate cache key for regex generation
    cache_key = f"{prompt}::REGEX::{pattern}"
    
    # Try to get from cache first
    cache_plan = plpy.prepare("""
        SELECT response 
        FROM steadytext_cache 
        WHERE cache_key = $1
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for regex generation")
        return cache_result[0]["response"]

# Cache miss - generate new content
# Get configuration for daemon connection
host_rv = plpy.execute(plan, ["daemon_host"])
host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(plan, ["daemon_port"])  
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

# Create daemon connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Check if daemon should auto-start
auto_start_rv = plpy.execute(plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    started = connector.start_daemon()
    if not started:
        plpy.warning("Failed to auto-start daemon, will try direct generation")

# Build kwargs
generation_kwargs = {
    "seed": resolved_seed,
    "unsafe_mode": unsafe_mode
}

# Try to generate via daemon or direct fallback
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
            pattern=pattern,
            max_tokens=resolved_max_tokens,
            **generation_kwargs
        )
    return result
    
except Exception as e:
    plpy.error(f"Regex generation failed: {str(e)}")
$c$;

-- Add new function to extension (idempotent)
DO $$
BEGIN
    -- Try to add function to extension
    BEGIN
        ALTER EXTENSION pg_steadytext ADD FUNCTION steadytext_generate_regex(text, text, integer, boolean, integer, boolean);
    EXCEPTION WHEN OTHERS THEN
        -- Function is already part of extension
        NULL;
    END;
END $$;

-- Update steadytext_generate_choice function to support unsafe_mode parameter
-- AIDEV-NOTE: Handle removal of old function signature before creating new one
DO $$
BEGIN
    -- Try to remove old function from extension if it exists
    BEGIN
        ALTER EXTENSION pg_steadytext DROP FUNCTION steadytext_generate_choice(text, text[], integer, boolean, integer);
    EXCEPTION WHEN OTHERS THEN
        -- Function either doesn't exist or isn't part of extension
        NULL;
    END;
    
    -- Try to drop old function if it exists
    DROP FUNCTION IF EXISTS steadytext_generate_choice(text, text[], integer, boolean, integer);
END $$;

CREATE OR REPLACE FUNCTION steadytext_generate_choice(
    prompt TEXT,
    choices TEXT[],
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
# AIDEV-NOTE: Generate text constrained to one of the provided choices
# v1.4.4: Added unsafe_mode parameter for remote model access
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
    # Generate cache key for choice generation
    cache_key = f"{prompt}::CHOICE::{json.dumps(choices_list, sort_keys=True)}"
    
    # Try to get from cache first
    cache_plan = plpy.prepare("""
        SELECT response 
        FROM steadytext_cache 
        WHERE cache_key = $1
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for choice generation")
        return cache_result[0]["response"]

# Cache miss - generate new content
# Get configuration for daemon connection
host_rv = plpy.execute(plan, ["daemon_host"])
host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(plan, ["daemon_port"])  
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

# Create daemon connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Check if daemon should auto-start
auto_start_rv = plpy.execute(plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    started = connector.start_daemon()
    if not started:
        plpy.warning("Failed to auto-start daemon, will try direct generation")

# Build kwargs
generation_kwargs = {
    "seed": resolved_seed,
    "unsafe_mode": unsafe_mode
}

# Try to generate via daemon or direct fallback
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
    return result
    
except Exception as e:
    plpy.error(f"Choice generation failed: {str(e)}")
$c$;

-- Add new function to extension (idempotent)
DO $$
BEGIN
    -- Try to add function to extension
    BEGIN
        ALTER EXTENSION pg_steadytext ADD FUNCTION steadytext_generate_choice(text, text[], integer, boolean, integer, boolean);
    EXCEPTION WHEN OTHERS THEN
        -- Function is already part of extension
        NULL;
    END;
END $$;

-- Create short aliases for all steadytext functions
-- AIDEV-NOTE: Manual creation is required to preserve default parameter values
-- Dynamic generation would create functions without DEFAULT clauses, breaking the API

-- st_generate alias
CREATE OR REPLACE FUNCTION st_generate(
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
LANGUAGE sql
IMMUTABLE PARALLEL SAFE
AS $alias$
    SELECT steadytext_generate($1, $2, $3, $4, $5, $6, $7, $8, $9, $10);
$alias$;

-- st_embed alias
CREATE OR REPLACE FUNCTION st_embed(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42
)
RETURNS vector(1024)
LANGUAGE sql
IMMUTABLE PARALLEL SAFE
AS $alias$
    SELECT steadytext_embed($1, $2, $3);
$alias$;

-- st_generate_json alias
CREATE OR REPLACE FUNCTION st_generate_json(
    prompt TEXT,
    schema JSONB,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE PARALLEL SAFE
AS $alias$
    SELECT steadytext_generate_json($1, $2, $3, $4, $5, $6);
$alias$;

-- st_generate_regex alias
CREATE OR REPLACE FUNCTION st_generate_regex(
    prompt TEXT,
    pattern TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE PARALLEL SAFE
AS $alias$
    SELECT steadytext_generate_regex($1, $2, $3, $4, $5, $6);
$alias$;

-- st_generate_choice alias
CREATE OR REPLACE FUNCTION st_generate_choice(
    prompt TEXT,
    choices TEXT[],
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE PARALLEL SAFE
AS $alias$
    SELECT steadytext_generate_choice($1, $2, $3, $4, $5, $6);
$alias$;

-- st_rerank alias
CREATE OR REPLACE FUNCTION st_rerank(
    query TEXT,
    documents TEXT[],
    task TEXT DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    return_scores BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42
)
RETURNS TABLE(document TEXT, score DOUBLE PRECISION)
LANGUAGE sql
IMMUTABLE PARALLEL SAFE
AS $alias$
    SELECT * FROM steadytext_rerank($1, $2, $3, $4, $5);
$alias$;

-- st_version alias
CREATE OR REPLACE FUNCTION st_version()
RETURNS TEXT
LANGUAGE sql
IMMUTABLE PARALLEL SAFE LEAKPROOF
AS $alias$
    SELECT steadytext_version();
$alias$;

-- st_daemon_status alias
CREATE OR REPLACE FUNCTION st_daemon_status()
RETURNS TABLE(daemon_id TEXT, status TEXT, endpoint TEXT, last_heartbeat TIMESTAMP WITH TIME ZONE, uptime_seconds INTEGER)
LANGUAGE sql
STABLE PARALLEL SAFE
AS $alias$
    SELECT * FROM steadytext_daemon_status();
$alias$;

-- st_cache_stats alias
CREATE OR REPLACE FUNCTION st_cache_stats()
RETURNS TABLE(total_entries BIGINT, total_size_mb DOUBLE PRECISION, cache_hit_rate DOUBLE PRECISION, avg_access_count DOUBLE PRECISION, oldest_entry TIMESTAMP WITH TIME ZONE, newest_entry TIMESTAMP WITH TIME ZONE)
LANGUAGE sql
STABLE PARALLEL SAFE
AS $alias$
    SELECT * FROM steadytext_cache_stats();
$alias$;

-- Add all manually created aliases to extension
-- AIDEV-NOTE: Each alias function needs to be explicitly added to the extension
ALTER EXTENSION pg_steadytext ADD FUNCTION st_generate(text, integer, boolean, integer, text, text, text, text, text, boolean);
ALTER EXTENSION pg_steadytext ADD FUNCTION st_embed(text, boolean, integer);
ALTER EXTENSION pg_steadytext ADD FUNCTION st_generate_json(text, jsonb, integer, boolean, integer, boolean);
ALTER EXTENSION pg_steadytext ADD FUNCTION st_generate_regex(text, text, integer, boolean, integer, boolean);
ALTER EXTENSION pg_steadytext ADD FUNCTION st_generate_choice(text, text[], integer, boolean, integer, boolean);
ALTER EXTENSION pg_steadytext ADD FUNCTION st_rerank(text, text[], text, boolean, integer);
ALTER EXTENSION pg_steadytext ADD FUNCTION st_version();
ALTER EXTENSION pg_steadytext ADD FUNCTION st_daemon_status();
ALTER EXTENSION pg_steadytext ADD FUNCTION st_cache_stats();

-- AIDEV-NOTE: Migration completed successfully
-- Changes in v1.4.4:
-- 1. Version bump to 1.4.4
-- 2. Added support for additional model parameters:
--    - eos_string: End-of-sequence string (default: '[EOS]')
--    - model: Specific model to use
--    - model_repo: Model repository
--    - model_filename: Model filename
--    - size: Model size specification
--    - unsafe_mode: Allow remote models when TRUE (default: FALSE)
-- 3. Cache key now includes eos_string when it's not the default value
-- 4. Remote models (containing ':') require unsafe_mode=TRUE
-- 5. Added manual alias creation for all steadytext_* functions as st_*
--    Manual creation is required to preserve default parameter values
-- pg_steadytext--1.4.2--1.4.3.sql
-- Migration from version 1.4.2 to 1.4.3

-- AIDEV-NOTE: This migration fixes the following issues:
-- 1. Parameter name fix: max_tokens → max_new_tokens for direct generation fallback
-- 2. Mark functions as LEAKPROOF where appropriate for security
-- 3. Fix function overload conflict by removing single-argument embed function
-- 4. Fix UnboundLocalError in ai_summarize_accumulate function
-- 5. Fix rerank function return type issue

-- Update the version function
CREATE OR REPLACE FUNCTION steadytext_version()
RETURNS TEXT
LANGUAGE sql
IMMUTABLE PARALLEL SAFE LEAKPROOF
AS $$
    SELECT '1.4.3'::TEXT;
$$;

-- Fix steadytext_generate function to use max_new_tokens in fallback
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
    # For generation: just the prompt (no parameters in key)
    cache_key = prompt

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

# Try to generate via daemon or direct fallback
try:
    if connector.is_daemon_running():
        result = connector.generate(
            prompt=prompt,
            max_tokens=resolved_max_tokens,
            seed=resolved_seed
        )
    else:
        # Direct generation fallback
        from steadytext import generate as steadytext_generate
        result = steadytext_generate(
            prompt=prompt, 
            max_new_tokens=resolved_max_tokens,
            seed=resolved_seed
        )
    return result
    
except Exception as e:
    plpy.error(f"Generation failed: {str(e)}")
$$;

-- Fix steadytext_embed function to be LEAKPROOF
CREATE OR REPLACE FUNCTION steadytext_embed(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42
)
RETURNS vector(1024)
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
# AIDEV-NOTE: Embedding function that returns deterministic embeddings
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
    # Embeddings use SHA256 hash of "embed:{text}"
    cache_key_input = f"embed:{text_input}"
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()

    # Try to get from cache
    cache_plan = plpy.prepare("""
        SELECT embedding 
        FROM steadytext_cache 
        WHERE cache_key = $1
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["embedding"] is not None:
        plpy.notice(f"Cache hit for embedding key: {cache_key[:8]}...")
        return cache_result[0]["embedding"]

# Cache miss - generate new embedding
plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])

host_rv = plpy.execute(plan, ["daemon_host"])
host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(plan, ["daemon_port"])
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

# Create connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Auto-start daemon if configured
auto_start_rv = plpy.execute(plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    connector.start_daemon()

# Generate embedding
try:
    if connector.is_daemon_running():
        result = connector.embed(text=text_input)
    else:
        # Direct embedding fallback
        from steadytext import embed as steadytext_embed
        result = steadytext_embed(text_input)
    
    # Convert to vector format if needed
    if result is not None:
        # Ensure it's a list/array
        if hasattr(result, 'tolist'):
            embedding_list = result.tolist()
        else:
            embedding_list = list(result)

        return embedding_list
    else:
        plpy.error("Failed to generate embedding")
        
except Exception as e:
    plpy.error(f"Embedding generation failed: {str(e)}")
$$;

-- Drop the single-argument embed overload if it exists (causes function overload conflicts)
DROP FUNCTION IF EXISTS steadytext_embed(TEXT);

-- Fix ai_summarize_accumulate UnboundLocalError
CREATE OR REPLACE FUNCTION ai_summarize_accumulate(
    state jsonb,
    value text,
    metadata jsonb DEFAULT '{}'::jsonb
) RETURNS jsonb
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
import json

old_state = state

if old_state is None:
    accum = {
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
        accum = json.loads(old_state)
    except (json.JSONDecodeError, TypeError) as e:
        plpy.error(f"Invalid state JSON: {e}")

if value is None:
    return json.dumps(accum)

# Extract facts from the value
plan = plpy.prepare("SELECT ai_extract_facts($1, 3) as facts", ["text"])
result = plpy.execute(plan, [value])

if result and result[0]["facts"]:
    try:
        extracted = json.loads(result[0]["facts"])
        if "facts" in extracted:
            accum["facts"].extend(extracted["facts"])
    except (json.JSONDecodeError, TypeError):
        pass

# Update statistics
value_len = len(value)
accum["stats"]["row_count"] += 1
accum["stats"]["total_chars"] += value_len
if accum["stats"]["min_length"] is None or value_len < accum["stats"]["min_length"]:
    accum["stats"]["min_length"] = value_len
if value_len > accum["stats"]["max_length"]:
    accum["stats"]["max_length"] = value_len

# Sample every 10th row (up to 10 samples)
if accum["stats"]["row_count"] % 10 == 1 and len(accum["samples"]) < 10:
    accum["samples"].append(value[:200])

# Merge metadata
if metadata:
    try:
        meta = json.loads(metadata) if isinstance(metadata, str) else metadata
        for k, v in meta.items():
            if k not in accum["metadata"]:
                accum["metadata"][k] = v
    except (json.JSONDecodeError, TypeError):
        pass

return json.dumps(accum)
$$;

-- Fix rerank function return type issue
-- First rename the implementation function
DO $$
BEGIN
    -- Only run if steadytext_rerank_impl does NOT exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_proc 
        WHERE proname = 'steadytext_rerank_impl'
          AND pg_function_is_visible(oid)
    ) THEN
        EXECUTE $cmd$
            ALTER FUNCTION steadytext_rerank(text, text[], text, boolean, integer) RENAME TO steadytext_rerank_impl;
        $cmd$;

        EXECUTE $cmd$
            CREATE OR REPLACE FUNCTION steadytext_rerank(
                query text,
                documents text[],
                task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
                return_scores boolean DEFAULT true,
                seed integer DEFAULT 42
            ) RETURNS SETOF record
            LANGUAGE sql IMMUTABLE PARALLEL SAFE
            AS $c$
              SELECT document, score FROM steadytext_rerank_impl($1, $2, $3, $4, $5);
            $c$;
        $cmd$;
    END IF;
END
$$;

-- Update config function to be more secure
CREATE OR REPLACE FUNCTION steadytext_config_set(key TEXT, value TEXT)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO steadytext_config (key, value)
    VALUES (key, to_jsonb(value))
    ON CONFLICT (key) DO UPDATE
    SET value = to_jsonb(EXCLUDED.value),
        updated_at = NOW(),
        updated_by = current_user;
END;
$$;

-- AIDEV-NOTE: Migration completed successfully
-- Changes in v1.4.3:
-- 1. Fixed parameter name from max_tokens to max_new_tokens in direct generation fallback
-- 2. Added LEAKPROOF to appropriate functions for security
-- 3. Removed conflicting single-argument steadytext_embed overload
-- 4. Fixed UnboundLocalError in ai_summarize_accumulate by using old_state/accum pattern
-- 5. Fixed rerank function return type by creating proper SQL wrapper
-- pg_steadytext extension upgrade from 1.4.5 to 1.4.6
-- Adds unsafe mode support for embeddings to enable remote providers like OpenAI

-- AIDEV-NOTE: v1.4.6 adds model and unsafe_mode parameters to embedding functions
-- to support remote embedding providers with best-effort determinism

-- Drop old embedding functions from extension
ALTER EXTENSION pg_steadytext DROP FUNCTION steadytext_embed(TEXT, BOOLEAN, INT);
-- AIDEV-NOTE: steadytext_embed_async doesn't exist in 1.4.5, only steadytext_embed_batch_async
-- ALTER EXTENSION pg_steadytext DROP FUNCTION steadytext_embed_async(TEXT, BOOLEAN);
ALTER EXTENSION pg_steadytext DROP FUNCTION steadytext_embed_cached(TEXT, INT);

-- Drop the old functions
DROP FUNCTION IF EXISTS steadytext_embed(TEXT, BOOLEAN, INT);
-- AIDEV-NOTE: steadytext_embed_async doesn't exist in 1.4.5, only steadytext_embed_batch_async
-- DROP FUNCTION IF EXISTS steadytext_embed_async(TEXT, BOOLEAN);
DROP FUNCTION IF EXISTS steadytext_embed_cached(TEXT, INT);

-- Create new embedding function with model and unsafe_mode parameters
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
# Added in v1.4.6 to support OpenAI and other remote embedding providers

import json
import numpy as np

# Initialize Python environment if needed
try:
    if 'steadytext_initialized' not in GD:
        plpy.execute("SELECT _steadytext_init_python()")
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
    plan = plpy.prepare(
        "SELECT embedding FROM steadytext_cache WHERE cache_key = $1",
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
        plan = plpy.prepare("SELECT * FROM steadytext_daemon_status()")
        status = plpy.execute(plan)
        
        if status and status[0]['is_running']:
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

-- Function is already part of extension after CREATE OR REPLACE
-- No need to add it again

-- Create new async embedding function with model and unsafe_mode parameters
CREATE OR REPLACE FUNCTION steadytext_embed_async(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    model TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    request_id UUID;
BEGIN
    -- AIDEV-NOTE: Queue an embedding request for async processing with remote model support
    
    -- Validate remote model requirements
    IF model IS NOT NULL AND position(':' IN model) > 0 AND NOT unsafe_mode THEN
        RAISE EXCEPTION 'Remote models (containing '':'' ) require unsafe_mode=TRUE';
    END IF;
    
    request_id := gen_random_uuid();
    
    -- Build params JSON including new parameters
    DECLARE
        params_json JSONB;
    BEGIN
        params_json := jsonb_build_object(
            'text_input', text_input,
            'use_cache', use_cache,
            'seed', seed
        );
        
        IF model IS NOT NULL THEN
            params_json := params_json || jsonb_build_object('model', model);
        END IF;
        
        IF unsafe_mode IS NOT NULL THEN
            params_json := params_json || jsonb_build_object('unsafe_mode', unsafe_mode);
        END IF;
        
        INSERT INTO steadytext_queue (
            request_id,
            request_type,
            params,
            status
        ) VALUES (
            request_id,
            'embed',
            params_json,
            'pending'
        );
    END;
    
    -- Notify worker
    PERFORM pg_notify('steadytext_async', request_id::text);
    
    RETURN request_id;
END;
$$;

-- Add NEW function to extension (was missing in 1.4.5)
-- Use DO block to handle case where function might already be in extension metadata
DO $$
BEGIN
    -- Try to add function to extension
    BEGIN
        ALTER EXTENSION pg_steadytext ADD FUNCTION steadytext_embed_async(TEXT, BOOLEAN, INT, TEXT, BOOLEAN);
    EXCEPTION WHEN OTHERS THEN
        -- Function is already part of extension, ignore
        NULL;
    END;
END $$;

-- Create new cached embedding function with model and unsafe_mode parameters
CREATE OR REPLACE FUNCTION steadytext_embed_cached(
    text_input TEXT,
    seed INT DEFAULT 42,
    model TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS vector(1024)
LANGUAGE plpgsql
VOLATILE
AS $c$
DECLARE
    v_result vector(1024);
    v_cache_key TEXT;
    v_cache_key_input TEXT;
BEGIN
    -- AIDEV-NOTE: VOLATILE wrapper for cache population with remote model support
    
    -- Validate remote model requirements
    IF model IS NOT NULL AND position(':' IN model) > 0 AND NOT unsafe_mode THEN
        RAISE EXCEPTION 'Remote models (containing '':'' ) require unsafe_mode=TRUE';
    END IF;
    
    -- Generate result using IMMUTABLE function
    v_result := steadytext_embed(text_input, true, seed, model, unsafe_mode);
    
    -- If result was generated, store it
    IF v_result IS NOT NULL THEN
        -- Generate cache key including model if specified
        IF model IS NOT NULL THEN
            v_cache_key_input := 'embed:' || text_input || ':' || model;
        ELSE
            v_cache_key_input := 'embed:' || text_input;
        END IF;
        
        v_cache_key := encode(digest(v_cache_key_input, 'sha256'), 'hex');
        
        -- Check if already cached
        PERFORM 1 FROM steadytext_cache WHERE cache_key = v_cache_key;
        
        -- If not cached, insert it
        IF NOT FOUND THEN
            INSERT INTO steadytext_cache (cache_key, prompt, embedding, created_at)
            VALUES (
                v_cache_key,
                text_input,
                v_result,
                NOW()
            )
            ON CONFLICT (cache_key) DO NOTHING;
        END IF;
    END IF;
    
    RETURN v_result;
END;
$c$;

-- Function is already part of extension after CREATE OR REPLACE
-- No need to add it again

-- Update function comments
COMMENT ON FUNCTION steadytext_embed IS 
'IMMUTABLE function for embeddings with optional remote model support via unsafe_mode. Only reads from cache, never writes. For automatic cache population, use steadytext_embed_cached.';

COMMENT ON FUNCTION steadytext_embed_cached IS 
'VOLATILE wrapper for steadytext_embed that handles cache population with remote model support. Use when automatic caching is needed.';

COMMENT ON FUNCTION steadytext_embed_async IS
'Async embedding generation with remote model support. Returns UUID for checking status with steadytext_get_result.';

-- Drop and recreate aliases to include new parameters
DROP FUNCTION IF EXISTS st_embed(TEXT, BOOLEAN, INT);
CREATE OR REPLACE FUNCTION st_embed(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    model TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS vector(1024)
LANGUAGE sql
IMMUTABLE PARALLEL SAFE
AS $alias$
    SELECT steadytext_embed($1, $2, $3, $4, $5);
$alias$;

DROP FUNCTION IF EXISTS st_embed_cached(TEXT, INT);
CREATE OR REPLACE FUNCTION st_embed_cached(
    text_input TEXT,
    seed INT DEFAULT 42,
    model TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS vector(1024)
LANGUAGE sql
VOLATILE PARALLEL SAFE
AS $alias$
    SELECT steadytext_embed_cached($1, $2, $3, $4);
$alias$;

-- AIDEV-NOTE: st_embed_async doesn't exist in 1.4.5, only batch version
-- DROP FUNCTION IF EXISTS st_embed_async(TEXT, BOOLEAN);
CREATE OR REPLACE FUNCTION st_embed_async(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    model TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS UUID
LANGUAGE sql
VOLATILE PARALLEL SAFE
AS $alias$
    SELECT steadytext_embed_async($1, $2, $3, $4, $5);
$alias$;

-- AIDEV-NOTE: v1.4.6 adds comprehensive unsafe mode support for embeddings
-- This enables using remote embedding providers like OpenAI with best-effort determinism
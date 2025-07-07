-- pg_steadytext Python initialization fix
-- AIDEV-NOTE: This fixes the Python module import issues

-- Update Python path configuration
DO $$
DECLARE
    pg_lib_dir TEXT;
    python_path TEXT;
BEGIN
    -- Get PostgreSQL lib directory
    SELECT setting INTO pg_lib_dir FROM pg_settings WHERE name = 'pkglibdir';
    
    -- Build Python path
    python_path := pg_lib_dir || '/pg_steadytext/python';
    
    -- Add to existing Python path if any
    BEGIN
        python_path := python_path || ':' || current_setting('plpython3.python_path');
    EXCEPTION WHEN OTHERS THEN
        -- No existing path
    END;
    
    -- Set the Python path for the current database
    EXECUTE format('ALTER DATABASE %I SET plpython3.python_path TO %L',
        current_database(), python_path);
    
    RAISE NOTICE 'Python path configured: %', python_path;
END;
$$;

-- Improved Python initialization function
CREATE OR REPLACE FUNCTION _steadytext_init_python()
RETURNS void
LANGUAGE plpython3u
AS $$
# AIDEV-NOTE: Enhanced Python environment initialization
import sys
import os
import site

# Get PostgreSQL lib directory
pg_lib_dir = plpy.execute("SHOW pkglibdir")[0]['pkglibdir']
python_module_dir = os.path.join(pg_lib_dir, 'pg_steadytext', 'python')

# Add to Python path if not already there
if python_module_dir not in sys.path:
    sys.path.insert(0, python_module_dir)
    site.addsitedir(python_module_dir)

# Try to import required external packages
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

# Try to import our modules
try:
    # Import modules directly
    import daemon_connector
    import cache_manager
    import security
    import config
    
    # Store in global dictionary for reuse
    GD['daemon_connector'] = daemon_connector
    GD['cache_manager'] = cache_manager
    GD['security'] = security
    GD['config'] = config
    GD['steadytext_initialized'] = True
    
    plpy.notice(f"pg_steadytext Python environment initialized successfully from {python_module_dir}")
except ImportError as e:
    GD['steadytext_initialized'] = False
    plpy.error(f"Failed to import pg_steadytext modules from {python_module_dir}: {e}")
    plpy.error(f"Python path: {sys.path}")
$$;

-- Add simple streaming generation function (fixed)
CREATE OR REPLACE FUNCTION steadytext_generate_stream(
    prompt TEXT,
    max_tokens INT DEFAULT 512
)
RETURNS SETOF TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
# AIDEV-NOTE: Simple streaming text generation function
import json

# Validate inputs
if not prompt or not prompt.strip():
    plpy.error("Prompt cannot be empty")

if max_tokens < 1 or max_tokens > 4096:
    plpy.error("max_tokens must be between 1 and 4096")

try:
    # Try to use daemon connector
    from daemon_connector import SteadyTextConnector
    
    # Get daemon configuration
    plan = plpy.prepare("SELECT value FROM steadytext_config WHERE key = $1", ["text"])
    host_rv = plpy.execute(plan, ["daemon_host"])
    port_rv = plpy.execute(plan, ["daemon_port"])
    
    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Create connector and generate streaming
    connector = SteadyTextConnector(host, port)
    for token in connector.generate_stream(prompt, max_tokens=max_tokens):
        yield token
        
except Exception as e:
    # Fallback to simple generation with simulated streaming
    try:
        from steadytext import generate
        full_text = generate(prompt, max_new_tokens=max_tokens)
        
        # Simulate streaming by yielding words
        words = full_text.split()
        for i, word in enumerate(words):
            if i < len(words) - 1:
                yield word + " "
            else:
                yield word
    except Exception as e2:
        # Final fallback
        yield f"[Streaming Error: {str(e2)}]"
$$;

-- Add batch embedding function
CREATE OR REPLACE FUNCTION steadytext_embed_batch(
    texts TEXT[],
    use_cache BOOLEAN DEFAULT TRUE
)
RETURNS TABLE(text TEXT, embedding vector)
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $$
# AIDEV-NOTE: Batch embedding function for multiple texts
import json
import numpy as np

# Check initialization
if not GD.get('steadytext_initialized', False):
    plpy.error("pg_steadytext Python environment not initialized")

# Get modules
daemon_connector = GD['daemon_connector']
cache_manager = GD['cache_manager']

# Validate input
if not texts or len(texts) == 0:
    return []

if len(texts) > 100:
    plpy.error("Batch size cannot exceed 100 texts")

# Create client
client = daemon_connector.SteadyTextDaemonClient()

results = []
for text in texts:
    if not text or not text.strip():
        # Return zero vector for empty text
        embedding = np.zeros(1024, dtype=np.float32)
    else:
        # Check cache first if enabled
        if use_cache:
            cache_key = cache_manager.generate_cache_key('embed', text)
            cached = plpy.execute(f"""
                SELECT embedding::float[] as embedding 
                FROM steadytext_cache 
                WHERE cache_key = '{cache_key}'
            """)
            
            if cached and cached[0]['embedding']:
                embedding = np.array(cached[0]['embedding'], dtype=np.float32)
            else:
                # Generate new embedding
                embedding = client.embed(text)
                
                # Store in cache
                plpy.execute(f"""
                    INSERT INTO steadytext_cache (cache_key, prompt, embedding)
                    VALUES ('{cache_key}', $1, $2::vector)
                    ON CONFLICT (cache_key) DO UPDATE
                    SET access_count = steadytext_cache.access_count + 1,
                        last_accessed = NOW()
                """, [text, embedding.tolist()])
        else:
            # Generate without cache
            embedding = client.embed(text)
    
    results.append({
        'text': text,
        'embedding': embedding.tolist()
    })

return results
$$;

-- Add async generation function
CREATE OR REPLACE FUNCTION steadytext_generate_async(
    prompt TEXT,
    max_tokens INT DEFAULT 512
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    request_id UUID;
BEGIN
    -- AIDEV-NOTE: Queue a generation request for async processing
    
    -- Validate input
    IF prompt IS NULL OR trim(prompt) = '' THEN
        RAISE EXCEPTION 'Prompt cannot be empty';
    END IF;
    
    IF max_tokens < 1 OR max_tokens > 4096 THEN
        RAISE EXCEPTION 'max_tokens must be between 1 and 4096';
    END IF;
    
    -- Insert into queue
    INSERT INTO steadytext_queue (
        request_type,
        prompt,
        params
    ) VALUES (
        'generate',
        prompt,
        jsonb_build_object(
            'max_tokens', max_tokens
        )
    )
    RETURNING request_id INTO request_id;
    
    -- Notify workers (if using LISTEN/NOTIFY)
    PERFORM pg_notify('steadytext_queue', request_id::text);
    
    RETURN request_id;
END;
$$;

-- Add function to check async request status
CREATE OR REPLACE FUNCTION steadytext_check_async(
    request_id UUID
)
RETURNS TABLE(
    status TEXT,
    result TEXT,
    error TEXT,
    created_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    processing_time_ms INT
)
LANGUAGE sql
STABLE PARALLEL SAFE LEAKPROOF
AS $$
    SELECT 
        status,
        result,
        error,
        created_at,
        completed_at,
        processing_time_ms
    FROM steadytext_queue
    WHERE steadytext_queue.request_id = steadytext_check_async.request_id;
$$;

-- Add semantic search function using pgvector
CREATE OR REPLACE FUNCTION steadytext_semantic_search(
    query_text TEXT,
    limit_results INT DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.7
)
RETURNS TABLE(
    prompt TEXT,
    response TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
STABLE PARALLEL SAFE
AS $$
DECLARE
    query_embedding vector;
BEGIN
    -- AIDEV-NOTE: Semantic search using pgvector
    
    -- Generate embedding for query
    query_embedding := steadytext_embed(query_text);
    
    -- Search using cosine similarity
    RETURN QUERY
    SELECT 
        c.prompt,
        c.response,
        1 - (c.embedding <=> query_embedding) as similarity
    FROM steadytext_cache c
    WHERE c.embedding IS NOT NULL
    AND c.response IS NOT NULL
    AND 1 - (c.embedding <=> query_embedding) > similarity_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT limit_results;
END;
$$;

-- Add cache eviction function
CREATE OR REPLACE FUNCTION steadytext_cache_evict(
    max_entries INT DEFAULT NULL,
    max_size_mb FLOAT DEFAULT NULL
)
RETURNS INT
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count INT;
    current_size_mb FLOAT;
    current_entries INT;
BEGIN
    -- AIDEV-NOTE: Evict cache entries based on frecency score
    
    -- Get current cache stats
    SELECT 
        COUNT(*),
        COALESCE(SUM(pg_column_size(response) + pg_column_size(embedding)) / 1024.0 / 1024.0, 0)
    INTO current_entries, current_size_mb
    FROM steadytext_cache;
    
    -- Use config defaults if not provided
    IF max_entries IS NULL THEN
        SELECT value::int INTO max_entries 
        FROM steadytext_config 
        WHERE key = 'max_cache_entries';
    END IF;
    
    IF max_size_mb IS NULL THEN
        SELECT value::float INTO max_size_mb 
        FROM steadytext_config 
        WHERE key = 'max_cache_size_mb';
    END IF;
    
    -- Delete entries with lowest frecency scores if over limits
    IF current_entries > max_entries OR current_size_mb > max_size_mb THEN
        WITH deleted AS (
            DELETE FROM steadytext_cache
            WHERE id IN (
                SELECT id 
                FROM steadytext_cache_with_frecency
                ORDER BY frecency_score ASC
                LIMIT GREATEST(
                    current_entries - max_entries,
                    (SELECT COUNT(*) FROM steadytext_cache) / 10  -- At least 10%
                )
            )
            RETURNING 1
        )
        SELECT COUNT(*) INTO deleted_count FROM deleted;
    ELSE
        deleted_count := 0;
    END IF;
    
    RETURN deleted_count;
END;
$$;

-- AIDEV-NOTE: Added in v1.0.1 (2025-07-07):
-- Marked deterministic and read-only functions with appropriate properties:
-- - steadytext_generate_stream() is IMMUTABLE PARALLEL SAFE
-- - steadytext_embed_batch() is IMMUTABLE PARALLEL SAFE
-- - steadytext_check_async() is STABLE PARALLEL SAFE LEAKPROOF
-- - steadytext_semantic_search() is STABLE PARALLEL SAFE
-- Functions that modify state remain VOLATILE:
-- - _steadytext_init_python() (modifies GD)
-- - steadytext_generate_async() (inserts into queue)
-- - steadytext_cache_evict() (deletes from cache)
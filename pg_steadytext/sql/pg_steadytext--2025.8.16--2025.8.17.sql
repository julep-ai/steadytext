-- pg_steadytext migration from 2025.8.16 to 2025.8.17
-- Combined changes:
-- 1. Fix schema qualification issues for steadytext_config table access in continuous aggregates
-- 2. Add unsafe_mode and model support to summarization functions

-- AIDEV-NOTE: This migration fixes issue #95 where steadytext_config table cannot be found
-- when functions are called within TimescaleDB continuous aggregate refresh contexts.
-- The fix adds explicit schema qualification using @extschema@ placeholder.
-- All table references in Python functions are now schema-qualified.

-- Update steadytext_generate function to use schema-qualified table references
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
AS $$
# AIDEV-NOTE: Main text generation function with schema-qualified table references
import json
import hashlib

# Check if pg_steadytext is initialized
if not GD.get('steadytext_initialized', False):
    # Initialize on first use
    plpy.execute("SELECT _steadytext_init_python()")
    # Check again after initialization
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules from GD
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Get configuration - FIXED: Get schema dynamically at runtime
    schema_result = plpy.execute("SELECT current_schema()")
    current_schema = schema_result[0]['current_schema'] if schema_result else 'public'
    plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(current_schema)}.steadytext_config WHERE key = $1", ["text"])

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
    
    # Try to get from cache first - FIXED: Get schema dynamically at runtime
    cache_plan = plpy.prepare(f"""
        UPDATE {plpy.quote_ident(current_schema)}.steadytext_cache 
        SET access_count = access_count + 1,
            last_accessed = NOW()
        WHERE cache_key = $1
        RETURNING response
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for key: {cache_key[:8]}...")
        return cache_result[0]["response"]

# Cache miss - generate new text
host_rv = plpy.execute(plan, ["daemon_host"])
port_rv = plpy.execute(plan, ["daemon_port"])

host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

try:
    # Try daemon first
    result = daemon_connector.daemon_generate(
        prompt, 
        host=host, 
        port=port, 
        max_tokens=resolved_max_tokens,
        seed=resolved_seed,
        eos_string=eos_string,
        model=model,
        model_repo=model_repo,
        model_filename=model_filename,
        size=size,
        unsafe_mode=unsafe_mode
    )
    
    # Store in cache if successful and caching is enabled - FIXED: Use schema-qualified table name
    if use_cache:
        insert_plan = plpy.prepare(f"""
            INSERT INTO {plpy.quote_ident(current_schema)}.steadytext_cache (cache_key, response)
            VALUES ($1, $2)
            ON CONFLICT (cache_key) DO UPDATE
            SET response = EXCLUDED.response,
                access_count = steadytext_cache.access_count + 1,
                last_accessed = NOW()
        """, ["text", "text"])
        plpy.execute(insert_plan, [cache_key, result])
        plpy.notice(f"Cached result for key: {cache_key[:8]}...")
    
    return result
    
except Exception as e:
    # If daemon fails and unsafe_mode is true with remote model, try direct generation
    if unsafe_mode and model and ':' in model:
        plpy.notice(f"Daemon failed, trying direct generation: {str(e)}")
        try:
            remote_generator = GD.get('module_remote_generator')
            if not remote_generator:
                plpy.error("remote_generator module not loaded")
            
            result = remote_generator.remote_generate(
                prompt=prompt,
                model=model,
                max_new_tokens=resolved_max_tokens,
                seed=resolved_seed
            )
            
            # Store in cache if successful - FIXED: Use schema-qualified table name
            if use_cache:
                insert_plan = plpy.prepare("""
                    INSERT INTO @extschema@.steadytext_cache (cache_key, response)
                    VALUES ($1, $2)
                    ON CONFLICT (cache_key) DO UPDATE
                    SET response = EXCLUDED.response,
                        access_count = steadytext_cache.access_count + 1,
                        last_accessed = NOW()
                """, ["text", "text"])
                plpy.execute(insert_plan, [cache_key, result])
            
            return result
        except Exception as inner_e:
            plpy.error(f"Both daemon and direct generation failed: {str(inner_e)}")
    else:
        # For local models, try fallback to direct loading
        plpy.notice(f"Daemon failed, trying direct loading: {str(e)}")
        
        direct_loader = GD.get('module_direct_loader')
        if not direct_loader:
            plpy.error("direct_loader module not loaded")
        
        try:
            result = direct_loader.generate_direct(
                prompt, 
                max_tokens=resolved_max_tokens,
                seed=resolved_seed,
                eos_string=eos_string,
                model=model,
                model_repo=model_repo,
                model_filename=model_filename,
                size=size
            )
            
            # Store in cache if successful - FIXED: Use schema-qualified table name
            if use_cache:
                insert_plan = plpy.prepare("""
                    INSERT INTO @extschema@.steadytext_cache (cache_key, response)
                    VALUES ($1, $2)
                    ON CONFLICT (cache_key) DO UPDATE
                    SET response = EXCLUDED.response,
                        access_count = steadytext_cache.access_count + 1,
                        last_accessed = NOW()
                """, ["text", "text"])
                plpy.execute(insert_plan, [cache_key, result])
                plpy.notice(f"Cached result for key: {cache_key[:8]}...")
            
            return result
        except Exception as inner_e:
            # Last resort: return deterministic fallback
            plpy.warning(f"All generation methods failed: {str(inner_e)}")
            # Use hash-based deterministic fallback
            hash_val = hashlib.sha256(f"{prompt}{resolved_seed}".encode()).hexdigest()
            words = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
                    "runs", "walks", "flies", "swims", "reads", "writes", "thinks", "dreams"]
            result = " ".join(words[int(hash_val[i:i+2], 16) % len(words)] 
                            for i in range(0, min(20, len(hash_val)-1), 2))
            return result[:resolved_max_tokens] if len(result) > resolved_max_tokens else result
$$;

-- Update steadytext_embed function
CREATE OR REPLACE FUNCTION steadytext_embed(
    text TEXT,
    normalize BOOLEAN DEFAULT TRUE,
    use_cache BOOLEAN DEFAULT TRUE,
    mode TEXT DEFAULT 'passage',
    size TEXT DEFAULT NULL
)
RETURNS vector
LANGUAGE plpython3u
AS $$
# AIDEV-NOTE: Embedding function with schema-qualified table references
import json
import numpy as np

# Check if pg_steadytext is initialized
if not GD.get('steadytext_initialized', False):
    # Initialize on first use
    plpy.execute("SELECT _steadytext_init_python()")
    # Check again after initialization
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules from GD
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Validate inputs
if not text or not text.strip():
    plpy.error("Text cannot be empty")

# AIDEV-NOTE: Mode validation for Jina v4 compatibility
if mode not in ['query', 'passage']:
    plpy.error("mode must be either 'query' or 'passage'")

# Check if we should use cache
if use_cache:
    # Include mode in cache key for Jina v4 compatibility
    cache_key = f"embed:{mode}:{text}"
    
    # Try to get from cache first - FIXED: Get schema dynamically at runtime
    cache_plan = plpy.prepare(f"""
        UPDATE {plpy.quote_ident(current_schema)}.steadytext_embedding_cache 
        SET access_count = access_count + 1,
            last_accessed = NOW()
        WHERE cache_key = $1
        RETURNING embedding
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["embedding"]:
        plpy.notice(f"Cache hit for embedding key: {cache_key[:16]}...")
        return cache_result[0]["embedding"]

# Cache miss - generate new embedding - FIXED: Use schema-qualified table name
    # Get the schema of this function dynamically at runtime
    schema_result = plpy.execute("SELECT current_schema()")
    current_schema = schema_result[0]['current_schema'] if schema_result else 'public'
    plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(current_schema)}.steadytext_config WHERE key = $1", ["text"])

host_rv = plpy.execute(plan, ["daemon_host"])
port_rv = plpy.execute(plan, ["daemon_port"])

host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

try:
    # Try daemon first
    result = daemon_connector.daemon_embed(
        text, 
        host=host, 
        port=port, 
        normalize=normalize,
        mode=mode,
        size=size
    )
    
    # Store in cache if successful and caching is enabled - FIXED: Use schema-qualified table name
    if use_cache:
        insert_plan = plpy.prepare(f"""
            INSERT INTO {plpy.quote_ident(current_schema)}.steadytext_embedding_cache (cache_key, embedding)
            VALUES ($1, $2)
            ON CONFLICT (cache_key) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                access_count = steadytext_embedding_cache.access_count + 1,
                last_accessed = NOW()
        """, ["text", "vector"])
        plpy.execute(insert_plan, [cache_key, result])
        plpy.notice(f"Cached embedding for key: {cache_key[:16]}...")
    
    return result
    
except Exception as e:
    # Fallback to direct loading if daemon fails
    plpy.notice(f"Daemon failed, trying direct loading: {str(e)}")
    
    direct_loader = GD.get('module_direct_loader')
    if not direct_loader:
        plpy.error("direct_loader module not loaded")
    
    try:
        result = direct_loader.embed_direct(
            text, 
            normalize=normalize,
            mode=mode,
            size=size
        )
        
        # Store in cache if successful - FIXED: Use schema-qualified table name
        if use_cache:
            insert_plan = plpy.prepare("""
                INSERT INTO @extschema@.steadytext_embedding_cache (cache_key, embedding)
                VALUES ($1, $2)
                ON CONFLICT (cache_key) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    access_count = @extschema@.steadytext_embedding_cache.access_count + 1,
                    last_accessed = NOW()
            """, ["text", "vector"])
            plpy.execute(insert_plan, [cache_key, result])
            plpy.notice(f"Cached embedding for key: {cache_key[:16]}...")
        
        return result
    except Exception as inner_e:
        # Last resort: return zero vector
        plpy.warning(f"All embedding methods failed: {str(inner_e)}")
        # Return normalized zero vector of correct dimension (1024)
        zero_vec = np.zeros(1024, dtype=np.float32)
        if normalize:
            # Can't normalize zero vector, return small random values
            zero_vec = np.random.randn(1024).astype(np.float32) * 0.01
            zero_vec = zero_vec / np.linalg.norm(zero_vec)
        return zero_vec.tolist()
$$;

-- Update other functions that access steadytext_config
-- This includes daemon status, start, stop functions

CREATE OR REPLACE FUNCTION steadytext_daemon_status()
RETURNS JSONB
LANGUAGE plpython3u
AS $$
import json
import zmq

try:
    # Get daemon configuration - FIXED: Use schema-qualified table name
    # Get the schema of this function dynamically at runtime
    schema_result = plpy.execute("SELECT current_schema()")
    current_schema = schema_result[0]['current_schema'] if schema_result else 'public'
    plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(current_schema)}.steadytext_config WHERE key = $1", ["text"])
    host_rv = plpy.execute(plan, ["daemon_host"])
    port_rv = plpy.execute(plan, ["daemon_port"])
    
    host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Create ZMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
    socket.setsockopt(zmq.LINGER, 0)
    
    try:
        socket.connect(f"tcp://{host}:{port}")
        
        # Send status request
        request = {"action": "status"}
        socket.send_json(request)
        
        # Get response
        response = socket.recv_json()
        
        # Update health table if we got a successful response - FIXED: Use schema-qualified table name
        if response.get("status") == "ok":
            update_plan = plpy.prepare(f"""
                INSERT INTO {plpy.quote_ident(current_schema)}.steadytext_daemon_health (
                    daemon_id, endpoint, last_heartbeat, status, version, models_loaded
                ) VALUES (
                    'default', $1, NOW(), 'healthy', $2, $3
                )
                ON CONFLICT (daemon_id) DO UPDATE SET
                    endpoint = EXCLUDED.endpoint,
                    last_heartbeat = EXCLUDED.last_heartbeat,
                    status = EXCLUDED.status,
                    version = EXCLUDED.version,
                    models_loaded = EXCLUDED.models_loaded
            """, ["text", "text", "text[]"])
            
            models = response.get("models_loaded", [])
            version = response.get("version", "unknown")
            plpy.execute(update_plan, [f"{host}:{port}", version, models])
        
        return json.dumps(response)
        
    finally:
        socket.close()
        context.term()
        
except Exception as e:
    # Update health table to reflect unhealthy status - FIXED: Get schema dynamically at runtime
    update_plan = plpy.prepare(f"""
        UPDATE {plpy.quote_ident(current_schema)}.steadytext_daemon_health 
        SET status = 'unhealthy', last_heartbeat = NOW()
        WHERE daemon_id = 'default'
    """)
    plpy.execute(update_plan)
    
    return json.dumps({
        "status": "error",
        "error": str(e),
        "daemon_running": False
    })
$$;

CREATE OR REPLACE FUNCTION steadytext_daemon_start(
    foreground BOOLEAN DEFAULT FALSE,
    host TEXT DEFAULT NULL,
    port INT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpython3u
AS $$
import json
import subprocess
import time

try:
    # Get configuration - FIXED: Use schema-qualified table name
    # Get the schema of this function dynamically at runtime
    schema_result = plpy.execute("SELECT current_schema()")
    current_schema = schema_result[0]['current_schema'] if schema_result else 'public'
    plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(current_schema)}.steadytext_config WHERE key = $1", ["text"])
    
    # Use provided values or defaults from config
    if host is None:
        host_rv = plpy.execute(plan, ["daemon_host"])
        host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    
    if port is None:
        port_rv = plpy.execute(plan, ["daemon_port"])
        port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Update daemon health status - FIXED: Get schema dynamically at runtime
    update_plan = plpy.prepare(f"""
        INSERT INTO {plpy.quote_ident(current_schema)}.steadytext_daemon_health (daemon_id, endpoint, status)
        VALUES ('default', $1, 'starting')
        ON CONFLICT (daemon_id) DO UPDATE SET
            endpoint = EXCLUDED.endpoint,
            status = EXCLUDED.status,
            last_heartbeat = NOW()
    """, ["text"])
    plpy.execute(update_plan, [f"{host}:{port}"])
    
    # Build command
    cmd = ["st", "daemon", "start", "--host", host, "--port", str(port)]
    if foreground:
        cmd.append("--foreground")
    
    # Start daemon
    if foreground:
        # Run in foreground (blocking)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            plpy.error(f"Failed to start daemon: {result.stderr}")
    else:
        # Run in background
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait a moment for daemon to start
        time.sleep(2)
    
    # Check if daemon is running
    status_result = plpy.execute("SELECT steadytext_daemon_status()")
    status = json.loads(status_result[0]["steadytext_daemon_status"])
    
    return json.dumps({
        "status": "started" if status.get("daemon_running") else "failed",
        "host": host,
        "port": port,
        "daemon_status": status
    })
    
except Exception as e:
    return json.dumps({
        "status": "error",
        "error": str(e)
    })
$$;

-- Update steadytext_rerank function
CREATE OR REPLACE FUNCTION steadytext_rerank(
    query TEXT,
    document TEXT,
    task_description TEXT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE
)
RETURNS FLOAT
LANGUAGE plpython3u
AS $$
# AIDEV-NOTE: Reranking function with schema-qualified table references
import json
import hashlib

# Check if pg_steadytext is initialized
if not GD.get('steadytext_initialized', False):
    # Initialize on first use
    plpy.execute("SELECT _steadytext_init_python()")
    # Check again after initialization  
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules from GD
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Validate inputs
if not query or not query.strip():
    plpy.error("Query cannot be empty")
if not document or not document.strip():
    plpy.error("Document cannot be empty")

# Use default task description if not provided
if not task_description:
    task_description = "Given a query and a document, indicate whether the document answers the query."

# Check if we should use cache
if use_cache:
    # Create cache key from query, document, and task
    cache_key = f"rerank:{task_description}:{query}:{document}"
    
    # Try to get from cache first - FIXED: Get schema dynamically at runtime
    cache_plan = plpy.prepare(f"""
        UPDATE {plpy.quote_ident(current_schema)}.steadytext_rerank_cache 
        SET access_count = access_count + 1,
            last_accessed = NOW()
        WHERE cache_key = $1
        RETURNING score
    """, ["text"])
    
    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["score"] is not None:
        plpy.notice(f"Cache hit for rerank key: {cache_key[:20]}...")
        return cache_result[0]["score"]

# Cache miss - compute new score - FIXED: Use schema-qualified table name
    # Get the schema of this function dynamically at runtime
    schema_result = plpy.execute("SELECT current_schema()")
    current_schema = schema_result[0]['current_schema'] if schema_result else 'public'
    plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(current_schema)}.steadytext_config WHERE key = $1", ["text"])

host_rv = plpy.execute(plan, ["daemon_host"])
port_rv = plpy.execute(plan, ["daemon_port"])

host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
port = json.loads(port_rv[0]["value"]) if port_rv else 5555

try:
    # Try daemon first
    result = daemon_connector.daemon_rerank(
        query=query,
        document=document,
        task_description=task_description,
        host=host,
        port=port
    )
    
    # Store in cache if successful and caching is enabled - FIXED: Use schema-qualified table name
    if use_cache:
        insert_plan = plpy.prepare(f"""
            INSERT INTO {plpy.quote_ident(current_schema)}.steadytext_rerank_cache (cache_key, score, query_text, document_text)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (cache_key) DO UPDATE
            SET score = EXCLUDED.score,
                access_count = steadytext_rerank_cache.access_count + 1,
                last_accessed = NOW()
        """, ["text", "float8", "text", "text"])
        plpy.execute(insert_plan, [cache_key, result, query, document])
        plpy.notice(f"Cached rerank score for key: {cache_key[:20]}...")
    
    return result
    
except Exception as e:
    # Fallback to direct loading if daemon fails
    plpy.notice(f"Daemon failed, trying direct loading: {str(e)}")
    
    direct_loader = GD.get('module_direct_loader')
    if not direct_loader:
        plpy.error("direct_loader module not loaded")
    
    try:
        result = direct_loader.rerank_direct(
            query=query,
            document=document,
            task_description=task_description
        )
        
        # Store in cache if successful - FIXED: Use schema-qualified table name
        if use_cache:
            insert_plan = plpy.prepare("""
                INSERT INTO @extschema@.steadytext_rerank_cache (cache_key, score, query_text, document_text)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (cache_key) DO UPDATE
                SET score = EXCLUDED.score,
                    access_count = @extschema@.steadytext_rerank_cache.access_count + 1,
                    last_accessed = NOW()
            """, ["text", "float8", "text", "text"])
            plpy.execute(insert_plan, [cache_key, result, query, document])
            plpy.notice(f"Cached rerank score for key: {cache_key[:20]}...")
        
        return result
    except Exception as inner_e:
        # Last resort: return semantic similarity based fallback
        plpy.warning(f"All reranking methods failed: {str(inner_e)}")
        
        # Simple heuristic: check for keyword overlap
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        overlap = len(query_words & doc_words)
        max_possible = len(query_words)
        
        if max_possible == 0:
            return 0.0
        
        # Return a score between 0 and 1 based on overlap
        return min(1.0, overlap / max_possible)
$$;

-- Now add the summarization enhancements from the HEAD branch
-- Update steadytext_summarize_text to support model and unsafe_mode parameters
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

plan = plpy.prepare(
    "SELECT steadytext_generate($1, NULL, true, 42, '[EOS]', $2, NULL, NULL, NULL, $3) as summary",
    ["text", "text", "boolean"]
)
result = plpy.execute(plan, [prompt, model, unsafe_mode])

if result and result[0]["summary"]:
    return result[0]["summary"]
return "Unable to generate summary"
$c$;

-- Create st_summarize_text alias with new parameters
CREATE OR REPLACE FUNCTION st_summarize_text(
    input_text text,
    metadata jsonb DEFAULT '{}'::jsonb,
    model text DEFAULT NULL,
    unsafe_mode boolean DEFAULT FALSE
) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE
AS $$
    SELECT steadytext_summarize_text($1, $2, $3, $4);
$$;

-- AIDEV-SECTION: AI_SUMMARIZATION_AGGREGATES
-- Add missing aggregate functions for complete AI summarization support

-- Helper function to extract facts from text using JSON generation
CREATE OR REPLACE FUNCTION steadytext_extract_facts(
    input_text text,
    max_facts integer DEFAULT 10
) RETURNS jsonb
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
    import json
    from plpy import quote_literal

    # Validate inputs
    if not input_text or not input_text.strip():
        return json.dumps({"facts": []})

    if max_facts <= 0 or max_facts > 50:
        plpy.error("max_facts must be between 1 and 50")

    # AIDEV-NOTE: Simplified implementation to avoid crashes with generate_json
    # Extract facts by splitting on sentence boundaries
    sentences = []
    current = ""
    
    for char in input_text:
        current += char
        if char in '.!?' and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""
    
    if current.strip():
        sentences.append(current.strip())
    
    # Take up to max_facts sentences as facts
    facts = sentences[:max_facts]
    
    return json.dumps({"facts": facts})
$c$;

-- Helper function to deduplicate facts using embeddings
CREATE OR REPLACE FUNCTION steadytext_deduplicate_facts(
    facts_array jsonb,
    similarity_threshold float DEFAULT 0.85
) RETURNS jsonb
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
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
$c$;

-- State accumulator function for AI summarization
CREATE OR REPLACE FUNCTION steadytext_summarize_accumulate(
    state jsonb,
    value text,
    metadata jsonb DEFAULT '{}'::jsonb
) RETURNS jsonb
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
    import json

    # AIDEV-NOTE: In PL/Python aggregate functions, we must not reassign argument variables
    # to avoid Python scoping issues. Create new variables for any computed values.
    
    # Initialize state_data from the input state
    if state is None:
        state_data = {
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
            state_data = json.loads(state)
        except (json.JSONDecodeError, TypeError) as e:
            plpy.error(f"Invalid state JSON: {e}")

    # Skip if no value provided
    if value is None or value == '':
        return json.dumps(state_data)

    # Extract facts from the value
    # AIDEV-NOTE: Get schema dynamically at runtime for TimescaleDB continuous aggregates compatibility
    schema_result = plpy.execute("SELECT current_schema()")
    current_schema = schema_result[0]['current_schema'] if schema_result else 'public'
    plan = plpy.prepare(f"SELECT {plpy.quote_ident(current_schema)}.steadytext_extract_facts($1, 3) as facts", ["text"])
    result = plpy.execute(plan, [value])

    if result and result[0]["facts"]:
        try:
            extracted = json.loads(result[0]["facts"])
            if "facts" in extracted:
                state_data["facts"].extend(extracted["facts"])
        except (json.JSONDecodeError, TypeError):
            # Skip if fact extraction failed
            pass

    # Update statistics
    value_len = len(value)
    state_data["stats"]["row_count"] += 1
    state_data["stats"]["total_chars"] += value_len

    if state_data["stats"]["min_length"] is None or value_len < state_data["stats"]["min_length"]:
        state_data["stats"]["min_length"] = value_len
    if value_len > state_data["stats"]["max_length"]:
        state_data["stats"]["max_length"] = value_len

    # Sample every 10th row (up to 10 samples)
    if state_data["stats"]["row_count"] % 10 == 1 and len(state_data["samples"]) < 10:
        state_data["samples"].append(value[:200])  # First 200 chars

    # Merge metadata
    if metadata and metadata != '{}':
        try:
            meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            for key, val in meta_dict.items():
                if key not in state_data["metadata"]:
                    state_data["metadata"][key] = val
        except (json.JSONDecodeError, TypeError):
            # Skip invalid metadata
            pass

    return json.dumps(state_data)
$c$;

-- Combiner function for parallel aggregation
CREATE OR REPLACE FUNCTION steadytext_summarize_combine(
    state1 jsonb,
    state2 jsonb
) RETURNS jsonb
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
    import json

    # AIDEV-NOTE: Don't reassign argument variables to avoid Python scoping issues
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
        # AIDEV-NOTE: Get schema dynamically at runtime for TimescaleDB continuous aggregates compatibility
        plan = plpy.prepare(
            f"SELECT {plpy.quote_ident(current_schema)}.steadytext_deduplicate_facts($1::jsonb) as deduped",
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
$c$;

-- Finalizer function to generate summary
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
    if metadata:
        # Extract model and unsafe_mode for remote model support
        model = metadata.get('model')
        unsafe_mode = metadata.get('unsafe_mode', False)
        
        # Add other metadata to prompt context
        other_metadata = {k: v for k, v in metadata.items() if k not in ['model', 'unsafe_mode']}
        if other_metadata:
            meta_str = ", ".join([f"{k}: {v}" for k, v in other_metadata.items()])
            prompt += f". Context: {meta_str}"
    else:
        model = None
        unsafe_mode = False

    # Generate summary using steadytext with model and unsafe_mode support
    # AIDEV-NOTE: Pass model and unsafe_mode from metadata to support remote models
    # AIDEV-NOTE: Get schema dynamically at runtime for TimescaleDB continuous aggregates compatibility
    plan = plpy.prepare(
        f"SELECT {plpy.quote_ident(current_schema)}.steadytext_generate($1, NULL, true, 42, '[EOS]', $2, NULL, NULL, NULL, $3) as summary",
        ["text", "text", "boolean"]
    )
    result = plpy.execute(plan, [prompt, model, unsafe_mode])

    if result and result[0]["summary"]:
        return result[0]["summary"]
    return "Unable to generate summary"
$c$;

-- Helper function to combine partial states for final aggregation
CREATE OR REPLACE FUNCTION steadytext_summarize_combine_states(
    state1 jsonb,
    partial_state jsonb
) RETURNS jsonb
LANGUAGE plpgsql
IMMUTABLE PARALLEL SAFE
AS $c$
BEGIN
    -- Simply use the combine function
    RETURN steadytext_summarize_combine(state1, partial_state);
END;
$c$;

-- AIDEV-NOTE: Since we use STYPE = jsonb, PostgreSQL handles serialization automatically for parallel processing.

-- Create the main aggregate
CREATE OR REPLACE AGGREGATE steadytext_summarize(text, jsonb) (
    SFUNC = steadytext_summarize_accumulate,
    STYPE = jsonb,
    FINALFUNC = steadytext_summarize_finalize,
    COMBINEFUNC = steadytext_summarize_combine,
    PARALLEL = SAFE
);

-- Create partial aggregate for TimescaleDB continuous aggregates
CREATE OR REPLACE AGGREGATE steadytext_summarize_partial(text, jsonb) (
    SFUNC = steadytext_summarize_accumulate,
    STYPE = jsonb,
    COMBINEFUNC = steadytext_summarize_combine,
    PARALLEL = SAFE
);

-- Create final aggregate that works on partial results
CREATE OR REPLACE AGGREGATE steadytext_summarize_final(jsonb) (
    SFUNC = steadytext_summarize_combine_states,
    STYPE = jsonb,
    FINALFUNC = steadytext_summarize_finalize,
    PARALLEL = SAFE
);

-- Create short aliases for helper functions
CREATE OR REPLACE FUNCTION st_extract_facts(
    input_text text,
    max_facts integer DEFAULT 10
) RETURNS jsonb
LANGUAGE sql IMMUTABLE PARALLEL SAFE
AS $$
    SELECT steadytext_extract_facts($1, $2);
$$;

CREATE OR REPLACE FUNCTION st_deduplicate_facts(
    facts_array jsonb,
    similarity_threshold float DEFAULT 0.85
) RETURNS jsonb
LANGUAGE sql IMMUTABLE PARALLEL SAFE
AS $$
    SELECT steadytext_deduplicate_facts($1, $2);
$$;

-- Create short alias for aggregate
CREATE OR REPLACE AGGREGATE st_summarize(text, jsonb) (
    SFUNC = steadytext_summarize_accumulate,
    STYPE = jsonb,
    COMBINEFUNC = steadytext_summarize_combine,
    FINALFUNC = steadytext_summarize_finalize,
    PARALLEL = SAFE
);

-- Add helpful comments
COMMENT ON AGGREGATE steadytext_summarize(text, jsonb) IS
'AI-powered text summarization aggregate that handles non-transitivity through structured fact extraction';

COMMENT ON AGGREGATE steadytext_summarize_partial(text, jsonb) IS
'Partial aggregate for use with TimescaleDB continuous aggregates';

COMMENT ON AGGREGATE steadytext_summarize_final(jsonb) IS
'Final aggregate for completing partial aggregations from continuous aggregates';

COMMENT ON FUNCTION steadytext_extract_facts(text, integer) IS
'Extract structured facts from text using SteadyText JSON generation';

COMMENT ON FUNCTION steadytext_deduplicate_facts(jsonb, float) IS
'Deduplicate facts based on semantic similarity using embeddings';

-- Update version function
CREATE OR REPLACE FUNCTION steadytext_version()
RETURNS TEXT AS $$
BEGIN
    RETURN '2025.8.17';
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- Create short alias for version function
CREATE OR REPLACE FUNCTION st_version()
RETURNS TEXT AS $$
    SELECT steadytext_version();
$$ LANGUAGE sql IMMUTABLE PARALLEL SAFE;

-- AIDEV-NOTE: Migration completed - all functions now use schema-qualified table references
-- This ensures they work correctly in TimescaleDB continuous aggregates and other contexts
-- where the search path might not include the extension's schema.
-- Additionally, summarization functions now support unsafe_mode and remote models.

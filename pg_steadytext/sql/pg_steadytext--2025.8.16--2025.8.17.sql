-- pg_steadytext migration from 2025.8.16 to 2025.8.17
-- Fix schema qualification issues for steadytext_config table access in continuous aggregates

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

# Get configuration - FIXED: Use schema-qualified table name
plan = plpy.prepare("SELECT value FROM @extschema@.steadytext_config WHERE key = $1", ["text"])

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
    
    # Try to get from cache first - FIXED: Use schema-qualified table name
    cache_plan = plpy.prepare("""
        UPDATE @extschema@.steadytext_cache 
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
    
    # Try to get from cache first - FIXED: Use schema-qualified table name
    cache_plan = plpy.prepare("""
        UPDATE @extschema@.steadytext_embedding_cache 
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
plan = plpy.prepare("SELECT value FROM @extschema@.steadytext_config WHERE key = $1", ["text"])

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
        insert_plan = plpy.prepare("""
            INSERT INTO @extschema@.steadytext_embedding_cache (cache_key, embedding)
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
                    access_count = steadytext_embedding_cache.access_count + 1,
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
    plan = plpy.prepare("SELECT value FROM @extschema@.steadytext_config WHERE key = $1", ["text"])
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
            update_plan = plpy.prepare("""
                INSERT INTO @extschema@.steadytext_daemon_health (
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
    # Update health table to reflect unhealthy status - FIXED: Use schema-qualified table name
    update_plan = plpy.prepare("""
        UPDATE @extschema@.steadytext_daemon_health 
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
    plan = plpy.prepare("SELECT value FROM @extschema@.steadytext_config WHERE key = $1", ["text"])
    
    # Use provided values or defaults from config
    if host is None:
        host_rv = plpy.execute(plan, ["daemon_host"])
        host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"
    
    if port is None:
        port_rv = plpy.execute(plan, ["daemon_port"])
        port = json.loads(port_rv[0]["value"]) if port_rv else 5555
    
    # Update daemon health status - FIXED: Use schema-qualified table name
    update_plan = plpy.prepare("""
        INSERT INTO @extschema@.steadytext_daemon_health (daemon_id, endpoint, status)
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
    
    # Try to get from cache first - FIXED: Use schema-qualified table name
    cache_plan = plpy.prepare("""
        UPDATE @extschema@.steadytext_rerank_cache 
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
plan = plpy.prepare("SELECT value FROM @extschema@.steadytext_config WHERE key = $1", ["text"])

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
        insert_plan = plpy.prepare("""
            INSERT INTO @extschema@.steadytext_rerank_cache (cache_key, score, query_text, document_text)
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
                    access_count = steadytext_rerank_cache.access_count + 1,
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

-- AIDEV-NOTE: Migration completed - all functions now use schema-qualified table references
-- This ensures they work correctly in TimescaleDB continuous aggregates and other contexts
-- where the search path might not include the extension's schema.
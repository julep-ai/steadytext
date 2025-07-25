-- pg_steadytext extension migration from 1.4.0 to 1.4.1
-- Fixes IMMUTABLE function violations by switching to write-once cache with age-based eviction

-- AIDEV-NOTE: This migration changes the cache strategy to support true IMMUTABLE functions
-- Previous versions tried to UPDATE cache statistics on reads and INSERT on cache misses,
-- which violates PostgreSQL's IMMUTABLE contract. 
--
-- In v1.4.1, IMMUTABLE functions:
-- - Only READ from cache (SELECT operations only)
-- - Never write to cache (no INSERT or UPDATE)
-- - Cache population must be done externally or via VOLATILE wrapper functions
--
-- Trade-offs:
-- - Lost: Automatic cache population on first use
-- - Lost: Frecency-based eviction 
-- - Gained: True immutability and PostgreSQL compliance
-- - Gained: Better query optimization potential

-- Update version
CREATE OR REPLACE FUNCTION steadytext_version()
RETURNS text AS $$
BEGIN
    RETURN '1.4.1';
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- AIDEV-SECTION: UPDATE_GENERATION_FUNCTIONS
-- Fix steadytext_generate to use SELECT-only cache reads
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
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
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
    
    # Try to get from cache first - SELECT ONLY (no UPDATE for immutability)
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
    
    # AIDEV-NOTE: Cache writes removed for IMMUTABLE compliance
    # To populate cache, use the VOLATILE wrapper functions or external processes
    
    return result
    
except Exception as e:
    plpy.error(f"Generation failed: {str(e)}")
$$;

-- Fix steadytext_embed to use SELECT-only cache reads
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
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
import json
import hashlib

# Check if initialized
if not GD.get('steadytext_initialized', False):
    plpy.execute("SELECT _steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules
daemon_connector = GD.get('module_daemon_connector')
if not daemon_connector:
    plpy.error("daemon_connector module not loaded")

# Validate input
if not text_input or not text_input.strip():
    plpy.error("Input text cannot be empty")

# Check cache if enabled
if use_cache:
    # Generate cache key consistent with SteadyText format
    # AIDEV-NOTE: Updated to match SteadyText's format from cache_manager.py
    # Embeddings use SHA256 hash of "embed:{text}"
    cache_key_input = f"embed:{text_input}"
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()
    
    # SELECT ONLY - no UPDATE for immutability
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
        
        # AIDEV-NOTE: Cache writes removed for IMMUTABLE compliance
        
        return embedding_list
    else:
        plpy.error("Failed to generate embedding")
        
except Exception as e:
    plpy.error(f"Embedding generation failed: {str(e)}")
$$;

-- Fix steadytext_generate_json to use SELECT-only cache reads
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
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
import json
import hashlib

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    plpy.execute("SELECT _steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules
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

# Validate inputs
if not prompt or not prompt.strip():
    plpy.error("Prompt cannot be empty")

if not schema:
    plpy.error("Schema cannot be empty")

schema_dict = json.loads(schema) if isinstance(schema, str) else schema

if resolved_max_tokens < 1 or resolved_max_tokens > 4096:
    plpy.error("max_tokens must be between 1 and 4096")

# Check cache if enabled
if use_cache:
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

# Create connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Auto-start daemon if configured
auto_start_rv = plpy.execute(plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    connector.start_daemon()

# Generate structured output
try:
    if connector.is_daemon_running():
        result = connector.generate_json(
            prompt=prompt,
            schema=schema_dict,
            max_tokens=resolved_max_tokens,
            seed=seed
        )
    else:
        # Direct generation fallback
        from steadytext import generate_json as steadytext_generate_json
        result = steadytext_generate_json(
            prompt=prompt,
            schema=schema_dict,
            max_tokens=resolved_max_tokens,
            seed=seed
        )
    
    # AIDEV-NOTE: Cache writes removed for IMMUTABLE compliance
    
    return result
    
except Exception as e:
    plpy.error(f"JSON generation failed: {str(e)}")
$$;

-- Fix steadytext_generate_regex to use SELECT-only cache reads
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
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
import json
import hashlib

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    plpy.execute("SELECT _steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules
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

# Validate inputs
if not prompt or not prompt.strip():
    plpy.error("Prompt cannot be empty")

if not pattern or not pattern.strip():
    plpy.error("Pattern cannot be empty")

if resolved_max_tokens < 1 or resolved_max_tokens > 4096:
    plpy.error("max_tokens must be between 1 and 4096")

# Check cache if enabled
if use_cache:
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

# Create connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Auto-start daemon if configured
auto_start_rv = plpy.execute(plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    connector.start_daemon()

# Generate structured output
try:
    if connector.is_daemon_running():
        result = connector.generate_regex(
            prompt=prompt,
            pattern=pattern,
            max_tokens=resolved_max_tokens,
            seed=seed
        )
    else:
        # Direct generation fallback
        from steadytext import generate_regex as steadytext_generate_regex
        result = steadytext_generate_regex(
            prompt=prompt,
            regex=pattern,
            max_tokens=resolved_max_tokens,
            seed=seed
        )
    
    # AIDEV-NOTE: Cache writes removed for IMMUTABLE compliance
    
    return result
    
except Exception as e:
    plpy.error(f"Regex generation failed: {str(e)}")
$$;

-- Fix steadytext_generate_choice to use SELECT-only cache reads
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
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
import json
import hashlib

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    plpy.execute("SELECT _steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get cached modules
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

# Validate inputs
if not prompt or not prompt.strip():
    plpy.error("Prompt cannot be empty")

if not choices or len(choices) == 0:
    plpy.error("Choices cannot be empty")

# Convert PostgreSQL array to Python list
choices_list = list(choices) if choices else []

if resolved_max_tokens < 1 or resolved_max_tokens > 4096:
    plpy.error("max_tokens must be between 1 and 4096")

# Check cache if enabled
if use_cache:
    # Generate cache key including choices
    cache_key_input = f"{prompt}|choice|{json.dumps(sorted(choices_list))}"
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

# Create connector
connector = daemon_connector.SteadyTextConnector(host=host, port=port)

# Auto-start daemon if configured
auto_start_rv = plpy.execute(plan, ["daemon_auto_start"])
auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

if auto_start and not connector.is_daemon_running():
    plpy.notice("Starting SteadyText daemon...")
    connector.start_daemon()

# Generate structured output
try:
    if connector.is_daemon_running():
        result = connector.generate_choice(
            prompt=prompt,
            choices=choices_list,
            max_tokens=resolved_max_tokens,
            seed=seed
        )
    else:
        # Direct generation fallback
        from steadytext import generate_choice as steadytext_generate_choice
        result = steadytext_generate_choice(
            prompt=prompt,
            choices=choices_list,
            max_tokens=resolved_max_tokens,
            seed=seed
        )
    
    # AIDEV-NOTE: Cache writes removed for IMMUTABLE compliance
    
    return result
    
except Exception as e:
    plpy.error(f"Choice generation failed: {str(e)}")
$$;

-- AIDEV-SECTION: UPDATE_EVICTION_FUNCTIONS
-- Replace frecency-based eviction with age-based eviction
CREATE OR REPLACE FUNCTION steadytext_cache_evict_by_age(
    target_entries INT DEFAULT NULL,
    target_size_mb FLOAT DEFAULT NULL,
    batch_size INT DEFAULT 100,
    min_age_hours INT DEFAULT 1
)
RETURNS TABLE(
    evicted_count INT,
    freed_size_mb FLOAT,
    remaining_entries BIGINT,
    remaining_size_mb FLOAT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_evicted_count INT := 0;
    v_freed_size_mb FLOAT := 0;
    v_current_entries BIGINT;
    v_current_size_mb FLOAT;
    v_batch_evicted INT;
    v_batch_freed_mb FLOAT;
BEGIN
    -- AIDEV-NOTE: Simplified eviction using age-based strategy (FIFO)
    -- This replaces the frecency-based eviction to maintain IMMUTABLE functions
    
    -- Get current cache statistics
    SELECT 
        COUNT(*),
        COALESCE(SUM(pg_column_size(response) + pg_column_size(embedding)) / 1024.0 / 1024.0, 0)
    INTO v_current_entries, v_current_size_mb
    FROM steadytext_cache;
    
    -- Set default targets if not provided
    IF target_entries IS NULL THEN
        SELECT value::INT INTO target_entries 
        FROM steadytext_config 
        WHERE key = 'cache_max_entries';
        target_entries := COALESCE(target_entries, 10000);
    END IF;
    
    IF target_size_mb IS NULL THEN
        SELECT value::FLOAT INTO target_size_mb 
        FROM steadytext_config 
        WHERE key = 'cache_max_size_mb';
        target_size_mb := COALESCE(target_size_mb, 1000);
    END IF;
    
    -- Evict in batches until we meet our targets
    WHILE (v_current_entries > target_entries OR v_current_size_mb > target_size_mb) LOOP
        -- Delete oldest entries (FIFO eviction)
        WITH deleted AS (
            DELETE FROM steadytext_cache
            WHERE id IN (
                SELECT id 
                FROM steadytext_cache
                WHERE created_at < NOW() - INTERVAL '1 hour' * min_age_hours
                ORDER BY created_at ASC  -- Oldest first
                LIMIT batch_size
            )
            RETURNING pg_column_size(response) + pg_column_size(embedding) as size_bytes
        )
        SELECT 
            COUNT(*),
            COALESCE(SUM(size_bytes) / 1024.0 / 1024.0, 0)
        INTO v_batch_evicted, v_batch_freed_mb
        FROM deleted;
        
        -- Break if nothing was evicted (all entries are too young)
        IF v_batch_evicted = 0 THEN
            EXIT;
        END IF;
        
        -- Update totals
        v_evicted_count := v_evicted_count + v_batch_evicted;
        v_freed_size_mb := v_freed_size_mb + v_batch_freed_mb;
        v_current_entries := v_current_entries - v_batch_evicted;
        v_current_size_mb := v_current_size_mb - v_batch_freed_mb;
        
        -- Log eviction batch
        INSERT INTO steadytext_audit_log (action, details)
        VALUES (
            'cache_eviction',
            jsonb_build_object(
                'evicted_count', v_batch_evicted,
                'freed_size_mb', v_batch_freed_mb,
                'eviction_type', 'age_based'
            )
        );
    END LOOP;
    
    -- Return results
    RETURN QUERY
    SELECT 
        v_evicted_count,
        v_freed_size_mb,
        v_current_entries,
        v_current_size_mb;
END;
$$;

-- Update the scheduled eviction function to use age-based eviction
CREATE OR REPLACE FUNCTION steadytext_cache_scheduled_eviction()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    v_enabled BOOLEAN;
    v_result RECORD;
BEGIN
    -- Check if eviction is enabled
    SELECT value::BOOLEAN INTO v_enabled 
    FROM steadytext_config 
    WHERE key = 'cache_eviction_enabled';
    
    IF NOT COALESCE(v_enabled, TRUE) THEN
        RETURN;
    END IF;
    
    -- Perform age-based eviction
    SELECT * INTO v_result
    FROM steadytext_cache_evict_by_age();
    
    -- Log results if anything was evicted
    IF v_result.evicted_count > 0 THEN
        RAISE NOTICE 'Cache eviction completed: % entries evicted, % MB freed',
            v_result.evicted_count, v_result.freed_size_mb;
    END IF;
END;
$$;

-- Update cache preview to show age instead of frecency
CREATE OR REPLACE FUNCTION steadytext_cache_preview_eviction(
    preview_count INT DEFAULT 10
)
RETURNS TABLE(
    cache_key TEXT,
    prompt TEXT,
    access_count INT,
    last_accessed TIMESTAMPTZ,
    created_at TIMESTAMPTZ,
    age_days FLOAT,
    size_bytes BIGINT
)
LANGUAGE sql
STABLE PARALLEL SAFE
AS $$
    SELECT 
        cache_key,
        LEFT(prompt, 50) || CASE WHEN LENGTH(prompt) > 50 THEN '...' ELSE '' END as prompt,
        access_count,
        last_accessed,
        created_at,
        EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 as age_days,
        pg_column_size(response) + pg_column_size(embedding) as size_bytes
    FROM steadytext_cache
    WHERE created_at < NOW() - INTERVAL '1 hour'  -- Respect min age
    ORDER BY created_at ASC  -- Oldest first
    LIMIT preview_count;
$$;

-- Update cache analysis to reflect age-based strategy
CREATE OR REPLACE FUNCTION steadytext_cache_analyze_usage()
RETURNS TABLE(
    age_bucket TEXT,
    entry_count BIGINT,
    avg_access_count FLOAT,
    total_size_mb FLOAT,
    percentage_of_cache FLOAT
)
LANGUAGE sql
STABLE PARALLEL SAFE
AS $$
    WITH cache_buckets AS (
        SELECT 
            CASE 
                WHEN created_at > NOW() - INTERVAL '1 hour' THEN '< 1 hour'
                WHEN created_at > NOW() - INTERVAL '1 day' THEN '1 hour - 1 day'
                WHEN created_at > NOW() - INTERVAL '7 days' THEN '1 day - 7 days'
                WHEN created_at > NOW() - INTERVAL '30 days' THEN '7 days - 30 days'
                ELSE '> 30 days'
            END as age_bucket,
            COUNT(*) as entry_count,
            AVG(access_count) as avg_access_count,
            SUM(pg_column_size(response) + pg_column_size(embedding)) / 1024.0 / 1024.0 as total_size_mb
        FROM steadytext_cache
        GROUP BY 1
    ),
    total AS (
        SELECT COUNT(*) as total_entries
        FROM steadytext_cache
    )
    SELECT 
        cb.age_bucket,
        cb.entry_count,
        cb.avg_access_count,
        cb.total_size_mb,
        (cb.entry_count::FLOAT / NULLIF(t.total_entries, 0) * 100)::FLOAT as percentage_of_cache
    FROM cache_buckets cb
    CROSS JOIN total t
    ORDER BY 
        CASE cb.age_bucket
            WHEN '< 1 hour' THEN 1
            WHEN '1 hour - 1 day' THEN 2
            WHEN '1 day - 7 days' THEN 3
            WHEN '7 days - 30 days' THEN 4
            ELSE 5
        END;
$$;

-- AIDEV-NOTE: The view steadytext_cache_with_frecency is kept for compatibility
-- but now shows age_score instead of frecency_score
CREATE OR REPLACE VIEW steadytext_cache_with_frecency AS
SELECT *,
    -- Simple age-based score for compatibility (higher = older = more likely to evict)
    EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 AS frecency_score
FROM steadytext_cache;

COMMENT ON VIEW steadytext_cache_with_frecency IS 
'Compatibility view - frecency_score now represents age in days (v1.4.1+)';

-- Add comment explaining the cache strategy change
COMMENT ON TABLE steadytext_cache IS 
'Write-once cache for SteadyText results. Uses age-based eviction (FIFO) as of v1.4.1 to maintain IMMUTABLE function guarantees.';

-- AIDEV-SECTION: VOLATILE_WRAPPER_FUNCTIONS
-- These functions provide cache-writing capability for users who need it
-- They wrap the IMMUTABLE functions and handle cache population

CREATE OR REPLACE FUNCTION steadytext_generate_cached(
    prompt TEXT,
    max_tokens INT DEFAULT NULL,
    seed INT DEFAULT 42
)
RETURNS TEXT
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    v_result TEXT;
    v_cache_key TEXT;
    v_generation_params JSONB;
BEGIN
    -- AIDEV-NOTE: This VOLATILE wrapper allows cache writes
    -- Use this when you need automatic cache population
    
    -- Generate result using IMMUTABLE function
    v_result := steadytext_generate(prompt, max_tokens, true, seed);
    
    -- If result was generated (not from cache), store it
    IF v_result IS NOT NULL THEN
        -- Generate cache key
        v_cache_key := prompt;
        
        -- Check if already cached
        PERFORM 1 FROM steadytext_cache WHERE cache_key = v_cache_key;
        
        IF NOT FOUND THEN
            -- Resolve max_tokens default
            IF max_tokens IS NULL THEN
                SELECT value::INT INTO max_tokens 
                FROM steadytext_config 
                WHERE key = 'default_max_tokens';
                max_tokens := COALESCE(max_tokens, 512);
            END IF;
            
            v_generation_params := jsonb_build_object(
                'max_tokens', max_tokens,
                'seed', seed
            );
            
            -- Store in cache
            INSERT INTO steadytext_cache 
            (cache_key, prompt, response, model_name, seed, generation_params)
            VALUES (v_cache_key, prompt, v_result, 'steadytext-default', seed, v_generation_params)
            ON CONFLICT (cache_key) DO NOTHING;
        END IF;
    END IF;
    
    RETURN v_result;
END;
$$;

CREATE OR REPLACE FUNCTION steadytext_embed_cached(
    text_input TEXT,
    seed INT DEFAULT 42
)
RETURNS vector(1024)
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    v_result vector(1024);
    v_cache_key TEXT;
    v_cache_key_input TEXT;
BEGIN
    -- AIDEV-NOTE: This VOLATILE wrapper allows cache writes
    
    -- Generate result using IMMUTABLE function
    v_result := steadytext_embed(text_input, true, seed);
    
    -- If result was generated, store it
    IF v_result IS NOT NULL THEN
        -- Generate cache key
        v_cache_key_input := 'embed:' || text_input;
        v_cache_key := encode(digest(v_cache_key_input, 'sha256'), 'hex');
        
        -- Check if already cached
        PERFORM 1 FROM steadytext_cache WHERE cache_key = v_cache_key;
        
        IF NOT FOUND THEN
            -- Store in cache
            INSERT INTO steadytext_cache 
            (cache_key, prompt, embedding, model_name, seed)
            VALUES (v_cache_key, text_input, v_result, 'steadytext-embedding', seed)
            ON CONFLICT (cache_key) DO NOTHING;
        END IF;
    END IF;
    
    RETURN v_result;
END;
$$;

-- Add comments explaining the wrapper functions
COMMENT ON FUNCTION steadytext_generate_cached IS 
'VOLATILE wrapper for steadytext_generate that handles cache population. Use when automatic caching is needed.';

COMMENT ON FUNCTION steadytext_embed_cached IS 
'VOLATILE wrapper for steadytext_embed that handles cache population. Use when automatic caching is needed.';

-- Add note about cache population strategies
COMMENT ON FUNCTION steadytext_generate IS 
'IMMUTABLE function for text generation. Only reads from cache, never writes. For automatic cache population, use steadytext_generate_cached.';

COMMENT ON FUNCTION steadytext_embed IS 
'IMMUTABLE function for embeddings. Only reads from cache, never writes. For automatic cache population, use steadytext_embed_cached.';
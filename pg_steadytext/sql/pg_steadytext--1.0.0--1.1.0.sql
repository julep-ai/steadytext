-- pg_steadytext--1.0.0--1.1.0.sql
-- Adds async counterparts for all generation and embedding functions
-- AIDEV-NOTE: This upgrade adds async support for structured generation and embeddings
-- AIDEV-NOTE: Implements non-blocking AI operations via queue-based processing
-- AIDEV-NOTE: Supports all SteadyText generation modes: text, JSON, regex, choice

-- Add new request types to queue constraint
ALTER TABLE steadytext_queue 
DROP CONSTRAINT steadytext_queue_request_type_check;

ALTER TABLE steadytext_queue 
ADD CONSTRAINT steadytext_queue_request_type_check 
CHECK (request_type IN ('generate', 'embed', 'batch_embed', 'generate_json', 'generate_regex', 'generate_choice'));

-- AIDEV-SECTION: ASYNC_EMBEDDING_FUNCTIONS

-- Async embedding function
CREATE OR REPLACE FUNCTION steadytext_embed_async(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT TRUE
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    request_id UUID;
BEGIN
    -- AIDEV-NOTE: Queue an embedding request for async processing
    
    -- Validate input
    IF text_input IS NULL OR trim(text_input) = '' THEN
        RAISE EXCEPTION 'Text input cannot be empty';
    END IF;
    
    -- Insert into queue
    INSERT INTO steadytext_queue (
        request_type,
        prompt,
        params
    ) VALUES (
        'embed',
        text_input,
        jsonb_build_object(
            'use_cache', use_cache
        )
    )
    RETURNING request_id INTO request_id;
    
    -- Notify workers
    PERFORM pg_notify('steadytext_queue', request_id::text);
    
    RETURN request_id;
END;
$$;

-- AIDEV-SECTION: ASYNC_STRUCTURED_GENERATION_FUNCTIONS

-- Async JSON generation with schema
CREATE OR REPLACE FUNCTION steadytext_generate_json_async(
    prompt TEXT,
    schema JSONB,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    request_id UUID;
    resolved_max_tokens INT;
BEGIN
    -- AIDEV-NOTE: Queue a JSON generation request for async processing
    
    -- Validate inputs
    IF prompt IS NULL OR trim(prompt) = '' THEN
        RAISE EXCEPTION 'Prompt cannot be empty';
    END IF;
    
    IF schema IS NULL THEN
        RAISE EXCEPTION 'Schema cannot be null';
    END IF;
    
    -- Resolve max_tokens from config if not provided
    IF max_tokens IS NULL THEN
        SELECT value::int INTO resolved_max_tokens 
        FROM steadytext_config 
        WHERE key = 'default_max_tokens';
        
        IF resolved_max_tokens IS NULL THEN
            resolved_max_tokens := 512;
        END IF;
    ELSE
        resolved_max_tokens := max_tokens;
    END IF;
    
    IF resolved_max_tokens < 1 OR resolved_max_tokens > 4096 THEN
        RAISE EXCEPTION 'max_tokens must be between 1 and 4096';
    END IF;
    
    -- Insert into queue
    INSERT INTO steadytext_queue (
        request_type,
        prompt,
        params
    ) VALUES (
        'generate_json',
        prompt,
        jsonb_build_object(
            'schema', schema,
            'max_tokens', resolved_max_tokens,
            'use_cache', use_cache,
            'seed', seed
        )
    )
    RETURNING request_id INTO request_id;
    
    -- Notify workers
    PERFORM pg_notify('steadytext_queue', request_id::text);
    
    RETURN request_id;
END;
$$;

-- Async regex-constrained generation
CREATE OR REPLACE FUNCTION steadytext_generate_regex_async(
    prompt TEXT,
    pattern TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    request_id UUID;
    resolved_max_tokens INT;
BEGIN
    -- AIDEV-NOTE: Queue a regex-constrained generation request
    
    -- Validate inputs
    IF prompt IS NULL OR trim(prompt) = '' THEN
        RAISE EXCEPTION 'Prompt cannot be empty';
    END IF;
    
    IF pattern IS NULL OR trim(pattern) = '' THEN
        RAISE EXCEPTION 'Pattern cannot be empty';
    END IF;
    
    -- Resolve max_tokens
    IF max_tokens IS NULL THEN
        SELECT value::int INTO resolved_max_tokens 
        FROM steadytext_config 
        WHERE key = 'default_max_tokens';
        
        IF resolved_max_tokens IS NULL THEN
            resolved_max_tokens := 512;
        END IF;
    ELSE
        resolved_max_tokens := max_tokens;
    END IF;
    
    IF resolved_max_tokens < 1 OR resolved_max_tokens > 4096 THEN
        RAISE EXCEPTION 'max_tokens must be between 1 and 4096';
    END IF;
    
    -- Insert into queue
    INSERT INTO steadytext_queue (
        request_type,
        prompt,
        params
    ) VALUES (
        'generate_regex',
        prompt,
        jsonb_build_object(
            'pattern', pattern,
            'max_tokens', resolved_max_tokens,
            'use_cache', use_cache,
            'seed', seed
        )
    )
    RETURNING request_id INTO request_id;
    
    -- Notify workers
    PERFORM pg_notify('steadytext_queue', request_id::text);
    
    RETURN request_id;
END;
$$;

-- Async choice-constrained generation
CREATE OR REPLACE FUNCTION steadytext_generate_choice_async(
    prompt TEXT,
    choices TEXT[],
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    request_id UUID;
BEGIN
    -- AIDEV-NOTE: Queue a choice-constrained generation request
    
    -- Validate inputs
    IF prompt IS NULL OR trim(prompt) = '' THEN
        RAISE EXCEPTION 'Prompt cannot be empty';
    END IF;
    
    IF choices IS NULL OR array_length(choices, 1) < 2 THEN
        RAISE EXCEPTION 'Must provide at least 2 choices';
    END IF;
    
    -- Insert into queue
    INSERT INTO steadytext_queue (
        request_type,
        prompt,
        params
    ) VALUES (
        'generate_choice',
        prompt,
        jsonb_build_object(
            'choices', choices,
            'use_cache', use_cache,
            'seed', seed
        )
    )
    RETURNING request_id INTO request_id;
    
    -- Notify workers
    PERFORM pg_notify('steadytext_queue', request_id::text);
    
    RETURN request_id;
END;
$$;

-- AIDEV-SECTION: ASYNC_RESULT_RETRIEVAL

-- Enhanced async check function that handles all result types
CREATE OR REPLACE FUNCTION steadytext_check_async(
    request_id UUID
)
RETURNS TABLE(
    status TEXT,
    result TEXT,
    results TEXT[],
    embedding vector(1024),
    embeddings vector(1024)[],
    error TEXT,
    created_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    processing_time_ms INT,
    request_type TEXT
)
LANGUAGE sql
STABLE PARALLEL SAFE LEAKPROOF
AS $$
    SELECT 
        status,
        result,
        results,
        embedding,
        embeddings,
        error,
        created_at,
        completed_at,
        processing_time_ms,
        request_type
    FROM steadytext_queue
    WHERE steadytext_queue.request_id = steadytext_check_async.request_id;
$$;

-- Convenience function to get result directly (blocks until ready or timeout)
CREATE OR REPLACE FUNCTION steadytext_get_async_result(
    request_id UUID,
    timeout_seconds INT DEFAULT 30
)
RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    start_time TIMESTAMPTZ := clock_timestamp();
    check_result RECORD;
BEGIN
    -- AIDEV-NOTE: Polls for async result with timeout
    
    LOOP
        -- Check status
        SELECT * INTO check_result
        FROM steadytext_check_async(request_id);
        
        -- Check if request exists
        IF check_result IS NULL THEN
            RAISE EXCEPTION 'Request ID % not found', request_id;
        END IF;
        
        -- Return result if completed
        IF check_result.status = 'completed' THEN
            RETURN check_result.result;
        END IF;
        
        -- Return error if failed
        IF check_result.status = 'failed' THEN
            RAISE EXCEPTION 'Async request failed: %', check_result.error;
        END IF;
        
        -- Check timeout
        IF clock_timestamp() - start_time > interval '1 second' * timeout_seconds THEN
            RAISE EXCEPTION 'Timeout waiting for async result (% seconds)', timeout_seconds;
        END IF;
        
        -- Wait a bit before checking again
        PERFORM pg_sleep(0.1);
    END LOOP;
END;
$$;

-- Function to cancel an async request
CREATE OR REPLACE FUNCTION steadytext_cancel_async(
    request_id UUID
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    rows_updated INT;
BEGIN
    -- AIDEV-NOTE: Cancel a pending async request
    
    UPDATE steadytext_queue
    SET status = 'cancelled',
        completed_at = NOW()
    WHERE steadytext_queue.request_id = steadytext_cancel_async.request_id
    AND status IN ('pending', 'processing');
    
    GET DIAGNOSTICS rows_updated = ROW_COUNT;
    
    RETURN rows_updated > 0;
END;
$$;

-- AIDEV-SECTION: BATCH_ASYNC_OPERATIONS

-- Batch generate async
CREATE OR REPLACE FUNCTION steadytext_generate_batch_async(
    prompts TEXT[],
    max_tokens INT DEFAULT 512
)
RETURNS UUID[]
LANGUAGE plpgsql
AS $$
DECLARE
    request_ids UUID[];
    prompt TEXT;
BEGIN
    -- AIDEV-NOTE: Create multiple async generation requests
    
    -- Validate input
    IF prompts IS NULL OR array_length(prompts, 1) = 0 THEN
        RAISE EXCEPTION 'Prompts array cannot be empty';
    END IF;
    
    IF array_length(prompts, 1) > 100 THEN
        RAISE EXCEPTION 'Batch size cannot exceed 100 prompts';
    END IF;
    
    -- Create requests for each prompt
    FOREACH prompt IN ARRAY prompts
    LOOP
        request_ids := array_append(
            request_ids, 
            steadytext_generate_async(prompt, max_tokens)
        );
    END LOOP;
    
    RETURN request_ids;
END;
$$;

-- Batch embed async
CREATE OR REPLACE FUNCTION steadytext_embed_batch_async(
    texts TEXT[],
    use_cache BOOLEAN DEFAULT TRUE
)
RETURNS UUID[]
LANGUAGE plpgsql
AS $$
DECLARE
    request_ids UUID[];
    text_item TEXT;
BEGIN
    -- AIDEV-NOTE: Create multiple async embedding requests
    
    -- Validate input
    IF texts IS NULL OR array_length(texts, 1) = 0 THEN
        RAISE EXCEPTION 'Texts array cannot be empty';
    END IF;
    
    IF array_length(texts, 1) > 100 THEN
        RAISE EXCEPTION 'Batch size cannot exceed 100 texts';
    END IF;
    
    -- Create requests for each text
    FOREACH text_item IN ARRAY texts
    LOOP
        request_ids := array_append(
            request_ids, 
            steadytext_embed_async(text_item, use_cache)
        );
    END LOOP;
    
    RETURN request_ids;
END;
$$;

-- Function to check multiple async requests at once
CREATE OR REPLACE FUNCTION steadytext_check_async_batch(
    request_ids UUID[]
)
RETURNS TABLE(
    request_id UUID,
    status TEXT,
    result TEXT,
    error TEXT,
    completed_at TIMESTAMPTZ
)
LANGUAGE sql
STABLE PARALLEL SAFE
AS $$
    SELECT 
        request_id,
        status,
        result,
        error,
        completed_at
    FROM steadytext_queue
    WHERE request_id = ANY(request_ids)
    ORDER BY array_position(request_ids, request_id);
$$;

-- AIDEV-NOTE: Version 1.1.0 adds comprehensive async support for all SteadyText functions
-- including structured generation (JSON, regex, choice) and batch operations.
-- The worker.py needs to be updated to handle the new request types.

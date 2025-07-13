-- pg_steadytext extension migration from 1.2.0 to 1.3.0
-- Adds reranking functions for query-document relevance scoring

-- AIDEV-NOTE: This migration adds reranking functions using Qwen3-Reranker model
-- for scoring query-document pairs and sorting by relevance

-- Update version
CREATE OR REPLACE FUNCTION steadytext_version()
RETURNS text AS $$
BEGIN
    RETURN '1.3.0';
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- Basic rerank function returning documents with scores
CREATE OR REPLACE FUNCTION steadytext_rerank(
    query text,
    documents text[],
    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    return_scores boolean DEFAULT true,
    seed integer DEFAULT 42
) RETURNS TABLE(document text, score float)
AS $$
    import json
    import logging
    from typing import List, Tuple
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not query:
        plpy.error("Query cannot be empty")
        
    if not documents or len(documents) == 0:
        return []
    
    # Import daemon connector
    try:
        from daemon_connector import SteadyTextConnector
        connector = SteadyTextConnector()
    except Exception as e:
        logger.error(f"Failed to initialize SteadyText connector: {e}")
        # Return empty result on error
        return []
    
    try:
        # Call rerank with scores always enabled for PostgreSQL
        results = connector.rerank(
            query=query,
            documents=list(documents),  # Convert from PostgreSQL array
            task=task,
            return_scores=True,  # Always get scores for PostgreSQL
            seed=seed
        )
        
        # Return results as tuples
        return results
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        # Return empty result on error
        return []
$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

-- Rerank function returning only documents (no scores)
CREATE OR REPLACE FUNCTION steadytext_rerank_docs_only(
    query text,
    documents text[],
    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    seed integer DEFAULT 42
) RETURNS TABLE(document text)
AS $$
    # Call the main rerank function and extract just documents
    results = plpy.execute(
        "SELECT document FROM steadytext_rerank($1, $2, $3, true, $4)",
        [query, documents, task, seed]
    )
    
    return [{"document": row["document"]} for row in results]
$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

-- Rerank function with top-k filtering
CREATE OR REPLACE FUNCTION steadytext_rerank_top_k(
    query text,
    documents text[],
    top_k integer,
    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    return_scores boolean DEFAULT true,
    seed integer DEFAULT 42
) RETURNS TABLE(document text, score float)
AS $$
    # Validate top_k
    if top_k <= 0:
        plpy.error("top_k must be positive")
    
    # Call the main rerank function
    results = plpy.execute(
        "SELECT document, score FROM steadytext_rerank($1, $2, $3, true, $4) LIMIT $5",
        [query, documents, task, seed, top_k]
    )
    
    if return_scores:
        return results
    else:
        # Return without scores
        return [{"document": row["document"], "score": None} for row in results]
$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

-- Async rerank function
CREATE OR REPLACE FUNCTION steadytext_rerank_async(
    query text,
    documents text[],
    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    return_scores boolean DEFAULT true,
    seed integer DEFAULT 42
) RETURNS uuid
AS $$
    import uuid
    import json
    import logging
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Prepare parameters
    params = {
        'query': query,
        'documents': documents,
        'task': task,
        'return_scores': return_scores,
        'seed': seed
    }
    
    # Insert into queue
    plpy.execute("""
        INSERT INTO steadytext_queue 
        (request_id, function_name, parameters, status, created_at, priority)
        VALUES ($1, 'rerank', $2::jsonb, 'pending', CURRENT_TIMESTAMP, 5)
    """, [request_id, json.dumps(params)])
    
    # Send notification to worker
    plpy.execute("NOTIFY steadytext_queue_notify")
    
    return request_id
$$ LANGUAGE plpython3u VOLATILE;

-- Batch rerank function for multiple queries
CREATE OR REPLACE FUNCTION steadytext_rerank_batch(
    queries text[],
    documents text[],
    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    return_scores boolean DEFAULT true,
    seed integer DEFAULT 42
) RETURNS TABLE(query_index integer, document text, score float)
AS $$
    import json
    import logging
    from typing import List, Tuple
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not queries or len(queries) == 0:
        plpy.error("Queries cannot be empty")
        
    if not documents or len(documents) == 0:
        return []
    
    # Import daemon connector
    try:
        from daemon_connector import SteadyTextConnector
        connector = SteadyTextConnector()
    except Exception as e:
        logger.error(f"Failed to initialize SteadyText connector: {e}")
        return []
    
    all_results = []
    
    # Process each query
    for idx, query in enumerate(queries):
        try:
            # Call rerank for this query
            results = connector.rerank(
                query=query,
                documents=list(documents),
                task=task,
                return_scores=True,
                seed=seed
            )
            
            # Add query index to results
            for doc, score in results:
                all_results.append({
                    "query_index": idx,
                    "document": doc,
                    "score": score
                })
                
        except Exception as e:
            logger.error(f"Reranking failed for query {idx}: {e}")
            # Continue with next query
            continue
    
    return all_results
$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

-- Batch async rerank function
CREATE OR REPLACE FUNCTION steadytext_rerank_batch_async(
    queries text[],
    documents text[],
    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    return_scores boolean DEFAULT true,
    seed integer DEFAULT 42
) RETURNS uuid[]
AS $$
    import uuid
    import json
    
    request_ids = []
    
    # Create separate async request for each query
    for query in queries:
        request_id = str(uuid.uuid4())
        request_ids.append(request_id)
        
        params = {
            'query': query,
            'documents': documents,
            'task': task,
            'return_scores': return_scores,
            'seed': seed
        }
        
        plpy.execute("""
            INSERT INTO steadytext_queue 
            (request_id, function_name, parameters, status, created_at, priority)
            VALUES ($1, 'rerank', $2::jsonb, 'pending', CURRENT_TIMESTAMP, 5)
        """, [request_id, json.dumps(params)])
    
    # Send notification to worker
    plpy.execute("NOTIFY steadytext_queue_notify")
    
    return request_ids
$$ LANGUAGE plpython3u VOLATILE;

-- Add rerank support to worker processing
-- This updates the worker to handle 'rerank' function_name in the queue
-- AIDEV-NOTE: The actual worker.py file handles this, but we document it here
COMMENT ON FUNCTION steadytext_rerank IS 'Rerank documents by relevance to a query using AI model';
COMMENT ON FUNCTION steadytext_rerank_docs_only IS 'Rerank documents returning only sorted documents without scores';
COMMENT ON FUNCTION steadytext_rerank_top_k IS 'Rerank documents and return only top K results';
COMMENT ON FUNCTION steadytext_rerank_async IS 'Asynchronously rerank documents (returns request UUID)';
COMMENT ON FUNCTION steadytext_rerank_batch IS 'Rerank documents for multiple queries in batch';
COMMENT ON FUNCTION steadytext_rerank_batch_async IS 'Asynchronously rerank documents for multiple queries';
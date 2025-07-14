# PostgreSQL Extension: Document Reranking

This guide covers the document reranking functionality in the SteadyText PostgreSQL extension.

## Overview

Document reranking allows you to reorder search results based on their relevance to a query using the Qwen3-Reranker-4B model. This is particularly useful for improving search quality in retrieval-augmented generation (RAG) systems.

## Core Functions

### Basic Reranking

```sql
-- Rerank a list of documents based on relevance to a query
SELECT * FROM steadytext_rerank(
    'machine learning applications',
    ARRAY['Introduction to ML', 'Python cookbook', 'Deep learning guide']
);

-- Returns:
-- position | document | score
-- 1        | Deep learning guide      | 0.89
-- 2        | Introduction to ML       | 0.76
-- 3        | Python cookbook          | 0.32
```

### Batch Reranking

```sql
-- Rerank documents for multiple queries
SELECT * FROM steadytext_rerank_batch(
    ARRAY['AI ethics', 'Python programming'],
    ARRAY[
        ARRAY['AI safety paper', 'Python tutorial'],
        ARRAY['Django guide', 'Ethics in technology']
    ]
);
```

### Async Reranking

```sql
-- Start async reranking job
SELECT request_id FROM steadytext_rerank_async(
    'customer support query',
    ARRAY(SELECT content FROM support_docs)
);

-- Check results
SELECT * FROM steadytext_check_async('uuid-here');
```

## Real-World Examples

### Search Result Improvement

```sql
-- Improve PostgreSQL full-text search results
WITH search_results AS (
    SELECT 
        doc_id,
        title,
        content,
        ts_rank(search_vector, query) as pg_score
    FROM documents,
         plainto_tsquery('english', 'machine learning') query
    WHERE search_vector @@ query
    ORDER BY pg_score DESC
    LIMIT 20
)
SELECT 
    r.doc_id,
    r.title,
    r.pg_score,
    rerank.score as ai_score,
    rerank.position
FROM search_results r
CROSS JOIN LATERAL (
    SELECT * FROM steadytext_rerank(
        'machine learning',
        ARRAY[r.content]
    )
) rerank
ORDER BY rerank.score DESC
LIMIT 10;
```

### Multi-Stage Retrieval

```sql
-- Stage 1: Fast vector search
WITH vector_candidates AS (
    SELECT doc_id, content, embedding
    FROM documents
    ORDER BY embedding <-> steadytext_embed('user query')
    LIMIT 100
),
-- Stage 2: Rerank top candidates
reranked AS (
    SELECT 
        v.*,
        rerank.score,
        rerank.position
    FROM vector_candidates v
    CROSS JOIN LATERAL (
        SELECT * FROM steadytext_rerank(
            'user query',
            ARRAY(SELECT content FROM vector_candidates)
        )
    ) rerank
    WHERE rerank.document = v.content
)
SELECT * FROM reranked
ORDER BY score DESC
LIMIT 10;
```

### Customer Support Optimization

```sql
-- Find most relevant support articles for a ticket
CREATE OR REPLACE FUNCTION find_relevant_articles(
    ticket_text TEXT,
    limit_count INT DEFAULT 5
) RETURNS TABLE(
    article_id INT,
    title TEXT,
    relevance_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH candidates AS (
        -- Get initial candidates using FTS
        SELECT 
            a.article_id,
            a.title,
            a.content
        FROM support_articles a
        WHERE to_tsvector('english', a.content) @@ 
              plainto_tsquery('english', ticket_text)
        LIMIT 50
    )
    SELECT 
        c.article_id,
        c.title,
        r.score as relevance_score
    FROM candidates c
    CROSS JOIN LATERAL (
        SELECT score 
        FROM steadytext_rerank(ticket_text, ARRAY[c.content])
    ) r
    ORDER BY r.score DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;
```

## Advanced Techniques

### Custom Task Descriptions

```sql
-- Use custom task description for domain-specific reranking
SELECT * FROM steadytext_rerank(
    'patient symptoms: headache, fever, fatigue',
    ARRAY[
        'Common cold treatment guide',
        'Migraine management',
        'COVID-19 symptoms'
    ],
    'Given a patient''s symptoms, rank medical articles by relevance for diagnosis'
);
```

### Combining with Embeddings

```sql
-- Hybrid scoring: embeddings + reranking
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    weight_embedding FLOAT DEFAULT 0.3,
    weight_rerank FLOAT DEFAULT 0.7
) RETURNS TABLE(
    doc_id INT,
    final_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH embedding_scores AS (
        SELECT 
            doc_id,
            1 - (embedding <-> steadytext_embed(query_text)) as embed_score,
            content
        FROM documents
        ORDER BY embed_score DESC
        LIMIT 100
    ),
    rerank_scores AS (
        SELECT 
            e.doc_id,
            e.embed_score,
            r.score as rerank_score
        FROM embedding_scores e
        CROSS JOIN LATERAL (
            SELECT score
            FROM steadytext_rerank(query_text, ARRAY[e.content])
        ) r
    )
    SELECT 
        doc_id,
        (embed_score * weight_embedding + 
         rerank_score * weight_rerank) as final_score
    FROM rerank_scores
    ORDER BY final_score DESC;
END;
$$ LANGUAGE plpgsql;
```

### Performance Optimization

```sql
-- Materialized view for frequently reranked content
CREATE MATERIALIZED VIEW popular_queries_reranked AS
WITH popular_queries AS (
    SELECT query, COUNT(*) as frequency
    FROM search_logs
    WHERE created_at > NOW() - INTERVAL '7 days'
    GROUP BY query
    ORDER BY frequency DESC
    LIMIT 100
)
SELECT 
    q.query,
    d.doc_id,
    r.score,
    r.position
FROM popular_queries q
CROSS JOIN documents d
CROSS JOIN LATERAL (
    SELECT score, position
    FROM steadytext_rerank(q.query, ARRAY[d.content])
) r
WHERE r.score > 0.5;

-- Refresh periodically
CREATE OR REPLACE FUNCTION refresh_reranked_cache()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY popular_queries_reranked;
END;
$$ LANGUAGE plpgsql;
```

## Configuration and Tuning

### Cache Configuration

```sql
-- Check reranking cache statistics
SELECT * FROM steadytext_cache_stats()
WHERE cache_type = 'reranking';

-- Clear reranking cache if needed
SELECT steadytext_clear_cache('reranking');
```

### Performance Considerations

1. **Batch Size**: Rerank in batches of 10-50 documents for optimal performance
2. **Caching**: Frequently reranked queries are cached automatically
3. **Async Operations**: Use async functions for large document sets
4. **Indexing**: Ensure proper indexes on columns used in WHERE clauses

### Error Handling

```sql
-- Safe reranking with error handling
CREATE OR REPLACE FUNCTION safe_rerank(
    query_text TEXT,
    documents TEXT[]
) RETURNS TABLE(
    position INT,
    document TEXT,
    score FLOAT,
    error TEXT
) AS $$
BEGIN
    BEGIN
        RETURN QUERY
        SELECT r.* 
        FROM steadytext_rerank(query_text, documents) r;
    EXCEPTION
        WHEN OTHERS THEN
            RETURN QUERY
            SELECT 
                generate_series(1, array_length(documents, 1)),
                unnest(documents),
                0.0::FLOAT,
                SQLERRM;
    END;
END;
$$ LANGUAGE plpgsql;
```

## Best Practices

### 1. Two-Stage Retrieval

Always use a fast first-stage retrieval (FTS, vector search) before reranking:

```sql
-- Good: Narrow down candidates first
WITH candidates AS (
    SELECT * FROM documents 
    WHERE search_vector @@ query 
    LIMIT 100
)
SELECT * FROM steadytext_rerank(
    'query', 
    ARRAY(SELECT content FROM candidates)
);

-- Bad: Reranking entire table
SELECT * FROM steadytext_rerank(
    'query',
    ARRAY(SELECT content FROM documents)  -- Too many!
);
```

### 2. Score Thresholds

Filter results by relevance score:

```sql
-- Only return highly relevant results
SELECT * FROM steadytext_rerank(query, docs)
WHERE score > 0.7;
```

### 3. Monitoring

Track reranking performance:

```sql
-- Create reranking metrics table
CREATE TABLE reranking_metrics (
    query_id UUID PRIMARY KEY,
    query_text TEXT,
    num_documents INT,
    execution_time_ms INT,
    avg_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Log reranking operations
CREATE OR REPLACE FUNCTION log_reranking(
    query_text TEXT,
    num_docs INT,
    exec_time_ms INT,
    avg_score FLOAT
) RETURNS void AS $$
INSERT INTO reranking_metrics 
    (query_id, query_text, num_documents, execution_time_ms, avg_score)
VALUES 
    (gen_random_uuid(), query_text, num_docs, exec_time_ms, avg_score);
$$ LANGUAGE sql;
```

## Integration Examples

### With pgvector

```sql
-- Combine with pgvector for hybrid search
CREATE OR REPLACE FUNCTION vector_rerank_search(
    query TEXT,
    limit_results INT DEFAULT 10
) RETURNS TABLE(
    id INT,
    content TEXT,
    vector_similarity FLOAT,
    rerank_score FLOAT,
    combined_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_search AS (
        SELECT 
            doc_id as id,
            content,
            1 - (embedding <-> steadytext_embed(query)) as similarity
        FROM documents
        ORDER BY similarity DESC
        LIMIT limit_results * 5  -- Get more candidates
    ),
    reranked AS (
        SELECT 
            v.*,
            r.score as rerank_score
        FROM vector_search v
        CROSS JOIN LATERAL (
            SELECT score 
            FROM steadytext_rerank(query, ARRAY[v.content])
        ) r
    )
    SELECT 
        id,
        content,
        similarity as vector_similarity,
        rerank_score,
        (similarity * 0.3 + rerank_score * 0.7) as combined_score
    FROM reranked
    ORDER BY combined_score DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;
```

### With Full-Text Search

```sql
-- Enhance PostgreSQL FTS with reranking
CREATE OR REPLACE FUNCTION fts_with_rerank(
    search_query TEXT,
    limit_results INT DEFAULT 10
) RETURNS TABLE(
    doc_id INT,
    headline TEXT,
    fts_rank FLOAT,
    ai_rank FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH fts_results AS (
        SELECT 
            d.doc_id,
            ts_headline('english', d.content, query) as headline,
            ts_rank(d.search_vector, query) as rank,
            d.content
        FROM documents d,
             plainto_tsquery('english', search_query) query
        WHERE d.search_vector @@ query
        ORDER BY rank DESC
        LIMIT limit_results * 3
    ),
    reranked AS (
        SELECT 
            f.*,
            r.score as ai_score
        FROM fts_results f
        CROSS JOIN LATERAL (
            SELECT score 
            FROM steadytext_rerank(search_query, ARRAY[f.content])
        ) r
    )
    SELECT 
        doc_id,
        headline,
        rank as fts_rank,
        ai_score as ai_rank
    FROM reranked
    ORDER BY ai_score DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;
```

## Related Documentation

- [PostgreSQL Extension Overview](postgresql-extension.md)
- [AI Integration Features](postgresql-extension-ai.md)
- [Async Operations](postgresql-extension-async.md)
- [Troubleshooting Guide](postgresql-extension-troubleshooting.md)
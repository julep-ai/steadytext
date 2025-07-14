# PostgreSQL Examples: Semantic Search

Examples for building powerful semantic search systems with SteadyText and PostgreSQL.

## Document Management System

### Schema Design

```sql
-- Create search schema
CREATE SCHEMA IF NOT EXISTS search;

-- Documents table
CREATE TABLE search.documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    document_type VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    embedding vector(1024),
    search_vector tsvector,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Search queries log
CREATE TABLE search.query_log (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding vector(1024),
    result_count INTEGER,
    clicked_results INTEGER[],
    search_time_ms INTEGER,
    user_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks for large documents
CREATE TABLE search.document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES search.documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    metadata JSONB DEFAULT '{}',
    UNIQUE(document_id, chunk_index)
);

-- Search feedback
CREATE TABLE search.relevance_feedback (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES search.query_log(id),
    document_id INTEGER REFERENCES search.documents(id),
    is_relevant BOOLEAN,
    feedback_type VARCHAR(20), -- 'explicit', 'implicit'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes for Performance

```sql
-- Vector similarity index
CREATE INDEX idx_documents_embedding ON search.documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_chunks_embedding ON search.document_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Full-text search index
CREATE INDEX idx_documents_search_vector ON search.documents 
USING gin (search_vector);

-- Update search vector trigger
CREATE OR REPLACE FUNCTION search.update_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_document_search_vector
    BEFORE INSERT OR UPDATE OF title, content ON search.documents
    FOR EACH ROW
    EXECUTE FUNCTION search.update_search_vector();
```

### Core Search Functions

```sql
-- Hybrid search combining vector and full-text
CREATE OR REPLACE FUNCTION search.hybrid_search(
    p_query TEXT,
    p_limit INTEGER DEFAULT 10,
    p_vector_weight FLOAT DEFAULT 0.7,
    p_text_weight FLOAT DEFAULT 0.3
) RETURNS TABLE(
    document_id INTEGER,
    title VARCHAR(500),
    content_preview TEXT,
    score FLOAT,
    match_type TEXT
) AS $$
DECLARE
    v_query_embedding vector(1024);
    v_query_tsquery tsquery;
BEGIN
    -- Generate query embedding
    v_query_embedding := steadytext_embed(p_query);
    
    -- Generate text search query
    v_query_tsquery := plainto_tsquery('english', p_query);
    
    -- Perform hybrid search
    RETURN QUERY
    WITH vector_search AS (
        SELECT 
            d.id,
            d.title,
            1 - (d.embedding <-> v_query_embedding) as vector_score
        FROM search.documents d
        WHERE v_query_embedding IS NOT NULL
        ORDER BY d.embedding <-> v_query_embedding
        LIMIT p_limit * 2
    ),
    text_search AS (
        SELECT 
            d.id,
            d.title,
            ts_rank(d.search_vector, v_query_tsquery) as text_score
        FROM search.documents d
        WHERE d.search_vector @@ v_query_tsquery
        ORDER BY text_score DESC
        LIMIT p_limit * 2
    ),
    combined_results AS (
        SELECT 
            COALESCE(v.id, t.id) as doc_id,
            COALESCE(v.title, t.title) as doc_title,
            COALESCE(v.vector_score, 0) * p_vector_weight +
            COALESCE(t.text_score, 0) * p_text_weight as combined_score,
            CASE 
                WHEN v.id IS NOT NULL AND t.id IS NOT NULL THEN 'hybrid'
                WHEN v.id IS NOT NULL THEN 'vector'
                ELSE 'text'
            END as match_type
        FROM vector_search v
        FULL OUTER JOIN text_search t ON v.id = t.id
    )
    SELECT 
        cr.doc_id as document_id,
        cr.doc_title as title,
        substring(d.content, 1, 200) || '...' as content_preview,
        cr.combined_score as score,
        cr.match_type
    FROM combined_results cr
    JOIN search.documents d ON cr.doc_id = d.id
    ORDER BY cr.combined_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Semantic search with reranking
CREATE OR REPLACE FUNCTION search.semantic_search_reranked(
    p_query TEXT,
    p_limit INTEGER DEFAULT 10,
    p_rerank_top_k INTEGER DEFAULT 30
) RETURNS TABLE(
    document_id INTEGER,
    title VARCHAR(500),
    content_preview TEXT,
    initial_score FLOAT,
    rerank_score FLOAT,
    final_rank INTEGER
) AS $$
DECLARE
    v_query_embedding vector(1024);
BEGIN
    -- Generate query embedding
    v_query_embedding := steadytext_embed(p_query);
    
    IF v_query_embedding IS NULL THEN
        RAISE NOTICE 'Failed to generate query embedding';
        RETURN;
    END IF;
    
    RETURN QUERY
    WITH initial_results AS (
        -- Get top-k candidates by vector similarity
        SELECT 
            d.id,
            d.title,
            d.content,
            1 - (d.embedding <-> v_query_embedding) as similarity
        FROM search.documents d
        ORDER BY d.embedding <-> v_query_embedding
        LIMIT p_rerank_top_k
    ),
    reranked AS (
        -- Rerank using SteadyText reranker
        SELECT 
            ir.id,
            ir.title,
            ir.content,
            ir.similarity,
            r.score as rerank_score,
            r.position
        FROM initial_results ir
        CROSS JOIN LATERAL (
            SELECT score, position
            FROM steadytext_rerank(
                p_query,
                ARRAY[ir.title || ' ' || substring(ir.content, 1, 500)]
            )
        ) r
    )
    SELECT 
        r.id as document_id,
        r.title,
        substring(r.content, 1, 200) || '...' as content_preview,
        r.similarity as initial_score,
        r.rerank_score,
        row_number() OVER (ORDER BY r.rerank_score DESC) as final_rank
    FROM reranked r
    ORDER BY r.rerank_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;
```

### Advanced Search Features

```sql
-- Multi-modal search (text + metadata filters)
CREATE OR REPLACE FUNCTION search.advanced_search(
    p_query TEXT,
    p_filters JSONB DEFAULT '{}',
    p_date_from DATE DEFAULT NULL,
    p_date_to DATE DEFAULT NULL,
    p_document_types TEXT[] DEFAULT NULL,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE(
    document_id INTEGER,
    title VARCHAR(500),
    document_type VARCHAR(50),
    score FLOAT,
    highlights TEXT[]
) AS $$
DECLARE
    v_query_embedding vector(1024);
    v_sql TEXT;
BEGIN
    -- Generate embedding
    v_query_embedding := steadytext_embed(p_query);
    
    -- Build dynamic query
    v_sql := format($sql$
        WITH filtered_docs AS (
            SELECT *
            FROM search.documents d
            WHERE 1=1
            %s  -- Date filter
            %s  -- Type filter
            %s  -- JSONB filters
        ),
        search_results AS (
            SELECT 
                d.id,
                d.title,
                d.document_type,
                d.content,
                1 - (d.embedding <-> %L::vector) as score
            FROM filtered_docs d
            WHERE %L::vector IS NOT NULL
            ORDER BY d.embedding <-> %L::vector
            LIMIT %s
        )
        SELECT 
            sr.id,
            sr.title,
            sr.document_type,
            sr.score,
            ARRAY[
                ts_headline('english', sr.content, plainto_tsquery('english', %L),
                    'StartSel=<mark>, StopSel=</mark>, MaxWords=20, MinWords=10')
            ] as highlights
        FROM search_results sr
        ORDER BY sr.score DESC
    $sql$,
        -- Date filter
        CASE 
            WHEN p_date_from IS NOT NULL OR p_date_to IS NOT NULL 
            THEN format('AND d.created_at BETWEEN %L AND %L', 
                        COALESCE(p_date_from, '1900-01-01'::date),
                        COALESCE(p_date_to, '2100-01-01'::date))
            ELSE ''
        END,
        -- Type filter
        CASE 
            WHEN p_document_types IS NOT NULL 
            THEN format('AND d.document_type = ANY(%L)', p_document_types)
            ELSE ''
        END,
        -- JSONB filters
        CASE 
            WHEN p_filters != '{}'::jsonb 
            THEN format('AND d.metadata @> %L', p_filters)
            ELSE ''
        END,
        v_query_embedding,
        v_query_embedding,
        v_query_embedding,
        p_limit,
        p_query
    );
    
    RETURN QUERY EXECUTE v_sql;
END;
$$ LANGUAGE plpgsql;

-- Query expansion for better recall
CREATE OR REPLACE FUNCTION search.expand_query(
    p_query TEXT,
    p_expansion_terms INTEGER DEFAULT 3
) RETURNS TEXT AS $$
DECLARE
    v_prompt TEXT;
    v_expanded TEXT;
BEGIN
    v_prompt := format(
        'Generate %s related search terms for the query "%s". Return as comma-separated list:',
        p_expansion_terms,
        p_query
    );
    
    v_expanded := steadytext_generate(v_prompt, 50);
    
    IF v_expanded IS NOT NULL THEN
        -- Combine original query with expansions
        RETURN p_query || ' ' || replace(v_expanded, ',', ' ');
    ELSE
        RETURN p_query;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Faceted search aggregation
CREATE OR REPLACE FUNCTION search.get_search_facets(
    p_query TEXT,
    p_facet_fields TEXT[] DEFAULT ARRAY['document_type', 'metadata.category', 'metadata.author']
) RETURNS TABLE(
    facet_field TEXT,
    facet_value TEXT,
    count BIGINT,
    sample_titles TEXT[]
) AS $$
DECLARE
    v_query_embedding vector(1024);
BEGIN
    v_query_embedding := steadytext_embed(p_query);
    
    RETURN QUERY
    WITH matching_docs AS (
        SELECT d.*
        FROM search.documents d
        WHERE 1 - (d.embedding <-> v_query_embedding) > 0.5
        ORDER BY d.embedding <-> v_query_embedding
        LIMIT 1000
    ),
    facet_counts AS (
        SELECT 
            unnest(p_facet_fields) as field,
            CASE 
                WHEN unnest(p_facet_fields) = 'document_type' THEN document_type
                WHEN unnest(p_facet_fields) LIKE 'metadata.%' THEN 
                    metadata #>> string_to_array(substring(unnest(p_facet_fields) from 10), '.')
                ELSE NULL
            END as value,
            COUNT(*) as cnt,
            array_agg(title ORDER BY embedding <-> v_query_embedding LIMIT 3) as samples
        FROM matching_docs
        GROUP BY 1, 2
    )
    SELECT 
        field as facet_field,
        value as facet_value,
        cnt as count,
        samples as sample_titles
    FROM facet_counts
    WHERE value IS NOT NULL
    ORDER BY field, cnt DESC;
END;
$$ LANGUAGE plpgsql;
```

### Search Analytics

```sql
-- Track search performance
CREATE OR REPLACE FUNCTION search.log_search_query(
    p_query TEXT,
    p_result_count INTEGER,
    p_search_time_ms INTEGER,
    p_user_id INTEGER DEFAULT NULL
) RETURNS INTEGER AS $$
DECLARE
    v_query_id INTEGER;
BEGIN
    INSERT INTO search.query_log (
        query_text, 
        query_embedding, 
        result_count, 
        search_time_ms, 
        user_id
    ) VALUES (
        p_query,
        steadytext_embed(p_query),
        p_result_count,
        p_search_time_ms,
        p_user_id
    ) RETURNING id INTO v_query_id;
    
    RETURN v_query_id;
END;
$$ LANGUAGE plpgsql;

-- Analyze search patterns
CREATE OR REPLACE FUNCTION search.analyze_search_patterns(
    p_days INTEGER DEFAULT 7
) RETURNS TABLE(
    pattern_type TEXT,
    pattern_value TEXT,
    frequency BIGINT,
    avg_results FLOAT,
    avg_clicks FLOAT,
    performance_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    -- Popular queries
    SELECT 
        'popular_query' as pattern_type,
        query_text as pattern_value,
        COUNT(*) as frequency,
        AVG(result_count) as avg_results,
        AVG(array_length(clicked_results, 1)) as avg_clicks,
        CASE 
            WHEN AVG(result_count) > 0 
            THEN AVG(array_length(clicked_results, 1)::float / result_count)
            ELSE 0
        END as performance_score
    FROM search.query_log
    WHERE created_at > NOW() - INTERVAL '1 day' * p_days
    GROUP BY query_text
    HAVING COUNT(*) > 5
    
    UNION ALL
    
    -- No-result queries
    SELECT 
        'no_results' as pattern_type,
        query_text as pattern_value,
        COUNT(*) as frequency,
        0 as avg_results,
        0 as avg_clicks,
        0 as performance_score
    FROM search.query_log
    WHERE result_count = 0
        AND created_at > NOW() - INTERVAL '1 day' * p_days
    GROUP BY query_text
    
    UNION ALL
    
    -- Low-click queries
    SELECT 
        'low_clicks' as pattern_type,
        query_text as pattern_value,
        COUNT(*) as frequency,
        AVG(result_count) as avg_results,
        0 as avg_clicks,
        0 as performance_score
    FROM search.query_log
    WHERE result_count > 0 
        AND (clicked_results IS NULL OR array_length(clicked_results, 1) = 0)
        AND created_at > NOW() - INTERVAL '1 day' * p_days
    GROUP BY query_text
    HAVING COUNT(*) > 3
    
    ORDER BY pattern_type, frequency DESC;
END;
$$ LANGUAGE plpgsql;

-- Generate search insights
CREATE OR REPLACE FUNCTION search.generate_search_insights(
    p_days INTEGER DEFAULT 30
) RETURNS TEXT AS $$
DECLARE
    v_stats RECORD;
    v_prompt TEXT;
    v_insights TEXT;
BEGIN
    -- Gather statistics
    WITH search_stats AS (
        SELECT 
            COUNT(*) as total_searches,
            COUNT(DISTINCT query_text) as unique_queries,
            AVG(result_count) as avg_results,
            AVG(search_time_ms) as avg_search_time,
            SUM(CASE WHEN result_count = 0 THEN 1 ELSE 0 END)::float / COUNT(*) as no_result_rate,
            AVG(CASE 
                WHEN result_count > 0 AND clicked_results IS NOT NULL 
                THEN array_length(clicked_results, 1)::float / result_count 
                ELSE 0 
            END) as avg_click_rate
        FROM search.query_log
        WHERE created_at > NOW() - INTERVAL '1 day' * p_days
    )
    SELECT * INTO v_stats FROM search_stats;
    
    -- Generate insights
    v_prompt := format(
        'Analyze these search metrics and provide 3 actionable insights: Total searches: %s, Unique queries: %s, Avg results: %s, No-result rate: %s%%, Avg click rate: %s%%',
        v_stats.total_searches,
        v_stats.unique_queries,
        round(v_stats.avg_results, 1),
        round(v_stats.no_result_rate * 100, 1),
        round(v_stats.avg_click_rate * 100, 1)
    );
    
    v_insights := steadytext_generate(v_prompt, 200);
    
    RETURN COALESCE(
        v_insights,
        format('Search volume: %s queries. Consider improving content for %s%% no-result queries.',
               v_stats.total_searches,
               round(v_stats.no_result_rate * 100))
    );
END;
$$ LANGUAGE plpgsql;
```

### Document Processing

```sql
-- Chunk large documents for better search
CREATE OR REPLACE FUNCTION search.chunk_document(
    p_document_id INTEGER,
    p_chunk_size INTEGER DEFAULT 500,
    p_overlap INTEGER DEFAULT 50
) RETURNS INTEGER AS $$
DECLARE
    v_content TEXT;
    v_chunks TEXT[];
    v_chunk TEXT;
    v_chunk_count INTEGER := 0;
    i INTEGER;
BEGIN
    -- Get document content
    SELECT content INTO v_content
    FROM search.documents
    WHERE id = p_document_id;
    
    -- Split into sentences (simple approach)
    v_chunks := string_to_array(v_content, '. ');
    
    -- Process chunks
    i := 1;
    WHILE i <= array_length(v_chunks, 1) LOOP
        -- Combine sentences to reach chunk size
        v_chunk := '';
        WHILE i <= array_length(v_chunks, 1) AND 
              length(v_chunk) + length(v_chunks[i]) < p_chunk_size LOOP
            v_chunk := v_chunk || v_chunks[i] || '. ';
            i := i + 1;
        END LOOP;
        
        -- Insert chunk
        IF length(v_chunk) > 50 THEN
            INSERT INTO search.document_chunks (
                document_id, 
                chunk_index, 
                content, 
                embedding
            ) VALUES (
                p_document_id,
                v_chunk_count,
                v_chunk,
                steadytext_embed(v_chunk)
            );
            v_chunk_count := v_chunk_count + 1;
            
            -- Overlap
            i := i - (p_overlap / 100);
        END IF;
    END LOOP;
    
    RETURN v_chunk_count;
END;
$$ LANGUAGE plpgsql;

-- Extract and index document metadata
CREATE OR REPLACE FUNCTION search.extract_document_metadata(
    p_content TEXT,
    p_document_type VARCHAR(50)
) RETURNS JSONB AS $$
DECLARE
    v_prompt TEXT;
    v_metadata_json TEXT;
    v_metadata JSONB;
BEGIN
    v_prompt := format(
        'Extract metadata from this %s document and return as JSON with fields: topic, keywords (array), summary, language: %s',
        p_document_type,
        substring(p_content, 1, 1000)
    );
    
    v_metadata_json := steadytext_generate_json(
        v_prompt,
        '{
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
                "language": {"type": "string"}
            }
        }'::json
    );
    
    IF v_metadata_json IS NOT NULL THEN
        v_metadata := v_metadata_json::jsonb;
    ELSE
        -- Fallback to basic extraction
        v_metadata := jsonb_build_object(
            'topic', 'Unknown',
            'keywords', ARRAY[]::text[],
            'summary', substring(p_content, 1, 200),
            'language', 'en'
        );
    END IF;
    
    RETURN v_metadata;
END;
$$ LANGUAGE plpgsql;
```

### Search Personalization

```sql
-- User search profiles
CREATE TABLE search.user_profiles (
    user_id INTEGER PRIMARY KEY,
    interest_embedding vector(1024),
    search_history JSONB DEFAULT '[]',
    preferences JSONB DEFAULT '{}',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Personalized search
CREATE OR REPLACE FUNCTION search.personalized_search(
    p_query TEXT,
    p_user_id INTEGER,
    p_limit INTEGER DEFAULT 10,
    p_personalization_weight FLOAT DEFAULT 0.3
) RETURNS TABLE(
    document_id INTEGER,
    title VARCHAR(500),
    score FLOAT,
    personalization_boost FLOAT
) AS $$
DECLARE
    v_query_embedding vector(1024);
    v_user_embedding vector(1024);
BEGIN
    -- Get embeddings
    v_query_embedding := steadytext_embed(p_query);
    
    SELECT interest_embedding INTO v_user_embedding
    FROM search.user_profiles
    WHERE user_id = p_user_id;
    
    RETURN QUERY
    WITH base_results AS (
        SELECT 
            d.id,
            d.title,
            1 - (d.embedding <-> v_query_embedding) as query_score,
            CASE 
                WHEN v_user_embedding IS NOT NULL 
                THEN 1 - (d.embedding <-> v_user_embedding)
                ELSE 0
            END as user_score
        FROM search.documents d
    )
    SELECT 
        br.id as document_id,
        br.title,
        (br.query_score * (1 - p_personalization_weight) + 
         br.user_score * p_personalization_weight) as score,
        br.user_score * p_personalization_weight as personalization_boost
    FROM base_results br
    ORDER BY score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Update user profile based on interactions
CREATE OR REPLACE FUNCTION search.update_user_profile(
    p_user_id INTEGER,
    p_query TEXT,
    p_clicked_document_id INTEGER
) RETURNS VOID AS $$
DECLARE
    v_doc_embedding vector(1024);
    v_current_embedding vector(1024);
    v_new_embedding vector(1024);
BEGIN
    -- Get document embedding
    SELECT embedding INTO v_doc_embedding
    FROM search.documents
    WHERE id = p_clicked_document_id;
    
    -- Get current user embedding
    SELECT interest_embedding INTO v_current_embedding
    FROM search.user_profiles
    WHERE user_id = p_user_id;
    
    -- Update or create profile
    IF v_current_embedding IS NULL THEN
        -- First interaction
        INSERT INTO search.user_profiles (user_id, interest_embedding)
        VALUES (p_user_id, v_doc_embedding)
        ON CONFLICT (user_id) DO UPDATE
        SET interest_embedding = v_doc_embedding,
            updated_at = NOW();
    ELSE
        -- Weighted average (90% current, 10% new)
        v_new_embedding := (v_current_embedding * 0.9 + v_doc_embedding * 0.1);
        
        UPDATE search.user_profiles
        SET interest_embedding = v_new_embedding,
            search_history = search_history || 
                jsonb_build_object(
                    'query', p_query,
                    'clicked', p_clicked_document_id,
                    'timestamp', NOW()
                ),
            updated_at = NOW()
        WHERE user_id = p_user_id;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

### Search Quality Monitoring

```sql
-- Monitor search quality metrics
CREATE OR REPLACE FUNCTION search.calculate_search_quality_metrics(
    p_days INTEGER DEFAULT 7
) RETURNS TABLE(
    metric_name TEXT,
    metric_value FLOAT,
    trend TEXT,
    status TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH current_metrics AS (
        SELECT 
            AVG(CASE WHEN result_count > 0 THEN 1 ELSE 0 END) as success_rate,
            AVG(CASE 
                WHEN result_count > 0 AND clicked_results IS NOT NULL 
                THEN array_length(clicked_results, 1)::float / LEAST(result_count, 10)
                ELSE 0 
            END) as click_through_rate,
            AVG(search_time_ms) as avg_latency,
            COUNT(DISTINCT query_text)::float / COUNT(*) as query_diversity
        FROM search.query_log
        WHERE created_at > NOW() - INTERVAL '1 day' * p_days
    ),
    previous_metrics AS (
        SELECT 
            AVG(CASE WHEN result_count > 0 THEN 1 ELSE 0 END) as success_rate,
            AVG(CASE 
                WHEN result_count > 0 AND clicked_results IS NOT NULL 
                THEN array_length(clicked_results, 1)::float / LEAST(result_count, 10)
                ELSE 0 
            END) as click_through_rate
        FROM search.query_log
        WHERE created_at > NOW() - INTERVAL '1 day' * (p_days * 2)
            AND created_at <= NOW() - INTERVAL '1 day' * p_days
    )
    SELECT 
        'Success Rate' as metric_name,
        cm.success_rate * 100 as metric_value,
        CASE 
            WHEN cm.success_rate > pm.success_rate THEN 'improving'
            WHEN cm.success_rate < pm.success_rate THEN 'declining'
            ELSE 'stable'
        END as trend,
        CASE 
            WHEN cm.success_rate > 0.9 THEN 'excellent'
            WHEN cm.success_rate > 0.7 THEN 'good'
            ELSE 'needs attention'
        END as status
    FROM current_metrics cm, previous_metrics pm
    
    UNION ALL
    
    SELECT 
        'Click-Through Rate',
        cm.click_through_rate * 100,
        CASE 
            WHEN cm.click_through_rate > pm.click_through_rate THEN 'improving'
            WHEN cm.click_through_rate < pm.click_through_rate THEN 'declining'
            ELSE 'stable'
        END,
        CASE 
            WHEN cm.click_through_rate > 0.3 THEN 'excellent'
            WHEN cm.click_through_rate > 0.1 THEN 'good'
            ELSE 'needs attention'
        END
    FROM current_metrics cm, previous_metrics pm
    
    UNION ALL
    
    SELECT 
        'Average Latency (ms)',
        cm.avg_latency,
        'stable',
        CASE 
            WHEN cm.avg_latency < 100 THEN 'excellent'
            WHEN cm.avg_latency < 500 THEN 'good'
            ELSE 'needs attention'
        END
    FROM current_metrics cm
    
    UNION ALL
    
    SELECT 
        'Query Diversity',
        cm.query_diversity * 100,
        'stable',
        CASE 
            WHEN cm.query_diversity > 0.8 THEN 'excellent'
            WHEN cm.query_diversity > 0.5 THEN 'good'
            ELSE 'needs attention'
        END
    FROM current_metrics cm;
END;
$$ LANGUAGE plpgsql;
```

## Related Documentation

- [PostgreSQL Extension Overview](../postgresql-extension.md)
- [Blog & CMS Examples](postgresql-blog-cms.md)
- [E-commerce Examples](postgresql-ecommerce.md)
- [Real-time Examples](postgresql-realtime.md)
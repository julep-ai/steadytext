# PostgreSQL Extension - Structured Generation & Reranking

This document covers structured text generation and document reranking features in the pg_steadytext PostgreSQL extension.

**Navigation**: [Main Documentation](postgresql-extension.md) | [AI Features](postgresql-extension-ai.md) | [Async Functions](postgresql-extension-async.md) | [Advanced Topics](postgresql-extension-advanced.md)

---

## Structured Generation (v2.4.1+)

New in v2.4.1, the PostgreSQL extension now supports structured text generation using llama.cpp's native grammar support.

### `steadytext_generate_json()`

Generate JSON that conforms to a JSON schema.

```sql
steadytext_generate_json(
    prompt TEXT,
    schema JSONB,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42,
    model TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT false
) RETURNS TEXT
-- Returns NULL if generation fails
-- v2.6.1+: Added model and unsafe_mode parameters for remote model support
```

**Examples:**

```sql
-- Simple JSON generation
SELECT steadytext_generate_json(
    'Create a user named John, age 30',
    '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'::jsonb
);

-- Generate product information
SELECT steadytext_generate_json(
    'Create a product listing for a laptop',
    '{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "specs": {
                "type": "object",
                "properties": {
                    "cpu": {"type": "string"},
                    "ram": {"type": "string"},
                    "storage": {"type": "string"}
                }
            }
        }
    }'::jsonb,
    seed := 999
);

-- Extract structured data from text
WITH schema AS (
    SELECT '{
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "confidence": {"type": "number"}
                    }
                }
            }
        }
    }'::jsonb AS value
)
SELECT 
    doc_id,
    steadytext_generate_json(
        'Extract entities from: ' || content,
        schema.value
    ) AS extracted_entities
FROM documents, schema
WHERE doc_type = 'research_paper';
```

### `steadytext_generate_regex()`

Generate text that matches a regular expression pattern.

```sql
steadytext_generate_regex(
    prompt TEXT,
    pattern TEXT,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42,
    model TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT false
) RETURNS TEXT
-- Returns NULL if generation fails
-- v2.6.1+: Added model and unsafe_mode parameters for remote model support
```

**Examples:**

```sql
-- Generate a phone number
SELECT steadytext_generate_regex(
    'Contact number: ',
    '\d{3}-\d{3}-\d{4}'
);

-- Generate a date
SELECT steadytext_generate_regex(
    'Event date: ',
    '\d{4}-\d{2}-\d{2}'
);

-- Generate an email
SELECT steadytext_generate_regex(
    'Email: ',
    '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
);

-- Validate and format user input
CREATE OR REPLACE FUNCTION format_phone_number(input TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN steadytext_generate_regex(
        'Format this phone number: ' || input || ' as: ',
        '\(\d{3}\) \d{3}-\d{4}'
    );
END;
$$ LANGUAGE plpgsql;

-- Generate SKU codes
SELECT 
    product_name,
    steadytext_generate_regex(
        'SKU for ' || product_name || ': ',
        '[A-Z]{3}-\d{4}-[A-Z0-9]{2}'
    ) AS sku
FROM products
WHERE sku IS NULL;
```

### `steadytext_generate_choice()`

Generate text that is one of the provided choices.

```sql
steadytext_generate_choice(
    prompt TEXT,
    choices TEXT[],
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42,
    model TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT false
) RETURNS TEXT
-- Returns NULL if generation fails
-- v2.6.1+: Added model and unsafe_mode parameters for remote model support
```

**Examples:**

```sql
-- Simple choice
SELECT steadytext_generate_choice(
    'The weather today is',
    ARRAY['sunny', 'cloudy', 'rainy']
);

-- Sentiment analysis
SELECT 
    review_id,
    review_text,
    steadytext_generate_choice(
        'Sentiment of this review: ' || review_text,
        ARRAY['positive', 'negative', 'neutral']
    ) AS sentiment
FROM product_reviews
WHERE sentiment IS NULL;

-- Classification with custom seed
SELECT steadytext_generate_choice(
    'This document is about',
    ARRAY['technology', 'business', 'health', 'sports', 'entertainment'],
    seed := 456
);

-- Multi-label classification
CREATE OR REPLACE FUNCTION classify_document(content TEXT)
RETURNS TABLE(category TEXT, relevance TEXT) AS $$
DECLARE
    categories TEXT[] := ARRAY['technology', 'business', 'health', 'sports', 'entertainment'];
    cat TEXT;
BEGIN
    FOREACH cat IN ARRAY categories
    LOOP
        RETURN QUERY SELECT 
            cat,
            steadytext_generate_choice(
                format('Is this document about %s? Document: %s', cat, content),
                ARRAY['yes', 'no', 'maybe']
            );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT * FROM classify_document('Apple announced new iPhone features...');
```

### Using Remote Models (v2.6.1+)

The structured generation functions now support remote models through the `model` and `unsafe_mode` parameters:

```sql
-- JSON generation with OpenAI
SELECT steadytext_generate_json(
    'Create a product listing',
    '{"type": "object", "properties": {"name": {"type": "string"}, "price": {"type": "number"}}}'::jsonb,
    model := 'openai:gpt-4o-mini',
    unsafe_mode := true
);

-- Regex pattern with Cerebras
SELECT steadytext_generate_regex(
    'My email is',
    '[a-z]+@[a-z]+\.com',
    model := 'cerebras:llama3.1-8b',
    unsafe_mode := true
);

-- Choice constraint with OpenAI
SELECT steadytext_generate_choice(
    'This review is',
    ARRAY['positive', 'negative', 'neutral'],
    model := 'openai:gpt-4o-mini',
    unsafe_mode := true
);

-- Note: Requires STEADYTEXT_UNSAFE_MODE environment variable to be set
-- and appropriate API keys (OPENAI_API_KEY or CEREBRAS_API_KEY)
```

### Structured Generation Best Practices

1. **Schema Design**: Keep schemas simple and well-structured for better results
2. **Prompt Engineering**: Include clear instructions in your prompts
3. **Error Handling**: Always check for NULL returns and handle appropriately
4. **Caching**: Structured generation results are cached by default - use consistent schemas
5. **Performance**: Complex schemas may take longer - consider async functions for batch processing

## Document Reranking (v1.3.0+)

PostgreSQL extension v1.3.0+ includes document reranking functionality powered by the Qwen3-Reranker-4B model.

### `steadytext_rerank()`

Rerank documents by relevance to a query.

```sql
steadytext_rerank(
    query TEXT,
    documents TEXT[],
    task TEXT DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TABLE(document TEXT, score FLOAT)
```

**Examples:**

```sql
-- Basic reranking
SELECT * FROM steadytext_rerank(
    'Python programming',
    ARRAY[
        'Python is a programming language',
        'Cats are cute animals',
        'Python snakes are found in Asia'
    ]
);

-- Custom task description
SELECT * FROM steadytext_rerank(
    'customer complaint about delivery',
    ARRAY(SELECT ticket_text FROM support_tickets WHERE created_at > NOW() - INTERVAL '7 days'),
    task := 'support ticket prioritization'
);

-- Integration with full-text search
WITH search_results AS (
    SELECT 
        doc_id,
        content,
        ts_rank(search_vector, query) AS text_score
    FROM documents, 
         plainto_tsquery('english', 'machine learning') query
    WHERE search_vector @@ query
    LIMIT 50  -- Get more candidates for reranking
)
SELECT 
    sr.doc_id,
    r.document,
    r.score as ai_score,
    sr.text_score,
    (0.7 * r.score + 0.3 * sr.text_score) as combined_score
FROM search_results sr,
     LATERAL steadytext_rerank(
         'machine learning',
         ARRAY_AGG(sr.content) OVER (),
         seed := 456
     ) r
WHERE sr.content = r.document
ORDER BY combined_score DESC
LIMIT 10;
```

### `steadytext_rerank_docs_only()`

Get reranked documents without scores.

```sql
steadytext_rerank_docs_only(
    query TEXT,
    documents TEXT[],
    task TEXT DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TABLE(document TEXT)
```

**Example:**

```sql
-- Get reranked documents for display
SELECT * FROM steadytext_rerank_docs_only(
    'machine learning',
    ARRAY(SELECT content FROM documents WHERE category = 'tech')
);

-- Create a search function
CREATE OR REPLACE FUNCTION search_documents(
    search_query TEXT,
    category_filter TEXT DEFAULT NULL,
    limit_results INTEGER DEFAULT 10
)
RETURNS TABLE(content TEXT, metadata JSONB) AS $$
BEGIN
    RETURN QUERY
    WITH candidates AS (
        SELECT content, metadata
        FROM documents
        WHERE (category_filter IS NULL OR category = category_filter)
        AND to_tsvector('english', content) @@ plainto_tsquery('english', search_query)
        LIMIT 100
    )
    SELECT c.content, c.metadata
    FROM candidates c,
         LATERAL steadytext_rerank_docs_only(
             search_query,
             ARRAY_AGG(c.content) OVER ()
         ) r
    WHERE c.content = r.document
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;
```

### `steadytext_rerank_top_k()`

Get top K most relevant documents.

```sql
steadytext_rerank_top_k(
    query TEXT,
    documents TEXT[],
    k INTEGER,
    task TEXT DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TABLE(document TEXT, score FLOAT)
```

**Example:**

```sql
-- Get top 5 most relevant support tickets
SELECT * FROM steadytext_rerank_top_k(
    'refund request',
    ARRAY(SELECT ticket_text FROM support_tickets WHERE status = 'open'),
    5
);

-- Dynamic top-k based on query complexity
CREATE OR REPLACE FUNCTION smart_search(query TEXT)
RETURNS TABLE(document TEXT, score FLOAT) AS $$
DECLARE
    k INTEGER;
BEGIN
    -- Adjust k based on query length/complexity
    k := CASE 
        WHEN length(query) < 10 THEN 3
        WHEN length(query) < 30 THEN 5
        ELSE 10
    END;
    
    RETURN QUERY
    SELECT * FROM steadytext_rerank_top_k(
        query,
        ARRAY(SELECT content FROM documents),
        k
    );
END;
$$ LANGUAGE plpgsql;
```

### `steadytext_rerank_batch()`

Batch reranking for multiple queries.

```sql
steadytext_rerank_batch(
    queries TEXT[],
    documents TEXT[],
    task TEXT DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TABLE(query_idx INTEGER, doc_idx INTEGER, score FLOAT)
```

**Example:**

```sql
-- Rerank documents for multiple queries
WITH batch_results AS (
    SELECT * FROM steadytext_rerank_batch(
        ARRAY['Python programming', 'machine learning', 'data science'],
        ARRAY(SELECT content FROM tutorials)
    )
)
SELECT 
    queries[br.query_idx + 1] as query,
    documents[br.doc_idx + 1] as document,
    br.score
FROM batch_results br,
     (SELECT ARRAY['Python programming', 'machine learning', 'data science'] as queries) q,
     (SELECT ARRAY_AGG(content) as documents FROM tutorials) d
ORDER BY br.query_idx, br.score DESC;

-- Create a recommendation matrix
CREATE OR REPLACE FUNCTION build_recommendation_matrix(
    user_queries TEXT[],
    content_items TEXT[]
)
RETURNS TABLE(user_query TEXT, content TEXT, relevance_score FLOAT) AS $$
BEGIN
    RETURN QUERY
    WITH scores AS (
        SELECT * FROM steadytext_rerank_batch(user_queries, content_items)
    )
    SELECT 
        user_queries[s.query_idx + 1],
        content_items[s.doc_idx + 1],
        s.score
    FROM scores s
    WHERE s.score > 0.5  -- Relevance threshold
    ORDER BY s.query_idx, s.score DESC;
END;
$$ LANGUAGE plpgsql;
```

### Async Reranking Functions

All reranking functions have async counterparts for non-blocking operations:

```sql
-- Queue async reranking
SELECT request_id FROM steadytext_rerank_async(
    'search query',
    ARRAY(SELECT content FROM documents)
);

-- Queue with custom parameters
SELECT request_id FROM steadytext_rerank_async(
    query := 'machine learning tutorials',
    documents := ARRAY(SELECT content FROM tutorials),
    task := 'find beginner-friendly tutorials',
    seed := 123
);

-- Check status and get results
SELECT * FROM steadytext_check_async(request_id);
SELECT * FROM steadytext_get_async_result(request_id, timeout_seconds := 30);

-- Batch async reranking
WITH requests AS (
    SELECT 
        user_id,
        steadytext_rerank_async(
            user_query,
            ARRAY(SELECT content FROM recommendations)
        ) AS request_id
    FROM user_searches
    WHERE created_at > NOW() - INTERVAL '1 hour'
)
SELECT 
    r.user_id,
    a.result
FROM requests r,
     LATERAL steadytext_get_async_result(r.request_id, 60) a;
```

### Reranking Best Practices

1. **Initial Retrieval**: Get 3-5x more candidates than final results needed
2. **Task Descriptions**: Use domain-specific task descriptions for better relevance
3. **Hybrid Scoring**: Combine reranking scores with other signals (e.g., recency, popularity)
4. **Caching Strategy**: Reranking results are cached - use consistent queries for better performance
5. **Batch Processing**: Use batch functions when reranking for multiple queries
6. **Async Operations**: Use async functions for large document sets or real-time applications

## Integration Examples

### Smart Search with Structured Extraction

```sql
-- Search and extract structured data
CREATE OR REPLACE FUNCTION search_and_extract(
    search_query TEXT,
    extract_schema JSONB
)
RETURNS TABLE(
    document TEXT,
    relevance_score FLOAT,
    extracted_data TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH ranked_docs AS (
        SELECT * FROM steadytext_rerank_top_k(
            search_query,
            ARRAY(SELECT content FROM documents),
            10
        )
    )
    SELECT 
        rd.document,
        rd.score,
        steadytext_generate_json(
            'Extract information from: ' || rd.document,
            extract_schema
        )
    FROM ranked_docs rd
    WHERE rd.score > 0.6;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT * FROM search_and_extract(
    'customer complaints about shipping',
    '{
        "type": "object",
        "properties": {
            "issue_type": {"type": "string"},
            "severity": {"type": "string"},
            "suggested_action": {"type": "string"}
        }
    }'::jsonb
);
```

### Document Classification Pipeline

```sql
-- Classify documents after reranking
CREATE OR REPLACE FUNCTION classify_relevant_documents(
    topic TEXT,
    classification_choices TEXT[]
)
RETURNS TABLE(
    document TEXT,
    relevance_score FLOAT,
    category TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH relevant_docs AS (
        SELECT * FROM steadytext_rerank_top_k(
            topic,
            ARRAY(SELECT content FROM unclassified_documents),
            20
        )
    )
    SELECT 
        rd.document,
        rd.score,
        steadytext_generate_choice(
            'Categorize this document: ' || rd.document,
            classification_choices
        )
    FROM relevant_docs rd;
END;
$$ LANGUAGE plpgsql;
```

## Performance Considerations

1. **Model Loading**: First calls may be slower if models aren't loaded
2. **Context Length**: Keep documents reasonably sized for reranking
3. **Batch Size**: Process documents in batches of 10-50 for optimal performance
4. **Caching**: Enable caching for repeated queries
5. **Async Processing**: Use async functions for large-scale operations

## Troubleshooting

### Common Issues

1. **NULL Returns**: Check daemon status and model availability
2. **Poor Reranking**: Verify task descriptions match your use case
3. **Slow Performance**: Consider using async functions or reducing document size
4. **Memory Usage**: Monitor model memory consumption for large batches

### Debugging

```sql
-- Check if structured generation is working
SELECT steadytext_generate_json(
    'Test: create {"test": true}',
    '{"type": "object", "properties": {"test": {"type": "boolean"}}}'::jsonb
);

-- Verify reranking model
SELECT * FROM steadytext_rerank(
    'test',
    ARRAY['test document', 'unrelated content']
);

-- Check daemon status
SELECT * FROM steadytext_daemon_status();
```

---

**Next**: [AI Summarization Features](postgresql-extension-ai.md) | [Async Functions](postgresql-extension-async.md)
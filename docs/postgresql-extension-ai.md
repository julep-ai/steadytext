# PostgreSQL Extension - AI Summarization Features

This document covers AI-powered text summarization and fact extraction features in the pg_steadytext PostgreSQL extension.

**Navigation**: [Main Documentation](postgresql-extension.md) | [Structured Generation](postgresql-extension-structured.md) | [Async Functions](postgresql-extension-async.md) | [Advanced Topics](postgresql-extension-advanced.md)

---

## AI Summarization (v1.1.0+)

The PostgreSQL extension includes powerful AI summarization aggregate functions that work seamlessly with TimescaleDB continuous aggregates.

### Core Summarization Functions

#### `ai_summarize_text()`

Summarize a single text with optional metadata.

```sql
ai_summarize_text(
    text_input TEXT,
    metadata JSONB DEFAULT NULL,
    max_tokens INTEGER DEFAULT 150,
    seed INTEGER DEFAULT 42
) RETURNS TEXT
```

**Examples:**

```sql
-- Simple text summarization
SELECT ai_summarize_text(
    'PostgreSQL is an advanced open-source relational database with ACID compliance, 
     JSON support, and extensibility through custom functions and types.',
    '{"source": "documentation"}'::jsonb
);

-- Summarize with custom parameters
SELECT ai_summarize_text(
    content,
    jsonb_build_object('importance', importance, 'category', category),
    max_tokens := 200,
    seed := 123
) AS summary
FROM documents
WHERE length(content) > 1000;

-- Batch summarization with metadata
SELECT 
    doc_id,
    title,
    ai_summarize_text(
        content,
        jsonb_build_object(
            'author', author,
            'date', created_at,
            'type', doc_type
        ),
        max_tokens := 100
    ) AS brief_summary
FROM articles
WHERE published = true
ORDER BY created_at DESC
LIMIT 10;
```

#### `ai_summarize()` Aggregate Function

Intelligently summarize multiple texts into a coherent summary.

```sql
-- Basic aggregate summarization
SELECT 
    category,
    ai_summarize(content) AS category_summary,
    count(*) AS doc_count
FROM documents
GROUP BY category;

-- With metadata
SELECT 
    department,
    ai_summarize(
        report_text,
        jsonb_build_object('priority', priority, 'date', report_date)
    ) AS department_summary
FROM reports
WHERE report_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY department;

-- Summarize customer feedback by product
SELECT 
    product_id,
    p.product_name,
    ai_summarize(
        r.review_text,
        jsonb_build_object(
            'rating', r.rating,
            'verified', r.verified_purchase
        )
    ) AS product_feedback_summary,
    avg(r.rating) AS avg_rating,
    count(*) AS review_count
FROM reviews r
JOIN products p ON r.product_id = p.id
WHERE r.created_at >= NOW() - INTERVAL '30 days'
GROUP BY product_id, p.product_name
HAVING count(*) > 5
ORDER BY avg_rating DESC;
```

### Partial Aggregation for TimescaleDB

The extension supports partial aggregation for use with TimescaleDB continuous aggregates:

#### `ai_summarize_partial()` and `ai_summarize_final()`

These functions enable efficient summarization in distributed and time-series scenarios.

```sql
-- Create continuous aggregate with partial summarization
CREATE MATERIALIZED VIEW hourly_log_summaries
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS hour,
    log_level,
    service_name,
    ai_summarize_partial(
        log_message,
        jsonb_build_object(
            'severity', severity,
            'service', service_name,
            'error_code', error_code
        )
    ) AS partial_summary,
    count(*) AS log_count,
    count(DISTINCT error_code) AS unique_errors
FROM application_logs
GROUP BY hour, log_level, service_name;

-- Query with final summarization
SELECT 
    time_bucket('1 day', hour) as day,
    log_level,
    ai_summarize_final(partial_summary) as daily_summary,
    sum(log_count) as total_logs,
    sum(unique_errors) as total_unique_errors
FROM hourly_log_summaries
WHERE hour >= NOW() - INTERVAL '7 days'
GROUP BY day, log_level
ORDER BY day DESC;

-- Create a hierarchical summarization system
CREATE MATERIALIZED VIEW daily_summaries AS
SELECT 
    date_trunc('day', hour) AS day,
    log_level,
    ai_summarize_final(partial_summary) AS daily_summary,
    sum(log_count) AS daily_logs
FROM hourly_log_summaries
GROUP BY day, log_level;

-- Weekly rollup
CREATE MATERIALIZED VIEW weekly_summaries AS
SELECT 
    date_trunc('week', day) AS week,
    log_level,
    ai_summarize(daily_summary) AS weekly_summary,
    sum(daily_logs) AS weekly_logs
FROM daily_summaries
GROUP BY week, log_level;
```

### Fact Extraction

#### `ai_extract_facts()`

Extract key facts from text content.

```sql
ai_extract_facts(
    text_input TEXT,
    max_facts INTEGER DEFAULT 5,
    seed INTEGER DEFAULT 42
) RETURNS TEXT[]
```

**Examples:**

```sql
-- Extract facts from a document
SELECT ai_extract_facts(
    'PostgreSQL supports JSON, arrays, full-text search, window functions, 
     CTEs, and has built-in replication. It also offers ACID compliance 
     and supports multiple programming languages for stored procedures.',
    max_facts := 7
);
-- Returns: {
--   "PostgreSQL supports JSON",
--   "PostgreSQL supports arrays",
--   "PostgreSQL has full-text search",
--   "PostgreSQL has window functions",
--   "PostgreSQL supports CTEs",
--   "PostgreSQL has built-in replication",
--   "PostgreSQL offers ACID compliance"
-- }

-- Extract facts from multiple documents
SELECT 
    doc_id,
    title,
    ai_extract_facts(content, 3) AS key_facts
FROM technical_docs
WHERE category = 'database'
LIMIT 10;

-- Build a fact database
CREATE TABLE extracted_facts (
    id SERIAL PRIMARY KEY,
    source_doc_id INTEGER REFERENCES documents(id),
    fact TEXT,
    extracted_at TIMESTAMP DEFAULT NOW(),
    confidence FLOAT DEFAULT 0.9
);

INSERT INTO extracted_facts (source_doc_id, fact)
SELECT 
    d.id,
    unnest(ai_extract_facts(d.content, 10))
FROM documents d
WHERE d.processed = false;

-- Mark documents as processed
UPDATE documents SET processed = true 
WHERE id IN (SELECT DISTINCT source_doc_id FROM extracted_facts);
```

#### `ai_deduplicate_facts()`

Deduplicate similar facts using semantic similarity comparison.

```sql
ai_deduplicate_facts(
    facts_jsonb JSONB,
    similarity_threshold FLOAT DEFAULT 0.8,
    seed INTEGER DEFAULT 42
) RETURNS JSONB
```

**Examples:**

```sql
-- Deduplicate facts from multiple sources
WITH all_facts AS (
    SELECT jsonb_agg(fact) AS facts
    FROM (
        SELECT unnest(ai_extract_facts(content, 10)) AS fact
        FROM documents
        WHERE category = 'PostgreSQL'
    ) extracted
)
SELECT ai_deduplicate_facts(facts, 0.85) AS unique_facts
FROM all_facts;

-- Process facts with metadata
WITH extracted_facts AS (
    SELECT 
        doc_id,
        jsonb_agg(
            jsonb_build_object(
                'fact', fact,
                'source', doc_id,
                'confidence', confidence
            )
        ) AS fact_objects
    FROM (
        SELECT 
            doc_id,
            unnest(ai_extract_facts(content)) AS fact,
            0.9 AS confidence
        FROM research_papers
    ) f
    GROUP BY doc_id
)
SELECT 
    ai_deduplicate_facts(
        jsonb_agg(fact_objects),
        similarity_threshold := 0.75
    ) AS deduplicated_facts
FROM extracted_facts;

-- Create a knowledge graph
CREATE OR REPLACE FUNCTION build_knowledge_graph(
    category_filter TEXT,
    similarity_threshold FLOAT DEFAULT 0.8
)
RETURNS TABLE(fact TEXT, sources TEXT[], confidence FLOAT) AS $$
BEGIN
    RETURN QUERY
    WITH raw_facts AS (
        SELECT 
            d.id AS doc_id,
            unnest(ai_extract_facts(d.content, 20)) AS fact
        FROM documents d
        WHERE d.category = category_filter
    ),
    fact_groups AS (
        SELECT 
            jsonb_agg(
                jsonb_build_object(
                    'fact', fact,
                    'source', doc_id::text
                )
            ) AS facts
        FROM raw_facts
    ),
    deduplicated AS (
        SELECT ai_deduplicate_facts(facts, similarity_threshold) AS result
        FROM fact_groups
    )
    SELECT 
        (fact_obj->>'fact')::TEXT AS fact,
        array_agg(DISTINCT fact_obj->>'source') AS sources,
        0.9::FLOAT AS confidence
    FROM deduplicated,
         jsonb_array_elements(result) AS fact_obj
    GROUP BY fact_obj->>'fact';
END;
$$ LANGUAGE plpgsql;
```

### Real-World Use Cases

#### Log Analysis Dashboard

```sql
-- Real-time error summarization
CREATE OR REPLACE VIEW error_summaries AS
SELECT 
    date_trunc('hour', timestamp) AS error_hour,
    service_name,
    ai_summarize(
        error_message,
        jsonb_build_object(
            'count', count(*),
            'unique_errors', count(DISTINCT error_code)
        )
    ) AS error_summary,
    array_agg(DISTINCT error_code) AS error_codes,
    count(*) AS error_count
FROM error_logs
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY error_hour, service_name
ORDER BY error_hour DESC;

-- Alert generation based on summaries
CREATE OR REPLACE FUNCTION generate_alerts()
RETURNS TABLE(
    service TEXT,
    severity TEXT,
    summary TEXT,
    action_required TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH recent_errors AS (
        SELECT * FROM error_summaries
        WHERE error_hour >= NOW() - INTERVAL '1 hour'
        AND error_count > 10
    )
    SELECT 
        re.service_name,
        CASE 
            WHEN re.error_count > 100 THEN 'CRITICAL'
            WHEN re.error_count > 50 THEN 'HIGH'
            ELSE 'MEDIUM'
        END AS severity,
        re.error_summary,
        steadytext_generate(
            format('Based on this error summary, suggest immediate action: %s', 
                   re.error_summary),
            max_tokens := 100
        ) AS action_required
    FROM recent_errors re;
END;
$$ LANGUAGE plpgsql;
```

#### Document Intelligence System

```sql
-- Automatic document categorization and summarization
CREATE OR REPLACE FUNCTION process_new_documents()
RETURNS TABLE(
    document_id INTEGER,
    title TEXT,
    summary TEXT,
    key_facts TEXT[],
    suggested_category TEXT,
    suggested_tags TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH doc_analysis AS (
        SELECT 
            d.id,
            d.title,
            ai_summarize_text(d.content, max_tokens := 150) AS summary,
            ai_extract_facts(d.content, 5) AS key_facts
        FROM documents d
        WHERE d.processed_at IS NULL
    )
    SELECT 
        da.id,
        da.title,
        da.summary,
        da.key_facts,
        steadytext_generate_choice(
            format('Category for document: %s', da.summary),
            ARRAY['technical', 'business', 'legal', 'marketing', 'other']
        ) AS suggested_category,
        string_to_array(
            steadytext_generate_regex(
                format('Generate 3 tags for: %s', da.summary),
                '[a-z]+, [a-z]+, [a-z]+'
            ),
            ', '
        ) AS suggested_tags
    FROM doc_analysis da;
END;
$$ LANGUAGE plpgsql;

-- Update documents with analysis
WITH analysis AS (
    SELECT * FROM process_new_documents()
)
UPDATE documents d
SET 
    summary = a.summary,
    category = a.suggested_category,
    tags = a.suggested_tags,
    processed_at = NOW()
FROM analysis a
WHERE d.id = a.document_id;
```

#### Customer Feedback Analysis

```sql
-- Analyze customer feedback trends
CREATE OR REPLACE FUNCTION analyze_feedback_trends(
    time_period INTERVAL DEFAULT '30 days'
)
RETURNS TABLE(
    period DATE,
    sentiment TEXT,
    summary TEXT,
    common_issues TEXT[],
    improvement_suggestions TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH feedback_by_period AS (
        SELECT 
            date_trunc('week', created_at) AS week,
            steadytext_generate_choice(
                format('Sentiment: %s', feedback_text),
                ARRAY['positive', 'negative', 'neutral']
            ) AS sentiment,
            feedback_text
        FROM customer_feedback
        WHERE created_at >= NOW() - time_period
    ),
    aggregated AS (
        SELECT 
            week,
            sentiment,
            ai_summarize(feedback_text) AS period_summary,
            array_agg(DISTINCT 
                unnest(ai_extract_facts(feedback_text, 3))
            ) AS issues
        FROM feedback_by_period
        GROUP BY week, sentiment
    )
    SELECT 
        a.week::DATE,
        a.sentiment,
        a.period_summary,
        a.issues[1:5], -- Top 5 issues
        steadytext_generate(
            format('Based on this feedback summary, suggest improvements: %s', 
                   a.period_summary),
            max_tokens := 150
        )
    FROM aggregated a
    ORDER BY a.week DESC, a.sentiment;
END;
$$ LANGUAGE plpgsql;
```

#### Research Paper Analysis

```sql
-- Extract and organize research insights
CREATE OR REPLACE FUNCTION analyze_research_papers(
    topic_filter TEXT
)
RETURNS TABLE(
    paper_id INTEGER,
    title TEXT,
    abstract_summary TEXT,
    key_findings TEXT[],
    methodology TEXT,
    future_work TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH paper_analysis AS (
        SELECT 
            p.id,
            p.title,
            ai_summarize_text(p.abstract, max_tokens := 100) AS abstract_summary,
            ai_extract_facts(p.content, 7) AS findings
        FROM papers p
        WHERE p.content ILIKE '%' || topic_filter || '%'
    )
    SELECT 
        pa.id,
        pa.title,
        pa.abstract_summary,
        pa.findings[1:5] AS key_findings,
        steadytext_generate(
            format('Extract methodology from: %s', pa.abstract_summary),
            max_tokens := 100
        ) AS methodology,
        array[
            steadytext_generate(
                format('Suggest future research based on: %s', 
                       array_to_string(pa.findings[1:3], ' ')),
                max_tokens := 50
            )
        ] AS future_work
    FROM paper_analysis pa;
END;
$$ LANGUAGE plpgsql;

-- Create a research knowledge base
CREATE MATERIALIZED VIEW research_knowledge_base AS
WITH all_research AS (
    SELECT * FROM analyze_research_papers('machine learning')
),
deduplicated_findings AS (
    SELECT ai_deduplicate_facts(
        jsonb_agg(
            jsonb_build_object(
                'fact', unnest(key_findings),
                'source', paper_id
            )
        ),
        0.7
    ) AS unique_findings
    FROM all_research
)
SELECT 
    (finding->>'fact')::TEXT AS finding,
    array_agg(DISTINCT (finding->>'source')::INTEGER) AS source_papers,
    count(*) AS mention_count
FROM deduplicated_findings,
     jsonb_array_elements(unique_findings) AS finding
GROUP BY finding->>'fact'
ORDER BY mention_count DESC;
```

### Best Practices

1. **Metadata Usage**: Always include relevant metadata for better context in summaries
2. **Token Limits**: Adjust max_tokens based on your needs - shorter for briefs, longer for detailed summaries
3. **Batch Processing**: Use async functions for large-scale summarization tasks
4. **Caching**: Summaries are cached by default - use consistent inputs for better performance
5. **Fact Extraction**: Extract more facts than needed, then deduplicate for comprehensive coverage
6. **Continuous Aggregates**: Use TimescaleDB integration for time-series data summarization

### Performance Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_documents_category_length 
ON documents(category, length(content));

-- Optimize fact extraction with parallel processing
CREATE OR REPLACE FUNCTION parallel_fact_extraction(
    batch_size INTEGER DEFAULT 100
)
RETURNS void AS $$
DECLARE
    doc_batch RECORD;
BEGIN
    FOR doc_batch IN 
        SELECT array_agg(id) AS doc_ids
        FROM (
            SELECT id 
            FROM documents 
            WHERE facts_extracted = false
            LIMIT batch_size
        ) t
    LOOP
        -- Process batch asynchronously
        PERFORM steadytext_generate_async(
            format('Extract facts from document %s', doc_id)
        )
        FROM unnest(doc_batch.doc_ids) AS doc_id;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Monitor summarization performance
CREATE OR REPLACE VIEW summarization_stats AS
SELECT 
    'ai_summarize' AS function_name,
    count(*) AS total_calls,
    avg(processing_time_ms) AS avg_time_ms,
    max(processing_time_ms) AS max_time_ms,
    sum(CASE WHEN cached THEN 1 ELSE 0 END) AS cache_hits
FROM steadytext_function_stats
WHERE function_name LIKE 'ai_summarize%'
GROUP BY function_name;
```

### Troubleshooting

```sql
-- Test summarization functions
SELECT ai_summarize_text('This is a test document for summarization.');

-- Check if fact extraction is working
SELECT ai_extract_facts('PostgreSQL has many features including JSON support.');

-- Verify deduplication
SELECT ai_deduplicate_facts(
    '[{"fact": "PostgreSQL supports JSON"}, 
      {"fact": "PostgreSQL has JSON support"}]'::jsonb,
    0.8
);

-- Debug partial aggregation
WITH test_data AS (
    SELECT ai_summarize_partial('Test text ' || generate_series::text)
    FROM generate_series(1, 5)
)
SELECT ai_summarize_final(partial_summary)
FROM test_data;
```

---

**Next**: [Async Functions](postgresql-extension-async.md) | [Advanced Topics](postgresql-extension-advanced.md)
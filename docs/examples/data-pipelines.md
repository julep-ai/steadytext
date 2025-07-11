# Data Pipelines with AI Enrichment

Build intelligent ETL pipelines that enrich, transform, and analyze data using SteadyText's AI capabilities directly in PostgreSQL.

## Overview

This tutorial shows how to create data pipelines that:
- Enrich raw data with AI-generated insights
- Transform unstructured data into structured formats
- Monitor data quality with AI validation
- Generate automated reports and summaries
- Create real-time data enrichment streams

## Prerequisites

```bash
# Start PostgreSQL with SteadyText
docker run -d -p 5432:5432 --name steadytext-etl julep/pg-steadytext

# Connect and setup
psql -h localhost -U postgres -c "CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;"
psql -h localhost -U postgres -c "CREATE EXTENSION IF NOT EXISTS pg_cron;"  # For scheduling
```

## Pipeline Architecture

Create a flexible pipeline schema:

```sql
-- Pipeline definitions
CREATE TABLE data_pipelines (
    id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    source_table VARCHAR(100),
    target_table VARCHAR(100),
    transform_function VARCHAR(100),
    schedule_cron VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    last_run TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Pipeline execution log
CREATE TABLE pipeline_runs (
    id SERIAL PRIMARY KEY,
    pipeline_id INTEGER REFERENCES data_pipelines(id),
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    status VARCHAR(20), -- 'running', 'completed', 'failed'
    records_processed INTEGER,
    records_enriched INTEGER,
    error_message TEXT,
    execution_stats JSONB
);

-- Raw data staging table
CREATE TABLE raw_data_staging (
    id SERIAL PRIMARY KEY,
    source_system VARCHAR(50),
    raw_content TEXT,
    metadata JSONB,
    processed BOOLEAN DEFAULT FALSE,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enriched data warehouse
CREATE TABLE enriched_data (
    id SERIAL PRIMARY KEY,
    source_id INTEGER,
    source_system VARCHAR(50),
    original_content TEXT,
    ai_summary TEXT,
    extracted_entities JSONB,
    sentiment_analysis JSONB,
    categories TEXT[],
    quality_score DECIMAL(3, 2),
    enriched_at TIMESTAMPTZ DEFAULT NOW()
);

-- Data quality monitoring
CREATE TABLE data_quality_issues (
    id SERIAL PRIMARY KEY,
    pipeline_id INTEGER REFERENCES data_pipelines(id),
    record_id INTEGER,
    issue_type VARCHAR(50),
    severity VARCHAR(20),
    description TEXT,
    ai_recommendation TEXT,
    detected_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Real-Time Data Enrichment Pipeline

Create a pipeline that enriches incoming data in real-time:

```sql
-- Main enrichment function
CREATE OR REPLACE FUNCTION enrich_raw_data()
RETURNS TRIGGER AS $$
DECLARE
    v_summary TEXT;
    v_entities JSONB;
    v_sentiment JSONB;
    v_categories TEXT[];
    v_quality_score DECIMAL(3, 2);
BEGIN
    -- Skip if already processed
    IF NEW.processed THEN
        RETURN NEW;
    END IF;
    
    -- Generate AI summary
    v_summary := steadytext_generate(
        format('Summarize this data in 2 sentences: %s',
            LEFT(NEW.raw_content, 1000)),
        max_tokens := 100
    );
    
    -- Extract entities
    v_entities := steadytext_generate_json(
        format('Extract entities from: %s', LEFT(NEW.raw_content, 500)),
        '{
            "people": {"type": "array", "items": {"type": "string"}},
            "organizations": {"type": "array", "items": {"type": "string"}},
            "locations": {"type": "array", "items": {"type": "string"}},
            "dates": {"type": "array", "items": {"type": "string"}},
            "amounts": {"type": "array", "items": {"type": "string"}}
        }'::json
    )::jsonb;
    
    -- Sentiment analysis
    v_sentiment := jsonb_build_object(
        'overall', steadytext_generate_choice(
            'Overall sentiment: ' || LEFT(NEW.raw_content, 500),
            ARRAY['positive', 'neutral', 'negative']
        ),
        'confidence', 0.85 + random() * 0.15  -- Simulated confidence
    );
    
    -- Categorization
    v_categories := string_to_array(
        steadytext_generate(
            format('List up to 3 categories for this content (comma-separated): %s',
                LEFT(NEW.raw_content, 500)),
            max_tokens := 30
        ),
        ', '
    );
    
    -- Calculate quality score
    v_quality_score := CASE
        WHEN length(NEW.raw_content) < 50 THEN 0.3
        WHEN v_entities IS NULL OR jsonb_typeof(v_entities) != 'object' THEN 0.5
        ELSE 0.7 + random() * 0.3
    END;
    
    -- Insert enriched data
    INSERT INTO enriched_data (
        source_id, source_system, original_content,
        ai_summary, extracted_entities, sentiment_analysis,
        categories, quality_score
    ) VALUES (
        NEW.id, NEW.source_system, NEW.raw_content,
        v_summary, v_entities, v_sentiment,
        v_categories, v_quality_score
    );
    
    -- Mark as processed
    NEW.processed := TRUE;
    
    -- Check for quality issues
    IF v_quality_score < 0.5 THEN
        INSERT INTO data_quality_issues (
            record_id, issue_type, severity, description, ai_recommendation
        ) VALUES (
            NEW.id,
            'low_quality_content',
            'medium',
            format('Quality score %s is below threshold', v_quality_score),
            steadytext_generate(
                'Suggest how to improve data quality for: ' || LEFT(NEW.raw_content, 200),
                max_tokens := 100
            )
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for real-time enrichment
CREATE TRIGGER enrich_on_insert
    BEFORE INSERT OR UPDATE ON raw_data_staging
    FOR EACH ROW
    EXECUTE FUNCTION enrich_raw_data();
```

## Batch Processing Pipeline

Create a batch pipeline for large-scale data processing:

```sql
-- Batch enrichment function
CREATE OR REPLACE FUNCTION batch_enrich_pipeline(
    p_pipeline_name VARCHAR,
    p_batch_size INTEGER DEFAULT 100
)
RETURNS TABLE (
    processed_count INTEGER,
    enriched_count INTEGER,
    error_count INTEGER,
    execution_time INTERVAL
) AS $$
DECLARE
    v_pipeline data_pipelines%ROWTYPE;
    v_run_id INTEGER;
    v_start_time TIMESTAMPTZ;
    v_processed INTEGER := 0;
    v_enriched INTEGER := 0;
    v_errors INTEGER := 0;
BEGIN
    v_start_time := NOW();
    
    -- Get pipeline configuration
    SELECT * INTO v_pipeline FROM data_pipelines 
    WHERE pipeline_name = p_pipeline_name AND is_active;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Pipeline % not found or inactive', p_pipeline_name;
    END IF;
    
    -- Create pipeline run record
    INSERT INTO pipeline_runs (pipeline_id, start_time, status)
    VALUES (v_pipeline.id, v_start_time, 'running')
    RETURNING id INTO v_run_id;
    
    -- Process in batches
    FOR r IN 
        SELECT * FROM raw_data_staging 
        WHERE NOT processed 
        ORDER BY ingested_at 
        LIMIT p_batch_size
    LOOP
        BEGIN
            -- Process individual record
            UPDATE raw_data_staging SET processed = TRUE WHERE id = r.id;
            v_processed := v_processed + 1;
            
            -- The trigger will handle enrichment
            v_enriched := v_enriched + 1;
            
        EXCEPTION WHEN OTHERS THEN
            v_errors := v_errors + 1;
            
            INSERT INTO data_quality_issues (
                pipeline_id, record_id, issue_type, severity, description
            ) VALUES (
                v_pipeline.id, r.id, 'processing_error', 'high', SQLERRM
            );
        END;
    END LOOP;
    
    -- Update pipeline run status
    UPDATE pipeline_runs 
    SET end_time = NOW(),
        status = 'completed',
        records_processed = v_processed,
        records_enriched = v_enriched,
        execution_stats = jsonb_build_object(
            'errors', v_errors,
            'avg_processing_time_ms', 
            EXTRACT(MILLISECONDS FROM (NOW() - v_start_time)) / NULLIF(v_processed, 0)
        )
    WHERE id = v_run_id;
    
    -- Update pipeline last run
    UPDATE data_pipelines 
    SET last_run = NOW() 
    WHERE id = v_pipeline.id;
    
    RETURN QUERY SELECT 
        v_processed,
        v_enriched,
        v_errors,
        NOW() - v_start_time;
END;
$$ LANGUAGE plpgsql;
```

## Data Quality Monitoring

Implement AI-powered data quality checks:

```sql
-- Data quality monitoring function
CREATE OR REPLACE FUNCTION monitor_data_quality(
    p_hours_back INTEGER DEFAULT 24
)
RETURNS TABLE (
    quality_metric VARCHAR,
    score DECIMAL,
    issues_found INTEGER,
    ai_insights TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH quality_metrics AS (
        SELECT 
            'completeness' AS metric,
            AVG(CASE 
                WHEN ai_summary IS NOT NULL 
                 AND extracted_entities IS NOT NULL 
                 AND categories IS NOT NULL 
                THEN 1.0 ELSE 0.0 
            END) AS score,
            COUNT(*) FILTER (WHERE quality_score < 0.5) AS issues
        FROM enriched_data
        WHERE enriched_at > NOW() - (p_hours_back || ' hours')::INTERVAL
        
        UNION ALL
        
        SELECT 
            'accuracy' AS metric,
            AVG(quality_score) AS score,
            COUNT(DISTINCT dqi.id) AS issues
        FROM enriched_data ed
        LEFT JOIN data_quality_issues dqi ON dqi.record_id = ed.source_id
        WHERE ed.enriched_at > NOW() - (p_hours_back || ' hours')::INTERVAL
        
        UNION ALL
        
        SELECT 
            'timeliness' AS metric,
            CASE 
                WHEN AVG(EXTRACT(EPOCH FROM (enriched_at - ed.enriched_at))) < 300 
                THEN 1.0 
                ELSE 0.5 
            END AS score,
            COUNT(*) FILTER (
                WHERE EXTRACT(EPOCH FROM (enriched_at - ed.enriched_at)) > 600
            ) AS issues
        FROM enriched_data ed
        JOIN raw_data_staging rs ON ed.source_id = rs.id
        WHERE ed.enriched_at > NOW() - (p_hours_back || ' hours')::INTERVAL
    )
    SELECT 
        metric,
        ROUND(score, 2),
        issues,
        steadytext_generate(
            format('Analyze data quality: %s score is %s with %s issues',
                metric, ROUND(score, 2), issues),
            max_tokens := 100
        ) AS ai_insights
    FROM quality_metrics;
END;
$$ LANGUAGE plpgsql;
```

## Automated Report Generation

Generate intelligent reports from pipeline data:

```sql
-- Automated report generation
CREATE OR REPLACE FUNCTION generate_pipeline_report(
    p_pipeline_id INTEGER,
    p_period INTERVAL DEFAULT INTERVAL '1 day'
)
RETURNS TABLE (
    report_section VARCHAR,
    content TEXT,
    metrics JSONB
) AS $$
DECLARE
    v_pipeline data_pipelines%ROWTYPE;
BEGIN
    SELECT * INTO v_pipeline FROM data_pipelines WHERE id = p_pipeline_id;
    
    RETURN QUERY
    -- Executive Summary
    WITH pipeline_stats AS (
        SELECT 
            COUNT(*) AS total_runs,
            SUM(records_processed) AS total_processed,
            SUM(records_enriched) AS total_enriched,
            AVG(EXTRACT(EPOCH FROM (end_time - start_time))) AS avg_duration_seconds,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed_runs
        FROM pipeline_runs
        WHERE pipeline_id = p_pipeline_id
          AND start_time > NOW() - p_period
    )
    SELECT 
        'executive_summary' AS report_section,
        steadytext_generate(
            format('Pipeline %s processed %s records in %s runs over the past %s. Average duration: %s seconds. Failed runs: %s',
                v_pipeline.pipeline_name,
                total_processed,
                total_runs,
                p_period,
                ROUND(avg_duration_seconds, 2),
                failed_runs
            ),
            max_tokens := 200
        ) AS content,
        to_jsonb(pipeline_stats.*) AS metrics
    FROM pipeline_stats
    
    UNION ALL
    
    -- Data Quality Analysis
    WITH quality_analysis AS (
        SELECT 
            AVG(quality_score) AS avg_quality,
            COUNT(*) FILTER (WHERE quality_score < 0.5) AS low_quality_count,
            array_agg(DISTINCT unnest(categories)) AS all_categories
        FROM enriched_data ed
        JOIN raw_data_staging rs ON ed.source_id = rs.id
        JOIN pipeline_runs pr ON pr.pipeline_id = p_pipeline_id
        WHERE ed.enriched_at BETWEEN pr.start_time AND COALESCE(pr.end_time, NOW())
          AND pr.start_time > NOW() - p_period
    )
    SELECT 
        'quality_analysis',
        steadytext_generate(
            format('Data quality analysis: Average score %s. Low quality records: %s. Categories covered: %s',
                ROUND(avg_quality, 2),
                low_quality_count,
                array_to_string(all_categories[1:5], ', ')
            ),
            max_tokens := 150
        ),
        jsonb_build_object(
            'avg_quality_score', avg_quality,
            'low_quality_count', low_quality_count,
            'category_count', array_length(all_categories, 1)
        )
    FROM quality_analysis
    
    UNION ALL
    
    -- Trend Analysis
    SELECT 
        'trend_analysis',
        steadytext_generate(
            format('Analyze trends for pipeline %s based on: %s',
                v_pipeline.pipeline_name,
                jsonb_pretty(
                    jsonb_build_object(
                        'processing_volume', 
                        (SELECT array_agg(records_processed ORDER BY start_time)
                         FROM pipeline_runs 
                         WHERE pipeline_id = p_pipeline_id 
                           AND start_time > NOW() - p_period
                         LIMIT 10)
                    )
                )
            ),
            max_tokens := 200
        ),
        NULL;
END;
$$ LANGUAGE plpgsql;
```

## Stream Processing Integration

Handle real-time data streams:

```sql
-- Streaming data handler
CREATE OR REPLACE FUNCTION process_data_stream(
    p_stream_data JSONB
)
RETURNS VOID AS $$
DECLARE
    v_record JSONB;
    v_source_system VARCHAR;
BEGIN
    -- Extract source system
    v_source_system := p_stream_data->>'source_system';
    
    -- Process each record in the stream
    FOR v_record IN SELECT * FROM jsonb_array_elements(p_stream_data->'records')
    LOOP
        INSERT INTO raw_data_staging (
            source_system,
            raw_content,
            metadata
        ) VALUES (
            v_source_system,
            v_record->>'content',
            v_record->'metadata'
        );
    END LOOP;
    
    -- Trigger batch processing if needed
    IF (SELECT COUNT(*) FROM raw_data_staging WHERE NOT processed) > 1000 THEN
        PERFORM batch_enrich_pipeline('main_pipeline', 1000);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- API endpoint for streaming
CREATE OR REPLACE FUNCTION api_ingest_stream(
    p_api_key VARCHAR,
    p_data JSONB
)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
BEGIN
    -- Validate API key (simplified)
    IF p_api_key != 'your-secret-key' THEN
        RETURN jsonb_build_object('error', 'Invalid API key');
    END IF;
    
    -- Process the stream
    PERFORM process_data_stream(p_data);
    
    -- Return success response
    RETURN jsonb_build_object(
        'status', 'success',
        'records_received', jsonb_array_length(p_data->'records'),
        'timestamp', NOW()
    );
END;
$$ LANGUAGE plpgsql;
```

## Pipeline Orchestration

Schedule and orchestrate complex pipelines:

```sql
-- Create sample pipelines
INSERT INTO data_pipelines (pipeline_name, description, schedule_cron) VALUES
('hourly_enrichment', 'Process and enrich data every hour', '0 * * * *'),
('daily_quality_check', 'Daily data quality monitoring', '0 9 * * *'),
('weekly_report', 'Generate weekly executive reports', '0 10 * * 1');

-- Schedule with pg_cron
SELECT cron.schedule(
    'hourly_enrichment_job',
    '0 * * * *',
    $$SELECT batch_enrich_pipeline('hourly_enrichment', 500);$$
);

SELECT cron.schedule(
    'daily_quality_job',
    '0 9 * * *',
    $$INSERT INTO data_quality_reports 
      SELECT NOW(), * FROM monitor_data_quality(24);$$
);

-- Complex pipeline with dependencies
CREATE OR REPLACE FUNCTION orchestrate_complex_pipeline()
RETURNS VOID AS $$
BEGIN
    -- Step 1: Ingest raw data
    PERFORM process_data_stream(
        jsonb_build_object(
            'source_system', 'automated_import',
            'records', (
                SELECT jsonb_agg(
                    jsonb_build_object(
                        'content', external_content,
                        'metadata', metadata
                    )
                )
                FROM external_data_source
                WHERE import_date = CURRENT_DATE
            )
        )
    );
    
    -- Step 2: Enrich data
    PERFORM batch_enrich_pipeline('hourly_enrichment', 1000);
    
    -- Step 3: Quality check
    INSERT INTO data_quality_reports
    SELECT NOW(), * FROM monitor_data_quality(1);
    
    -- Step 4: Generate insights
    INSERT INTO executive_insights
    SELECT * FROM generate_pipeline_report(
        (SELECT id FROM data_pipelines WHERE pipeline_name = 'hourly_enrichment'),
        INTERVAL '1 hour'
    );
    
    -- Step 5: Alert on issues
    PERFORM pg_notify('pipeline_complete', 
        json_build_object(
            'pipeline', 'complex_orchestration',
            'status', 'completed',
            'timestamp', NOW()
        )::text
    );
END;
$$ LANGUAGE plpgsql;
```

## Sample Data and Testing

```sql
-- Insert test data
INSERT INTO raw_data_staging (source_system, raw_content, metadata) VALUES
('crm', 'Customer John Smith called about product issue. He was frustrated with the delayed shipping and wants a refund. Order #12345.', 
 '{"customer_id": "C123", "call_duration": "15:32"}'::jsonb),
('social_media', 'Just received my new headphones from @YourCompany! Amazing sound quality and super comfortable. Best purchase this year! #Happy', 
 '{"platform": "twitter", "engagement": {"likes": 45, "retweets": 12}}'::jsonb),
('support_email', 'Subject: Technical Issue\n\nDear Support,\n\nI am experiencing connectivity issues with model XZ-500. The device keeps disconnecting every few minutes. I have tried resetting but the problem persists.\n\nPlease help.\n\nRegards,\nJane Doe', 
 '{"ticket_id": "T789", "priority": "high"}'::jsonb);

-- Run enrichment pipeline
SELECT * FROM batch_enrich_pipeline('hourly_enrichment', 10);

-- Check enriched data
SELECT 
    source_system,
    ai_summary,
    sentiment_analysis->>'overall' AS sentiment,
    categories,
    quality_score
FROM enriched_data
ORDER BY enriched_at DESC
LIMIT 5;

-- Monitor quality
SELECT * FROM monitor_data_quality(24);

-- Generate report
SELECT * FROM generate_pipeline_report(1, INTERVAL '1 day');
```

## Performance Optimization

```sql
-- Parallel processing function
CREATE OR REPLACE FUNCTION parallel_enrich_pipeline(
    p_pipeline_name VARCHAR,
    p_parallel_workers INTEGER DEFAULT 4
)
RETURNS VOID AS $$
BEGIN
    -- Use PostgreSQL parallel queries
    SET max_parallel_workers_per_gather = p_parallel_workers;
    
    -- Process in parallel
    UPDATE raw_data_staging rs
    SET processed = TRUE
    FROM (
        SELECT id, 
               steadytext_generate('Summarize: ' || raw_content, 100) AS summary
        FROM raw_data_staging
        WHERE NOT processed
        LIMIT 1000
    ) enriched
    WHERE rs.id = enriched.id;
    
    RESET max_parallel_workers_per_gather;
END;
$$ LANGUAGE plpgsql;
```

## Best Practices

1. **Batch Size**: Tune batch sizes based on your hardware
2. **Error Handling**: Always implement comprehensive error handling
3. **Monitoring**: Set up alerts for pipeline failures
4. **Caching**: Use SteadyText's caching for repeated AI operations
5. **Scheduling**: Use pg_cron for reliable pipeline scheduling

## Next Steps

- [TimescaleDB Integration →](../integrations/timescaledb.md)
- [Production Deployment →](../deployment/production.md)
- [Migration Guides →](../migration/from-openai.md)

---

!!! tip "Pro Tip"
    For high-volume pipelines, consider partitioning your staging tables by date and using parallel workers to maximize throughput.
# TimescaleDB + SteadyText: AI-Powered Time-Series Analytics

Combine TimescaleDB's time-series superpowers with SteadyText's AI capabilities for intelligent, automated analytics at scale.

## Overview

TimescaleDB + SteadyText enables:
- **Continuous AI aggregates** that summarize data automatically
- **Real-time pattern detection** in time-series data
- **Intelligent alerting** based on AI analysis
- **Automated report generation** from historical data
- **Predictive insights** from time-series trends

## Prerequisites

```bash
# Option 1: Docker with both extensions
docker run -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  timescale/timescaledb-ha:pg16 \
  -c shared_preload_libraries='timescaledb,pg_steadytext'

# Option 2: Install on existing TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;
```

## Core Concepts

### Hypertables Meet AI

```sql
-- Create a hypertable for sensor data
CREATE TABLE sensor_data (
    time TIMESTAMPTZ NOT NULL,
    sensor_id INTEGER,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    status TEXT,
    error_message TEXT
);

SELECT create_hypertable('sensor_data', 'time');

-- Add AI analysis column
ALTER TABLE sensor_data 
ADD COLUMN ai_analysis TEXT;

-- Automatically analyze anomalies on insert
CREATE OR REPLACE FUNCTION analyze_sensor_reading()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.temperature > 80 OR NEW.temperature < 20 THEN
        NEW.ai_analysis := steadytext_generate(
            format('Analyze abnormal temperature reading: %s°C at sensor %s. Previous status: %s',
                NEW.temperature, NEW.sensor_id, NEW.status),
            max_tokens := 100
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER analyze_on_insert
    BEFORE INSERT ON sensor_data
    FOR EACH ROW
    EXECUTE FUNCTION analyze_sensor_reading();
```

## Continuous AI Aggregates

The killer feature: AI summaries that update automatically!

### Example 1: Hourly Log Summaries

```sql
-- Create continuous aggregate with AI summaries
CREATE MATERIALIZED VIEW hourly_system_insights
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour'::interval, time) AS hour,
    count(*) AS event_count,
    count(*) FILTER (WHERE severity = 'ERROR') AS error_count,
    count(*) FILTER (WHERE severity = 'WARNING') AS warning_count,
    steadytext_generate(
        format('Summarize system status: %s total events, %s errors, %s warnings. Key messages: %s',
            count(*),
            count(*) FILTER (WHERE severity = 'ERROR'),
            count(*) FILTER (WHERE severity = 'WARNING'),
            string_agg(
                CASE WHEN severity IN ('ERROR', 'WARNING') 
                THEN message ELSE NULL END, 
                '; ' 
                ORDER BY time
            )
        ),
        max_tokens := 200
    ) AS ai_summary
FROM system_logs
GROUP BY hour;

-- Refresh policy for real-time updates
SELECT add_continuous_aggregate_policy('hourly_system_insights',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '10 minutes'
);
```

### Example 2: Daily Business Metrics

```sql
-- Sales analysis with AI insights
CREATE MATERIALIZED VIEW daily_sales_intelligence
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day'::interval, order_time) AS day,
    product_category,
    COUNT(*) AS orders,
    SUM(amount) AS revenue,
    AVG(amount) AS avg_order_value,
    COUNT(DISTINCT customer_id) AS unique_customers,
    steadytext_generate(
        format('Analyze sales performance for %s: $%s revenue from %s orders (%s customers). AOV: $%s',
            product_category,
            ROUND(SUM(amount), 2),
            COUNT(*),
            COUNT(DISTINCT customer_id),
            ROUND(AVG(amount), 2)
        ),
        max_tokens := 150
    ) AS performance_analysis,
    steadytext_generate_json(
        format('Suggest 3 actions to improve %s sales based on: revenue=$%s, orders=%s, AOV=$%s',
            product_category,
            ROUND(SUM(amount), 2),
            COUNT(*),
            ROUND(AVG(amount), 2)
        ),
        '{"recommendations": {"type": "array", "items": {"type": "string"}}}'::json
    )::jsonb AS ai_recommendations
FROM orders
GROUP BY day, product_category;

-- Auto-refresh every hour
SELECT add_continuous_aggregate_policy('daily_sales_intelligence',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);
```

## Real-Time Pattern Detection

Detect complex patterns in streaming data:

```sql
-- Function to detect patterns across time windows
CREATE OR REPLACE FUNCTION detect_anomaly_patterns(
    p_hours_back INTEGER DEFAULT 24
)
RETURNS TABLE (
    pattern_type VARCHAR,
    severity VARCHAR,
    affected_sensors INTEGER[],
    ai_diagnosis TEXT,
    recommended_action TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH recent_data AS (
        SELECT 
            sensor_id,
            time,
            temperature,
            humidity,
            LAG(temperature) OVER (PARTITION BY sensor_id ORDER BY time) AS prev_temp,
            LAG(humidity) OVER (PARTITION BY sensor_id ORDER BY time) AS prev_humidity
        FROM sensor_data
        WHERE time > NOW() - (p_hours_back || ' hours')::INTERVAL
    ),
    anomalies AS (
        SELECT 
            sensor_id,
            COUNT(*) AS anomaly_count,
            AVG(ABS(temperature - prev_temp)) AS avg_temp_change,
            MAX(temperature) AS max_temp,
            MIN(temperature) AS min_temp
        FROM recent_data
        WHERE ABS(temperature - prev_temp) > 5  -- Rapid changes
           OR temperature > 75 
           OR temperature < 25
        GROUP BY sensor_id
        HAVING COUNT(*) > 3  -- Persistent anomalies
    )
    SELECT 
        'temperature_instability' AS pattern_type,
        CASE 
            WHEN MAX(anomaly_count) > 10 THEN 'critical'
            WHEN MAX(anomaly_count) > 5 THEN 'high'
            ELSE 'medium'
        END AS severity,
        array_agg(sensor_id) AS affected_sensors,
        steadytext_generate(
            format('Diagnose temperature instability: %s sensors affected, max variations: %s°C',
                COUNT(DISTINCT sensor_id),
                ROUND(MAX(avg_temp_change), 2)
            ),
            max_tokens := 150
        ) AS ai_diagnosis,
        steadytext_generate(
            format('Recommend action for %s sensors with temperature anomalies (severity: %s)',
                COUNT(DISTINCT sensor_id),
                CASE 
                    WHEN MAX(anomaly_count) > 10 THEN 'critical'
                    WHEN MAX(anomaly_count) > 5 THEN 'high'
                    ELSE 'medium'
                END
            ),
            max_tokens := 100
        ) AS recommended_action
    FROM anomalies
    GROUP BY pattern_type;
END;
$$ LANGUAGE plpgsql;
```

## Intelligent Data Retention

Use AI to decide what historical data to keep:

```sql
-- Intelligent compression policy
CREATE OR REPLACE FUNCTION intelligent_compression_policy()
RETURNS VOID AS $$
DECLARE
    v_chunk RECORD;
    v_importance_score DECIMAL;
    v_compression_decision TEXT;
BEGIN
    FOR v_chunk IN 
        SELECT 
            chunk_name,
            range_start,
            range_end,
            chunk_table_size,
            compression_status
        FROM timescaledb_information.chunks
        WHERE hypertable_name = 'sensor_data'
          AND range_end < NOW() - INTERVAL '7 days'
          AND compression_status IS NULL
    LOOP
        -- AI evaluates chunk importance
        v_compression_decision := steadytext_generate_choice(
            format('Should we compress sensor data from %s to %s? Size: %s. Analyze for historical importance.',
                v_chunk.range_start::date,
                v_chunk.range_end::date,
                pg_size_pretty(v_chunk.chunk_table_size)
            ),
            ARRAY['compress_aggressive', 'compress_normal', 'keep_uncompressed']
        );
        
        -- Execute decision
        CASE v_compression_decision
            WHEN 'compress_aggressive' THEN
                -- Compress with aggressive settings
                PERFORM compress_chunk(v_chunk.chunk_name::regclass, if_not_compressed => true);
                
            WHEN 'compress_normal' THEN
                -- Standard compression
                PERFORM compress_chunk(v_chunk.chunk_name::regclass);
                
            ELSE
                -- Keep uncompressed for now
                RAISE NOTICE 'Keeping chunk % uncompressed due to importance', v_chunk.chunk_name;
        END CASE;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Schedule intelligent compression
SELECT cron.schedule('intelligent_compression', '0 2 * * *', 'SELECT intelligent_compression_policy()');
```

## Predictive Analytics

Combine time-series data with AI predictions:

```sql
-- Predictive maintenance system
CREATE OR REPLACE FUNCTION predict_equipment_failure(
    p_sensor_id INTEGER,
    p_hours_ahead INTEGER DEFAULT 24
)
RETURNS TABLE (
    prediction_time TIMESTAMPTZ,
    failure_probability DECIMAL,
    risk_factors JSONB,
    maintenance_recommendation TEXT
) AS $$
DECLARE
    v_recent_patterns TEXT;
    v_historical_failures INTEGER;
BEGIN
    -- Gather recent patterns
    SELECT 
        format('Recent: Avg temp %s°C, %s errors in last 24h, %s maintenance events',
            ROUND(AVG(temperature), 1),
            COUNT(*) FILTER (WHERE error_message IS NOT NULL),
            COUNT(DISTINCT maintenance_id)
        ) INTO v_recent_patterns
    FROM sensor_data
    WHERE sensor_id = p_sensor_id
      AND time > NOW() - INTERVAL '24 hours';
    
    -- Get historical context
    SELECT COUNT(*) INTO v_historical_failures
    FROM equipment_failures
    WHERE sensor_id = p_sensor_id
      AND time > NOW() - INTERVAL '90 days';
    
    RETURN QUERY
    SELECT 
        NOW() + (p_hours_ahead || ' hours')::INTERVAL AS prediction_time,
        (steadytext_generate_json(
            format('Predict failure probability (0-1) for sensor %s: %s. Historical failures: %s',
                p_sensor_id, v_recent_patterns, v_historical_failures),
            '{"probability": {"type": "number", "minimum": 0, "maximum": 1}}'::json
        )::jsonb->>'probability')::DECIMAL AS failure_probability,
        steadytext_generate_json(
            format('Identify risk factors for sensor %s: %s',
                p_sensor_id, v_recent_patterns),
            '{
                "temperature_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                "usage_pattern_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                "age_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                "maintenance_overdue": {"type": "boolean"}
            }'::json
        )::jsonb AS risk_factors,
        steadytext_generate(
            format('Recommend maintenance for sensor %s based on: %s',
                p_sensor_id, v_recent_patterns),
            max_tokens := 150
        ) AS maintenance_recommendation;
END;
$$ LANGUAGE plpgsql;
```

## Performance Optimization

### Parallel AI Processing

```sql
-- Enable parallel processing for large aggregates
ALTER DATABASE mydb SET max_parallel_workers_per_gather = 4;
ALTER DATABASE mydb SET max_parallel_workers = 8;

-- Parallel AI analysis function
CREATE OR REPLACE FUNCTION parallel_analyze_time_range(
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    bucket_size INTERVAL DEFAULT '1 hour'
)
RETURNS TABLE (
    bucket TIMESTAMPTZ,
    analysis TEXT
) AS $$
BEGIN
    -- Force parallel execution
    SET LOCAL max_parallel_workers_per_gather = 4;
    
    RETURN QUERY
    SELECT 
        time_bucket(bucket_size, time) AS bucket,
        steadytext_generate(
            'Summarize: ' || string_agg(message, '; '),
            max_tokens := 100
        ) AS analysis
    FROM sensor_data
    WHERE time BETWEEN start_time AND end_time
    GROUP BY time_bucket(bucket_size, time)
    ORDER BY bucket;
END;
$$ LANGUAGE plpgsql;
```

### Caching Strategies

```sql
-- Cache AI results for frequently accessed time periods
CREATE TABLE ai_analysis_cache (
    time_bucket TIMESTAMPTZ,
    cache_key VARCHAR(255),
    analysis_result TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time_bucket, cache_key)
);

-- Auto-expire old cache entries
SELECT create_hypertable('ai_analysis_cache', 'time_bucket');
SELECT add_retention_policy('ai_analysis_cache', INTERVAL '7 days');
```

## Real-World Use Cases

### IoT Sensor Networks

```sql
-- Complete IoT monitoring solution
CREATE MATERIALIZED VIEW iot_fleet_status
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 minutes'::interval, time) AS bucket,
    device_type,
    COUNT(DISTINCT device_id) AS active_devices,
    AVG(battery_level) AS avg_battery,
    COUNT(*) FILTER (WHERE status = 'ERROR') AS errors,
    steadytext_generate(
        format('Fleet status: %s %s devices, %s%% avg battery, %s errors',
            COUNT(DISTINCT device_id),
            device_type,
            ROUND(AVG(battery_level)),
            COUNT(*) FILTER (WHERE status = 'ERROR')
        ),
        max_tokens := 100
    ) AS fleet_summary
FROM iot_telemetry
GROUP BY bucket, device_type;
```

### Financial Trading

```sql
-- Market analysis with AI insights
CREATE MATERIALIZED VIEW market_intelligence
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 minute'::interval, time) AS minute,
    symbol,
    AVG(price) AS avg_price,
    SUM(volume) AS total_volume,
    MAX(price) - MIN(price) AS price_range,
    steadytext_generate(
        format('Analyze %s: price movement $%s, volume %s, volatility %s%%',
            symbol,
            ROUND(MAX(price) - MIN(price), 2),
            SUM(volume),
            ROUND((MAX(price) - MIN(price)) / AVG(price) * 100, 2)
        ),
        max_tokens := 100
    ) AS market_analysis
FROM trades
GROUP BY minute, symbol;
```

## Monitoring & Alerting

```sql
-- AI-powered alert system
CREATE OR REPLACE FUNCTION check_alerts()
RETURNS VOID AS $$
DECLARE
    v_alert RECORD;
BEGIN
    FOR v_alert IN 
        SELECT * FROM detect_anomaly_patterns(1)
        WHERE severity IN ('high', 'critical')
    LOOP
        -- Send intelligent alerts
        PERFORM pg_notify(
            'ai_alert',
            jsonb_build_object(
                'severity', v_alert.severity,
                'diagnosis', v_alert.ai_diagnosis,
                'action', v_alert.recommended_action,
                'timestamp', NOW()
            )::text
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Schedule alert checks
SELECT cron.schedule('ai_alerts', '*/5 * * * *', 'SELECT check_alerts()');
```

## Best Practices

1. **Chunk Size**: Optimize chunk_time_interval for your workload
2. **Aggregate Design**: Pre-compute AI summaries in continuous aggregates
3. **Compression**: Use AI to identify compressible chunks
4. **Indexes**: Create indexes on AI-generated columns for fast queries
5. **Parallel Processing**: Enable for large-scale AI operations

## Performance Benchmarks

```sql
-- Benchmark AI processing speed
DO $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    record_count INTEGER;
BEGIN
    start_time := clock_timestamp();
    
    -- Process 1 million records
    SELECT COUNT(*) INTO record_count
    FROM (
        SELECT steadytext_generate('Analyze: ' || message, 50)
        FROM system_logs
        LIMIT 1000000
    ) t;
    
    end_time := clock_timestamp();
    
    RAISE NOTICE 'Processed % records in % seconds (% records/sec)',
        record_count,
        EXTRACT(EPOCH FROM (end_time - start_time)),
        record_count / EXTRACT(EPOCH FROM (end_time - start_time));
END $$;
```

## Troubleshooting

### Common Issues

1. **Slow continuous aggregates**
   - Solution: Reduce AI token count in aggregates
   - Use sampling for very large time buckets

2. **Memory usage**
   - Solution: Tune work_mem for AI operations
   - Use batching for large datasets

3. **Lock contention**
   - Solution: Use CONCURRENTLY option
   - Schedule refreshes during low-traffic periods

## Next Steps

- [Log Analysis Example →](../examples/log-analysis.md)
- [Production Deployment →](../deployment/production.md)
- [PostgreSQL Extension Docs →](../postgresql-extension-hero.md)

---

!!! tip "Pro Tip"
    Start with hourly aggregates and tune based on your needs. The combination of TimescaleDB's efficiency and SteadyText's determinism makes even minute-level AI aggregates feasible for many workloads.
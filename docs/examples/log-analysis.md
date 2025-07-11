# Log Analysis with AI-Powered Summarization

Transform your logs from noise into insights using SteadyText's AI capabilities directly in PostgreSQL.

## Overview

This tutorial shows how to build an intelligent log analysis system that:
- Automatically summarizes error patterns
- Identifies security threats in real-time
- Creates hourly/daily AI-powered reports
- Integrates seamlessly with TimescaleDB for time-series analysis

## Prerequisites

```bash
# Install PostgreSQL with SteadyText
docker run -d -p 5432:5432 --name steadytext-logs julep/pg-steadytext

# Connect to the database
psql -h localhost -U postgres
```

## Setting Up the Schema

First, let's create a table for our application logs:

```sql
-- Create the logs table
CREATE TABLE application_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level VARCHAR(10) NOT NULL,
    service VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    request_id UUID,
    user_id INTEGER,
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_logs_timestamp ON application_logs(timestamp DESC);
CREATE INDEX idx_logs_level ON application_logs(level);
CREATE INDEX idx_logs_service ON application_logs(service);
CREATE INDEX idx_logs_metadata ON application_logs USING GIN(metadata);

-- Enable the SteadyText extension
CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;
```

## Real-Time Error Summarization

Create a function that summarizes errors in real-time:

```sql
-- Function to analyze error patterns
CREATE OR REPLACE FUNCTION analyze_error_patterns(
    time_window INTERVAL DEFAULT '1 hour'
)
RETURNS TABLE (
    error_summary TEXT,
    affected_services TEXT[],
    error_count INTEGER,
    severity_score INTEGER,
    recommended_actions TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH recent_errors AS (
        SELECT 
            level,
            service,
            message,
            COUNT(*) as count
        FROM application_logs
        WHERE timestamp > NOW() - time_window
          AND level IN ('ERROR', 'CRITICAL')
        GROUP BY level, service, message
    ),
    aggregated AS (
        SELECT 
            string_agg(
                format('%s (%s): %s [%s times]', 
                    level, service, 
                    LEFT(message, 100), count::text
                ), 
                '; '
            ) AS error_details,
            array_agg(DISTINCT service) AS services,
            SUM(count)::INTEGER AS total_errors
        FROM recent_errors
    )
    SELECT 
        steadytext_generate(
            'Analyze these application errors and provide a concise summary: ' || 
            error_details
        ) AS error_summary,
        services AS affected_services,
        total_errors AS error_count,
        CASE 
            WHEN total_errors > 100 THEN 5
            WHEN total_errors > 50 THEN 4
            WHEN total_errors > 20 THEN 3
            WHEN total_errors > 5 THEN 2
            ELSE 1
        END AS severity_score,
        steadytext_generate(
            'Based on these errors, suggest 3 immediate actions: ' || 
            error_details
        ) AS recommended_actions
    FROM aggregated
    WHERE total_errors > 0;
END;
$$ LANGUAGE plpgsql;
```

## TimescaleDB Integration for Historical Analysis

If you're using TimescaleDB, create continuous aggregates for automatic summarization:

```sql
-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('application_logs', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create hourly error summaries
CREATE MATERIALIZED VIEW hourly_error_insights
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour'::interval, timestamp) AS hour,
    level,
    service,
    COUNT(*) AS error_count,
    steadytext_generate(
        format('Summarize these %s errors from %s service: %s',
            level,
            service,
            string_agg(LEFT(message, 200), '; ' ORDER BY timestamp)
        )
    ) AS ai_summary,
    array_agg(DISTINCT user_id) FILTER (WHERE user_id IS NOT NULL) AS affected_users,
    array_agg(DISTINCT ip_address) FILTER (WHERE ip_address IS NOT NULL) AS source_ips
FROM application_logs
WHERE level IN ('ERROR', 'CRITICAL', 'WARNING')
  AND timestamp > NOW() - INTERVAL '7 days'
GROUP BY hour, level, service;

-- Add refresh policy
SELECT add_continuous_aggregate_policy('hourly_error_insights',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '10 minutes'
);
```

## Security Threat Detection

Use AI to identify potential security threats in your logs:

```sql
-- Security analysis view
CREATE OR REPLACE VIEW security_alerts AS
WITH suspicious_activity AS (
    SELECT 
        timestamp,
        ip_address,
        user_id,
        service,
        message,
        metadata,
        steadytext_generate_choice(
            'Classify security risk: ' || message,
            ARRAY['safe', 'low_risk', 'medium_risk', 'high_risk', 'critical']
        ) AS risk_level
    FROM application_logs
    WHERE timestamp > NOW() - INTERVAL '1 hour'
      AND (
        message ILIKE '%failed login%'
        OR message ILIKE '%unauthorized%'
        OR message ILIKE '%injection%'
        OR message ILIKE '%suspicious%'
        OR metadata->>'status_code' IN ('401', '403')
      )
)
SELECT 
    timestamp,
    ip_address,
    risk_level,
    COUNT(*) OVER (PARTITION BY ip_address) AS attempts_from_ip,
    steadytext_generate(
        format('Analyze security threat: IP %s attempted: %s',
            ip_address,
            string_agg(message, '; ')
        )
    ) AS threat_analysis
FROM suspicious_activity
WHERE risk_level NOT IN ('safe', 'low_risk')
GROUP BY timestamp, ip_address, risk_level, user_id, service, message;
```

## Daily Executive Summary

Create automated daily reports for stakeholders:

```sql
-- Daily summary function
CREATE OR REPLACE FUNCTION generate_daily_log_report(
    report_date DATE DEFAULT CURRENT_DATE - 1
)
RETURNS TABLE (
    report_date DATE,
    executive_summary TEXT,
    key_metrics JSONB,
    top_issues TEXT[],
    recommendations TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH daily_stats AS (
        SELECT 
            COUNT(*) AS total_logs,
            COUNT(*) FILTER (WHERE level = 'ERROR') AS error_count,
            COUNT(*) FILTER (WHERE level = 'CRITICAL') AS critical_count,
            COUNT(DISTINCT service) AS active_services,
            COUNT(DISTINCT user_id) AS active_users,
            array_agg(DISTINCT service) FILTER (
                WHERE level IN ('ERROR', 'CRITICAL')
            ) AS problematic_services
        FROM application_logs
        WHERE DATE(timestamp) = report_date
    ),
    error_details AS (
        SELECT 
            string_agg(
                format('%s: %s (%s times)', 
                    service, 
                    LEFT(message, 100), 
                    COUNT(*)::text
                ),
                '; '
            ) AS error_summary
        FROM application_logs
        WHERE DATE(timestamp) = report_date
          AND level IN ('ERROR', 'CRITICAL')
        GROUP BY service, message
        ORDER BY COUNT(*) DESC
        LIMIT 10
    )
    SELECT 
        report_date,
        steadytext_generate(
            format('Executive summary for %s: Total logs: %s, Errors: %s, Critical: %s. Top errors: %s',
                report_date,
                total_logs,
                error_count,
                critical_count,
                error_summary
            )
        ) AS executive_summary,
        jsonb_build_object(
            'total_logs', total_logs,
            'error_count', error_count,
            'critical_count', critical_count,
            'error_rate', ROUND((error_count::NUMERIC / NULLIF(total_logs, 0) * 100), 2),
            'active_services', active_services,
            'active_users', active_users,
            'problematic_services', problematic_services
        ) AS key_metrics,
        ARRAY(
            SELECT DISTINCT 
                service || ': ' || LEFT(message, 100)
            FROM application_logs
            WHERE DATE(timestamp) = report_date
              AND level IN ('ERROR', 'CRITICAL')
            ORDER BY 1
            LIMIT 5
        ) AS top_issues,
        steadytext_generate(
            'Based on these metrics, provide 3 actionable recommendations: ' || 
            format('Error rate: %s%%, Critical issues: %s, Problematic services: %s',
                ROUND((error_count::NUMERIC / NULLIF(total_logs, 0) * 100), 2),
                critical_count,
                array_to_string(problematic_services, ', ')
            )
        ) AS recommendations
    FROM daily_stats, error_details;
END;
$$ LANGUAGE plpgsql;
```

## Pattern Recognition and Anomaly Detection

Identify unusual patterns in your logs:

```sql
-- Anomaly detection view
CREATE OR REPLACE VIEW log_anomalies AS
WITH baseline AS (
    -- Calculate baseline metrics for the past week
    SELECT 
        service,
        level,
        EXTRACT(HOUR FROM timestamp) AS hour_of_day,
        AVG(COUNT(*)) OVER (
            PARTITION BY service, level, EXTRACT(HOUR FROM timestamp)
        ) AS avg_count,
        STDDEV(COUNT(*)) OVER (
            PARTITION BY service, level, EXTRACT(HOUR FROM timestamp)
        ) AS stddev_count
    FROM application_logs
    WHERE timestamp > NOW() - INTERVAL '7 days'
      AND timestamp < NOW() - INTERVAL '1 hour'
    GROUP BY service, level, EXTRACT(HOUR FROM timestamp), DATE(timestamp)
),
current_hour AS (
    -- Get current hour's metrics
    SELECT 
        service,
        level,
        COUNT(*) AS current_count
    FROM application_logs
    WHERE timestamp > NOW() - INTERVAL '1 hour'
    GROUP BY service, level
)
SELECT 
    c.service,
    c.level,
    c.current_count,
    ROUND(b.avg_count, 2) AS expected_count,
    CASE 
        WHEN b.stddev_count > 0 AND 
             ABS(c.current_count - b.avg_count) > 2 * b.stddev_count 
        THEN 'ANOMALY'
        ELSE 'NORMAL'
    END AS status,
    steadytext_generate(
        format('Analyze anomaly: %s service %s level - Current: %s, Expected: %s (±%s)',
            c.service,
            c.level,
            c.current_count,
            ROUND(b.avg_count, 2),
            ROUND(b.stddev_count, 2)
        )
    ) AS anomaly_analysis
FROM current_hour c
JOIN baseline b ON 
    c.service = b.service 
    AND c.level = b.level 
    AND EXTRACT(HOUR FROM NOW()) = b.hour_of_day
WHERE ABS(c.current_count - b.avg_count) > b.stddev_count;
```

## Sample Data and Testing

Let's insert some sample data to test our log analysis:

```sql
-- Insert sample log data
INSERT INTO application_logs (timestamp, level, service, message, metadata, user_id, ip_address)
VALUES 
    (NOW() - INTERVAL '2 hours', 'ERROR', 'auth', 'Failed login attempt', '{"attempts": 3}'::jsonb, 123, '192.168.1.100'::inet),
    (NOW() - INTERVAL '90 minutes', 'ERROR', 'auth', 'Failed login attempt', '{"attempts": 5}'::jsonb, 123, '192.168.1.100'::inet),
    (NOW() - INTERVAL '1 hour', 'CRITICAL', 'auth', 'Potential brute force attack detected', '{"attempts": 10}'::jsonb, NULL, '192.168.1.100'::inet),
    (NOW() - INTERVAL '45 minutes', 'ERROR', 'api', 'Database connection timeout', '{"duration": 5000}'::jsonb, 456, '10.0.0.50'::inet),
    (NOW() - INTERVAL '30 minutes', 'WARNING', 'api', 'Slow query detected', '{"query_time": 3.5}'::jsonb, 789, '10.0.0.51'::inet),
    (NOW() - INTERVAL '15 minutes', 'ERROR', 'payment', 'Payment processing failed', '{"error": "Gateway timeout"}'::jsonb, 321, '172.16.0.10'::inet),
    (NOW() - INTERVAL '5 minutes', 'INFO', 'api', 'User logged in successfully', '{"method": "OAuth"}'::jsonb, 654, '192.168.1.50'::inet);

-- Test our analysis functions
SELECT * FROM analyze_error_patterns('2 hours');
SELECT * FROM security_alerts;
SELECT * FROM generate_daily_log_report();
```

## Automation with pg_cron

Schedule automatic reports using pg_cron:

```sql
-- Enable pg_cron
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Schedule hourly error analysis
SELECT cron.schedule(
    'hourly-error-analysis',
    '0 * * * *',
    $$INSERT INTO error_analysis_history 
      SELECT NOW(), * FROM analyze_error_patterns('1 hour')$$
);

-- Schedule daily reports
SELECT cron.schedule(
    'daily-log-report',
    '0 8 * * *',
    $$INSERT INTO daily_reports 
      SELECT * FROM generate_daily_log_report()$$
);
```

## Best Practices

1. **Index Strategy**: Always index timestamp and frequently queried fields
2. **Partitioning**: Use TimescaleDB or native partitioning for large datasets
3. **Caching**: SteadyText caches AI results automatically
4. **Batch Processing**: Process logs in batches for better performance
5. **Retention**: Set up automatic data retention policies

## Performance Optimization

```sql
-- Create a summary table for faster queries
CREATE TABLE log_summaries (
    id SERIAL PRIMARY KEY,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    service VARCHAR(50),
    level VARCHAR(10),
    count INTEGER,
    ai_summary TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for fast lookups
CREATE INDEX idx_log_summaries_period 
ON log_summaries(period_start, period_end);

-- Batch process historical data
INSERT INTO log_summaries (period_start, period_end, service, level, count, ai_summary)
SELECT 
    date_trunc('hour', timestamp) AS period_start,
    date_trunc('hour', timestamp) + INTERVAL '1 hour' AS period_end,
    service,
    level,
    COUNT(*) as count,
    steadytext_generate(
        'Summarize: ' || string_agg(LEFT(message, 100), '; ')
    ) AS ai_summary
FROM application_logs
WHERE timestamp < NOW() - INTERVAL '1 day'
GROUP BY date_trunc('hour', timestamp), service, level;
```

## Next Steps

- [Content Management Examples →](content-management.md)
- [Customer Intelligence Tutorial →](customer-intelligence.md)
- [TimescaleDB Integration Guide →](../integrations/timescaledb.md)

---

!!! tip "Pro Tip"
    Use materialized views with SteadyText for pre-computed AI summaries. This gives you instant query performance while keeping the AI insights fresh.
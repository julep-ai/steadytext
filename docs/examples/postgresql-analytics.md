# PostgreSQL Examples: Analytics & Performance Monitoring

Examples for building analytics and monitoring systems with AI-powered insights using SteadyText.

## Performance Monitoring System

### Schema Design

```sql
-- Create analytics schema
CREATE SCHEMA IF NOT EXISTS analytics;

-- Application metrics
CREATE TABLE analytics.app_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC NOT NULL,
    metric_type VARCHAR(50), -- counter, gauge, histogram
    tags JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Error logs with AI analysis
CREATE TABLE analytics.error_logs (
    id SERIAL PRIMARY KEY,
    error_hash VARCHAR(64),
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    occurrence_count INTEGER DEFAULT 1,
    severity VARCHAR(20),
    ai_analysis TEXT,
    suggested_fix TEXT,
    embedding vector(1024),
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance traces
CREATE TABLE analytics.traces (
    id SERIAL PRIMARY KEY,
    trace_id UUID DEFAULT gen_random_uuid(),
    operation_name VARCHAR(200),
    duration_ms INTEGER,
    status VARCHAR(20),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Anomaly detection
CREATE TABLE analytics.anomalies (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100),
    anomaly_type VARCHAR(50),
    severity FLOAT,
    description TEXT,
    ai_explanation TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- User behavior analytics
CREATE TABLE analytics.user_events (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    event_type VARCHAR(100),
    event_data JSONB DEFAULT '{}',
    session_id UUID,
    device_info JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Intelligent Error Analysis

```sql
-- Analyze and categorize errors
CREATE OR REPLACE FUNCTION analytics.analyze_error(
    p_error_message TEXT,
    p_stack_trace TEXT
) RETURNS TABLE(
    severity VARCHAR(20),
    category VARCHAR(50),
    root_cause TEXT,
    suggested_fix TEXT
) AS $$
DECLARE
    v_prompt TEXT;
    v_analysis TEXT;
BEGIN
    -- Generate analysis prompt
    v_prompt := format(
        'Analyze this error and provide: 1) Severity (critical/high/medium/low), 2) Category (database/network/logic/user), 3) Root cause, 4) Suggested fix. Error: %s Stack: %s',
        substring(p_error_message, 1, 500),
        substring(p_stack_trace, 1, 500)
    );
    
    v_analysis := steadytext_generate(v_prompt, 200);
    
    -- Parse AI response (simplified - in production use structured generation)
    RETURN QUERY
    SELECT 
        CASE 
            WHEN p_error_message ~* 'fatal|critical|emergency' THEN 'critical'
            WHEN p_error_message ~* 'error|fail' THEN 'high'
            WHEN p_error_message ~* 'warning|warn' THEN 'medium'
            ELSE 'low'
        END,
        CASE 
            WHEN p_error_message ~* 'database|sql|query' THEN 'database'
            WHEN p_error_message ~* 'network|timeout|connection' THEN 'network'
            WHEN p_error_message ~* 'null|undefined|type' THEN 'logic'
            ELSE 'other'
        END,
        COALESCE(
            substring(v_analysis FROM 'Root cause: ([^.]+)'),
            'Error in application logic'
        ),
        COALESCE(
            substring(v_analysis FROM 'Fix: ([^.]+)'),
            v_analysis,
            'Review error context and stack trace'
        );
END;
$$ LANGUAGE plpgsql;

-- Group similar errors
CREATE OR REPLACE FUNCTION analytics.group_similar_errors(
    p_error_message TEXT,
    p_stack_trace TEXT
) RETURNS VARCHAR(64) AS $$
DECLARE
    v_embedding vector(1024);
    v_similar_hash VARCHAR(64);
BEGIN
    -- Generate embedding for error
    v_embedding := steadytext_embed(
        p_error_message || ' ' || COALESCE(substring(p_stack_trace, 1, 500), '')
    );
    
    -- Find similar existing error
    SELECT error_hash INTO v_similar_hash
    FROM analytics.error_logs
    WHERE embedding IS NOT NULL
        AND 1 - (embedding <-> v_embedding) > 0.9
    ORDER BY embedding <-> v_embedding
    LIMIT 1;
    
    -- Return existing hash or generate new one
    RETURN COALESCE(
        v_similar_hash,
        md5(p_error_message || COALESCE(p_stack_trace, ''))
    );
END;
$$ LANGUAGE plpgsql;

-- Process and store errors intelligently
CREATE OR REPLACE FUNCTION analytics.log_error(
    p_error_message TEXT,
    p_stack_trace TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
) RETURNS INTEGER AS $$
DECLARE
    v_error_hash VARCHAR(64);
    v_analysis RECORD;
    v_error_id INTEGER;
BEGIN
    -- Get error hash (groups similar errors)
    v_error_hash := analytics.group_similar_errors(p_error_message, p_stack_trace);
    
    -- Check if error exists
    SELECT id INTO v_error_id
    FROM analytics.error_logs
    WHERE error_hash = v_error_hash;
    
    IF v_error_id IS NOT NULL THEN
        -- Update existing error
        UPDATE analytics.error_logs
        SET occurrence_count = occurrence_count + 1,
            last_seen = NOW()
        WHERE id = v_error_id;
    ELSE
        -- Analyze new error
        SELECT * INTO v_analysis
        FROM analytics.analyze_error(p_error_message, p_stack_trace);
        
        -- Insert new error
        INSERT INTO analytics.error_logs (
            error_hash,
            error_message,
            stack_trace,
            severity,
            ai_analysis,
            suggested_fix,
            embedding
        ) VALUES (
            v_error_hash,
            p_error_message,
            p_stack_trace,
            v_analysis.severity,
            v_analysis.root_cause,
            v_analysis.suggested_fix,
            steadytext_embed(p_error_message || ' ' || COALESCE(p_stack_trace, ''))
        ) RETURNING id INTO v_error_id;
    END IF;
    
    RETURN v_error_id;
END;
$$ LANGUAGE plpgsql;
```

### Anomaly Detection

```sql
-- Detect metric anomalies
CREATE OR REPLACE FUNCTION analytics.detect_anomalies(
    p_metric_name VARCHAR(100),
    p_lookback_hours INTEGER DEFAULT 24
) RETURNS TABLE(
    is_anomaly BOOLEAN,
    severity FLOAT,
    description TEXT,
    explanation TEXT
) AS $$
DECLARE
    v_stats RECORD;
    v_recent_value NUMERIC;
    v_prompt TEXT;
    v_explanation TEXT;
BEGIN
    -- Calculate statistics
    WITH metric_stats AS (
        SELECT 
            AVG(metric_value) as mean_val,
            STDDEV(metric_value) as std_val,
            MIN(metric_value) as min_val,
            MAX(metric_value) as max_val,
            COUNT(*) as sample_count
        FROM analytics.app_metrics
        WHERE metric_name = p_metric_name
            AND created_at > NOW() - INTERVAL '1 hour' * p_lookback_hours
    ),
    recent AS (
        SELECT metric_value
        FROM analytics.app_metrics
        WHERE metric_name = p_metric_name
        ORDER BY created_at DESC
        LIMIT 1
    )
    SELECT 
        ms.*,
        r.metric_value as recent_value
    INTO v_stats
    FROM metric_stats ms, recent r;
    
    -- Check for anomaly
    IF v_stats.sample_count < 10 THEN
        RETURN QUERY SELECT FALSE, 0.0::FLOAT, 'Insufficient data'::TEXT, NULL::TEXT;
        RETURN;
    END IF;
    
    -- Calculate anomaly score
    DECLARE
        v_z_score FLOAT;
        v_is_anomaly BOOLEAN;
        v_severity FLOAT;
    BEGIN
        v_z_score := ABS((v_stats.recent_value - v_stats.mean_val) / NULLIF(v_stats.std_val, 0));
        v_is_anomaly := v_z_score > 3;
        v_severity := LEAST(v_z_score / 5, 1.0);
        
        IF v_is_anomaly THEN
            -- Generate explanation
            v_prompt := format(
                'Explain why metric "%s" with value %s is anomalous. Normal range: %s-%s, average: %s',
                p_metric_name,
                v_stats.recent_value,
                round(v_stats.mean_val - 2 * v_stats.std_val, 2),
                round(v_stats.mean_val + 2 * v_stats.std_val, 2),
                round(v_stats.mean_val, 2)
            );
            
            v_explanation := steadytext_generate(v_prompt, 100);
            
            -- Log anomaly
            INSERT INTO analytics.anomalies (
                metric_name,
                anomaly_type,
                severity,
                description,
                ai_explanation
            ) VALUES (
                p_metric_name,
                CASE 
                    WHEN v_stats.recent_value > v_stats.mean_val THEN 'spike'
                    ELSE 'drop'
                END,
                v_severity,
                format('%s detected: %.2f (normal: %.2f)',
                    CASE WHEN v_stats.recent_value > v_stats.mean_val THEN 'Spike' ELSE 'Drop' END,
                    v_stats.recent_value,
                    v_stats.mean_val
                ),
                v_explanation
            );
        END IF;
        
        RETURN QUERY
        SELECT 
            v_is_anomaly,
            v_severity,
            format('%s: %.2f (z-score: %.2f)', p_metric_name, v_stats.recent_value, v_z_score),
            v_explanation;
    END;
END;
$$ LANGUAGE plpgsql;

-- Batch anomaly detection
CREATE OR REPLACE FUNCTION analytics.detect_all_anomalies()
RETURNS TABLE(
    metric_name VARCHAR(100),
    is_anomaly BOOLEAN,
    severity FLOAT,
    description TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH distinct_metrics AS (
        SELECT DISTINCT metric_name
        FROM analytics.app_metrics
        WHERE created_at > NOW() - INTERVAL '1 hour'
    )
    SELECT 
        dm.metric_name,
        da.is_anomaly,
        da.severity,
        da.description
    FROM distinct_metrics dm
    CROSS JOIN LATERAL analytics.detect_anomalies(dm.metric_name, 24) da
    WHERE da.is_anomaly = TRUE
    ORDER BY da.severity DESC;
END;
$$ LANGUAGE plpgsql;
```

### Performance Analysis

```sql
-- Analyze slow queries/operations
CREATE OR REPLACE FUNCTION analytics.analyze_performance_trace(
    p_operation_name VARCHAR(200),
    p_duration_ms INTEGER,
    p_metadata JSONB DEFAULT '{}'
) RETURNS TEXT AS $$
DECLARE
    v_percentile FLOAT;
    v_prompt TEXT;
    v_analysis TEXT;
BEGIN
    -- Calculate percentile
    SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)
    INTO v_percentile
    FROM analytics.traces
    WHERE operation_name = p_operation_name
        AND created_at > NOW() - INTERVAL '1 hour';
    
    -- Analyze if slow
    IF p_duration_ms > COALESCE(v_percentile, 100) * 2 THEN
        v_prompt := format(
            'Analyze why operation "%s" took %sms (95th percentile: %sms). Context: %s. Suggest optimizations:',
            p_operation_name,
            p_duration_ms,
            round(v_percentile),
            p_metadata::text
        );
        
        v_analysis := steadytext_generate(v_prompt, 150);
        
        RETURN COALESCE(
            v_analysis,
            format('Operation slower than usual. Consider caching or query optimization.')
        );
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Generate performance insights
CREATE OR REPLACE FUNCTION analytics.generate_performance_report(
    p_hours INTEGER DEFAULT 24
) RETURNS TABLE(
    section TEXT,
    insight TEXT,
    recommendation TEXT,
    priority VARCHAR(10)
) AS $$
BEGIN
    -- Slowest operations
    INSERT INTO analytics.temp_insights
    SELECT 
        'Slow Operations',
        format('Operation "%s" averaging %sms (called %s times)',
            operation_name,
            round(AVG(duration_ms)),
            COUNT(*)
        ),
        analytics.analyze_performance_trace(
            operation_name,
            round(AVG(duration_ms))::INTEGER,
            '{}'::jsonb
        ),
        CASE 
            WHEN AVG(duration_ms) > 1000 THEN 'high'
            WHEN AVG(duration_ms) > 500 THEN 'medium'
            ELSE 'low'
        END
    FROM analytics.traces
    WHERE created_at > NOW() - INTERVAL '1 hour' * p_hours
    GROUP BY operation_name
    HAVING AVG(duration_ms) > 100
    ORDER BY AVG(duration_ms) DESC
    LIMIT 5;
    
    -- Error patterns
    INSERT INTO analytics.temp_insights
    SELECT 
        'Error Patterns',
        format('Error "%s" occurred %s times',
            substring(error_message, 1, 50),
            occurrence_count
        ),
        suggested_fix,
        severity
    FROM analytics.error_logs
    WHERE last_seen > NOW() - INTERVAL '1 hour' * p_hours
        AND occurrence_count > 5
    ORDER BY occurrence_count DESC
    LIMIT 5;
    
    -- Resource usage anomalies
    INSERT INTO analytics.temp_insights
    SELECT 
        'Resource Anomalies',
        description,
        ai_explanation,
        CASE 
            WHEN severity > 0.8 THEN 'high'
            WHEN severity > 0.5 THEN 'medium'
            ELSE 'low'
        END
    FROM analytics.anomalies
    WHERE detected_at > NOW() - INTERVAL '1 hour' * p_hours
        AND resolved_at IS NULL
    ORDER BY severity DESC
    LIMIT 5;
    
    RETURN QUERY
    SELECT * FROM analytics.temp_insights
    ORDER BY 
        CASE priority 
            WHEN 'critical' THEN 1
            WHEN 'high' THEN 2
            WHEN 'medium' THEN 3
            ELSE 4
        END;
    
    DROP TABLE analytics.temp_insights;
END;
$$ LANGUAGE plpgsql;
```

### User Behavior Analytics

```sql
-- Analyze user patterns
CREATE OR REPLACE FUNCTION analytics.analyze_user_behavior(
    p_user_id INTEGER,
    p_days INTEGER DEFAULT 7
) RETURNS TABLE(
    metric TEXT,
    value NUMERIC,
    insight TEXT
) AS $$
DECLARE
    v_stats RECORD;
    v_prompt TEXT;
    v_insights TEXT;
BEGIN
    -- Gather user statistics
    WITH user_stats AS (
        SELECT 
            COUNT(*) as total_events,
            COUNT(DISTINCT date_trunc('day', created_at)) as active_days,
            COUNT(DISTINCT session_id) as total_sessions,
            array_agg(DISTINCT event_type) as event_types,
            AVG(EXTRACT(epoch FROM (
                lead(created_at) OVER (PARTITION BY session_id ORDER BY created_at) - created_at
            ))) as avg_time_between_events
        FROM analytics.user_events
        WHERE user_id = p_user_id
            AND created_at > NOW() - INTERVAL '1 day' * p_days
    )
    SELECT * INTO v_stats FROM user_stats;
    
    -- Generate insights
    v_prompt := format(
        'Analyze user behavior: %s events over %s days, %s sessions. Event types: %s. What patterns do you see?',
        v_stats.total_events,
        v_stats.active_days,
        v_stats.total_sessions,
        array_to_string(v_stats.event_types, ', ')
    );
    
    v_insights := steadytext_generate(v_prompt, 100);
    
    RETURN QUERY
    SELECT 'Total Events', v_stats.total_events::NUMERIC, 'Activity level'
    UNION ALL
    SELECT 'Active Days', v_stats.active_days::NUMERIC, 'Engagement frequency'
    UNION ALL
    SELECT 'Sessions', v_stats.total_sessions::NUMERIC, 'Usage pattern'
    UNION ALL
    SELECT 'Avg Session Duration', 
           round(v_stats.avg_time_between_events)::NUMERIC, 
           COALESCE(v_insights, 'Regular user activity');
END;
$$ LANGUAGE plpgsql;

-- Segment users based on behavior
CREATE OR REPLACE FUNCTION analytics.segment_users()
RETURNS TABLE(
    user_id INTEGER,
    segment VARCHAR(50),
    characteristics TEXT[],
    recommendations TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH user_metrics AS (
        SELECT 
            user_id,
            COUNT(*) as event_count,
            COUNT(DISTINCT date_trunc('day', created_at)) as active_days,
            COUNT(DISTINCT event_type) as event_diversity,
            MAX(created_at) as last_active
        FROM analytics.user_events
        WHERE created_at > NOW() - INTERVAL '30 days'
        GROUP BY user_id
    ),
    user_segments AS (
        SELECT 
            user_id,
            CASE 
                WHEN active_days >= 25 AND event_count > 500 THEN 'power_user'
                WHEN active_days >= 15 AND event_count > 100 THEN 'regular_user'
                WHEN active_days >= 5 THEN 'casual_user'
                WHEN last_active < NOW() - INTERVAL '14 days' THEN 'churning_user'
                ELSE 'new_user'
            END as segment,
            ARRAY[
                format('%s events', event_count),
                format('%s active days', active_days),
                format('%s event types', event_diversity)
            ] as characteristics
        FROM user_metrics
    )
    SELECT 
        us.user_id,
        us.segment,
        us.characteristics,
        CASE us.segment
            WHEN 'power_user' THEN 'Offer premium features and early access'
            WHEN 'regular_user' THEN 'Encourage deeper feature adoption'
            WHEN 'casual_user' THEN 'Send engagement campaigns'
            WHEN 'churning_user' THEN 'Re-engagement campaign needed'
            ELSE 'Onboarding and education'
        END as recommendations
    FROM user_segments us;
END;
$$ LANGUAGE plpgsql;
```

### Predictive Analytics

```sql
-- Predict metric values
CREATE OR REPLACE FUNCTION analytics.predict_metric_value(
    p_metric_name VARCHAR(100),
    p_hours_ahead INTEGER DEFAULT 1
) RETURNS TABLE(
    predicted_value NUMERIC,
    confidence_interval NUMERIC[],
    trend TEXT,
    factors TEXT[]
) AS $$
DECLARE
    v_recent_data RECORD;
    v_prompt TEXT;
    v_prediction TEXT;
BEGIN
    -- Analyze recent trends
    WITH trend_analysis AS (
        SELECT 
            AVG(metric_value) as avg_value,
            STDDEV(metric_value) as std_value,
            regr_slope(metric_value, extract(epoch from created_at)) as trend_slope,
            COUNT(*) as data_points
        FROM analytics.app_metrics
        WHERE metric_name = p_metric_name
            AND created_at > NOW() - INTERVAL '24 hours'
    )
    SELECT * INTO v_recent_data FROM trend_analysis;
    
    -- Simple prediction (in production, use proper time series models)
    RETURN QUERY
    SELECT 
        v_recent_data.avg_value + (v_recent_data.trend_slope * p_hours_ahead * 3600),
        ARRAY[
            v_recent_data.avg_value - 2 * v_recent_data.std_value,
            v_recent_data.avg_value + 2 * v_recent_data.std_value
        ],
        CASE 
            WHEN v_recent_data.trend_slope > 0.01 THEN 'increasing'
            WHEN v_recent_data.trend_slope < -0.01 THEN 'decreasing'
            ELSE 'stable'
        END,
        ARRAY[
            format('Based on %s data points', v_recent_data.data_points),
            format('Trend: %s', 
                CASE 
                    WHEN v_recent_data.trend_slope > 0 THEN 'upward'
                    ELSE 'downward'
                END
            )
        ];
END;
$$ LANGUAGE plpgsql;

-- Predict system failures
CREATE OR REPLACE FUNCTION analytics.predict_failure_risk()
RETURNS TABLE(
    component TEXT,
    risk_score FLOAT,
    predicted_failure_time TIMESTAMP,
    prevention_steps TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH error_trends AS (
        SELECT 
            substring(error_message from '^([^:]+)') as component,
            COUNT(*) as error_count,
            MAX(last_seen) as last_error,
            AVG(EXTRACT(epoch FROM (last_seen - first_seen))) as error_frequency
        FROM analytics.error_logs
        WHERE last_seen > NOW() - INTERVAL '7 days'
        GROUP BY substring(error_message from '^([^:]+)')
    ),
    performance_degradation AS (
        SELECT 
            operation_name as component,
            regr_slope(duration_ms, extract(epoch from created_at)) as perf_slope
        FROM analytics.traces
        WHERE created_at > NOW() - INTERVAL '7 days'
        GROUP BY operation_name
        HAVING regr_slope(duration_ms, extract(epoch from created_at)) > 0.1
    )
    SELECT 
        COALESCE(et.component, pd.component),
        LEAST(
            (COALESCE(et.error_count, 0)::FLOAT / 100) +
            (CASE WHEN pd.perf_slope > 0 THEN pd.perf_slope ELSE 0 END),
            1.0
        ) as risk_score,
        CASE 
            WHEN et.error_frequency IS NOT NULL 
            THEN NOW() + (et.error_frequency || ' seconds')::INTERVAL
            ELSE NOW() + INTERVAL '7 days'
        END as predicted_failure_time,
        ARRAY[
            CASE 
                WHEN et.error_count > 10 THEN 'Review and fix recurring errors'
                ELSE NULL
            END,
            CASE 
                WHEN pd.perf_slope > 0 THEN 'Optimize performance bottlenecks'
                ELSE NULL
            END
        ]
    FROM error_trends et
    FULL OUTER JOIN performance_degradation pd ON et.component = pd.component
    WHERE COALESCE(et.error_count, 0) > 5 
       OR COALESCE(pd.perf_slope, 0) > 0.1
    ORDER BY risk_score DESC;
END;
$$ LANGUAGE plpgsql;
```

### Dashboard and Reporting

```sql
-- Real-time dashboard data
CREATE OR REPLACE FUNCTION analytics.get_dashboard_metrics()
RETURNS TABLE(
    metric_category TEXT,
    metric_name TEXT,
    current_value NUMERIC,
    change_percent NUMERIC,
    status TEXT,
    mini_chart JSONB
) AS $$
BEGIN
    -- System health metrics
    RETURN QUERY
    WITH current_window AS (
        SELECT 
            metric_name,
            AVG(metric_value) as current_avg,
            array_agg(
                json_build_object(
                    'time', extract(epoch from created_at),
                    'value', metric_value
                ) ORDER BY created_at
            ) as chart_data
        FROM analytics.app_metrics
        WHERE created_at > NOW() - INTERVAL '1 hour'
        GROUP BY metric_name
    ),
    previous_window AS (
        SELECT 
            metric_name,
            AVG(metric_value) as previous_avg
        FROM analytics.app_metrics
        WHERE created_at > NOW() - INTERVAL '2 hours'
            AND created_at <= NOW() - INTERVAL '1 hour'
        GROUP BY metric_name
    )
    SELECT 
        'System Health',
        cw.metric_name,
        round(cw.current_avg, 2),
        round(((cw.current_avg - pw.previous_avg) / NULLIF(pw.previous_avg, 0)) * 100, 1),
        CASE 
            WHEN cw.current_avg > pw.previous_avg * 1.2 THEN 'warning'
            WHEN cw.current_avg < pw.previous_avg * 0.8 THEN 'warning'
            ELSE 'normal'
        END,
        to_jsonb(cw.chart_data)
    FROM current_window cw
    LEFT JOIN previous_window pw ON cw.metric_name = pw.metric_name
    
    UNION ALL
    
    -- Error metrics
    SELECT 
        'Errors',
        'Error Rate',
        COUNT(*)::NUMERIC,
        0,
        CASE 
            WHEN COUNT(*) > 100 THEN 'critical'
            WHEN COUNT(*) > 50 THEN 'warning'
            ELSE 'normal'
        END,
        '{}'::jsonb
    FROM analytics.error_logs
    WHERE last_seen > NOW() - INTERVAL '1 hour'
    
    UNION ALL
    
    -- Performance metrics
    SELECT 
        'Performance',
        'Avg Response Time',
        round(AVG(duration_ms))::NUMERIC,
        0,
        CASE 
            WHEN AVG(duration_ms) > 1000 THEN 'critical'
            WHEN AVG(duration_ms) > 500 THEN 'warning'
            ELSE 'normal'
        END,
        '{}'::jsonb
    FROM analytics.traces
    WHERE created_at > NOW() - INTERVAL '1 hour';
END;
$$ LANGUAGE plpgsql;

-- Generate executive summary
CREATE OR REPLACE FUNCTION analytics.generate_executive_summary(
    p_period_days INTEGER DEFAULT 7
) RETURNS TEXT AS $$
DECLARE
    v_stats RECORD;
    v_prompt TEXT;
    v_summary TEXT;
BEGIN
    -- Gather key statistics
    WITH summary_stats AS (
        SELECT 
            (SELECT COUNT(*) FROM analytics.app_metrics 
             WHERE created_at > NOW() - INTERVAL '1 day' * p_period_days) as total_metrics,
            (SELECT COUNT(*) FROM analytics.error_logs 
             WHERE last_seen > NOW() - INTERVAL '1 day' * p_period_days) as total_errors,
            (SELECT COUNT(*) FROM analytics.anomalies 
             WHERE detected_at > NOW() - INTERVAL '1 day' * p_period_days) as total_anomalies,
            (SELECT AVG(duration_ms) FROM analytics.traces 
             WHERE created_at > NOW() - INTERVAL '1 day' * p_period_days) as avg_performance,
            (SELECT COUNT(DISTINCT user_id) FROM analytics.user_events 
             WHERE created_at > NOW() - INTERVAL '1 day' * p_period_days) as active_users
    )
    SELECT * INTO v_stats FROM summary_stats;
    
    -- Generate AI summary
    v_prompt := format(
        'Write an executive summary for the past %s days: %s metrics collected, %s errors, %s anomalies detected, %sms avg response time, %s active users. Highlight key insights and recommendations.',
        p_period_days,
        v_stats.total_metrics,
        v_stats.total_errors,
        v_stats.total_anomalies,
        round(v_stats.avg_performance),
        v_stats.active_users
    );
    
    v_summary := steadytext_generate(v_prompt, 200);
    
    RETURN COALESCE(
        v_summary,
        format('System performance over the past %s days: %s errors logged, %s anomalies detected. Average response time: %sms.',
            p_period_days,
            v_stats.total_errors,
            v_stats.total_anomalies,
            round(v_stats.avg_performance)
        )
    );
END;
$$ LANGUAGE plpgsql;
```

### Alert Configuration

```sql
-- Alert rules table
CREATE TABLE analytics.alert_rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) UNIQUE NOT NULL,
    metric_name VARCHAR(100),
    condition_type VARCHAR(20), -- threshold, rate, pattern
    condition_value JSONB,
    severity VARCHAR(20),
    notification_channels TEXT[],
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluate alert conditions
CREATE OR REPLACE FUNCTION analytics.evaluate_alerts()
RETURNS TABLE(
    alert_id INTEGER,
    rule_name VARCHAR(100),
    severity VARCHAR(20),
    message TEXT,
    recommendation TEXT
) AS $$
DECLARE
    v_rule RECORD;
    v_should_alert BOOLEAN;
    v_message TEXT;
BEGIN
    FOR v_rule IN 
        SELECT * FROM analytics.alert_rules WHERE is_active = TRUE
    LOOP
        v_should_alert := FALSE;
        
        -- Evaluate based on condition type
        CASE v_rule.condition_type
            WHEN 'threshold' THEN
                SELECT metric_value > (v_rule.condition_value->>'threshold')::NUMERIC
                INTO v_should_alert
                FROM analytics.app_metrics
                WHERE metric_name = v_rule.metric_name
                ORDER BY created_at DESC
                LIMIT 1;
                
            WHEN 'rate' THEN
                SELECT COUNT(*) > (v_rule.condition_value->>'count')::INTEGER
                INTO v_should_alert
                FROM analytics.app_metrics
                WHERE metric_name = v_rule.metric_name
                    AND created_at > NOW() - ((v_rule.condition_value->>'window')::TEXT)::INTERVAL;
                
            WHEN 'pattern' THEN
                -- Use AI to detect complex patterns
                v_message := analytics.detect_pattern(
                    v_rule.metric_name,
                    v_rule.condition_value->>'pattern'
                );
                v_should_alert := v_message IS NOT NULL;
        END CASE;
        
        IF v_should_alert THEN
            RETURN QUERY
            SELECT 
                v_rule.id,
                v_rule.rule_name,
                v_rule.severity,
                COALESCE(v_message, format('Alert: %s condition met', v_rule.rule_name)),
                steadytext_generate(
                    format('Provide recommendation for alert: %s', v_rule.rule_name),
                    50
                );
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

## Related Documentation

- [PostgreSQL Extension Overview](../postgresql-extension.md)
- [Search Examples](postgresql-search.md)
- [Real-time Examples](postgresql-realtime.md)
- [E-commerce Examples](postgresql-ecommerce.md)
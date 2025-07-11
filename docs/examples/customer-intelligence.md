# Customer Intelligence with AI-Powered Analytics

Transform raw customer data into actionable insights using SteadyText's AI capabilities directly in PostgreSQL.

## Overview

This tutorial demonstrates how to build a comprehensive customer intelligence system that:
- Analyzes customer feedback at scale
- Tracks sentiment trends over time
- Identifies churn signals automatically
- Creates customer segment profiles
- Generates personalized recommendations

## Prerequisites

```bash
# Start PostgreSQL with SteadyText
docker run -d -p 5432:5432 --name steadytext-intel julep/pg-steadytext

# Connect and enable extensions
psql -h localhost -U postgres -c "CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;"
psql -h localhost -U postgres -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"
```

## Database Schema

Create a comprehensive customer intelligence schema:

```sql
-- Customers table
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    customer_id UUID DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    segment VARCHAR(50),
    lifetime_value DECIMAL(10, 2) DEFAULT 0,
    acquisition_date DATE,
    last_active_date DATE,
    churn_risk_score DECIMAL(3, 2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Customer interactions
CREATE TABLE customer_interactions (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    interaction_type VARCHAR(50), -- 'support', 'purchase', 'review', 'email', 'chat'
    channel VARCHAR(50), -- 'web', 'mobile', 'email', 'phone'
    content TEXT,
    metadata JSONB,
    sentiment_score DECIMAL(3, 2), -- -1 to 1
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Product reviews
CREATE TABLE product_reviews (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    product_id INTEGER,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    title VARCHAR(200),
    review_text TEXT,
    verified_purchase BOOLEAN DEFAULT FALSE,
    helpful_count INTEGER DEFAULT 0,
    ai_summary TEXT,
    sentiment VARCHAR(20),
    key_themes TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Support tickets
CREATE TABLE support_tickets (
    id SERIAL PRIMARY KEY,
    ticket_number VARCHAR(20) UNIQUE,
    customer_id INTEGER REFERENCES customers(id),
    category VARCHAR(50),
    priority VARCHAR(20),
    subject VARCHAR(200),
    description TEXT,
    resolution TEXT,
    satisfaction_score INTEGER,
    ai_analysis JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

-- Customer segments with AI insights
CREATE TABLE customer_segments_analysis (
    id SERIAL PRIMARY KEY,
    segment_name VARCHAR(50) UNIQUE,
    customer_count INTEGER,
    avg_lifetime_value DECIMAL(10, 2),
    common_behaviors TEXT[],
    ai_profile TEXT,
    recommendations JSONB,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_interactions_customer ON customer_interactions(customer_id, timestamp);
CREATE INDEX idx_reviews_customer ON product_reviews(customer_id);
CREATE INDEX idx_reviews_sentiment ON product_reviews(sentiment);
CREATE INDEX idx_tickets_customer ON support_tickets(customer_id);
CREATE INDEX idx_customers_segment ON customers(segment);
CREATE INDEX idx_customers_churn ON customers(churn_risk_score);
```

## Real-Time Review Analysis

Automatically analyze customer reviews as they come in:

```sql
-- Trigger function to analyze reviews
CREATE OR REPLACE FUNCTION analyze_review_on_insert()
RETURNS TRIGGER AS $$
DECLARE
    v_sentiment VARCHAR;
    v_summary TEXT;
    v_themes TEXT[];
    v_sentiment_score DECIMAL(3, 2);
BEGIN
    -- Determine sentiment
    v_sentiment := steadytext_generate_choice(
        format('Classify sentiment of this review (rating %s/5): %s',
            NEW.rating, NEW.review_text),
        ARRAY['very_positive', 'positive', 'neutral', 'negative', 'very_negative']
    );
    
    -- Generate summary
    v_summary := steadytext_generate(
        format('Summarize this customer review in one sentence: %s',
            NEW.review_text),
        max_tokens := 50
    );
    
    -- Extract key themes
    v_themes := string_to_array(
        steadytext_generate(
            format('List 3 key themes from this review (comma-separated): %s',
                NEW.review_text),
            max_tokens := 30
        ),
        ', '
    );
    
    -- Calculate numeric sentiment score
    v_sentiment_score := CASE v_sentiment
        WHEN 'very_positive' THEN 1.0
        WHEN 'positive' THEN 0.5
        WHEN 'neutral' THEN 0.0
        WHEN 'negative' THEN -0.5
        WHEN 'very_negative' THEN -1.0
    END;
    
    -- Update the review record
    NEW.sentiment := v_sentiment;
    NEW.ai_summary := v_summary;
    NEW.key_themes := v_themes;
    
    -- Also log this as an interaction
    INSERT INTO customer_interactions (
        customer_id, interaction_type, channel, 
        content, sentiment_score, metadata
    ) VALUES (
        NEW.customer_id, 'review', 'web',
        NEW.review_text, v_sentiment_score,
        jsonb_build_object(
            'rating', NEW.rating,
            'product_id', NEW.product_id,
            'themes', v_themes
        )
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create the trigger
CREATE TRIGGER analyze_review_before_insert
    BEFORE INSERT ON product_reviews
    FOR EACH ROW
    EXECUTE FUNCTION analyze_review_on_insert();
```

## Customer Sentiment Tracking

Track sentiment trends over time:

```sql
-- Customer sentiment dashboard view
CREATE OR REPLACE VIEW customer_sentiment_dashboard AS
WITH sentiment_data AS (
    SELECT 
        c.id,
        c.email,
        c.segment,
        AVG(ci.sentiment_score) AS avg_sentiment,
        COUNT(ci.id) AS interaction_count,
        MAX(ci.timestamp) AS last_interaction,
        array_agg(DISTINCT ci.interaction_type) AS interaction_types
    FROM customers c
    LEFT JOIN customer_interactions ci ON c.id = ci.customer_id
    WHERE ci.timestamp > NOW() - INTERVAL '90 days'
    GROUP BY c.id, c.email, c.segment
),
recent_issues AS (
    SELECT 
        customer_id,
        COUNT(*) AS issue_count,
        AVG(CASE WHEN satisfaction_score IS NOT NULL 
            THEN satisfaction_score ELSE 3 END) AS avg_satisfaction
    FROM support_tickets
    WHERE created_at > NOW() - INTERVAL '30 days'
    GROUP BY customer_id
)
SELECT 
    sd.*,
    ri.issue_count,
    ri.avg_satisfaction,
    CASE 
        WHEN sd.avg_sentiment < -0.3 AND ri.issue_count > 2 THEN 'high_risk'
        WHEN sd.avg_sentiment < 0 OR ri.issue_count > 3 THEN 'medium_risk'
        WHEN sd.avg_sentiment > 0.5 AND ri.issue_count = 0 THEN 'loyal'
        ELSE 'normal'
    END AS customer_status,
    steadytext_generate(
        format('Analyze customer behavior: Sentiment: %s, Interactions: %s, Issues: %s',
            ROUND(sd.avg_sentiment, 2),
            sd.interaction_count,
            COALESCE(ri.issue_count, 0)
        ),
        max_tokens := 100
    ) AS ai_insights
FROM sentiment_data sd
LEFT JOIN recent_issues ri ON sd.id = ri.customer_id;
```

## Churn Prediction System

Identify customers at risk of churning:

```sql
-- Churn risk calculation function
CREATE OR REPLACE FUNCTION calculate_churn_risk()
RETURNS TABLE (
    customer_id INTEGER,
    risk_score DECIMAL(3, 2),
    risk_factors JSONB,
    retention_strategy TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH customer_metrics AS (
        SELECT 
            c.id,
            c.last_active_date,
            c.lifetime_value,
            COUNT(DISTINCT ci.id) AS interaction_count,
            AVG(ci.sentiment_score) AS avg_sentiment,
            MAX(ci.timestamp) AS last_interaction,
            COUNT(DISTINCT st.id) AS support_tickets,
            AVG(st.satisfaction_score) AS avg_satisfaction
        FROM customers c
        LEFT JOIN customer_interactions ci ON c.id = ci.customer_id
            AND ci.timestamp > NOW() - INTERVAL '90 days'
        LEFT JOIN support_tickets st ON c.id = st.customer_id
            AND st.created_at > NOW() - INTERVAL '90 days'
        GROUP BY c.id, c.last_active_date, c.lifetime_value
    ),
    risk_scoring AS (
        SELECT 
            id,
            -- Calculate risk score based on multiple factors
            LEAST(1.0, GREATEST(0.0,
                0.3 * (EXTRACT(EPOCH FROM (NOW() - last_active_date)) / 86400.0 / 30.0) + -- Days inactive
                0.2 * (1.0 - COALESCE(avg_sentiment + 1, 1.0) / 2.0) + -- Sentiment
                0.2 * (support_tickets::FLOAT / GREATEST(interaction_count, 1)) + -- Support ratio
                0.3 * (CASE WHEN avg_satisfaction < 3 THEN 1.0 ELSE 0.0 END) -- Low satisfaction
            )) AS risk_score,
            jsonb_build_object(
                'days_inactive', EXTRACT(EPOCH FROM (NOW() - last_active_date)) / 86400.0,
                'sentiment_score', COALESCE(avg_sentiment, 0),
                'support_tickets', support_tickets,
                'satisfaction', COALESCE(avg_satisfaction, 3),
                'lifetime_value', lifetime_value
            ) AS risk_factors
        FROM customer_metrics
    )
    SELECT 
        rs.id,
        rs.risk_score,
        rs.risk_factors,
        steadytext_generate(
            format('Create retention strategy for customer with risk score %s and factors: %s',
                ROUND(rs.risk_score, 2),
                rs.risk_factors::TEXT
            ),
            max_tokens := 150
        ) AS retention_strategy
    FROM risk_scoring rs
    WHERE rs.risk_score > 0.3;
    
    -- Update customer records
    UPDATE customers c
    SET churn_risk_score = rs.risk_score,
        updated_at = NOW()
    FROM risk_scoring rs
    WHERE c.id = rs.id;
END;
$$ LANGUAGE plpgsql;
```

## Customer Segment Analysis

Generate AI-powered insights for customer segments:

```sql
-- Analyze and profile customer segments
CREATE OR REPLACE FUNCTION analyze_customer_segments()
RETURNS VOID AS $$
DECLARE
    v_segment RECORD;
BEGIN
    -- Clear previous analysis
    TRUNCATE customer_segments_analysis;
    
    -- Analyze each segment
    FOR v_segment IN 
        SELECT DISTINCT segment FROM customers WHERE segment IS NOT NULL
    LOOP
        INSERT INTO customer_segments_analysis (
            segment_name,
            customer_count,
            avg_lifetime_value,
            common_behaviors,
            ai_profile,
            recommendations
        )
        WITH segment_data AS (
            SELECT 
                COUNT(DISTINCT c.id) AS customer_count,
                AVG(c.lifetime_value) AS avg_ltv,
                array_agg(DISTINCT ci.interaction_type) AS interaction_types,
                AVG(ci.sentiment_score) AS avg_sentiment,
                COUNT(DISTINCT pr.id) AS review_count,
                AVG(pr.rating) AS avg_rating
            FROM customers c
            LEFT JOIN customer_interactions ci ON c.id = ci.customer_id
            LEFT JOIN product_reviews pr ON c.id = pr.customer_id
            WHERE c.segment = v_segment.segment
        ),
        behavior_analysis AS (
            SELECT 
                array_agg(DISTINCT theme) AS common_themes
            FROM (
                SELECT unnest(key_themes) AS theme
                FROM product_reviews pr
                JOIN customers c ON pr.customer_id = c.id
                WHERE c.segment = v_segment.segment
            ) t
        )
        SELECT 
            v_segment.segment,
            sd.customer_count,
            sd.avg_ltv,
            sd.interaction_types,
            steadytext_generate(
                format('Create detailed profile for %s customer segment with %s customers, $%s avg LTV, %s sentiment',
                    v_segment.segment,
                    sd.customer_count,
                    ROUND(sd.avg_ltv, 2),
                    CASE 
                        WHEN sd.avg_sentiment > 0.5 THEN 'very positive'
                        WHEN sd.avg_sentiment > 0 THEN 'positive'
                        WHEN sd.avg_sentiment > -0.5 THEN 'neutral'
                        ELSE 'negative'
                    END
                ),
                max_tokens := 200
            ) AS ai_profile,
            steadytext_generate_json(
                format('Suggest 3 marketing strategies for %s segment', v_segment.segment),
                '{
                    "strategies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "expected_impact": {"type": "string"}
                            }
                        }
                    }
                }'::json
            )::jsonb AS recommendations
        FROM segment_data sd, behavior_analysis ba;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

## Support Ticket Intelligence

Extract insights from support interactions:

```sql
-- Analyze support tickets with AI
CREATE OR REPLACE FUNCTION analyze_support_ticket(
    p_ticket_id INTEGER
)
RETURNS VOID AS $$
DECLARE
    v_ticket support_tickets%ROWTYPE;
    v_analysis JSONB;
BEGIN
    SELECT * INTO v_ticket FROM support_tickets WHERE id = p_ticket_id;
    
    -- Generate comprehensive analysis
    v_analysis := steadytext_generate_json(
        format('Analyze support ticket: Subject: %s, Description: %s, Category: %s',
            v_ticket.subject,
            v_ticket.description,
            v_ticket.category
        ),
        '{
            "urgency": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "sentiment": {"type": "string", "enum": ["angry", "frustrated", "neutral", "satisfied", "happy"]},
            "root_cause": {"type": "string"},
            "suggested_resolution": {"type": "string"},
            "follow_up_needed": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}}
        }'::json
    )::jsonb;
    
    -- Update ticket with analysis
    UPDATE support_tickets
    SET ai_analysis = v_analysis,
        priority = COALESCE(priority, v_analysis->>'urgency')
    WHERE id = p_ticket_id;
    
    -- Log as interaction
    INSERT INTO customer_interactions (
        customer_id, interaction_type, channel,
        content, sentiment_score, metadata
    ) VALUES (
        v_ticket.customer_id,
        'support',
        'ticket',
        v_ticket.description,
        CASE v_analysis->>'sentiment'
            WHEN 'angry' THEN -1.0
            WHEN 'frustrated' THEN -0.5
            WHEN 'neutral' THEN 0.0
            WHEN 'satisfied' THEN 0.5
            WHEN 'happy' THEN 1.0
        END,
        v_analysis
    );
END;
$$ LANGUAGE plpgsql;
```

## Personalized Recommendations

Generate personalized product recommendations:

```sql
-- Generate personalized recommendations
CREATE OR REPLACE FUNCTION generate_customer_recommendations(
    p_customer_id INTEGER,
    p_num_recommendations INTEGER DEFAULT 5
)
RETURNS TABLE (
    recommendation_type VARCHAR,
    title VARCHAR,
    description TEXT,
    priority INTEGER
) AS $$
DECLARE
    v_customer RECORD;
    v_profile TEXT;
BEGIN
    -- Get customer profile
    SELECT 
        c.*,
        cs.ai_profile,
        array_agg(DISTINCT pr.key_themes) AS interests,
        AVG(pr.rating) AS avg_rating_given
    INTO v_customer
    FROM customers c
    LEFT JOIN customer_segments_analysis cs ON c.segment = cs.segment_name
    LEFT JOIN product_reviews pr ON c.id = pr.customer_id
    WHERE c.id = p_customer_id
    GROUP BY c.id, c.customer_id, c.email, c.first_name, c.last_name, 
             c.segment, c.lifetime_value, c.acquisition_date, c.last_active_date,
             c.churn_risk_score, c.created_at, c.updated_at, cs.ai_profile;
    
    -- Build customer profile for AI
    v_profile := format('Customer: %s %s, Segment: %s, LTV: $%s, Risk: %s, Interests: %s',
        v_customer.first_name,
        v_customer.last_name,
        v_customer.segment,
        v_customer.lifetime_value,
        COALESCE(v_customer.churn_risk_score, 0),
        array_to_string(v_customer.interests, ', ')
    );
    
    -- Generate recommendations
    RETURN QUERY
    WITH recommendations AS (
        SELECT 
            'product' AS rec_type,
            steadytext_generate(
                format('Suggest product for: %s', v_profile),
                max_tokens := 50
            ) AS title,
            steadytext_generate(
                format('Why this product is perfect for: %s', v_profile),
                max_tokens := 100
            ) AS description,
            1 AS priority
        UNION ALL
        SELECT 
            'retention' AS rec_type,
            steadytext_generate(
                format('Create retention offer for: %s', v_profile),
                max_tokens := 50
            ) AS title,
            steadytext_generate(
                format('Explain retention offer benefits for: %s', v_profile),
                max_tokens := 100
            ) AS description,
            CASE WHEN v_customer.churn_risk_score > 0.5 THEN 1 ELSE 2 END AS priority
        UNION ALL
        SELECT 
            'upsell' AS rec_type,
            steadytext_generate(
                format('Suggest upsell opportunity for: %s', v_profile),
                max_tokens := 50
            ) AS title,
            steadytext_generate(
                format('Upsell pitch for: %s', v_profile),
                max_tokens := 100
            ) AS description,
            3 AS priority
    )
    SELECT * FROM recommendations
    ORDER BY priority
    LIMIT p_num_recommendations;
END;
$$ LANGUAGE plpgsql;
```

## Sample Data and Dashboards

```sql
-- Insert sample customers
INSERT INTO customers (email, first_name, last_name, segment, lifetime_value, acquisition_date)
VALUES 
    ('john.doe@email.com', 'John', 'Doe', 'premium', 2500.00, '2023-01-15'),
    ('jane.smith@email.com', 'Jane', 'Smith', 'regular', 450.00, '2023-06-20'),
    ('bob.wilson@email.com', 'Bob', 'Wilson', 'budget', 150.00, '2024-01-10');

-- Insert sample reviews
INSERT INTO product_reviews (customer_id, product_id, rating, title, review_text)
VALUES
    (1, 101, 5, 'Excellent product!', 'This product exceeded my expectations. The quality is outstanding and the customer service was impeccable. Would definitely recommend to friends and family.'),
    (2, 102, 3, 'Decent but could be better', 'The product works as advertised but the shipping took forever and the packaging was damaged. The product itself is okay for the price.'),
    (3, 103, 1, 'Very disappointed', 'Product broke after just one week of use. Customer support was unhelpful and refused to honor the warranty. Will not buy from this company again.');

-- Insert sample support tickets
INSERT INTO support_tickets (ticket_number, customer_id, category, subject, description)
VALUES
    ('TICK-001', 2, 'shipping', 'Late delivery', 'My order was supposed to arrive last week but still hasnt been delivered. Tracking shows no updates.'),
    ('TICK-002', 3, 'product_issue', 'Product defect', 'The product stopped working after one week. It wont turn on anymore despite following all troubleshooting steps.');

-- Run analysis
SELECT calculate_churn_risk();
SELECT analyze_customer_segments();

-- View insights
SELECT * FROM customer_sentiment_dashboard;
SELECT * FROM generate_customer_recommendations(1);
```

## Executive Dashboard Query

```sql
-- Executive customer intelligence summary
CREATE OR REPLACE VIEW executive_customer_summary AS
WITH summary_stats AS (
    SELECT 
        COUNT(DISTINCT c.id) AS total_customers,
        COUNT(DISTINCT c.id) FILTER (WHERE c.churn_risk_score > 0.7) AS high_risk_customers,
        AVG(c.lifetime_value) AS avg_ltv,
        COUNT(DISTINCT pr.id) AS total_reviews,
        AVG(pr.rating) AS avg_rating,
        COUNT(DISTINCT st.id) AS open_tickets
    FROM customers c
    LEFT JOIN product_reviews pr ON c.id = pr.customer_id
    LEFT JOIN support_tickets st ON c.id = st.customer_id 
        AND st.resolved_at IS NULL
)
SELECT 
    *,
    steadytext_generate(
        format('Executive summary: %s customers, %s at high risk, $%s avg LTV, %s rating, %s open tickets',
            total_customers,
            high_risk_customers,
            ROUND(avg_ltv, 2),
            ROUND(avg_rating, 1),
            open_tickets
        ),
        max_tokens := 200
    ) AS executive_insights
FROM summary_stats;
```

## Best Practices

1. **Privacy First**: Always anonymize data in AI prompts
2. **Batch Processing**: Use background jobs for large-scale analysis
3. **Caching Strategy**: Leverage SteadyText's caching for repeated analyses
4. **Feedback Loop**: Use AI insights to improve models over time
5. **Human Review**: Always have humans validate critical decisions

## Next Steps

- [Data Pipelines Example →](data-pipelines.md)
- [TimescaleDB Integration →](../integrations/timescaledb.md)
- [Production Deployment →](../deployment/production.md)

---

!!! tip "Pro Tip"
    Combine customer intelligence with TimescaleDB continuous aggregates for real-time dashboards that update automatically as new data arrives.
# PostgreSQL Examples: E-commerce

Examples for building e-commerce platforms with AI-powered features using SteadyText.

## E-commerce Product Catalog

### Schema Design

```sql
-- Create e-commerce schema
CREATE SCHEMA IF NOT EXISTS ecommerce;

-- Products table with AI fields
CREATE TABLE ecommerce.products (
    id SERIAL PRIMARY KEY,
    sku VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    ai_description TEXT,
    features TEXT[],
    price DECIMAL(10, 2) NOT NULL,
    category_id INTEGER,
    brand VARCHAR(100),
    embedding vector(1024),
    search_vector tsvector,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Categories with hierarchical structure
CREATE TABLE ecommerce.categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES ecommerce.categories(id),
    description TEXT,
    embedding vector(1024),
    path ltree,
    UNIQUE(parent_id, name)
);

-- Customer profiles
CREATE TABLE ecommerce.customers (
    id SERIAL PRIMARY KEY,
    email VARCHAR(200) UNIQUE NOT NULL,
    preferences JSONB DEFAULT '{}',
    preference_embedding vector(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reviews with sentiment
CREATE TABLE ecommerce.reviews (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES ecommerce.products(id),
    customer_id INTEGER REFERENCES ecommerce.customers(id),
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    title VARCHAR(200),
    content TEXT,
    sentiment_score FLOAT,
    helpful_count INTEGER DEFAULT 0,
    verified_purchase BOOLEAN DEFAULT FALSE,
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Shopping behavior tracking
CREATE TABLE ecommerce.customer_events (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES ecommerce.customers(id),
    event_type VARCHAR(50),
    product_id INTEGER REFERENCES ecommerce.products(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Product Description Generation

```sql
-- Generate engaging product descriptions
CREATE OR REPLACE FUNCTION ecommerce.generate_product_description(
    p_name TEXT,
    p_features TEXT[],
    p_brand TEXT,
    p_category TEXT,
    p_style TEXT DEFAULT 'professional'
) RETURNS TEXT AS $$
DECLARE
    v_prompt TEXT;
    v_description TEXT;
    v_features_text TEXT;
BEGIN
    v_features_text := array_to_string(p_features, ', ');
    
    v_prompt := format(
        'Write a %s product description for: %s by %s. Category: %s. Features: %s. Make it engaging and highlight benefits.',
        p_style,
        p_name,
        COALESCE(p_brand, 'our brand'),
        p_category,
        v_features_text
    );
    
    v_description := steadytext_generate(v_prompt, 200);
    
    IF v_description IS NULL THEN
        -- Fallback to template-based description
        v_description := format(
            'Introducing the %s from %s. This %s features %s. Shop now for the best selection.',
            p_name,
            COALESCE(p_brand, 'our collection'),
            p_category,
            v_features_text
        );
    END IF;
    
    RETURN v_description;
END;
$$ LANGUAGE plpgsql;

-- Generate product features from description
CREATE OR REPLACE FUNCTION ecommerce.extract_product_features(
    p_description TEXT,
    p_max_features INTEGER DEFAULT 5
) RETURNS TEXT[] AS $$
DECLARE
    v_prompt TEXT;
    v_features_text TEXT;
    v_features TEXT[];
BEGIN
    v_prompt := format(
        'Extract %s key features from this product description as a comma-separated list: %s',
        p_max_features,
        p_description
    );
    
    v_features_text := steadytext_generate(v_prompt, 100);
    
    IF v_features_text IS NOT NULL THEN
        v_features := string_to_array(
            regexp_replace(v_features_text, '^\s*[-•*]?\s*', '', 'gm'),
            E'\n'
        );
        -- Clean up array
        v_features := array_remove(v_features, '');
        v_features := array_remove(v_features, NULL);
    ELSE
        v_features := ARRAY[]::TEXT[];
    END IF;
    
    RETURN v_features[1:p_max_features];
END;
$$ LANGUAGE plpgsql;

-- Generate SEO-optimized product titles
CREATE OR REPLACE FUNCTION ecommerce.optimize_product_title(
    p_name TEXT,
    p_brand TEXT,
    p_category TEXT,
    p_key_features TEXT[]
) RETURNS TEXT AS $$
DECLARE
    v_prompt TEXT;
    v_title TEXT;
BEGIN
    v_prompt := format(
        'Create an SEO-optimized product title (max 60 chars) for: %s %s in %s category with features: %s',
        p_brand,
        p_name,
        p_category,
        array_to_string(p_key_features[1:2], ', ')
    );
    
    v_title := steadytext_generate(v_prompt, 20);
    
    IF v_title IS NULL OR length(v_title) > 60 THEN
        -- Fallback to simple concatenation
        v_title := substring(
            format('%s %s - %s', p_brand, p_name, p_key_features[1]),
            1, 60
        );
    END IF;
    
    RETURN v_title;
END;
$$ LANGUAGE plpgsql;
```

### Product Recommendations

```sql
-- Personalized product recommendations
CREATE OR REPLACE FUNCTION ecommerce.get_personalized_recommendations(
    p_customer_id INTEGER,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE(
    product_id INTEGER,
    product_name VARCHAR(200),
    score FLOAT,
    reason TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH customer_profile AS (
        -- Build customer preference profile
        SELECT 
            c.preference_embedding,
            array_agg(DISTINCT cat.name) as preferred_categories,
            avg(e.metadata->>'price_range') as avg_price_range
        FROM ecommerce.customers c
        LEFT JOIN ecommerce.customer_events e ON c.id = e.customer_id
        LEFT JOIN ecommerce.products p ON e.product_id = p.id
        LEFT JOIN ecommerce.categories cat ON p.category_id = cat.id
        WHERE c.id = p_customer_id
        GROUP BY c.id, c.preference_embedding
    ),
    product_scores AS (
        SELECT 
            p.id,
            p.name,
            -- Combine embedding similarity with business rules
            (
                CASE 
                    WHEN cp.preference_embedding IS NOT NULL 
                    THEN 0.6 * (1 - (p.embedding <-> cp.preference_embedding))
                    ELSE 0.3
                END +
                CASE 
                    WHEN cat.name = ANY(cp.preferred_categories) 
                    THEN 0.3
                    ELSE 0.0
                END +
                CASE 
                    WHEN abs(p.price - COALESCE(cp.avg_price_range::numeric, p.price)) < 50 
                    THEN 0.1
                    ELSE 0.0
                END
            ) as score,
            cp.preferred_categories
        FROM ecommerce.products p
        JOIN ecommerce.categories cat ON p.category_id = cat.id
        CROSS JOIN customer_profile cp
        WHERE p.id NOT IN (
            -- Exclude already purchased
            SELECT DISTINCT product_id 
            FROM ecommerce.customer_events 
            WHERE customer_id = p_customer_id 
            AND event_type = 'purchase'
        )
    )
    SELECT 
        ps.id as product_id,
        ps.name as product_name,
        ps.score,
        CASE 
            WHEN ps.preferred_categories IS NOT NULL 
            THEN 'Based on your interest in ' || ps.preferred_categories[1]
            ELSE 'Trending product you might like'
        END as reason
    FROM product_scores ps
    ORDER BY ps.score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Cross-sell recommendations
CREATE OR REPLACE FUNCTION ecommerce.get_cross_sell_products(
    p_product_id INTEGER,
    p_limit INTEGER DEFAULT 5
) RETURNS TABLE(
    product_id INTEGER,
    product_name VARCHAR(200),
    confidence FLOAT,
    relationship TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH product_pairs AS (
        -- Find products frequently bought together
        SELECT 
            ce2.product_id as related_id,
            COUNT(*) as co_purchase_count
        FROM ecommerce.customer_events ce1
        JOIN ecommerce.customer_events ce2 
            ON ce1.customer_id = ce2.customer_id 
            AND ce1.product_id != ce2.product_id
            AND ce2.created_at BETWEEN ce1.created_at AND ce1.created_at + INTERVAL '1 hour'
        WHERE ce1.product_id = p_product_id
            AND ce1.event_type = 'purchase'
            AND ce2.event_type = 'purchase'
        GROUP BY ce2.product_id
    ),
    semantic_similarity AS (
        -- Find semantically similar products
        SELECT 
            p2.id as related_id,
            1 - (p1.embedding <-> p2.embedding) as similarity
        FROM ecommerce.products p1
        JOIN ecommerce.products p2 ON p1.id != p2.id
        WHERE p1.id = p_product_id
    )
    SELECT 
        p.id as product_id,
        p.name as product_name,
        GREATEST(
            COALESCE(pp.co_purchase_count::float / 100, 0),
            COALESCE(ss.similarity, 0)
        ) as confidence,
        CASE 
            WHEN pp.co_purchase_count > 0 THEN 'Frequently bought together'
            ELSE 'Similar product'
        END as relationship
    FROM ecommerce.products p
    LEFT JOIN product_pairs pp ON p.id = pp.related_id
    LEFT JOIN semantic_similarity ss ON p.id = ss.related_id
    WHERE p.id != p_product_id
        AND (pp.co_purchase_count > 0 OR ss.similarity > 0.7)
    ORDER BY confidence DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;
```

### Review Analysis

```sql
-- Analyze review sentiment
CREATE OR REPLACE FUNCTION ecommerce.analyze_review_sentiment(
    p_content TEXT
) RETURNS FLOAT AS $$
DECLARE
    v_prompt TEXT;
    v_result TEXT;
    v_sentiment FLOAT;
BEGIN
    v_prompt := format(
        'Rate the sentiment of this product review from -1 (very negative) to 1 (very positive). Return only the number: %s',
        substring(p_content, 1, 500)
    );
    
    v_result := steadytext_generate(v_prompt, 10);
    
    IF v_result ~ '^-?[0-9]*\.?[0-9]+$' THEN
        v_sentiment := v_result::FLOAT;
        v_sentiment := GREATEST(-1, LEAST(1, v_sentiment));
    ELSE
        v_sentiment := 0.0;
    END IF;
    
    RETURN v_sentiment;
END;
$$ LANGUAGE plpgsql;

-- Generate review summary
CREATE OR REPLACE FUNCTION ecommerce.generate_review_summary(
    p_product_id INTEGER
) RETURNS TABLE(
    summary TEXT,
    pros TEXT[],
    cons TEXT[],
    overall_sentiment FLOAT
) AS $$
DECLARE
    v_reviews TEXT;
    v_prompt TEXT;
    v_summary TEXT;
    v_pros_text TEXT;
    v_cons_text TEXT;
BEGIN
    -- Aggregate reviews
    SELECT string_agg(
        format('Rating: %s/5 - %s', rating, substring(content, 1, 200)),
        E'\n'
    ) INTO v_reviews
    FROM (
        SELECT rating, content
        FROM ecommerce.reviews
        WHERE product_id = p_product_id
        ORDER BY helpful_count DESC, created_at DESC
        LIMIT 10
    ) r;
    
    -- Generate summary
    v_prompt := format(
        'Summarize these product reviews in 2-3 sentences: %s',
        v_reviews
    );
    v_summary := steadytext_generate(v_prompt, 100);
    
    -- Extract pros
    v_prompt := format(
        'List 3 main pros mentioned in these reviews as comma-separated values: %s',
        v_reviews
    );
    v_pros_text := steadytext_generate(v_prompt, 50);
    
    -- Extract cons
    v_prompt := format(
        'List 3 main cons mentioned in these reviews as comma-separated values: %s',
        v_reviews
    );
    v_cons_text := steadytext_generate(v_prompt, 50);
    
    RETURN QUERY
    SELECT 
        COALESCE(v_summary, 'No reviews yet'),
        string_to_array(COALESCE(v_pros_text, ''), ','),
        string_to_array(COALESCE(v_cons_text, ''), ','),
        (SELECT AVG(sentiment_score) FROM ecommerce.reviews WHERE product_id = p_product_id);
END;
$$ LANGUAGE plpgsql;

-- Auto-moderate reviews
CREATE OR REPLACE FUNCTION ecommerce.moderate_review()
RETURNS TRIGGER AS $$
BEGIN
    -- Analyze sentiment
    NEW.sentiment_score := ecommerce.analyze_review_sentiment(NEW.content);
    
    -- Generate embedding for similarity
    NEW.embedding := steadytext_embed(
        COALESCE(NEW.title, '') || ' ' || NEW.content
    );
    
    -- Check for potential issues
    IF NEW.sentiment_score < -0.8 THEN
        -- Flag for manual review
        INSERT INTO ecommerce.moderation_queue (
            review_id, reason, created_at
        ) VALUES (
            NEW.id, 'Very negative sentiment', NOW()
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER review_moderation
    BEFORE INSERT OR UPDATE ON ecommerce.reviews
    FOR EACH ROW
    EXECUTE FUNCTION ecommerce.moderate_review();
```

### Dynamic Pricing

```sql
-- Price optimization suggestions
CREATE OR REPLACE FUNCTION ecommerce.suggest_optimal_price(
    p_product_id INTEGER
) RETURNS TABLE(
    current_price DECIMAL(10,2),
    suggested_price DECIMAL(10,2),
    expected_impact TEXT,
    reasoning TEXT
) AS $$
DECLARE
    v_product RECORD;
    v_metrics RECORD;
    v_prompt TEXT;
    v_suggestion TEXT;
BEGIN
    -- Get product info
    SELECT * INTO v_product
    FROM ecommerce.products
    WHERE id = p_product_id;
    
    -- Calculate metrics
    WITH sales_data AS (
        SELECT 
            COUNT(*) as total_sales,
            COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as recent_sales,
            AVG(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) as view_rate
        FROM ecommerce.customer_events
        WHERE product_id = p_product_id
    ),
    competitor_prices AS (
        SELECT 
            AVG(price) as avg_competitor_price,
            MIN(price) as min_price,
            MAX(price) as max_price
        FROM ecommerce.products
        WHERE category_id = v_product.category_id
            AND id != p_product_id
    )
    SELECT * INTO v_metrics
    FROM sales_data, competitor_prices;
    
    -- Generate pricing suggestion
    v_prompt := format(
        'Analyze pricing: Current price $%s. Recent sales: %s/month. Category avg price: $%s. Suggest optimal price and explain why.',
        v_product.price,
        v_metrics.recent_sales,
        round(v_metrics.avg_competitor_price, 2)
    );
    
    v_suggestion := steadytext_generate(v_prompt, 150);
    
    -- Parse suggestion (in real implementation, use structured generation)
    RETURN QUERY
    SELECT 
        v_product.price,
        CASE 
            WHEN v_metrics.recent_sales < 10 AND v_product.price > v_metrics.avg_competitor_price
            THEN v_product.price * 0.9
            WHEN v_metrics.recent_sales > 50 AND v_product.price < v_metrics.avg_competitor_price
            THEN v_product.price * 1.1
            ELSE v_product.price
        END,
        CASE 
            WHEN v_metrics.recent_sales < 10 THEN 'Increase sales volume'
            WHEN v_metrics.recent_sales > 50 THEN 'Maximize revenue'
            ELSE 'Maintain current position'
        END,
        COALESCE(v_suggestion, 'Price appears optimal for current market conditions');
END;
$$ LANGUAGE plpgsql;
```

### Customer Service Automation

```sql
-- Generate customer service responses
CREATE OR REPLACE FUNCTION ecommerce.generate_cs_response(
    p_inquiry_type TEXT,
    p_context JSONB
) RETURNS TEXT AS $$
DECLARE
    v_prompt TEXT;
    v_response TEXT;
BEGIN
    v_prompt := CASE p_inquiry_type
        WHEN 'order_status' THEN format(
            'Write a friendly response about order #%s status: %s. Estimated delivery: %s',
            p_context->>'order_id',
            p_context->>'status',
            p_context->>'delivery_date'
        )
        WHEN 'return_request' THEN format(
            'Write a helpful response for a return request. Product: %s. Reason: %s. Policy: 30-day returns.',
            p_context->>'product_name',
            p_context->>'reason'
        )
        WHEN 'product_question' THEN format(
            'Answer this product question: %s. Product info: %s',
            p_context->>'question',
            p_context->>'product_info'
        )
        ELSE 'Write a friendly customer service response acknowledging the inquiry.'
    END;
    
    v_response := steadytext_generate(v_prompt, 150);
    
    RETURN COALESCE(
        v_response,
        'Thank you for contacting us. A customer service representative will assist you shortly.'
    );
END;
$$ LANGUAGE plpgsql;

-- Categorize customer inquiries
CREATE OR REPLACE FUNCTION ecommerce.categorize_inquiry(
    p_message TEXT
) RETURNS TEXT AS $$
DECLARE
    v_category TEXT;
BEGIN
    v_category := steadytext_generate_choice(
        format('Categorize this customer inquiry: %s', p_message),
        ARRAY[
            'order_status',
            'return_request',
            'product_question',
            'shipping_inquiry',
            'payment_issue',
            'technical_support',
            'general_inquiry'
        ]
    );
    
    RETURN COALESCE(v_category, 'general_inquiry');
END;
$$ LANGUAGE plpgsql;
```

### Inventory Intelligence

```sql
-- Predict inventory needs
CREATE OR REPLACE FUNCTION ecommerce.predict_inventory_needs(
    p_product_id INTEGER,
    p_days_ahead INTEGER DEFAULT 30
) RETURNS TABLE(
    predicted_demand INTEGER,
    confidence_level TEXT,
    factors TEXT[],
    recommendation TEXT
) AS $$
DECLARE
    v_sales_history RECORD;
    v_prompt TEXT;
    v_prediction TEXT;
BEGIN
    -- Analyze sales patterns
    WITH sales_analysis AS (
        SELECT 
            COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as recent_sales,
            COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '90 days') / 3.0 as monthly_avg,
            EXTRACT(DOW FROM NOW()) as day_of_week,
            EXTRACT(MONTH FROM NOW()) as current_month
        FROM ecommerce.customer_events
        WHERE product_id = p_product_id
            AND event_type = 'purchase'
    )
    SELECT * INTO v_sales_history FROM sales_analysis;
    
    -- Generate prediction
    v_prompt := format(
        'Predict inventory needs for next %s days. Recent monthly sales: %s. Average: %s. Current month: %s. Provide a number.',
        p_days_ahead,
        v_sales_history.recent_sales,
        round(v_sales_history.monthly_avg, 1),
        to_char(to_timestamp(v_sales_history.current_month::text, 'MM'), 'Month')
    );
    
    v_prediction := steadytext_generate(v_prompt, 100);
    
    RETURN QUERY
    SELECT 
        GREATEST(
            round(v_sales_history.monthly_avg * (p_days_ahead / 30.0) * 1.2)::INTEGER,
            10
        ),
        CASE 
            WHEN v_sales_history.recent_sales > v_sales_history.monthly_avg * 1.5 THEN 'High'
            WHEN v_sales_history.recent_sales < v_sales_history.monthly_avg * 0.5 THEN 'Low'
            ELSE 'Medium'
        END,
        ARRAY[
            'Historical sales: ' || v_sales_history.recent_sales,
            'Trend: ' || CASE 
                WHEN v_sales_history.recent_sales > v_sales_history.monthly_avg THEN 'Increasing'
                ELSE 'Stable'
            END,
            'Season: ' || to_char(NOW(), 'Month')
        ],
        COALESCE(
            v_prediction,
            format('Recommend stocking %s units based on historical data', 
                   round(v_sales_history.monthly_avg * (p_days_ahead / 30.0) * 1.2))
        );
END;
$$ LANGUAGE plpgsql;
```

### A/B Testing for Products

```sql
-- A/B test different product descriptions
CREATE TABLE ecommerce.ab_tests (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES ecommerce.products(id),
    variant_name VARCHAR(50),
    description TEXT,
    embedding vector(1024),
    impressions INTEGER DEFAULT 0,
    conversions INTEGER DEFAULT 0,
    revenue DECIMAL(10,2) DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Get A/B test variant
CREATE OR REPLACE FUNCTION ecommerce.get_ab_test_variant(
    p_product_id INTEGER,
    p_customer_id INTEGER
) RETURNS TABLE(
    variant_id INTEGER,
    description TEXT
) AS $$
DECLARE
    v_hash INTEGER;
BEGIN
    -- Consistent hashing for customer assignment
    v_hash := abs(hashtext(p_product_id::text || p_customer_id::text));
    
    RETURN QUERY
    SELECT 
        id,
        description
    FROM ecommerce.ab_tests
    WHERE product_id = p_product_id
        AND is_active = TRUE
    ORDER BY id
    LIMIT 1
    OFFSET (v_hash % (
        SELECT COUNT(*) 
        FROM ecommerce.ab_tests 
        WHERE product_id = p_product_id AND is_active = TRUE
    ));
END;
$$ LANGUAGE plpgsql;

-- Analyze A/B test results
CREATE OR REPLACE FUNCTION ecommerce.analyze_ab_test(
    p_product_id INTEGER
) RETURNS TABLE(
    variant_name VARCHAR(50),
    conversion_rate FLOAT,
    avg_revenue DECIMAL(10,2),
    statistical_significance TEXT,
    recommendation TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH test_results AS (
        SELECT 
            variant_name,
            impressions,
            conversions,
            revenue,
            conversions::FLOAT / NULLIF(impressions, 0) as conv_rate,
            revenue / NULLIF(conversions, 0) as avg_order_value
        FROM ecommerce.ab_tests
        WHERE product_id = p_product_id
    ),
    winner AS (
        SELECT variant_name
        FROM test_results
        ORDER BY conv_rate DESC NULLS LAST
        LIMIT 1
    )
    SELECT 
        tr.variant_name,
        tr.conv_rate,
        tr.avg_order_value,
        CASE 
            WHEN tr.impressions < 100 THEN 'Insufficient data'
            WHEN tr.conv_rate > (SELECT AVG(conv_rate) * 1.2 FROM test_results) THEN 'Significant improvement'
            WHEN tr.conv_rate < (SELECT AVG(conv_rate) * 0.8 FROM test_results) THEN 'Significant decline'
            ELSE 'No significant difference'
        END,
        CASE 
            WHEN tr.variant_name = (SELECT variant_name FROM winner) 
            THEN 'Winning variant - consider making permanent'
            ELSE 'Continue testing or discontinue'
        END
    FROM test_results tr;
END;
$$ LANGUAGE plpgsql;
```

## Related Documentation

- [PostgreSQL Extension Overview](../postgresql-extension.md)
- [Blog & CMS Examples](postgresql-blog-cms.md)
- [Search Examples](postgresql-search.md)
- [Analytics Examples](postgresql-analytics.md)
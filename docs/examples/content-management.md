# Content Management with AI-Powered Generation

Build a smart content management system that generates, optimizes, and personalizes content using SteadyText's deterministic AI capabilities.

## Overview

This tutorial demonstrates how to:
- Auto-generate product descriptions and SEO metadata
- Create content variations for A/B testing
- Build a content moderation pipeline
- Personalize content based on user segments
- Generate multilingual content variants

## Prerequisites

```bash
# Start PostgreSQL with SteadyText
docker run -d -p 5432:5432 --name steadytext-cms julep/pg-steadytext

# Connect to the database
psql -h localhost -U postgres

# Enable required extensions
CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- For UUIDs
```

## Database Schema

Let's create a comprehensive content management schema:

```sql
-- Products table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    sku VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100),
    brand VARCHAR(100),
    price DECIMAL(10, 2),
    features JSONB,
    specifications JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Product descriptions with versions
CREATE TABLE product_descriptions (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id),
    version INTEGER DEFAULT 1,
    language VARCHAR(10) DEFAULT 'en',
    title VARCHAR(200),
    short_description TEXT,
    long_description TEXT,
    seo_title VARCHAR(70),
    seo_description VARCHAR(160),
    seo_keywords TEXT[],
    generated_by VARCHAR(50), -- 'human' or 'ai'
    is_active BOOLEAN DEFAULT FALSE,
    performance_score DECIMAL(5, 2), -- A/B test performance
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Content templates
CREATE TABLE content_templates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    template_type VARCHAR(50), -- 'product', 'email', 'landing_page'
    prompt_template TEXT NOT NULL,
    variables JSONB, -- Expected variables
    output_format VARCHAR(20) DEFAULT 'text', -- 'text', 'html', 'json'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User segments for personalization
CREATE TABLE user_segments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    criteria JSONB, -- Segment definition
    preferences JSONB, -- Content preferences
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Content moderation log
CREATE TABLE moderation_log (
    id SERIAL PRIMARY KEY,
    content_type VARCHAR(50),
    content_id INTEGER,
    original_content TEXT,
    moderated_content TEXT,
    issues_found JSONB,
    action_taken VARCHAR(50), -- 'approved', 'modified', 'rejected'
    moderated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_descriptions_product ON product_descriptions(product_id);
CREATE INDEX idx_descriptions_active ON product_descriptions(is_active);
CREATE INDEX idx_moderation_content ON moderation_log(content_type, content_id);
```

## Auto-Generate Product Descriptions

Create a sophisticated product description generator:

```sql
-- Function to generate product descriptions
CREATE OR REPLACE FUNCTION generate_product_description(
    p_product_id INTEGER,
    p_style VARCHAR DEFAULT 'professional', -- 'professional', 'casual', 'technical'
    p_length VARCHAR DEFAULT 'medium' -- 'short', 'medium', 'long'
)
RETURNS TABLE (
    title VARCHAR,
    short_description TEXT,
    long_description TEXT,
    seo_title VARCHAR,
    seo_description VARCHAR,
    seo_keywords TEXT[]
) AS $$
DECLARE
    v_product products%ROWTYPE;
    v_features TEXT;
    v_specs TEXT;
BEGIN
    -- Get product details
    SELECT * INTO v_product FROM products WHERE id = p_product_id;
    
    -- Format features and specifications
    v_features := COALESCE(
        (SELECT string_agg(key || ': ' || value, ', ')
         FROM jsonb_each_text(v_product.features)),
        'Standard features'
    );
    
    v_specs := COALESCE(
        (SELECT string_agg(key || ': ' || value, ', ')
         FROM jsonb_each_text(v_product.specifications)),
        'Standard specifications'
    );
    
    RETURN QUERY
    SELECT
        -- Generate compelling title
        steadytext_generate(
            format('Create a compelling product title for: %s %s (max 60 chars)',
                v_product.brand, v_product.name),
            max_tokens := 20
        )::VARCHAR AS title,
        
        -- Short description for product cards
        steadytext_generate(
            format('Write a %s style product description for %s %s with features: %s (max 150 chars)',
                p_style, v_product.brand, v_product.name, v_features),
            max_tokens := 50
        ) AS short_description,
        
        -- Long description for product pages
        steadytext_generate(
            format('Write a detailed %s style product description for %s %s. Features: %s. Specs: %s. Price: $%s',
                p_style, v_product.brand, v_product.name, v_features, v_specs, v_product.price),
            max_tokens := CASE p_length 
                WHEN 'short' THEN 100
                WHEN 'long' THEN 300
                ELSE 200
            END
        ) AS long_description,
        
        -- SEO title (max 70 chars)
        steadytext_generate(
            format('Create SEO title for: %s %s %s (max 60 chars)',
                v_product.brand, v_product.name, v_product.category),
            max_tokens := 20
        )::VARCHAR AS seo_title,
        
        -- SEO meta description (max 160 chars)
        steadytext_generate(
            format('Write SEO meta description for %s %s with benefits and call-to-action (max 150 chars)',
                v_product.brand, v_product.name),
            max_tokens := 50
        )::VARCHAR AS seo_description,
        
        -- SEO keywords
        string_to_array(
            steadytext_generate(
                format('List 5 SEO keywords for: %s %s %s (comma separated)',
                    v_product.brand, v_product.name, v_product.category),
                max_tokens := 30
            ),
            ', '
        ) AS seo_keywords;
END;
$$ LANGUAGE plpgsql;
```

## A/B Testing Content Variations

Generate multiple content variations for testing:

```sql
-- Generate content variations for A/B testing
CREATE OR REPLACE FUNCTION create_content_variations(
    p_product_id INTEGER,
    p_num_variations INTEGER DEFAULT 3
)
RETURNS TABLE (
    variation_id INTEGER,
    title VARCHAR,
    description TEXT,
    style VARCHAR
) AS $$
DECLARE
    v_styles VARCHAR[] := ARRAY['professional', 'casual', 'technical', 'enthusiastic', 'minimalist'];
    v_style VARCHAR;
    v_counter INTEGER := 1;
BEGIN
    -- Generate variations with different styles
    WHILE v_counter <= p_num_variations LOOP
        v_style := v_styles[1 + (v_counter % array_length(v_styles, 1))];
        
        RETURN QUERY
        WITH generated AS (
            SELECT * FROM generate_product_description(
                p_product_id, 
                v_style, 
                'medium'
            )
        )
        INSERT INTO product_descriptions (
            product_id, version, title, 
            short_description, long_description,
            seo_title, seo_description, seo_keywords,
            generated_by, is_active
        )
        SELECT 
            p_product_id,
            (SELECT COALESCE(MAX(version), 0) + 1 
             FROM product_descriptions 
             WHERE product_id = p_product_id),
            g.title,
            g.short_description,
            g.long_description,
            g.seo_title,
            g.seo_description,
            g.seo_keywords,
            'ai',
            v_counter = 1 -- First variation is active by default
        FROM generated g
        RETURNING 
            id AS variation_id,
            title,
            short_description AS description,
            v_style AS style;
        
        v_counter := v_counter + 1;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

## Content Moderation Pipeline

Automatically moderate user-generated content:

```sql
-- Content moderation function
CREATE OR REPLACE FUNCTION moderate_content(
    p_content TEXT,
    p_content_type VARCHAR,
    p_content_id INTEGER
)
RETURNS TABLE (
    moderated_content TEXT,
    is_safe BOOLEAN,
    issues JSONB,
    action VARCHAR
) AS $$
DECLARE
    v_toxicity_check VARCHAR;
    v_issues JSONB := '[]'::JSONB;
    v_cleaned_content TEXT;
BEGIN
    -- Check for toxicity/inappropriate content
    v_toxicity_check := steadytext_generate_choice(
        'Is this content appropriate for all audiences: ' || LEFT(p_content, 500),
        ARRAY['safe', 'potentially_inappropriate', 'inappropriate', 'toxic']
    );
    
    -- Build issues list
    IF v_toxicity_check != 'safe' THEN
        v_issues := v_issues || jsonb_build_object(
            'type', 'toxicity',
            'severity', v_toxicity_check
        );
    END IF;
    
    -- Check for spam patterns
    IF p_content ~* '(click here|buy now|limited time|act now){3,}' THEN
        v_issues := v_issues || jsonb_build_object(
            'type', 'spam',
            'severity', 'high'
        );
    END IF;
    
    -- Clean or reject content based on issues
    IF v_toxicity_check IN ('inappropriate', 'toxic') THEN
        -- Reject toxic content
        v_cleaned_content := '[Content removed due to policy violation]';
        
        INSERT INTO moderation_log (
            content_type, content_id, original_content,
            moderated_content, issues_found, action_taken
        ) VALUES (
            p_content_type, p_content_id, p_content,
            v_cleaned_content, v_issues, 'rejected'
        );
        
        RETURN QUERY SELECT 
            v_cleaned_content,
            FALSE,
            v_issues,
            'rejected'::VARCHAR;
    ELSE
        -- Clean up mild issues
        v_cleaned_content := steadytext_generate(
            'Rewrite this content to be more appropriate while keeping the same meaning: ' || p_content,
            max_tokens := 200
        );
        
        INSERT INTO moderation_log (
            content_type, content_id, original_content,
            moderated_content, issues_found, action_taken
        ) VALUES (
            p_content_type, p_content_id, p_content,
            v_cleaned_content, v_issues, 
            CASE WHEN jsonb_array_length(v_issues) > 0 THEN 'modified' ELSE 'approved' END
        );
        
        RETURN QUERY SELECT 
            v_cleaned_content,
            jsonb_array_length(v_issues) = 0,
            v_issues,
            CASE WHEN jsonb_array_length(v_issues) > 0 THEN 'modified' ELSE 'approved' END::VARCHAR;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

## Dynamic Content Templates

Create reusable content templates:

```sql
-- Insert sample templates
INSERT INTO content_templates (name, template_type, prompt_template, variables) VALUES
(
    'product_email_campaign',
    'email',
    'Write a promotional email for {product_name} highlighting {key_feature} with a {tone} tone. Include subject line.',
    '{"product_name": "string", "key_feature": "string", "tone": "string"}'::JSONB
),
(
    'category_landing_page',
    'landing_page',
    'Create landing page copy for {category} products. Include hero text, 3 benefit points, and CTA. Style: {style}',
    '{"category": "string", "style": "string"}'::JSONB
),
(
    'seasonal_promotion',
    'product',
    'Write {season} promotional text for {product_name}. Emphasize seasonal benefits and include urgency.',
    '{"season": "string", "product_name": "string"}'::JSONB
);

-- Function to use templates
CREATE OR REPLACE FUNCTION generate_from_template(
    p_template_name VARCHAR,
    p_variables JSONB
)
RETURNS TEXT AS $$
DECLARE
    v_template content_templates%ROWTYPE;
    v_prompt TEXT;
    v_key TEXT;
    v_value TEXT;
BEGIN
    -- Get template
    SELECT * INTO v_template FROM content_templates WHERE name = p_template_name;
    
    -- Replace variables in prompt
    v_prompt := v_template.prompt_template;
    FOR v_key, v_value IN SELECT * FROM jsonb_each_text(p_variables) LOOP
        v_prompt := REPLACE(v_prompt, '{' || v_key || '}', v_value);
    END LOOP;
    
    -- Generate content
    RETURN steadytext_generate(v_prompt, max_tokens := 300);
END;
$$ LANGUAGE plpgsql;
```

## Personalized Content Generation

Generate content tailored to user segments:

```sql
-- Insert sample user segments
INSERT INTO user_segments (name, criteria, preferences) VALUES
('budget_conscious', 
 '{"income": "low_medium", "behavior": "price_sensitive"}'::JSONB,
 '{"tone": "value_focused", "highlight": "savings"}'::JSONB),
('premium_buyers',
 '{"income": "high", "behavior": "quality_focused"}'::JSONB,
 '{"tone": "luxurious", "highlight": "exclusivity"}'::JSONB),
('tech_enthusiasts',
 '{"interests": ["technology", "gadgets"], "age": "18-35"}'::JSONB,
 '{"tone": "technical", "highlight": "specifications"}'::JSONB);

-- Personalized content function
CREATE OR REPLACE FUNCTION generate_personalized_content(
    p_product_id INTEGER,
    p_segment_id INTEGER
)
RETURNS TEXT AS $$
DECLARE
    v_product products%ROWTYPE;
    v_segment user_segments%ROWTYPE;
    v_prompt TEXT;
BEGIN
    SELECT * INTO v_product FROM products WHERE id = p_product_id;
    SELECT * INTO v_segment FROM user_segments WHERE id = p_segment_id;
    
    -- Build personalized prompt
    v_prompt := format(
        'Write product description for %s %s targeting %s customers. Tone: %s. Highlight: %s. Price: $%s',
        v_product.brand,
        v_product.name,
        v_segment.name,
        v_segment.preferences->>'tone',
        v_segment.preferences->>'highlight',
        v_product.price
    );
    
    RETURN steadytext_generate(v_prompt, max_tokens := 150);
END;
$$ LANGUAGE plpgsql;
```

## Bulk Content Operations

Process multiple items efficiently:

```sql
-- Bulk generate descriptions for all products
CREATE OR REPLACE FUNCTION bulk_generate_descriptions(
    p_category VARCHAR DEFAULT NULL,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    product_id INTEGER,
    product_name VARCHAR,
    status VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    WITH products_to_process AS (
        SELECT p.id, p.name
        FROM products p
        LEFT JOIN product_descriptions pd ON p.id = pd.product_id AND pd.is_active
        WHERE pd.id IS NULL  -- No active description
          AND (p_category IS NULL OR p.category = p_category)
        LIMIT p_limit
    )
    SELECT 
        ptp.id,
        ptp.name,
        CASE 
            WHEN generate_product_description(ptp.id) IS NOT NULL THEN 'generated'
            ELSE 'failed'
        END AS status
    FROM products_to_process ptp;
END;
$$ LANGUAGE plpgsql;
```

## Content Performance Analytics

Track and analyze content performance:

```sql
-- Content performance view
CREATE OR REPLACE VIEW content_performance AS
WITH performance_metrics AS (
    SELECT 
        pd.id,
        pd.product_id,
        pd.version,
        pd.title,
        pd.generated_by,
        pd.performance_score,
        p.name AS product_name,
        p.category,
        pd.created_at,
        RANK() OVER (PARTITION BY pd.product_id ORDER BY pd.performance_score DESC) AS rank
    FROM product_descriptions pd
    JOIN products p ON pd.product_id = p.id
    WHERE pd.performance_score IS NOT NULL
)
SELECT 
    *,
    steadytext_generate(
        format('Analyze why this content performed %s: Title: %s, Category: %s, Score: %s',
            CASE 
                WHEN performance_score > 80 THEN 'excellently'
                WHEN performance_score > 60 THEN 'well'
                ELSE 'poorly'
            END,
            title,
            category,
            performance_score
        ),
        max_tokens := 100
    ) AS performance_analysis
FROM performance_metrics
WHERE rank <= 3;  -- Top 3 versions per product
```

## Sample Data and Testing

```sql
-- Insert sample products
INSERT INTO products (sku, name, category, brand, price, features, specifications) VALUES
('WH-1000XM5', 'Wireless Noise-Cancelling Headphones', 'Audio', 'Sony', 399.99,
 '{"noise_cancelling": "Industry-leading", "battery": "30 hours", "comfort": "Premium"}'::JSONB,
 '{"weight": "250g", "bluetooth": "5.2", "drivers": "30mm"}'::JSONB),
('MBA-M2-2023', 'MacBook Air M2', 'Computers', 'Apple', 1199.00,
 '{"processor": "M2 chip", "display": "13.6-inch Retina", "battery": "18 hours"}'::JSONB,
 '{"ram": "8GB", "storage": "256GB SSD", "weight": "1.24kg"}'::JSONB),
('OLED55C3', '55" OLED Smart TV', 'Electronics', 'LG', 1299.99,
 '{"display": "OLED", "resolution": "4K", "smart": "webOS"}'::JSONB,
 '{"size": "55 inches", "refresh": "120Hz", "hdr": "Dolby Vision"}'::JSONB);

-- Generate content for products
SELECT * FROM create_content_variations(1, 3);
SELECT * FROM bulk_generate_descriptions('Audio', 10);

-- Test content moderation
SELECT * FROM moderate_content(
    'This is an amazing product! Buy now for 50% off!!!!! Click here!!!',
    'review',
    1
);

-- Generate from template
SELECT generate_from_template(
    'seasonal_promotion',
    '{"season": "Summer", "product_name": "Wireless Headphones"}'::JSONB
);
```

## Best Practices

1. **Version Control**: Keep all generated content versions for comparison
2. **A/B Testing**: Always test AI-generated content against human-written
3. **Moderation**: Review AI outputs before publishing
4. **Caching**: Leverage SteadyText's built-in caching for repeated generations
5. **Templates**: Use templates for consistent brand voice

## Next Steps

- [Customer Intelligence Tutorial →](customer-intelligence.md)
- [Data Pipelines Example →](data-pipelines.md)
- [Migration from OpenAI →](../migration/from-openai.md)

---

!!! tip "Pro Tip"
    Use database triggers to automatically generate content when new products are added. This ensures every product has optimized descriptions from day one.
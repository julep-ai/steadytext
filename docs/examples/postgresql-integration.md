# PostgreSQL Integration Examples

This guide provides comprehensive examples for integrating SteadyText with PostgreSQL using the `pg_steadytext` extension. Learn how to build powerful applications that combine structured data with AI-generated content and embeddings.

## Table of Contents

- [Basic Setup](#basic-setup)
- [Content Management](#content-management)
- [Semantic Search](#semantic-search)
- [Real-time Applications](#real-time-applications)
- [Advanced Workflows](#advanced-workflows)
- [Performance Optimization](#performance-optimization)

## Basic Setup

### Installation and Configuration

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;
CREATE EXTENSION IF NOT EXISTS omni_python CASCADE;
CREATE EXTENSION IF NOT EXISTS pgvector CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;

-- Verify installation
SELECT steadytext_version();
SELECT * FROM steadytext_config;

-- Configure for your environment
SELECT steadytext_config_set('default_max_tokens', '256');
SELECT steadytext_config_set('default_seed', '42');
SELECT steadytext_config_set('cache_enabled', 'true');

-- Start the daemon for better performance
SELECT steadytext_daemon_start();
SELECT * FROM steadytext_daemon_status();
```

### Basic Text Generation

```sql
-- Simple text generation (uses default seed 42)
SELECT steadytext_generate('Write a product description for a smartphone');

-- With custom parameters and error handling
SELECT 
    CASE 
        WHEN steadytext_generate(
            'Explain machine learning',
            max_tokens := 200,
            seed := 123
        ) IS NOT NULL 
        THEN steadytext_generate(
            'Explain machine learning',
            max_tokens := 200,
            seed := 123
        )
        ELSE 'Error: Failed to generate content. Please check daemon status.'
    END AS result;

-- Batch generation with NULL handling
WITH prompts AS (
    SELECT unnest(ARRAY[
        'Describe artificial intelligence',
        'Explain quantum computing',
        'What is blockchain technology'
    ]) AS prompt
)
SELECT 
    prompt,
    COALESCE(
        steadytext_generate(prompt, max_tokens := 150, seed := 42),
        '[Generation failed for this prompt]'
    ) AS response,
    CASE 
        WHEN steadytext_generate(prompt, max_tokens := 150, seed := 42) IS NOT NULL 
        THEN 'Success'
        ELSE 'Failed'
    END AS status
FROM prompts;
```

### Basic Embeddings

```sql
-- Generate embeddings (uses default seed 42)
SELECT steadytext_embed('artificial intelligence');

-- Generate with error checking
SELECT 
    'artificial intelligence' AS text,
    CASE 
        WHEN steadytext_embed('artificial intelligence') IS NOT NULL 
        THEN 'Embedding generated successfully'
        ELSE 'Embedding generation failed'
    END AS status;

-- Create a table with embeddings
CREATE TABLE concepts (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    embedding vector(1024),
    embedding_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Populate with embeddings (with error handling)
WITH embedding_data AS (
    SELECT 
        name,
        description,
        text,
        steadytext_embed(text, seed := 42) AS embedding
    FROM (VALUES 
        ('AI', 'Artificial Intelligence', 'artificial intelligence'),
        ('ML', 'Machine Learning', 'machine learning'),
        ('DL', 'Deep Learning', 'deep learning')
    ) AS concepts(name, description, text)
)
INSERT INTO concepts (name, description, embedding, embedding_status)
SELECT 
    name,
    description,
    embedding,
    CASE 
        WHEN embedding IS NOT NULL THEN 'success'
        ELSE 'failed'
    END AS embedding_status
FROM embedding_data;

-- Find similar concepts (with NULL handling)
WITH query_embedding AS (
    SELECT steadytext_embed('neural networks', seed := 42) AS embedding
)
SELECT 
    c.name,
    c.description,
    CASE 
        WHEN c.embedding IS NOT NULL AND qe.embedding IS NOT NULL 
        THEN 1 - (c.embedding <=> qe.embedding)
        ELSE NULL
    END AS similarity,
    CASE 
        WHEN c.embedding IS NULL THEN 'Missing concept embedding'
        WHEN qe.embedding IS NULL THEN 'Query embedding failed'
        ELSE 'OK'
    END AS status
FROM concepts c
CROSS JOIN query_embedding qe
WHERE c.embedding_status = 'success'
ORDER BY similarity DESC NULLS LAST;
```

## Content Management

### Blog Platform

Build a complete blog platform with AI-generated content and semantic search.

```sql
-- Create blog schema
CREATE TABLE authors (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    bio TEXT,
    bio_embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE blog_posts (
    id SERIAL PRIMARY KEY,
    author_id INTEGER REFERENCES authors(id),
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    tags TEXT[],
    title_embedding vector(1024),
    content_embedding vector(1024),
    published_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES blog_posts(id),
    author_name TEXT NOT NULL,
    content TEXT NOT NULL,
    sentiment_score REAL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for vector similarity search
CREATE INDEX ON blog_posts USING ivfflat (title_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON blog_posts USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON authors USING ivfflat (bio_embedding vector_cosine_ops) WITH (lists = 100);

-- Function to auto-generate blog content with error handling
CREATE OR REPLACE FUNCTION generate_blog_post(
    topic TEXT,
    target_length INTEGER DEFAULT 500,
    writing_style TEXT DEFAULT 'informative',
    post_seed INTEGER DEFAULT 42
)
RETURNS TABLE(title TEXT, content TEXT, summary TEXT, tags TEXT[], generation_status TEXT) AS $$
DECLARE
    generated_title TEXT;
    generated_content TEXT;
    generated_summary TEXT;
    generated_tags_text TEXT;
    status TEXT := 'success';
BEGIN
    -- Generate title with error handling
    generated_title := steadytext_generate(
        format('Create an engaging blog post title about: %s', topic),
        max_tokens := 20,
        seed := post_seed
    );
    
    IF generated_title IS NULL THEN
        generated_title := format('[Generated title for: %s]', topic);
        status := 'partial_failure';
    END IF;
    
    -- Generate content with error handling
    generated_content := steadytext_generate(
        format('Write a %s word blog post about %s in a %s style', 
               target_length, topic, writing_style),
        max_tokens := target_length,
        seed := post_seed + 1
    );
    
    IF generated_content IS NULL THEN
        generated_content := format('[Content about %s could not be generated]', topic);
        status := 'partial_failure';
    END IF;
    
    -- Generate summary with error handling
    IF generated_content IS NOT NULL AND generated_content NOT LIKE '[Content%' THEN
        generated_summary := steadytext_generate(
            format('Write a brief summary of this blog post: %s', generated_content),
            max_tokens := 100,
            seed := post_seed + 2
        );
    END IF;
    
    IF generated_summary IS NULL THEN
        generated_summary := format('Summary of %s blog post', topic);
        status := 'partial_failure';
    END IF;
    
    -- Generate tags with error handling
    generated_tags_text := steadytext_generate(
        format('List 5 relevant tags for a blog post about: %s (comma-separated)', topic),
        max_tokens := 30,
        seed := post_seed + 3
    );
    
    IF generated_tags_text IS NOT NULL THEN
        tags := string_to_array(generated_tags_text, ',');
    ELSE
        tags := ARRAY[topic, 'technology', 'blog'];
        status := 'partial_failure';
    END IF;
    
    -- Return results
    title := generated_title;
    content := generated_content;
    summary := generated_summary;
    generation_status := status;
    
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Auto-populate blog with generated content (with error handling)
WITH author_embeddings AS (
    SELECT 
        name,
        bio,
        steadytext_embed(bio, seed := 100 + ROW_NUMBER() OVER()) AS bio_embedding
    FROM (VALUES 
        ('AI Writer', 'An AI-powered content creator specializing in technology topics.'),
        ('Tech Analyst', 'Expert in emerging technologies and digital transformation.')
    ) AS authors(name, bio)
)
INSERT INTO authors (name, bio, bio_embedding)
SELECT 
    name,
    bio,
    CASE 
        WHEN bio_embedding IS NOT NULL THEN bio_embedding
        ELSE vector(array_fill(0::real, ARRAY[1024]))  -- Zero vector fallback
    END
FROM author_embeddings;

-- Generate blog posts with comprehensive error handling
WITH generated_posts AS (
    SELECT 
        1 as author_id,
        'Machine Learning' as topic,
        generate_blog_post('Machine Learning', 400, 'technical', 200) as post_data
    UNION ALL
    SELECT 
        2 as author_id,
        'Quantum Computing' as topic,
        generate_blog_post('Quantum Computing', 350, 'educational', 300) as post_data
    UNION ALL
    SELECT 
        1 as author_id,
        'Blockchain Technology' as topic,
        generate_blog_post('Blockchain Technology', 450, 'analytical', 400) as post_data
),
embedded_posts AS (
    SELECT 
        gp.author_id,
        (gp.post_data).title,
        (gp.post_data).content,
        (gp.post_data).summary,
        (gp.post_data).tags,
        (gp.post_data).generation_status,
        steadytext_embed((gp.post_data).title, seed := 500) AS title_embedding,
        steadytext_embed((gp.post_data).content, seed := 600) AS content_embedding
    FROM generated_posts gp
)
INSERT INTO blog_posts (author_id, title, content, summary, tags, title_embedding, content_embedding)
SELECT 
    author_id,
    title,
    content,
    summary,
    tags,
    COALESCE(title_embedding, vector(array_fill(0::real, ARRAY[1024]))),
    COALESCE(content_embedding, vector(array_fill(0::real, ARRAY[1024])))
FROM embedded_posts
WHERE generation_status IS NOT NULL;  -- Only insert posts that were generated

-- Semantic blog search function with NULL handling
CREATE OR REPLACE FUNCTION search_blog_posts(
    search_query TEXT,
    max_results INTEGER DEFAULT 10,
    search_seed INTEGER DEFAULT 42
)
RETURNS TABLE(
    id INTEGER,
    title TEXT,
    summary TEXT,
    author_name TEXT,
    similarity_score REAL,
    published_at TIMESTAMP,
    search_status TEXT
) AS $$
DECLARE
    query_embedding vector(1024);
BEGIN
    -- Generate query embedding with error handling
    query_embedding := steadytext_embed(search_query, seed := search_seed);
    
    -- Return empty result if embedding generation failed
    IF query_embedding IS NULL THEN
        RAISE WARNING 'Failed to generate embedding for search query: %', search_query;
        RETURN;
    END IF;
    
    RETURN QUERY
    SELECT 
        bp.id,
        bp.title,
        bp.summary,
        a.name as author_name,
        1 - (bp.content_embedding <=> query_embedding) AS similarity_score,
        bp.published_at,
        'success' AS search_status
    FROM blog_posts bp
    JOIN authors a ON bp.author_id = a.id
    WHERE bp.content_embedding IS NOT NULL
    ORDER BY bp.content_embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Search for posts with status checking
SELECT 
    id,
    title,
    summary,
    author_name,
    similarity_score,
    published_at,
    search_status
FROM search_blog_posts('artificial intelligence applications', 5);

-- Handle failed searches
WITH search_results AS (
    SELECT * FROM search_blog_posts('future technology trends', 3)
)
SELECT 
    CASE 
        WHEN COUNT(*) > 0 THEN 'Search completed successfully'
        ELSE 'Search failed - no results returned'
    END AS search_summary,
    COUNT(*) AS result_count
FROM search_results;

-- Auto-generate related posts
CREATE OR REPLACE FUNCTION suggest_related_posts(
    post_id_input INTEGER,
    max_suggestions INTEGER DEFAULT 3
)
RETURNS TABLE(
    suggested_id INTEGER,
    suggested_title TEXT,
    similarity_score REAL
) AS $$
DECLARE
    source_embedding vector(1024);
BEGIN
    -- Get the embedding of the source post
    SELECT content_embedding INTO source_embedding
    FROM blog_posts
    WHERE id = post_id_input;
    
    IF source_embedding IS NULL THEN
        RAISE EXCEPTION 'Post not found or has no embedding';
    END IF;
    
    RETURN QUERY
    SELECT 
        bp.id as suggested_id,
        bp.title as suggested_title,
        1 - (bp.content_embedding <=> source_embedding) AS similarity_score
    FROM blog_posts bp
    WHERE bp.id != post_id_input 
      AND bp.content_embedding IS NOT NULL
    ORDER BY bp.content_embedding <=> source_embedding
    LIMIT max_suggestions;
END;
$$ LANGUAGE plpgsql;

-- Find related posts
SELECT * FROM suggest_related_posts(1, 3);
```

### E-commerce Product Catalog

Create an intelligent product catalog with AI-generated descriptions and semantic search.

```sql
-- E-commerce schema
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    description_embedding vector(1024)
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    category_id INTEGER REFERENCES categories(id),
    name TEXT NOT NULL,
    base_features JSONB,
    ai_description TEXT,
    ai_marketing_copy TEXT,
    technical_specs TEXT,
    price DECIMAL(10,2),
    embedding vector(1024),
    marketing_embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE product_reviews (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id),
    customer_name TEXT,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    review_text TEXT,
    sentiment_analysis TEXT,
    embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for vector search
CREATE INDEX ON products USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON products USING ivfflat (marketing_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON product_reviews USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Product content generation function
CREATE OR REPLACE FUNCTION generate_product_content(
    product_name TEXT,
    features JSONB,
    category_name TEXT,
    price_value DECIMAL,
    content_seed INTEGER DEFAULT 42
)
RETURNS TABLE(
    description TEXT,
    marketing_copy TEXT,
    technical_specs TEXT
) AS $$
DECLARE
    features_text TEXT;
BEGIN
    -- Convert JSONB features to readable text
    features_text := (
        SELECT string_agg(key || ': ' || value, ', ')
        FROM jsonb_each_text(features)
    );
    
    -- Generate product description
    description := steadytext_generate(
        format('Write a detailed product description for %s in the %s category. Features: %s. Price: $%s',
               product_name, category_name, features_text, price_value),
        max_tokens := 200,
        seed := content_seed
    );
    
    -- Generate marketing copy
    marketing_copy := steadytext_generate(
        format('Create compelling marketing copy for %s. Highlight benefits and unique selling points. Features: %s',
               product_name, features_text),
        max_tokens := 150,
        seed := content_seed + 100
    );
    
    -- Generate technical specifications
    technical_specs := steadytext_generate(
        format('Create detailed technical specifications for %s based on these features: %s',
               product_name, features_text),
        max_tokens := 250,
        seed := content_seed + 200
    );
    
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Populate categories
INSERT INTO categories (name, description, description_embedding)
VALUES 
    ('Smartphones', 'Mobile devices with advanced computing capabilities',
     steadytext_embed('Mobile devices with advanced computing capabilities', seed := 1000)),
    ('Laptops', 'Portable computers for professional and personal use',
     steadytext_embed('Portable computers for professional and personal use', seed := 1001)),
    ('Smartwatches', 'Wearable devices with health and connectivity features',
     steadytext_embed('Wearable devices with health and connectivity features', seed := 1002));

-- Generate products with AI content
WITH product_data AS (
    SELECT 
        'iPhone 15 Pro' as name,
        1 as category_id,
        '{"screen": "6.1 inch OLED", "storage": "256GB", "camera": "48MP", "battery": "3274mAh"}'::jsonb as features,
        'Smartphones' as category_name,
        999.99 as price,
        2000 as seed
    UNION ALL
    SELECT 
        'MacBook Air M3' as name,
        2 as category_id,
        '{"processor": "Apple M3", "memory": "16GB", "storage": "512GB SSD", "display": "13.6 inch Retina"}'::jsonb as features,
        'Laptops' as category_name,
        1299.99 as price,
        2100 as seed
    UNION ALL
    SELECT 
        'Apple Watch Series 9' as name,
        3 as category_id,
        '{"display": "45mm Always-On Retina", "health": "Blood Oxygen, ECG", "battery": "18 hours", "connectivity": "GPS + Cellular"}'::jsonb as features,
        'Smartwatches' as category_name,
        429.99 as price,
        2200 as seed
)
INSERT INTO products (category_id, name, base_features, ai_description, ai_marketing_copy, technical_specs, price, embedding, marketing_embedding)
SELECT 
    pd.category_id,
    pd.name,
    pd.features,
    (generate_product_content(pd.name, pd.features, pd.category_name, pd.price, pd.seed)).description,
    (generate_product_content(pd.name, pd.features, pd.category_name, pd.price, pd.seed)).marketing_copy,
    (generate_product_content(pd.name, pd.features, pd.category_name, pd.price, pd.seed)).technical_specs,
    pd.price,
    steadytext_embed(pd.name || ' ' || (generate_product_content(pd.name, pd.features, pd.category_name, pd.price, pd.seed)).description, seed := 3000),
    steadytext_embed((generate_product_content(pd.name, pd.features, pd.category_name, pd.price, pd.seed)).marketing_copy, seed := 3100)
FROM product_data pd;

-- Product search functions
CREATE OR REPLACE FUNCTION search_products(
    search_query TEXT,
    search_type TEXT DEFAULT 'general', -- 'general', 'marketing', 'technical'
    max_results INTEGER DEFAULT 10,
    search_seed INTEGER DEFAULT 42
)
RETURNS TABLE(
    id INTEGER,
    name TEXT,
    description TEXT,
    price DECIMAL,
    category_name TEXT,
    similarity_score REAL
) AS $$
DECLARE
    query_embedding vector(1024);
    embedding_column TEXT;
BEGIN
    -- Generate embedding for search query
    query_embedding := steadytext_embed(search_query, seed := search_seed);
    
    -- Choose embedding column based on search type
    embedding_column := CASE search_type
        WHEN 'marketing' THEN 'marketing_embedding'
        ELSE 'embedding'
    END;
    
    RETURN QUERY EXECUTE format('
        SELECT 
            p.id,
            p.name,
            p.ai_description as description,
            p.price,
            c.name as category_name,
            1 - (p.%I <=> $1) AS similarity_score
        FROM products p
        JOIN categories c ON p.category_id = c.id
        WHERE p.%I IS NOT NULL
        ORDER BY p.%I <=> $1
        LIMIT $2
    ', embedding_column, embedding_column, embedding_column)
    USING query_embedding, max_results;
END;
$$ LANGUAGE plpgsql;

-- Search examples
SELECT * FROM search_products('high-performance laptop for programming', 'general', 5);
SELECT * FROM search_products('best smartphone camera', 'marketing', 3);

-- Auto-generate product comparisons
CREATE OR REPLACE FUNCTION compare_products(
    product_ids INTEGER[],
    comparison_seed INTEGER DEFAULT 42
)
RETURNS TEXT AS $$
DECLARE
    product_info TEXT;
    comparison_result TEXT;
BEGIN
    -- Gather product information
    SELECT string_agg(
        format('%s: %s (Price: $%s)', name, ai_description, price),
        E'\n'
    ) INTO product_info
    FROM products
    WHERE id = ANY(product_ids);
    
    -- Generate comparison
    comparison_result := steadytext_generate(
        format('Compare these products and highlight their key differences and advantages:\n%s', product_info),
        max_tokens := 400,
        seed := comparison_seed
    );
    
    RETURN comparison_result;
END;
$$ LANGUAGE plpgsql;

-- Compare products
SELECT compare_products(ARRAY[1, 2, 3], 4000);
```

## Semantic Search

### Document Management System

Build a comprehensive document search and analysis system.

```sql
-- Document management schema
CREATE TABLE document_types (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    processing_instructions TEXT,
    embedding vector(1024)
);

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    type_id INTEGER REFERENCES document_types(id),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    key_points TEXT[],
    metadata JSONB,
    content_embedding vector(1024),
    summary_embedding vector(1024),
    created_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP
);

CREATE TABLE document_sections (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    section_title TEXT,
    section_content TEXT NOT NULL,
    section_order INTEGER,
    embedding vector(1024)
);

CREATE TABLE document_relationships (
    id SERIAL PRIMARY KEY,
    source_doc_id INTEGER REFERENCES documents(id),
    target_doc_id INTEGER REFERENCES documents(id),
    relationship_type TEXT, -- 'similar', 'references', 'updated_version'
    confidence_score REAL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for document search
CREATE INDEX ON documents USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON documents USING ivfflat (summary_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON document_sections USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Document processing function
CREATE OR REPLACE FUNCTION process_document(
    doc_id INTEGER,
    processing_seed INTEGER DEFAULT 42
)
RETURNS BOOLEAN AS $$
DECLARE
    doc_record documents%ROWTYPE;
    doc_summary TEXT;
    doc_key_points TEXT[];
    section_texts TEXT[];
    section_text TEXT;
    i INTEGER;
BEGIN
    -- Get document
    SELECT * INTO doc_record FROM documents WHERE id = doc_id;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Document not found';
    END IF;
    
    -- Generate summary
    doc_summary := steadytext_generate(
        format('Summarize the following document in 2-3 sentences:\n%s', 
               left(doc_record.content, 2000)),
        max_tokens := 150,
        seed := processing_seed
    );
    
    -- Extract key points
    SELECT string_to_array(
        steadytext_generate(
            format('Extract 5 key points from this document (one per line):\n%s',
                   left(doc_record.content, 1500)),
            max_tokens := 200,
            seed := processing_seed + 100
        ),
        E'\n'
    ) INTO doc_key_points;
    
    -- Update document with processed information
    UPDATE documents SET
        summary = doc_summary,
        key_points = doc_key_points,
        content_embedding = steadytext_embed(doc_record.content, seed := processing_seed + 200),
        summary_embedding = steadytext_embed(doc_summary, seed := processing_seed + 300),
        processed_at = NOW()
    WHERE id = doc_id;
    
    -- Create document sections (split content into chunks)
    section_texts := string_to_array(doc_record.content, E'\n\n');
    
    FOR i IN 1..array_length(section_texts, 1) LOOP
        section_text := section_texts[i];
        IF length(section_text) > 50 THEN  -- Only process substantial sections
            INSERT INTO document_sections (document_id, section_content, section_order, embedding)
            VALUES (
                doc_id,
                section_text,
                i,
                steadytext_embed(section_text, seed := processing_seed + 400 + i)
            );
        END IF;
    END LOOP;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Advanced search function
CREATE OR REPLACE FUNCTION search_documents(
    search_query TEXT,
    search_mode TEXT DEFAULT 'content', -- 'content', 'summary', 'sections'
    doc_type_filter INTEGER DEFAULT NULL,
    max_results INTEGER DEFAULT 10,
    search_seed INTEGER DEFAULT 42
)
RETURNS TABLE(
    document_id INTEGER,
    title TEXT,
    summary TEXT,
    similarity_score REAL,
    doc_type TEXT,
    section_match TEXT
) AS $$
DECLARE
    query_embedding vector(1024);
BEGIN
    query_embedding := steadytext_embed(search_query, seed := search_seed);
    
    IF search_mode = 'sections' THEN
        -- Search within document sections
        RETURN QUERY
        SELECT 
            d.id as document_id,
            d.title,
            d.summary,
            1 - (ds.embedding <=> query_embedding) AS similarity_score,
            dt.name as doc_type,
            left(ds.section_content, 200) as section_match
        FROM document_sections ds
        JOIN documents d ON ds.document_id = d.id
        JOIN document_types dt ON d.type_id = dt.id
        WHERE (doc_type_filter IS NULL OR d.type_id = doc_type_filter)
          AND ds.embedding IS NOT NULL
        ORDER BY ds.embedding <=> query_embedding
        LIMIT max_results;
    
    ELSIF search_mode = 'summary' THEN
        -- Search within summaries
        RETURN QUERY
        SELECT 
            d.id as document_id,
            d.title,
            d.summary,
            1 - (d.summary_embedding <=> query_embedding) AS similarity_score,
            dt.name as doc_type,
            NULL::TEXT as section_match
        FROM documents d
        JOIN document_types dt ON d.type_id = dt.id
        WHERE (doc_type_filter IS NULL OR d.type_id = doc_type_filter)
          AND d.summary_embedding IS NOT NULL
        ORDER BY d.summary_embedding <=> query_embedding
        LIMIT max_results;
    
    ELSE
        -- Default: search within full content
        RETURN QUERY
        SELECT 
            d.id as document_id,
            d.title,
            d.summary,
            1 - (d.content_embedding <=> query_embedding) AS similarity_score,
            dt.name as doc_type,
            NULL::TEXT as section_match
        FROM documents d
        JOIN document_types dt ON d.type_id = dt.id
        WHERE (doc_type_filter IS NULL OR d.type_id = doc_type_filter)
          AND d.content_embedding IS NOT NULL
        ORDER BY d.content_embedding <=> query_embedding
        LIMIT max_results;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Setup document types and sample data
INSERT INTO document_types (name, description, processing_instructions, embedding)
VALUES 
    ('Technical Manual', 'Technical documentation and user guides',
     'Focus on technical details and step-by-step instructions',
     steadytext_embed('Technical documentation and user guides', seed := 5000)),
    ('Research Paper', 'Academic and research publications',
     'Extract methodology, results, and conclusions',
     steadytext_embed('Academic and research publications', seed := 5001)),
    ('Legal Document', 'Contracts, agreements, and legal texts',
     'Identify key terms, obligations, and legal implications',
     steadytext_embed('Contracts, agreements, and legal texts', seed := 5002));

-- Add sample documents
INSERT INTO documents (type_id, title, content)
VALUES 
    (1, 'API Integration Guide', 
     'This guide explains how to integrate our REST API into your application. The API provides endpoints for user authentication, data retrieval, and real-time updates. Authentication is handled via JWT tokens that must be included in request headers. Rate limiting is enforced at 1000 requests per hour per API key.'),
    (2, 'Machine Learning in Healthcare',
     'Recent advances in machine learning have shown significant promise in healthcare applications. This study examines the effectiveness of neural networks in medical diagnosis, analyzing data from 10,000 patient records. Results indicate 95% accuracy in early disease detection when compared to traditional diagnostic methods.'),
    (3, 'Software License Agreement',
     'This agreement governs the use of the software product. The licensee agrees to use the software solely for internal business purposes. Redistribution is prohibited without written consent. The license term is perpetual but may be terminated for breach of terms. Limitation of liability applies to all claims.');

-- Process all documents
SELECT process_document(id, 6000 + id) FROM documents;

-- Search examples
SELECT * FROM search_documents('API authentication methods', 'content', NULL, 5);
SELECT * FROM search_documents('machine learning accuracy', 'summary', 2, 3);
SELECT * FROM search_documents('license terms', 'sections', 3, 5);

-- Find document relationships
CREATE OR REPLACE FUNCTION find_related_documents(
    source_doc_id INTEGER,
    similarity_threshold REAL DEFAULT 0.7,
    max_related INTEGER DEFAULT 5
)
RETURNS TABLE(
    related_doc_id INTEGER,
    related_title TEXT,
    similarity_score REAL,
    relationship_type TEXT
) AS $$
DECLARE
    source_embedding vector(1024);
    source_type_id INTEGER;
BEGIN
    -- Get source document embedding and type
    SELECT content_embedding, type_id 
    INTO source_embedding, source_type_id
    FROM documents 
    WHERE id = source_doc_id;
    
    IF source_embedding IS NULL THEN
        RAISE EXCEPTION 'Source document not found or not processed';
    END IF;
    
    RETURN QUERY
    SELECT 
        d.id as related_doc_id,
        d.title as related_title,
        1 - (d.content_embedding <=> source_embedding) AS similarity_score,
        CASE 
            WHEN d.type_id = source_type_id THEN 'same_type'
            ELSE 'different_type'
        END as relationship_type
    FROM documents d
    WHERE d.id != source_doc_id
      AND d.content_embedding IS NOT NULL
      AND 1 - (d.content_embedding <=> source_embedding) >= similarity_threshold
    ORDER BY d.content_embedding <=> source_embedding
    LIMIT max_related;
END;
$$ LANGUAGE plpgsql;

-- Find related documents
SELECT * FROM find_related_documents(1, 0.5, 3);
```

## Real-time Applications

### Chat System with AI Assistance

Build a real-time chat system with AI-powered features.

```sql
-- Chat system schema
CREATE TABLE chat_rooms (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    ai_assistant_enabled BOOLEAN DEFAULT false,
    ai_personality TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chat_participants (
    id SERIAL PRIMARY KEY,
    room_id INTEGER REFERENCES chat_rooms(id),
    user_name TEXT NOT NULL,
    joined_at TIMESTAMP DEFAULT NOW(),
    is_ai_user BOOLEAN DEFAULT false
);

CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    room_id INTEGER REFERENCES chat_rooms(id),
    participant_id INTEGER REFERENCES chat_participants(id),
    message_text TEXT NOT NULL,
    ai_generated BOOLEAN DEFAULT false,
    ai_confidence REAL,
    message_embedding vector(1024),
    reply_to_message_id INTEGER REFERENCES chat_messages(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chat_context (
    id SERIAL PRIMARY KEY,
    room_id INTEGER REFERENCES chat_rooms(id),
    context_summary TEXT,
    key_topics TEXT[],
    context_embedding vector(1024),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for chat search and recommendations
CREATE INDEX ON chat_messages USING ivfflat (message_embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX ON chat_context USING ivfflat (context_embedding vector_cosine_ops) WITH (lists = 20);
CREATE INDEX ON chat_messages (room_id, created_at DESC);

-- AI assistant function
CREATE OR REPLACE FUNCTION generate_ai_response(
    room_id_input INTEGER,
    context_messages TEXT,
    ai_personality_input TEXT DEFAULT 'helpful assistant',
    response_seed INTEGER DEFAULT 42
)
RETURNS TEXT AS $$
DECLARE
    ai_response TEXT;
    prompt_text TEXT;
BEGIN
    -- Build context-aware prompt
    prompt_text := format(
        'You are a %s in a chat room. Based on this recent conversation context, provide a helpful and relevant response:\n\n%s\n\nResponse:',
        ai_personality_input,
        context_messages
    );
    
    -- Generate AI response
    ai_response := steadytext_generate(
        prompt_text,
        max_tokens := 200,
        seed := response_seed
    );
    
    RETURN ai_response;
END;
$$ LANGUAGE plpgsql;

-- Context summarization function
CREATE OR REPLACE FUNCTION update_chat_context(
    room_id_input INTEGER,
    context_seed INTEGER DEFAULT 42
)
RETURNS BOOLEAN AS $$
DECLARE
    recent_messages TEXT;
    context_summary_new TEXT;
    key_topics_new TEXT[];
BEGIN
    -- Get recent messages (last 20)
    SELECT string_agg(
        format('[%s] %s: %s', 
               to_char(cm.created_at, 'HH24:MI'), 
               cp.user_name, 
               cm.message_text),
        E'\n' ORDER BY cm.created_at DESC
    )
    INTO recent_messages
    FROM chat_messages cm
    JOIN chat_participants cp ON cm.participant_id = cp.id
    WHERE cm.room_id = room_id_input
      AND cm.created_at > NOW() - INTERVAL '1 hour'
    LIMIT 20;
    
    IF recent_messages IS NULL THEN
        RETURN FALSE;
    END IF;
    
    -- Generate context summary
    context_summary_new := steadytext_generate(
        format('Summarize the key points and current discussion topics from this chat conversation:\n%s', recent_messages),
        max_tokens := 150,
        seed := context_seed
    );
    
    -- Extract key topics
    SELECT string_to_array(
        steadytext_generate(
            format('Extract 5 main topics being discussed in this chat (comma-separated):\n%s', recent_messages),
            max_tokens := 50,
            seed := context_seed + 100
        ),
        ','
    ) INTO key_topics_new;
    
    -- Update or insert context
    INSERT INTO chat_context (room_id, context_summary, key_topics, context_embedding, updated_at)
    VALUES (
        room_id_input,
        context_summary_new,
        key_topics_new,
        steadytext_embed(context_summary_new, seed := context_seed + 200),
        NOW()
    )
    ON CONFLICT (room_id) DO UPDATE SET
        context_summary = EXCLUDED.context_summary,
        key_topics = EXCLUDED.key_topics,
        context_embedding = EXCLUDED.context_embedding,
        updated_at = EXCLUDED.updated_at;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Message processing trigger function
CREATE OR REPLACE FUNCTION process_chat_message()
RETURNS TRIGGER AS $$
DECLARE
    room_has_ai BOOLEAN;
    ai_personality_val TEXT;
    recent_context TEXT;
    ai_participant_id INTEGER;
    ai_response TEXT;
BEGIN
    -- Generate embedding for the new message
    NEW.message_embedding := steadytext_embed(NEW.message_text, seed := 42);
    
    -- Check if room has AI assistant enabled
    SELECT ai_assistant_enabled, ai_personality 
    INTO room_has_ai, ai_personality_val
    FROM chat_rooms 
    WHERE id = NEW.room_id;
    
    -- If AI is enabled and this isn't an AI message, potentially generate response
    IF room_has_ai AND NOT NEW.ai_generated THEN
        -- Get recent conversation context
        SELECT string_agg(
            format('%s: %s', cp.user_name, cm.message_text),
            E'\n' ORDER BY cm.created_at DESC
        )
        INTO recent_context
        FROM chat_messages cm
        JOIN chat_participants cp ON cm.participant_id = cp.id
        WHERE cm.room_id = NEW.room_id
          AND cm.created_at > NOW() - INTERVAL '10 minutes'
        LIMIT 5;
        
        -- Decide if AI should respond (simple logic - respond to questions or mentions)
        IF NEW.message_text ILIKE '%?%' OR NEW.message_text ILIKE '%ai%' THEN
            -- Get AI participant
            SELECT id INTO ai_participant_id
            FROM chat_participants
            WHERE room_id = NEW.room_id AND is_ai_user = true
            LIMIT 1;
            
            IF ai_participant_id IS NOT NULL THEN
                -- Generate AI response
                ai_response := generate_ai_response(
                    NEW.room_id,
                    recent_context,
                    ai_personality_val,
                    extract(epoch from NOW())::integer % 10000
                );
                
                -- Insert AI response
                INSERT INTO chat_messages (
                    room_id, 
                    participant_id, 
                    message_text, 
                    ai_generated, 
                    ai_confidence,
                    reply_to_message_id
                )
                VALUES (
                    NEW.room_id,
                    ai_participant_id,
                    ai_response,
                    true,
                    0.8,
                    NEW.id
                );
            END IF;
        END IF;
        
        -- Update chat context periodically
        PERFORM update_chat_context(NEW.room_id);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for message processing
CREATE TRIGGER chat_message_processing
    BEFORE INSERT ON chat_messages
    FOR EACH ROW
    EXECUTE FUNCTION process_chat_message();

-- Message search function
CREATE OR REPLACE FUNCTION search_chat_history(
    room_id_input INTEGER,
    search_query TEXT,
    max_results INTEGER DEFAULT 10,
    search_seed INTEGER DEFAULT 42
)
RETURNS TABLE(
    message_id INTEGER,
    user_name TEXT,
    message_text TEXT,
    similarity_score REAL,
    created_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cm.id as message_id,
        cp.user_name,
        cm.message_text,
        1 - (cm.message_embedding <=> steadytext_embed(search_query, seed := search_seed)) AS similarity_score,
        cm.created_at
    FROM chat_messages cm
    JOIN chat_participants cp ON cm.participant_id = cp.id
    WHERE cm.room_id = room_id_input
      AND cm.message_embedding IS NOT NULL
    ORDER BY cm.message_embedding <=> steadytext_embed(search_query, seed := search_seed)
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Setup sample chat room
INSERT INTO chat_rooms (name, description, ai_assistant_enabled, ai_personality)
VALUES ('Tech Discussion', 'General technology discussion room', true, 'knowledgeable tech expert');

-- Add participants
INSERT INTO chat_participants (room_id, user_name, is_ai_user)
VALUES 
    (1, 'Alice', false),
    (1, 'Bob', false),
    (1, 'TechBot', true);

-- Simulate conversation
INSERT INTO chat_messages (room_id, participant_id, message_text, ai_generated)
VALUES 
    (1, 1, 'Hi everyone! What do you think about the latest AI developments?', false),
    (1, 2, 'I think machine learning is advancing really fast. What are your thoughts on GPT models?', false);

-- Search chat history
SELECT * FROM search_chat_history(1, 'artificial intelligence developments', 5);
```

## Advanced Workflows

### Content Pipeline with Quality Control

```sql
-- Content pipeline schema
CREATE TABLE content_templates (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    template_text TEXT NOT NULL,
    variables JSONB,
    quality_criteria JSONB,
    embedding vector(1024)
);

CREATE TABLE content_jobs (
    id SERIAL PRIMARY KEY,
    template_id INTEGER REFERENCES content_templates(id),
    input_parameters JSONB,
    status TEXT DEFAULT 'pending', -- pending, processing, completed, failed, review_needed
    priority INTEGER DEFAULT 5,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE generated_content (
    id SERIAL PRIMARY KEY,
    job_id INTEGER REFERENCES content_jobs(id),
    content_text TEXT NOT NULL,
    quality_score REAL,
    quality_issues TEXT[],
    embedding vector(1024),
    approved BOOLEAN DEFAULT false,
    human_review_notes TEXT,
    generation_seed INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Quality assessment function
CREATE OR REPLACE FUNCTION assess_content_quality(
    content_text TEXT,
    quality_criteria JSONB,
    assessment_seed INTEGER DEFAULT 42
)
RETURNS TABLE(
    quality_score REAL,
    quality_issues TEXT[]
) AS $$
DECLARE
    criteria_text TEXT;
    assessment_prompt TEXT;
    assessment_result TEXT;
    score_match TEXT;
    issues_text TEXT;
BEGIN
    -- Convert criteria to readable format
    SELECT string_agg(key || ': ' || value, ', ')
    INTO criteria_text
    FROM jsonb_each_text(quality_criteria);
    
    -- Generate quality assessment
    assessment_prompt := format(
        'Assess the quality of this content against these criteria: %s\n\nContent to assess:\n%s\n\nProvide a score from 0.0 to 1.0 and list any issues (format: Score: X.X, Issues: issue1, issue2)',
        criteria_text,
        content_text
    );
    
    assessment_result := steadytext_generate(
        assessment_prompt,
        max_tokens := 200,
        seed := assessment_seed
    );
    
    -- Extract score (simple pattern matching)
    score_match := substring(assessment_result from 'Score: ([0-9.]+)');
    quality_score := COALESCE(score_match::real, 0.5);
    
    -- Extract issues
    issues_text := substring(assessment_result from 'Issues: (.+)');
    IF issues_text IS NOT NULL THEN
        quality_issues := string_to_array(trim(issues_text), ', ');
    ELSE
        quality_issues := ARRAY[]::TEXT[];
    END IF;
    
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- Content generation worker function
CREATE OR REPLACE FUNCTION process_content_job(
    job_id_input INTEGER
)
RETURNS BOOLEAN AS $$
DECLARE
    job_record content_jobs%ROWTYPE;
    template_record content_templates%ROWTYPE;
    generation_prompt TEXT;
    generated_text TEXT;
    quality_result RECORD;
    generation_seed INTEGER;
BEGIN
    -- Get job and template
    SELECT * INTO job_record FROM content_jobs WHERE id = job_id_input;
    SELECT * INTO template_record FROM content_templates WHERE id = job_record.template_id;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Job or template not found';
    END IF;
    
    -- Update job status
    UPDATE content_jobs SET 
        status = 'processing',
        started_at = NOW()
    WHERE id = job_id_input;
    
    -- Build generation prompt by substituting variables
    generation_prompt := template_record.template_text;
    
    -- Simple variable substitution (in practice, you'd want more sophisticated templating)
    FOR variable_key, variable_value IN 
        SELECT key, value FROM jsonb_each_text(job_record.input_parameters)
    LOOP
        generation_prompt := replace(generation_prompt, '{{' || variable_key || '}}', variable_value);
    END LOOP;
    
    -- Generate unique seed for this job
    generation_seed := 50000 + job_id_input;
    
    -- Generate content
    generated_text := steadytext_generate(
        generation_prompt,
        max_tokens := 500,
        seed := generation_seed
    );
    
    -- Assess quality
    SELECT * INTO quality_result FROM assess_content_quality(
        generated_text,
        template_record.quality_criteria,
        generation_seed + 1000
    );
    
    -- Store generated content
    INSERT INTO generated_content (
        job_id,
        content_text,
        quality_score,
        quality_issues,
        embedding,
        approved,
        generation_seed
    ) VALUES (
        job_id_input,
        generated_text,
        quality_result.quality_score,
        quality_result.quality_issues,
        steadytext_embed(generated_text, seed := generation_seed + 2000),
        quality_result.quality_score >= 0.8, -- Auto-approve high quality content
        generation_seed
    );
    
    -- Update job status
    UPDATE content_jobs SET
        status = CASE 
            WHEN quality_result.quality_score >= 0.8 THEN 'completed'
            ELSE 'review_needed'
        END,
        completed_at = NOW()
    WHERE id = job_id_input;
    
    RETURN TRUE;
EXCEPTION
    WHEN OTHERS THEN
        UPDATE content_jobs SET status = 'failed' WHERE id = job_id_input;
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- Setup content templates
INSERT INTO content_templates (name, template_text, variables, quality_criteria, embedding)
VALUES 
    ('Product Description',
     'Write a compelling product description for {{product_name}} in the {{category}} category. Key features: {{features}}. Target audience: {{audience}}.',
     '{"product_name": "string", "category": "string", "features": "string", "audience": "string"}',
     '{"clarity": "high", "engagement": "high", "accuracy": "high", "length": "150-300 words"}',
     steadytext_embed('Product description template for e-commerce', seed := 60000)),
    
    ('Blog Post Introduction',
     'Write an engaging introduction for a blog post about {{topic}}. The post should appeal to {{audience}} and have a {{tone}} tone.',
     '{"topic": "string", "audience": "string", "tone": "string"}',
     '{"hook": "strong", "clarity": "high", "engagement": "high", "length": "100-200 words"}',
     steadytext_embed('Blog post introduction template', seed := 60001));

-- Create content jobs
INSERT INTO content_jobs (template_id, input_parameters, priority)
VALUES 
    (1, '{"product_name": "Smart Fitness Tracker", "category": "wearables", "features": "heart rate monitoring, sleep tracking, waterproof", "audience": "fitness enthusiasts"}', 1),
    (2, '{"topic": "sustainable technology", "audience": "environmentally conscious consumers", "tone": "informative yet inspiring"}', 2);

-- Process content jobs
SELECT process_content_job(1);
SELECT process_content_job(2);

-- Review generated content
SELECT 
    cj.id as job_id,
    ct.name as template_name,
    gc.content_text,
    gc.quality_score,
    gc.quality_issues,
    gc.approved,
    cj.status
FROM content_jobs cj
JOIN content_templates ct ON cj.template_id = ct.id
JOIN generated_content gc ON cj.id = gc.job_id
ORDER BY cj.created_at DESC;
```

## Performance Optimization

### Monitoring and Analytics

```sql
-- Performance monitoring schema
CREATE TABLE api_usage_logs (
    id SERIAL PRIMARY KEY,
    function_name TEXT NOT NULL,
    input_text TEXT,
    input_length INTEGER,
    output_text TEXT,
    output_length INTEGER,
    processing_time_ms INTEGER,
    cache_hit BOOLEAN,
    seed_used INTEGER,
    embedding_similarity REAL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE cache_performance (
    id SERIAL PRIMARY KEY,
    cache_type TEXT NOT NULL, -- 'generation', 'embedding'
    hit_rate REAL,
    total_requests INTEGER,
    cache_hits INTEGER,
    cache_misses INTEGER,
    avg_response_time_ms REAL,
    measured_at TIMESTAMP DEFAULT NOW()
);

-- Performance monitoring functions
CREATE OR REPLACE FUNCTION log_steadytext_usage(
    func_name TEXT,
    input_text TEXT,
    output_text TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    was_cache_hit BOOLEAN DEFAULT false,
    seed_value INTEGER DEFAULT 42
)
RETURNS VOID AS $$
DECLARE
    processing_ms INTEGER;
BEGIN
    processing_ms := EXTRACT(epoch FROM (end_time - start_time)) * 1000;
    
    INSERT INTO api_usage_logs (
        function_name,
        input_text,
        input_length,
        output_text,
        output_length,
        processing_time_ms,
        cache_hit,
        seed_used
    ) VALUES (
        func_name,
        input_text,
        length(input_text),
        output_text,
        length(output_text),
        processing_ms,
        was_cache_hit,
        seed_value
    );
END;
$$ LANGUAGE plpgsql;

-- Enhanced generation function with monitoring
CREATE OR REPLACE FUNCTION monitored_generate(
    prompt TEXT,
    max_tokens INTEGER DEFAULT 512,
    seed INTEGER DEFAULT 42
)
RETURNS TEXT AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    result TEXT;
BEGIN
    start_time := clock_timestamp();
    
    result := steadytext_generate(prompt, max_tokens, true, seed);
    
    end_time := clock_timestamp();
    
    -- Log the usage
    PERFORM log_steadytext_usage(
        'steadytext_generate',
        prompt,
        result,
        start_time,
        end_time,
        false, -- We'd need to modify steadytext to return cache hit info
        seed
    );
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Performance analytics views
CREATE VIEW performance_summary AS
SELECT 
    function_name,
    COUNT(*) as total_calls,
    AVG(processing_time_ms) as avg_time_ms,
    MIN(processing_time_ms) as min_time_ms,
    MAX(processing_time_ms) as max_time_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY processing_time_ms) as median_time_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY processing_time_ms) as p95_time_ms,
    AVG(input_length) as avg_input_length,
    AVG(output_length) as avg_output_length,
    COUNT(*) FILTER (WHERE cache_hit) as cache_hits,
    ROUND(100.0 * COUNT(*) FILTER (WHERE cache_hit) / COUNT(*), 2) as cache_hit_rate_pct
FROM api_usage_logs
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY function_name;

CREATE VIEW hourly_usage AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    function_name,
    COUNT(*) as calls,
    AVG(processing_time_ms) as avg_time_ms,
    COUNT(*) FILTER (WHERE cache_hit) as cache_hits
FROM api_usage_logs
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', created_at), function_name
ORDER BY hour DESC;

-- Batch optimization function
CREATE OR REPLACE FUNCTION optimize_batch_processing(
    prompts TEXT[],
    base_seed INTEGER DEFAULT 42
)
RETURNS TABLE(
    prompt_index INTEGER,
    generated_text TEXT,
    processing_time_ms INTEGER
) AS $$
DECLARE
    prompt_text TEXT;
    result TEXT;
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    i INTEGER := 1;
BEGIN
    FOREACH prompt_text IN ARRAY prompts LOOP
        start_time := clock_timestamp();
        
        result := steadytext_generate(
            prompt_text,
            max_tokens := 200,
            seed := base_seed + i
        );
        
        end_time := clock_timestamp();
        
        prompt_index := i;
        generated_text := result;
        processing_time_ms := EXTRACT(epoch FROM (end_time - start_time)) * 1000;
        
        RETURN NEXT;
        i := i + 1;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Usage examples and performance testing
SELECT * FROM performance_summary;
SELECT * FROM hourly_usage LIMIT 10;

-- Test batch optimization
SELECT * FROM optimize_batch_processing(ARRAY[
    'Explain machine learning',
    'Describe quantum computing',
    'What is blockchain technology',
    'How does AI work',
    'Define neural networks'
], 70000);

-- Cache warming function
CREATE OR REPLACE FUNCTION warm_cache(
    common_prompts TEXT[],
    base_seed INTEGER DEFAULT 42
)
RETURNS TABLE(
    prompt TEXT,
    cached BOOLEAN
) AS $$
DECLARE
    prompt_text TEXT;
    result TEXT;
BEGIN
    FOREACH prompt_text IN ARRAY common_prompts LOOP
        -- Generate to populate cache
        result := steadytext_generate(prompt_text, max_tokens := 100, seed := base_seed);
        
        prompt := prompt_text;
        cached := true;
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Warm cache with common prompts
SELECT * FROM warm_cache(ARRAY[
    'Write a professional email',
    'Create a product summary',
    'Explain a technical concept',
    'Generate a brief description'
]);

-- Performance monitoring dashboard query
WITH recent_stats AS (
    SELECT 
        function_name,
        COUNT(*) as calls_last_hour,
        AVG(processing_time_ms) as avg_time_last_hour,
        COUNT(*) FILTER (WHERE cache_hit) as cache_hits_last_hour
    FROM api_usage_logs 
    WHERE created_at > NOW() - INTERVAL '1 hour'
    GROUP BY function_name
),
daily_stats AS (
    SELECT 
        function_name,
        COUNT(*) as calls_last_day,
        AVG(processing_time_ms) as avg_time_last_day
    FROM api_usage_logs 
    WHERE created_at > NOW() - INTERVAL '24 hours'
    GROUP BY function_name
)
SELECT 
    COALESCE(r.function_name, d.function_name) as function_name,
    COALESCE(r.calls_last_hour, 0) as calls_last_hour,
    COALESCE(d.calls_last_day, 0) as calls_last_day,
    COALESCE(r.avg_time_last_hour, 0) as avg_time_last_hour,
    COALESCE(d.avg_time_last_day, 0) as avg_time_last_day,
    COALESCE(r.cache_hits_last_hour, 0) as cache_hits_last_hour,
    CASE 
        WHEN r.calls_last_hour > 0 THEN 
            ROUND(100.0 * r.cache_hits_last_hour / r.calls_last_hour, 2)
        ELSE 0
    END as cache_hit_rate_pct
FROM recent_stats r
FULL OUTER JOIN daily_stats d ON r.function_name = d.function_name
ORDER BY calls_last_day DESC;
```

## Structured Generation Examples

SteadyText v2.4.1+ supports structured text generation using native llama.cpp grammars. Generate JSON, regex-constrained text, or multiple-choice responses directly in PostgreSQL.

### JSON Generation

```sql
-- Generate structured user profiles
CREATE TABLE user_profiles (
    id SERIAL PRIMARY KEY,
    user_data JSONB,
    generated_at TIMESTAMP DEFAULT NOW()
);

-- Function to generate user profiles
CREATE OR REPLACE FUNCTION generate_user_profile(
    description TEXT,
    profile_seed INTEGER DEFAULT 42
)
RETURNS JSONB AS $$
DECLARE
    schema JSONB;
    result TEXT;
    parsed_json JSONB;
BEGIN
    -- Define JSON schema
    schema := '{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
            "occupation": {"type": "string"},
            "interests": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }'::jsonb;
    
    -- Generate JSON with error handling
    result := steadytext_generate_json(
        'Create a user profile: ' || description,
        schema,
        max_tokens := 200,
        seed := profile_seed
    );
    
    IF result IS NOT NULL THEN
        -- Extract JSON from the output (it contains <json-output>...</json-output>)
        result := substring(result FROM '<json-output>(.*)</json-output>');
        parsed_json := result::jsonb;
    ELSE
        -- Fallback on generation failure
        parsed_json := jsonb_build_object(
            'name', 'Unknown User',
            'age', 0,
            'email', 'unknown@example.com',
            'error', 'Generation failed'
        );
    END IF;
    
    RETURN parsed_json;
END;
$$ LANGUAGE plpgsql;

-- Generate multiple user profiles
INSERT INTO user_profiles (user_data)
SELECT generate_user_profile(description, seed)
FROM (VALUES 
    ('software engineer who loves hiking', 100),
    ('data scientist interested in ML', 200),
    ('product manager with startup experience', 300)
) AS profiles(description, seed);

-- Generate product catalog with structured data
CREATE OR REPLACE FUNCTION generate_product_listing(
    product_type TEXT,
    listing_seed INTEGER DEFAULT 42
)
RETURNS TABLE(
    name TEXT,
    price NUMERIC,
    specs JSONB,
    description TEXT
) AS $$
DECLARE
    schema JSONB;
    result TEXT;
    product_json JSONB;
BEGIN
    schema := '{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "specs": {
                "type": "object",
                "properties": {
                    "weight": {"type": "string"},
                    "dimensions": {"type": "string"},
                    "material": {"type": "string"}
                }
            }
        }
    }'::jsonb;
    
    result := steadytext_generate_json(
        format('Create a %s product listing', product_type),
        schema,
        seed := listing_seed
    );
    
    IF result IS NOT NULL THEN
        result := substring(result FROM '<json-output>(.*)</json-output>');
        product_json := result::jsonb;
        
        name := product_json->>'name';
        price := (product_json->>'price')::numeric;
        specs := product_json->'specs';
        
        -- Generate description separately
        description := steadytext_generate(
            format('Write a brief description for %s', product_json->>'name'),
            max_tokens := 100,
            seed := listing_seed + 1000
        );
    END IF;
    
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;
```

### Regex-Constrained Generation

```sql
-- Generate formatted data matching patterns
CREATE TABLE formatted_data (
    id SERIAL PRIMARY KEY,
    data_type TEXT,
    formatted_value TEXT,
    pattern_used TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Phone number generation
INSERT INTO formatted_data (data_type, formatted_value, pattern_used)
SELECT 
    'phone',
    steadytext_generate_regex(
        'Customer support number: ',
        '\(\d{3}\) \d{3}-\d{4}',
        seed := 1000 + generate_series
    ),
    '\(\d{3}\) \d{3}-\d{4}'
FROM generate_series(1, 5);

-- Generate SKUs with specific format
INSERT INTO formatted_data (data_type, formatted_value, pattern_used)
SELECT 
    'sku',
    steadytext_generate_regex(
        'Product SKU: ',
        '[A-Z]{3}-\d{4}-[A-Z]\d{2}',
        seed := 2000 + generate_series
    ),
    '[A-Z]{3}-\d{4}-[A-Z]\d{2}'
FROM generate_series(1, 10);

-- Generate formatted dates
CREATE OR REPLACE FUNCTION generate_event_schedule(
    num_events INTEGER,
    base_seed INTEGER DEFAULT 42
)
RETURNS TABLE(
    event_name TEXT,
    event_date TEXT,
    event_time TEXT,
    event_code TEXT
) AS $$
BEGIN
    FOR i IN 1..num_events LOOP
        event_name := steadytext_generate(
            'Event name: ',
            max_tokens := 20,
            seed := base_seed + i
        );
        
        event_date := steadytext_generate_regex(
            'Date: ',
            '\d{4}-\d{2}-\d{2}',
            seed := base_seed + i + 1000
        );
        
        event_time := steadytext_generate_regex(
            'Time: ',
            '\d{2}:\d{2}',
            seed := base_seed + i + 2000
        );
        
        event_code := steadytext_generate_regex(
            'Code: ',
            'EVT-[A-Z]{2}\d{3}',
            seed := base_seed + i + 3000
        );
        
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

### Multiple Choice Generation

```sql
-- Sentiment analysis with choices
CREATE TABLE review_sentiments (
    id SERIAL PRIMARY KEY,
    review_text TEXT,
    sentiment TEXT,
    confidence_level TEXT,
    analyzed_at TIMESTAMP DEFAULT NOW()
);

-- Analyze sentiments
INSERT INTO review_sentiments (review_text, sentiment, confidence_level)
SELECT 
    review,
    steadytext_generate_choice(
        'Sentiment of this review: ' || review,
        ARRAY['positive', 'negative', 'neutral'],
        seed := 5000 + ROW_NUMBER() OVER()
    ) AS sentiment,
    steadytext_generate_choice(
        'Confidence level for sentiment analysis: ' || review,
        ARRAY['high', 'medium', 'low'],
        seed := 6000 + ROW_NUMBER() OVER()
    ) AS confidence_level
FROM (VALUES 
    ('This product exceeded my expectations! Highly recommend.'),
    ('Terrible quality, broke after one day of use.'),
    ('It works as described, nothing special.')
) AS reviews(review);

-- Content categorization
CREATE OR REPLACE FUNCTION categorize_content(
    content TEXT,
    categories TEXT[],
    cat_seed INTEGER DEFAULT 42
)
RETURNS TABLE(
    primary_category TEXT,
    secondary_category TEXT,
    content_type TEXT
) AS $$
BEGIN
    primary_category := steadytext_generate_choice(
        'Primary category for: ' || content,
        categories,
        seed := cat_seed
    );
    
    secondary_category := steadytext_generate_choice(
        'Secondary category for: ' || content,
        categories,
        seed := cat_seed + 1000
    );
    
    content_type := steadytext_generate_choice(
        'Content type: ' || content,
        ARRAY['article', 'tutorial', 'news', 'opinion', 'guide'],
        seed := cat_seed + 2000
    );
    
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;
```

## AI Summarization Examples

Leverage AI-powered summarization for intelligent data aggregation and insights extraction.

### Document Summarization

```sql
-- Create a document repository with AI summaries
CREATE TABLE documents_with_ai (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    summary TEXT,
    key_facts TEXT[],
    category TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Populate with AI-generated summaries
INSERT INTO documents_with_ai (title, content, summary, key_facts, category)
SELECT 
    title,
    content,
    ai_summarize_text(content, jsonb_build_object('doc_type', 'technical')) AS summary,
    ai_extract_facts(content, 5) AS key_facts,
    steadytext_generate_choice(
        'Document category: ' || title,
        ARRAY['technical', 'business', 'research', 'tutorial'],
        seed := 7000 + id
    ) AS category
FROM source_documents
WHERE length(content) > 500;

-- Create materialized view for category summaries
CREATE MATERIALIZED VIEW category_summaries AS
SELECT 
    category,
    count(*) AS document_count,
    ai_summarize(
        content,
        jsonb_build_object(
            'category', category,
            'doc_count', count(*) OVER (PARTITION BY category)
        )
    ) AS category_overview,
    array_agg(DISTINCT key_facts) AS all_key_facts
FROM documents_with_ai
GROUP BY category;

-- Real-time log analysis with AI
CREATE TABLE system_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    service_name TEXT,
    log_level TEXT,
    message TEXT,
    metadata JSONB
);

-- Create hourly log summaries
CREATE OR REPLACE VIEW hourly_log_analysis AS
SELECT 
    date_trunc('hour', timestamp) AS hour,
    service_name,
    log_level,
    count(*) AS log_count,
    ai_summarize(
        message,
        jsonb_build_object(
            'service', service_name,
            'level', log_level,
            'count', count(*)
        )
    ) AS hour_summary,
    array_agg(DISTINCT 
        CASE 
            WHEN log_level IN ('ERROR', 'CRITICAL') 
            THEN message 
        END
    ) FILTER (WHERE log_level IN ('ERROR', 'CRITICAL')) AS critical_messages
FROM system_logs
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY date_trunc('hour', timestamp), service_name, log_level;
```

### TimescaleDB Integration

```sql
-- Enable TimescaleDB (if available)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create hypertable for time-series data
CREATE TABLE metrics (
    time TIMESTAMPTZ NOT NULL,
    device_id TEXT,
    metric_name TEXT,
    value NUMERIC,
    description TEXT
);

SELECT create_hypertable('metrics', 'time');

-- Create continuous aggregate with AI summarization
CREATE MATERIALIZED VIEW daily_metric_summaries
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS day,
    device_id,
    metric_name,
    avg(value) AS avg_value,
    min(value) AS min_value,
    max(value) AS max_value,
    count(*) AS data_points,
    ai_summarize_partial(
        description,
        jsonb_build_object(
            'metric', metric_name,
            'device', device_id,
            'stats', jsonb_build_object(
                'avg', avg(value),
                'min', min(value),
                'max', max(value)
            )
        )
    ) AS partial_summary
FROM metrics
GROUP BY day, device_id, metric_name;

-- Query with final summarization
CREATE OR REPLACE VIEW weekly_insights AS
SELECT 
    time_bucket('1 week', day) AS week,
    device_id,
    ai_summarize_final(partial_summary) AS weekly_summary,
    jsonb_agg(
        jsonb_build_object(
            'metric', metric_name,
            'avg', avg_value,
            'min', min_value,
            'max', max_value
        )
    ) AS metric_stats
FROM daily_metric_summaries
WHERE day >= NOW() - INTERVAL '4 weeks'
GROUP BY week, device_id;
```

### Intelligent Report Generation

```sql
-- Generate executive summaries from multiple data sources
CREATE OR REPLACE FUNCTION generate_executive_report(
    report_date DATE DEFAULT CURRENT_DATE
)
RETURNS TABLE(
    section TEXT,
    summary TEXT,
    key_metrics JSONB,
    recommendations TEXT[]
) AS $$
BEGIN
    -- Sales summary
    section := 'Sales Performance';
    SELECT 
        ai_summarize_text(
            string_agg(
                format('Product %s: %s units sold for $%s', 
                    product_name, units_sold, revenue),
                '; '
            ),
            jsonb_build_object('report_type', 'sales', 'date', report_date)
        ),
        jsonb_agg(
            jsonb_build_object(
                'product', product_name,
                'units', units_sold,
                'revenue', revenue
            )
        )
    INTO summary, key_metrics
    FROM sales_data
    WHERE date = report_date;
    
    recommendations := ai_extract_facts(
        format('Based on sales data: %s. What actions should be taken?', summary),
        3
    );
    
    RETURN NEXT;
    
    -- Customer feedback summary
    section := 'Customer Feedback';
    SELECT 
        ai_summarize(
            feedback_text,
            jsonb_build_object(
                'sentiment', sentiment_score,
                'category', feedback_category
            )
        ),
        jsonb_build_object(
            'avg_sentiment', avg(sentiment_score),
            'total_feedback', count(*),
            'categories', jsonb_agg(DISTINCT feedback_category)
        )
    INTO summary, key_metrics
    FROM customer_feedback
    WHERE date_trunc('day', created_at) = report_date
    GROUP BY date_trunc('day', created_at);
    
    recommendations := ai_extract_facts(
        format('Customer feedback summary: %s. Recommended improvements?', summary),
        4
    );
    
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;
```

## Async Functions Examples

Build high-performance applications with non-blocking AI operations.

### Basic Async Operations

```sql
-- Create a task queue for async processing
CREATE TABLE ai_tasks (
    id SERIAL PRIMARY KEY,
    request_id UUID,
    task_type TEXT,
    input_data JSONB,
    status TEXT DEFAULT 'pending',
    result TEXT,
    error TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Function to process multiple prompts asynchronously
CREATE OR REPLACE FUNCTION process_bulk_content(
    prompts TEXT[],
    task_type TEXT DEFAULT 'generation'
)
RETURNS TABLE(
    task_id INTEGER,
    request_id UUID,
    prompt TEXT
) AS $$
DECLARE
    prompt_text TEXT;
    req_id UUID;
    task_id_val INTEGER;
BEGIN
    FOREACH prompt_text IN ARRAY prompts LOOP
        -- Start async generation
        req_id := steadytext_generate_async(
            prompt_text,
            max_tokens := 200,
            priority := 5
        );
        
        -- Store in task queue
        INSERT INTO ai_tasks (request_id, task_type, input_data)
        VALUES (req_id, task_type, jsonb_build_object('prompt', prompt_text))
        RETURNING id INTO task_id_val;
        
        task_id := task_id_val;
        request_id := req_id;
        prompt := prompt_text;
        
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Process results with retry logic
CREATE OR REPLACE FUNCTION collect_async_results(
    timeout_seconds INTEGER DEFAULT 30
)
RETURNS INTEGER AS $$
DECLARE
    pending_task RECORD;
    result_data RECORD;
    completed_count INTEGER := 0;
BEGIN
    FOR pending_task IN 
        SELECT id, request_id 
        FROM ai_tasks 
        WHERE status = 'pending'
    LOOP
        -- Check async status
        SELECT * INTO result_data
        FROM steadytext_check_async(pending_task.request_id);
        
        IF result_data.status = 'completed' THEN
            UPDATE ai_tasks
            SET status = 'completed',
                result = result_data.result,
                completed_at = NOW()
            WHERE id = pending_task.id;
            
            completed_count := completed_count + 1;
        ELSIF result_data.status = 'failed' THEN
            UPDATE ai_tasks
            SET status = 'failed',
                error = result_data.error,
                completed_at = NOW()
            WHERE id = pending_task.id;
        END IF;
    END LOOP;
    
    RETURN completed_count;
END;
$$ LANGUAGE plpgsql;
```

### Advanced Async Patterns

```sql
-- Async document processing pipeline
CREATE TABLE document_pipeline (
    id SERIAL PRIMARY KEY,
    document_id INTEGER,
    stage TEXT,
    request_id UUID,
    result TEXT,
    next_stage TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Multi-stage async processing
CREATE OR REPLACE FUNCTION process_document_async(
    doc_id INTEGER,
    doc_content TEXT
)
RETURNS VOID AS $$
DECLARE
    summary_req UUID;
    facts_req UUID;
    category_req UUID;
BEGIN
    -- Stage 1: Generate summary
    summary_req := steadytext_generate_async(
        'Summarize: ' || doc_content,
        max_tokens := 150,
        priority := 8
    );
    
    INSERT INTO document_pipeline (document_id, stage, request_id, next_stage)
    VALUES (doc_id, 'summary', summary_req, 'facts');
    
    -- Stage 2: Extract facts (higher priority)
    facts_req := steadytext_generate_async(
        'Extract 5 key facts from: ' || doc_content,
        max_tokens := 100,
        priority := 9
    );
    
    INSERT INTO document_pipeline (document_id, stage, request_id, next_stage)
    VALUES (doc_id, 'facts', facts_req, 'category');
    
    -- Stage 3: Categorize
    category_req := steadytext_generate_choice_async(
        'Category for document: ' || substring(doc_content, 1, 200),
        ARRAY['technical', 'business', 'research', 'general'],
        priority := 7
    );
    
    INSERT INTO document_pipeline (document_id, stage, request_id, next_stage)
    VALUES (doc_id, 'category', category_req, 'complete');
END;
$$ LANGUAGE plpgsql;

-- LISTEN/NOTIFY based processing
CREATE OR REPLACE FUNCTION setup_async_notifications()
RETURNS VOID AS $$
BEGIN
    -- Create notification trigger
    CREATE OR REPLACE FUNCTION notify_async_complete()
    RETURNS TRIGGER AS $trigger$
    BEGIN
        IF NEW.status = 'completed' AND OLD.status != 'completed' THEN
            PERFORM pg_notify(
                'ai_task_complete',
                json_build_object(
                    'task_id', NEW.id,
                    'request_id', NEW.request_id,
                    'task_type', NEW.task_type
                )::text
            );
        END IF;
        RETURN NEW;
    END;
    $trigger$ LANGUAGE plpgsql;
    
    -- Attach trigger
    DROP TRIGGER IF EXISTS async_complete_notify ON ai_tasks;
    CREATE TRIGGER async_complete_notify
        AFTER UPDATE ON ai_tasks
        FOR EACH ROW
        EXECUTE FUNCTION notify_async_complete();
END;
$$ LANGUAGE plpgsql;

-- Batch embedding generation with progress tracking
CREATE OR REPLACE FUNCTION generate_embeddings_async(
    texts TEXT[],
    batch_size INTEGER DEFAULT 10
)
RETURNS TABLE(
    batch_id INTEGER,
    request_ids UUID[],
    text_count INTEGER
) AS $$
DECLARE
    batch_texts TEXT[];
    batch_requests UUID[];
    start_idx INTEGER := 1;
    end_idx INTEGER;
    batch_num INTEGER := 1;
BEGIN
    WHILE start_idx <= array_length(texts, 1) LOOP
        end_idx := LEAST(start_idx + batch_size - 1, array_length(texts, 1));
        batch_texts := texts[start_idx:end_idx];
        
        -- Generate embeddings for batch
        batch_requests := steadytext_embed_batch_async(batch_texts);
        
        -- Store batch info
        INSERT INTO embedding_batches (batch_id, request_ids, status)
        VALUES (batch_num, batch_requests, 'processing');
        
        batch_id := batch_num;
        request_ids := batch_requests;
        text_count := array_length(batch_texts, 1);
        
        RETURN NEXT;
        
        start_idx := end_idx + 1;
        batch_num := batch_num + 1;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Worker management
CREATE OR REPLACE FUNCTION manage_async_workers()
RETURNS TABLE(
    action TEXT,
    status TEXT,
    details JSONB
) AS $$
DECLARE
    worker_status RECORD;
BEGIN
    -- Check worker status
    SELECT * INTO worker_status FROM steadytext_worker_status();
    
    action := 'check_status';
    status := worker_status.status;
    details := to_jsonb(worker_status);
    RETURN NEXT;
    
    -- Start worker if not running
    IF worker_status.status != 'running' THEN
        PERFORM steadytext_worker_start();
        
        action := 'start_worker';
        status := 'started';
        details := jsonb_build_object('message', 'Worker started successfully');
        RETURN NEXT;
    END IF;
    
    -- Configure worker settings
    PERFORM steadytext_config_set('worker_batch_size', '20');
    PERFORM steadytext_config_set('worker_poll_interval_ms', '500');
    
    action := 'configure_worker';
    status := 'configured';
    details := jsonb_build_object(
        'batch_size', 20,
        'poll_interval_ms', 500
    );
    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;
```

### Real-World Async Use Cases

```sql
-- Real-time content moderation system
CREATE TABLE content_moderation_queue (
    id SERIAL PRIMARY KEY,
    content_id INTEGER,
    content_text TEXT,
    moderation_request_id UUID,
    moderation_result TEXT,
    action_taken TEXT,
    processed_at TIMESTAMP
);

CREATE OR REPLACE FUNCTION moderate_content_async(
    content_items JSONB[]
)
RETURNS VOID AS $$
DECLARE
    item JSONB;
    req_id UUID;
BEGIN
    FOREACH item IN ARRAY content_items LOOP
        -- Check content appropriateness
        req_id := steadytext_generate_choice_async(
            format('Is this content appropriate for all audiences? Content: %s', 
                   item->>'text'),
            ARRAY['appropriate', 'needs_review', 'inappropriate'],
            priority := 10  -- High priority for moderation
        );
        
        INSERT INTO content_moderation_queue (
            content_id,
            content_text,
            moderation_request_id
        ) VALUES (
            (item->>'id')::integer,
            item->>'text',
            req_id
        );
    END LOOP;
    
    -- Notify moderation service
    PERFORM pg_notify('content_moderation', 'new_items_queued');
END;
$$ LANGUAGE plpgsql;

-- Async email generation system
CREATE TABLE email_generation_queue (
    id SERIAL PRIMARY KEY,
    recipient_id INTEGER,
    email_type TEXT,
    context JSONB,
    subject_request_id UUID,
    body_request_id UUID,
    subject TEXT,
    body TEXT,
    status TEXT DEFAULT 'queued',
    created_at TIMESTAMP DEFAULT NOW(),
    sent_at TIMESTAMP
);

CREATE OR REPLACE FUNCTION generate_personalized_emails_async(
    recipient_data JSONB
)
RETURNS INTEGER AS $$
DECLARE
    recipient JSONB;
    subject_req UUID;
    body_req UUID;
    email_count INTEGER := 0;
BEGIN
    FOR recipient IN SELECT * FROM jsonb_array_elements(recipient_data)
    LOOP
        -- Generate subject
        subject_req := steadytext_generate_async(
            format('Email subject for %s promotion to %s',
                   recipient->>'product',
                   recipient->>'name'),
            max_tokens := 20,
            priority := 6
        );
        
        -- Generate body
        body_req := steadytext_generate_async(
            format('Professional marketing email about %s for %s who is interested in %s',
                   recipient->>'product',
                   recipient->>'name',
                   recipient->>'interests'),
            max_tokens := 300,
            priority := 5
        );
        
        INSERT INTO email_generation_queue (
            recipient_id,
            email_type,
            context,
            subject_request_id,
            body_request_id
        ) VALUES (
            (recipient->>'id')::integer,
            'marketing',
            recipient,
            subject_req,
            body_req
        );
        
        email_count := email_count + 1;
    END LOOP;
    
    RETURN email_count;
END;
$$ LANGUAGE plpgsql;

-- Process completed emails
CREATE OR REPLACE FUNCTION process_completed_emails()
RETURNS INTEGER AS $$
DECLARE
    email RECORD;
    subject_result RECORD;
    body_result RECORD;
    processed_count INTEGER := 0;
BEGIN
    FOR email IN 
        SELECT * FROM email_generation_queue 
        WHERE status = 'queued'
    LOOP
        -- Check subject generation
        SELECT * INTO subject_result
        FROM steadytext_check_async(email.subject_request_id);
        
        -- Check body generation
        SELECT * INTO body_result
        FROM steadytext_check_async(email.body_request_id);
        
        -- If both completed, update and send
        IF subject_result.status = 'completed' AND 
           body_result.status = 'completed' THEN
            
            UPDATE email_generation_queue
            SET subject = subject_result.result,
                body = body_result.result,
                status = 'ready_to_send'
            WHERE id = email.id;
            
            processed_count := processed_count + 1;
            
            -- Trigger email sending
            PERFORM pg_notify(
                'send_email',
                json_build_object(
                    'email_id', email.id,
                    'recipient_id', email.recipient_id
                )::text
            );
        END IF;
    END LOOP;
    
    RETURN processed_count;
END;
$$ LANGUAGE plpgsql;
```

This comprehensive guide demonstrates how to integrate SteadyText with PostgreSQL for building powerful AI-enhanced applications. The examples cover everything from basic setup to advanced workflows, providing a solid foundation for developing production-ready systems that combine structured data with AI-generated content and embeddings.
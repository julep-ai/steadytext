# PostgreSQL Examples: Blog & Content Management

Examples for building blog platforms and content management systems with SteadyText.

## Blog Platform

### Schema Design

```sql
-- Create blog schema
CREATE SCHEMA IF NOT EXISTS blog;

-- Authors table
CREATE TABLE blog.authors (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    bio TEXT,
    bio_embedding vector(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Posts table with AI fields
CREATE TABLE blog.posts (
    id SERIAL PRIMARY KEY,
    author_id INTEGER REFERENCES blog.authors(id),
    title VARCHAR(200) NOT NULL,
    slug VARCHAR(200) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    embedding vector(1024),
    tags TEXT[],
    status VARCHAR(20) DEFAULT 'draft',
    published_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Comments with sentiment analysis
CREATE TABLE blog.comments (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES blog.posts(id),
    author_name VARCHAR(100),
    content TEXT NOT NULL,
    sentiment FLOAT,
    embedding vector(1024),
    is_spam BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Categories with embeddings
CREATE TABLE blog.categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    embedding vector(1024)
);

-- Post categories junction
CREATE TABLE blog.post_categories (
    post_id INTEGER REFERENCES blog.posts(id),
    category_id INTEGER REFERENCES blog.categories(id),
    PRIMARY KEY (post_id, category_id)
);
```

### Content Generation Functions

```sql
-- Generate blog post summary
CREATE OR REPLACE FUNCTION blog.generate_post_summary(
    p_content TEXT,
    p_max_words INTEGER DEFAULT 50
) RETURNS TEXT AS $$
DECLARE
    v_prompt TEXT;
    v_summary TEXT;
BEGIN
    v_prompt := format(
        'Summarize this blog post in approximately %s words: %s',
        p_max_words,
        substring(p_content, 1, 2000)
    );
    
    v_summary := steadytext_generate(v_prompt, 150);
    
    IF v_summary IS NULL THEN
        -- Fallback to simple extraction
        v_summary := substring(p_content, 1, 200) || '...';
    END IF;
    
    RETURN v_summary;
END;
$$ LANGUAGE plpgsql;

-- Generate SEO-friendly slug
CREATE OR REPLACE FUNCTION blog.generate_slug(
    p_title TEXT
) RETURNS TEXT AS $$
DECLARE
    v_prompt TEXT;
    v_slug TEXT;
BEGIN
    v_prompt := format(
        'Convert this title to a URL-friendly slug (lowercase, hyphens, no special chars): "%s"',
        p_title
    );
    
    v_slug := steadytext_generate(v_prompt, 50);
    
    IF v_slug IS NULL THEN
        -- Fallback to regex-based conversion
        v_slug := lower(p_title);
        v_slug := regexp_replace(v_slug, '[^a-z0-9]+', '-', 'g');
        v_slug := trim(both '-' from v_slug);
    END IF;
    
    RETURN v_slug;
END;
$$ LANGUAGE plpgsql;

-- Generate tags for post
CREATE OR REPLACE FUNCTION blog.generate_tags(
    p_content TEXT,
    p_max_tags INTEGER DEFAULT 5
) RETURNS TEXT[] AS $$
DECLARE
    v_prompt TEXT;
    v_tags_text TEXT;
    v_tags TEXT[];
BEGIN
    v_prompt := format(
        'Extract %s relevant tags from this content as a comma-separated list: %s',
        p_max_tags,
        substring(p_content, 1, 1000)
    );
    
    v_tags_text := steadytext_generate(v_prompt, 100);
    
    IF v_tags_text IS NOT NULL THEN
        v_tags := string_to_array(
            regexp_replace(v_tags_text, '[\s,]+', ',', 'g'),
            ','
        );
    ELSE
        v_tags := ARRAY[]::TEXT[];
    END IF;
    
    RETURN v_tags;
END;
$$ LANGUAGE plpgsql;
```

### Automated Content Processing

```sql
-- Trigger to auto-generate content on insert/update
CREATE OR REPLACE FUNCTION blog.process_post()
RETURNS TRIGGER AS $$
BEGIN
    -- Generate summary if not provided
    IF NEW.summary IS NULL OR NEW.summary = '' THEN
        NEW.summary := blog.generate_post_summary(NEW.content);
    END IF;
    
    -- Generate slug if not provided
    IF NEW.slug IS NULL OR NEW.slug = '' THEN
        NEW.slug := blog.generate_slug(NEW.title);
        -- Ensure uniqueness
        WHILE EXISTS (SELECT 1 FROM blog.posts WHERE slug = NEW.slug AND id != COALESCE(NEW.id, -1)) LOOP
            NEW.slug := NEW.slug || '-' || floor(random() * 1000)::text;
        END LOOP;
    END IF;
    
    -- Generate embedding
    NEW.embedding := steadytext_embed(
        NEW.title || ' ' || COALESCE(NEW.summary, '') || ' ' || substring(NEW.content, 1, 1000)
    );
    
    -- Generate tags if empty
    IF array_length(NEW.tags, 1) IS NULL THEN
        NEW.tags := blog.generate_tags(NEW.content);
    END IF;
    
    -- Update timestamp
    NEW.updated_at := CURRENT_TIMESTAMP;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER blog_post_process
    BEFORE INSERT OR UPDATE ON blog.posts
    FOR EACH ROW
    EXECUTE FUNCTION blog.process_post();
```

### Content Recommendation System

```sql
-- Find related posts
CREATE OR REPLACE FUNCTION blog.find_related_posts(
    p_post_id INTEGER,
    p_limit INTEGER DEFAULT 5
) RETURNS TABLE(
    post_id INTEGER,
    title VARCHAR(200),
    similarity FLOAT,
    reason TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH current_post AS (
        SELECT embedding, tags
        FROM blog.posts
        WHERE id = p_post_id
    )
    SELECT 
        p.id as post_id,
        p.title,
        1 - (p.embedding <-> cp.embedding) as similarity,
        CASE 
            WHEN array_length(array_intersect(p.tags, cp.tags), 1) > 0 
            THEN 'Similar tags: ' || array_to_string(array_intersect(p.tags, cp.tags), ', ')
            ELSE 'Similar content'
        END as reason
    FROM blog.posts p, current_post cp
    WHERE p.id != p_post_id
        AND p.status = 'published'
    ORDER BY similarity DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Generate content recommendations
CREATE OR REPLACE FUNCTION blog.generate_recommendations(
    p_user_id INTEGER,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE(
    post_id INTEGER,
    title VARCHAR(200),
    score FLOAT,
    recommendation_reason TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH user_interests AS (
        -- Get user's reading history
        SELECT 
            p.embedding,
            p.tags,
            COUNT(*) as read_count
        FROM blog.posts p
        JOIN blog.post_views pv ON p.id = pv.post_id
        WHERE pv.user_id = p_user_id
        GROUP BY p.id
    ),
    interest_profile AS (
        -- Create aggregate interest profile
        SELECT 
            pg_stat_avg_vector(embedding) as avg_embedding,
            array_agg(DISTINCT tag) as all_tags
        FROM user_interests, unnest(tags) as tag
    )
    SELECT 
        p.id as post_id,
        p.title,
        (
            0.7 * (1 - (p.embedding <-> ip.avg_embedding)) +
            0.3 * (array_length(array_intersect(p.tags, ip.all_tags), 1)::float / array_length(p.tags, 1))
        ) as score,
        'Based on your reading history' as recommendation_reason
    FROM blog.posts p, interest_profile ip
    WHERE p.status = 'published'
        AND p.id NOT IN (
            SELECT post_id FROM blog.post_views WHERE user_id = p_user_id
        )
    ORDER BY score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;
```

### Comment Moderation

```sql
-- Analyze comment sentiment and spam detection
CREATE OR REPLACE FUNCTION blog.analyze_comment(
    p_content TEXT
) RETURNS TABLE(
    sentiment FLOAT,
    is_spam BOOLEAN,
    reason TEXT
) AS $$
DECLARE
    v_sentiment_prompt TEXT;
    v_spam_prompt TEXT;
    v_sentiment_result TEXT;
    v_spam_result TEXT;
BEGIN
    -- Sentiment analysis
    v_sentiment_prompt := format(
        'Rate the sentiment of this comment from -1 (very negative) to 1 (very positive), return only the number: %s',
        p_content
    );
    
    v_sentiment_result := steadytext_generate(v_sentiment_prompt, 10);
    
    -- Spam detection
    v_spam_prompt := format(
        'Is this comment spam? Reply only "yes" or "no": %s',
        p_content
    );
    
    v_spam_result := steadytext_generate_choice(
        v_spam_prompt,
        ARRAY['yes', 'no']
    );
    
    RETURN QUERY
    SELECT 
        CASE 
            WHEN v_sentiment_result ~ '^-?[0-9]*\.?[0-9]+$' 
            THEN v_sentiment_result::FLOAT
            ELSE 0.0
        END,
        v_spam_result = 'yes',
        CASE 
            WHEN v_spam_result = 'yes' THEN 'Detected as spam'
            WHEN v_sentiment_result::FLOAT < -0.5 THEN 'Very negative sentiment'
            ELSE 'Approved'
        END;
END;
$$ LANGUAGE plpgsql;

-- Auto-moderate comments
CREATE OR REPLACE FUNCTION blog.moderate_comment()
RETURNS TRIGGER AS $$
DECLARE
    v_analysis RECORD;
BEGIN
    -- Analyze comment
    SELECT * INTO v_analysis
    FROM blog.analyze_comment(NEW.content);
    
    NEW.sentiment := v_analysis.sentiment;
    NEW.is_spam := v_analysis.is_spam;
    
    -- Generate embedding for similarity search
    NEW.embedding := steadytext_embed(NEW.content);
    
    -- Auto-approve or flag for review
    IF v_analysis.is_spam OR v_analysis.sentiment < -0.7 THEN
        -- Could implement notification system here
        RAISE NOTICE 'Comment flagged for review: %', v_analysis.reason;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER comment_moderation
    BEFORE INSERT ON blog.comments
    FOR EACH ROW
    EXECUTE FUNCTION blog.moderate_comment();
```

### Content Analytics

```sql
-- Analyze content performance
CREATE OR REPLACE FUNCTION blog.analyze_post_performance(
    p_days INTEGER DEFAULT 30
) RETURNS TABLE(
    post_id INTEGER,
    title VARCHAR(200),
    views BIGINT,
    avg_time_on_page INTERVAL,
    bounce_rate FLOAT,
    sentiment_score FLOAT,
    engagement_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH post_metrics AS (
        SELECT 
            pv.post_id,
            COUNT(*) as view_count,
            AVG(pv.time_spent) as avg_time,
            SUM(CASE WHEN pv.bounced THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as bounce,
            AVG(c.sentiment) as avg_sentiment,
            COUNT(DISTINCT c.id) as comment_count
        FROM blog.post_views pv
        LEFT JOIN blog.comments c ON c.post_id = pv.post_id
        WHERE pv.viewed_at > CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days
        GROUP BY pv.post_id
    )
    SELECT 
        p.id,
        p.title,
        pm.view_count,
        pm.avg_time,
        pm.bounce,
        COALESCE(pm.avg_sentiment, 0),
        (
            0.4 * (pm.view_count::FLOAT / NULLIF(MAX(pm.view_count) OVER (), 0)) +
            0.3 * (extract(epoch from pm.avg_time) / NULLIF(MAX(extract(epoch from pm.avg_time)) OVER (), 0)) +
            0.2 * (1 - pm.bounce) +
            0.1 * ((pm.avg_sentiment + 1) / 2)
        ) as engagement_score
    FROM blog.posts p
    JOIN post_metrics pm ON p.id = pm.post_id
    ORDER BY engagement_score DESC;
END;
$$ LANGUAGE plpgsql;

-- Generate content insights
CREATE OR REPLACE FUNCTION blog.generate_content_insights(
    p_post_id INTEGER
) RETURNS TEXT AS $$
DECLARE
    v_metrics RECORD;
    v_prompt TEXT;
    v_insights TEXT;
BEGIN
    -- Get post metrics
    SELECT * INTO v_metrics
    FROM blog.analyze_post_performance(30)
    WHERE post_id = p_post_id;
    
    -- Generate insights
    v_prompt := format(
        'Analyze these blog post metrics and provide insights: Views: %s, Avg time: %s, Bounce rate: %s%%, Sentiment: %s. What does this suggest about the content?',
        v_metrics.views,
        v_metrics.avg_time_on_page,
        round(v_metrics.bounce_rate * 100),
        round(v_metrics.sentiment_score::numeric, 2)
    );
    
    v_insights := steadytext_generate(v_prompt, 200);
    
    RETURN COALESCE(v_insights, 'Insufficient data for insights.');
END;
$$ LANGUAGE plpgsql;
```

### RSS Feed Generation

```sql
-- Generate RSS feed with AI summaries
CREATE OR REPLACE FUNCTION blog.generate_rss_feed(
    p_limit INTEGER DEFAULT 20
) RETURNS XML AS $$
DECLARE
    v_feed XML;
BEGIN
    v_feed := xmlelement(
        name rss,
        xmlattributes('2.0' as version),
        xmlelement(
            name channel,
            xmlelement(name title, 'My Blog'),
            xmlelement(name link, 'https://myblog.com'),
            xmlelement(name description, 'Latest posts from My Blog'),
            xmlelement(name language, 'en-us'),
            xmlelement(name lastBuildDate, to_char(CURRENT_TIMESTAMP, 'Dy, DD Mon YYYY HH24:MI:SS TZ')),
            (
                SELECT xmlagg(
                    xmlelement(
                        name item,
                        xmlelement(name title, p.title),
                        xmlelement(name link, 'https://myblog.com/posts/' || p.slug),
                        xmlelement(name description, xmlcdata(p.summary)),
                        xmlelement(name pubDate, to_char(p.published_at, 'Dy, DD Mon YYYY HH24:MI:SS TZ')),
                        xmlelement(name guid, 'https://myblog.com/posts/' || p.id),
                        (
                            SELECT xmlagg(
                                xmlelement(name category, tag)
                            )
                            FROM unnest(p.tags) as tag
                        )
                    )
                )
                FROM blog.posts p
                WHERE p.status = 'published'
                ORDER BY p.published_at DESC
                LIMIT p_limit
            )
        )
    );
    
    RETURN v_feed;
END;
$$ LANGUAGE plpgsql;
```

## Content Versioning

```sql
-- Version control for posts
CREATE TABLE blog.post_versions (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES blog.posts(id),
    version_number INTEGER NOT NULL,
    title VARCHAR(200),
    content TEXT,
    summary TEXT,
    change_summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER,
    UNIQUE(post_id, version_number)
);

-- Auto-generate change summaries
CREATE OR REPLACE FUNCTION blog.create_post_version()
RETURNS TRIGGER AS $$
DECLARE
    v_version_number INTEGER;
    v_change_summary TEXT;
    v_old_content TEXT;
BEGIN
    -- Skip if no actual changes
    IF OLD.content = NEW.content AND OLD.title = NEW.title THEN
        RETURN NEW;
    END IF;
    
    -- Get next version number
    SELECT COALESCE(MAX(version_number), 0) + 1
    INTO v_version_number
    FROM blog.post_versions
    WHERE post_id = NEW.id;
    
    -- Generate change summary
    v_change_summary := steadytext_generate(
        format(
            'Summarize the changes between these versions in one sentence: OLD: %s NEW: %s',
            substring(OLD.title || ' ' || OLD.content, 1, 500),
            substring(NEW.title || ' ' || NEW.content, 1, 500)
        ),
        50
    );
    
    -- Insert version record
    INSERT INTO blog.post_versions (
        post_id, version_number, title, content, 
        summary, change_summary
    ) VALUES (
        NEW.id, v_version_number, OLD.title, OLD.content,
        OLD.summary, COALESCE(v_change_summary, 'Content updated')
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER post_versioning
    BEFORE UPDATE ON blog.posts
    FOR EACH ROW
    WHEN (OLD.content IS DISTINCT FROM NEW.content OR OLD.title IS DISTINCT FROM NEW.title)
    EXECUTE FUNCTION blog.create_post_version();
```

## Related Documentation

- [PostgreSQL Extension Overview](../postgresql-extension.md)
- [E-commerce Examples](postgresql-ecommerce.md)
- [Search Examples](postgresql-search.md)
- [Real-time Examples](postgresql-realtime.md)
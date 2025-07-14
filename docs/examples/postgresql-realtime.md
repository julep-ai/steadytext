# PostgreSQL Examples: Real-time Applications

Examples for building real-time applications with AI features using SteadyText and PostgreSQL.

## Chat System with AI Assistance

### Schema Design

```sql
-- Create chat schema
CREATE SCHEMA IF NOT EXISTS chat;

-- Users table
CREATE TABLE chat.users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100),
    status VARCHAR(20) DEFAULT 'offline',
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Channels/Rooms
CREATE TABLE chat.channels (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    channel_type VARCHAR(20) DEFAULT 'public', -- public, private, direct
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Messages with AI fields
CREATE TABLE chat.messages (
    id SERIAL PRIMARY KEY,
    channel_id INTEGER REFERENCES chat.channels(id),
    user_id INTEGER REFERENCES chat.users(id),
    content TEXT NOT NULL,
    message_type VARCHAR(20) DEFAULT 'user', -- user, ai_suggestion, system
    embedding vector(1024),
    sentiment FLOAT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AI conversation memory
CREATE TABLE chat.conversation_memory (
    id SERIAL PRIMARY KEY,
    channel_id INTEGER REFERENCES chat.channels(id),
    summary TEXT,
    key_points TEXT[],
    context_embedding vector(1024),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Message reactions
CREATE TABLE chat.reactions (
    id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES chat.messages(id),
    user_id INTEGER REFERENCES chat.users(id),
    reaction VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(message_id, user_id, reaction)
);
```

### Real-time Triggers and Notifications

```sql
-- Function to notify on new messages
CREATE OR REPLACE FUNCTION chat.notify_new_message()
RETURNS TRIGGER AS $$
DECLARE
    v_payload JSONB;
BEGIN
    v_payload := jsonb_build_object(
        'message_id', NEW.id,
        'channel_id', NEW.channel_id,
        'user_id', NEW.user_id,
        'content', NEW.content,
        'message_type', NEW.message_type,
        'created_at', NEW.created_at
    );
    
    PERFORM pg_notify(
        'chat_message_' || NEW.channel_id,
        v_payload::text
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER notify_message_insert
    AFTER INSERT ON chat.messages
    FOR EACH ROW
    EXECUTE FUNCTION chat.notify_new_message();

-- Function to process messages with AI
CREATE OR REPLACE FUNCTION chat.process_message_ai()
RETURNS TRIGGER AS $$
BEGIN
    -- Generate embedding
    NEW.embedding := steadytext_embed(NEW.content);
    
    -- Analyze sentiment
    NEW.sentiment := chat.analyze_message_sentiment(NEW.content);
    
    -- Check for AI assistance triggers
    IF NEW.content ILIKE '%@ai%' OR NEW.content ILIKE '%help%' THEN
        -- Schedule AI response
        INSERT INTO chat.ai_response_queue (
            message_id,
            channel_id,
            priority,
            created_at
        ) VALUES (
            NEW.id,
            NEW.channel_id,
            CASE WHEN NEW.content ILIKE '%urgent%' THEN 1 ELSE 5 END,
            NOW()
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER process_message_before_insert
    BEFORE INSERT ON chat.messages
    FOR EACH ROW
    WHEN (NEW.message_type = 'user')
    EXECUTE FUNCTION chat.process_message_ai();
```

### AI-Powered Features

```sql
-- Generate AI responses
CREATE OR REPLACE FUNCTION chat.generate_ai_response(
    p_message_id INTEGER
) RETURNS TEXT AS $$
DECLARE
    v_message RECORD;
    v_context TEXT;
    v_prompt TEXT;
    v_response TEXT;
BEGIN
    -- Get message and context
    SELECT 
        m.content,
        m.channel_id,
        c.name as channel_name
    INTO v_message
    FROM chat.messages m
    JOIN chat.channels c ON m.channel_id = c.id
    WHERE m.id = p_message_id;
    
    -- Get conversation context
    SELECT string_agg(
        format('%s: %s', u.username, m.content),
        E'\n'
        ORDER BY m.created_at DESC
    ) INTO v_context
    FROM chat.messages m
    JOIN chat.users u ON m.user_id = u.id
    WHERE m.channel_id = v_message.channel_id
        AND m.created_at > NOW() - INTERVAL '10 minutes'
        AND m.id < p_message_id
    LIMIT 10;
    
    -- Generate response
    v_prompt := format(
        'You are a helpful AI assistant in the "%s" chat. Recent conversation:\n%s\n\nUser asks: %s\n\nProvide a helpful response:',
        v_message.channel_name,
        v_context,
        v_message.content
    );
    
    v_response := steadytext_generate(v_prompt, 150);
    
    RETURN COALESCE(v_response, 'I can help you with that. Could you provide more details?');
END;
$$ LANGUAGE plpgsql;

-- Smart message suggestions
CREATE OR REPLACE FUNCTION chat.suggest_replies(
    p_channel_id INTEGER,
    p_limit INTEGER DEFAULT 3
) RETURNS TABLE(
    suggestion TEXT,
    confidence FLOAT
) AS $$
DECLARE
    v_recent_context TEXT;
    v_prompt TEXT;
    v_suggestions TEXT;
BEGIN
    -- Get recent conversation
    SELECT string_agg(
        content,
        ' '
        ORDER BY created_at DESC
    ) INTO v_recent_context
    FROM (
        SELECT content
        FROM chat.messages
        WHERE channel_id = p_channel_id
        ORDER BY created_at DESC
        LIMIT 5
    ) recent;
    
    -- Generate suggestions
    v_prompt := format(
        'Based on this conversation: "%s", suggest %s possible replies. Return as numbered list:',
        v_recent_context,
        p_limit
    );
    
    v_suggestions := steadytext_generate(v_prompt, 100);
    
    -- Parse suggestions (simplified)
    IF v_suggestions IS NOT NULL THEN
        RETURN QUERY
        SELECT 
            trim(regexp_split_to_table(v_suggestions, E'\n')),
            0.8 + random() * 0.2 -- Mock confidence
        FROM regexp_split_to_table(v_suggestions, E'\n') s
        WHERE length(trim(s)) > 0
        LIMIT p_limit;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Conversation summarization
CREATE OR REPLACE FUNCTION chat.summarize_conversation(
    p_channel_id INTEGER,
    p_hours INTEGER DEFAULT 24
) RETURNS TEXT AS $$
DECLARE
    v_messages TEXT;
    v_prompt TEXT;
    v_summary TEXT;
BEGIN
    -- Get messages
    SELECT string_agg(
        format('%s: %s', u.username, m.content),
        E'\n'
        ORDER BY m.created_at
    ) INTO v_messages
    FROM chat.messages m
    JOIN chat.users u ON m.user_id = u.id
    WHERE m.channel_id = p_channel_id
        AND m.created_at > NOW() - INTERVAL '1 hour' * p_hours;
    
    IF v_messages IS NULL THEN
        RETURN 'No messages in the specified time period.';
    END IF;
    
    -- Generate summary
    v_prompt := format(
        'Summarize this conversation in 3-4 key points:\n%s',
        substring(v_messages, 1, 2000)
    );
    
    v_summary := steadytext_generate(v_prompt, 150);
    
    -- Update conversation memory
    UPDATE chat.conversation_memory
    SET summary = v_summary,
        context_embedding = steadytext_embed(v_summary),
        updated_at = NOW()
    WHERE channel_id = p_channel_id;
    
    IF NOT FOUND THEN
        INSERT INTO chat.conversation_memory (channel_id, summary, context_embedding)
        VALUES (p_channel_id, v_summary, steadytext_embed(v_summary));
    END IF;
    
    RETURN COALESCE(v_summary, 'Unable to generate summary.');
END;
$$ LANGUAGE plpgsql;

-- Sentiment analysis
CREATE OR REPLACE FUNCTION chat.analyze_message_sentiment(
    p_content TEXT
) RETURNS FLOAT AS $$
DECLARE
    v_result TEXT;
    v_sentiment FLOAT;
BEGIN
    v_result := steadytext_generate(
        format('Rate sentiment from -1 to 1 for: "%s". Return only number:', 
               substring(p_content, 1, 200)),
        10
    );
    
    IF v_result ~ '^-?[0-9]*\.?[0-9]+$' THEN
        v_sentiment := v_result::FLOAT;
        RETURN GREATEST(-1, LEAST(1, v_sentiment));
    ELSE
        RETURN 0.0;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

### Message Search and Discovery

```sql
-- Semantic message search
CREATE OR REPLACE FUNCTION chat.search_messages(
    p_query TEXT,
    p_channel_id INTEGER DEFAULT NULL,
    p_user_id INTEGER DEFAULT NULL,
    p_limit INTEGER DEFAULT 20
) RETURNS TABLE(
    message_id INTEGER,
    channel_name VARCHAR(100),
    username VARCHAR(50),
    content TEXT,
    similarity FLOAT,
    created_at TIMESTAMP
) AS $$
DECLARE
    v_query_embedding vector(1024);
BEGIN
    v_query_embedding := steadytext_embed(p_query);
    
    RETURN QUERY
    SELECT 
        m.id as message_id,
        ch.name as channel_name,
        u.username,
        m.content,
        1 - (m.embedding <-> v_query_embedding) as similarity,
        m.created_at
    FROM chat.messages m
    JOIN chat.channels ch ON m.channel_id = ch.id
    JOIN chat.users u ON m.user_id = u.id
    WHERE (p_channel_id IS NULL OR m.channel_id = p_channel_id)
        AND (p_user_id IS NULL OR m.user_id = p_user_id)
        AND m.embedding IS NOT NULL
    ORDER BY m.embedding <-> v_query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Find similar conversations
CREATE OR REPLACE FUNCTION chat.find_similar_conversations(
    p_channel_id INTEGER,
    p_limit INTEGER DEFAULT 5
) RETURNS TABLE(
    channel_id INTEGER,
    channel_name VARCHAR(100),
    similarity FLOAT,
    common_topics TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH current_context AS (
        SELECT 
            context_embedding,
            key_points
        FROM chat.conversation_memory
        WHERE channel_id = p_channel_id
    ),
    similarities AS (
        SELECT 
            cm.channel_id,
            ch.name,
            1 - (cm.context_embedding <-> cc.context_embedding) as similarity,
            cm.key_points
        FROM chat.conversation_memory cm
        JOIN chat.channels ch ON cm.channel_id = ch.id
        CROSS JOIN current_context cc
        WHERE cm.channel_id != p_channel_id
    )
    SELECT 
        s.channel_id,
        s.name as channel_name,
        s.similarity,
        array_intersect(s.key_points, cc.key_points) as common_topics
    FROM similarities s
    CROSS JOIN current_context cc
    ORDER BY s.similarity DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;
```

### Real-time Analytics

```sql
-- Message activity dashboard
CREATE OR REPLACE FUNCTION chat.get_activity_metrics(
    p_interval INTERVAL DEFAULT '1 hour'
) RETURNS TABLE(
    metric_name TEXT,
    metric_value NUMERIC,
    change_percent NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    WITH current_period AS (
        SELECT 
            COUNT(*) as message_count,
            COUNT(DISTINCT user_id) as active_users,
            COUNT(DISTINCT channel_id) as active_channels,
            AVG(sentiment) as avg_sentiment
        FROM chat.messages
        WHERE created_at > NOW() - p_interval
    ),
    previous_period AS (
        SELECT 
            COUNT(*) as message_count,
            COUNT(DISTINCT user_id) as active_users,
            COUNT(DISTINCT channel_id) as active_channels,
            AVG(sentiment) as avg_sentiment
        FROM chat.messages
        WHERE created_at > NOW() - (p_interval * 2)
            AND created_at <= NOW() - p_interval
    )
    SELECT 
        'Messages' as metric_name,
        c.message_count::NUMERIC,
        CASE 
            WHEN p.message_count > 0 
            THEN ((c.message_count - p.message_count)::NUMERIC / p.message_count * 100)
            ELSE 0
        END
    FROM current_period c, previous_period p
    
    UNION ALL
    
    SELECT 
        'Active Users',
        c.active_users::NUMERIC,
        CASE 
            WHEN p.active_users > 0 
            THEN ((c.active_users - p.active_users)::NUMERIC / p.active_users * 100)
            ELSE 0
        END
    FROM current_period c, previous_period p
    
    UNION ALL
    
    SELECT 
        'Average Sentiment',
        ROUND(c.avg_sentiment::NUMERIC, 3),
        CASE 
            WHEN p.avg_sentiment IS NOT NULL 
            THEN ((c.avg_sentiment - p.avg_sentiment)::NUMERIC * 100)
            ELSE 0
        END
    FROM current_period c, previous_period p;
END;
$$ LANGUAGE plpgsql;

-- Real-time trending topics
CREATE OR REPLACE FUNCTION chat.get_trending_topics(
    p_hours INTEGER DEFAULT 1,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE(
    topic TEXT,
    mentions INTEGER,
    channels TEXT[],
    sentiment_avg FLOAT
) AS $$
DECLARE
    v_messages TEXT;
    v_topics_prompt TEXT;
    v_topics_text TEXT;
BEGIN
    -- Aggregate recent messages
    SELECT string_agg(content, ' ') INTO v_messages
    FROM (
        SELECT content
        FROM chat.messages
        WHERE created_at > NOW() - INTERVAL '1 hour' * p_hours
        ORDER BY created_at DESC
        LIMIT 1000
    ) recent;
    
    -- Extract topics using AI
    v_topics_prompt := format(
        'Extract %s trending topics from these messages as comma-separated list: %s',
        p_limit,
        substring(v_messages, 1, 2000)
    );
    
    v_topics_text := steadytext_generate(v_topics_prompt, 100);
    
    IF v_topics_text IS NOT NULL THEN
        -- Analyze each topic
        RETURN QUERY
        WITH topics AS (
            SELECT trim(unnest(string_to_array(v_topics_text, ','))) as topic
        ),
        topic_stats AS (
            SELECT 
                t.topic,
                COUNT(DISTINCT m.id) as mention_count,
                array_agg(DISTINCT ch.name) as channel_list,
                AVG(m.sentiment) as avg_sent
            FROM topics t
            JOIN chat.messages m ON m.content ILIKE '%' || t.topic || '%'
            JOIN chat.channels ch ON m.channel_id = ch.id
            WHERE m.created_at > NOW() - INTERVAL '1 hour' * p_hours
            GROUP BY t.topic
        )
        SELECT 
            topic,
            mention_count::INTEGER,
            channel_list,
            avg_sent
        FROM topic_stats
        ORDER BY mention_count DESC
        LIMIT p_limit;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

### Moderation and Safety

```sql
-- Content moderation
CREATE OR REPLACE FUNCTION chat.moderate_content(
    p_content TEXT
) RETURNS TABLE(
    is_safe BOOLEAN,
    risk_level TEXT,
    reasons TEXT[]
) AS $$
DECLARE
    v_analysis TEXT;
    v_is_safe BOOLEAN := TRUE;
    v_reasons TEXT[] := '{}';
BEGIN
    -- Quick checks
    IF p_content ~* '\y(spam|scam|phishing)\y' THEN
        v_is_safe := FALSE;
        v_reasons := array_append(v_reasons, 'Potential spam/scam content');
    END IF;
    
    -- AI-based analysis
    v_analysis := steadytext_generate_choice(
        format('Is this message safe for a public chat? "%s" Answer:', 
               substring(p_content, 1, 200)),
        ARRAY['safe', 'warning', 'unsafe']
    );
    
    RETURN QUERY
    SELECT 
        v_analysis != 'unsafe',
        v_analysis,
        CASE 
            WHEN v_analysis = 'unsafe' THEN array_append(v_reasons, 'AI flagged as unsafe')
            WHEN v_analysis = 'warning' THEN array_append(v_reasons, 'Requires review')
            ELSE v_reasons
        END;
END;
$$ LANGUAGE plpgsql;

-- Auto-moderation trigger
CREATE OR REPLACE FUNCTION chat.auto_moderate_message()
RETURNS TRIGGER AS $$
DECLARE
    v_moderation RECORD;
BEGIN
    -- Check content
    SELECT * INTO v_moderation
    FROM chat.moderate_content(NEW.content);
    
    IF NOT v_moderation.is_safe THEN
        -- Flag message
        NEW.metadata := NEW.metadata || 
            jsonb_build_object(
                'moderated', true,
                'risk_level', v_moderation.risk_level,
                'reasons', v_moderation.reasons
            );
        
        -- Notify moderators
        INSERT INTO chat.moderation_queue (
            message_id,
            risk_level,
            reasons,
            created_at
        ) VALUES (
            NEW.id,
            v_moderation.risk_level,
            v_moderation.reasons,
            NOW()
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

### Notification System

```sql
-- Smart notification preferences
CREATE TABLE chat.notification_preferences (
    user_id INTEGER REFERENCES chat.users(id),
    channel_id INTEGER REFERENCES chat.channels(id),
    notification_level VARCHAR(20) DEFAULT 'all', -- all, mentions, important, none
    keywords TEXT[],
    quiet_hours JSONB DEFAULT '{"start": "22:00", "end": "08:00"}',
    PRIMARY KEY (user_id, channel_id)
);

-- Determine if user should be notified
CREATE OR REPLACE FUNCTION chat.should_notify_user(
    p_user_id INTEGER,
    p_message RECORD
) RETURNS BOOLEAN AS $$
DECLARE
    v_prefs RECORD;
    v_importance_score FLOAT;
    v_current_hour INTEGER;
BEGIN
    -- Get user preferences
    SELECT * INTO v_prefs
    FROM chat.notification_preferences
    WHERE user_id = p_user_id
        AND channel_id = p_message.channel_id;
    
    -- Default to notify if no preferences
    IF NOT FOUND THEN
        RETURN TRUE;
    END IF;
    
    -- Check notification level
    IF v_prefs.notification_level = 'none' THEN
        RETURN FALSE;
    END IF;
    
    -- Check quiet hours
    v_current_hour := EXTRACT(HOUR FROM NOW());
    IF v_current_hour >= (v_prefs.quiet_hours->>'start')::INTEGER 
       OR v_current_hour < (v_prefs.quiet_hours->>'end')::INTEGER THEN
        RETURN FALSE;
    END IF;
    
    -- Check if mentioned
    IF p_message.content ~* ('@' || (
        SELECT username FROM chat.users WHERE id = p_user_id
    )) THEN
        RETURN TRUE;
    END IF;
    
    -- Check keywords
    IF v_prefs.keywords IS NOT NULL AND array_length(v_prefs.keywords, 1) > 0 THEN
        IF EXISTS (
            SELECT 1 
            FROM unnest(v_prefs.keywords) k 
            WHERE p_message.content ~* k
        ) THEN
            RETURN TRUE;
        END IF;
    END IF;
    
    -- Calculate importance score
    v_importance_score := chat.calculate_message_importance(p_message);
    
    RETURN CASE v_prefs.notification_level
        WHEN 'all' THEN TRUE
        WHEN 'important' THEN v_importance_score > 0.7
        ELSE FALSE
    END;
END;
$$ LANGUAGE plpgsql;

-- Calculate message importance
CREATE OR REPLACE FUNCTION chat.calculate_message_importance(
    p_message RECORD
) RETURNS FLOAT AS $$
DECLARE
    v_prompt TEXT;
    v_score_text TEXT;
    v_score FLOAT;
BEGIN
    v_prompt := format(
        'Rate the importance of this message from 0 to 1 based on urgency and relevance: "%s". Return only the number:',
        substring(p_message.content, 1, 200)
    );
    
    v_score_text := steadytext_generate(v_prompt, 10);
    
    IF v_score_text ~ '^[0-9]*\.?[0-9]+$' THEN
        v_score := v_score_text::FLOAT;
        
        -- Boost score for certain patterns
        IF p_message.content ~* '\y(urgent|important|asap|emergency)\y' THEN
            v_score := LEAST(v_score + 0.3, 1.0);
        END IF;
        
        RETURN v_score;
    ELSE
        RETURN 0.5; -- Default medium importance
    END IF;
END;
$$ LANGUAGE plpgsql;
```

### Performance Optimization

```sql
-- Message caching for hot channels
CREATE MATERIALIZED VIEW chat.hot_channel_cache AS
WITH hot_channels AS (
    SELECT channel_id
    FROM chat.messages
    WHERE created_at > NOW() - INTERVAL '1 hour'
    GROUP BY channel_id
    HAVING COUNT(*) > 100
)
SELECT 
    m.id,
    m.channel_id,
    m.user_id,
    m.content,
    m.created_at,
    u.username,
    u.display_name
FROM chat.messages m
JOIN chat.users u ON m.user_id = u.id
WHERE m.channel_id IN (SELECT channel_id FROM hot_channels)
    AND m.created_at > NOW() - INTERVAL '1 hour'
ORDER BY m.channel_id, m.created_at DESC;

CREATE INDEX idx_hot_cache_channel ON chat.hot_channel_cache(channel_id, created_at DESC);

-- Refresh cache periodically
CREATE OR REPLACE FUNCTION chat.refresh_hot_cache()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY chat.hot_channel_cache;
END;
$$ LANGUAGE plpgsql;

-- Connection pooling for real-time queries
CREATE OR REPLACE FUNCTION chat.get_recent_messages_optimized(
    p_channel_id INTEGER,
    p_limit INTEGER DEFAULT 50
) RETURNS TABLE(
    message_id INTEGER,
    user_id INTEGER,
    username VARCHAR(50),
    content TEXT,
    created_at TIMESTAMP
) AS $$
BEGIN
    -- Try hot cache first
    IF EXISTS (
        SELECT 1 FROM chat.hot_channel_cache 
        WHERE channel_id = p_channel_id 
        LIMIT 1
    ) THEN
        RETURN QUERY
        SELECT 
            id,
            user_id,
            username,
            content,
            created_at
        FROM chat.hot_channel_cache
        WHERE channel_id = p_channel_id
        ORDER BY created_at DESC
        LIMIT p_limit;
    ELSE
        -- Fall back to main tables
        RETURN QUERY
        SELECT 
            m.id,
            m.user_id,
            u.username,
            m.content,
            m.created_at
        FROM chat.messages m
        JOIN chat.users u ON m.user_id = u.id
        WHERE m.channel_id = p_channel_id
        ORDER BY m.created_at DESC
        LIMIT p_limit;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

## WebSocket Integration Example

```sql
-- WebSocket event queue
CREATE TABLE chat.websocket_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    channel_id INTEGER,
    user_id INTEGER,
    payload JSONB NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Queue WebSocket events
CREATE OR REPLACE FUNCTION chat.queue_websocket_event(
    p_event_type VARCHAR(50),
    p_channel_id INTEGER,
    p_payload JSONB
) RETURNS VOID AS $$
BEGIN
    INSERT INTO chat.websocket_events (
        event_type,
        channel_id,
        payload
    ) VALUES (
        p_event_type,
        p_channel_id,
        p_payload
    );
    
    -- Notify WebSocket server
    PERFORM pg_notify(
        'websocket_event',
        jsonb_build_object(
            'event_type', p_event_type,
            'channel_id', p_channel_id
        )::text
    );
END;
$$ LANGUAGE plpgsql;

-- Process typing indicators
CREATE OR REPLACE FUNCTION chat.handle_typing_indicator(
    p_channel_id INTEGER,
    p_user_id INTEGER,
    p_is_typing BOOLEAN
) RETURNS VOID AS $$
BEGIN
    -- Update user status
    UPDATE chat.users
    SET metadata = metadata || 
        jsonb_build_object(
            'typing_in_channel', 
            CASE WHEN p_is_typing THEN p_channel_id ELSE NULL END
        )
    WHERE id = p_user_id;
    
    -- Queue event for other users
    PERFORM chat.queue_websocket_event(
        'typing_indicator',
        p_channel_id,
        jsonb_build_object(
            'user_id', p_user_id,
            'is_typing', p_is_typing
        )
    );
END;
$$ LANGUAGE plpgsql;
```

## Related Documentation

- [PostgreSQL Extension Overview](../postgresql-extension.md)
- [Blog & CMS Examples](postgresql-blog-cms.md)
- [Search Examples](postgresql-search.md)
- [Analytics Examples](postgresql-analytics.md)
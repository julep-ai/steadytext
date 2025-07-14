# PostgreSQL Integration Examples

This section provides comprehensive examples for integrating SteadyText with PostgreSQL using the `pg_steadytext` extension. 

## Overview

The SteadyText PostgreSQL extension enables you to use AI-powered text generation, embeddings, and reranking directly within your database. This allows you to build intelligent applications without external API calls, maintaining data locality and improving performance.

## Example Categories

We've organized our PostgreSQL examples into specific use cases to help you find relevant patterns for your application:

### 📝 [Blog & Content Management](postgresql-blog-cms.md)
Build intelligent content management systems with features like:
- Automatic content generation and summarization
- SEO optimization
- Comment moderation with sentiment analysis
- Content recommendations
- Version control with AI-generated change summaries

### 🛒 [E-commerce Applications](postgresql-ecommerce.md)
Create AI-enhanced e-commerce platforms featuring:
- Product description generation
- Personalized recommendations
- Review analysis and summarization
- Dynamic pricing suggestions
- Customer service automation

### 🔍 [Semantic Search Systems](postgresql-search.md)
Implement powerful search functionality including:
- Hybrid search (vector + full-text)
- Document reranking
- Query expansion
- Search personalization
- Faceted search with AI insights

### 💬 [Real-time Applications](postgresql-realtime.md)
Build responsive real-time systems with:
- AI-powered chat assistance
- Message sentiment analysis
- Smart notifications
- Conversation summarization
- Real-time analytics

### 📊 [Analytics & Monitoring](postgresql-analytics.md)
Create intelligent monitoring systems featuring:
- Error analysis and categorization
- Anomaly detection
- Performance predictions
- User behavior analytics
- Executive summaries

## Getting Started

### Prerequisites

1. PostgreSQL 14+ with the following extensions:
   - `plpython3u` - Python language support
   - `pgvector` - Vector similarity search
   - `pg_steadytext` - SteadyText integration

2. SteadyText Python library installed in your PostgreSQL Python environment

### Basic Setup

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;
CREATE EXTENSION IF NOT EXISTS pgvector CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;

-- Verify installation
SELECT steadytext_version();

-- Start the daemon for better performance
SELECT steadytext_daemon_start();
```

### Quick Example

```sql
-- Generate text
SELECT steadytext_generate('Write a product description for organic coffee');

-- Create embeddings
SELECT steadytext_embed('premium organic coffee beans');

-- Rerank search results
SELECT * FROM steadytext_rerank(
    'best coffee for espresso',
    ARRAY['Colombian beans', 'Italian roast', 'Organic blend']
);
```

## Best Practices

### 1. Use Caching
The extension includes built-in caching. Repeated operations with the same inputs return cached results for consistency and performance.

### 2. Batch Operations
When processing multiple items, use batch functions for better performance:
```sql
SELECT * FROM steadytext_generate_batch(
    ARRAY['prompt1', 'prompt2', 'prompt3']
);
```

### 3. Async for Long Operations
Use async functions for operations that might take time:
```sql
SELECT steadytext_generate_async('Complex analysis task...', 1000);
```

### 4. Error Handling
Functions return NULL on error rather than throwing exceptions:
```sql
SELECT COALESCE(
    steadytext_generate('prompt'),
    'Fallback text'
);
```

## Performance Tips

1. **Start the Daemon**: Always run with the daemon for 160x faster first requests
2. **Index Embeddings**: Use ivfflat indexes for vector similarity search
3. **Preload Models**: Use `steadytext_download_models()` to ensure models are ready
4. **Monitor Cache**: Check cache statistics with `steadytext_cache_stats()`

## Architecture Patterns

### Pattern 1: Triggers for Automatic Processing
```sql
CREATE TRIGGER auto_generate_summary
    BEFORE INSERT ON articles
    FOR EACH ROW
    EXECUTE FUNCTION generate_article_summary();
```

### Pattern 2: Materialized Views for Performance
```sql
CREATE MATERIALIZED VIEW product_embeddings AS
SELECT id, steadytext_embed(name || ' ' || description) as embedding
FROM products;
```

### Pattern 3: Async Job Queues
```sql
-- Queue long-running tasks
INSERT INTO ai_job_queue (task_type, payload)
VALUES ('generate_report', '{"report_id": 123}');

-- Process with background worker
SELECT process_ai_queue();
```

## Troubleshooting

For detailed troubleshooting information, see the [PostgreSQL Extension Troubleshooting Guide](../postgresql-extension-troubleshooting.md).

Common issues:
- **NULL returns**: Check daemon status with `steadytext_daemon_status()`
- **Slow performance**: Ensure daemon is running and models are preloaded
- **Out of memory**: Adjust PostgreSQL memory settings and model cache size

## Related Documentation

- [PostgreSQL Extension Overview](../postgresql-extension.md)
- [Function Reference](../postgresql-extension-reference.md)
- [Advanced Features](../postgresql-extension-advanced.md)
- [AI Integration](../postgresql-extension-ai.md)
- [Async Operations](../postgresql-extension-async.md)
- [Troubleshooting](../postgresql-extension-troubleshooting.md)
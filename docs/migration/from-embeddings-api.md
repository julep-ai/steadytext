# Migrating from Embeddings APIs to SteadyText

Replace external embedding services with fast, deterministic, local embeddings that work directly in your database.

## Why Migrate from Embedding APIs?

| **External Embedding APIs** | **SteadyText Embeddings** |
|----------------------------|---------------------------|
| Pay per embedding | Free after installation |
| Network latency (50-200ms) | Local execution (<1ms) |
| Rate limits apply | Unlimited embeddings |
| Internet required | Works offline |
| Privacy concerns | Data never leaves your server |
| Non-deterministic* | 100% deterministic |

*Some providers vary embeddings slightly between calls

## Quick Start

### Python Migration

**Before (OpenAI/Cohere/etc):**
```python
# OpenAI
import openai
openai.api_key = "sk-..."
response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input="Hello world"
)
embedding = response.data[0].embedding  # 1536 dims

# Cohere
import cohere
co = cohere.Client("api-key")
response = co.embed(texts=["Hello world"])
embedding = response.embeddings[0]  # 768/1024 dims

# Hugging Face
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# Complex setup...
```

**After (SteadyText):**
```python
import steadytext

# That's it! No API keys, no setup
embedding = steadytext.embed("Hello world")  # 1024 dims
# Always returns the same vector for the same input
```

### PostgreSQL Migration

**Before (Storing external embeddings):**
```sql
-- Complex setup with external calls
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding FLOAT[]  -- Or vector type
);

-- Need application code to generate embeddings
-- INSERT happens from Python/Node/etc
```

**After (Native PostgreSQL):**
```sql
CREATE EXTENSION pg_steadytext;
CREATE EXTENSION pgvector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1024)
);

-- Generate embeddings directly in SQL!
INSERT INTO documents (content, embedding)
VALUES 
    ('Hello world', steadytext_embed('Hello world')::vector),
    ('PostgreSQL rocks', steadytext_embed('PostgreSQL rocks')::vector);

-- Or update existing data
UPDATE documents 
SET embedding = steadytext_embed(content)::vector
WHERE embedding IS NULL;
```

## Common Embedding API Migrations

### 1. OpenAI text-embedding-ada-002

**Dimension Mapping:**
- OpenAI: 1536 dimensions
- SteadyText: 1024 dimensions

**Migration Script:**
```sql
-- Add new column for SteadyText embeddings
ALTER TABLE documents ADD COLUMN embedding_new vector(1024);

-- Generate new embeddings
UPDATE documents 
SET embedding_new = steadytext_embed(content)::vector;

-- Once verified, swap columns
ALTER TABLE documents DROP COLUMN embedding;
ALTER TABLE documents RENAME COLUMN embedding_new TO embedding;

-- Update indexes
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

### 2. Cohere Embeddings

**Before:**
```python
import cohere
co = cohere.Client('api-key')

def get_embeddings_batch(texts, batch_size=96):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = co.embed(
            texts=batch,
            model='embed-english-v3.0',
            input_type='search_document'
        )
        embeddings.extend(response.embeddings)
    return embeddings
```

**After:**
```python
import steadytext

def get_embeddings_batch(texts, batch_size=1000):
    # No rate limits! Process as fast as your CPU allows
    return [steadytext.embed(text) for text in texts]
```

### 3. Sentence Transformers

**Before:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)  # 384 dims
```

**After:**
```python
import steadytext

embeddings = [steadytext.embed(s) for s in sentences]  # 1024 dims
# Higher dimensional embeddings often perform better
```

### 4. Voyage AI

**Before:**
```python
import voyageai

vo = voyageai.Client(api_key="...")
result = vo.embed(texts, model="voyage-02")
embeddings = result.embeddings
```

**After:**
```python
embeddings = [steadytext.embed(text) for text in texts]
# No API key needed!
```

## Semantic Search Migration

### Vector Database Migration

**Before (Pinecone/Weaviate/Qdrant):**
```python
import pinecone

pinecone.init(api_key="...", environment="...")
index = pinecone.Index("my-index")

# Upload embeddings
for i, text in enumerate(texts):
    embedding = get_external_embedding(text)
    index.upsert([(str(i), embedding, {"text": text})])

# Search
query_embedding = get_external_embedding(query)
results = index.query(query_embedding, top_k=10)
```

**After (PostgreSQL + pgvector):**
```sql
-- Everything in your database!
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1024),
    metadata JSONB
);

-- Index for fast search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);

-- Insert with embeddings
INSERT INTO documents (content, embedding)
SELECT content, steadytext_embed(content)::vector
FROM raw_documents;

-- Search is just SQL
WITH query AS (
    SELECT steadytext_embed('search query')::vector AS q_embedding
)
SELECT d.*, 1 - (d.embedding <=> q.q_embedding) AS similarity
FROM documents d, query q
WHERE d.embedding <=> q.q_embedding < 0.3  -- Distance threshold
ORDER BY d.embedding <=> q.q_embedding
LIMIT 10;
```

## Bulk Migration Strategies

### Parallel Processing

```sql
-- Process embeddings in parallel
CREATE OR REPLACE FUNCTION migrate_embeddings_parallel(
    batch_size INTEGER DEFAULT 1000
)
RETURNS VOID AS $$
DECLARE
    v_count INTEGER := 0;
BEGIN
    -- Enable parallel processing
    SET max_parallel_workers_per_gather = 4;
    
    -- Update in batches
    LOOP
        WITH batch AS (
            SELECT id, content
            FROM documents
            WHERE embedding IS NULL
            LIMIT batch_size
            FOR UPDATE SKIP LOCKED
        )
        UPDATE documents d
        SET embedding = steadytext_embed(b.content)::vector
        FROM batch b
        WHERE d.id = b.id;
        
        GET DIAGNOSTICS v_count = ROW_COUNT;
        EXIT WHEN v_count = 0;
        
        -- Progress notification
        RAISE NOTICE 'Processed % records', v_count;
        COMMIT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

### Progress Tracking

```sql
-- Track migration progress
CREATE TABLE embedding_migration_progress (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100),
    total_records INTEGER,
    processed_records INTEGER,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'running'
);

-- Monitor progress
CREATE OR REPLACE VIEW migration_status AS
SELECT 
    table_name,
    processed_records || '/' || total_records AS progress,
    ROUND(processed_records::NUMERIC / total_records * 100, 2) || '%' AS percentage,
    CASE 
        WHEN completed_at IS NOT NULL THEN 
            'Completed in ' || (completed_at - started_at)::TEXT
        ELSE 
            'Running for ' || (NOW() - started_at)::TEXT
    END AS duration
FROM embedding_migration_progress;
```

## Quality Comparison

### A/B Testing Embeddings

```python
def compare_embedding_quality(texts, queries):
    results = {
        'external': {'precision': [], 'recall': []},
        'steadytext': {'precision': [], 'recall': []}
    }
    
    for query in queries:
        # External API results
        external_embedding = get_external_embedding(query)
        external_results = search_with_embedding(external_embedding)
        
        # SteadyText results
        steady_embedding = steadytext.embed(query)
        steady_results = search_with_embedding(steady_embedding)
        
        # Compare precision/recall
        # ... calculation logic ...
    
    return results
```

### Embedding Space Analysis

```sql
-- Analyze embedding space distribution
CREATE OR REPLACE FUNCTION analyze_embedding_distribution()
RETURNS TABLE (
    metric VARCHAR,
    value NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 'avg_magnitude', AVG(sqrt(sum(v * v)))::NUMERIC
    FROM (
        SELECT unnest(embedding::float[]) AS v, id
        FROM documents
    ) t
    GROUP BY id
    
    UNION ALL
    
    SELECT 'avg_similarity', AVG(1 - (a.embedding <=> b.embedding))::NUMERIC
    FROM documents a, documents b
    WHERE a.id < b.id
    LIMIT 10000;  -- Sample for performance
END;
$$ LANGUAGE plpgsql;
```

## Cost Calculator

```python
def calculate_savings(monthly_embeddings, provider="openai"):
    costs = {
        "openai": 0.0001,      # per 1K tokens
        "cohere": 0.0002,      # per 1K embeddings
        "voyage": 0.00012,     # per 1K embeddings
        "anthropic": 0.0001    # per 1K tokens
    }
    
    monthly_cost = (monthly_embeddings / 1000) * costs.get(provider, 0.0001)
    yearly_cost = monthly_cost * 12
    
    print(f"Current {provider} costs:")
    print(f"  Monthly: ${monthly_cost:,.2f}")
    print(f"  Yearly: ${yearly_cost:,.2f}")
    print(f"\nSteadyText costs:")
    print(f"  Monthly: $0")
    print(f"  Yearly: $0")
    print(f"\nYearly savings: ${yearly_cost:,.2f}")
    
# Example: 10M embeddings/month
calculate_savings(10_000_000, "openai")
```

## Performance Benchmarks

### Speed Comparison

```python
import time
import statistics

def benchmark_embedding_speed(texts, iterations=5):
    # SteadyText
    steady_times = []
    for _ in range(iterations):
        start = time.time()
        for text in texts:
            steadytext.embed(text)
        steady_times.append(time.time() - start)
    
    # External API (simulated)
    api_times = []
    for _ in range(iterations):
        start = time.time()
        for text in texts:
            time.sleep(0.05)  # Simulate 50ms API latency
        api_times.append(time.time() - start)
    
    print(f"SteadyText: {statistics.mean(steady_times):.3f}s "
          f"({len(texts)/statistics.mean(steady_times):.0f} embeddings/sec)")
    print(f"External API: {statistics.mean(api_times):.3f}s "
          f"({len(texts)/statistics.mean(api_times):.0f} embeddings/sec)")
```

## Common Issues & Solutions

### Issue 1: Dimension Mismatch
```sql
-- Solution: Re-create vector columns with correct dimensions
ALTER TABLE documents 
ALTER COLUMN embedding TYPE vector(1024) 
USING embedding::vector(1024);
```

### Issue 2: Different Similarity Scores
```python
# Normalize similarity scores for comparison
def normalize_similarity(score, method="cosine"):
    if method == "cosine":
        return (score + 1) / 2  # Map [-1, 1] to [0, 1]
    elif method == "euclidean":
        return 1 / (1 + score)  # Map [0, ∞) to (0, 1]
```

### Issue 3: Batch Size Optimization
```sql
-- Find optimal batch size for your hardware
DO $$
DECLARE
    batch_sizes INTEGER[] := ARRAY[100, 500, 1000, 5000];
    size INTEGER;
    start_time TIMESTAMP;
    duration INTERVAL;
BEGIN
    FOREACH size IN ARRAY batch_sizes
    LOOP
        start_time := clock_timestamp();
        
        PERFORM steadytext_embed(content)
        FROM documents
        LIMIT size;
        
        duration := clock_timestamp() - start_time;
        RAISE NOTICE 'Batch size %: % seconds', size, duration;
    END LOOP;
END $$;
```

## Migration Checklist

- [ ] Benchmark current embedding quality
- [ ] Install SteadyText and pgvector
- [ ] Create new vector columns with correct dimensions
- [ ] Migrate embeddings (use parallel processing)
- [ ] Update application code
- [ ] Compare search quality (A/B test)
- [ ] Update vector indexes
- [ ] Remove external API dependencies
- [ ] Calculate and celebrate cost savings! 🎉

## Next Steps

- [PostgreSQL Extension Setup →](../postgresql-extension-hero.md)
- [Customer Intelligence Examples →](../examples/customer-intelligence.md)
- [Performance Tuning →](../examples/performance-tuning.md)

---

!!! tip "Pro Tip"
    Start by migrating a small subset of your data to compare quality. SteadyText's embeddings are optimized for semantic similarity and often outperform general-purpose embedding APIs for domain-specific content.
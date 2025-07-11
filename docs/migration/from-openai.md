# Migrating from OpenAI API to SteadyText

A practical guide to replacing OpenAI API calls with SteadyText's deterministic, local AI capabilities.

## Why Migrate?

| **OpenAI API** | **SteadyText** |
|----------------|----------------|
| $0.01-0.12 per 1K tokens | $0 after installation |
| 100-500ms latency | <1ms local execution |
| Rate limits and quotas | Unlimited local usage |
| Non-deterministic outputs | 100% deterministic |
| Internet required | Works offline |
| API key management | No keys needed |

## Quick Comparison

### Text Generation

**Before (OpenAI):**
```python
import openai

openai.api_key = "sk-..."

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Summarize this text: " + text}],
    temperature=0,  # Still not deterministic!
    max_tokens=150
)
summary = response.choices[0].message.content
```

**After (SteadyText):**
```python
import steadytext

# No API key needed!
summary = steadytext.generate(
    f"Summarize this text: {text}",
    max_tokens=150
)
# Same input ALWAYS produces same output
```

### Embeddings

**Before (OpenAI):**
```python
response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input="Hello world"
)
embedding = response.data[0].embedding  # 1536 dimensions
```

**After (SteadyText):**
```python
embedding = steadytext.embed("Hello world")  # 1024 dimensions
# Deterministic - same text always produces same vector
```

## PostgreSQL Migration

### Database Functions

**Before (OpenAI via HTTP):**
```sql
-- Complex function making HTTP requests
CREATE OR REPLACE FUNCTION summarize_with_openai(text_input TEXT)
RETURNS TEXT AS $$
DECLARE
    api_response JSONB;
BEGIN
    -- Using pg_http or similar
    SELECT content::JSONB INTO api_response
    FROM http_post(
        'https://api.openai.com/v1/chat/completions',
        jsonb_build_object(
            'model', 'gpt-3.5-turbo',
            'messages', jsonb_build_array(
                jsonb_build_object('role', 'user', 'content', text_input)
            )
        )::TEXT,
        'application/json',
        ARRAY[['Authorization', 'Bearer sk-...']]
    );
    
    RETURN api_response->'choices'->0->'message'->>'content';
END;
$$ LANGUAGE plpgsql;
```

**After (SteadyText):**
```sql
-- Simple, fast, deterministic
CREATE EXTENSION pg_steadytext;

-- That's it! Now just use:
SELECT steadytext_generate('Summarize: ' || text_column) 
FROM your_table;
```

### Batch Processing

**Before (OpenAI with rate limits):**
```python
import time
import openai
from openai.error import RateLimitError

summaries = []
for i, text in enumerate(texts):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Summarize: {text}"}]
        )
        summaries.append(response.choices[0].message.content)
        
        # Respect rate limits
        if i % 10 == 0:
            time.sleep(1)
            
    except RateLimitError:
        time.sleep(60)  # Wait a minute
        # Retry logic here...
```

**After (SteadyText unlimited):**
```sql
-- Process millions of rows without rate limits
UPDATE articles 
SET summary = steadytext_generate('Summarize: ' || content)
WHERE summary IS NULL;

-- Or in Python with no rate limits
summaries = [steadytext.generate(f"Summarize: {text}") for text in texts]
```

## Common Use Case Migrations

### 1. Content Moderation

**Before:**
```python
def moderate_content_openai(text):
    response = openai.Moderation.create(input=text)
    return response.results[0].flagged
```

**After:**
```python
def moderate_content_steadytext(text):
    result = steadytext.generate_choice(
        f"Is this content inappropriate: {text}",
        choices=["safe", "inappropriate"]
    )
    return result == "inappropriate"
```

### 2. Structured Data Extraction

**Before:**
```python
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{
        "role": "system", 
        "content": "Extract JSON data"
    }, {
        "role": "user",
        "content": text
    }],
    response_format={"type": "json_object"}  # Still can fail!
)
```

**After:**
```python
# Guaranteed valid JSON with schema
result = steadytext.generate_json(
    text,
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phone": {"type": "string"}
        }
    }
)
```

### 3. Semantic Search

**Before:**
```python
# Store OpenAI embeddings
def create_embedding_openai(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Search with cosine similarity
query_embedding = create_embedding_openai(query)
# Complex vector similarity search...
```

**After:**
```sql
-- Native PostgreSQL with pgvector
ALTER TABLE documents ADD COLUMN embedding vector(1024);

UPDATE documents 
SET embedding = steadytext_embed(content)::vector;

-- Search is just SQL
SELECT * FROM documents
WHERE embedding <=> steadytext_embed('search query')::vector < 0.3
ORDER BY embedding <=> steadytext_embed('search query')::vector
LIMIT 10;
```

## Cost Analysis

### OpenAI Costs (Monthly)
```
10M tokens/day × 30 days = 300M tokens/month

GPT-3.5 Turbo: 300M × $0.001/1K = $300/month
GPT-4: 300M × $0.03/1K = $9,000/month
Embeddings: 100M × $0.0001/1K = $10/month

Total: $310-9,010/month + rate limit delays
```

### SteadyText Costs
```
One-time: $0 (open source)
Monthly: $0 (runs on your infrastructure)
Rate limits: None
Latency: <1ms (vs 100-500ms)
```

## Testing Strategy

### Making Tests Deterministic

**Before (Flaky):**
```python
def test_summarization():
    # This test randomly fails!
    summary = call_openai_api("Summarize: " + text)
    assert "important point" in summary  # Sometimes true, sometimes false
```

**After (Reliable):**
```python
def test_summarization():
    # Always passes with same input
    summary = steadytext.generate("Summarize: " + text)
    assert summary == "Expected exact output"  # Deterministic!
```

## Migration Checklist

- [ ] Install SteadyText (`pip install steadytext`)
- [ ] For PostgreSQL: Install pg_steadytext extension
- [ ] Replace OpenAI initialization with SteadyText import
- [ ] Update function calls (see mapping below)
- [ ] Remove API key management code
- [ ] Remove rate limit handling
- [ ] Update error handling (no more network errors!)
- [ ] Update tests to expect deterministic outputs
- [ ] Calculate cost savings 🎉

## Function Mapping Reference

| **OpenAI Function** | **SteadyText Equivalent** |
|---------------------|---------------------------|
| `ChatCompletion.create()` | `steadytext.generate()` |
| `Embedding.create()` | `steadytext.embed()` |
| `ChatCompletion.create(stream=True)` | `steadytext.generate_iter()` |
| `response_format={"type": "json_object"}` | `steadytext.generate_json()` |
| `functions=[...]` | `steadytext.generate_json(schema=...)` |
| `Moderation.create()` | `steadytext.generate_choice()` |

## Advanced Patterns

### Caching Layer

**Before (Complex Redis setup):**
```python
def get_summary_with_cache(text):
    cache_key = hashlib.md5(text.encode()).hexdigest()
    
    # Check Redis
    cached = redis_client.get(cache_key)
    if cached:
        return cached
    
    # Call OpenAI
    summary = call_openai_api(text)
    
    # Cache with TTL
    redis_client.setex(cache_key, 3600, summary)
    return summary
```

**After (Built-in caching):**
```python
# SteadyText automatically caches deterministic outputs
summary = steadytext.generate(f"Summarize: {text}")
# Subsequent calls with same input are instant!
```

### Async Operations

**Before:**
```python
async def process_batch_openai(texts):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for text in texts:
            task = call_openai_async(session, text)
            tasks.append(task)
        return await asyncio.gather(*tasks)
```

**After (PostgreSQL):**
```sql
-- Process asynchronously in database
SELECT steadytext_generate_async(
    'Summarize: ' || content
) FROM articles;

-- Check results
SELECT * FROM steadytext_check_async_batch(array_of_ids);
```

## Gradual Migration Strategy

### Phase 1: Development Environment
1. Install SteadyText alongside OpenAI
2. A/B test outputs for quality
3. Measure performance improvements

### Phase 2: Non-Critical Features
1. Migrate internal tools first
2. Move test environments
3. Validate deterministic behavior

### Phase 3: Production Migration
1. Start with read-heavy workloads
2. Migrate batch processing
3. Finally migrate real-time features

### Rollback Plan
```python
# Feature flag approach
USE_STEADYTEXT = os.getenv("USE_STEADYTEXT", "false") == "true"

def generate_text(prompt):
    if USE_STEADYTEXT:
        return steadytext.generate(prompt)
    else:
        return call_openai_api(prompt)
```

## Common Gotchas

1. **Output Length**: SteadyText defaults to 512 tokens (configurable)
2. **Model Size**: 2GB download on first use
3. **Embedding Dimensions**: 1024 vs OpenAI's 1536
4. **JSON Mode**: Use `generate_json()` with schema for guaranteed structure

## Support & Community

- GitHub Issues: [github.com/julep-ai/steadytext/issues](https://github.com/julep-ai/steadytext/issues)
- Discord: [discord.gg/steadytext](https://discord.gg/steadytext)
- More Questions: See our [FAQ](../faq.md)

---

!!! success "Ready to Save Money?"
    Most teams see 100% cost reduction and 100x performance improvement after migration. Start with a small proof-of-concept and scale from there!
# Why SteadyText

Understanding the technical rationale behind deterministic AI.

## The Problem: Non-Deterministic AI

Traditional AI models produce different outputs for the same input, causing:

- **Flaky tests**: Tests that pass locally but fail in CI
- **Debugging difficulties**: Cannot reproduce issues reliably
- **Caching challenges**: Results cannot be cached effectively
- **API dependencies**: External services introduce latency and failure points

```python
# Traditional approach - unpredictable
def test_ai_feature():
    result = ai_generate("Summarize this")
    assert "summary" in result  # May randomly fail
```

## The Solution: Deterministic Generation

SteadyText ensures identical inputs always produce identical outputs by:

1. **Fixed random seeds**: All randomness is seeded with deterministic values
2. **Greedy decoding**: Always selecting the highest probability token
3. **Quantized models**: Consistent numerical precision across platforms
4. **Aggressive caching**: Deterministic outputs enable perfect caching

```python
# SteadyText approach - predictable
def test_ai_feature():
    result = steadytext.generate("Summarize this")
    assert result == steadytext.generate("Summarize this")  # Always true
```

## Technical Architecture

### Local-First Design

- **No network calls**: Models run entirely on your infrastructure
- **No API keys**: Self-contained system with no external dependencies
- **Predictable latency**: Consistent sub-millisecond response times
- **Data locality**: AI processing happens where your data lives

### PostgreSQL Integration

The PostgreSQL extension enables AI operations directly in SQL:

```sql
-- AI as a native database function
SELECT 
    id,
    steadytext_generate('Summarize: ' || content) AS summary
FROM documents
WHERE created_at > CURRENT_DATE - 7;
```

Benefits:
- **Transactional consistency**: AI operations participate in ACID transactions
- **Backup integration**: AI results included in standard database backups
- **Security model**: Leverages existing PostgreSQL authentication and permissions
- **Performance**: Eliminates round-trips between application and database

### Caching Strategy

Deterministic outputs enable sophisticated caching:

```python
# Cache key includes all parameters affecting output
cache_key = hash(prompt + str(seed) + model_params)

# Perfect cache hits for repeated queries
if cache_key in cache:
    return cache[cache_key]  # <1ms response
```

Cache features:
- **Frecency-based eviction**: Balances recency and frequency
- **Distributed backends**: Support for SQLite, D1, and memory caches
- **Size limits**: Configurable capacity and memory constraints

## Use Cases

SteadyText excels in scenarios requiring predictable AI:

1. **Automated testing**: Reliable assertions on AI-generated content
2. **Data pipelines**: Reproducible ETL operations with AI components
3. **Content generation**: Consistent outputs for documentation and reports
4. **Semantic search**: Stable embeddings for similarity matching
5. **Log analysis**: Deterministic summarization of system events

## Performance Characteristics

- **Inference latency**: <100ms for most generation tasks
- **Embedding speed**: ~1ms per text with caching
- **Memory usage**: 2-4GB for model storage
- **Cache hit rate**: >90% in typical workloads

## Design Principles

1. **Determinism by default**: Same input → same output, always
2. **Zero configuration**: Works out of the box without setup
3. **Local execution**: No external dependencies or network calls
4. **SQL-native**: AI as a first-class database primitive
5. **Production-ready**: Designed for reliability over novelty

## Next Steps

- [Quick Start Guide](quick-start.md) - Get running in minutes
- [PostgreSQL Extension](postgresql-extension.md) - Database integration
- [API Reference](api/index.md) - Complete function documentation
- [Examples](examples/index.md) - Real-world usage patterns
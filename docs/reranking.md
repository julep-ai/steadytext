# Document Reranking

SteadyText v1.3.0+ introduces powerful document reranking capabilities, allowing you to reorder search results based on their relevance to a query using the state-of-the-art Qwen3-Reranker-4B model.

## Overview

Document reranking is a crucial technique in information retrieval that improves search quality by re-scoring and reordering an initial set of retrieved documents. While traditional search methods (like BM25 or vector similarity) are fast and efficient for initial retrieval, a reranking model can provide more accurate relevance scores by deeply understanding the semantic relationship between queries and documents.

### Why Use Reranking?

1. **Improved Search Quality**: Rerankers understand context and semantics better than traditional retrieval methods
2. **Domain Adaptation**: Custom task descriptions allow the model to understand your specific use case
3. **Hybrid Search**: Combine fast initial retrieval with accurate reranking for optimal performance
4. **Deterministic Results**: With custom seeds, you get reproducible rankings for testing and debugging

## How It Works

SteadyText's reranking uses the Qwen3-Reranker-4B model, which:

1. Takes a query and a list of documents
2. Evaluates each query-document pair using a binary relevance judgment ("yes" or "no")
3. Converts the model's confidence into a relevance score
4. Returns documents sorted by relevance score

The model uses a special prompt format that includes:
- A system prompt explaining the task
- Your custom task description
- The query and document to evaluate
- A "thinking" section where the model reasons about relevance

## Basic Usage

### Python API

```python
import steadytext

# Basic reranking
documents = [
    "Python is a high-level programming language known for its simplicity",
    "Cats are independent pets that many people love",
    "Python snakes are non-venomous constrictors found in Africa and Asia"
]

# Rerank documents by relevance to the query
results = steadytext.rerank("Python programming language", documents)

# Results are sorted by relevance (highest score first)
for doc, score in results:
    print(f"Score: {score:.3f} | Document: {doc}")
```

### CLI Usage

```bash
# Rerank documents from files
st rerank "machine learning tutorial" doc1.txt doc2.txt doc3.txt

# Rerank with custom output format
st rerank "customer complaint" *.txt --format json

# Rerank from stdin
echo -e "Document 1\n---\nDocument 2\n---\nDocument 3" | \
  st rerank "Python programming" --delimiter "---"
```

## Advanced Features

### Custom Task Descriptions

Different domains require different relevance criteria. Use custom task descriptions to guide the model:

```python
# Legal document search
legal_results = steadytext.rerank(
    query="intellectual property infringement",
    documents=legal_documents,
    task="legal document retrieval for case law research"
)

# Customer support prioritization
support_results = steadytext.rerank(
    query="payment failed refund request",
    documents=support_tickets,
    task="prioritize support tickets by urgency and relevance"
)

# Academic paper retrieval
academic_results = steadytext.rerank(
    query="transformer architectures for NLP",
    documents=research_papers,
    task="find relevant academic papers for literature review"
)
```

### Reproducible Results with Seeds

Use custom seeds for deterministic, reproducible rankings:

```python
# Same seed = same rankings
results1 = steadytext.rerank("AI safety", documents, seed=42)
results2 = steadytext.rerank("AI safety", documents, seed=42)
assert results1 == results2  # True

# Different seed = potentially different rankings
results3 = steadytext.rerank("AI safety", documents, seed=123)
# results3 may have different ordering than results1
```

### Getting Documents Without Scores

If you only need the reordered documents without scores:

```python
# Returns just the documents in relevance order
sorted_docs = steadytext.rerank(
    "machine learning",
    documents,
    return_scores=False
)

# Useful for direct display or further processing
for doc in sorted_docs:
    print(doc)
```

## Integration Patterns

### Hybrid Search Pipeline

Combine fast initial retrieval with accurate reranking:

```python
import steadytext
from your_search_engine import search

def hybrid_search(query, index, rerank_top_k=20, return_top_k=5):
    # Step 1: Fast initial retrieval (e.g., BM25, vector search)
    initial_results = search(query, index, top_k=rerank_top_k)
    
    # Step 2: Extract documents for reranking
    documents = [result['text'] for result in initial_results]
    
    # Step 3: Rerank with SteadyText
    reranked = steadytext.rerank(query, documents)
    
    # Step 4: Return top results with metadata
    final_results = []
    for doc, score in reranked[:return_top_k]:
        # Find original metadata
        original = next(r for r in initial_results if r['text'] == doc)
        final_results.append({
            **original,
            'rerank_score': score
        })
    
    return final_results
```

### PostgreSQL Integration

Use reranking directly in your PostgreSQL queries:

```sql
-- Rerank search results from full-text search
WITH search_results AS (
    SELECT id, title, content, 
           ts_rank(search_vector, query) AS text_score
    FROM documents, 
         plainto_tsquery('english', 'machine learning') query
    WHERE search_vector @@ query
    LIMIT 50  -- Get more candidates for reranking
)
SELECT sr.*, r.score as rerank_score
FROM search_results sr,
     LATERAL steadytext_rerank(
         'machine learning',
         ARRAY_AGG(sr.content) OVER (),
         seed := 42
     ) r
WHERE sr.content = r.document
ORDER BY r.score DESC
LIMIT 10;  -- Return top reranked results

-- Rerank with custom task for support tickets
SELECT *
FROM steadytext_rerank(
    'payment processing error',
    ARRAY(
        SELECT concat(subject, ' ', description)
        FROM support_tickets
        WHERE status = 'open'
        AND created_at > NOW() - INTERVAL '24 hours'
    ),
    task := 'identify high-priority payment-related support tickets'
);
```

### Batch Processing

Efficiently rerank multiple queries:

```python
def batch_rerank(queries, document_sets, task=None):
    """Rerank multiple queries against their respective document sets."""
    results = []
    
    for query, documents in zip(queries, document_sets):
        reranked = steadytext.rerank(
            query,
            documents,
            task=task,
            return_scores=True
        )
        results.append({
            'query': query,
            'results': reranked
        })
    
    return results

# Example usage
queries = ["Python tutorial", "Machine learning", "Web development"]
document_sets = [docs_set1, docs_set2, docs_set3]

batch_results = batch_rerank(queries, document_sets)
```

## Performance Considerations

### Caching

Reranking results are automatically cached for identical query-document pairs:

```python
# First call: computes scores
results1 = steadytext.rerank("Python", ["doc1", "doc2"])

# Second call: returns cached results (fast)
results2 = steadytext.rerank("Python", ["doc1", "doc2"])

# Different query or documents: new computation
results3 = steadytext.rerank("Java", ["doc1", "doc2"])
```

### Model Loading

The reranker model is loaded on first use and cached:

```python
# Preload the reranker model
steadytext.preload_models(reranker=True)

# Now reranking calls will be faster
results = steadytext.rerank(query, documents)
```

### Context Length Limits

The reranker has context length limits. For very long documents:

```python
def chunk_and_rerank(query, long_documents, chunk_size=500):
    """Rerank long documents by chunking them first."""
    all_chunks = []
    chunk_to_doc = {}
    
    # Create chunks
    for doc_idx, doc in enumerate(long_documents):
        words = doc.split()
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            all_chunks.append(chunk)
            chunk_to_doc[chunk] = doc_idx
    
    # Rerank chunks
    reranked_chunks = steadytext.rerank(query, all_chunks)
    
    # Aggregate scores by document
    doc_scores = {}
    for chunk, score in reranked_chunks:
        doc_idx = chunk_to_doc[chunk]
        if doc_idx not in doc_scores:
            doc_scores[doc_idx] = []
        doc_scores[doc_idx].append(score)
    
    # Average scores per document
    final_scores = [
        (long_documents[idx], np.mean(scores))
        for idx, scores in doc_scores.items()
    ]
    
    return sorted(final_scores, key=lambda x: x[1], reverse=True)
```

## Fallback Behavior

When the reranker model is unavailable, SteadyText falls back to a simple word overlap scoring:

```python
# If model loading fails, you still get results
# based on word overlap between query and documents
results = steadytext.rerank("Python programming", documents)
# Returns deterministic scores based on word overlap
```

This ensures your application never fails, even if the model cannot be loaded.

## Best Practices

1. **Initial Retrieval Size**: Retrieve 3-5x more documents than you need, then rerank and take the top K
2. **Task Descriptions**: Write clear, specific task descriptions for your domain
3. **Document Length**: Keep documents reasonably sized (under 1000 words) for best performance
4. **Caching**: Take advantage of automatic caching for repeated queries
5. **Evaluation**: Use consistent seeds when evaluating reranking quality

## Examples

### E-commerce Product Search

```python
products = [
    "iPhone 15 Pro Max 256GB Natural Titanium - Latest Apple smartphone with A17 Pro chip",
    "Samsung Galaxy S24 Ultra 512GB - Android flagship with S Pen and AI features",
    "Google Pixel 8 Pro 128GB - Pure Android experience with advanced camera AI",
    "OnePlus 12 256GB - Fast charging flagship killer with Hasselblad cameras"
]

# Customer searching for an iPhone
results = steadytext.rerank(
    "iPhone Pro Max best price",
    products,
    task="e-commerce product search - prioritize exact product matches"
)

# The iPhone will rank highest despite other products mentioning "Pro"
```

### Research Paper Discovery

```python
papers = load_research_papers()  # Your paper abstracts

# Find relevant papers for a literature review
relevant_papers = steadytext.rerank(
    "transformer models for code generation",
    papers,
    task="academic literature review - find papers directly related to the research topic",
    return_scores=True
)

# Filter by score threshold
high_quality = [(p, s) for p, s in relevant_papers if s > 0.7]
print(f"Found {len(high_quality)} highly relevant papers")
```

### Content Recommendation

```python
articles = fetch_blog_posts()  # Your content

# Personalized recommendations based on user interest
recommendations = steadytext.rerank(
    "machine learning for beginners Python tutorials",
    articles,
    task="content recommendation - match user interest and skill level",
    seed=user_id  # Use user ID as seed for consistent personalization
)

# Show top 5 recommendations
for article, score in recommendations[:5]:
    display_article_preview(article, score)
```

## Troubleshooting

### Low Relevance Scores

If you're getting unexpectedly low scores:
1. Check your task description - make it more specific
2. Ensure documents contain enough context
3. Try different seed values
4. Verify documents are in the same language as the query

### Performance Issues

If reranking is slow:
1. Reduce the number of documents to rerank
2. Preload the model with `preload_models(reranker=True)`
3. Use shorter documents or implement chunking
4. Enable daemon mode for better performance

### Memory Usage

The reranker model uses ~4GB of memory. If you're running out of memory:
1. Use a smaller initial retrieval set
2. Process documents in batches
3. Ensure other models are unloaded when not needed

## See Also

- [API Reference](api.md#document-reranking-v130) - Detailed API documentation
- [PostgreSQL Extension](postgresql-extension.md#document-reranking-v130) - Database integration
- [Vector Indexing](vector-indexing.md) - Building the initial retrieval pipeline
- [Examples](examples/index.md) - More code examples
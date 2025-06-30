# Embeddings API

Functions for creating deterministic text embeddings.

## embed()

Create deterministic embeddings for text input.

```python
def embed(text_input: Union[str, List[str]], seed: int = DEFAULT_SEED) -> np.ndarray
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_input` | `Union[str, List[str]]` | *required* | Text string or list of strings to embed |
| `seed` | `int` | `42` | Random seed for deterministic embedding generation |

### Returns

**Returns**: `np.ndarray` - 1024-dimensional L2-normalized float32 array

### Examples

=== "Single Text"

    ```python
    import steadytext
    import numpy as np

    # Embed single text
    vector = steadytext.embed("Hello world")
    
    print(f"Shape: {vector.shape}")        # (1024,)
    print(f"Type: {vector.dtype}")         # float32
    print(f"Norm: {np.linalg.norm(vector):.6f}")  # 1.000000 (L2 normalized)
    ```

=== "Custom Seed"

    ```python
    # Generate different embeddings with different seeds
    vec1 = steadytext.embed("Hello world", seed=123)
    vec2 = steadytext.embed("Hello world", seed=123)  # Same as vec1
    vec3 = steadytext.embed("Hello world", seed=456)  # Different from vec1
    
    print(f"Seed 123 vs 123 equal: {np.array_equal(vec1, vec2)}")  # True
    print(f"Seed 123 vs 456 equal: {np.array_equal(vec1, vec3)}")  # False
    
    # Calculate similarity between different seed embeddings
    similarity = np.dot(vec1, vec3)  # Cosine similarity (vectors are normalized)
    print(f"Similarity between seeds: {similarity:.3f}")
    ```

=== "Multiple Texts"

    ```python
    # Embed multiple texts (averaged)
    texts = ["machine learning", "artificial intelligence", "deep learning"]
    vector = steadytext.embed(texts)
    
    print(f"Combined embedding shape: {vector.shape}")  # (1024,)
    # Result is averaged across all input texts
    ```

=== "Similarity Comparison"

    ```python
    import numpy as np
    
    # Create embeddings for comparison with consistent seed
    seed = 42
    vec1 = steadytext.embed("machine learning", seed=seed)
    vec2 = steadytext.embed("artificial intelligence", seed=seed) 
    vec3 = steadytext.embed("cooking recipes", seed=seed)
    
    # Calculate cosine similarity (vectors are already L2 normalized)
    sim_ml_ai = np.dot(vec1, vec2)
    sim_ml_cooking = np.dot(vec1, vec3)
    
    print(f"ML vs AI similarity: {sim_ml_ai:.3f}")
    print(f"ML vs Cooking similarity: {sim_ml_cooking:.3f}")
    # ML and AI should have higher similarity than ML and cooking
    
    # Compare same text with different seeds
    vec_seed1 = steadytext.embed("machine learning", seed=100)
    vec_seed2 = steadytext.embed("machine learning", seed=200)
    seed_similarity = np.dot(vec_seed1, vec_seed2)
    print(f"Same text, different seeds similarity: {seed_similarity:.3f}")
    ```

---

## Advanced Usage

### Deterministic Behavior

Embeddings are completely deterministic for the same input text and seed:

```python
# Same text, same seed - always identical
vec1 = steadytext.embed("test text")
vec2 = steadytext.embed("test text")
assert np.array_equal(vec1, vec2)  # Always passes!

# Same text, explicit same seed - always identical
vec3 = steadytext.embed("test text", seed=42)
vec4 = steadytext.embed("test text", seed=42)
assert np.array_equal(vec3, vec4)  # Always passes!

# Same text, different seeds - different results
vec5 = steadytext.embed("test text", seed=123)
vec6 = steadytext.embed("test text", seed=456)
assert not np.array_equal(vec5, vec6)  # Different seeds produce different embeddings

# But each seed is still deterministic
vec7 = steadytext.embed("test text", seed=123)
assert np.array_equal(vec5, vec7)  # Same seed always produces same result
```

### Seed Use Cases

```python
# Experimental variations - try different embeddings for the same text
text = "artificial intelligence"
baseline_embedding = steadytext.embed(text, seed=42)
variation1 = steadytext.embed(text, seed=100)
variation2 = steadytext.embed(text, seed=200)

# Compare variations
print(f"Baseline vs Variation 1: {np.dot(baseline_embedding, variation1):.3f}")
print(f"Baseline vs Variation 2: {np.dot(baseline_embedding, variation2):.3f}")
print(f"Variation 1 vs Variation 2: {np.dot(variation1, variation2):.3f}")

# Reproducible research - document your seeds
research_texts = ["AI", "ML", "DL"]
research_seed = 42
embeddings = []
for text in research_texts:
    embedding = steadytext.embed(text, seed=research_seed)
    embeddings.append(embedding)
    print(f"Text: {text}, Seed: {research_seed}")
```

### Preprocessing

Text is automatically preprocessed before embedding:

```python
# These produce different embeddings due to different text
vec1 = steadytext.embed("Hello World")
vec2 = steadytext.embed("hello world")
vec3 = steadytext.embed("HELLO WORLD")

# Case sensitivity matters
assert not np.array_equal(vec1, vec2)
```

### Batch Processing

For multiple texts, pass as a list with consistent seeding:

```python
# Individual embeddings with consistent seed
seed = 42
vec1 = steadytext.embed("first text", seed=seed)
vec2 = steadytext.embed("second text", seed=seed) 
vec3 = steadytext.embed("third text", seed=seed)

# Batch embedding (averaged) with same seed
vec_batch = steadytext.embed(["first text", "second text", "third text"], seed=seed)

# The batch result is the average of individual embeddings
expected = (vec1 + vec2 + vec3) / 3
expected = expected / np.linalg.norm(expected)  # Re-normalize after averaging
assert np.allclose(vec_batch, expected, atol=1e-6)

# Different seeds produce different batch results
vec_batch_alt = steadytext.embed(["first text", "second text", "third text"], seed=123)
assert not np.array_equal(vec_batch, vec_batch_alt)
```

### Caching

Embeddings are cached for performance, with seed as part of the cache key:

```python
# First call: computes and caches embedding for default seed
vec1 = steadytext.embed("common text")  # ~0.5 seconds

# Second call with same seed: returns cached result
vec2 = steadytext.embed("common text")  # ~0.01 seconds
assert np.array_equal(vec1, vec2)  # Same result, much faster

# Different seed: computes and caches separately
vec3 = steadytext.embed("common text", seed=123)  # ~0.5 seconds (new cache entry)
vec4 = steadytext.embed("common text", seed=123)  # ~0.01 seconds (cached)

assert np.array_equal(vec3, vec4)  # Same seed, same cached result
assert not np.array_equal(vec1, vec3)  # Different seeds, different results

# Each seed gets its own cache entry
for seed in [100, 200, 300]:
    steadytext.embed("cache test", seed=seed)  # Each gets cached separately
```

### Fallback Behavior

When models can't be loaded, deterministic fallback vectors are generated using the seed:

```python
# Even without models, function never fails and respects seeds
vector1 = steadytext.embed("any text", seed=42)
vector2 = steadytext.embed("any text", seed=42)
vector3 = steadytext.embed("any text", seed=123)

assert vector1.shape == (1024,)     # Correct shape
assert vector1.dtype == np.float32  # Correct type
assert np.array_equal(vector1, vector2)  # Same seed, same fallback
assert not np.array_equal(vector1, vector3)  # Different seed, different fallback

# Fallback vectors are normalized and deterministic
assert abs(np.linalg.norm(vector1) - 1.0) < 1e-6  # Properly normalized
```

---

## Use Cases

### Document Similarity

```python
import steadytext
import numpy as np

def document_similarity(doc1: str, doc2: str, seed: int = 42) -> float:
    """Calculate similarity between two documents."""
    vec1 = steadytext.embed(doc1, seed=seed)
    vec2 = steadytext.embed(doc2, seed=seed)
    return np.dot(vec1, vec2)  # Already L2 normalized

# Usage
similarity = document_similarity(
    "Machine learning algorithms",
    "AI and neural networks"
)
print(f"Similarity: {similarity:.3f}")
```

### Semantic Search

```python
def semantic_search(query: str, documents: List[str], top_k: int = 5, seed: int = 42):
    """Find most similar documents to query."""
    query_vec = steadytext.embed(query, seed=seed)
    doc_vecs = [steadytext.embed(doc, seed=seed) for doc in documents]
    
    similarities = [np.dot(query_vec, doc_vec) for doc_vec in doc_vecs]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [(documents[i], similarities[i]) for i in top_indices]

# Usage  
docs = ["AI research", "Machine learning", "Cooking recipes", "Data science"]
results = semantic_search("artificial intelligence", docs, top_k=2)

for doc, score in results:
    print(f"{doc}: {score:.3f}")
```

### Clustering

```python
from sklearn.cluster import KMeans
import numpy as np

def cluster_texts(texts: List[str], n_clusters: int = 3, seed: int = 42):
    """Cluster texts using their embeddings."""
    embeddings = np.array([steadytext.embed(text, seed=seed) for text in texts])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    return clusters

# Usage
texts = [
    "machine learning", "deep learning", "neural networks",  # AI cluster
    "pizza recipe", "pasta cooking", "italian food",        # Food cluster  
    "stock market", "trading", "investment"                 # Finance cluster
]

clusters = cluster_texts(texts, n_clusters=3)
for text, cluster in zip(texts, clusters):
    print(f"Cluster {cluster}: {text}")
```

---

## Performance Notes

!!! tip "Optimization Tips"
    - **Preload models**: Call `steadytext.preload_models()` at startup
    - **Batch similar texts**: Group related texts together for cache efficiency  
    - **Memory usage**: ~610MB for embedding model (loaded once)
    - **Speed**: ~100-500 embeddings/second depending on text length
    - **Seed consistency**: Use consistent seeds across related embeddings for comparable results
    - **Cache efficiency**: Different seeds create separate cache entries, so choose seeds wisely
# Embeddings API

Functions for creating deterministic text embeddings.

## embed()

Create deterministic embeddings for text input.

```python
def embed(text_input: Union[str, List[str]]) -> np.ndarray
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `text_input` | `Union[str, List[str]]` | Text string or list of strings to embed |

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
    
    # Create embeddings for comparison
    vec1 = steadytext.embed("machine learning")
    vec2 = steadytext.embed("artificial intelligence") 
    vec3 = steadytext.embed("cooking recipes")
    
    # Calculate cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_ml_ai = cosine_similarity(vec1, vec2)
    sim_ml_cooking = cosine_similarity(vec1, vec3)
    
    print(f"ML vs AI similarity: {sim_ml_ai:.3f}")
    print(f"ML vs Cooking similarity: {sim_ml_cooking:.3f}")
    # ML and AI should have higher similarity than ML and cooking
    ```

---

## Advanced Usage

### Deterministic Behavior

Embeddings are completely deterministic:

```python
# These will always be identical
vec1 = steadytext.embed("test text")
vec2 = steadytext.embed("test text")

assert np.array_equal(vec1, vec2)  # Always passes!
assert np.allclose(vec1, vec2)     # Always passes!
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

For multiple texts, pass as a list:

```python
# Individual embeddings
vec1 = steadytext.embed("first text")
vec2 = steadytext.embed("second text") 
vec3 = steadytext.embed("third text")

# Batch embedding (averaged)
vec_batch = steadytext.embed(["first text", "second text", "third text"])

# The batch result is the average of individual embeddings
expected = (vec1 + vec2 + vec3) / 3
assert np.allclose(vec_batch, expected, atol=1e-6)
```

### Caching

Embeddings are cached for performance:

```python
# First call: computes and caches embedding
vec1 = steadytext.embed("common text")  # ~0.5 seconds

# Second call: returns cached result
vec2 = steadytext.embed("common text")  # ~0.01 seconds

assert np.array_equal(vec1, vec2)  # Same result, much faster
```

### Fallback Behavior

When models can't be loaded, zero vectors are returned:

```python
# Even without models, function never fails
vector = steadytext.embed("any text")

assert vector.shape == (1024,)     # Correct shape
assert vector.dtype == np.float32  # Correct type
assert np.linalg.norm(vector) == 0 # Zero vector fallback
```

---

## Use Cases

### Document Similarity

```python
import steadytext
import numpy as np

def document_similarity(doc1: str, doc2: str) -> float:
    """Calculate similarity between two documents."""
    vec1 = steadytext.embed(doc1)
    vec2 = steadytext.embed(doc2)
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
def semantic_search(query: str, documents: List[str], top_k: int = 5):
    """Find most similar documents to query."""
    query_vec = steadytext.embed(query)
    doc_vecs = [steadytext.embed(doc) for doc in documents]
    
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

def cluster_texts(texts: List[str], n_clusters: int = 3):
    """Cluster texts using their embeddings."""
    embeddings = np.array([steadytext.embed(text) for text in texts])
    
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
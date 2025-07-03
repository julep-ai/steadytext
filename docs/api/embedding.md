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
    # Embed multiple texts (returns a single, averaged embedding)
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

---

## Advanced Examples

### Vector Database Integration

```python
import steadytext
import numpy as np
import faiss

class VectorDB:
    """Simple vector database using FAISS."""
    
    def __init__(self, dimension: int = 1024, seed: int = 42):
        self.dimension = dimension
        self.seed = seed
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
    
    def add_documents(self, documents: list, ids: list = None):
        """Add documents to the vector database."""
        embeddings = []
        
        for i, doc in enumerate(documents):
            # Use consistent seed for all documents
            vec = steadytext.embed(doc, seed=self.seed)
            embeddings.append(vec)
            
            # Store metadata
            self.metadata.append({
                'id': ids[i] if ids else i,
                'text': doc,
                'embedding': vec
            })
        
        # Add to FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
    
    def search(self, query: str, k: int = 5):
        """Search for similar documents."""
        # Use same seed as documents
        query_vec = steadytext.embed(query, seed=self.seed).reshape(1, -1)
        
        # Search in FAISS
        distances, indices = self.index.search(query_vec.astype('float32'), k)
        
        # Return results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    'id': self.metadata[idx]['id'],
                    'text': self.metadata[idx]['text'],
                    'distance': distances[0][i],
                    'similarity': 1 / (1 + distances[0][i])  # Convert distance to similarity
                })
        
        return results

# Example usage
db = VectorDB(seed=100)  # Custom seed for this database

# Add documents
documents = [
    "Introduction to machine learning algorithms",
    "Deep learning with neural networks",
    "Natural language processing basics",
    "Computer vision applications",
    "Reinforcement learning in robotics"
]

db.add_documents(documents, ids=['ML101', 'DL201', 'NLP301', 'CV401', 'RL501'])

# Search
results = db.search("text processing and NLP", k=3)
for result in results:
    print(f"ID: {result['id']}, Similarity: {result['similarity']:.3f}")
    print(f"Text: {result['text']}\n")
```

### Multi-Modal Embeddings

```python
import steadytext
import numpy as np
from typing import Dict, Any

class MultiModalEmbedder:
    """Create combined embeddings from multiple modalities."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.modality_seeds = {
            'text': base_seed,
            'title': base_seed + 1000,
            'tags': base_seed + 2000,
            'category': base_seed + 3000
        }
    
    def embed_document(self, document: Dict[str, Any]) -> np.ndarray:
        """Create a combined embedding from multiple fields."""
        embeddings = []
        weights = []
        
        # Embed each modality with its own seed
        if 'text' in document and document['text']:
            vec = steadytext.embed(document['text'], seed=self.modality_seeds['text'])
            embeddings.append(vec)
            weights.append(0.5)  # Main content gets highest weight
        
        if 'title' in document and document['title']:
            vec = steadytext.embed(document['title'], seed=self.modality_seeds['title'])
            embeddings.append(vec)
            weights.append(0.3)
        
        if 'tags' in document and document['tags']:
            # Combine tags into single text
            tags_text = " ".join(document['tags'])
            vec = steadytext.embed(tags_text, seed=self.modality_seeds['tags'])
            embeddings.append(vec)
            weights.append(0.15)
        
        if 'category' in document and document['category']:
            vec = steadytext.embed(document['category'], seed=self.modality_seeds['category'])
            embeddings.append(vec)
            weights.append(0.05)
        
        if not embeddings:
            # Fallback to zero vector if no content
            return np.zeros(1024, dtype=np.float32)
        
        # Weighted average
        weights = np.array(weights) / sum(weights)  # Normalize weights
        combined = np.average(embeddings, axis=0, weights=weights)
        
        # Re-normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined

# Example usage
embedder = MultiModalEmbedder(base_seed=200)

# Document with multiple fields
doc1 = {
    'title': 'Introduction to Machine Learning',
    'text': 'Machine learning is a subset of artificial intelligence...',
    'tags': ['ML', 'AI', 'tutorial', 'beginner'],
    'category': 'Education'
}

doc2 = {
    'title': 'Advanced Deep Learning Techniques',
    'text': 'Deep learning has revolutionized computer vision...',
    'tags': ['DL', 'neural networks', 'advanced'],
    'category': 'Research'
}

# Create multi-modal embeddings
vec1 = embedder.embed_document(doc1)
vec2 = embedder.embed_document(doc2)

# Compare similarity
similarity = np.dot(vec1, vec2)
print(f"Document similarity: {similarity:.3f}")
```

### Incremental Embedding Updates

```python
import steadytext
import numpy as np
from collections import deque

class IncrementalEmbedder:
    """Maintain running average embeddings for evolving content."""
    
    def __init__(self, window_size: int = 10, seed: int = 42):
        self.window_size = window_size
        self.seed = seed
        self.history = deque(maxlen=window_size)
        self.current_embedding = None
    
    def add_text(self, text: str) -> np.ndarray:
        """Add new text and update running embedding."""
        # Embed new text
        new_embedding = steadytext.embed(text, seed=self.seed)
        self.history.append(new_embedding)
        
        # Calculate running average
        if len(self.history) > 0:
            avg_embedding = np.mean(list(self.history), axis=0)
            # Re-normalize
            self.current_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return self.current_embedding
    
    def get_evolution(self) -> list:
        """Get the evolution of embeddings over time."""
        evolution = []
        temp_history = []
        
        for emb in self.history:
            temp_history.append(emb)
            avg = np.mean(temp_history, axis=0)
            avg = avg / np.linalg.norm(avg)
            evolution.append(avg)
        
        return evolution

# Example: Track topic drift in conversation
embedder = IncrementalEmbedder(window_size=5, seed=300)

conversation = [
    "Let's talk about machine learning",
    "Neural networks are fascinating",
    "Deep learning has many applications",
    "But what about traditional algorithms?",
    "Random forests are still useful",
    "Statistical methods have their place",
    "Linear regression is fundamental"
]

print("Conversation evolution:")
for i, text in enumerate(conversation):
    embedding = embedder.add_text(text)
    
    if i > 0:
        # Compare to previous state
        evolution = embedder.get_evolution()
        similarity = np.dot(evolution[-1], evolution[0])
        print(f"Step {i}: '{text[:30]}...' - Drift from start: {1-similarity:.3f}")
```

### Embedding Dimensionality Reduction

```python
import steadytext
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class EmbeddingVisualizer:
    """Visualize high-dimensional embeddings in 2D/3D."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.embeddings = []
        self.labels = []
    
    def add_texts(self, texts: list, labels: list = None):
        """Add texts with optional labels."""
        for i, text in enumerate(texts):
            emb = steadytext.embed(text, seed=self.seed)
            self.embeddings.append(emb)
            self.labels.append(labels[i] if labels else str(i))
    
    def reduce_pca(self, n_components: int = 2) -> np.ndarray:
        """Reduce dimensions using PCA."""
        if not self.embeddings:
            return np.array([])
        
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(np.array(self.embeddings))
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        return reduced
    
    def reduce_tsne(self, n_components: int = 2) -> np.ndarray:
        """Reduce dimensions using t-SNE."""
        if not self.embeddings:
            return np.array([])
        
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=5)
        reduced = tsne.fit_transform(np.array(self.embeddings))
        return reduced
    
    def plot_2d(self, method: str = 'pca'):
        """Create 2D visualization."""
        if method == 'pca':
            reduced = self.reduce_pca(2)
        else:
            reduced = self.reduce_tsne(2)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1])
        
        for i, label in enumerate(self.labels):
            plt.annotate(label, (reduced[i, 0], reduced[i, 1]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title(f'Embedding Visualization ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)
        return plt

# Example usage
viz = EmbeddingVisualizer(seed=400)

# Add different categories of text
categories = {
    'AI': ["machine learning", "neural networks", "deep learning"],
    'Food': ["pizza recipe", "pasta cooking", "italian cuisine"],
    'Finance': ["stock market", "investment strategy", "trading"]
}

for category, texts in categories.items():
    for text in texts:
        viz.add_texts([text], labels=[f"{category}: {text}"])

# Visualize (would display plot in Jupyter)
# plot = viz.plot_2d('tsne')
# plot.show()
```

### Cross-Lingual Embeddings

```python
import steadytext
import numpy as np

class CrossLingualEmbedder:
    """Create aligned embeddings across languages using seed variations."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        # Different seed offsets for different languages
        self.language_seeds = {
            'en': base_seed,
            'es': base_seed + 10000,
            'fr': base_seed + 20000,
            'de': base_seed + 30000,
            'zh': base_seed + 40000
        }
    
    def embed(self, text: str, language: str = 'en') -> np.ndarray:
        """Embed text with language-specific seed."""
        if language not in self.language_seeds:
            language = 'en'  # Fallback to English
        
        seed = self.language_seeds[language]
        return steadytext.embed(text, seed=seed)
    
    def align_embeddings(self, source_texts: list, target_texts: list, 
                        source_lang: str, target_lang: str) -> tuple:
        """Create aligned embeddings for parallel texts."""
        source_embeddings = [self.embed(text, source_lang) for text in source_texts]
        target_embeddings = [self.embed(text, target_lang) for text in target_texts]
        
        # Simple alignment: compute transformation matrix
        # In practice, you'd use more sophisticated methods
        S = np.array(source_embeddings)
        T = np.array(target_embeddings)
        
        # Compute pseudo-inverse for alignment
        # W = T @ S.T @ np.linalg.inv(S @ S.T)
        # For simplicity, we'll just return the embeddings
        
        return source_embeddings, target_embeddings
    
    def cross_lingual_similarity(self, text1: str, lang1: str, 
                               text2: str, lang2: str) -> float:
        """Compute similarity across languages."""
        vec1 = self.embed(text1, lang1)
        vec2 = self.embed(text2, lang2)
        
        # Apply simple heuristic adjustment for cross-lingual comparison
        # In practice, you'd use learned alignment
        if lang1 != lang2:
            # Reduce similarity slightly for different languages
            adjustment = 0.9
        else:
            adjustment = 1.0
        
        return np.dot(vec1, vec2) * adjustment

# Example usage
embedder = CrossLingualEmbedder(base_seed=500)

# Embed in different languages
en_vec = embedder.embed("Hello world", "en")
es_vec = embedder.embed("Hola mundo", "es")
fr_vec = embedder.embed("Bonjour le monde", "fr")

# Compare cross-lingual similarities
print("Cross-lingual similarities:")
print(f"EN-ES: {embedder.cross_lingual_similarity('Hello world', 'en', 'Hola mundo', 'es'):.3f}")
print(f"EN-FR: {embedder.cross_lingual_similarity('Hello world', 'en', 'Bonjour le monde', 'fr'):.3f}")
print(f"ES-FR: {embedder.cross_lingual_similarity('Hola mundo', 'es', 'Bonjour le monde', 'fr'):.3f}")

# Same language comparison
en_sim = embedder.cross_lingual_similarity('Hello world', 'en', 'Hi earth', 'en')
print(f"\nSame language (EN-EN): {en_sim:.3f}")
```

### Real-time Embedding Stream

```python
import steadytext
import numpy as np
import time
from typing import Iterator, Tuple

class EmbeddingStream:
    """Process streaming text data with real-time embeddings."""
    
    def __init__(self, chunk_size: int = 100, overlap: int = 20, seed: int = 42):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.seed = seed
        self.buffer = ""
        self.processed_count = 0
    
    def process_stream(self, text_stream: Iterator[str]) -> Iterator[Tuple[str, np.ndarray]]:
        """Process streaming text and yield embeddings."""
        for text in text_stream:
            self.buffer += text
            
            # Process complete chunks
            while len(self.buffer) >= self.chunk_size:
                # Extract chunk
                chunk = self.buffer[:self.chunk_size]
                
                # Generate embedding with position-based seed
                chunk_seed = self.seed + self.processed_count
                embedding = steadytext.embed(chunk, seed=chunk_seed)
                
                yield chunk, embedding
                
                # Move buffer forward with overlap
                self.buffer = self.buffer[self.chunk_size - self.overlap:]
                self.processed_count += 1
        
        # Process remaining buffer
        if self.buffer:
            final_seed = self.seed + self.processed_count
            embedding = steadytext.embed(self.buffer, seed=final_seed)
            yield self.buffer, embedding

# Example: Simulate streaming text
def text_generator():
    """Simulate streaming text data."""
    texts = [
        "Machine learning is transforming how we process information. ",
        "Neural networks can learn complex patterns from data. ",
        "Deep learning models require large amounts of training data. ",
        "Transfer learning helps when data is limited. ",
        "Embeddings capture semantic meaning in vector space. "
    ]
    
    for text in texts:
        # Simulate streaming by yielding words
        words = text.split()
        for word in words:
            yield word + " "
            time.sleep(0.1)  # Simulate real-time stream

# Process stream
stream_processor = EmbeddingStream(chunk_size=50, overlap=10, seed=600)

print("Processing text stream...")
embeddings_collected = []

for chunk, embedding in stream_processor.process_stream(text_generator()):
    print(f"Processed chunk: '{chunk[:30]}...' -> Embedding shape: {embedding.shape}")
    embeddings_collected.append(embedding)

# Analyze progression
if len(embeddings_collected) > 1:
    print(f"\nTotal chunks processed: {len(embeddings_collected)}")
    
    # Check similarity progression
    for i in range(1, len(embeddings_collected)):
        sim = np.dot(embeddings_collected[i-1], embeddings_collected[i])
        print(f"Similarity between chunk {i-1} and {i}: {sim:.3f}")
```

---

## Troubleshooting

### Common Issues

**Issue: Embeddings not deterministic**
```python
# Problem: Different results each run
vec1 = steadytext.embed("test")
# ... restart Python ...
vec2 = steadytext.embed("test")
# vec1 != vec2

# Solution: Ensure consistent seed and environment
import os
os.environ['PYTHONHASHSEED'] = '0'  # Set before importing steadytext
import steadytext

vec1 = steadytext.embed("test", seed=42)
vec2 = steadytext.embed("test", seed=42)
assert np.array_equal(vec1, vec2)  # Now deterministic
```

**Issue: Out of memory with large batches**
```python
# Problem: OOM with large text list
texts = ["text"] * 10000
vectors = [steadytext.embed(t) for t in texts]  # May OOM

# Solution: Process in batches
def embed_in_batches(texts, batch_size=100, seed=42):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            embeddings.append(steadytext.embed(text, seed=seed))
    return np.array(embeddings)

vectors = embed_in_batches(texts)
```

**Issue: Slow embedding generation**
```python
# Problem: First embedding is slow
import time

start = time.time()
vec1 = steadytext.embed("test")  # ~2-3 seconds (model loading)
print(f"First: {time.time() - start:.2f}s")

start = time.time()
vec2 = steadytext.embed("test")  # ~0.01 seconds (cached)
print(f"Second: {time.time() - start:.2f}s")

# Solution: Preload models
steadytext.preload_models()  # Load once at startup
# Now all embeddings will be fast
```
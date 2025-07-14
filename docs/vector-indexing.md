# Vector Indexing

SteadyText v1.3.3+ introduces vector indexing capabilities for building Retrieval-Augmented Generation (RAG) applications. Create searchable document indices using FAISS and automatically retrieve relevant context for your prompts.

## Overview

Vector indexing allows you to:

1. **Build Document Indices**: Convert text documents into searchable vector databases
2. **Semantic Search**: Find relevant information based on meaning, not just keywords  
3. **Automatic Context Retrieval**: Enhance generation with relevant document context
4. **Deterministic Chunking**: Reproducible text splitting using chonkie
5. **Efficient Storage**: FAISS indices for fast similarity search

## How It Works

The indexing system:

1. **Chunks documents** using chonkie with configurable chunk sizes and overlap
2. **Generates embeddings** for each chunk using SteadyText's embedding model
3. **Stores vectors** in a FAISS index for efficient similarity search
4. **Retrieves relevant chunks** based on query similarity
5. **Augments prompts** with retrieved context for better generation

## Installation Requirements

Vector indexing requires additional dependencies:

```bash
# Install required packages
pip install faiss-cpu chonkie

# Or install SteadyText with index extras
pip install "steadytext[index]"
```

## Basic Usage

### Creating an Index

Build an index from text files:

```bash
# Create index from a single file
st index create document.txt --output my_docs.faiss

# Create index from multiple files
st index create *.txt --output knowledge_base.faiss

# Create index with custom chunking
st index create docs/*.md --output docs.faiss --chunk-size 256 --chunk-overlap 30

# Overwrite existing index
st index create *.txt --output updated.faiss --force
```

### Searching an Index

Find relevant chunks:

```bash
# Basic search
st index search my_docs.faiss "how to install Python"

# Get more results
st index search docs.faiss "configuration options" --top-k 10

# Output as JSON for programmatic use
st index search knowledge.faiss "API reference" --json
```

### Getting Index Information

View index metadata:

```bash
# Show index info
st index info my_docs.faiss

# Output:
# Index: my_docs.faiss
#   Version: 1.0
#   Chunks: 245
#   Dimension: 1024
#   Chunk size: 512 tokens
#   Chunk overlap: 50 tokens
#   Index size: 245 vectors
# 
# Source files:
#   - document.txt
#     Hash: a3f5b2c1d4e6f8...
#     Size: 45,678 bytes
```

## Python API

### Creating Indices Programmatically

```python
from steadytext.index import create_index, search_index
from pathlib import Path

# Create index from files
files = ["doc1.txt", "doc2.txt", "doc3.md"]
index_path = create_index(
    input_files=files,
    output_path="my_index.faiss",
    chunk_size=512,      # tokens per chunk
    chunk_overlap=50,    # overlap between chunks
    force=True,          # overwrite if exists
    seed=42             # for deterministic embeddings
)

# Create index from text content
texts = [
    ("Python Tutorial", "Python is a high-level programming language..."),
    ("ML Guide", "Machine learning is a subset of AI..."),
]

chunks = []
for title, content in texts:
    # Chunk the content
    text_chunks = chunk_text(content, chunk_size=256)
    for chunk in text_chunks:
        chunks.append({
            "text": chunk,
            "source": title
        })

# Build index from chunks
index = build_faiss_index(chunks)
save_index(index, "content.faiss")
```

### Searching Indices

```python
from steadytext.index import load_index, search_index

# Load and search
index, metadata = load_index("my_index.faiss")
results = search_index(
    index=index,
    metadata=metadata,
    query="Python installation guide",
    top_k=5,
    seed=42
)

# Process results
for chunk, score, source in results:
    print(f"Score: {score:.3f}")
    print(f"Source: {source}")
    print(f"Content: {chunk[:100]}...")
    print("---")
```

## Automatic Context Retrieval

### Using Default Index

When a `default.faiss` index exists, SteadyText automatically retrieves context:

```bash
# Create default index
st index create *.txt --output default.faiss

# Generation now automatically uses context
echo "How do I configure logging?" | st

# Disable automatic context retrieval
echo "Write a poem" | st --no-index
```

### Specifying Custom Index

Use a specific index for context:

```bash
# Use custom index file
echo "What are the API endpoints?" | st --index-file api_docs.faiss

# Combine with other options
st generate "Explain the configuration" \
  --index-file config_docs.faiss \
  --top-k 5 \
  --json
```

## RAG Pipeline Examples

### Basic RAG Implementation

```python
import steadytext
from steadytext.index import search_index, load_index

def rag_generate(query, index_path="default.faiss", top_k=3):
    """Generate text with retrieved context."""
    
    # 1. Retrieve relevant chunks
    index, metadata = load_index(index_path)
    results = search_index(index, metadata, query, top_k=top_k)
    
    # 2. Build context from retrieved chunks
    context_parts = []
    for chunk, score, source in results:
        context_parts.append(f"From {source}:\n{chunk}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # 3. Create augmented prompt
    augmented_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    # 4. Generate response
    response = steadytext.generate(augmented_prompt)
    
    return response, results

# Example usage
answer, sources = rag_generate("How do I configure the database?")
print(answer)
```

### Advanced RAG with Reranking

Combine indexing with reranking for better results:

```python
import steadytext
from steadytext.index import search_index, load_index

def advanced_rag(query, index_path="default.faiss", initial_k=20, final_k=3):
    """RAG with reranking for improved relevance."""
    
    # 1. Initial retrieval (get more candidates)
    index, metadata = load_index(index_path)
    initial_results = search_index(index, metadata, query, top_k=initial_k)
    
    # 2. Extract chunks for reranking
    chunks = [chunk for chunk, _, _ in initial_results]
    
    # 3. Rerank for better relevance
    reranked = steadytext.rerank(
        query=query,
        documents=chunks,
        task="find passages that directly answer the question"
    )
    
    # 4. Use top reranked chunks
    context = "\n\n".join([doc for doc, _ in reranked[:final_k]])
    
    # 5. Generate with high-quality context
    prompt = f"Answer based on this context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    
    return steadytext.generate(prompt)
```

### Streaming RAG

Stream responses with context:

```python
def stream_rag(query, index_path="default.faiss"):
    """Stream RAG responses."""
    
    # Retrieve context
    index, metadata = load_index(index_path)
    results = search_index(index, metadata, query, top_k=3)
    
    # Build context
    context = "\n\n".join([chunk for chunk, _, _ in results])
    
    # Stream generation
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    for token in steadytext.generate_iter(prompt):
        yield token
```

## Chunking Strategies

### Standard Chunking

Default chunking with token overlap:

```python
# Using CLI
st index create docs.txt --chunk-size 512 --chunk-overlap 50

# Result: 512-token chunks with 50-token overlap for context continuity
```

### Semantic Chunking

Chunk by paragraphs or sections:

```python
def semantic_chunk(text, min_size=100, max_size=500):
    """Chunk text by paragraphs within size limits."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para.split())
        
        if current_size + para_size > max_size and current_chunk:
            # Save current chunk
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks
```

### Document-Aware Chunking

Preserve document structure:

```python
def chunk_markdown(content, chunk_size=400):
    """Chunk markdown while preserving headers."""
    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_headers = []
    current_size = 0
    
    for line in lines:
        # Track headers
        if line.startswith('#'):
            level = len(line.split()[0])
            current_headers = current_headers[:level-1] + [line]
        
        current_chunk.append(line)
        current_size += len(line.split())
        
        if current_size >= chunk_size:
            # Include headers in chunk for context
            full_chunk = '\n'.join(current_headers + current_chunk)
            chunks.append(full_chunk)
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append('\n'.join(current_headers + current_chunk))
    
    return chunks
```

## Performance Optimization

### Index Building

Speed up index creation:

```python
# 1. Batch embedding generation
def batch_create_embeddings(texts, batch_size=10):
    """Generate embeddings in batches."""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # SteadyText handles batching internally
        for text in batch:
            emb = steadytext.embed(text)
            embeddings.append(emb)
    
    return embeddings

# 2. Use appropriate index type
def create_optimized_index(embeddings, use_gpu=False):
    """Create FAISS index optimized for size/speed."""
    dimension = embeddings.shape[1]
    
    if len(embeddings) < 10000:
        # For small datasets, use flat index (exact search)
        index = faiss.IndexFlatL2(dimension)
    else:
        # For larger datasets, use IVF index
        nlist = int(np.sqrt(len(embeddings)))
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(embeddings)
    
    index.add(embeddings)
    return index
```

### Search Optimization

Improve search performance:

```python
# 1. Cache search results
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_search(index_path, query, top_k):
    """Cache search results for repeated queries."""
    index, metadata = load_index(index_path)
    return search_index(index, metadata, query, top_k)

# 2. Pre-filter candidates
def filtered_search(index, metadata, query, filter_source=None, top_k=5):
    """Search with source filtering."""
    # Get more results initially
    results = search_index(index, metadata, query, top_k=top_k*3)
    
    # Filter by source if specified
    if filter_source:
        results = [r for r in results if filter_source in r[2]]
    
    return results[:top_k]
```

## Best Practices

### 1. Chunk Size Selection

- **Small chunks (128-256 tokens)**: Better for precise retrieval
- **Medium chunks (256-512 tokens)**: Balanced context and precision
- **Large chunks (512-1024 tokens)**: Better for maintaining context

### 2. Index Management

```bash
# Organize indices by purpose
steadytext/
├── indices/
│   ├── default.faiss       # General knowledge base
│   ├── api_docs.faiss      # API documentation
│   ├── tutorials.faiss     # Tutorial content
│   └── troubleshooting.faiss  # Common issues
```

### 3. Metadata Tracking

Track document versions and updates:

```python
def create_versioned_index(files, version):
    """Create index with version tracking."""
    metadata = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "files": {}
    }
    
    for file in files:
        metadata["files"][file] = {
            "hash": calculate_file_hash(file),
            "modified": os.path.getmtime(file)
        }
    
    # Create index with metadata
    index_path = f"index_v{version}.faiss"
    create_index(files, index_path)
    save_metadata(metadata, f"{index_path}.meta")
```

### 4. Regular Updates

Keep indices current:

```bash
#!/bin/bash
# Update index if files changed

CHECKSUM_FILE=".index_checksum"
NEW_CHECKSUM=$(find docs -type f -name "*.md" -exec md5sum {} \; | sort | md5sum)

if [ ! -f "$CHECKSUM_FILE" ] || [ "$NEW_CHECKSUM" != "$(cat $CHECKSUM_FILE)" ]; then
    echo "Updating index..."
    st index create docs/*.md --output default.faiss --force
    echo "$NEW_CHECKSUM" > "$CHECKSUM_FILE"
fi
```

## Troubleshooting

### Common Issues

1. **"faiss-cpu not installed"**
   ```bash
   pip install faiss-cpu
   ```

2. **Index not found**
   ```bash
   # Check index location
   st index info ~/.cache/steadytext/indices/my_index.faiss
   ```

3. **Poor retrieval results**
   - Try smaller chunk sizes
   - Increase chunk overlap
   - Use reranking for better relevance

4. **Memory issues with large indices**
   - Use IVF indices for large datasets
   - Process files in batches
   - Consider dimension reduction

## Integration Examples

### FastAPI RAG Service

```python
from fastapi import FastAPI
import steadytext
from steadytext.index import load_index, search_index

app = FastAPI()

# Load index once at startup
INDEX, METADATA = load_index("knowledge_base.faiss")

@app.post("/ask")
async def ask_question(question: str, top_k: int = 3):
    # Retrieve context
    results = search_index(INDEX, METADATA, question, top_k)
    
    # Build context
    context = "\n\n".join([chunk for chunk, _, _ in results])
    
    # Generate answer
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = steadytext.generate(prompt)
    
    return {
        "question": question,
        "answer": answer,
        "sources": [{"text": chunk[:100], "source": src} 
                   for chunk, _, src in results]
    }
```

### Gradio Chat Interface

```python
import gradio as gr
import steadytext
from steadytext.index import load_index, search_index

def chat_with_docs(message, history, index_path="default.faiss"):
    # Load index
    index, metadata = load_index(index_path)
    
    # Retrieve context
    results = search_index(index, metadata, message, top_k=3)
    context = "\n".join([chunk for chunk, _, _ in results])
    
    # Generate response
    prompt = f"Context: {context}\n\nUser: {message}\nAssistant:"
    response = steadytext.generate(prompt)
    
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    chat_with_docs,
    title="Document Chat",
    description="Ask questions about your documents"
)

demo.launch()
```

## See Also

- [Reranking](reranking.md) - Improve retrieval quality with reranking
- [API Reference](api.md) - Detailed API documentation
- [CLI Reference](api/cli.md) - Complete CLI command reference
- [Examples](examples/index.md) - More code examples
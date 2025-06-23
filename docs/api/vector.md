# Vector Operations CLI

The `vector` command group provides various operations on text embeddings, enabling similarity comparisons, searches, and arithmetic operations.

## Overview

All vector operations work with SteadyText's 1024-dimensional L2-normalized embeddings. These embeddings are deterministic - the same text always produces the same embedding vector.

## Commands

### similarity

Compute similarity between two text embeddings.

```bash
st vector similarity TEXT1 TEXT2 [OPTIONS]
```

**Options:**
- `--metric [cosine|dot]`: Similarity metric to use (default: cosine)
- `--json`: Output as JSON

**Examples:**
```bash
# Basic cosine similarity
st vector similarity "apple" "orange"
# Output: 0.875432

# Dot product similarity
st vector similarity "king" "queen" --metric dot

# JSON output
st vector similarity "hello" "world" --json
```

### distance

Compute distance between two text embeddings.

```bash
st vector distance TEXT1 TEXT2 [OPTIONS]
```

**Options:**
- `--metric [euclidean|manhattan|cosine]`: Distance metric (default: euclidean)
- `--json`: Output as JSON

**Examples:**
```bash
# Euclidean distance
st vector distance "cat" "dog"

# Manhattan (L1) distance
st vector distance "yes" "no" --metric manhattan

# Cosine distance (1 - cosine_similarity)
st vector distance "hot" "cold" --metric cosine --json
```

### search

Find the most similar texts from a list of candidates.

```bash
st vector search QUERY [OPTIONS]
```

**Options:**
- `--candidates PATH`: File containing candidate texts (one per line)
- `--stdin`: Read candidates from stdin
- `--top N`: Number of top results to return (default: 1)
- `--metric [cosine|euclidean]`: Similarity/distance metric (default: cosine)
- `--json`: Output as JSON

**Examples:**
```bash
# Search from stdin
echo -e "apple\norange\ncar\ntruck" | st vector search "fruit" --stdin

# Search from file, get top 3
st vector search "python" --candidates languages.txt --top 3

# Search with euclidean distance
st vector search "query" --stdin --metric euclidean
```

### average

Compute the average of multiple text embeddings.

```bash
st vector average TEXT1 TEXT2 [TEXT3...] [OPTIONS]
```

**Options:**
- `--json`: Output as JSON with full embedding

**Examples:**
```bash
# Average animal embeddings
st vector average "cat" "dog" "hamster"

# Average programming languages
st vector average "python" "javascript" "rust" --json
```

### arithmetic

Perform vector arithmetic operations on embeddings.

```bash
st vector arithmetic BASE [ADD_TERMS...] [OPTIONS]
```

**Options:**
- `--subtract TEXT`: Terms to subtract (can be used multiple times)
- `--normalize/--no-normalize`: Whether to L2 normalize result (default: normalize)
- `--json`: Output as JSON

**Examples:**
```bash
# Classic word analogy: king + woman - man ≈ queen
st vector arithmetic "king" "woman" --subtract "man"

# Location arithmetic: paris - france + italy ≈ rome
st vector arithmetic "paris" "italy" --subtract "france"

# Multiple additions
st vector arithmetic "good" "better" "best"

# Without normalization
st vector arithmetic "hot" --subtract "cold" --no-normalize
```

## Output Formats

### Default Output
- **similarity/distance**: Single floating-point number
- **search**: Tab-separated lines of text and score
- **average/arithmetic**: First 50 values of resulting vector

### JSON Output
All commands support `--json` flag for structured output suitable for programmatic use.

## Technical Details

- **Embedding Dimension**: 1024
- **Normalization**: All embeddings are L2-normalized (unit vectors)
- **Determinism**: Same input always produces same output
- **Precision**: Float32 for all calculations
- **Cosine Similarity**: Since vectors are normalized, computed as simple dot product

## Use Cases

1. **Semantic Search**: Find related concepts from a database
2. **Document Similarity**: Compare similarity between texts
3. **Clustering**: Group similar items using distance metrics
4. **Analogies**: Explore semantic relationships with vector arithmetic
5. **Concept Interpolation**: Create blended concepts with averaging
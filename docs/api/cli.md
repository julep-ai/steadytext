# CLI Reference

Complete command-line interface documentation for SteadyText.

## Installation

The CLI is automatically installed with SteadyText:

```bash
# Using UV (recommended)
uv add steadytext

# Or using pip
pip install steadytext
```

Two commands are available:
- `steadytext` - Full command name
- `st` - Short alias

## Global Options

```bash
st --version     # Show version
st --help        # Show help
```

---

## generate

Generate deterministic text from a prompt.

### Usage

```bash
# New pipe syntax (recommended)
echo "prompt" | st [OPTIONS]
echo "prompt" | steadytext [OPTIONS]

# Legacy syntax (still supported)
st generate [OPTIONS] PROMPT
steadytext generate [OPTIONS] PROMPT
```

### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--wait` | `-w` | flag | `false` | Wait for complete output (disable streaming) |
| `--json` | `-j` | flag | `false` | Output as JSON with metadata |
| `--logprobs` | `-l` | flag | `false` | Include log probabilities |
| `--eos-string` | `-e` | string | `"[EOS]"` | Custom end-of-sequence string |
| `--max-new-tokens` | | int | `512` | Maximum number of tokens to generate |
| `--seed` | | int | `42` | Random seed for deterministic generation |
| `--size` | | choice | | Model size: small (2B, default), large (4B) |
| `--model` | | string | | Model name from registry (e.g., "qwen2.5-3b") |
| `--model-repo` | | string | | Custom model repository |
| `--model-filename` | | string | | Custom model filename |
| `--no-index` | | flag | `false` | Disable automatic index search |
| `--index-file` | | path | | Use specific index file |
| `--top-k` | | int | `3` | Number of context chunks to retrieve |
| `--schema` | | string | | JSON schema for structured output (file path or inline JSON) |
| `--regex` | | string | | Regular expression pattern for structured output |
| `--choices` | | string | | Comma-separated list of allowed choices |

### Examples

=== "Basic Generation"

    ```bash
    # New pipe syntax
    echo "Write a Python function to calculate fibonacci" | st
    
    # Legacy syntax
    st generate "Write a Python function to calculate fibonacci"
    ```

=== "Wait for Complete Output"

    ```bash
    # Disable streaming
    echo "Explain machine learning" | st --wait
    ```


=== "JSON Output"

    ```bash
    st generate "Hello world" --json
    # Output:
    # {
    #   "text": "Hello! How can I help you today?...",
    #   "tokens": 15,
    #   "cached": false
    # }
    ```

=== "With Log Probabilities"

    ```bash
    st generate "Explain AI" --logprobs --json
    # Includes token probabilities in JSON output
    ```

=== "Custom Stop String"

    ```bash
    st generate "List colors until STOP" --eos-string "STOP"
    ```

=== "Custom Seed for Reproducibility"

    ```bash
    # Generate with specific seed for reproducible results
    echo "Write a story" | st --seed 123
    
    # Same seed always produces same output
    st generate "Tell me a joke" --seed 456
    st generate "Tell me a joke" --seed 456  # Identical result
    
    # Different seeds produce different outputs
    st generate "Explain AI" --seed 100
    st generate "Explain AI" --seed 200  # Different result
    ```

=== "Custom Length"

    ```bash
    # Generate shorter responses
    echo "Quick summary of Python" | st --max-new-tokens 50
    
    # Generate longer responses
    echo "Detailed explanation of ML" | st --max-new-tokens 200
    ```

=== "Using Size Parameter"

    ```bash
    # Fast generation with small model
    st generate "Quick response" --size small
    
    # High quality with large model  
    st generate "Complex analysis" --size large
    
    # Combine size with custom seed
    st generate "Technical explanation" --size large --seed 789
    ```

=== "Model Selection"

    ```bash
    # Use specific model size
    st generate "Technical explanation" --size large
    
    # Use custom model (advanced)
    st generate "Write code" --model-repo ggml-org/gemma-3n-E4B-it-GGUF \
        --model-filename gemma-3n-E4B-it-Q8_0.gguf
    
    # Custom model with seed and length control
    st generate "Complex task" --model-repo ggml-org/gemma-3n-E4B-it-GGUF \
        --model-filename gemma-3n-E4B-it-Q8_0.gguf \
        --seed 999 --max-new-tokens 100
    ```

=== "Structured JSON Output"

    ```bash
    # Generate JSON with inline schema
    echo "Create a person" | st --schema '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}' --wait
    
    # Generate JSON from schema file
    echo "Generate user data" | st --schema user_schema.json --wait
    
    # Complex schema example
    echo "Create a product listing" | st --schema '{"type": "object", "properties": {"title": {"type": "string"}, "price": {"type": "number"}, "inStock": {"type": "boolean"}}}' --wait
    ```

=== "Regex Pattern Matching"

    ```bash
    # Phone number pattern
    echo "My phone number is" | st --regex '\d{3}-\d{3}-\d{4}' --wait
    
    # Date pattern
    echo "Today's date is" | st --regex '\d{4}-\d{2}-\d{2}' --wait
    
    # Custom pattern
    echo "The product code is" | st --regex '[A-Z]{3}-\d{4}' --wait
    ```

=== "Choice Constraints"

    ```bash
    # Simple yes/no choice
    echo "Is Python a good language?" | st --choices "yes,no" --wait
    
    # Multiple choice
    echo "What's the weather like?" | st --choices "sunny,cloudy,rainy,snowy" --wait
    
    # Decision making
    echo "Should we proceed with deployment?" | st --choices "proceed,wait,cancel" --wait
    ```

### Structured Generation Notes

!!! warning "Structured Generation Requirements"
    - **Streaming not supported**: Always use `--wait` flag with structured options
    - **Mutually exclusive**: Only one of `--schema`, `--regex`, or `--choices` can be used at a time
    - **Schema format**: Can be inline JSON or path to a `.json` file
    - **Choices format**: Comma-separated values without spaces around commas

### Stdin Support

Generate from stdin when no prompt provided:

```bash
echo "Write a haiku" | st generate
cat prompts.txt | st generate --stream
```

---

## embed

Create deterministic embeddings for text.

### Usage

```bash
st embed [OPTIONS] TEXT
steadytext embed [OPTIONS] TEXT
```

### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | choice | `json` | Output format: `json`, `numpy`, `hex` |
| `--output` | `-o` | path | `-` | Output file (default: stdout) |
| `--seed` | | int | `42` | Random seed for deterministic embedding generation |

### Examples

=== "Basic Embedding"

    ```bash
    st embed "machine learning"
    # Outputs JSON array with 1024 float values
    ```

=== "Custom Seed"

    ```bash
    # Generate reproducible embeddings
    st embed "artificial intelligence" --seed 123
    st embed "artificial intelligence" --seed 123  # Same result
    st embed "artificial intelligence" --seed 456  # Different result
    
    # Compare embeddings with different seeds
    st embed "test text" --seed 100 --format json > embed1.json
    st embed "test text" --seed 200 --format json > embed2.json
    ```

=== "Numpy Format"

    ```bash
    st embed "text to embed" --format numpy
    # Outputs binary numpy array
    ```

=== "Hex Format"

    ```bash
    st embed "hello world" --format hex
    # Outputs hex-encoded float32 array
    ```

=== "Save to File"

    ```bash
    st embed "important text" --output embedding.json
    st embed "data" --format numpy --output embedding.npy
    
    # Save with custom seed
    st embed "research data" --seed 42 --output research_embedding.json
    st embed "experiment" --seed 123 --format numpy --output exp_embed.npy
    ```

### Stdin Support

Embed text from stdin:

```bash
echo "text to embed" | st embed
cat document.txt | st embed --format numpy --output doc_embedding.npy

# Stdin with custom seed
echo "text to embed" | st embed --seed 789
cat document.txt | st embed --seed 42 --format numpy --output doc_embed_s42.npy
```

---

## models

Manage SteadyText models.

### Usage

```bash
st models [OPTIONS]
steadytext models [OPTIONS]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--list` | `-l` | List available models |
| `--preload` | `-p` | Preload all models |
| `--cache-dir` |  | Show model cache directory |
| `--json` | flag | `false` | Output as JSON |
| `--seed` | | int | Random seed for model operations |

### Commands

| Command | Description |
|---------|-------------|
| `status` | Check model download status |
| `list` | List available models |
| `download` | Pre-download models |
| `delete` | Delete cached models |
| `preload` | Preload models into memory |
| `path` | Show model cache directory |

### Examples

=== "List Models"

    ```bash
    st models list
    # Output:
    # Size Shortcuts:
    #   small → gemma-3n-2b
    #   large → gemma-3n-4b
    #
    # Available Models:
    #   gemma-3n-2b
    #     Repository: ggml-org/gemma-3n-E2B-it-GGUF
    #     Filename: gemma-3n-E2B-it-Q8_0.gguf
    #   gemma-3n-4b
    #     Repository: ggml-org/gemma-3n-E4B-it-GGUF
    #     Filename: gemma-3n-E4B-it-Q8_0.gguf
    ```

=== "Download Models"

    ```bash
    # Download default models
    st models download

    # Download by size
    st models download --size small

    # Download by name
    st models download --model gemma-3n-4b

    # Download all models
    st models download --all
    ```

=== "Delete Models"

    ```bash
    # Delete by size
    st models delete --size small

    # Delete by name
    st models delete --model gemma-3n-4b

    # Delete all models with confirmation
    st models delete --all

    # Force delete all models without confirmation
    st models delete --all --force
    ```

=== "Preload Models"

    ```bash
    st models preload
    # Downloads and loads all models
    
    # Preload with specific seed for deterministic initialization
    st models preload --seed 42
    ```

=== "Cache Information"

    ```bash
    st models path
    # /home/user/.cache/steadytext/models/

    st models status
    # {
    #   "model_directory": "/home/user/.cache/steadytext/models",
    #   "models": { ... }
    # }
    ```

---

## vector

Perform vector operations on embeddings.

### Usage

```bash
st vector COMMAND [OPTIONS]
steadytext vector COMMAND [OPTIONS]
```

### Commands

| Command | Description |
|---------|-------------|
| `similarity` | Compute similarity between text embeddings |
| `distance` | Compute distance between text embeddings |
| `search` | Find most similar texts from candidates |
| `average` | Compute average of multiple embeddings |
| `arithmetic` | Perform vector arithmetic operations |

### Global Vector Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--seed` | int | `42` | Random seed for deterministic embeddings |
| `--json` | flag | `false` | Output as JSON with metadata |

### Examples

=== "Similarity"

    ```bash
    # Cosine similarity
    st vector similarity "cat" "dog"
    # 0.823456
    
    # With JSON output
    st vector similarity "king" "queen" --json
    
    # Reproducible similarity with custom seed
    st vector similarity "king" "queen" --seed 123
    st vector similarity "king" "queen" --seed 123  # Same result
    st vector similarity "king" "queen" --seed 456  # Different result
    ```

=== "Distance"

    ```bash
    # Euclidean distance
    st vector distance "hot" "cold"
    
    # Manhattan distance
    st vector distance "yes" "no" --metric manhattan
    ```

=== "Search"

    ```bash
    # Find similar from stdin
    echo -e "apple\norange\ncar" | st vector search "fruit" --stdin
    
    # From file, top 3
    st vector search "python" --candidates langs.txt --top 3
    
    # Reproducible search with custom seed
    echo -e "apple\norange\ncar" | st vector search "fruit" --stdin --seed 789
    st vector search "programming" --candidates langs.txt --top 3 --seed 42
    ```

=== "Average"

    ```bash
    # Average embeddings
    st vector average "cat" "dog" "hamster"
    
    # With full embedding output
    st vector average "red" "green" "blue" --json
    
    # Reproducible averaging with custom seed
    st vector average "cat" "dog" "hamster" --seed 555
    st vector average "colors" "shapes" "sizes" --seed 666 --json
    ```

=== "Arithmetic"

    ```bash
    # Classic analogy: king + woman - man ≈ queen
    st vector arithmetic "king" "woman" --subtract "man"
    
    # Location arithmetic
    st vector arithmetic "paris" "italy" --subtract "france"
    
    # Reproducible arithmetic with custom seed
    st vector arithmetic "king" "woman" --subtract "man" --seed 777
    st vector arithmetic "tokyo" "italy" --subtract "japan" --seed 888 --json
    ```

See [Vector Operations](#vector) for detailed usage.

---

## rerank

Rerank documents based on relevance to a query (v2.3.0+).

### Usage

```bash
st rerank [OPTIONS] QUERY [DOCUMENTS...]
steadytext rerank [OPTIONS] QUERY [DOCUMENTS...]
```

### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--file` | `-f` | path | | Read documents from file (one per line) |
| `--stdin` | | flag | `false` | Read documents from stdin |
| `--top-k` | `-k` | int | | Return only top K results |
| `--json` | `-j` | flag | `false` | Output as JSON with scores |
| `--task` | `-t` | string | `"text retrieval for user question"` | Task description for better results |
| `--seed` | | int | `42` | Random seed for deterministic reranking |

### Examples

=== "Basic Reranking"

    ```bash
    # Rerank files
    st rerank "Python programming" doc1.txt doc2.txt doc3.txt
    
    # With custom seed
    st rerank "Python programming" doc1.txt doc2.txt doc3.txt --seed 123
    ```

=== "From File"

    ```bash
    # Documents in file (one per line)
    st rerank "machine learning" --file documents.txt
    
    # Top 5 results with custom seed
    st rerank "deep learning" --file papers.txt --top-k 5 --seed 456
    ```

=== "From Stdin"

    ```bash
    # Pipe documents
    cat documents.txt | st rerank "search query" --stdin
    
    # From command output
    find . -name "*.md" -exec cat {} \; | st rerank "installation guide" --stdin --top-k 3
    ```

=== "JSON Output"

    ```bash
    # Get scores with documents
    st rerank "Python" doc1.txt doc2.txt --json
    # Output:
    # [
    #   {"document": "Python is a programming language...", "score": 0.95},
    #   {"document": "Cats are cute animals...", "score": 0.12}
    # ]
    ```

=== "Custom Task"

    ```bash
    # Customer support prioritization
    st rerank "billing issue" --file tickets.txt --task "support ticket prioritization"
    
    # Legal document search with custom seed
    st rerank "contract breach" --file legal_docs.txt \
        --task "legal document retrieval for case research" \
        --seed 789
    ```

### Notes

!!! info "Reranking Model"
    Uses the Qwen3-Reranker-4B model for binary relevance scoring based on yes/no token logits.

!!! tip "Task Descriptions"
    Custom task descriptions help the model understand your specific reranking context:
    - `"support ticket prioritization"` for customer service
    - `"code snippet relevance"` for programming searches
    - `"academic paper retrieval"` for research
    - `"product search ranking"` for e-commerce

---

## cache

Manage result caches.

### Usage

```bash
st cache [OPTIONS]
steadytext cache [OPTIONS]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--clear` | `-c` | Clear all caches |
| `--status` | `-s` | Show cache status |
| `--generation-only` |  | Target only generation cache |
| `--embedding-only` |  | Target only embedding cache |

### Examples

=== "Cache Status"

    ```bash
    st cache --status
    # Generation Cache: 45 entries, 12.3MB
    # Embedding Cache: 128 entries, 34.7MB
    ```

=== "Clear Caches"

    ```bash
    st cache --clear
    # Cleared all caches

    st cache --clear --generation-only
    # Cleared generation cache only
    ```

---

## daemon

Manage the SteadyText daemon for persistent model serving.

### Usage

```bash
st daemon COMMAND [OPTIONS]
steadytext daemon COMMAND [OPTIONS]
```

### Commands

| Command | Description |
|---------|-------------|
| `start` | Start the daemon server |
| `stop` | Stop the daemon server |
| `status` | Check daemon status |
| `restart` | Restart the daemon server |

### Global Daemon Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--seed` | int | `42` | Default seed for daemon operations |

### Options

#### start

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host` | string | `127.0.0.1` | Bind address |
| `--port` | int | `5557` | Port number |
| `--foreground` | flag | `false` | Run in foreground |

#### stop

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--force` | flag | `false` | Force kill if graceful shutdown fails |

#### status

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--json` | flag | `false` | Output as JSON |

### Examples

=== "Start Daemon"

    ```bash
    # Start in background (default)
    st daemon start
    
    # Start in foreground for debugging
    st daemon start --foreground
    
    # Custom host/port
    st daemon start --host 0.0.0.0 --port 5557
    
    # Start with custom default seed
    st daemon start --seed 123
    
    # Combined options
    st daemon start --host 0.0.0.0 --port 5557 --seed 456 --foreground
    ```

=== "Check Status"

    ```bash
    st daemon status
    # Output: Daemon is running (PID: 12345)
    
    # JSON output
    st daemon status --json
    # {"running": true, "pid": 12345, "host": "127.0.0.1", "port": 5557}
    ```

=== "Stop/Restart"

    ```bash
    # Graceful stop
    st daemon stop
    
    # Force stop
    st daemon stop --force
    
    # Restart
    st daemon restart
    ```

### Benefits

- **160x faster first request**: No model loading overhead
- **Persistent cache**: Shared across all operations
- **Automatic fallback**: Operations work without daemon
- **Zero configuration**: Used by default when available

---

## index

Manage FAISS vector indexes for retrieval-augmented generation.

### Usage

```bash
st index COMMAND [OPTIONS]
steadytext index COMMAND [OPTIONS]
```

### Commands

| Command | Description |
|---------|-------------|
| `create` | Create index from text files |
| `search` | Search index for similar chunks |
| `info` | Show index information |

### Global Index Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--seed` | int | `42` | Random seed for embedding generation |

### Options

#### create

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` | path | required | Output index file |
| `--chunk-size` | int | `512` | Chunk size in tokens |
| `--glob` | string | | File glob pattern |

#### search

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--top-k` | int | `5` | Number of results |
| `--threshold` | float | | Similarity threshold |

### Examples

=== "Create Index"

    ```bash
    # From specific files
    st index create doc1.txt doc2.txt --output docs.faiss
    
    # From glob pattern
    st index create --glob "**/*.md" --output project.faiss
    
    # Custom chunk size
    st index create *.txt --output custom.faiss --chunk-size 256
    
    # Reproducible index creation with custom seed
    st index create doc1.txt doc2.txt --output docs_s123.faiss --seed 123
    st index create --glob "**/*.md" --output project_s456.faiss --seed 456
    ```

=== "Search Index"

    ```bash
    # Basic search
    st index search docs.faiss "query text"
    
    # Top 10 results
    st index search docs.faiss "error message" --top-k 10
    
    # With threshold
    st index search docs.faiss "specific term" --threshold 0.8
    
    # Reproducible search with custom seed
    st index search docs.faiss "query text" --seed 789
    st index search docs.faiss "error message" --top-k 10 --seed 123
    ```

=== "Index Info"

    ```bash
    st index info docs.faiss
    # Output:
    # Index: docs.faiss
    # Chunks: 1,234
    # Dimension: 1024
    # Size: 5.2MB
    ```

---

## Advanced Usage

### Environment Variables

Set these before running CLI commands:

```bash
# Cache configuration
export STEADYTEXT_GENERATION_CACHE_CAPACITY=512
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=100

# Allow model downloads (for development)
export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true

# Set default seed for all operations
export STEADYTEXT_DEFAULT_SEED=42

# Then run commands
st generate "test prompt"
st generate "test prompt" --seed 123  # Override default seed
```

### Pipeline Usage

Chain commands with other tools:

```bash
# Batch processing
cat prompts.txt | while read prompt; do
  echo "Prompt: $prompt"
  st generate "$prompt" --json | jq '.text'
  echo "---"
done

# Generate and embed
text=$(st generate "explain AI")
echo "$text" | st embed --format hex > ai_explanation.hex
```

### Scripting Examples

=== "Bash Script"

    ```bash
    #!/bin/bash
    # generate_docs.sh

    prompts=(
      "Explain machine learning"
      "What is deep learning?"
      "Define neural networks"
    )

    for prompt in "${prompts[@]}"; do
      echo "=== $prompt ==="
      st generate "$prompt" --stream
      echo -e "\n---\n"
    done
    ```

=== "Python Integration"

    ```python
    import subprocess
    import json

    def cli_generate(prompt):
        """Use CLI from Python."""
        result = subprocess.run([
            'st', 'generate', prompt, '--json'
        ], capture_output=True, text=True)
        
        return json.loads(result.stdout)

    # Usage
    result = cli_generate("Hello world")
    print(result['text'])
    ```

### Performance Tips

!!! tip "CLI Optimization"
    - **Preload models**: Run `st models --preload` once at startup
    - **Use JSON output**: Easier to parse in scripts with `--json`
    - **Batch operations**: Process multiple items in single session
    - **Cache warmup**: Generate common prompts to populate cache

---

## Real-World Examples

### Content Generation Pipeline

```bash
#!/bin/bash
# blog_generator.sh - Generate blog posts with consistent style

SEED=12345  # Consistent seed for reproducible content

# Function to generate blog post
generate_post() {
    local topic="$1"
    local style="$2"
    
    echo "Generating post about: $topic"
    
    # Generate title
    title=$(st generate "Create an engaging blog title about $topic" --seed $SEED --wait)
    
    # Generate introduction
    intro=$(st generate "Write a compelling introduction for a blog post about $topic" --seed $(($SEED + 1)) --wait)
    
    # Generate main content
    content=$(st generate "Write the main content for a blog post about $topic in a $style style" --seed $(($SEED + 2)) --max-new-tokens 800 --wait)
    
    # Generate conclusion
    conclusion=$(st generate "Write a strong conclusion for a blog post about $topic" --seed $(($SEED + 3)) --wait)
    
    # Combine into final post
    cat <<EOF
# $title

## Introduction
$intro

## Main Content
$content

## Conclusion
$conclusion

---
Generated with SteadyText (seed: $SEED)
EOF
}

# Generate multiple posts
topics=("Machine Learning" "Web Development" "Data Science")
styles=("technical" "beginner-friendly" "professional")

for i in "${!topics[@]}"; do
    generate_post "${topics[$i]}" "${styles[$i]}" > "blog_${i}.md"
    echo "Created blog_${i}.md"
done
```

### Semantic Search CLI Tool

```bash
#!/bin/bash
# semantic_search.sh - Search documents using embeddings

INDEX_FILE="documents.faiss"
SEED=42

# Function to build index
build_index() {
    echo "Building search index..."
    st index create --glob "**/*.md" --output "$INDEX_FILE" --chunk-size 256 --seed $SEED
    echo "Index created: $INDEX_FILE"
}

# Function to search
search_docs() {
    local query="$1"
    local num_results="${2:-5}"
    
    echo "Searching for: $query"
    echo "========================"
    
    # Search and format results
    st index search "$INDEX_FILE" "$query" --top-k $num_results --seed $SEED | \
    while IFS= read -r line; do
        if [[ $line =~ ^([0-9]+)\.\s+(.+):\s+(.+)$ ]]; then
            rank="${BASH_REMATCH[1]}"
            file="${BASH_REMATCH[2]}"
            snippet="${BASH_REMATCH[3]}"
            
            echo -e "\n[$rank] $file"
            echo "   $snippet"
        fi
    done
}

# Main menu
while true; do
    echo -e "\nSemantic Search Tool"
    echo "1. Build/Rebuild index"
    echo "2. Search documents"
    echo "3. Exit"
    read -p "Choose option: " choice
    
    case $choice in
        1) build_index ;;
        2) 
            read -p "Enter search query: " query
            read -p "Number of results (default 5): " num
            search_docs "$query" "${num:-5}"
            ;;
        3) exit 0 ;;
        *) echo "Invalid option" ;;
    esac
done
```

### AI-Powered Code Documentation Generator

```python
#!/usr/bin/env python3
"""
docgen.py - Generate documentation from code using SteadyText CLI
"""

import subprocess
import json
import re
from pathlib import Path
import argparse

def run_steadytext(prompt, seed=42, max_tokens=512):
    """Run SteadyText CLI and return result."""
    cmd = [
        'st', 'generate', prompt,
        '--json',
        '--wait',
        '--seed', str(seed),
        '--max-new-tokens', str(max_tokens)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        data = json.loads(result.stdout)
        return data['text']
    else:
        raise Exception(f"SteadyText error: {result.stderr}")

def extract_functions(code):
    """Extract function definitions from Python code."""
    pattern = r'def\s+(\w+)\s*\([^)]*\):'
    return re.findall(pattern, code)

def generate_function_docs(file_path, seed=42):
    """Generate documentation for a Python file."""
    with open(file_path, 'r') as f:
        code = f.read()
    
    functions = extract_functions(code)
    docs = []
    
    # Generate module overview
    module_prompt = f"Write a brief overview of a Python module containing these functions: {', '.join(functions)}"
    overview = run_steadytext(module_prompt, seed=seed)
    docs.append(f"# {file_path.name}\n\n{overview}\n")
    
    # Generate documentation for each function
    for i, func in enumerate(functions):
        # Extract function code
        func_pattern = rf'(def\s+{func}\s*\([^)]*\):.*?)(?=\ndef|\Z)'
        match = re.search(func_pattern, code, re.DOTALL)
        
        if match:
            func_code = match.group(1)
            
            # Generate documentation
            doc_prompt = f"Write clear documentation for this Python function:\n\n{func_code}"
            func_doc = run_steadytext(doc_prompt, seed=seed + i + 1, max_tokens=300)
            
            docs.append(f"\n## `{func}()`\n\n{func_doc}\n")
    
    return '\n'.join(docs)

def main():
    parser = argparse.ArgumentParser(description='Generate documentation from Python code')
    parser.add_argument('files', nargs='+', help='Python files to document')
    parser.add_argument('--output', '-o', help='Output directory', default='./docs')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    for file_path in args.files:
        file_path = Path(file_path)
        if file_path.suffix == '.py':
            print(f"Generating documentation for {file_path}...")
            
            try:
                docs = generate_function_docs(file_path, seed=args.seed)
                
                # Save documentation
                doc_path = output_dir / f"{file_path.stem}_docs.md"
                with open(doc_path, 'w') as f:
                    f.write(docs)
                
                print(f"  → Saved to {doc_path}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

if __name__ == '__main__':
    main()
```

### Batch Text Analysis Tool

```bash
#!/bin/bash
# analyze_texts.sh - Analyze multiple texts for sentiment, topics, etc.

SEED=999
OUTPUT_DIR="analysis_results"
mkdir -p "$OUTPUT_DIR"

# Function to analyze single text
analyze_text() {
    local file="$1"
    local filename=$(basename "$file" .txt)
    local output_file="$OUTPUT_DIR/${filename}_analysis.json"
    
    echo "Analyzing: $file"
    
    # Read content
    content=$(cat "$file")
    
    # Generate various analyses
    sentiment=$(st generate "Analyze the sentiment of this text and respond with only: POSITIVE, NEGATIVE, or NEUTRAL: $content" --seed $SEED --wait --max-new-tokens 10)
    
    summary=$(st generate "Write a one-sentence summary of: $content" --seed $(($SEED + 1)) --wait --max-new-tokens 50)
    
    topics=$(st generate "List the main topics in this text as comma-separated values: $content" --seed $(($SEED + 2)) --wait --max-new-tokens 30)
    
    # Create embedding
    embedding=$(echo "$content" | st embed --seed $SEED --format json)
    
    # Combine results
    cat > "$output_file" <<EOF
{
  "file": "$file",
  "sentiment": "$sentiment",
  "summary": "$summary",
  "topics": "$topics",
  "embedding_sample": $(echo "$embedding" | jq '.[0:5]'),
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    echo "  → Saved to $output_file"
}

# Process all text files
for file in *.txt; do
    if [ -f "$file" ]; then
        analyze_text "$file"
    fi
done

# Generate summary report
echo -e "\n\nGenerating summary report..."

st generate "Based on these analysis results, write a summary report: $(cat $OUTPUT_DIR/*.json | jq -s '.')" \
    --seed $(($SEED + 100)) \
    --max-new-tokens 500 \
    --wait > "$OUTPUT_DIR/summary_report.md"

echo "Analysis complete! Results in $OUTPUT_DIR/"
```

### Interactive Q&A System

```python
#!/usr/bin/env python3
"""
qa_system.py - Interactive Q&A using SteadyText with context
"""

import subprocess
import json
import readline  # For better input handling
from datetime import datetime

class QASystem:
    def __init__(self, seed=42):
        self.seed = seed
        self.context = []
        self.max_context = 5
        
    def ask(self, question):
        """Ask a question with context."""
        # Build context prompt
        if self.context:
            context_str = "Previous Q&A:\n"
            for qa in self.context[-self.max_context:]:
                context_str += f"Q: {qa['q']}\nA: {qa['a'][:100]}...\n\n"
            full_prompt = f"{context_str}\nNow answer this question: {question}"
        else:
            full_prompt = question
        
        # Generate answer
        cmd = [
            'st', 'generate', full_prompt,
            '--seed', str(self.seed + len(self.context)),
            '--max-new-tokens', '300',
            '--wait',
            '--json'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            answer = data['text']
            
            # Store in context
            self.context.append({
                'q': question,
                'a': answer,
                'timestamp': datetime.now().isoformat()
            })
            
            return answer
        else:
            return f"Error: {result.stderr}"
    
    def save_session(self, filename):
        """Save Q&A session to file."""
        with open(filename, 'w') as f:
            json.dump(self.context, f, indent=2)
        print(f"Session saved to {filename}")
    
    def run_interactive(self):
        """Run interactive Q&A session."""
        print("SteadyText Q&A System")
        print("Type 'quit' to exit, 'save' to save session")
        print("-" * 50)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() == 'quit':
                    break
                elif question.lower() == 'save':
                    filename = input("Save as: ") or "qa_session.json"
                    self.save_session(filename)
                    continue
                elif not question:
                    continue
                
                print("\nThinking...")
                answer = self.ask(question)
                print(f"\nAnswer: {answer}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load', help='Load previous session')
    
    args = parser.parse_args()
    
    qa = QASystem(seed=args.seed)
    
    if args.load:
        with open(args.load, 'r') as f:
            qa.context = json.load(f)
        print(f"Loaded {len(qa.context)} previous Q&As")
    
    qa.run_interactive()
```

### Multi-Language Code Generator

```bash
#!/bin/bash
# polyglot_codegen.sh - Generate code in multiple languages

generate_code() {
    local task="$1"
    local lang="$2"
    local seed="$3"
    
    prompt="Write a $lang function that $task. Include only the code, no explanations."
    
    echo "=== $lang ==="
    st generate "$prompt" --seed $seed --max-new-tokens 200 --wait
    echo -e "\n"
}

# Main
echo "Multi-Language Code Generator"
echo "============================"
read -p "What should the function do? " task

# Generate in multiple languages with consistent seeds
LANGUAGES=("Python" "JavaScript" "Go" "Rust" "Java" "C++" "Ruby" "PHP")
BASE_SEED=1000

for i in "${!LANGUAGES[@]}"; do
    generate_code "$task" "${LANGUAGES[$i]}" $(($BASE_SEED + $i))
done

# Generate comparison
echo "=== Performance Comparison ==="
st generate "Compare the performance characteristics of these languages for $task: ${LANGUAGES[*]}" \
    --seed $(($BASE_SEED + 100)) \
    --max-new-tokens 300 \
    --wait
```

---

## Troubleshooting

### Common Issues

**Issue: Command not found**
```bash
# Problem
$ st generate "test"
bash: st: command not found

# Solution
# Ensure SteadyText is installed
pip install steadytext

# Or add to PATH if using local install
export PATH="$HOME/.local/bin:$PATH"
```

**Issue: Slow first generation**
```bash
# Problem: First call takes 2-3 seconds

# Solution 1: Preload models
st models preload

# Solution 2: Use daemon mode
st daemon start
st generate "test"  # Now fast!
```

**Issue: Different results across runs**
```bash
# Problem: Results vary between sessions

# Solution: Use explicit seeds
st generate "test" --seed 42  # Always same result
st embed "test" --seed 42     # Always same embedding
```

**Issue: JSON parsing errors**
```bash
# Problem: Invalid JSON output

# Solution: Use proper error handling
result=$(st generate "test" --json 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "$result" | jq '.text'
else
    echo "Error generating text"
fi
```

### Best Practices

!!! success "CLI Best Practices"
    1. **Always use seeds** for reproducible results in production
    2. **Start daemon** for better performance in scripts
    3. **Use JSON output** for reliable parsing
    4. **Handle errors** properly in scripts
    5. **Batch operations** when possible
    6. **Set environment variables** for consistent configuration
    7. **Use appropriate output formats** (JSON for parsing, plain for display)
    8. **Chain commands** efficiently with pipes
    9. **Cache warmup** for frequently used prompts
    10. **Monitor performance** with timing commands
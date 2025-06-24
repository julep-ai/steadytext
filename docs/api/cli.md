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
| `--think` | | flag | `false` | Enable Qwen3 thinking mode (shows reasoning) |
| `--eos-string` | `-e` | string | `"[EOS]"` | Custom end-of-sequence string |
| `--size` | | choice | | Model size: small (0.6B), medium (1.7B), large (4B) |
| `--model` | | string | | Model name from registry (e.g., "qwen2.5-3b") |
| `--model-repo` | | string | | Custom model repository |
| `--model-filename` | | string | | Custom model filename |
| `--no-index` | | flag | `false` | Disable automatic index search |
| `--index-file` | | path | | Use specific index file |
| `--top-k` | | int | `3` | Number of context chunks to retrieve |

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

=== "Thinking Mode"

    ```bash
    # Enable Qwen3 thinking mode to show reasoning
    echo "Solve this complex problem" | st --think
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

=== "Using Size Parameter"

    ```bash
    # Fast generation with small model
    st generate "Quick response" --size small
    
    # High quality with large model  
    st generate "Complex analysis" --size large
    ```

=== "Model Selection"

    ```bash
    # Use specific model from registry
    st generate "Technical explanation" --model qwen2.5-3b
    
    # Use custom model
    st generate "Write code" --model-repo Qwen/Qwen2.5-7B-Instruct-GGUF \
        --model-filename qwen2.5-7b-instruct-q8_0.gguf
    ```

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

### Examples

=== "Basic Embedding"

    ```bash
    st embed "machine learning"
    # Outputs JSON array with 1024 float values
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
    ```

### Stdin Support

Embed text from stdin:

```bash
echo "text to embed" | st embed
cat document.txt | st embed --format numpy --output doc_embedding.npy
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
| `--cache-info` |  | Show cache usage information |

### Examples

=== "List Models"

    ```bash
    st models --list
    # Output:
    # Generation Model: Qwen3-1.7B-Q8_0 (1.83GB)
    # Embedding Model: Qwen3-0.6B-Q8_0 (610MB)
    ```

=== "Preload Models"

    ```bash
    st models --preload
    # Downloads and loads all models
    ```

=== "Cache Information"

    ```bash
    st models --cache-dir
    # /home/user/.cache/steadytext/models/

    st models --cache-info
    # Cache directory: /home/user/.cache/steadytext/models/
    # Generation model: 1.83GB (downloaded)
    # Embedding model: 610MB (not downloaded)
    # Total size: 1.83GB / 2.44GB
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

### Examples

=== "Similarity"

    ```bash
    # Cosine similarity
    st vector similarity "cat" "dog"
    # 0.823456
    
    # With JSON output
    st vector similarity "king" "queen" --json
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
    ```

=== "Average"

    ```bash
    # Average embeddings
    st vector average "cat" "dog" "hamster"
    
    # With full embedding output
    st vector average "red" "green" "blue" --json
    ```

=== "Arithmetic"

    ```bash
    # Classic analogy: king + woman - man ≈ queen
    st vector arithmetic "king" "woman" --subtract "man"
    
    # Location arithmetic
    st vector arithmetic "paris" "italy" --subtract "france"
    ```

See [Vector Operations Documentation](vector.md) for detailed usage.

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
    st daemon start --host 0.0.0.0 --port 8080
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
    ```

=== "Search Index"

    ```bash
    # Basic search
    st index search docs.faiss "query text"
    
    # Top 10 results
    st index search docs.faiss "error message" --top-k 10
    
    # With threshold
    st index search docs.faiss "specific term" --threshold 0.8
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

# Then run commands
st generate "test prompt"
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
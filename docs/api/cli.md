# CLI Reference

Complete command-line interface documentation for SteadyText.

## Installation

The CLI is automatically installed with SteadyText:

```bash
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
st generate [OPTIONS] PROMPT
steadytext generate [OPTIONS] PROMPT
```

### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--stream` | `-s` | flag | `false` | Stream output token by token |
| `--json` | `-j` | flag | `false` | Output as JSON with metadata |
| `--logprobs` | `-l` | flag | `false` | Include log probabilities |
| `--eos-string` | `-e` | string | `"[EOS]"` | Custom end-of-sequence string |

### Examples

=== "Basic Generation"

    ```bash
    st generate "Write a Python function to calculate fibonacci"
    ```

=== "Streaming Output"

    ```bash
    st generate "Explain machine learning" --stream
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
    # Generation Model: BitCPM4-1B-Q8_0 (1.3GB)
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
    # Generation model: 1.3GB (downloaded)
    # Embedding model: 610MB (not downloaded)
    # Total size: 1.3GB / 1.9GB
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
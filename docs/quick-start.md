# Python Quick Start

This guide walks you from installation to your first deterministic workflows in Python. If you're targeting SQL-first scenarios, head to the [Postgres Quick Start](postgresql-extension.md).

---

## 1. Prerequisites

- **Python** `>= 3.10` (3.11+ recommended)
- **RAM**: 4 GB minimum (8 GB+ for larger models)
- **Disk**: 5–15 GB for cached models
- **OS**: Linux, macOS, or Windows

---

## 2. Install the SDK

=== "uv (recommended)"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv add steadytext
    ```

=== "pip"
    ```bash
    pip install steadytext
    ```

=== "Poetry"
    ```bash
    poetry add steadytext
    ```

!!! note "First model download"
    On first use SteadyText downloads roughly 5 GB of models to `~/.cache/steadytext/models/` (or `%LOCALAPPDATA%\\steadytext\\steadytext\\models\\` on Windows). Downloads are cached for future runs.

---

## 3. Optional: Start the Daemon

The background daemon keeps models warm and shared between processes. You can skip this step, but your first request will be slower.

```bash
st daemon start
st daemon status  # confirm it's running
```

Shut it down with `st daemon stop` when you're done.

---

## 4. First Deterministic Generation

```python
import steadytext

code = steadytext.generate("Implement binary search in Python")
assert "def binary_search" in code

variant = steadytext.generate(
    "Implement binary search in Python",
    seed=1234,  # choose any integer seed
)
```

- Same prompt + seed ⇒ same output on any machine.
- Different seeds ⇒ controlled, reproducible variations for tests and reviews.

Streaming uses the same deterministic guarantees:

```python
for token in steadytext.generate_iter("Explain dynamic programming", seed=42):
    print(token, end="", flush=True)
```

---

## 5. Deterministic Embeddings

```python
vector = steadytext.embed("Hello world")
print(vector.shape)  # (1024,)

pair = steadytext.embed(["coffee", "espresso"])
```

Embeddings also respect seeds:

```python
vec_a = steadytext.embed("release checklist", seed=100)
vec_b = steadytext.embed("release checklist", seed=100)
assert (vec_a == vec_b).all()
```

Use CLI helpers for quick experiments:

```bash
echo "Explain retrievers" | st --seed 7
echo "incident report" | st embed --seed 99
```

---

## 6. Structured Outputs & Validation

```python
from pydantic import BaseModel
from steadytext import generate

class Ticket(BaseModel):
    title: str
    severity: str

result = generate(
    "Create a sev2 incident summary about retries failing",
    schema=Ticket,
)
print(result)  # deterministic JSON payload
```

Key follow-ups:

- [Structured generation guide](structured-generation.md)
- [Tooling examples](examples/tooling.md)

---

## 7. Reranking & Retrieval Workflows

```python
from steadytext import rerank

docs = ["Reset password steps", "SSL handshake failure", "CORS issue"]
ranked = rerank("TLS handshake failed", docs)
```

Combine embeddings, indexes, and reranking to build full retrieval pipelines. See:

- [Vector indexing](vector-indexing.md)
- [Reranking overview](reranking.md)
- [Log analysis example](examples/log-analysis.md)

---

## 8. Keep Going

- Explore end-to-end scenarios in the [Python tutorials](examples/index.md#python-library).
- Configure caching, seeds, and fallbacks via [Configuration Reference](configuration-reference.md).
- Need SQL parity? Jump to the [Postgres Extension Journey](postgresql-extension.md#postgres-quick-start).

Let us know what you build—anchor your questions with `AIDEV-ANCHOR:` references for faster feedback.
variations = []

for i, style_seed in enumerate([300, 400, 500], 1):
    variation = steadytext.generate(base_prompt, seed=style_seed)
    variations.append(f"Version {i}: {variation}")
    
for variation in variations:
    print(variation[:80] + "...\n")
```

## PostgreSQL Integration

SteadyText now includes a PostgreSQL extension:

```bash
# Install the PostgreSQL extension
git clone https://github.com/julep-ai/steadytext.git
cd steadytext/pg_steadytext
make && sudo make install

# Enable in PostgreSQL
psql -c "CREATE EXTENSION pg_steadytext CASCADE;"
```

```sql
-- Use in SQL queries
SELECT steadytext_generate('Write a product description', max_tokens := 200, seed := 123);

-- Generate embeddings
SELECT steadytext_embed('machine learning', seed := 456);

-- Semantic search with pgvector
SELECT title, content <-> steadytext_embed('AI technology') AS distance
FROM documents
ORDER BY distance
LIMIT 5;
```

## Next Steps

- **[API Reference](api/index.md)** - Complete function documentation with seed parameters
- **[Custom Seeds Guide](examples/custom-seeds.md)** - Comprehensive seed usage examples
- **[PostgreSQL Integration](postgresql-extension.md)** - Complete PostgreSQL extension guide
- **[CLI Reference](api/cli.md)** - Command-line interface with `--seed` flag details
- **[Examples](examples/index.md)** - Real-world usage patterns

## Common Issues and Solutions

### Model Loading Errors
If you see "Failed to load model":
```bash
# Use fallback models
export STEADYTEXT_USE_FALLBACK_MODEL=true

# Or clear model cache and re-download
rm -rf ~/.cache/steadytext/models/
```

### llama-cpp-python Build Issues
If installation fails with llama-cpp-python errors:
```bash
# Set required build flags
export FORCE_CMAKE=1
export CMAKE_ARGS="-DLLAVA_BUILD=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF"

# Then reinstall
pip install --force-reinstall steadytext
```

### Daemon Connection Issues
```bash
# Check if daemon is running
st daemon status

# Start daemon if not running
st daemon start

# Or disable daemon and use direct loading
export STEADYTEXT_DISABLE_DAEMON=1
```

## Need Help?

- **Issues**: [GitHub Issues](https://github.com/julep-ai/steadytext/issues)
- **Discussions**: [GitHub Discussions](https://github.com/julep-ai/steadytext/discussions)
- **Documentation**: [Full Documentation](https://github.com/julep-ai/steadytext/tree/main/docs)

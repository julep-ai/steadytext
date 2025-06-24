# SteadyText Examples

This directory contains example code demonstrating various use cases for SteadyText.

## Examples

### basic_usage.py
Core functionality demonstration:
- Text generation with and without logprobs
- Streaming generation
- Creating embeddings

### testing_with_ai.py
Using SteadyText for testing:
- Deterministic test assertions
- Mock AI services
- Test fixture generation
- Fuzz testing with reproducible inputs

### cli_tools.py
Building command-line tools:
- Motivational quotes
- Error message explanations
- Git command generation
- Click-based CLI examples

### content_generation.py
Content and data generation:
- ASCII art
- Game NPC dialogue
- Product reviews and user bios
- Auto-documentation
- Story generation
- Semantic cache keys

### vector_operations.py
Vector operations on embeddings:
- Cosine similarity between texts
- Distance calculations (euclidean, manhattan, cosine)
- Similarity search across multiple files
- Embedding averaging
- Vector arithmetic (king - man + woman)
- Stdin input support

### index_management.py
FAISS index creation and search:
- Creating indices from text documents
- Searching for similar chunks
- Context-enhanced generation (RAG)
- Index information and statistics
- Default index usage
- Deterministic document retrieval

## Running the Examples

### Using UV (Recommended)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run examples with UV
uv run python examples/basic_usage.py
uv run python examples/testing_with_ai.py
uv run python examples/cli_tools.py
uv run python examples/content_generation.py
uv run python examples/vector_operations.py
uv run python examples/index_management.py
```

### Legacy Method

```bash
python examples/basic_usage.py
python examples/testing_with_ai.py
python examples/cli_tools.py
python examples/content_generation.py
python examples/vector_operations.py
python examples/index_management.py
```

The CLI tools example also supports command-line arguments:

```bash
# Using UV
uv run python examples/cli_tools.py quote
uv run python examples/cli_tools.py error ECONNREFUSED
uv run python examples/cli_tools.py git "undo last commit"

# Legacy method
python examples/cli_tools.py quote
python examples/cli_tools.py error ECONNREFUSED
python examples/cli_tools.py git "undo last commit"
```

## Daemon Mode Examples

SteadyText includes a daemon mode for faster responses:

```bash
# Start the daemon
st daemon start

# All generation now uses the daemon automatically
echo "Write a function" | st

# Python code also uses daemon when available
uv run python examples/basic_usage.py  # Uses daemon automatically

# Check daemon status
st daemon status

# Stop daemon when done
st daemon stop
```

### Daemon in Python Code

```python
import steadytext
from steadytext.daemon import use_daemon

# Daemon is used automatically when available
text = steadytext.generate("Hello world")  # Fast if daemon running

# Explicitly use daemon for a scope
with use_daemon():
    text = steadytext.generate("Complex prompt")
    embedding = steadytext.embed("Some text")
    # All operations in this block use the daemon
```

## Note

All examples use deterministic generation, so running them multiple times will produce identical outputs. This is the core feature of SteadyText - predictable, reproducible AI outputs.
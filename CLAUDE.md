# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Use Anchor comments

Add specially formatted comments throughout the codebase, where appropriate, for yourself as inline knowledge that can be easily `grep`ped for.

- Use `AIDEV-NOTE:`, `AIDEV-TODO:`, `AIDEV-QUESTION:`, or `AIDEV-SECTION:` as prefix as appropriate.

- *Important:* Before scanning files, always first try to grep for existing `AIDEV-…`.

- Update relevant anchors, after finishing any task.

- Make sure to add relevant anchor comments, whenever a file or piece of code is:

  * too complex, or
  * very important, or
  * could have a bug

## Daemon Architecture

SteadyText includes a daemon mode for persistent model serving via ZeroMQ. This avoids the overhead of repeated model loading.

### Key Components

**daemon/protocol.py**
- Defines JSON-based request/response protocol
- Request types: generate, generate_iter, embed, ping, shutdown
- Uses unique IDs for request-response matching

**daemon/server.py**
- ZeroMQ REP server that keeps models loaded in memory
- Handles multiple request types with error recovery
- Supports streaming generation with acknowledgment protocol
- Graceful shutdown via signals (SIGINT, SIGTERM)

**daemon/client.py**
- Client wrapper with automatic fallback to direct model loading
- `use_daemon()` context manager for scoped daemon usage
- Environment variable support (STEADYTEXT_DISABLE_DAEMON to opt-out)

### Usage Patterns

**CLI Commands:**
```bash
# Start daemon
st daemon start [--host HOST] [--port PORT] [--foreground]

# Check status
st daemon status [--json]

# Stop daemon
st daemon stop [--force]

# Restart daemon
st daemon restart
```

**SDK Usage:**
```python
# Daemon is used by default - no setup needed
text = generate("Hello world")  # Uses daemon automatically
embedding = embed("Some text")   # Uses daemon automatically

# Use daemon for all operations in a scope (ensures connection)
with use_daemon():
    text = generate("Hello world")
    embedding = embed("Some text")

# Disable daemon globally via environment
os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
```

### AIDEV Considerations

- AIDEV-NOTE: The daemon maintains singleton model instances to avoid memory duplication.
- AIDEV-NOTE: Streaming uses a special acknowledgment protocol for flow control.
- AIDEV-NOTE: All operations gracefully fall back to direct model loading on connection failure.
- AIDEV-NOTE: The centralized cache system ensures consistent caching between the daemon and direct access.
- AIDEV-NOTE: The cache is shared via an SQLite backend for thread/process safety.
- AIDEV-NOTE: The daemon server is now fully integrated with the centralized cache manager (v1.3.3+).
- AIDEV-NOTE: Cache hits in daemon mode return identical results to direct access mode.
- AIDEV-NOTE: Streaming generation simulates streaming from cached results for consistency.
- AIDEV-NOTE: Streaming now populates the cache after completion in both daemon and direct modes (v1.3.3+).
- AIDEV-NOTE: Cached streaming results may have normalized whitespace when simulated from the cache.
- AIDEV-NOTE: Logprobs requests bypass the cache in both daemon and direct modes.
- AIDEV-NOTE: Custom eos_string parameters are included in cache keys to prevent collisions.
- AIDEV-NOTE: The CLI cache fix (v1.3.3+) uses the main API layer (generate, generate_iter) to enable full cache functionality.
- AIDEV-TODO: Consider adding connection pooling for high-concurrency scenarios.
- AIDEV-TODO: Add metrics/monitoring endpoints for production deployments.
- AIDEV-QUESTION: Should we support multiple daemon instances for load balancing?

## Gemma-3n Models

SteadyText v2.0+ uses Gemma-3n models for generation and Qwen3 for embeddings.

### AIDEV Considerations

- AIDEV-NOTE: The default generation model is `gemma-3n-E2B-it-GGUF`.
- AIDEV-NOTE: The default embedding model is `Qwen3-Embedding-0.6B-GGUF`.

## Reranking Support (v1.3.0+)

SteadyText v1.3.0+ includes document reranking functionality using the Qwen3-Reranker-4B model.

### Key Components

**core/reranker.py**
- DeterministicReranker class following the same patterns as generation/embedding
- Uses yes/no token logits for binary relevance scoring
- Fallback to simple word overlap scoring when model unavailable

**Reranking Features:**
- Query-document relevance scoring
- Batch document reranking
- Custom task descriptions for domain-specific reranking
- Caching support via dedicated reranking cache
- CLI command: `st rerank`
- PostgreSQL functions: `steadytext_rerank()` and async variants

### AIDEV Considerations

- AIDEV-NOTE: The default reranking model is `Qwen3-Reranker-4B-GGUF`.
- AIDEV-NOTE: Reranking uses a specific prompt format with system/user/assistant tags.
- AIDEV-NOTE: Scores are derived from yes/no token probabilities.
- AIDEV-TODO: Consider adding support for cross-encoder models.
- AIDEV-TODO: Add streaming support for large document sets.

## Cache Management

SteadyText v1.3+ uses a centralized cache management system with pluggable backends (v2.2.0+).

### Key Components

**cache_manager.py**
- Centralized cache management with singleton pattern
- Support for multiple cache backends via factory pattern
- Thread-safe and process-safe access across all components

**Cache Backends (v2.2.0+):**
- **SQLite** (default): Thread-safe local storage with WAL mode
- **D1**: Cloudflare's distributed SQLite for edge deployments
- **Memory**: In-memory cache for testing/ephemeral workloads

### Cache Backend Selection

```bash
# Select cache backend
export STEADYTEXT_CACHE_BACKEND=sqlite  # Default
export STEADYTEXT_CACHE_BACKEND=d1      # Cloudflare D1
export STEADYTEXT_CACHE_BACKEND=memory  # In-memory

# D1-specific configuration
export STEADYTEXT_D1_API_URL=https://your-worker.workers.dev
export STEADYTEXT_D1_API_KEY=your-api-key
export STEADYTEXT_D1_BATCH_SIZE=50
```

### Cache Configuration

Environment variables affect all backends:

**Generation Cache:**
- `STEADYTEXT_GENERATION_CACHE_CAPACITY` (default: 256)
- `STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB` (default: 50.0)

**Embedding Cache:**
- `STEADYTEXT_EMBEDDING_CACHE_CAPACITY` (default: 512)
- `STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB` (default: 100.0)

### Cache Usage

```python
from steadytext import get_cache_manager

# Get cache statistics
cache_manager = get_cache_manager()
stats = cache_manager.get_cache_stats()
print(f"Generation cache size: {stats['generation']['size']}")

# Clear all caches (for testing)
cache_manager.clear_all_caches()

# Programmatic backend selection
from steadytext.disk_backed_frecency_cache import DiskBackedFrecencyCache
cache = DiskBackedFrecencyCache(backend_type="d1", api_url="...", api_key="...")
```

AIDEV-NOTE: The cache backend system uses a factory pattern in `cache/factory.py`
AIDEV-NOTE: D1 backend requires a proxy Worker due to Cloudflare access restrictions
AIDEV-NOTE: All backends implement the same CacheBackend interface for consistency
AIDEV-TODO: Consider adding Redis backend for traditional distributed caching

## AI Assistant Workflow: Step-by-Step Methodology

When responding to user instructions, the AI assistant (Claude, Cursor, GPT, etc.) should follow this process to ensure clarity, correctness, and maintainability:

1. **Consult Relevant Guidance**: When the user gives an instruction, consult the relevant instructions from `CLAUDE.md` files (both root and directory-specific) for the request.
2. **Clarify Ambiguities**: Based on what you could gather, see if there's any need for clarifications. If so, ask the user targeted questions before proceeding.
3. **Break Down & Plan**: Break down the task at hand and chalk out a rough plan for carrying it out, referencing project conventions and best practices.
4. **Trivial Tasks**: If the plan/request is trivial, go ahead and get started immediately.
5. **Non-Trivial Tasks**: Otherwise, present the plan to the user for review and iterate based on their feedback.
6. **Track Progress**: Use a to-do list (internally, or optionally in a `TODOS.md` file) to keep track of your progress on multi-step or complex tasks.
7. **If Stuck, Re-plan**: If you get stuck or blocked, return to step 3 to re-evaluate and adjust your plan.
8. **Update Documentation**: Once the user's request is fulfilled, update relevant anchor comments (`AIDEV-NOTE`, etc.) and `CLAUDE.md` files in the files and directories you touched.
9. **User Review**: After completing the task, ask the user to review what you've done, and repeat the process as needed.
10. **Session Boundaries**: If the user's request isn't directly related to the current context and can be safely started in a fresh session, suggest starting from scratch to avoid context confusion.


## Deterministic Fallback Removal (v2.1.0+)

AIDEV-NOTE: The deterministic fallback generator has been disabled in v2.1.0+. When models are unavailable or errors occur, functions now return `None` instead of generating deterministic but meaningless text. This change was made because the fallback was causing more confusion than it was solving.

**Key Changes:**
- `generate()` returns `None` when model is not loaded or on invalid input
- `generate_iter()` returns an empty iterator when model is not loaded
- `embed()` returns `None` instead of zero vectors when model is not loaded
- PostgreSQL extension returns NULL on errors instead of fallback text
- Tests updated to expect `None` instead of fallback outputs

The original fallback functions (`_deterministic_fallback_generate` and `_deterministic_fallback_generate_iter`) are preserved but marked as DEPRECATED.

## pytest Collection Hanging Fix

AIDEV-NOTE: Fixed in v2.0.1+ - pytest collection was hanging due to module-level code execution. The fixes include removing module-level execution, adding environment checks for model downloads, lazy cache initialization, and early environment setup in conftest.py.

## Context Window Management (v2.3.0+)

AIDEV-NOTE: SteadyText now dynamically manages context windows to maximize available context while preventing errors.

### Key Features

**Dynamic Context Window Sizing:**
- Automatically uses the largest context window supported by each model
- Can be overridden via `STEADYTEXT_MAX_CONTEXT_WINDOW` environment variable
- Known model limits are hardcoded for safety (e.g., Qwen2.5-3B: 32768 tokens)

**Input Length Validation:**
- Validates input length before generation to prevent mid-generation failures
- Raises `ContextLengthExceededError` with detailed token counts
- Reserves space for output tokens (default: 512) plus 10% safety margin
- Uses model's tokenizer for accurate counting, falls back to estimation

**Deterministic Behavior:**
- Output remains identical regardless of context window size
- The same input produces the same output whether n_ctx=2048 or n_ctx=32768
- Only the maximum processable input length changes

### Usage

```python
# Context window is set automatically
text = generate("Your prompt here")  # Uses optimal context for loaded model

# Override context window size
os.environ["STEADYTEXT_MAX_CONTEXT_WINDOW"] = "8192"
text = generate("Your prompt here")  # Limited to 8192 tokens

# Handle long inputs gracefully
try:
    text = generate(very_long_prompt)
except ContextLengthExceededError as e:
    print(f"Input too long: {e.input_tokens} tokens, max: {e.max_tokens}")
```

AIDEV-NOTE: The context window affects only input capacity, not output quality or consistency
AIDEV-TODO: Add automatic input truncation option for oversized inputs
AIDEV-TODO: Support for sliding window or chunking for very long documents

## Structured Generation (v2.4.0+)

### Native Grammar Support

AIDEV-NOTE: As of v2.4.0, SteadyText uses llama.cpp's native GBNF grammar support instead of Outlines. This resolves compatibility issues with Gemma-3n and other models that had vocabulary processing errors.

**Implementation Details:**
- `core/grammar.py`: Converts JSON schemas, regex patterns, and choices to GBNF grammars
- `core/structured.py`: Uses llama-cpp-python's `grammar` parameter for constrained generation
- Outlines dependency has been removed from the project

**Grammar Conversion Features:**
- JSON schemas (including Pydantic models) → GBNF
- Simple regex patterns → GBNF (complex patterns fall back to generic string)
- Choice lists → GBNF
- Python types (int, float, bool, str) → GBNF

AIDEV-NOTE: The grammar-based approach is fully compatible with all models supported by llama.cpp
AIDEV-TODO: Expand regex to GBNF conversion for more complex patterns
AIDEV-TODO: Add support for recursive JSON schemas

### Feature Overview

SteadyText supports structured text generation using llama.cpp grammars, enabling:
- JSON generation with schemas or Pydantic models
- Regex pattern matching for formatted output
- Choice constraints for multiple-choice selection
- Type constraints for basic Python types (int, float, bool, str)

### Usage Examples

```python
from steadytext import generate, generate_json, generate_regex, generate_choice
from pydantic import BaseModel

# JSON with Pydantic model
class Person(BaseModel):
    name: str
    age: int

result = generate("Create a person", schema=Person)
# Returns: "Let me create a person...<json-output>{"name": "Alice", "age": 30}</json-output>"

# Regex pattern matching
phone = generate("My number is", regex=r"\d{3}-\d{3}-\d{4}")
# Returns: "555-123-4567"

# Choice constraints
answer = generate("Is Python good?", choices=["yes", "no", "maybe"])
# Returns: "yes"

# JSON schema
schema = {"type": "object", "properties": {"color": {"type": "string"}}}
result = generate_json("Pick a color", schema)
```

### Technical Implementation

- AIDEV-NOTE: Structured generation uses a two-phase approach:
  1. Generate thoughts/reasoning up to `<json-` tag
  2. Use llama.cpp grammars to generate constrained output after `<json-output>`
- AIDEV-NOTE: All structured outputs are deterministic and cacheable
- AIDEV-NOTE: Structured generation bypasses cache for logprobs requests
- AIDEV-NOTE: The daemon protocol supports grammar parameters natively
- AIDEV-NOTE: Grammar conversion happens on-the-fly for each request
- AIDEV-TODO: Add streaming support for structured generation
- AIDEV-TODO: Cache compiled grammars for frequently used schemas

## Development Commands

### Testing

**Using UV (recommended):**
```bash
# Run all tests with UV
uv run python -m pytest

# Run tests with coverage
uv run python -m pytest --cov=steadytext --cov-report=xml

# Run specific test files
uv run python -m pytest tests/test_steadytext.py
uv run python -m pytest test_gen.py
uv run python -m pytest test_fallback_gen.py

# Allow model downloads in tests (models are downloaded on first use)
STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true uv run python -m pytest

# Configure cache settings
STEADYTEXT_GENERATION_CACHE_CAPACITY=512 uv run python -m pytest
STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=100.0 uv run python -m pytest
STEADYTEXT_EMBEDDING_CACHE_CAPACITY=1024 uv run python -m pytest
STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=200.0 uv run python -m pytest

# Alternative: run pytest as tool (if not project dependency)
uvx pytest
```

**Legacy method:**
```bash
# Run all tests with plain Python
python -m pytest
```

All tests are designed to pass even if models cannot be downloaded. Model-dependent tests are automatically skipped unless `STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true` is set.

### Linting and Formatting

**Using UV (recommended):**
```bash
# Run tools without installing them in project environment
uvx ruff check .
uvx ruff format .
uvx black .
uvx mypy .

# Install and run pre-commit hooks
uvx pre-commit install
uvx pre-commit run --all-files

# If tools are added as dev dependencies
uv add --dev ruff black mypy pre-commit
uv run ruff check .
uv run black .
```

**Legacy methods:**
```bash
# Check code quality with flake8
python -m flake8 .

# Using poethepoet tasks (if available)
poe lint
poe format
poe check
```

AIDEV-NOTE: UV's tool runner (uvx) allows using linting tools without polluting project dependencies

### Index Management
```bash
# Create FAISS index from text files
st index create document1.txt document2.txt --output my_index.faiss
st index create *.txt --output project.faiss --chunk-size 256

# View index information
st index info my_index.faiss

# Search index
st index search my_index.faiss "query text" --top-k 5

# Use index with generation (automatic with default.faiss)
echo "What is Python?" | st --index-file my_index.faiss
echo "explain this error" | st --no-index  # Disable index search
```

AIDEV-NOTE: The index functionality uses chonkie for deterministic text chunking, faiss-cpu for vector storage, automatic context retrieval when default.faiss exists, and aggressive caching of search results for determinism.

### Installation

**Preferred method using UV (recommended):**
```bash
# Install in development mode with UV
uv pip install -e .

# Or if project uses pyproject.toml with UV
uv sync --all-extras --dev

# Build package with UV
uv build
```

**Alternative method using pip:**
```bash
# Install in development mode with pip (legacy)
python -m pip install -e .
```

AIDEV-NOTE: Always prefer UV for new development - it's faster and handles virtual environments automatically

## Architecture Overview

SteadyText provides deterministic text generation and embedding with zero configuration. The core principle is "Never Fails" - all functions return deterministic outputs even when models can't be loaded.

### Key Components

**Core Layer (`steadytext/core/`)**
- `generator.py`: Text generation with `DeterministicGenerator` class and deterministic fallback function
- `embedder.py`: Embedding creation with L2 normalization and deterministic fallback to zero vectors

**Models Layer (`steadytext/models/`)**
- `cache.py`: Downloads and caches GGUF models from Hugging Face
- `loader.py`: Singleton model loading with thread-safe caching via `_ModelInstanceCache`

**Configuration (`steadytext/utils.py`)**
- Model configurations for llama-cpp-python
- Deterministic environment setup (seeds, PYTHONHASHSEED)
- Cache directory management across platforms

### Deterministic Design

**Text Generation:**
- Uses Gemma-3n with deterministic sampling parameters
- Fallback generates text using hash-based word selection when model unavailable
- Always returns strings, never raises exceptions
- Supports both batch generation (`generate()`) and streaming generation (`generate_iter()`)

**Embeddings:**
- Uses Qwen3-Embedding-0.6B
- Always returns 1024-dimensional L2-normalized float32 numpy arrays
- Fallback returns zero vectors when model unavailable

**Model Loading:**
- Models auto-download to platform-specific cache directories on first use
- Thread-safe singleton pattern prevents multiple model instances
- Graceful degradation when models can't be loaded

### Testing Strategy

The test suite in `tests/test_steadytext.py` covers:
- API determinism across multiple calls
- Graceful error handling and fallback behavior
- Edge cases (empty inputs, invalid types)
- Model-dependent tests (skipped if models unavailable)

Two standalone test files (`test_gen.py`, `test_fallback_gen.py`) provide direct testing of core components.

## Important Constants

- `DEFAULT_SEED = 42`: Used throughout for deterministic behavior
- `GENERATION_MAX_NEW_TOKENS = 512`: Fixed output length for text generation
- `EMBEDDING_DIMENSION = 1024`: Fixed embedding dimensionality
- Models are cached to `~/.cache/steadytext/models/` (Linux/Mac) or `%LOCALAPPDATA%\steadytext\steadytext\models\` (Windows)

## CLI Architecture

SteadyText includes a command-line interface built with Click:

**Main CLI (`steadytext/cli/main.py`)**
- Entry point for both `steadytext` and `st` commands
- Supports stdin pipe input when no subcommand provided
- Version flag support
- Quiet by default with `--verbose` option for informational output

**Commands (`steadytext/cli/commands/`)**
- `generate.py`: Text generation with streaming by default, JSON output, and logprobs support
- `embed.py`: Embedding creation with multiple output formats (JSON, numpy, hex)
- `cache.py`: Cache management and status commands
- `models.py`: Model management (list, preload, etc.)
- `completion.py`: Shell completion script generation for bash/zsh/fish

**CLI Features:**
- Deterministic outputs matching the Python API
- Multiple output formats (raw text, JSON with metadata, structured data)
- Streaming by default for real-time text generation (use `--wait` to disable)
- Quiet by default (use `--verbose` to enable informational output)
- Stdin/pipe support for unix-style command chaining
- Log probability access for advanced use cases
- Shell completion support for all commands and options

## Shell Integration and ZSH Plugins

SteadyText provides advanced shell integration through ZSH plugins that enable AI-powered command suggestions.

### Shell Completions
Basic tab completion for commands and options:
```bash
# Install completions
st completion --install

# Manual installation for specific shell
st completion --shell zsh
```

### ZSH Plugins (`steadytext/cli/zsh-plugin/`)

**Context-Aware Suggestions (`steadytext-context.plugin.zsh`)**
- AIDEV-NOTE: Gathers shell context (pwd, git, env, history) for AI suggestions
- Triggered with `Ctrl-X Ctrl-S` by default
- Configurable via environment variables
- Non-intrusive manual activation

**Autosuggestions (`steadytext-autosuggestions.zsh`)**
- AIDEV-NOTE: Fish-like autosuggestions powered by SteadyText
- Shows AI predictions as you type in gray text
- Async processing to avoid blocking
- Suggestion caching for performance
- Integrates with zsh-autosuggestions if available
- Multiple strategies: context, history, or mixed

**Installation:**
```bash
# Quick install (interactive installer)
bash /path/to/steadytext/cli/zsh-plugin/install.sh

# Manual oh-my-zsh
plugins=(... steadytext-context steadytext-autosuggestions)

# Manual standalone
source /path/to/steadytext/cli/zsh-plugin/steadytext-context.plugin.zsh
source /path/to/steadytext/cli/zsh-plugin/steadytext-autosuggestions.zsh
```

**Configuration:**
```bash
export STEADYTEXT_SUGGEST_ENABLED=1          # Enable/disable
export STEADYTEXT_SUGGEST_MODEL_SIZE=small   # Model size
export STEADYTEXT_SUGGEST_STRATEGY=context   # Suggestion strategy
export STEADYTEXT_SUGGEST_ASYNC=1            # Async mode
```

AIDEV-NOTE: The ZSH plugins send minimal context to preserve privacy
AIDEV-TODO: Consider adding bash/fish equivalents of the context-aware plugins
AIDEV-TODO: Add telemetry for suggestion acceptance rates (opt-in)
AIDEV-QUESTION: Should we support project-specific contexts via .steadytext files?

## Cache Configuration

SteadyText uses disk-backed frecency caches for both generation and embedding results. The caches can be configured via environment variables:

**Generation Cache:**
- `STEADYTEXT_GENERATION_CACHE_CAPACITY`: Maximum number of cache entries (default: 256)
- `STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB`: Maximum cache file size in MB (default: 50.0)

**Embedding Cache:**
- `STEADYTEXT_EMBEDDING_CACHE_CAPACITY`: Maximum number of cache entries (default: 512)
- `STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB`: Maximum cache file size in MB (default: 100.0)

Cache files are stored in:
- `~/.cache/steadytext/caches/` (Linux/Mac)
- `%LOCALAPPDATA%\steadytext\steadytext\caches\` (Windows)

## Todos Directory

The `todos/` directory contains task descriptions and implementation notes for features that are planned or in progress. These are typically detailed technical specifications or design documents that outline how specific features should be implemented.

When working on features described in `todos/`:
1. Read the relevant todo file thoroughly before implementation
2. Follow the technical specifications and design decisions outlined
3. Move or archive completed todo files once implemented
4. Update todo files if implementation details change during development

## Benchmarking

The `benchmarks/` directory contains comprehensive speed and accuracy benchmarks:

### Running Benchmarks

**Using UV (recommended):**
```bash
# Run all benchmarks
uv run python benchmarks/run_all_benchmarks.py

# Quick benchmarks for CI
uv run python benchmarks/run_all_benchmarks.py --quick

# Test benchmarks are working
uv run python benchmarks/test_benchmarks.py
```

**Legacy method:**
```bash
# Run with plain Python
python benchmarks/run_all_benchmarks.py
```

### Key Metrics
- **Speed**: Generation/embedding throughput, latency percentiles, memory usage
- **Accuracy**: Determinism verification, quality checks, LightEval standard benchmarks

AIDEV-NOTE: When modifying core generation/embedding code, always run benchmarks to check for performance regressions

## UV Package Manager

UV is a modern, blazing-fast Python package and project manager written in Rust. It serves as a drop-in replacement for pip, virtualenv, poetry, and other Python tooling, offering 10-100x speed improvements.

AIDEV-NOTE: UV is now the preferred package manager for SteadyText development. It automatically handles virtual environments, avoids activation/deactivation issues, and provides superior dependency resolution.

### Key Benefits

- **Speed**: 10-100x faster than pip for package installation and dependency resolution
- **Automatic Virtual Environments**: Creates and manages `.venv` automatically when needed
- **No Activation Required**: Commands work without manual virtual environment activation
- **Superior Dependency Resolution**: Modern resolver prevents version conflicts
- **Unified Tooling**: Replaces multiple tools (pip, virtualenv, poetry, pyenv) with one
- **Drop-in Compatibility**: Works with existing requirements.txt and pyproject.toml files

### Installation

Install UV system-wide using the official installer:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell (run as administrator)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: using pip (not recommended)
pip install uv
```

### Basic Usage

**Project Initialization:**
```bash
# Initialize new project with pyproject.toml
uv init steadytext-project
cd steadytext-project

# Initialize in existing directory
uv init .
```

**Virtual Environment Management:**
```bash
# Create virtual environment (done automatically with uv add)
uv venv

# Create with specific Python version
uv venv --python 3.11

# UV automatically finds and uses .venv when present - no activation needed!
```

**Package Management:**
```bash
# Add dependencies (creates .venv automatically if needed)
uv add requests numpy pandas

# Add development dependencies
uv add --dev pytest black ruff

# Add optional dependencies
uv add --optional test pytest coverage

# Remove dependencies
uv remove requests

# Install from requirements.txt
uv pip install -r requirements.txt

# Install project in development mode
uv pip install -e .

# Sync dependencies from lock file
uv sync
```

**Running Code:**
```bash
# Run Python scripts (automatically uses project's .venv)
uv run python script.py
uv run pytest
uv run python -m pytest

# Run tools without installing in project
uv tool run black .
uv tool run ruff check .

# Short alias for tool run
uvx black .
uvx ruff check .
```

### Python Version Management

```bash
# Install Python versions
uv python install 3.10 3.11 3.12

# List available Python versions
uv python list

# Use specific Python version for project
uv python pin 3.11

# Create venv with specific Python version
uv venv --python 3.11
```

### Advanced Features

**Lock Files and Reproducibility:**
```bash
# Generate lock file (done automatically with uv add)
uv lock

# Export to requirements.txt format
uv export -o requirements.txt

# Install from lock file
uv sync
```

**Development Workflow:**
```bash
# Install project with all development dependencies
uv sync --all-extras --dev

# Update dependencies
uv lock --upgrade

# Check for dependency conflicts
uv tree
```

### Migration from pip/virtualenv

Replace common commands:
```bash
# Old way                          # New way
python -m venv .venv              # uv venv (automatic)
source .venv/bin/activate         # (not needed)
pip install requests              # uv add requests
pip install -r requirements.txt  # uv pip install -r requirements.txt
pip freeze > requirements.txt    # uv export -o requirements.txt
deactivate                        # (not needed)
```

### Common Patterns for SteadyText Development

**Setting up development environment:**
```bash
# Clone and setup
git clone <repo>
cd steadytext
uv sync --all-extras --dev

# Run tests
uv run python -m pytest

# Run linting
uvx ruff check .
uvx black .

# Install in development mode
uv pip install -e .
```

**Working with dependencies:**
```bash
# Add ML libraries commonly used
uv add numpy torch transformers

# Add development tools
uv add --dev pytest ruff black mypy

# Check installed packages
uv pip list

# Show dependency tree
uv tree
```

### Troubleshooting

**Common Issues:**
- If UV can't find Python version, install it: `uv python install 3.11`
- For permission errors on Linux/Mac: `sudo chown -R $USER ~/.local/share/uv`
- To force recreation of virtual environment: `rm -rf .venv && uv sync`

**Cache Management:**
```bash
# Show cache info
uv cache info

# Clean cache
uv cache clean
```

AIDEV-TODO: Consider adding UV-specific CI/CD configurations for faster builds
AIDEV-NOTE: UV's automatic virtual environment management eliminates common "forgot to activate venv" issues

## Dependency Management and Optional Dependencies

AIDEV-NOTE: The project includes optional dependencies that may pull in large packages like torch/nvidia:
- `lighteval` (in benchmark extras) depends on `accelerate` → `torch` → nvidia CUDA packages
- UV correctly includes all dependencies (including optional) in `uv.lock` for reproducibility
- These packages are NOT installed during regular installation unless explicitly requested
- Use `uv sync` for minimal installation (no extras)
- Use `uv sync --extra benchmark` only when running benchmarks
- The code gracefully handles missing optional dependencies (e.g., lighteval)

## PostgreSQL Extension (pg_steadytext)

### Known Issues and Fixes

**Python Path Configuration Error (Fixed)**
- AIDEV-NOTE: Fixed SQL syntax error when setting plpython3.python_path in pg_steadytext--1.0.0.sql
- Issue: `ALTER DATABASE ... SET plpython3.python_path TO NULL` causes syntax error during Docker initialization
- Root cause: `current_setting('plpython3.python_path', true)` returns NULL when setting doesn't exist, causing invalid SQL
- Fix: Properly handle NULL values by checking if current_path exists before concatenation
- The fixed code uses a DECLARE block with proper exception handling to avoid NULL concatenation

### Docker Development

**Building and Running:**
```bash
cd pg_steadytext
docker build -t pg_steadytext .
docker run -d -p 5432:5432 --name pg_steadytext pg_steadytext
```

**Testing the Extension:**
```bash
docker exec -it pg_steadytext psql -U postgres -c "SELECT steadytext_generate('Hello Docker!');"
docker exec -it pg_steadytext psql -U postgres -c "SELECT steadytext_version();"
```

## PostgreSQL Extension Structured Output (v2.4.0+)

AIDEV-NOTE: The PostgreSQL extension now supports structured output generation using the same grammar-based approach as the main library.

## PostgreSQL Extension Async Functions (v1.1.0+)

AIDEV-NOTE: The PostgreSQL extension now includes async counterparts for all generation and embedding functions.

**Key Features:**
- Non-blocking execution: Functions return UUID immediately
- Queue-based processing with priority support
- Background worker handles AI operations
- LISTEN/NOTIFY integration for responsive processing
- Batch operations for efficiency

**New Async SQL Functions:**
```sql
-- Async generation
steadytext_generate_async(prompt, max_tokens) → UUID
steadytext_embed_async(text, use_cache) → UUID
steadytext_generate_json_async(prompt, schema, max_tokens, use_cache, seed) → UUID
steadytext_generate_regex_async(prompt, pattern, max_tokens, use_cache, seed) → UUID
steadytext_generate_choice_async(prompt, choices, use_cache, seed) → UUID

-- Batch operations
steadytext_generate_batch_async(prompts[], max_tokens) → UUID[]
steadytext_embed_batch_async(texts[], use_cache) → UUID[]

-- Result retrieval
steadytext_check_async(request_id) → (status, result, error, ...)
steadytext_get_async_result(request_id, timeout_seconds) → TEXT
steadytext_cancel_async(request_id) → BOOLEAN
steadytext_check_async_batch(request_ids[]) → TABLE
```

**Implementation Details:**
- Queue table (`steadytext_queue`) stores all async requests
- Python worker (`pg_steadytext/python/worker.py`) processes queue
- Updated to handle all structured generation types
- Supports concurrent workers with `FOR UPDATE SKIP LOCKED`
- All async functions follow the same patterns as synchronous versions
- Functions use the same `daemon_connector.py` methods that wrap the main library

AIDEV-TODO: Add tests for PostgreSQL structured generation functions
AIDEV-TODO: Consider adding support for Pydantic models in PostgreSQL (would need JSON serialization)

## Distribution Packaging (v1.2.0+)

AIDEV-NOTE: SteadyText includes comprehensive packaging scripts for multiple Linux distributions with minimal maintenance overhead.

### Package Types

**Supported Formats:**
- **Debian/Ubuntu** (.deb) - For apt-based systems with multiple PostgreSQL versions
- **RHEL/Rocky/Fedora** (.rpm) - For yum/dnf-based systems with multiple PostgreSQL versions
- **PGXN** - PostgreSQL Extension Network compatible packages
- **Pigsty** - Configuration for Pigsty PostgreSQL distribution

### Packaging Architecture

**Key Components:**
- `packaging/build-deb.sh` - Builds Debian packages for PostgreSQL 14-17
- `packaging/build-rpm.sh` - Builds RPM packages for PostgreSQL 14-17
- `packaging/pgxn-upload.sh` - Prepares PGXN distribution with Pigsty config
- `packaging/test-builds.sh` - Validates built packages
- `.github/workflows/build-packages.yml` - Automated CI/CD for releases

**Version Management:**
- Extension version read from `pg_steadytext/META.json`
- Python version read from `pyproject.toml`
- No manual version updates needed in packaging scripts

### Building Packages

```bash
# Build all packages
./build-packages.sh

# Build specific type
./build-packages.sh deb
./build-packages.sh rpm
./build-packages.sh pgxn
```

**Package Contents:**
- PostgreSQL extension files (.control, .sql)
- Python support modules in `/opt/steadytext/`
- Systemd service for async worker
- Virtual environment with SteadyText installed
- Documentation and license files

AIDEV-NOTE: Packages handle Python dependencies via virtual environments to avoid system conflicts
AIDEV-NOTE: Post-install scripts automatically set up the Python environment and systemd service
AIDEV-TODO: Consider adding support for Alpine Linux (apk) packages
AIDEV-TODO: Add package signing for security-conscious deployments

## Development Workflow

### Additional Process Guidance

- At the end of code changes, please make sure to run `poe format` and `poe lint` (in that order) to make sure the code follows the style guide.
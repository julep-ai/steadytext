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

- AIDEV-NOTE: Remote models (containing ':' in the name) with unsafe_mode=True skip daemon entirely
- AIDEV-NOTE: This prevents unnecessary local model loading when using remote embeddings
- AIDEV-NOTE: Daemon supports --skip-embeddings flag to avoid loading embedding model when only using remote embeddings

### Usage Patterns

**CLI Commands:**
```bash
# Start daemon
st daemon start [--host HOST] [--port PORT] [--foreground]

# Check status
st daemon status [--json]

# Stop daemon
st daemon stop [--force]
```

**SDK Usage:**
```python
# Use daemon for all operations in a scope (ensures connection)
with use_daemon():
    text = generate("Hello world")
    embedding = embed("Some text")

# Disable daemon globally via environment
os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
```

### AIDEV Considerations

- AIDEV-NOTE: The daemon maintains singleton model instances and falls back to direct loading on connection failure
- AIDEV-NOTE: Daemon requires explicit startup with 'st daemon start' - no automatic startup in main library
- AIDEV-NOTE: Cache integration: Daemon and direct access share the same centralized SQLite cache for consistency
- AIDEV-NOTE: Streaming handles caching by collecting tokens and populating cache after completion
- AIDEV-NOTE: Special cases: Logprobs bypass cache, custom eos_string included in cache keys
- AIDEV-TODO: Consider adding connection pooling for high-concurrency scenarios
- AIDEV-TODO: Add metrics/monitoring endpoints for production deployments

## Models (v2025.8.26+)

- AIDEV-NOTE: Documentation previously mentioned Gemma-3n models but code actually uses Qwen3 models
- AIDEV-NOTE: As of v2025.8.26, default generation models are Qwen3-4B (small) and Qwen3-30B (large)

SteadyText v2025.8.26+ uses Qwen3 models for generation and Jina v4 for embeddings:

**Generation Models:**
- **Small** (default): Qwen3-4B-Instruct - Fast, efficient for most tasks
- **Large**: Qwen3-30B-A3B-Instruct - Higher quality for complex tasks
- Use `size="small"` or `size="large"` parameter, or set `STEADYTEXT_GENERATION_SIZE` env var

**Embedding Model:**
- **Jina v4 Text Retrieval**: 2048-dim embeddings truncated to 1024 using Matryoshka
- Requires Query/Passage prefix: `mode="query"` for searches, `mode="passage"` for documents
- Set default mode via `STEADYTEXT_EMBEDDING_MODE` env var

- AIDEV-NOTE: Jina v4 outputs 2048 dimensions but we truncate to 1024 for backward compatibility
- AIDEV-NOTE: Query/Passage prefixes are critical for optimal retrieval performance
- AIDEV-TODO: Consider exposing full 2048 dimensions as an option

## Mini Models Support (v2.7.0+)

- AIDEV-NOTE: Mini models provide ~10x faster CI/testing with smaller model sizes
- Generation: Gemma-3-270M (~97MB), Embedding: BGE-large (~130MB), Reranking: BGE-base (~300MB)
- Use STEADYTEXT_USE_MINI_MODELS=true environment variable for CI/testing
- CLI supports --size mini flag for all commands (generate, embed, rerank, daemon)
- AIDEV-TODO: Consider adding more mini model variants for different use cases

## Temperature Parameter (v2025.8.15+)

SteadyText supports temperature control for text generation, allowing fine-tuned control over output randomness:

**Temperature Values:**
- `0.0` (default): Fully deterministic, always picks highest probability token
- `0.1-0.5`: Low randomness, mostly coherent and focused
- `0.6-1.0`: Balanced creativity, good for general use
- `1.0-2.0`: High creativity, more varied and unpredictable outputs

**Implementation Details:**
- Temperature is integrated into cache keys to prevent collisions between different temperature values
- Sampling parameters (top_k, top_p, min_p) automatically adjust for non-zero temperatures
- Same seed + temperature combination always produces identical output (reproducible randomness)
- Supported across all interfaces: Python API, CLI, daemon, and remote providers

- AIDEV-NOTE: Temperature affects cache keys - "prompt::TEMP::0.5" format for non-default values
- AIDEV-NOTE: Temperature=0.0 uses greedy decoding (top_k=1), temperature>0 uses top_k=40, top_p=0.95
- AIDEV-TODO: Consider adding temperature presets (e.g., "creative", "balanced", "focused")

## Reranking Support (v1.3.0+)

SteadyText v1.3.0+ includes document reranking functionality using the Qwen3-Reranker-4B model.

**Reranking Features:**
- Query-document relevance scoring
- Batch document reranking
- Custom task descriptions for domain-specific reranking
- Caching support via dedicated reranking cache
- CLI command: `st rerank`
- PostgreSQL functions: `steadytext_rerank()` and async variants

- AIDEV-NOTE: Default model `Qwen3-Reranker-4B-GGUF` uses binary scoring (1.0/0.0) based on yes/no tokens
- AIDEV-NOTE: v2.5.2+ caches all scores and improved fallback with semantic heuristics
- AIDEV-TODO: Consider cross-encoder models, streaming support, and probabilistic scoring

## Cache Management

SteadyText v1.3+ uses a centralized cache management system with pluggable backends (v2.2.0+).

**Cache Backends (v2.2.0+):**
- **SQLite** (default): Thread-safe local storage with WAL mode
- **D1**: Cloudflare's distributed SQLite for edge deployments
- **Memory**: In-memory cache for testing/ephemeral workloads

- AIDEV-NOTE: Cache backends use factory pattern (`cache/factory.py`), all implement CacheBackend interface
- AIDEV-NOTE: D1 backend requires proxy Worker due to Cloudflare restrictions
- AIDEV-TODO: Consider adding Redis backend for distributed caching

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

## Structured Generation (v2.4.0+)

- AIDEV-NOTE: Uses LlamaGrammar.from_json_schema() for JSON/Pydantic schemas - more reliable than custom GBNF generation
- AIDEV-NOTE: Mini models (Gemma-3-270M QAT) work with LlamaGrammar.from_json_schema() but had issues with custom GBNF grammars
- Conversion support: JSON schemas, Pydantic models, regex, choices, Python types → GBNF
- AIDEV-NOTE: Remote models (v2.6.2+) support structured generation via unsafe_mode=True
- AIDEV-FIXED: Resolved mini model issues by switching to LlamaGrammar.from_json_schema() instead of custom grammar generation
- AIDEV-TODO: Expand regex conversion and add recursive schema support

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

# Remote models with structured generation (v2.6.2+)
result = generate_json(
    "Create a person", 
    Person,
    model="openai:gpt-4o-mini",
    unsafe_mode=True
)
```

- AIDEV-NOTE: Two-phase generation: reasoning → `<json-` tag → grammar-constrained output
- AIDEV-NOTE: Deterministic/cacheable, bypasses cache for logprobs, on-the-fly grammar conversion
- AIDEV-TODO: Add streaming support and grammar caching

## Versioning Policy (As of 2025.8.26)

- AIDEV-NOTE: Version 2025.8.26 includes complete schema qualification for all PostgreSQL extension functions

Both SteadyText components now use **date-based versioning** instead of semantic versioning:

- **Format:** `yyyy.mm.dd` (no zero-padding, e.g., `2025.8.16`, `2025.12.3`)
- **Applies to:** Both the Python package (steadytext) and PostgreSQL extension (pg_steadytext)
- **Rationale:** The rapid pace of model improvements and feature changes made semantic versioning impractical. Date-based versioning provides clearer insight into release recency and better aligns with our continuous improvement philosophy.
- **Migration:** Existing installations can upgrade using standard commands (pip for Python, PostgreSQL extension commands for pg_steadytext)
- **Documentation:** The CHANGELOG.md files explain the versioning scheme and track changes

- AIDEV-NOTE: When creating new releases, use current date in yyyy.mm.dd format for both components
- AIDEV-NOTE: Version consistency - keep both Python package and pg_steadytext versions aligned when possible

## Development Commands

### Testing

```bash
# Run all tests with UV
uv run poe test

# Allow model downloads in tests (models are downloaded on first use)
STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true uv run poe test
```

All tests are designed to pass even if models cannot be downloaded. Model-dependent tests are automatically skipped unless `STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true` is set.

### Linting and Formatting

```bash
uv run poe format
uv run poe lint
uv run poe test
uv run poe check # typecheck
uv run poe docs-build # build the docs

uv run poe build # Runs all of the above
```

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

- AIDEV-NOTE: Index uses chonkie (chunking), faiss-cpu (vectors), auto-retrieval with default.faiss

## Architecture Overview

SteadyText provides deterministic text generation and embedding with zero configuration. The core principle is "Never Fails" - all functions return deterministic outputs even when models can't be loaded.

### Key Components

**Core Layer (`steadytext/core/`)**
- `generator.py`: Text generation with `DeterministicGenerator` class and deterministic fallback function
- `embedder.py`: Embedding creation with L2 normalization and deterministic fallback to zero vectors

**Models Layer (`steadytext/models/`)**
- `cache.py`: Downloads and caches GGUF models from Hugging Face
- `loader.py`: Singleton model loading with thread-safe caching via `_ModelInstanceCache`

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

## Cache Configuration

SteadyText uses disk-backed frecency caches for both generation and embedding results. The caches can be configured via environment variables:

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
uv run python benchmarks/run_all_benchmarks.py --quick  # Optional, for CI
```

## UV Package Manager

UV is a modern, blazing-fast Python package and project manager written in Rust. It serves as a drop-in replacement for pip, virtualenv, poetry, and other Python tooling, offering 10-100x speed improvements.

### Installation

Install UV system-wide using the official installer:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Basic Usage

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

### Troubleshooting

**Common Issues:**
- If UV can't find Python version, install it: `uv python install 3.11`
- For permission errors on Linux/Mac: `sudo chown -R $USER ~/.local/share/uv`
- To force recreation of virtual environment: `rm -rf .venv && uv sync`

- AIDEV-TODO: Add UV-specific CI/CD configurations for faster builds

## Recent PostgreSQL Extension Updates (v2025.8.26)

- AIDEV-NOTE: Complete schema qualification for ALL functions accessing extension tables (v2025.8.26)
- AIDEV-NOTE: Functions use dynamic schema resolution via plpy.execute() and plpy.quote_ident()
- AIDEV-NOTE: SQL alias functions use @extschema@ placeholder for proper schema references
- AIDEV-NOTE: Fixed Python scoping issues in aggregate functions (argument reassignment problem)
- AIDEV-NOTE: Enhanced summarization functions with model and unsafe_mode parameters for remote models
- AIDEV-TODO: Consider adding more comprehensive tests for remote model summarization
- AIDEV-TODO: Document the schema qualification pattern more thoroughly for extension developers

## PostgreSQL Extension (pg_steadytext)

### pgTAP Testing Best Practices

**Running Tests:**
```bash
# Run all tests
PGHOST=postgres PGPORT=5432 PGUSER=postgres PGPASSWORD=password ./run_pgtap_tests.sh

# Run specific test
PGHOST=postgres PGPORT=5432 PGUSER=postgres PGPASSWORD=password psql test_postgres -X -f test/pgtap/08_reranking.sql

# Use mini models to prevent timeouts (critical for reranking tests)
PGHOST=postgres PGPORT=5432 PGUSER=postgres PGPASSWORD=password STEADYTEXT_USE_MINI_MODELS=true ./run_pgtap_tests.sh
```

**Common Test Issues and Solutions:**

1. **Tests hanging on model loading:**
   - AIDEV-FIX: Set `STEADYTEXT_USE_MINI_MODELS=true` environment variable
   - Mini models are ~10x smaller and prevent timeouts
   - Must be set at container level for PostgreSQL extension tests

2. **plpy.Error in tests:**
   - Usually means incorrect plpy.execute() usage
   - Check that parameterized queries use plpy.prepare()
   - Example error: `plpy.Error: plpy.execute expected a query or a plan`

3. **Type mismatch errors:**
   - Python strings must be explicitly cast to PostgreSQL types
   - Common: UUID casting with `$1::uuid`, JSONB with `$2::jsonb`

4. **Function output format issues:**
   - Table-returning functions must use `yield` not `return`
   - Check test expectations match actual output format

5. **Test syntax errors:**
   - Functions with OUT parameters don't need column definitions
   - Error: `a column definition list is redundant for a function with OUT parameters`

AIDEV-NOTE: Always run tests with mini models in CI to prevent timeouts
AIDEV-NOTE: Check test output carefully - PL/Python errors can be misleading
AIDEV-TODO: Add automated test suite for all PL/Python edge cases

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

**DevContainer Workflow:**
For rapid development in the devcontainer environment:
```bash
# Quick rebuild after making changes (recommended)
/workspace/.devcontainer/rebuild_extension_simple.sh

# Auto-rebuild on file changes  
/workspace/.devcontainer/watch_extension.sh
```
AIDEV-NOTE: The devcontainer uses a copy-and-build approach for fast iteration
AIDEV-NOTE: Scripts handle all the complexity of copying files and rebuilding

## PostgreSQL Extension Async Functions (v1.1.0+)

**Features:** Non-blocking UUID returns, queue-based processing, LISTEN/NOTIFY, batch operations

**Implementation:** Queue table + Python worker using `FOR UPDATE SKIP LOCKED` for concurrency

- AIDEV-TODO: Add tests for structured generation, consider Pydantic model support

## Distribution Packaging (v1.2.0+)

**Formats:** Debian/Ubuntu (.deb), RHEL/Fedora (.rpm), PGXN, Pigsty

**Build:** `./build-packages.sh [deb|rpm|pgxn]`

- AIDEV-NOTE: Uses virtual environments to avoid system Python conflicts
- AIDEV-TODO: Add Alpine support and package signing

## Development Workflow

### Additional Process Guidance

- At the end of code changes, please make sure to run `poe build`.

## Development Container (v2.6.0+)

**Usage:**
```bash
# Open in VSCode with Dev Containers extension
# Or use GitHub Codespaces

# PostgreSQL is available at:
# - Host: localhost (or postgres service name)
# - Port: 5432
# - User: postgres
# - Password: password
# - Database: postgres
```

**Development Workflow:**
1. Container automatically installs SteadyText and pg_steadytext
2. PostgreSQL starts automatically with required extensions
3. Run tests with `uv run pytest` or `make test` in pg_steadytext/
4. Develop with full IDE support and database access

**PostgreSQL Extension Development (2025.8.27+):**
```bash
# Quick rebuild after changes (takes ~2-3 seconds)
/workspace/.devcontainer/rebuild_extension_simple.sh

# Auto-rebuild on file changes
/workspace/.devcontainer/watch_extension.sh
```

- AIDEV-NOTE: DevContainer runs Docker-in-Docker, limiting mount options for postgres container
- AIDEV-NOTE: Solution: Copy-and-build approach via rebuild_extension_simple.sh
- AIDEV-NOTE: Watch script monitors SQL/Python files and triggers automatic rebuilds
- AIDEV-TODO: Add support for GPU passthrough in devcontainer for CUDA models
- AIDEV-TODO: Consider adding Redis service for distributed cache testing
- AIDEV-NOTE: The devcontainer mounts Docker socket for testing containerized builds

## Unsafe Mode: Remote Models (v2.6.0+)

SteadyText includes an "unsafe mode" that allows using remote AI models with best-effort determinism via seed parameters.

**Provider Support (v2.6.2+):**
- **OpenAI**: Text generation (gpt-4o, gpt-4o-mini, GPT-5 series) and embeddings (text-embedding-3-small/large) with seed parameter
  - AIDEV-NOTE: GPT-5 and o1 reasoning models require temperature=1.0 and are automatically adjusted
- **Cerebras**: Fast Llama model generation with seed parameter
- **VoyageAI**: Specialized embeddings (voyage-3, voyage-large-2, etc.) - no seed support
- **Jina AI**: Multilingual embeddings (jina-embeddings-v3, v2-base variants) - no seed support

**Custom Options Support (v2025.8.17+):**
You can pass additional provider-specific options using the `options` parameter:
```python
# Python API
result = generate("Hello", model="openai:gpt-4o-mini", unsafe_mode=True, 
                 options={"top_p": 0.95, "presence_penalty": 0.5})

# CLI
echo "Hello" | st --unsafe-mode --model openai:gpt-4o-mini --options '{"top_p": 0.95}'
```

AIDEV-TODO: Add support for more providers (Anthropic when they add seed support, Together.ai, etc.)
AIDEV-TODO: Consider adding structured output support for remote models
AIDEV-TODO: Add telemetry to track unsafe mode usage patterns

### PostgreSQL Extension Unsafe Mode Testing (v2025.8.26+)

The pg_steadytext extension supports unsafe_mode for remote models. This must be thoroughly tested before production deployment.

**Functions Supporting unsafe_mode:**
- `steadytext_generate` - Text generation with remote models
- `steadytext_embed` - Embedding generation with remote models
- `steadytext_summarize` aggregate - Summarization with metadata containing model/unsafe_mode
- `steadytext_summarize_text` - Single-value summarization with model/unsafe_mode parameters

**Testing Procedure:**

```sql
-- AIDEV-NOTE: Always test in a separate database to avoid production impact
CREATE DATABASE manual_test_db;
\c manual_test_db

-- Install required extensions
CREATE EXTENSION IF NOT EXISTS plpython3u;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_steadytext;

-- Verify correct version (should be 2025.8.26 or later)
SELECT extname, extversion FROM pg_extension WHERE extname = 'pg_steadytext';

-- Test 1: Error validation - remote model without unsafe_mode should fail
SELECT steadytext_generate('Hello', model := 'openai:gpt-4o-mini');
-- Expected: ERROR: Remote models (containing ':') require unsafe_mode=TRUE

-- Test 2: Error validation - unsafe_mode without model should fail  
SELECT steadytext_generate('Hello', unsafe_mode := TRUE);
-- Expected: ERROR: unsafe_mode=TRUE requires a model parameter to be specified

-- Test 3: Successful remote model generation
SELECT steadytext_generate(
    'What is the capital of France? Answer in one word.',
    model := 'openai:gpt-4o-mini',
    unsafe_mode := TRUE
);
-- Expected: "Paris." or similar

-- Test 4: Remote embedding generation
SELECT vector_dims(steadytext_embed(
    'Hello world',
    model := 'openai:text-embedding-3-small',
    unsafe_mode := TRUE
));
-- Expected: 1024 dimensions

-- Test 5: Summarization with remote model
SELECT steadytext_summarize_text(
    'PostgreSQL is an advanced open-source database system.',
    metadata := '{}'::jsonb,
    model := 'openai:gpt-4o-mini',
    unsafe_mode := TRUE
);
-- Expected: A concise summary

-- Test 6: Aggregate summarization with metadata
CREATE TABLE test_docs (content TEXT);
INSERT INTO test_docs VALUES 
    ('PostgreSQL is powerful'),
    ('It supports SQL'),
    ('Extensions are flexible');

SELECT steadytext_summarize(
    content,
    jsonb_build_object('model', 'openai:gpt-4o-mini', 'unsafe_mode', true)
) FROM test_docs;
-- Expected: A summary of all documents
```

**Important Validation Rules:**
- AIDEV-NOTE: Remote models (containing ':' in the model name) MUST have unsafe_mode=TRUE
- AIDEV-NOTE: unsafe_mode=TRUE MUST have a model parameter specified  
- AIDEV-NOTE: The aggregate function passes model/unsafe_mode through metadata JSONB
- AIDEV-NOTE: All remote model calls skip the daemon and use direct API calls

**Supported Remote Models (as of v2025.8.26):**
- OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, text-embedding-3-small, text-embedding-3-large
- Cerebras: cerebras:llama3.1-8b, cerebras:llama3.1-70b
- VoyageAI: voyage:voyage-3, voyage:voyage-large-2 (embeddings only, no seed support)
- Jina: jina:jina-embeddings-v3 (embeddings only, no seed support)

AIDEV-TODO: Add automated tests for all unsafe_mode scenarios in pgTAP test suite
AIDEV-TODO: Create performance benchmarks comparing local vs remote models
AIDEV-TODO: Document rate limiting and error handling for remote model failures

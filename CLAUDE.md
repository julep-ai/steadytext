# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General Guidelines

| # | AI may do | AI must NOT do |
|---|-----------|----------------|
| G-0 | Ask for clarification when unsure about project-specific features or decisions | Write changes or use tools when uncertain about project context |
| G-1 | Generate code in relevant source directories (e.g., `steadytext/`, `pg_steadytext/`) | Touch test files or specs without explicit permission |
| G-2 | Add/update `AIDEV-ANCHOR:`, `AIDEV-REF:`, and `AIDEV-NOTE:` comments; keep anchors current | Remove anchor comments without updating references accordingly |
| G-3 | Follow project's lint/style configs (`pyproject.toml`, `.ruff.toml`) | Re-format code to any other style |
| G-4 | Ask for confirmation before changes >300 LOC or >3 files | Refactor large modules without human guidance |
| G-5 | Stay within current task context; suggest starting fresh if needed | Continue work from prior unrelated context |

## Anchor Comments

Structured, greppable comments for navigation and documentation.

**Tags:** `AIDEV-ANCHOR:` (headings), `AIDEV-REF:` (cross-refs), `AIDEV-TODO:` (tasks), `AIDEV-NOTE:` (context), `AIDEV-QUESTION:`, `FIXME:`

**Guidelines:**
- Ultra-short phrases (≤60 chars), no filler words
- Place anchors above major blocks (classes, functions, routes)
- Keep ~1 anchor per 40-60 lines
- Format refs: `AIDEV-REF: path/file.py -> anchor text`
- For documentation rewrite, follow the addendum at `specs/003-documentation-rewrite/anchor-guidelines.md`

> For detailed anchor management, use the **anchor-comment-manager** agent.

## Daemon Architecture

Persistent model serving via ZeroMQ to avoid repeated loading overhead.

### Usage

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
with use_daemon():
    text = generate("Hello world")
    embedding = embed("Some text")
```

- Remote models skip daemon when unsafe_mode=True

## Models & Configuration

### Standard Models
- **Generation:** Qwen3-4B-Instruct (default), Qwen3-30B-A3B-Instruct (large)
- **Embedding:** Jina v4 (2048-dim → 1024 truncated, requires query/passage prefixes)
- **Reranking:** Qwen3-Reranker-4B

### Mini Models (CI/Testing)
~10x faster: Gemma-3-270M, BGE-large, BGE-base
Enable: `STEADYTEXT_USE_MINI_MODELS=true` or `--size mini`

### Remote Models (Unsafe Mode)
OpenAI, Cerebras, VoyageAI, Jina, OpenRouter - use `model="openai:gpt-4o-mini" unsafe_mode=True`
OpenRouter provides unified access to multiple providers: `model="openrouter:anthropic/claude-3.5-sonnet"`

> **Embedding overrides:** Setting `EMBEDDING_OPENAI_BASE_URL` and `EMBEDDING_OPENAI_API_KEY` auto-routes embeddings (library, CLI, pg extension). Use `EMBEDDING_OPENAI_MODEL` to change the default; explicit `model=` arguments still take precedence.



## Cache Management

**Backends:** SQLite (default), D1 (Cloudflare edge), Memory (testing)

- Factory pattern in `cache/factory.py`, all implement CacheBackend interface

## AI Assistant Workflow

1. **Grep for Anchors** - Understand codebase structure first
2. **Consult CLAUDE.md** - Check project guidelines
3. **Clarify if needed** - Ask targeted questions
4. **Plan approach** - Break down complex tasks
5. **Execute** - Start immediately for trivial tasks, get approval for complex ones
6. **Track progress** - Use AIDEV-TODO comments and TodoWrite tool
7. **Update documentation** - Keep anchors and docs current
8. **Review** - Get user feedback and iterate

## Structured Generation

Generate structured output using llama.cpp grammars - JSON schemas, Pydantic models, regex patterns, choices.

### Examples

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

## Versioning Policy

**Date-based versioning:** `yyyy.mm.dd` format (e.g., `2025.8.27`)

Applies to both Python package and pg_steadytext extension. See CHANGELOG.md for version history.

> For release management, use the **github-pr-manager** agent.

## Prompt Registry

Jinja2-based template management for PostgreSQL with immutable versioning and automatic variable extraction.

**Key Functions:**
```sql
-- Management functions
SELECT st_prompt_create(slug, template, description, metadata);
SELECT st_prompt_update(slug, template, metadata);
SELECT st_prompt_delete(slug);

-- Rendering and retrieval
SELECT st_prompt_render(slug, variables, version, strict_mode);
SELECT * FROM st_prompt_get(slug, version);

-- Discovery functions  
SELECT * FROM st_prompt_list();
SELECT * FROM st_prompt_versions(slug);
```

**Real-World Usage:**
```sql
-- AI prompt management with versioning
SELECT st_prompt_create(
    'code-review', 
    'Review this {{ language }} code:\n\n```{{ language }}\n{{ code }}\n```\n\nFocus areas:\n{% for area in focus_areas -%}\n- {{ area }}\n{% endfor %}',
    'AI code review prompt template',
    '{"category": "ai", "model": "gpt-4", "version": "v2.1"}'::jsonb
);

-- Dynamic email templates with personalization
SELECT st_prompt_create(
    'order-confirmation',
    'Dear {{ customer.name }},\n\nYour order #{{ order.id }} is confirmed.\n\n{% if order.total > 100 -%}\n🎉 You qualify for free shipping!\n{% endif %}',
    'Order confirmation email with conditional content'
);

-- Render with variables
SELECT st_prompt_render(
    'code-review', 
    '{"language": "Python", "code": "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)", "focus_areas": ["performance", "readability"]}'::jsonb
);
```

See `docs/PROMPT_REGISTRY.md` for comprehensive documentation.
## Development Commands

### Testing

```bash
# Run all tests with UV
uv run poe test

# Allow model downloads in tests (models are downloaded on first use)
STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true uv run poe test
```

All tests are designed to pass even if models cannot be downloaded. Model-dependent tests are automatically skipped unless `STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true` is set.

- Targeted env override coverage: `pytest tests/test_embedding_env_overrides.py`

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

- Index uses chonkie (chunking), faiss-cpu (vectors), auto-retrieval with default.faiss

## Architecture Overview

Deterministic text generation and embedding with zero configuration. "Never Fails" principle - deterministic outputs even without models.

### Key Components

**Core Layer (`steadytext/core/`)**
- `generator.py`: Text generation with `DeterministicGenerator` class and deterministic fallback function
- `embedder.py`: Embedding creation with L2 normalization and deterministic fallback to zero vectors

**Models Layer (`steadytext/models/`)**
- `cache.py`: Downloads and caches GGUF models from Hugging Face
- `loader.py`: Singleton model loading with thread-safe caching via `_ModelInstanceCache`

- **Generation:** Deterministic sampling, hash-based fallback
- **Embeddings:** 1024-dim L2-normalized vectors, zero-vector fallback
- **Models:** Singleton pattern with thread-safe caching
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



## UV Package Manager

Blazing-fast Python package manager (10-100x faster than pip).

### Quick Start
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add dependencies (creates .venv automatically)
uv add requests numpy pandas

# Run code (uses .venv automatically)
uv run python script.py
uv run pytest

# Install Python version
uv python install 3.11
```

### pgTAP Testing

> For comprehensive pgTAP test execution, monitoring, and troubleshooting, use the **pgtap-test-runner** agent.

Quick reference:

```bash
# Always use mini models to prevent timeouts
PGHOST=postgres PGPORT=5432 PGUSER=postgres PGPASSWORD=password \
  STEADYTEXT_USE_MINI_MODELS=true ./run_pgtap_tests.sh
```

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

The devcontainer uses a copy-and-build approach for fast iteration
Scripts handle all the complexity of copying files and rebuilding

### Async Functions

Non-blocking UUID returns with queue-based processing, LISTEN/NOTIFY, batch operations.

### Distribution Packaging

**Formats:** .deb, .rpm, PGXN, Pigsty - Build with `./build-packages.sh`

### Development Workflow

Always run `poe build` after code changes.

### Development Container

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

**Quick Extension Development:**

```bash
# Rebuild after changes (2-3 seconds)
/workspace/.devcontainer/rebuild_extension_simple.sh

# Auto-rebuild on file changes
/workspace/.devcontainer/watch_extension.sh
```

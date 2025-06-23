# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Use Anchor comments

Add specially formatted comments throughout the codebase, where appropriate, for yourself as inline knowledge that can be easily `grep`ped for.

- Use `AIDEV-NOTE:`, `AIDEV-TODO:`, or `AIDEV-QUESTION:` as prefix as appropriate.

- *Important:* Before scanning files, always first try to grep for existing `AIDEV-…`.

- Update relevant anchors, after finishing any task.

- Make sure to add relevant anchor comments, whenever a file or piece of code is:

  * too complex, or
  * very important, or
  * could have a bug

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


## Development Commands

### Testing
```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=steadytext --cov-report=xml

# Run specific test files
python -m pytest tests/test_steadytext.py
python -m pytest test_gen.py
python -m pytest test_fallback_gen.py

# Allow model downloads in tests (models are downloaded on first use)
STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true python -m pytest

# Configure cache settings
STEADYTEXT_GENERATION_CACHE_CAPACITY=512 python -m pytest
STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=100.0 python -m pytest
STEADYTEXT_EMBEDDING_CACHE_CAPACITY=1024 python -m pytest
STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=200.0 python -m pytest
```

All tests are designed to pass even if models cannot be downloaded. Model-dependent tests are automatically skipped unless `STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true` is set.

### Linting and Formatting
```bash
# Check code quality
python -m flake8 .

# Install and run pre-commit hooks
pre-commit install
pre-commit run --all-files

# Using poethepoet tasks (if available)
poe lint
poe format
poe check
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
st "What is Python?" --index-file my_index.faiss
st "explain this error" --no-index  # Disable index search
```

AIDEV-NOTE: The index functionality uses:
- chonkie for deterministic text chunking (512 token default)
- faiss-cpu for vector storage (IndexFlatL2 for exact search)
- Automatic context retrieval when default.faiss exists
- Aggressive caching of search results for determinism

### Installation
```bash
# Install in development mode
python -m pip install -e .

# Install with uv (if available)
uv pip install -e .

# Build package with uv
uv build
```

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
- Uses openbmb.BitCPM4-1B.Q8_0.gguf with deterministic sampling parameters
- Fallback generates text using hash-based word selection when model unavailable
- Always returns strings, never raises exceptions
- Supports both batch generation (`generate()`) and streaming generation (`generate_iter()`)

**Embeddings:**
- Uses Qwen3-Embedding-0.6B-Q8_0.gguf configured for embeddings
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

**Commands (`steadytext/cli/commands/`)**
- `generate.py`: Text generation with streaming, JSON output, and logprobs support
- `embed.py`: Embedding creation with multiple output formats (JSON, numpy, hex)
- `cache.py`: Cache management and status commands
- `models.py`: Model management (list, preload, etc.)

**CLI Features:**
- Deterministic outputs matching the Python API
- Multiple output formats (raw text, JSON with metadata, structured data)
- Streaming support for real-time text generation
- Stdin/pipe support for unix-style command chaining
- Log probability access for advanced use cases

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
```bash
# Run all benchmarks
python benchmarks/run_all_benchmarks.py

# Quick benchmarks for CI
python benchmarks/run_all_benchmarks.py --quick

# Test benchmarks are working
python benchmarks/test_benchmarks.py
```

### Key Metrics
- **Speed**: Generation/embedding throughput, latency percentiles, memory usage
- **Accuracy**: Determinism verification, quality checks, LightEval standard benchmarks

AIDEV-NOTE: When modifying core generation/embedding code, always run benchmarks to check for performance regressions

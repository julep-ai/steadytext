# Changelog

> [!IMPORTANT]
> **Versioning Change Notice (2025.8.16)**
> 
> SteadyText has transitioned from semantic versioning to date-based versioning (yyyy.mm.dd format).
> This change reflects the rapid pace of AI model improvements and feature additions, where traditional
> semantic versioning became impractical. Date-based versioning provides clearer insight into release
> recency and better aligns with our continuous improvement philosophy.

## Version 2025.8.27 (2025-08-27)

### Performance Improvements
- **Remote Embedding Optimization:** Fixed issue where remote embeddings (e.g., OpenAI) were unnecessarily loading local embedding models
  - The `embed()` function now skips daemon connection entirely for remote models with `unsafe_mode=True`
  - This prevents the daemon from preloading the local embedding model when only using remote embeddings
  - Matches the existing behavior of `generate()` function for remote models

### New Features
- **Daemon Skip Embeddings Flag:** Added `--skip-embeddings` flag to daemon commands
  - Use `st daemon start --skip-embeddings` to start daemon without loading embedding model
  - Useful when only using remote embeddings to save memory and startup time
  - Also available for `st daemon restart --skip-embeddings`

### Bug Fixes
- **Embedding Path Optimization:** Remote embeddings with `unsafe_mode=True` now bypass local model loading entirely
  - Previously, remote embeddings would still trigger local model loading through the daemon
  - Now correctly skips daemon for remote models, going directly to `core_embed()`

## Version 2025.8.26 (2025-08-26)

### Changes
- **Version Bump:** Date-based version update

## Version 2025.8.17 (2025-08-17)

### New Features
- **GPT-5 Series Support:** Added full support for OpenAI's GPT-5 reasoning models
  - Automatic temperature adjustment to 1.0 for GPT-5 models (gpt-5-mini, gpt-5-pro, etc.)
  - Similar handling to existing o1 series reasoning models
  - Models require temperature=1.0 and don't support temperature tuning

- **Custom Provider Options:** Added ability to pass provider-specific parameters
  - New `options` parameter in `generate()` and `generate_iter()` functions
  - CLI support via `--options` flag accepting JSON strings
  - Allows passing custom parameters like `top_p`, `presence_penalty`, etc. to remote providers
  - Example: `echo "text" | st --unsafe-mode --model openai:gpt-4o-mini --options '{"top_p": 0.95}'`

### PostgreSQL Extension Updates
- **AI Summarization Enhancement:** Renamed `ai_*` functions to `steadytext_*` with `st_*` aliases for consistency
- **Remote Model Support:** Added `model` and `unsafe_mode` parameters to summarization functions
- **Schema Qualification Fix:** Fixed TimescaleDB continuous aggregate compatibility by adding schema qualification to all table references
- **Python Scoping Fix:** Resolved Python scoping issues in PL/Python aggregate functions that caused NameError
- **Default Improvements:** Increased default max_facts from 5 to 10 in fact extraction

### Documentation Updates
- **Model Documentation:** Corrected README to reflect actual Qwen3 models being used (not Gemma-3n)
- **Version Alignment:** Updated all version references to 2025.8.17
- **Feature Documentation:** Added comprehensive documentation for new summarization features
- **GPT-5 Documentation:** Updated documentation to include GPT-5 series support and custom options

## Version 2025.8.16 (2025-08-16)

### Changes
- **Version Bump:** Date-based version update
- **Documentation:** Updated all version references throughout the codebase
- **PostgreSQL Extension:** Aligned extension version with main package

## Version 2025.8.15 (2025-08-15)

### Major Changes
- **Date-Based Versioning:** Switched from semantic versioning (2.6.2) to date-based versioning (yyyy.mm.dd)
  - This change aligns both the Python package and PostgreSQL extension versioning schemes
  - Existing installations can upgrade normally using pip or standard upgrade commands
  - Rationale: The rapid evolution of AI models and features made semantic versioning impractical

## Version 2.6.3 (Unreleased - Now 2025.8.15)

### New Features
- **Temperature Parameter Support:** Added temperature parameter for controlled text generation randomness
  - Available in `generate()` and `generate_iter()` functions with default value of 0.0 (fully deterministic)
  - CLI support via `--temperature` flag (e.g., `echo "prompt" | st --temperature 0.8`)
  - Integrated with cache key generation to prevent temperature value collisions
  - Automatically adjusts sampling parameters (top_k, top_p, min_p) for non-zero temperatures
  - Full daemon support with temperature parameter passed through client/server
  - Remote provider support for OpenAI and Cerebras models
  - Maintains backward compatibility with default temperature=0.0 for deterministic behavior
  - Same seed + temperature combination always produces identical output

## Version 2.6.2 (2025-08-14)

### New Features
- **Jina AI Embeddings Support:** Added Jina as a remote embedding provider
  - Support for multilingual embeddings with high-quality semantic understanding
  - Available models: jina-embeddings-v3, jina-embeddings-v2-base-* variants
  - Configurable dimensions for jina-embeddings-v3 model
  - Note: Jina doesn't support seed parameters, so embeddings are not deterministic

- **VoyageAI Integration Improvements:** Enhanced support for VoyageAI embeddings
  - Better error handling and API compatibility
  - Support for all VoyageAI embedding models

### Documentation Changes
- **Clarified Daemon Behavior:** Updated documentation to clarify that daemon requires explicit startup
  - Removed misleading "zero configuration" and "automatic startup" claims
  - Added clear messages when daemon is not available, directing users to start it with `st daemon start`
  - Updated README, docs, and examples to reflect that daemon must be started explicitly
  - Main library already behaves correctly (no automatic startup), only documentation was misleading

### Internal Changes
- **Version Tracking:** Updated __version__ in __init__.py to match pyproject.toml

## Version 2.6.1 (2025-08-02)

### Internal Changes
- **Version Bump:** Incremented version number for maintenance release

## Version 2.6.0 (2025-07-31)

### New Features
- **Unsafe Mode: Remote Model Support (Experimental):** Added support for remote AI models with best-effort determinism
  - Requires explicit opt-in via `STEADYTEXT_UNSAFE_MODE=true` environment variable
  - Support for OpenAI models (gpt-4o, gpt-4o-mini, etc.) with seed parameter
  - Support for Cerebras Cloud API (Llama models)
  - Provider-based architecture for easy extension to other remote models
  - Prominent warnings about determinism limitations of remote models
  - Remote models specified as "provider:model" (e.g., "openai:gpt-4o-mini")
  - Supports both regular and streaming generation
  - Note: Does NOT support structured output, logprobs, or embeddings with remote models

- **Development Container Support:** Added comprehensive VSCode Dev Container configuration
  - Full PostgreSQL 17 setup with extensions (plpython3u, pgvector, pg_cron)
  - Pre-configured development environment with all dependencies
  - Docker-in-Docker support for testing containerized builds
  - Automatic SteadyText and pg_steadytext installation in editable mode
  - Multi-container setup with PostgreSQL service

### Documentation
- **Unsafe Mode Documentation:** Added comprehensive documentation about remote model support
  - Clear warnings about best-effort determinism vs true determinism
  - Provider configuration and API key setup
  - Usage examples and limitations
- **Enhanced CLAUDE.md:** Added comprehensive development container documentation
- **Updated AIDEV Comments:** Added documentation for recent fixes and new features

## Version 2.5.3 (2025-07-18)

### New Features
- **Gemma License Compliance Framework:** Added license compliance for Google's Gemma models
  - Display license notice when downloading Gemma models
  - Added `LICENSE-GEMMA.txt` placeholder for Gemma Terms of Use
  - Automatic detection of Gemma models by checking repo_id or filename
  - Clear user notification before model download begins

### Bug Fixes
- **CLI Formatting:** Applied code formatting and linting fixes to CLI commands

### Documentation
- **License Documentation:** Added comprehensive documentation about Gemma model licensing
- **Updated README:** Added information about model licenses and compliance

## Version 2.5.2 (2025-07-15)

### Bug Fixes
- **Reranker Improvements:** Fixed reranking model compatibility and enhanced fallback scoring
  - Removed unsupported `top_logprobs` parameter from model call
  - Improved fallback scoring with basic semantic heuristics for common phrases
  - Enhanced caching to include both model-generated and fallback scores
  - Better error handling and logging for debugging reranking issues

### Internal Changes
- **Version Sync:** Updated `__version__` in `__init__.py` to match pyproject.toml
- **Cache Improvements:** Reranker now caches all valid scores for better performance

## Version 2.5.1 (2025-07-14)

### New Features
- **Document Reranking:** Added powerful document reranking capabilities using the Qwen3-Reranker-4B model
  - New `rerank()` function for reordering documents by relevance to a query
  - CLI command `st rerank` with support for multiple input formats
  - PostgreSQL extension functions: `steadytext_rerank()`, `steadytext_rerank_docs_only()`, `steadytext_rerank_top_k()`
  - Async variants of all reranking functions for non-blocking operations
  - Custom task descriptions for domain-specific reranking
  - Deterministic scoring with custom seed support
  - Automatic caching of reranking results
  - Fallback to word overlap scoring when model unavailable

### Dependencies
- **Upgrade to Official llama-cpp-python:** Replaced `llama-cpp-python-bundled>=0.3.9` with official `llama-cpp-python>=0.3.12`
  - Provides better compatibility and performance with the latest GGUF models
  - Removes dependency on the bundled fork which may have compatibility issues
  - Maintains all existing functionality without API changes

### Bug Fixes
- **Temporarily Disable lighteval Dependency:** Commented out `lighteval` from benchmark extras to avoid pulling in large torch/nvidia CUDA packages
  - Prevents unnecessary installation of ~6GB+ of PyTorch and CUDA packages for users not running benchmarks
  - Optional dependency chain: lighteval → accelerate → torch → nvidia CUDA packages
  - Will be re-enabled once lighteval dependency management is improved

### Documentation
- **Enhanced Dependency Management Guide:** Added comprehensive documentation in `CLAUDE.md` explaining:
  - Optional dependency management patterns and best practices
  - How to use `uv sync` for minimal installation vs. full installation with extras
  - Graceful handling of missing optional dependencies in the codebase
  - Torch/nvidia dependency chain through optional packages
- **Comprehensive Reranking Documentation:** Added detailed guide for document reranking in `docs/reranking.md`

### Internal Changes
- Updated UV lock file to reflect the new dependency versions
- Enhanced project documentation for better developer experience
- Added dedicated reranking cache to the cache manager system

## Version 2.4.1 (2025-07-04)

### Bug Fixes & Improvements
- **Grammar-Based Generation:** Replaced Outlines with llama.cpp's native GBNF grammar support for structured generation.
  - Resolves compatibility issues with Gemma-3n models and other models that had vocabulary processing errors
  - Provides better performance and reliability
  - No API changes - existing structured generation code continues to work unchanged
  - Added new `core/grammar.py` module for JSON schema to GBNF conversion
  - Removed `outlines` dependency from the project

### New Features  
- **PostgreSQL Structured Generation:** Added structured output support to the PostgreSQL extension.
  - New SQL functions: `steadytext_generate_json()`, `steadytext_generate_regex()`, `steadytext_generate_choice()`
  - Full integration with the same grammar-based approach as the main library
  - Includes fallback methods for when SteadyText is unavailable
  - All structured functions support caching with schema/pattern/choices included in cache keys

### Internal Changes
- Implemented `GrammarConverter` class for converting JSON schemas, regex patterns, and choice lists to GBNF
- Updated `StructuredGenerator` to use llama-cpp-python's `grammar` parameter directly
- Enhanced PostgreSQL extension's `daemon_connector.py` with structured generation methods

## Version 2.4.0 (2025-07-03)

### New Features
- **Structured Generation:** Introduced structured generation capabilities using Outlines library.
  - Generate JSON output conforming to a JSON schema or Pydantic model
  - Constrain output to specific regular expression patterns
  - Limit output to a predefined list of choices
  - Support for basic Python types (int, float, bool, str)
  - New API functions: `generate_json()`, `generate_regex()`, `generate_choice()`, and `generate_format()`
  - New parameters for `generate()`: `schema`, `regex`, `choices`, and `response_format`
  - Two-phase generation approach: reasoning followed by structured output
  - Full integration with daemon mode and caching system
  - Comprehensive examples in `examples/structured_generation.py`
  - Added `outlines>=1.0.3` as a new dependency

### Documentation
- Added comprehensive structured generation documentation in `docs/structured-generation.md`
- Added structured generation examples showcasing all features
- Updated API documentation with new structured generation parameters

### Known Issues
- Some models (Gemma-3n, Qwen1.5, Phi-2, Llama 3.x) have vocabulary compatibility issues with Outlines 1.0.3+
- Tracked in: https://github.com/outlines-dev/outlines/issues/820

## Version 2.3.0 (2025-07-03)

### New Features
- **Context Window Management:** Added dynamic context window sizing and input validation.
  - Automatically uses the largest context window supported by each model
  - Input length validation before generation to prevent mid-generation failures
  - Raises `ContextLengthExceededError` with detailed token counts when input is too long
  - Support for environment variable override via `STEADYTEXT_MAX_CONTEXT_WINDOW`
  - Token counting using model's tokenizer with fallback to estimation
  - Safety margins and output token reservation (default: 512 tokens + 10% margin)
  - Maintains deterministic behavior across different context window sizes
  - Added `get_optimal_context_window()` function for automatic context sizing
  - Comprehensive test suite for context window features

### Bug Fixes
- Fixed PostgreSQL extension embed connector functionality
- Applied formatting and lint fixes across the codebase

### Internal Changes
- Added `steadytext/exceptions.py` with new `ContextLengthExceededError` exception
- Enhanced model loader with context window configuration
- Updated generator with input validation and token counting

## Version 2.2.0 (2025-06-30)

### New Features
- **Pluggable Cache Backend System:** Added support for multiple cache backends with a factory pattern:
  - **SQLite Backend** (default): Thread-safe local storage with WAL mode
  - **D1 Backend**: Cloudflare's distributed SQLite for edge deployments
  - **Memory Backend**: In-memory cache for testing/ephemeral workloads
- **Cache Backend Configuration:** Environment variables for backend selection and configuration:
  - `STEADYTEXT_CACHE_BACKEND` to select backend type
  - D1-specific configuration (`STEADYTEXT_D1_API_URL`, `STEADYTEXT_D1_API_KEY`)
- **PostgreSQL Extension Improvements:** Enhanced pg_steadytext with daemon connectivity and better error handling
- **Cloudflare Workers Integration:** Added D1 cache proxy worker for distributed caching scenarios

### Architecture Improvements
- **Cache Factory Pattern:** Unified cache backend interface for consistent behavior across all backends
- **Enhanced Documentation:** New documentation structure with dedicated pages for architecture, deployment, and integrations
- **Test Coverage:** Added comprehensive tests for all cache backends and PostgreSQL extension

### Bug Fixes
- **PostgreSQL Path Configuration:** Fixed SQL syntax error in pg_steadytext extension initialization
- **Test Suite Improvements:** Fixed pytest skip usage and enhanced test reliability
- **Type Safety:** Improved typechecker compliance across test files

### Documentation
- Added architecture overview documentation
- Added cache backends configuration guide
- Added deployment and integration guides
- Enhanced FAQ and migration documentation

## Version 2.1.1 (2025-06-30)

### Bug Fixes
- **Fixed Llama CPP Fork:** Switched to the `inference-sh` fork of `llama-cpp-python` to resolve build issues and ensure compatibility with the latest GGUF models.

## Version 2.1.0 (2025-06-29)

### New Features
- **Custom Seed Support:** Added support for custom seed parameter in generation and embedding functions for enhanced deterministic control.

### Bug Fixes
- Various stability improvements and minor fixes.

## Version 2.0.4 (2025-06-28)

### Bug Fixes
- Documentation updates and code formatting improvements.
- Fixed various linting and type checking issues.

## Version 2.0.3 (2025-06-28)

### Bug Fixes
- Minor bug fixes and performance improvements.

## Version 2.0.2 (2025-06-28)

### Bug Fixes
- Fixed model loading and caching issues.

## Version 2.0.1 (2025-06-28)

### Bug Fixes
- **Fixed Model Repository:** Updated Gemma-3n model repository from `ggml-org` to `ggml-org` which hosts the latest GGUF versions
  - E2B model: Now uses `ggml-org/gemma-3n-E2B-it-GGUF` with filename `gemma-3n-E2B-it-Q8_0.gguf`
  - E4B model: Now uses `ggml-org/gemma-3n-E4B-it-GGUF` with filename `gemma-3n-E4B-it-Q8_0.gguf`

## Version 2.0.0 (2025-06-28)

### Major Changes
- **Switched to Gemma-3n:** The default generation model is now `gemma-3n-E2B-it-GGUF` (ggml-org/gemma-3n-E2B-it-GGUF).
- **Changed Default Model Size:** Default model changed from Gemma-3n-4B to Gemma-3n-2B for faster generation while maintaining quality.
- **Deprecated Thinking Mode:** The `thinking_mode` parameter has been removed from all functions and the CLI. Temperature=0 deterministic generation works better without thinking mode.
- **Model Registry Update:** Updated to focus on Gemma-3n models (2B and 4B variants).

### New Features
- **Configurable Generation Length:** Added `max_new_tokens` parameter to `generate()` and `generate_iter()` functions to control output length.
- **CLI Support:** Added `--max-new-tokens` flag to CLI for controlling generation length.

### Configuration Changes
- Reduced default context window from 3072 to 2048 tokens.
- Reduced default max new tokens for generation from 1024 to 512.
- Embedding model remains `Qwen3-Embedding-0.6B-GGUF` with 1024 dimensions.

### Breaking Changes
- Removed `thinking_mode` parameter from `generate()`, `generate_iter()`, and CLI
- Removed `--think` flag from CLI
- Changed default generation model from Qwen3-1.7B to Gemma-3n-E2B
- Changed default model size from "large" (4B) to "small" (2B)

## Version 1.3.5 (2025-06-23)

- Minor bug fixes and performance improvements.

## Version 1.3.3 (2025-06-20)

### New Features
- **Vector Index Support:** Added FAISS-based vector indexing for RAG applications
  - CLI commands: `st index create`, `st index search`, `st index info`
  - Deterministic text chunking with chonkie
  - Automatic context retrieval when `default.faiss` exists
  - Integration with text generation via `--index-file` flag
  - Caching of index search results for deterministic retrieval
- **SQLite Concurrent Cache:** Replaced pickle-based cache with SQLite for thread-safe operations
  - WAL mode for optimal concurrent performance
  - Automatic migration from legacy pickle format
  - Microsecond precision timestamps for accurate frecency ordering
  - Graceful handling of corrupted databases
- **CLI Enhancements:**
  - Added `--quiet` flag to suppress informational output
  - Added `--eos-string` parameter for custom end-of-sequence strings
  - Added shell completion support via `st completion --install`
  - Added `cache path` and `cache status` commands
  - Added `models list` command to show available models
  - Multiple text input support for embed command

### PostgreSQL Extension Updates
- **AI Summarization Aggregate Functions:** 
  - `ai_summarize()` aggregate for intelligent text summarization
  - `ai_summarize_partial()` and `ai_summarize_final()` for TimescaleDB continuous aggregates
  - `ai_extract_facts()` for structured fact extraction
  - `ai_deduplicate_facts()` for semantic deduplication
- **Async Function Support:** Added async variants of all generation and embedding functions
  - Queue-based processing with priority support
  - LISTEN/NOTIFY integration
  - Batch operations for efficiency
- **Performance Improvements:**
  - Functions marked as PARALLEL SAFE, LEAKPROOF, and IMMUTABLE
  - Better query optimization with PostgreSQL planner

### Internal Changes
- Thread-local database connections for cache performance
- Comprehensive concurrent access test coverage
- Enhanced error handling and recovery mechanisms

## Version 1.4.0 (2025-06-22)

### PostgreSQL Extension Features
- **Automatic Cache Eviction with pg_cron:** Added scheduled cache management
  - `steadytext_setup_cache_eviction()` function for automatic configuration
  - `steadytext_evict_cache()` for manual eviction with custom parameters
  - Configurable eviction intervals, age limits, and size targets
  - Integration with pg_cron for scheduled execution
  - Monitoring support via cron.job and cron.job_run_details tables
  - Performance index for optimal frecency-based queries

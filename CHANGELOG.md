# Changelog

## Version 2.5.1 (2025-07-14)

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

### Internal Changes
- Updated UV lock file to reflect the new dependency versions
- Enhanced project documentation for better developer experience

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

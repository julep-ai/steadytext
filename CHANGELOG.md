# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-06-23

### Added
- New `vector` CLI command group for vector operations on embeddings
- `vector similarity` - Compute cosine or dot product similarity between texts
- `vector distance` - Calculate euclidean, manhattan, or cosine distance
- `vector search` - Find most similar texts from a list of candidates
- `vector average` - Compute average of multiple text embeddings
- `vector arithmetic` - Perform vector arithmetic (addition/subtraction)
- Support for stdin input and JSON output in all vector commands
- Comprehensive documentation for vector operations
- New `index` CLI command group for FAISS index management
- `index create` - Create deterministic FAISS indices from text files
- `index info` - Display index statistics and metadata
- `index search` - Search for similar chunks in an index
- Automatic context retrieval in `generate` command when index exists
- Deterministic text chunking with chonkie library (512 token default)
- FAISS-based vector storage with exact search (IndexFlatL2)
- Index search result caching for deterministic retrieval
- `--no-index`, `--index-file`, and `--top-k` options for generate command

### Dependencies
- Added `chonkie>=0.2.1` for deterministic text chunking
- Added `faiss-cpu>=1.7.0` for vector index storage and search

## [0.2.3] - 2025-06-19

### Added
- Custom EOS (End-of-Sequence) string support for text generation
- `eos_string` parameter in both `generate()` and `generate_iter()` functions
- `--eos-string` CLI parameter for custom stop sequences
- `--quiet` / `-q` CLI flag to silence logging output
- Enhanced caching logic that includes eos_string in cache keys
- Comprehensive test coverage for EOS string functionality

### Changed
- Updated Python API signatures to include `eos_string` parameter (default: "[EOS]")
- Enhanced CLI generate command with custom stop sequence support
- Improved logging control with quiet mode for production use

## [0.2.2] - 2025-06-19

### Added
- Comprehensive CLI interface with both `steadytext` and `st` commands
- Text generation with stdin support and multiple output formats (raw, JSON, streaming)
- Embedding generation with format options (JSON, numpy, hex)
- Cache management commands (stats, clear, export, import)
- Model management commands (status, download, path)
- Pipeline support for Unix-style workflows
- SQLite-based concurrent disk-backed frecency cache
- Thread-safe and process-safe cache operations
- Automatic migration from legacy pickle format to SQLite
- Examples directory with practical use cases

### Changed
- Replaced pickle-based cache with SQLite for safe concurrent access
- Enhanced README with better examples and documentation
- Improved CI/CD workflows and testing infrastructure
- Updated model configurations and caching behavior

### Fixed
- Eliminated file-level race conditions and corruption issues
- Improved cache sharing across multiple processes/threads
- Enhanced reliability for production concurrent workloads

## [0.1.0] - 2025-06-16

### Added
- Initial release of SteadyText library
- Core deterministic text generation functionality
- Deterministic embedding generation with L2 normalization
- GGUF model downloading from Hugging Face Hub
- Thread-safe model loading with singleton pattern
- Deterministic fallback functions for robust error handling
- Comprehensive test suite with model-dependent and fallback testing
- Support for openbmb.BitCPM4-1B.Q8_0.gguf for text generation
- Support for Qwen3-Embedding-0.6B-Q8_0.gguf for embeddings
- Cross-platform cache directory management
- Environment-based configuration for cache limits
- Anchor comment system for AI-assisted development

### Technical Details
- Uses deterministic sampling parameters with fixed seed (42)
- Always returns 1024-dimensional L2-normalized embeddings
- Graceful degradation when models cannot be loaded
- Zero-configuration setup with automatic model downloads
- "Never Fails" architecture - all functions return deterministic outputs

## [Initial] - 2025-06-11

### Added
- Project foundation and structure
- MIT license
- Basic Python package configuration
- Development tooling setup (pre-commit, CI/CD workflows)
- Initial documentation and README
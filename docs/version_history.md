# Version History

This document outlines the major versions of SteadyText and the key features introduced in each.

**Latest Version**: 2.1.0+ - Custom Seeds & PostgreSQL Extension

| Version | Key Features                                                                                                                            | Default Generation Model                               | Default Embedding Model                                | Python Versions |
| :------ | :-------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- | :----------------------------------------------------- | :-------------- |
| **2.1.x** | - **Custom Seeds**: Added seed parameter to all generation and embedding functions.<br>- **PostgreSQL Extension**: Released pg_steadytext extension.<br>- **Enhanced Reproducibility**: Full control over deterministic generation. | `ggml-org/gemma-3n-E2B-it-GGUF` (gemma-3n-E2B-it-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | `>=3.10, <3.14` |
| **2.0.x** | - **Daemon Mode**: Persistent model serving with ZeroMQ.<br>- **Gemma-3n Models**: Switched to `gemma-3n` for generation.<br>- **Thinking Mode Deprecated**: Removed thinking mode. | `ggml-org/gemma-3n-E2B-it-GGUF` (gemma-3n-E2B-it-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | `>=3.10, <3.14` |
| **1.x** | - **Model Switching**: Added support for switching models via environment variables and a model registry.<br>- **Qwen3 Models**: Switched to `qwen3-1.7b` for generation.<br>- **Indexing**: Added support for FAISS indexing. | `Qwen/Qwen3-1.7B-GGUF` (Qwen3-1.7B-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | `>=3.10, <3.14` |
| **0.x** | - **Initial Release**: Deterministic text generation and embedding.                                                                      | `Qwen/Qwen1.5-0.5B-Chat-GGUF` (qwen1_5-0_5b-chat-q4_k_m.gguf) | `Qwen/Qwen1.5-0.5B-Chat-GGUF` (qwen1_5-0_5b-chat-q8_0.gguf) | `>=3.10`        |

## Detailed Release Notes

### Version 2.1.0+ - Custom Seeds & PostgreSQL Extension

**Release Date**: June 2025

#### 🎯 Custom Seed Support

**Major Enhancement**: Added comprehensive custom seed support across all SteadyText APIs.

- **Python API**: All functions now accept optional `seed: int = DEFAULT_SEED` parameter
  - `steadytext.generate(prompt, seed=123)`
  - `steadytext.generate_iter(prompt, seed=456)`
  - `steadytext.embed(text, seed=789)`

- **CLI Support**: Added `--seed` flag to all commands
  - `st generate "prompt" --seed 123`
  - `st embed "text" --seed 456`
  - `st vector similarity "text1" "text2" --seed 789`

- **Daemon Integration**: Seeds are properly passed through daemon protocol
- **Fallback Behavior**: Deterministic fallbacks now respect custom seeds
- **Cache Keys**: Seeds are included in cache keys to prevent collisions

**Use Cases**:
- **Reproducible Research**: Document and reproduce exact results
- **A/B Testing**: Generate controlled variations of content
- **Experimental Design**: Systematic exploration of model behavior
- **Content Variations**: Create different versions while maintaining quality

#### 🐘 PostgreSQL Extension (pg_steadytext)

**New Release**: Complete PostgreSQL extension for SteadyText integration.

**Core Features**:
- **SQL Functions**: Native PostgreSQL functions for text generation and embeddings
  - `steadytext_generate(prompt, max_tokens, use_cache, seed)`
  - `steadytext_embed(text, use_cache, seed)`
  - `steadytext_daemon_start()`, `steadytext_daemon_status()`, `steadytext_daemon_stop()`

- **Vector Integration**: Full compatibility with pgvector extension
- **Built-in Caching**: PostgreSQL-based frecency cache with eviction
- **Daemon Support**: Integrates with SteadyText's ZeroMQ daemon for performance
- **Configuration Management**: SQL-based configuration with `steadytext_config` table

**Installation**:
```bash
# Install Python dependencies
pip3 install steadytext>=2.1.0

# Build and install extension
git clone https://github.com/julep-ai/steadytext.git
cd steadytext/pg_steadytext
make && sudo make install

# Enable in PostgreSQL
psql -c "CREATE EXTENSION pg_steadytext CASCADE;"
```

**Docker Support**:
```bash
# Standard build
docker build -t pg_steadytext .

# With fallback model for compatibility
docker build --build-arg STEADYTEXT_USE_FALLBACK_MODEL=true -t pg_steadytext .
```

#### 🔧 Technical Improvements

- **Validation**: Added `validate_seed()` function for input validation
- **Environment Setup**: Enhanced `set_deterministic_environment()` with custom seeds
- **Error Handling**: Improved error messages and fallback behavior
- **Documentation**: Comprehensive documentation and examples

#### 📖 Documentation Updates

- **API Documentation**: Updated all function signatures with seed parameters
- **CLI Reference**: Added `--seed` flag documentation for all commands
- **Examples**: New comprehensive examples for custom seed usage
- **PostgreSQL Guide**: Complete integration guide for pg_steadytext
- **Migration Guide**: Instructions for upgrading from previous versions

#### 🔄 Breaking Changes

**None** - Version 2.1.0+ is fully backward compatible with 2.0.x. All existing code continues to work unchanged, with new seed parameters being optional.

#### 🐛 Bug Fixes

- Fixed cache key generation to include seed for proper isolation
- Improved daemon protocol to handle seed parameters correctly
- Enhanced fallback behavior to be deterministic with custom seeds
- Resolved edge cases in streaming generation with custom seeds

#### 📋 Requirements

- **Python**: 3.10+ (unchanged)
- **PostgreSQL**: 14+ (for pg_steadytext extension)
- **Dependencies**: All existing dependencies remain compatible

---

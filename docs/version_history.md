# Version History

This document outlines the major versions of SteadyText and the key features introduced in each.

**Latest Version**: 2.4.1 - Structured Generation with Native Grammar Support

| Version | Key Features                                                                                                                            | Default Generation Model                               | Default Embedding Model                                | Default Reranking Model | Python Versions |
| :------ | :-------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- | :----------------------------------------------------- | :---------------------- | :-------------- |
| **2.4.x** | - **Native Grammar Support**: Replaced Outlines with llama.cpp's native GBNF grammars for structured generation.<br>- **PostgreSQL Structured Generation**: Added `steadytext_generate_json()`, `steadytext_generate_regex()`, `steadytext_generate_choice()` SQL functions.<br>- **Better Compatibility**: Fixes issues with Gemma-3n and other models. | `ggml-org/gemma-3n-E2B-it-GGUF` (gemma-3n-E2B-it-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | `Qwen/Qwen3-Reranker-4B-GGUF` (Qwen3-Reranker-4B-Q8_0.gguf) | `>=3.10, <3.14` |
| **2.3.x** | - **Document Reranking**: Added reranking functionality with `Qwen3-Reranker-4B` model.<br>- **Structured Generation**: Added support for JSON, Regex, and Choice-constrained generation via `outlines`.<br>- **New API parameters**: `schema`, `regex`, `choices` added to `generate()`.<br>- **New convenience functions**: `generate_json()`, `generate_regex()`, `generate_choice()`. | `ggml-org/gemma-3n-E2B-it-GGUF` (gemma-3n-E2B-it-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | `Qwen/Qwen3-Reranker-4B-GGUF` (Qwen3-Reranker-4B-Q8_0.gguf) | `>=3.10, <3.14` |
| **2.1.x** | - **Custom Seeds**: Added seed parameter to all generation and embedding functions.<br>- **PostgreSQL Extension**: Released pg_steadytext extension.<br>- **Enhanced Reproducibility**: Full control over deterministic generation. | `ggml-org/gemma-3n-E2B-it-GGUF` (gemma-3n-E2B-it-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | - | `>=3.10, <3.14` |
| **2.0.x** | - **Daemon Mode**: Persistent model serving with ZeroMQ.<br>- **Gemma-3n Models**: Switched to `gemma-3n` for generation.<br>- **Thinking Mode Deprecated**: Removed thinking mode. | `ggml-org/gemma-3n-E2B-it-GGUF` (gemma-3n-E2B-it-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | - | `>=3.10, <3.14` |
| **1.x** | - **Model Switching**: Added support for switching models via environment variables.<br>- **Centralized Cache**: Unified cache system with SQLite backend.<br>- **CLI Improvements**: Streaming by default, quiet output, new pipe syntax. | `Qwen/Qwen3-1.7B-GGUF` (Qwen3-1.7B-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | - | `>=3.10, <3.14` |
| **0.x** | - **Initial Release**: Deterministic text generation and embedding.                                                                      | `Qwen/Qwen1.5-0.5B-Chat-GGUF` (qwen1_5-0_5b-chat-q4_k_m.gguf) | `Qwen/Qwen1.5-0.5B-Chat-GGUF` (qwen1_5-0_5b-chat-q8_0.gguf) | - | `>=3.10`        |

## Detailed Release Notes

### Version 2.4.1 - Native Grammar Support

**Release Date**: July 2025

#### 🔧 Grammar-Based Structured Generation

**Major Improvement**: Replaced Outlines with llama.cpp's native GBNF (Grammatical Backus-Naur Form) grammar support.

**Benefits**:
- **Better Compatibility**: Fixes vocabulary processing errors with Gemma-3n, Qwen1.5, Phi-2, and Llama 3.x models
- **Improved Performance**: Native integration with llama.cpp eliminates external library overhead
- **No API Changes**: Existing structured generation code continues to work unchanged
- **Deterministic Output**: Grammar-based generation maintains SteadyText's determinism guarantees

**Technical Details**:
- New `core/grammar.py` module converts JSON schemas, regex patterns, and choice lists to GBNF
- `StructuredGenerator` now uses llama-cpp-python's `grammar` parameter directly
- Removed `outlines` dependency, simplifying the dependency tree

#### 🐘 PostgreSQL Structured Generation

**New Feature**: Added structured generation support to the PostgreSQL extension.

**New SQL Functions**:
- `steadytext_generate_json(prompt, schema)` - Generate JSON conforming to a schema
- `steadytext_generate_regex(prompt, pattern)` - Generate text matching a regex
- `steadytext_generate_choice(prompt, choices)` - Generate one of the provided choices

**Example Usage**:
```sql
-- Generate structured JSON
SELECT steadytext_generate_json(
    'Create a person named Alice',
    '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'::jsonb
);

-- Generate text matching a pattern
SELECT steadytext_generate_regex(
    'My phone number is',
    '\d{3}-\d{3}-\d{4}'
);

-- Generate from choices
SELECT steadytext_generate_choice(
    'Is Python good?',
    ARRAY['yes', 'no', 'maybe']
);
```

### Version 2.3.0 - Document Reranking & Structured Generation

**Release Date**: July 2025

#### 🔍 Document Reranking

**Major Feature**: Added document reranking functionality powered by the Qwen3-Reranker-4B model.

- **Python API**: New `steadytext.rerank()` function with customizable task descriptions
  - `steadytext.rerank(query, documents, task="custom search task")`
  - Support for both single document and list of documents
  - Optional score returning with `return_scores` parameter
- **CLI Command**: `st rerank` for command-line reranking operations
  - `st rerank "query" doc1.txt doc2.txt --top-k 5`
- **Fallback Support**: Simple word overlap scoring when model unavailable
- **Dedicated Cache**: Separate frecency cache for reranking results

#### ✨ Structured Generation

**Major Feature**: Introduced structured generation capabilities powered by the [Outlines](https://github.com/outlines-dev/outlines) library.

- **JSON Generation**: Generate JSON that conforms to a JSON schema or a Pydantic model.
  - `steadytext.generate(prompt, schema=MyPydanticModel)`
  - `steadytext.generate_json(prompt, schema={"type": "object", ...})`
- **Regex-Constrained Generation**: Force output to match a regular expression.
  - `steadytext.generate(prompt, regex=r"\d{3}-\d{3}-\d{4}")`
- **Multiple Choice**: Force model to choose from a list of options.
  - `steadytext.generate(prompt, choices=["A", "B", "C"])`

**Use Cases**:
- Reliable data extraction
- Building robust function-calling systems
- Creating predictable application logic
- Generating structured data for databases

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

---

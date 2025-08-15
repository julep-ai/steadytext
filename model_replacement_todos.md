# Model Replacement Tasks

## Phase 1: Core Model Configuration Updates

### 1.1 Update Model Constants in utils.py
- [ ] Replace DEFAULT_GENERATION_MODEL_REPO with two new constants for small and large models
- [ ] Replace DEFAULT_EMBEDDING_MODEL_REPO with Jina v4 model
- [ ] Update GENERATION_MODEL_FILENAME for both small and large models
- [ ] Update EMBEDDING_MODEL_FILENAME for Jina model
- [ ] Remove DEFAULT_RERANKING_MODEL_REPO (keeping existing for now)
- [ ] Update MODEL_REGISTRY with new Qwen3 models
- [ ] Update SIZE_TO_MODEL mapping

### 1.2 Embedding Dimension Changes
- [ ] Update EMBEDDING_DIMENSION from 1024 to 2048 (Jina v4 default)
- [ ] Add support for Matryoshka truncation dimensions [128, 256, 512, 1024, 2048]
- [ ] Update validate_normalized_embedding to handle new dimension

### 1.3 Add Query/Passage Prefix Support
- [ ] Modify core/embedder.py to add "Query:" or "Passage:" prefix
- [ ] Add parameter to embed() function to specify query vs passage mode
- [ ] Update DeterministicEmbedder class to handle prefixes
- [ ] Ensure prefix is added before caching (part of cache key)

## Phase 2: Model Loading and Compatibility

### 2.1 Update Model Loader
- [ ] Modify get_generation_model_path to handle small/large model selection
- [ ] Update get_embedding_model_path for Jina model
- [ ] Add size parameter handling for generation models
- [ ] Update model compatibility checks for new models

### 2.2 Update Cache Download Logic
- [ ] Ensure cache.py can download from direct URLs (not just HF repo format)
- [ ] Add logic to handle the download=true parameter in URLs
- [ ] Update model path resolution for direct URL downloads

## Phase 3: API and Interface Updates

### 3.1 Update Public API
- [ ] Modify generate() function to accept size="small" or size="large" parameter
- [ ] Update embed() function to accept mode="query" or mode="passage" parameter
- [ ] Ensure backward compatibility with existing code

### 3.2 CLI Updates
- [ ] Update cli/commands/generate.py to support --size flag
- [ ] Update cli/commands/embed.py to support --mode flag
- [ ] Update help text and documentation

### 3.3 Daemon Updates
- [ ] Update daemon server to handle new model parameters
- [ ] Update daemon client to pass size and mode parameters
- [ ] Ensure daemon can load both small and large models

## Phase 4: PostgreSQL Extension Updates

### 4.1 SQL Function Updates
- [ ] Update steadytext_generate to accept size parameter
- [ ] Update steadytext_embed to accept mode parameter
- [ ] Update Python connector functions accordingly

### 4.2 Migration Scripts
- [ ] Create migration SQL for existing installations
- [ ] Update version to reflect new models

## Phase 5: Documentation Updates

### 5.1 Core Documentation
- [ ] Update README.md with new model information
- [ ] Update CLAUDE.md to reflect Qwen3 models instead of Gemma-3n
- [ ] Update CHANGELOG.md with version notes

### 5.2 API Documentation
- [ ] Update docs/api.md with new parameters
- [ ] Update docs/architecture.md with new model details
- [ ] Update docs/benchmarks.md if needed

### 5.3 Migration Guide
- [ ] Create migration guide for users upgrading from Gemma/Qwen2.5 models
- [ ] Document embedding dimension change (1024 -> 2048)
- [ ] Document Query/Passage prefix requirement

## Phase 6: Testing and Validation

### 6.1 Update Tests
- [ ] Update test fixtures for new embedding dimension
- [ ] Add tests for query/passage prefix functionality
- [ ] Add tests for small/large model selection
- [ ] Update benchmark tests

### 6.2 Compatibility Testing
- [ ] Test backward compatibility with existing code
- [ ] Test cache compatibility (may need cache clearing)
- [ ] Test daemon mode with new models

## Phase 7: Environment and Configuration

### 7.1 Environment Variables
- [ ] Add STEADYTEXT_GENERATION_SIZE for default size selection
- [ ] Add STEADYTEXT_EMBEDDING_MODE for default query/passage mode
- [ ] Update existing env var documentation

### 7.2 Fallback Handling
- [ ] Update fallback model logic for new models
- [ ] Ensure graceful degradation if models unavailable

## Implementation Notes

### Model URLs and Details:
- **Large Generation Model**: Qwen3-30B-A3B-Instruct (Q4_K_XL quantization)
  - URL: https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF
  - File: Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf
  
- **Small Generation Model**: Qwen3-4B-Instruct (Q6_K_XL quantization)
  - URL: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF
  - File: Qwen3-4B-Instruct-2507-UD-Q6_K_XL.gguf
  
- **Embedding Model**: Jina Embeddings v4 Text Retrieval (Q5_K_S quantization)
  - URL: https://huggingface.co/jinaai/jina-embeddings-v4-text-retrieval-GGUF
  - File: jina-embeddings-v4-text-retrieval-Q5_K_S.gguf
  - Requires "Query:" or "Passage:" prefix
  - Default dimension: 2048 (supports truncation to 128, 256, 512, 1024)

### Breaking Changes:
1. Embedding dimension changes from 1024 to 2048
2. Embeddings now require Query/Passage prefix
3. Generation model selection via size parameter
4. Cache invalidation due to model changes

### Migration Strategy:
1. Bump version to 2025.8.16 (or appropriate date)
2. Clear existing model cache on first run
3. Provide clear migration documentation
4. Consider compatibility layer for old embedding dimension
# Migration Assessment: llama-cpp-python to xllamacpp

**Date**: 2025-08-15  
**Author**: Claude (AI Assistant)  
**Status**: Assessment Only - Migration NOT Recommended

## Executive Summary

Switching from llama-cpp-python to xllamacpp would require a **complete rewrite** of SteadyText's core functionality, taking an estimated **3-4 weeks** for an experienced developer with **HIGH risk** and **MEDIUM-HIGH complexity**.

## Effort and Complexity Assessment

- **Overall Effort**: HIGH (3-4 weeks for experienced developer)
- **Complexity**: MEDIUM-HIGH
- **Risk Level**: HIGH
- **Breaking Changes**: Yes - Complete API change for all users

## Key Architectural Differences

### 1. API Architecture (HIGH COMPLEXITY)
| Aspect | llama-cpp-python | xllamacpp |
|--------|-----------------|-----------|
| Wrapper Type | ctypes | Cython |
| API Style | Direct Llama class | Server & CommonParams |
| Model Loading | Simple instantiation | Parameter-based configuration |
| Thread Safety | Limited | Built-in |

### 2. Model Loading Pattern
```python
# Current (llama-cpp-python)
model = Llama(
    model_path=str(path),
    n_ctx=4096,
    seed=42,
    **params
)

# xllamacpp approach
params = xlc.CommonParams()
params.model.path = "model.gguf"
params.n_ctx = 4096
params.sampling.seed = 42
server = xlc.Server(params)
```

### 3. Text Generation
```python
# Current (llama-cpp-python)
result = model(prompt, max_tokens=512, stop=["</s>"])
text = result["choices"][0]["text"]

# xllamacpp approach
server.handle_completions(
    json.dumps({"prompt": prompt, "max_tokens": 512}),
    on_chunk_callback,
    on_complete_callback
)
```

### 4. Embeddings
```python
# Current (llama-cpp-python)
embeddings = model.embed(texts)  # Returns numpy arrays directly

# xllamacpp approach
server.handle_embeddings(
    json.dumps({"input": texts}),
    on_error_callback,
    on_success_callback  # JSON response with embeddings
)
```

## Specific Components Requiring Rewrite

### Core Components (~1,300 lines)
1. **`steadytext/models/loader.py`** (300+ lines)
   - Complete rewrite of singleton model management
   - New caching strategy for Server instances
   - Thread-safe model access patterns

2. **`steadytext/core/generator.py`** (500+ lines)
   - Rewrite generation logic to use Server API
   - Handle async/callback-based responses
   - Adapt streaming generation
   - Token counting and validation

3. **`steadytext/core/embedder.py`** (200+ lines)
   - New embedding extraction logic
   - JSON response parsing
   - Dimension handling (1024 vs 2048)

4. **`steadytext/core/structured.py`** (300+ lines)
   - **CRITICAL**: May lose structured generation
   - Grammar support uncertain in xllamacpp
   - Would need alternative implementation

### Supporting Components (~500 lines)
5. **Daemon Mode** (`steadytext/daemon/`)
   - Complete redesign for xllamacpp's server architecture
   - Different IPC mechanism

6. **CLI Commands** (`steadytext/cli/commands/`)
   - Update all generation and embedding commands
   - New parameter handling

### Testing (~1,000+ lines)
7. **All test files**
   - Complete test suite rewrite
   - New mocking strategies
   - Different assertion patterns

## Major Technical Challenges

### 1. Structured Generation (CRITICAL)
- Current: Uses `LlamaGrammar.from_json_schema()` for JSON/Pydantic schemas
- xllamacpp: Grammar support unclear/different API
- **Risk**: May lose this feature entirely

### 2. Thread Safety & Concurrency
- Current: Managed through singleton with locks
- xllamacpp: Built-in thread safety but different patterns
- **Risk**: Race conditions during migration

### 3. Caching System
- Current: Cache keys based on llama-cpp-python internals
- xllamacpp: Would need new cache key generation
- **Risk**: Cache invalidation issues

### 4. Backward Compatibility
- **Breaking changes** for all existing code
- Python API would completely change
- PostgreSQL extension would need updates

### 5. Feature Parity
- Logprobs support uncertain
- Stop sequences handling different
- Temperature/sampling parameter names differ

## Cost-Benefit Analysis

### Potential Benefits
✅ Better thread safety (native support)  
✅ Continuous batching support  
✅ C++ server implementation (potentially faster)  
✅ Prebuilt wheels (easier installation)  
✅ Active development from Xorbits team  

### Costs and Risks
❌ 3-4 weeks development time  
❌ High risk of bugs/regressions  
❌ Possible loss of structured generation  
❌ Breaking API changes for all users  
❌ Extensive testing required (2+ weeks)  
❌ Documentation rewrite needed  
❌ PostgreSQL extension updates  
❌ Daemon mode redesign  
❌ Unknown edge cases  

## Migration Timeline (If Pursued)

| Phase | Duration | Tasks |
|-------|----------|-------|
| Planning | 3 days | Detailed API mapping, feature gap analysis |
| Core Migration | 10 days | Rewrite loader, generator, embedder |
| Feature Restoration | 5 days | Structured generation, special features |
| Testing | 7 days | Unit tests, integration tests, benchmarks |
| Documentation | 3 days | API docs, migration guide |
| **Total** | **28 days** | Plus buffer for unknowns |

## Alternatives to Migration

If performance is the main concern, consider these alternatives:

1. **Optimize Current Implementation**
   - Profile and optimize hot paths
   - Better batching strategies
   - Memory pool management

2. **Upgrade llama-cpp-python**
   - Stay on latest version
   - Use new features like server mode
   - Leverage performance improvements

3. **Hybrid Approach**
   - Keep llama-cpp-python for core features
   - Add xllamacpp for specific use cases
   - Gradual migration if needed

4. **Wait and Evaluate**
   - Monitor xllamacpp development
   - Wait for feature parity
   - Reassess in 6 months

## Recommendation

### ❌ **DO NOT MIGRATE** at this time

**Rationale:**
1. The effort (3-4 weeks) is not justified by the benefits
2. High risk of losing critical features (structured generation)
3. Would cause major disruption to users
4. Current implementation is stable and working
5. Performance gains uncertain and unquantified

**Suggested Action:**
- Continue with llama-cpp-python
- Monitor xllamacpp development
- Consider migration only if:
  - Critical features become available only in xllamacpp
  - Performance bottlenecks cannot be solved otherwise
  - llama-cpp-python becomes unmaintained

## AIDEV-NOTE: Migration Complexity Points

- AIDEV-NOTE: The Server/CommonParams API is fundamentally different from Llama class API
- AIDEV-NOTE: Callback-based async pattern vs synchronous calls would affect entire codebase
- AIDEV-NOTE: Grammar/structured generation is critical for JSON output - unclear if supported
- AIDEV-NOTE: Cython vs ctypes means different error handling and debugging approaches
- AIDEV-TODO: If migration considered, create proof-of-concept with structured generation first
- AIDEV-TODO: Benchmark both libraries with identical models before committing to migration
- AIDEV-QUESTION: Can xllamacpp support all sampling parameters we currently use?
- AIDEV-QUESTION: How would streaming generation work with callback-based API?
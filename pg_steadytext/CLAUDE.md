# AIDEV Notes for pg_steadytext

This file contains important development notes and architectural decisions for AI assistants working on pg_steadytext.

## Recent Fixes

### v1.4.5 - Version Bump and Library Update
- **Updated**: SteadyText library dependency to >= 2.6.1
- **Version**: Extension version bumped to 1.4.5

### v1.4.4 - Extended Model Parameters, Unsafe Mode, and Short Aliases (Updated)
- **Added**: Support for additional generation parameters:
  - `eos_string`: End-of-sequence string (default: '[EOS]')
  - `model`: Specific model to use
  - `model_repo`: Model repository
  - `model_filename`: Model filename
  - `size`: Model size specification
  - `unsafe_mode`: Allow remote models when TRUE (default: FALSE)
- **Added**: Automatic short aliases for all functions (`st_*` for `steadytext_*`)
  - Examples: `st_generate()`, `st_embed()`, `st_generate_json()`, etc.
  - Aliases preserve all function properties (IMMUTABLE, PARALLEL SAFE, etc.)
  - Created dynamically during migration to catch all current and future functions
- **Added**: Missing `steadytext_generate_async()` function and async aliases
  - Function was referenced but never implemented in earlier versions
  - Added `st_generate_async`, `st_rerank_async`, `st_check_async`, etc.
- **Security**: Remote models (containing ':' in name) require `unsafe_mode=TRUE`
- **Fixed**: Upgrade script pattern for changing function signatures
- AIDEV-NOTE: Cache key includes eos_string when non-default
- AIDEV-NOTE: Added validation to prevent remote model usage without explicit unsafe_mode flag
- AIDEV-NOTE: When changing function signatures in upgrades, use ALTER EXTENSION DROP/ADD pattern:
  ```sql
  ALTER EXTENSION pg_steadytext DROP FUNCTION old_signature;
  DROP FUNCTION IF EXISTS old_signature;
  CREATE OR REPLACE FUNCTION new_signature...;
  ALTER EXTENSION pg_steadytext ADD FUNCTION new_signature;
  ```
- AIDEV-NOTE: Aliases must be created manually to preserve default parameters:
  ```sql
  -- Manual creation preserves DEFAULT clauses
  CREATE FUNCTION st_generate(
    prompt TEXT,
    max_tokens INT DEFAULT NULL,
    ...
  ) RETURNS TEXT LANGUAGE sql AS $$ 
    SELECT steadytext_generate($1, $2, ...); 
  $$;
  ```
  Dynamic generation would lose default values, requiring all parameters.
- **Fixed**: Remote model performance issue (2025-08-01):
  - SQL function now skips daemon checks entirely for remote models (containing ':')
  - `is_daemon_running()` now uses lightweight ZMQ ping instead of model loading
  - Prevents unnecessary delays when using OpenAI or other remote models with unsafe_mode
  - AIDEV-NOTE: Remote models go directly to steadytext.generate() without daemon involvement

### v1.4.3 - Parameter Naming
- **Fixed**: `max_tokens` → `max_new_tokens` in direct generation fallback
- AIDEV-NOTE: Daemon API uses `max_tokens`, direct Python API uses `max_new_tokens`

### v1.4.2 - Public Methods
- **Fixed**: Added public `start_daemon()`, `is_daemon_running()`, `check_health()` methods

## Security Fixes (v1.0.2)

1. **SQL Injection**: Added table name validation in cache_manager.py
2. **Missing Methods**: Added daemon status methods to connector
3. **Cache Keys**: Aligned with SteadyText format for compatibility
4. **Rate Limiting**: Implemented sliding window with SQL atomicity
5. **Input Validation**: Added host/port validation in daemon_connector
6. **Code Cleanup**: Removed unused SAFE_TEXT_PATTERN

### Future Work

- AIDEV-TODO: Bidirectional cache sync, ZeroMQ pooling, prepared statement caching
- AIDEV-TODO: Enhanced prompt validation and injection detection
- AIDEV-QUESTION: Multiple daemon instances for load balancing?

## pgTAP Testing Framework (v1.0.3)

- AIDEV-NOTE: Uses pgTAP for TAP output, rich assertions, transaction safety

**Run tests:** `make test-pgtap` or `./run_pgtap_tests.sh test/pgtap/01_basic.sql`

**Key functions:** `plan()`, `has_function()`, `is()`, `throws_ok()`, etc.

## v1.0.1 Fixes

1. **Removed thinking_mode**: Not supported by core library
2. **Python Init**: On-demand initialization in each function
3. **Docker Optimization**: Layer ordering for better caching
4. **Model Compatibility**: Gemma-3n issues with inference-sh fork, added Qwen fallback

## Python Version Constraints

- AIDEV-NOTE: plpython3u is compiled against specific Python version - cannot change at runtime
- **Solution**: Custom build with `Dockerfile.python313` or install packages in correct Python
- **Verify**: `DO $$ import sys; plpy.notice(sys.version) $$ LANGUAGE plpython3u;`

## IMMUTABLE Functions and Cache Strategy (v1.4.1+)

- AIDEV-NOTE: IMMUTABLE functions use SELECT-only cache reads (no writes)
- **Change**: Frecency eviction → Age-based FIFO eviction
- **Cache population**: Use VOLATILE wrapper functions (`steadytext_generate_cached()`)
- **Trade-off**: Lost access tracking, gained true immutability for query optimization

## Architecture Overview

**Principles**: Leverage daemon, mirror cache, graceful degradation, security first

**Key Components**:
- `daemon_connector.py`: ZeroMQ client
- `cache_manager.py`: Age-based cache (was frecency)
- `security.py`: Input validation/rate limiting
- `worker.py`: Async queue processor

## Python Module Loading

- AIDEV-NOTE: plpython3u uses different Python env - modules in PostgreSQL's path
- **v1.0.0 Fix**: Resolve $libdir, add to sys.path, cache in GD
- **Debug**: `SELECT _steadytext_init_python();` if ImportError

### Implementation Patterns

**Daemon**: Singleton client, auto-startup, fallback to direct loading

**Cache**: Age-based eviction (was frecency), matches SteadyText key format

**Security**: Input validation, rate limiting (implemented)

- AIDEV-TODO: Connection pooling, prepared statements, batch optimizations

## Development Quick Reference

**Add function**: SQL → Python → Tests → Docs

**Debug imports**: Check sys.path and module locations

**Test daemon**: `SELECT * FROM steadytext_daemon_status();`


## Troubleshooting

**Common Issues**:
1. **Not initialized**: Run `SELECT _steadytext_init_python();`
2. **Daemon down**: Check `st daemon status`
3. **Cache hit**: Normal - use ON CONFLICT
4. **Model issues**: Use `STEADYTEXT_USE_FALLBACK_MODEL=true` for Gemma-3n problems

**Compatible Models**: Qwen2.5-3B (recommended), Qwen3-1.7B (smaller)

- AIDEV-TODO: Track gemma-3n compatibility updates


## Async Functions (v1.1.0)

- AIDEV-NOTE: Queue-based async with UUID returns, worker processes with SKIP LOCKED
- AIDEV-NOTE: `steadytext_generate_async` was missing until v1.4.4 (only JSON/regex/choice async existed)

**Components**: Queue table → *_async functions → Python worker → Result retrieval

**Available async functions**:
- `steadytext_generate_async()` - Basic text generation (added v1.4.4)
- `steadytext_embed_async()` - Embeddings
- `steadytext_generate_json_async()` - JSON with schema
- `steadytext_generate_regex_async()` - Regex-constrained
- `steadytext_generate_choice_async()` - Choice-constrained
- `steadytext_rerank_async()` - Document reranking

**Test**: `SELECT st_generate_async('Test', 100);`

- AIDEV-TODO: SSE streaming, worker auto-scaling, distributed workers

## Cache Eviction (v1.4.0+)

- AIDEV-NOTE: Now uses age-based eviction (FIFO) for IMMUTABLE compliance
- **Setup**: `CREATE EXTENSION pg_cron; SELECT steadytext_setup_cache_eviction_cron();`
- **Config**: Set max_entries, max_size_mb, min_age_hours via config table

- AIDEV-TODO: Adaptive thresholds, alternative strategies (LRU/ARC)

## Python Package Installation (v1.4.0+)

- AIDEV-NOTE: Auto-installs to `$(pkglibdir)/pg_steadytext/site-packages`
- **Install**: `sudo make install` or manual pip with --target
- **Test**: `./test_installation.sh`

- AIDEV-TODO: Virtual env support, package version checking

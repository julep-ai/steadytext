# AIDEV Notes for pg_steadytext

This file contains important development notes and architectural decisions for AI assistants working on pg_steadytext.

## Recent Fixes

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

**Components**: Queue table → *_async functions → Python worker → Result retrieval

**Test**: `SELECT steadytext_generate_async('Test', 10);`

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

# AIDEV Notes for pg_steadytext

This file contains important development notes and architectural decisions for AI assistants working on pg_steadytext.

## Extension Table Creation Pattern

### IMPORTANT: Table Creation in Extension Scripts
- **AIDEV-NOTE**: Always use `DROP TABLE IF EXISTS table_name CASCADE;` before `CREATE TABLE`
- **Issue**: Using `CREATE TABLE IF NOT EXISTS` can cause tables to not be added to extension
- **Why**: PostgreSQL only adds objects to extensions when they're created within the extension script
- **Pattern**:
  ```sql
  -- Wrong: Table won't be added to extension if it already exists
  CREATE TABLE IF NOT EXISTS my_table (...);
  
  -- Correct: Ensures table is always added to extension
  DROP TABLE IF EXISTS my_table CASCADE;
  CREATE TABLE my_table (...);
  ```
- **Fixed in**: v1.4.5 and v1.4.6 (2025-08-14)

## Recent Fixes

### v2025.8.26 - PL/Python Function Fixes and pgTAP Test Improvements
- **Fixed**: PL/Python plpy.execute() usage patterns
  - plpy.execute() expects either a SQL string or a prepared plan
  - Must use plpy.prepare() for parameterized queries, not plpy.execute(query, [args])
  - Example: `plan = plpy.prepare("SELECT * WHERE id = $1", ["integer"]); plpy.execute(plan, [123])`
- **Fixed**: UUID type casting in async functions
  - Python strings must be explicitly cast to UUID in SQL: `VALUES ($1::uuid, ...)`
  - PostgreSQL doesn't auto-cast text to UUID in parameterized queries
- **Fixed**: PL/Python row output patterns
  - Use `yield` to output individual rows in table-returning functions
  - Using `return [...]` outputs all rows as a single array value
  - Example: `for row in results: yield row["column"]`
- **Fixed**: pgTAP test syntax errors
  - Functions with OUT parameters already define result columns
  - Don't use `AS (column definitions)` clause with such functions
- **Performance**: Reranking tests with mini models
  - Set STEADYTEXT_USE_MINI_MODELS=true to use smaller models in tests
  - Prevents timeouts when loading large reranking models
  - Must be set at container level for PostgreSQL extension tests
- AIDEV-NOTE: Always test PL/Python functions thoroughly - error messages can be cryptic
- AIDEV-NOTE: Use dynamic schema resolution pattern for all extension table access
- AIDEV-TODO: Add automated tests for all PL/Python edge cases

### v2025.8.26 - Complete Schema Qualification for All Functions
- **Fixed**: Extended schema qualification to ALL functions accessing extension tables
  - Added dynamic schema resolution to daemon control functions
  - Added schema qualification to structured generation functions (JSON, regex, choice)
  - Added schema qualification to summarization helper functions
  - Uses `plpy.execute()` and `plpy.quote_ident()` for dynamic schema lookup
  - Ensures all functions work correctly with TimescaleDB continuous aggregates
- **Added**: Proper `@extschema@` qualification for all alias functions
  - `st_daemon_start()` and `st_daemon_stop()` now use proper schema references
- **Migration**: Comprehensive migration script from v2025.8.17 to v2025.8.26
  - Includes all 17 function updates with proper schema qualification
- AIDEV-NOTE: Always use dynamic schema resolution for any function accessing extension tables
- AIDEV-NOTE: Use `@extschema@` placeholder in SQL alias functions for proper schema references

### v2025.8.26 - AI Summarization Enhancement, Schema Qualification & GPT-5 Support
- **Added**: Enhanced AI summarization with remote model support
  - Renamed `ai_*` functions to `steadytext_*` with `st_*` aliases for consistency  
  - Added `model` and `unsafe_mode` parameters to summarization functions
  - Support for remote models like `openai:gpt-4o-mini` with `unsafe_mode=TRUE`
  - Increased default max_facts from 5 to 10
- **Added**: GPT-5 reasoning model support
  - OpenAI's GPT-5 series models (gpt-5-mini, gpt-5-pro) now recognized as reasoning models
  - Temperature automatically adjusted to 1.0 for GPT-5 models (requirement from OpenAI)
  - AIDEV-NOTE: Reasoning models (o1 series, GPT-5 series) require temperature=1.0
- **Added**: Custom provider options support
  - New `options` parameter for all generation functions to pass provider-specific settings
  - Supports JSON object with provider parameters like top_p, presence_penalty, etc.
  - Example: `SELECT st_generate('Hello', options => '{"top_p": 0.95}'::jsonb);`
  - AIDEV-NOTE: Options are passed as **kwargs to remote providers
- **Fixed**: Schema qualification for TimescaleDB continuous aggregates
  - All table references now use `@extschema@.table_name` pattern
  - Fixes issue #95 where functions failed when called from continuous aggregates
  - AIDEV-NOTE: Critical for any function that accesses extension tables
- **Fixed**: Python scoping issues in PL/Python aggregate functions
  - Resolved NameError caused by reassigning argument variables
  - AIDEV-NOTE: In PL/Python, reassigning an argument makes it local for entire scope
  - Solution: Use new local variables instead of reassigning arguments
- AIDEV-TODO: Add comprehensive tests for remote model summarization
- AIDEV-TODO: Consider adding support for streaming in summarization functions
- AIDEV-TODO: Add tests for GPT-5 models and custom options parameter

### v1.4.6 - Unsafe Mode Support for Embeddings
- **Added**: `model` and `unsafe_mode` parameters to embedding functions
  - `steadytext_embed()`, `steadytext_embed_cached()`, `steadytext_embed_async()`
  - Remote embedding models (e.g., `openai:text-embedding-3-small`) supported with `unsafe_mode=TRUE`
  - Cache keys include model name when specified: `embed:{text}:{model}`
- **Updated**: Python `daemon_connector.py` embed() method supports new parameters
- **Security**: Remote embedding models require explicit `unsafe_mode=TRUE`
- AIDEV-NOTE: Skip daemon for remote embedding models to improve performance
- AIDEV-NOTE: Consistent behavior with generation functions' unsafe_mode support

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

## PL/Python Best Practices

### Common Pitfalls and Solutions

1. **plpy.execute() Usage**
   - WRONG: `plpy.execute("SELECT * WHERE id = $1", [123])`
   - RIGHT: `plan = plpy.prepare("SELECT * WHERE id = $1", ["integer"]); plpy.execute(plan, [123])`
   - Or for simple queries: `plpy.execute("SELECT * WHERE id = 123")`

2. **Type Casting**
   - Python strings aren't auto-cast to PostgreSQL UUIDs
   - Always use explicit casting: `INSERT INTO table VALUES ($1::uuid)`
   - Same applies to other types like JSONB: `$2::jsonb`

3. **Row Output**
   - For table-returning functions, use `yield` not `return`
   - `yield` outputs one row at a time
   - `return [...]` outputs entire list as single value

4. **Schema Qualification**
   - Always use dynamic schema resolution for extension tables:
   ```python
   ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
   ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
   query = f"SELECT * FROM {plpy.quote_ident(ext_schema)}.my_table"
   ```

5. **Error Handling**
   - PL/Python errors can be cryptic
   - Always test with actual data
   - Use try/except blocks and plpy.warning() for debugging

## Python Module Loading

- AIDEV-NOTE: plpython3u uses different Python env - modules in PostgreSQL's path
- **v1.0.0 Fix**: Resolve $libdir, add to sys.path, cache in GD
- **Debug**: `SELECT _steadytext_init_python();` if ImportError

### Implementation Patterns

**Daemon**: Singleton client, auto-startup, fallback to direct loading

**Cache**: Age-based eviction (was frecency), matches SteadyText key format

**Security**: Input validation, rate limiting (implemented)

- AIDEV-TODO: Connection pooling, prepared statements, batch optimizations

## TimescaleDB Integration (v2025.8.27+)

- AIDEV-NOTE: TimescaleDB package is installed in Docker image for optional time-series testing
- AIDEV-NOTE: Due to Omnigres base image constraints, TimescaleDB requires manual configuration
- AIDEV-NOTE: To enable TimescaleDB in container: ALTER SYSTEM SET shared_preload_libraries = 'omni--0.2.11.so', 'timescaledb';
- Test file: test/pgtap/16_timescaledb_compat.sql - Tests basic compatibility
- Test file: test/pgtap/16_timescaledb_integration.sql - Comprehensive continuous aggregate tests (requires TimescaleDB)
- AIDEV-NOTE: Use STEADYTEXT_USE_MINI_MODELS=true when running tests to prevent timeouts
- AIDEV-TODO: Simplify TimescaleDB configuration once Omnigres updates their base image

## DevContainer Testing Instructions

### Testing Extension Changes in DevContainer

When working inside the devcontainer (you can check if `/workspace` exists with source code), PostgreSQL runs in a separate container.

- AIDEV-NOTE: DevContainer limitation - Docker daemon runs on host, not in devcontainer
- AIDEV-NOTE: Cannot mount host paths directly from inside devcontainer's docker-compose
- AIDEV-NOTE: Solution: Use copy-and-build approach for fast iteration (~2-3 seconds)

**Quick Rebuild Method (Recommended):**
```bash
# From anywhere in the devcontainer, run:
/workspace/.devcontainer/rebuild_extension_simple.sh

# This script automatically:
# - Copies all extension files to the postgres container
# - Builds and installs the extension
# - Reinstalls it in the database  
# - Verifies the installation

# For auto-rebuild on file changes (requires inotify-tools):
/workspace/.devcontainer/watch_extension.sh
```

- AIDEV-NOTE: rebuild_extension_simple.sh uses docker cp for file transfer (works around mount limitations)
- AIDEV-NOTE: watch_extension.sh uses inotifywait for efficient file monitoring with polling fallback
- AIDEV-NOTE: Both scripts handle all complexity - just run and develop

**Manual Method:**

**1. Check Container Status:**
```bash
# PostgreSQL is available at: postgres://postgres:password@postgres
docker ps | grep pg_steadytext_db  # Should show the postgres container
```

**2. Copy and Build Extension:**
```bash
# Copy all source files to container
docker exec pg_steadytext_db mkdir -p /tmp/pg_steadytext_build
docker cp /workspace/pg_steadytext/Makefile pg_steadytext_db:/tmp/pg_steadytext_build/
docker cp /workspace/pg_steadytext/pg_steadytext.control pg_steadytext_db:/tmp/pg_steadytext_build/
docker cp /workspace/pg_steadytext/sql pg_steadytext_db:/tmp/pg_steadytext_build/
docker cp /workspace/pg_steadytext/python pg_steadytext_db:/tmp/pg_steadytext_build/
docker cp /workspace/pg_steadytext/test pg_steadytext_db:/tmp/pg_steadytext_build/

# Build and install
docker exec pg_steadytext_db bash -c 'cd /tmp/pg_steadytext_build && make clean && make install'
```

**3. Test Extension Installation:**
```bash
# Connect to PostgreSQL
PGPASSWORD=password psql -h postgres -U postgres -d postgres

# Reinstall extension
DROP EXTENSION IF EXISTS pg_steadytext CASCADE;
CREATE EXTENSION pg_steadytext;

# Verify installation
SELECT steadytext_version();
SELECT extname, extversion FROM pg_extension WHERE extname = 'pg_steadytext';
```

**4. Run Tests:**
```bash
# pgTAP tests (requires copying test files first)
docker exec pg_steadytext_db bash -c "cd /tmp/pg_steadytext_build && ./run_pgtap_tests.sh"

# Or specific test
docker exec pg_steadytext_db psql -U postgres -f /tmp/pg_steadytext_build/test/pgtap/01_basic.sql
```

- AIDEV-NOTE: The postgres container runs separately without direct mounts for security
- AIDEV-NOTE: Use rebuild_extension_simple.sh for fastest development iteration
- AIDEV-NOTE: The watch script provides automatic rebuilds on file changes
- AIDEV-NOTE: PostgreSQL version is 17 in the current devcontainer setup

## Development Quick Reference

**Add function**: SQL → Python → Tests → Docs

**Debug imports**: Check sys.path and module locations

**Test daemon**: `SELECT * FROM steadytext_daemon_status();`


## Troubleshooting

**Common Issues**:
1. **Not initialized**: Run `SELECT _steadytext_init_python();`
2. **Daemon down**: Check `st daemon status`
3. **Cache hit**: Normal - use ON CONFLICT
4. **Model issues**: Use `STEADYTEXT_USE_FALLBACK_MODEL=true` for model loading problems

**Compatible Models**: Qwen3-4B (default small), Qwen3-30B (large)


## Async Functions (v1.1.0)

- AIDEV-NOTE: Queue-based async with UUID returns, worker processes with SKIP LOCKED
- AIDEV-NOTE: `steadytext_generate_async` was missing until v1.4.4 (only JSON/regex/choice async existed)

**Components**: Queue table → *_async functions → Python worker → Result retrieval

**Available async functions**:
- `steadytext_generate_async()` - Basic text generation (added v1.4.4)
- `steadytext_embed_async()` - Embeddings
- `steadytext_rerank_async()` - Document reranking

**Test**: `SELECT st_generate_async('Test', 100);`

- AIDEV-TODO: SSE streaming, worker auto-scaling, distributed workers

## Cache Eviction (v1.4.0+)

- AIDEV-NOTE: Now uses age-based eviction (FIFO) for IMMUTABLE compliance
- **Config**: Set max_entries, max_size_mb, min_age_hours via config table
- **Function**: `steadytext_cache_evict_by_age()` for manual eviction

- AIDEV-TODO: Adaptive thresholds, alternative strategies (LRU/ARC)

## Python Package Installation (v1.4.0+)

- AIDEV-NOTE: Auto-installs to `$(pkglibdir)/pg_steadytext/site-packages`
- **Install**: `sudo make install` or manual pip with --target
- **Test**: `./test_installation.sh`

- AIDEV-TODO: Virtual env support, package version checking

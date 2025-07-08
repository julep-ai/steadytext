# AIDEV Notes for pg_steadytext

This file contains important development notes and architectural decisions for AI assistants working on pg_steadytext.

## Recent Security Fixes (v1.0.2)

This document summarizes the critical fixes implemented in the pg_steadytext PostgreSQL extension.

### Critical Issues Fixed

#### 1. SQL Injection Vulnerability (HIGH SEVERITY) ✅
**File**: `python/cache_manager.py`
**Issue**: Table name was directly interpolated into SQL queries using f-strings
**Fix**: Added validation in `__init__` to only allow alphanumeric characters and underscores
**AIDEV-NOTE**: Added validation to prevent SQL injection by restricting table names

#### 2. Missing Method (HIGH SEVERITY) ✅  
**File**: `python/daemon_connector.py`
**Issue**: `worker.py` called non-existent `is_daemon_running()` method
**Fix**: Added `is_daemon_running()` and `check_health()` methods to SteadyTextConnector
**AIDEV-NOTE**: These methods check daemon availability and health status

#### 3. Cache Key Inconsistency (HIGH SEVERITY) ✅
**Files**: `python/cache_manager.py`, `sql/pg_steadytext--1.0.0.sql`
**Issue**: PostgreSQL used SHA256 hashing while SteadyText used simple string format
**Fix**: Updated to match SteadyText's format:
  - Generation: `"{prompt}"` or `"{prompt}::EOS::{eos_string}"`
  - Embeddings: SHA256 hash of `"embed:{text}"`
**AIDEV-NOTE**: Cache keys now match SteadyText for cross-system compatibility

#### 4. Rate Limiting (MEDIUM SEVERITY) ✅
**File**: `python/security.py`
**Issue**: `check_rate_limit()` was a placeholder returning True
**Fix**: Implemented sliding window rate limiting with minute/hour/day buckets
**AIDEV-NOTE**: Uses atomic SQL operations to update counters and check limits

#### 5. Input Validation (MEDIUM SEVERITY) ✅
**File**: `python/daemon_connector.py`
**Issue**: Host and port parameters weren't validated
**Fix**: Added validation for:
  - Host: alphanumeric, dots, hyphens, underscores only
  - Port: integer between 1-65535
**AIDEV-NOTE**: Prevents potential command injection attacks

#### 6. Unused Code (LOW SEVERITY) ✅
**File**: `python/security.py`
**Issue**: `SAFE_TEXT_PATTERN` regex was defined but never used
**Fix**: Removed the unused pattern with explanatory comment
**AIDEV-NOTE**: The validate_prompt method uses a more nuanced approach

### AIDEV-TODO Items for Future Work

1. **Cache Synchronization** (`cache_manager.py`)
   - Implement bidirectional sync between PostgreSQL and SteadyText SQLite cache
   - Read from SteadyText's cache database and import entries

2. **Connection Pooling** (`daemon_connector.py`)
   - Implement ZeroMQ connection pooling for high-concurrency scenarios
   - Reuse connections instead of creating new ones

3. **Enhanced Security** (`security.py`)
   - Consider making prompt validation stricter based on requirements
   - Add more sophisticated prompt injection detection

4. **Performance Optimizations**
   - Implement prepared statement caching across requests
   - Add batch operation improvements
   - Create performance benchmarking suite

### AIDEV-QUESTIONS for Review

1. Should the prompt validation be more restrictive, or is logging sufficient?
2. Should we support multiple daemon instances for load balancing?
3. Is the current rate limiting granularity (minute/hour/day) sufficient?

### Testing Recommendations

1. Test SQL injection prevention with various malicious table names
2. Verify cache key compatibility between PostgreSQL and SteadyText
3. Test rate limiting with concurrent requests
4. Validate daemon failover behavior
5. Check input validation edge cases

## pgTAP Testing Framework (v1.0.3)

### AIDEV-NOTE: pgTAP Implementation

The extension now uses pgTAP (PostgreSQL Test Anything Protocol) for comprehensive testing. This provides:

1. **TAP Output**: Standard test output format for CI/CD integration
2. **Rich Assertions**: Specialized PostgreSQL testing functions
3. **Transaction Safety**: Each test runs in a transaction and rolls back
4. **Better Coverage**: More thorough testing of edge cases

### Test Organization

```
test/pgtap/
├── 00_setup.sql        # Verify pgTAP installation
├── 01_basic.sql        # Core functionality tests
├── 02_embeddings.sql   # Embedding and vector tests
├── 03_async.sql        # Async queue tests
├── 04_structured.sql   # JSON/regex/choice tests
└── 05_cache_daemon.sql # Cache and daemon tests
```

### Running pgTAP Tests

```bash
# Install pgTAP (if not in Docker)
sudo apt-get install postgresql-17-pgtap

# Run all tests
make test-pgtap

# Run with verbose output
make test-pgtap-verbose

# Run for CI (TAP output)
make test-pgtap-tap

# Run specific test
./run_pgtap_tests.sh test/pgtap/01_basic.sql
```

### Writing New pgTAP Tests

```sql
-- test/pgtap/99_new_feature.sql
BEGIN;
SELECT plan(5);  -- Number of tests

-- Test function exists
SELECT has_function('public', 'new_function', 'Function should exist');

-- Test return type
SELECT function_returns('public', 'new_function', ARRAY['text'], 'text', 
    'Function should return text');

-- Test behavior
SELECT is(new_function('input'), 'expected', 'Should return expected value');

-- Test error handling
SELECT throws_ok(
    $$ SELECT new_function(NULL) $$,
    'P0001',
    'Input cannot be null',
    'Should fail with null input'
);

-- Test performance (optional)
SELECT lives_ok(
    $$ SELECT new_function('test') $$,
    'Function should complete without error'
);

SELECT * FROM finish();
ROLLBACK;
```

### AIDEV-NOTE: Key pgTAP Functions

- `plan(n)` - Declare number of tests
- `has_function()` - Check function exists
- `function_returns()` - Check return type
- `has_table()` - Check table exists
- `has_column()` - Check column exists
- `has_index()` - Check index exists
- `is()` - Exact equality
- `isnt()` - Inequality
- `ok()` - Boolean assertion
- `matches()` - Regex matching
- `throws_ok()` - Exception testing
- `lives_ok()` - No exception testing
- `results_eq()` - Compare query results
- `columns_are()` - Check column names

### CI Integration

The extension includes a GitHub Actions workflow for pgTAP tests:

```yaml
# .github/workflows/test-pgtap.yml
- Runs on push/PR to pg_steadytext/
- Sets up PostgreSQL 17 with required extensions
- Runs tests with TAP output
- Uploads test results as artifacts
```

## Recent Fixes (v1.0.1)

### 1. Removed thinking_mode Parameter
- AIDEV-NOTE: The `thinking_mode` parameter was removed from all functions as it's not supported by the core SteadyText library
- Fixed in: SQL functions, daemon_connector.py, worker.py, cache_manager.py, config.py, and test files
- The parameter was causing "unexpected keyword argument" errors

### 2. Fixed Python Initialization
- AIDEV-NOTE: Changed from immediate initialization to on-demand initialization in each function
- This ensures proper initialization even across session boundaries
- Functions now call `_steadytext_init_python()` automatically if not initialized

### 3. Optimized Dockerfile for Caching
- AIDEV-NOTE: Restructured Dockerfile to maximize layer caching
- Dependencies are installed first (rarely change)
- Source files are copied in order of change frequency: Makefile → SQL → Python
- This significantly reduces rebuild time when only Python files change

### 4. Model Compatibility Issues with Gemma-3n
- AIDEV-NOTE: The gemma-3n models may not be compatible with the inference-sh fork of llama-cpp-python
- This causes "Failed to load model from file" errors even though the model downloads successfully
- Added fallback model support using known-working Qwen models
- Added diagnostic scripts to help troubleshoot model loading issues

## Architecture Overview

### Core Design Principles

1. **Minimal Reimplementation**: We leverage SteadyText's existing daemon architecture rather than reimplementing functionality
2. **Cache Synchronization**: PostgreSQL cache mirrors SteadyText's SQLite cache for consistency
3. **Graceful Degradation**: All functions have fallbacks when the daemon is unavailable
4. **Security First**: Input validation and rate limiting are built-in

### Component Map

```
pg_steadytext/
├── sql/
│   └── pg_steadytext--1.0.0.sql    # AIDEV-SECTION: Core schema and functions
├── python/
│   ├── daemon_connector.py          # AIDEV-NOTE: ZeroMQ client for SteadyText daemon
│   ├── cache_manager.py            # AIDEV-NOTE: Frecency cache implementation
│   ├── security.py                 # AIDEV-NOTE: Input validation and rate limiting
│   ├── config.py                   # AIDEV-NOTE: Configuration management
│   └── worker.py                   # AIDEV-NOTE: Background queue processor
└── test/
    └── sql/                        # AIDEV-NOTE: pgTAP-compatible test files
```

## Key Implementation Details

### Python Module Loading (CRITICAL)

**AIDEV-NOTE**: PostgreSQL's plpython3u has a different Python environment than the system Python. Modules must be installed in PostgreSQL's Python path.

**Fixed in v1.0.0**: The module loading issue has been resolved by:
1. Resolving `$libdir` to actual path using `pg_settings`
2. Adding module directory to `sys.path` in initialization
3. Caching modules in GD (Global Dictionary) for reuse
4. Enhanced error messages with debugging information

```sql
-- The SQL schema now properly resolves the library path
SELECT setting FROM pg_settings WHERE name = 'pkglibdir';
-- Then builds the full path: <pkglibdir>/pg_steadytext/python
```

**Key Changes Made**:
- `_steadytext_init_python()` now adds module path to sys.path
- All Python functions use cached modules from GD
- Docker build includes module verification step
- Better error handling with descriptive messages

**Common Issues**:
1. ImportError: Module not found → Run `SELECT _steadytext_init_python();`
2. Permission denied → Ensure postgres user can read Python files
3. Version mismatch → PostgreSQL Python != System Python
4. Docker issues → Check that modules exist in container at `/usr/lib/postgresql/17/lib/pg_steadytext/python/`

### Daemon Integration

**AIDEV-NOTE**: The daemon connector uses these patterns:

1. **Singleton Client**: Reuse ZeroMQ connections
2. **Automatic Startup**: Start daemon if not running
3. **Fallback Mode**: Direct model loading if daemon fails

```python
# Always check daemon status first
if client.is_daemon_running():
    result = client.generate(prompt)
else:
    # Fallback to direct generation
    from steadytext import generate
    result = generate(prompt)
```

### Cache Design

**AIDEV-NOTE**: Frecency = Frequency + Recency

```python
score = access_count * (1 / (1 + time_since_last_access_hours))
```

**Cache Key Generation**:
- Must match SteadyText's format exactly
- Includes: prompt, max_tokens, thinking_mode, model_name
- Uses MD5 for consistency

### Security Considerations

**AIDEV-TODO**: Implement these security features:

1. **SQL Injection Prevention**: Use parameterized queries
2. **Prompt Injection**: Validate and sanitize inputs
3. **Rate Limiting**: Per-user and per-session limits
4. **Resource Limits**: Max tokens, timeout, memory

### Performance Optimizations

**AIDEV-NOTE**: These areas need optimization:

1. **Prepared Statements**: Cache frequently used queries
2. **Connection Pooling**: Reuse daemon connections
3. **Batch Operations**: Process multiple requests together
4. **Index Usage**: Ensure queries use indexes

## Common Development Tasks

### Adding a New Function

1. Add SQL function definition to schema
2. Implement Python logic if needed
3. Add tests to test/sql/
4. Update documentation
5. Add AIDEV-NOTE comments

### Debugging Import Issues

```python
# Add to any Python function to debug
import sys
plpy.notice(f"Python path: {sys.path}")
plpy.notice(f"Module locations: {[m.__file__ for m in sys.modules.values() if hasattr(m, '__file__')]}")
```

### Testing Daemon Connection

```sql
-- Check daemon status
SELECT * FROM steadytext_daemon_status();

-- Test connection
SELECT steadytext_generate('test', 10);

-- Check logs
SELECT * FROM steadytext_daemon_health;
```

## Future Enhancements

### AIDEV-TODO: Priority Items

1. **Connection Pooling**: Implement ZeroMQ connection pool
2. **Streaming Optimization**: True streaming instead of simulated
3. **Cache Sync**: Bidirectional sync with SteadyText SQLite
4. **Metrics Export**: Prometheus/OpenTelemetry integration
5. **GPU Support**: Detect and use GPU-enabled models

### AIDEV-QUESTION: Design Decisions

1. Should we support multiple daemon instances?
2. How to handle model versioning?
3. Best approach for distributed caching?
4. Should we implement our own model loading?

## Troubleshooting Guide

### Common Errors and Solutions

1. **"pg_steadytext Python environment not initialized"**
   - Check Python modules are installed in correct path
   - Verify plpython3u extension is created
   - Run `SELECT _steadytext_init_python();`

2. **"Failed to connect to daemon"**
   - Check if daemon is running: `st daemon status`
   - Verify ZeroMQ port is not blocked
   - Check daemon logs

3. **"Cache key already exists"**
   - This is normal - cache hit
   - Use ON CONFLICT clause to handle

4. **"Model not found"**
   - SteadyText models auto-download on first use
   - Check disk space in ~/.cache/steadytext/
   - Verify internet connectivity

5. **"Failed to load model from file: /path/to/gemma-3n-*.gguf"**
   - This is a known compatibility issue with gemma-3n models and the inference-sh fork of llama-cpp-python
   - **Solution 1**: Use the fallback model by setting environment variable:
     ```bash
     # For Docker build:
     docker build --build-arg STEADYTEXT_USE_FALLBACK_MODEL=true -t pg_steadytext .
     
     # For Docker run:
     docker run -e STEADYTEXT_USE_FALLBACK_MODEL=true -p 5432:5432 pg_steadytext
     
     # For direct usage:
     export STEADYTEXT_USE_FALLBACK_MODEL=true
     ```
   - **Solution 2**: Manually specify a compatible model:
     ```bash
     export STEADYTEXT_GENERATION_MODEL_REPO=lmstudio-community/Qwen2.5-3B-Instruct-GGUF
     export STEADYTEXT_GENERATION_MODEL_FILENAME=Qwen2.5-3B-Instruct-Q8_0.gguf
     ```
   - **Diagnostics**: Run the diagnostic script to check your environment:
     ```bash
     # Inside Docker container:
     docker exec -it pg_steadytext /usr/local/bin/diagnose_pg_model
     
     # Outside Docker:
     python /path/to/steadytext/diagnose_model.py
     ```

### Model Compatibility Matrix

| Model | Repository | Filename | Status | Notes |
|-------|------------|----------|---------|-------|
| Gemma-3n 2B | ggml-org/gemma-3n-E2B-it-GGUF | gemma-3n-E2B-it-Q8_0.gguf | ⚠️ Issues | May fail with inference-sh fork |
| Gemma-3n 4B | ggml-org/gemma-3n-E4B-it-GGUF | gemma-3n-E4B-it-Q8_0.gguf | ⚠️ Issues | May fail with inference-sh fork |
| Qwen2.5 3B | lmstudio-community/Qwen2.5-3B-Instruct-GGUF | Qwen2.5-3B-Instruct-Q8_0.gguf | ✅ Working | Recommended fallback |
| Qwen3 1.7B | lmstudio-community/qwen3-1.7b-llama-cpp-python-GGUF | qwen3-1.7b-q8_0.gguf | ✅ Working | Smaller alternative |

AIDEV-TODO: Track updates to the inference-sh fork for gemma-3n support

## Development Workflow

1. **Make Changes**: Edit SQL/Python files
2. **Rebuild**: `make clean && make install`
3. **Test**: `make test` or `./run_tests.sh`
4. **Debug**: Check PostgreSQL logs and daemon output
5. **Document**: Add AIDEV-NOTE comments

## Version Compatibility

- PostgreSQL: 14+ (tested on 14, 15, 16)
- Python: 3.8+ (matches plpython3u version)
- SteadyText: 2.1.0+ (for daemon support)
- pgvector: 0.5.0+ (for embedding storage)

---

## Async Functions Implementation (v1.1.0)

### AIDEV-NOTE: Async Architecture Overview

The async implementation uses a queue-based approach that allows PostgreSQL to continue processing while AI generation happens in the background:

1. **Queue Table**: `steadytext_queue` stores all async requests with metadata
2. **Request Functions**: `*_async` functions insert into queue and return UUID immediately
3. **Background Worker**: Python worker polls queue and processes requests
4. **Result Retrieval**: Multiple ways to check status and get results

### Key Design Decisions

1. **Why Queue Table?**: 
   - Native PostgreSQL solution (no external dependencies)
   - Survives restarts (persistent storage)
   - Supports LISTEN/NOTIFY for responsive processing
   - Allows multiple workers with SKIP LOCKED

2. **Request Types**:
   - `generate`: Basic text generation
   - `embed`: Embedding generation
   - `generate_json`: JSON with schema validation
   - `generate_regex`: Regex-constrained generation
   - `generate_choice`: Choice-constrained generation

3. **Worker Design**:
   - Single-threaded Python worker (can run multiple instances)
   - Uses daemon_connector for all AI operations
   - Handles retries and error recovery
   - Updates queue status atomically

### AIDEV-TODO: Future Async Enhancements

1. **Streaming Support**: Add Server-Sent Events (SSE) endpoint for streaming results
2. **Priority Queue Optimization**: Implement more sophisticated scheduling algorithms
3. **Worker Pool Management**: Auto-scale workers based on queue depth
4. **Result Callbacks**: Support PostgreSQL triggers on completion
5. **Distributed Workers**: Support workers on different machines

### Testing Async Functions

```sql
-- Quick test of all async functions
SELECT steadytext_generate_async('Test', 10);
SELECT steadytext_embed_async('Test');
SELECT steadytext_generate_json_async('Test', '{"type": "string"}'::jsonb);
SELECT steadytext_generate_regex_async('Test', '\d+');
SELECT steadytext_generate_choice_async('Test', ARRAY['a', 'b']);

-- Check all results
SELECT request_id, request_type, status FROM steadytext_queue 
WHERE created_at > NOW() - INTERVAL '1 minute';
```

### Common Async Issues

1. **Worker Not Running**: Queue fills with 'pending' requests
   - Start worker: `python /path/to/worker.py`
   - Check worker logs for errors

2. **Slow Processing**: Requests take too long
   - Check daemon status: `SELECT * FROM steadytext_daemon_health;`
   - Monitor queue depth: `SELECT COUNT(*) FROM steadytext_queue WHERE status = 'pending';`

3. **Failed Requests**: Requests consistently fail
   - Check error messages: `SELECT * FROM steadytext_queue WHERE status = 'failed';`
   - Verify model availability and daemon connection

**AIDEV-NOTE**: This file should be updated whenever architectural decisions change or new patterns are established.

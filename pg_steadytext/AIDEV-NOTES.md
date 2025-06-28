# AIDEV Notes for pg_steadytext

This file contains important development notes and architectural decisions for AI assistants working on pg_steadytext.

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

```sql
-- The SQL schema sets up Python path
ALTER DATABASE dbname SET plpython3.python_path TO '$libdir/pg_steadytext/python:...';
```

**Common Issues**:
1. ImportError: Module not found → Check installation path
2. Permission denied → Ensure postgres user can read Python files
3. Version mismatch → PostgreSQL Python != System Python

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

## Development Workflow

1. **Make Changes**: Edit SQL/Python files
2. **Rebuild**: `make clean && make install`
3. **Test**: `make test` or `./run_tests.sh`
4. **Debug**: Check PostgreSQL logs and daemon output
5. **Document**: Add AIDEV-NOTE comments

## Version Compatibility

- PostgreSQL: 14+ (tested on 14, 15, 16)
- Python: 3.8+ (matches plpython3u version)
- SteadyText: 1.3.0+ (for daemon support)
- pgvector: 0.5.0+ (for embedding storage)

---

**AIDEV-NOTE**: This file should be updated whenever architectural decisions change or new patterns are established.
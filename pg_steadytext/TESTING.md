# Testing Guide for pg_steadytext

This guide covers the various testing approaches for the pg_steadytext PostgreSQL extension.

## Test Scripts Overview

### 1. `test_integration_localhost.sh`
Comprehensive integration test script that connects from localhost to a PostgreSQL instance (typically in Docker).

**Features:**
- Basic functionality tests (version, generate, embed)
- Cache functionality tests
- Async queue tests
- Structured generation tests (JSON, regex, choice)
- Optional pgTAP test integration
- Optional performance benchmarks
- TAP format output for CI integration

**Usage:**
```bash
# Basic run (connects to localhost:5432)
./test_integration_localhost.sh

# With custom connection parameters
./test_integration_localhost.sh -h localhost -p 5432 -U postgres -W postgres

# Verbose output
./test_integration_localhost.sh -v

# Run pgTAP tests
./test_integration_localhost.sh --pgtap

# Run performance benchmarks
./test_integration_localhost.sh --benchmark

# TAP format for CI
./test_integration_localhost.sh --tap
```

### 2. `test_e2e_docker.sh`
End-to-end test that builds and tests the extension in a fresh Docker container.

**Features:**
- Uses cimg/postgres image
- Installs all dependencies
- Builds extension from source
- Runs integration tests
- Optional container preservation for debugging

**Usage:**
```bash
# Basic run
./test_e2e_docker.sh

# Verbose output
./test_e2e_docker.sh -v

# Include pgTAP tests
./test_e2e_docker.sh --pgtap

# Run benchmarks
./test_e2e_docker.sh --benchmark

# Keep container after tests
./test_e2e_docker.sh --keep-container
```

### 3. `run_pgtap_tests.sh`
Runs pgTAP tests against a local PostgreSQL installation.

## Test Categories

### Basic Functionality Tests
- Extension version check
- Text generation
- Embedding generation
- Determinism verification
- NULL/empty input handling
- Special character handling
- Parameter validation

### Cache Tests
- Cache hit/miss behavior
- Cache statistics
- Cache clearing
- Cache differentiation by parameters
- Cache eviction
- Extended statistics
- Usage analysis

### Async Queue Tests
- Async request submission
- UUID generation
- Queue status checking
- Batch operations
- Priority handling
- Request cancellation
- Result retrieval

### Structured Generation Tests
- JSON generation with schemas
- Regex pattern matching
- Choice constraints
- Determinism
- Complex schemas
- Error handling
- NULL parameter handling

### pgTAP Tests
Located in `test/pgtap/`:
- `00_setup.sql` - Environment verification
- `01_basic.sql` - Core functionality
- `02_embeddings.sql` - Embedding tests
- `03_async.sql` - Async queue tests
- `04_structured.sql` - Structured generation
- `05_cache_daemon.sql` - Cache and daemon tests

### Performance Benchmarks
- Generation speed by prompt length
- Cache performance (hit vs miss)
- Embedding generation speed
- Structured generation overhead
- Concurrent request handling
- Memory usage

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run integration tests
  run: |
    docker run -d -p 5432:5432 --name test-db -e POSTGRES_PASSWORD=postgres postgres:17
    sleep 10
    ./test_integration_localhost.sh --tap
```

### GitLab CI
```yaml
test:
  script:
    - ./test_e2e_docker.sh --tap
  artifacts:
    reports:
      junit: test-results.xml
```

## Debugging Failed Tests

### Check Extension Installation
```bash
# In container
docker exec -it pg_steadytext_e2e_test bash
su - postgres
psql -c "SELECT * FROM pg_extension WHERE extname = 'pg_steadytext';"
```

### View Python Modules
```bash
# Check Python path
psql -c "DO \$\$ import sys; plpy.notice(sys.path) \$\$ LANGUAGE plpython3u;"

# Check installed packages
pip3 list | grep steadytext
```

### Check Daemon Status
```bash
psql -c "SELECT * FROM steadytext_daemon_status();"
psql -c "SELECT * FROM steadytext_daemon_health;"
```

### View Queue Status
```bash
psql -c "SELECT * FROM steadytext_queue ORDER BY created_at DESC LIMIT 10;"
```

## Common Issues

### Python Module Not Found
- Ensure steadytext is installed in the correct Python environment
- Check PostgreSQL's Python version matches system Python
- Verify module path in `_steadytext_init_python()`

### Model Loading Failures
- Set `STEADYTEXT_USE_FALLBACK_MODEL=true` for compatibility
- Check disk space for model downloads
- Verify internet connectivity

### Cache Issues
- Clear cache with `SELECT steadytext_cache_clear()`
- Check cache statistics with `SELECT * FROM steadytext_cache_stats()`
- Verify cache table exists

### Async Processing
- Ensure worker is running for async tests
- Check queue entries with `SELECT * FROM steadytext_queue`
- Monitor worker logs for errors

## Contributing Tests

When adding new features:
1. Add unit tests to appropriate pgTAP test file
2. Add integration tests to `test_integration_localhost.sh`
3. Update this documentation
4. Ensure tests pass in Docker environment
5. Add performance benchmarks if applicable
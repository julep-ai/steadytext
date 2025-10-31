# pgTAP CI Setup Documentation

## Overview

This document describes how to enable pgTAP tests in GitHub Actions for the pg_steadytext PostgreSQL extension.

## Implementation

### Workflow File

The pgTAP tests workflow is configured in `.github/workflows/pgtap.yml`. Due to GitHub App permission restrictions, this file needs to be manually moved from `pg_steadytext/pgtap-workflow.yml` to `.github/workflows/pgtap.yml`.

### Key Configuration Points

1. **PostgreSQL Service**: Uses PostgreSQL 16 as a service container with health checks
2. **Mini Models**: Sets `STEADYTEXT_USE_MINI_MODELS=true` to prevent test timeouts (as noted in CLAUDE.md)
3. **TAP Output**: Uses `--tap` flag for CI-friendly output format
4. **Path Triggers**: Only runs when pg_steadytext files are modified to save CI resources

### Test Execution

The workflow performs the following steps:

1. Sets up PostgreSQL 16 service container
2. Installs pgTAP from source
3. Installs Python dependencies with uv
4. Installs required PostgreSQL extensions (plpython3u, vector, pgcrypto)
5. Builds and installs pg_steadytext extension
6. Runs pgTAP tests using the `run_pgtap_tests.sh` script

### Environment Variables

Required environment variables for test execution:

- `PGHOST`: localhost
- `PGPORT`: 5432
- `PGUSER`: postgres
- `PGPASSWORD`: password
- `PGDATABASE`: test_postgres
- `STEADYTEXT_USE_MINI_MODELS`: true (prevents timeouts)

### Test Runner Script

The existing `run_pgtap_tests.sh` script handles:

- Database setup and teardown
- Extension installation
- Test execution with TAP or human-readable output
- Test result aggregation

### Manual Setup Required

To enable pgTAP tests in CI:

1. Move the workflow file:
   ```bash
   mv pg_steadytext/pgtap-workflow.yml .github/workflows/pgtap.yml
   ```

2. Commit and push the changes:
   ```bash
   git add .github/workflows/pgtap.yml
   git commit -m "ci: Add pgTAP tests to GitHub Actions"
   git push
   ```

### Alternative: Integration into Existing CI

If preferred, the pgTAP test job can be added to the existing `ci.yml` workflow instead of creating a separate workflow. This would involve adding the `pgtap` job definition to `.github/workflows/ci.yml`.

## Test Coverage

The pgTAP test suite includes 19 test files covering:

- Basic functionality
- Embeddings
- Async operations
- Structured generation
- Cache and daemon management
- Configuration
- Reranking
- AI summarization
- Security validation
- Performance edge cases
- Unsafe mode
- TimescaleDB integration
- Prompt registry

## Troubleshooting

### Common Issues

1. **Test Timeouts**: Ensure `STEADYTEXT_USE_MINI_MODELS=true` is set
2. **Connection Issues**: Verify PostgreSQL service is healthy before running tests
3. **Extension Build Failures**: Check that all build dependencies are installed
4. **pgTAP Not Found**: Ensure pgTAP is properly installed from source

### Debugging

To debug test failures:

1. Check the workflow logs in GitHub Actions
2. Run tests locally with verbose output: `./run_pgtap_tests.sh -v`
3. Use the `--tap` flag for raw TAP output that can be parsed by CI tools

## Future Improvements

1. **Test Result Reporting**: Integrate with GitHub's test reporting features
2. **Parallel Test Execution**: Run test files in parallel to speed up CI
3. **Coverage Reporting**: Add code coverage metrics for the extension
4. **Matrix Testing**: Test against multiple PostgreSQL versions (14, 15, 16)
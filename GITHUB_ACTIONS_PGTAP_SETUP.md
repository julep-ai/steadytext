# GitHub Actions pgTAP Test Setup

This document provides the complete configuration needed to enable pgTAP tests in GitHub Actions for the SteadyText project.

## Required GitHub Actions Workflow

Create the following file at `.github/workflows/test-pgtap.yml`:

```yaml
name: pgTAP Tests

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'pg_steadytext/**'
      - '.github/workflows/test-pgtap.yml'
  pull_request:
    branches: [ main, develop ]
    paths:
      - 'pg_steadytext/**'
      - '.github/workflows/test-pgtap.yml'

jobs:
  pgtap-tests:
    name: Run pgTAP Tests
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:17
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install system dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y \
            postgresql-client-17 \
            postgresql-17-pgtap \
            postgresql-17-python3 \
            postgresql-17-pgvector \
            build-essential \
            git

      - name: Install SteadyText Python package
        run: |
          pip install --upgrade pip uv
          cd ${{ github.workspace }}
          uv pip install -e .
          
      - name: Set up PostgreSQL extensions
        env:
          PGPASSWORD: postgres
        run: |
          # Create extensions in the postgres database
          psql -h localhost -U postgres -d postgres <<EOF
          CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;
          CREATE EXTENSION IF NOT EXISTS vector CASCADE;
          CREATE EXTENSION IF NOT EXISTS pgtap CASCADE;
          EOF

      - name: Build and install pg_steadytext
        env:
          PGPASSWORD: postgres
        run: |
          cd pg_steadytext
          
          # Build the extension
          make clean
          make
          
          # Install the extension files
          sudo make install
          
          # Verify installation
          psql -h localhost -U postgres -d postgres -c "CREATE EXTENSION pg_steadytext CASCADE;"
          psql -h localhost -U postgres -d postgres -c "SELECT steadytext_version();"

      - name: Run pgTAP tests
        env:
          PGPASSWORD: postgres
          PGHOST: localhost
          PGUSER: postgres
          PGPORT: 5432
        run: |
          cd pg_steadytext
          chmod +x run_pgtap_tests.sh
          ./run_pgtap_tests.sh --tap > test-results.tap
          
      - name: Parse TAP results
        if: always()
        run: |
          # Install tap-parser for better output formatting
          npm install -g tap-parser
          
          # Parse and display results
          cat pg_steadytext/test-results.tap | tap-parser -j > test-results.json
          
          # Display summary
          echo "Test Results Summary:"
          cat test-results.json | jq '.stats'
          
          # Check for failures
          if [ $(cat test-results.json | jq '.stats.failures') -gt 0 ]; then
            echo "Tests failed!"
            cat test-results.json | jq '.failures[]'
            exit 1
          fi

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: pgtap-test-results
          path: |
            pg_steadytext/test-results.tap
            test-results.json

  pgtap-tests-docker:
    name: Run pgTAP Tests in Docker
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run end-to-end Docker tests
        run: |
          cd pg_steadytext
          chmod +x test_e2e_docker.sh
          ./test_e2e_docker.sh --pgtap --tap

      - name: Upload Docker test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: docker-pgtap-test-results
          path: pg_steadytext/test-results-docker.tap
```

## Alternative: Matrix Testing for Multiple PostgreSQL Versions

For testing against multiple PostgreSQL versions, use this workflow instead:

```yaml
name: pgTAP Tests Matrix

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'pg_steadytext/**'
      - '.github/workflows/test-pgtap-matrix.yml'
  pull_request:
    branches: [ main, develop ]
    paths:
      - 'pg_steadytext/**'
      - '.github/workflows/test-pgtap-matrix.yml'

jobs:
  pgtap-tests:
    name: pgTAP Tests (PostgreSQL ${{ matrix.postgres }})
    runs-on: ubuntu-latest
    
    strategy:
      fail-fast: false
      matrix:
        postgres: [14, 15, 16, 17]
        
    services:
      postgres:
        image: postgres:${{ matrix.postgres }}
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      # Same steps as above, but adjust package names for PostgreSQL version
      # e.g., postgresql-client-${{ matrix.postgres }}
```

## Integration with Existing CI

If you have an existing CI workflow, add this job to it:

```yaml
  pgtap:
    name: pgTAP Tests
    needs: [other-job]  # Add dependencies as needed
    runs-on: ubuntu-latest
    # ... rest of the job configuration from above
```

## Required Repository Settings

1. **Secrets**: No additional secrets needed (uses postgres/postgres for testing)
2. **Branch Protection**: Consider requiring pgTAP tests to pass before merging
3. **Status Checks**: Add "pgTAP Tests" as a required status check

## Local Testing

To test the workflow locally before pushing:

```bash
# Install act (GitHub Actions local runner)
brew install act  # or your package manager

# Run the workflow locally
act -j pgtap-tests

# Or with specific PostgreSQL version
act -j pgtap-tests --matrix postgres:16
```

## Troubleshooting

### Common Issues and Solutions

1. **pgTAP not found**
   - Ensure the PostgreSQL version matches the pgTAP package version
   - Check that the pgTAP extension is available: `apt-cache search postgresql-.*-pgtap`

2. **Python module not found**
   - Verify Python version compatibility with PostgreSQL
   - Ensure SteadyText is installed in the correct Python environment
   - Check PostgreSQL's python path: `SHOW plpython3.python_path;`

3. **Extension creation fails**
   - Check PostgreSQL logs in the GitHub Actions output
   - Verify all dependencies are installed
   - Ensure the extension files are properly installed with `make install`

4. **Tests timeout**
   - Add timeout configuration to the job: `timeout-minutes: 30`
   - Check for infinite loops in async tests
   - Ensure the worker process starts for async tests

## Monitoring and Notifications

Add these optional features:

### Slack Notifications
```yaml
      - name: Notify Slack on failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "pgTAP tests failed on ${{ github.ref }}",
              "blocks": [{
                "type": "section",
                "text": {
                  "type": "mrkdwn",
                  "text": "pgTAP tests failed in <${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|${{ github.workflow }}>"
                }
              }]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### Test Report Comments on PRs
```yaml
      - name: Comment PR with test results
        if: github.event_name == 'pull_request' && always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('test-results.json', 'utf8'));
            const comment = `## pgTAP Test Results
            
            - **Total Tests**: ${results.stats.total}
            - **Passed**: ${results.stats.passes}
            - **Failed**: ${results.stats.failures}
            - **Duration**: ${results.stats.duration}ms
            
            ${results.stats.failures > 0 ? '❌ Tests failed' : '✅ All tests passed'}`;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

## Next Steps

1. Create the `.github/workflows/test-pgtap.yml` file with the configuration above
2. Test the workflow by creating a pull request that modifies files in `pg_steadytext/`
3. Monitor the Actions tab for the workflow execution
4. Add the workflow badge to your README:
   ```markdown
   ![pgTAP Tests](https://github.com/julep-ai/steadytext/workflows/pgTAP%20Tests/badge.svg)
   ```

## AIDEV-NOTE: Implementation Details

- The workflow uses PostgreSQL 17 by default but can be adapted for other versions
- TAP output mode is used for better CI integration and parsing
- Both direct installation and Docker-based tests are included for comprehensive coverage
- The workflow only triggers on changes to pg_steadytext/ to avoid unnecessary runs
- Test results are uploaded as artifacts for debugging failed runs
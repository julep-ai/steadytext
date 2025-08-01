# pgTAP CI Implementation Steps

Follow these steps to enable pgTAP tests in GitHub Actions for the SteadyText project.

## Step 1: Create the GitHub Actions Workflow

Since the `.github/workflows/` directory doesn't exist in your repository:

1. Create the directory structure:
   ```bash
   mkdir -p .github/workflows
   ```

2. Copy the example workflow:
   ```bash
   cp example-pgtap-workflow.yml .github/workflows/test-pgtap.yml
   ```

3. Commit the workflow:
   ```bash
   git add .github/workflows/test-pgtap.yml
   git commit -m "ci: Add pgTAP tests to GitHub Actions
   
   - Configure PostgreSQL 17 service container
   - Install pgTAP from source for compatibility
   - Build and install pg_steadytext extension
   - Run all pgTAP tests with TAP output
   - Upload test results as artifacts"
   ```

## Step 2: Verify the Workflow

1. Push to a feature branch:
   ```bash
   git push origin feature/pgtap-ci
   ```

2. Create a pull request to trigger the workflow

3. Monitor the Actions tab to see the workflow execution

## Step 3: Add Workflow Badge (Optional)

Add to your README.md:

```markdown
# SteadyText

![pgTAP Tests](https://github.com/julep-ai/steadytext/workflows/pgTAP%20Tests/badge.svg)

... rest of README ...
```

## Step 4: Configure Branch Protection (Optional)

1. Go to Settings → Branches in your repository
2. Add a branch protection rule for `main`
3. Enable "Require status checks to pass before merging"
4. Select "pgTAP Tests" as a required check

## Key Implementation Details

### Why This Configuration?

1. **PostgreSQL 17 Service Container**: Matches the version used in development and provides a clean test environment

2. **pgTAP from Source**: Ubuntu packages might not have pgTAP for PostgreSQL 17, so we build from source

3. **TAP Output Format**: The `--tap` flag provides machine-readable output that can be parsed by CI tools

4. **Path Filtering**: The workflow only runs when files in `pg_steadytext/` are modified to save CI resources

5. **Artifact Upload**: Test results are uploaded for debugging failed runs

### Customization Options

1. **Multiple PostgreSQL Versions**: See the matrix configuration in `GITHUB_ACTIONS_PGTAP_SETUP.md`

2. **Parallel Test Execution**: Split tests into groups for faster execution:
   ```yaml
   strategy:
     matrix:
       test-group: [basic, async, structured, performance]
   ```

3. **Integration with Existing CI**: Add as a job dependency in your existing workflow

### Troubleshooting

If tests fail in CI but pass locally:

1. Check the PostgreSQL version mismatch
2. Verify Python path configuration: The CI uses system Python, not a virtual environment
3. Review the uploaded artifacts for detailed error messages
4. Add debug output to the failing tests

## Step 5: Clean Up

After implementation:

1. Remove the documentation files from the repository root:
   ```bash
   rm GITHUB_ACTIONS_PGTAP_SETUP.md
   rm example-pgtap-workflow.yml
   rm PGTAP_CI_IMPLEMENTATION_STEPS.md
   ```

2. Optionally, move relevant documentation to `pg_steadytext/docs/` or `.github/`

## AIDEV-NOTE: Future Enhancements

Consider these improvements after initial implementation:

1. **Test Parallelization**: Run test files in parallel to reduce CI time
2. **Coverage Reports**: Add code coverage tracking for PL/Python functions
3. **Performance Regression Detection**: Add benchmarks to detect performance issues
4. **Database Migration Testing**: Test upgrade paths between extension versions
5. **Multi-Architecture Testing**: Test on ARM64 runners for broader compatibility
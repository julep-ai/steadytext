# DevContainer CI/CD Documentation

This document explains how the DevContainer CI/CD setup works and how to use it.

## GitHub Actions Workflows

We have two GitHub Actions workflows for testing the devcontainer:

### 1. `devcontainer-ci.yml` (Full CI)
- Builds and pushes the PostgreSQL image to GitHub Container Registry
- Tests the complete devcontainer setup with all services
- Uses image caching for faster subsequent builds
- Suitable for production-ready CI/CD pipelines

### 2. `devcontainer-simple.yml` (Simple CI)
- Builds the PostgreSQL image locally without pushing to registry
- Runs a comprehensive test suite inside the devcontainer
- Faster for simple validation of changes
- Good for PR validation and quick checks

## Local Development

### Building the PostgreSQL Image

The devcontainer requires a PostgreSQL image with the pg_steadytext extension pre-installed. You have several options:

#### Option 1: Build locally (recommended for development)
```bash
# Build the postgres image
docker build -t steadytext_devcontainer-postgres:latest ./pg_steadytext

# Or use docker-compose with build override
docker compose -f .devcontainer/docker-compose.yml -f .devcontainer/docker-compose.build.yml build
```

#### Option 2: Use pre-built image from GitHub Container Registry
```bash
# Pull from registry (if available)
docker pull ghcr.io/YOUR_ORG/steadytext_devcontainer-postgres:latest
docker tag ghcr.io/YOUR_ORG/steadytext_devcontainer-postgres:latest steadytext_devcontainer-postgres:latest
```

### Starting the DevContainer

1. **VS Code**: Open the folder in VS Code and use "Reopen in Container"
2. **GitHub Codespaces**: Create a new codespace from the repository
3. **Manual**: Use docker compose commands

### Docker Compose Files

- `docker-compose.yml`: Main configuration with two services (dev and postgres)
- `docker-compose.build.yml`: Override to build postgres image locally
- `docker-compose.override.yml.example`: Example for custom configurations

### Testing the Setup

Run the test script inside the devcontainer:
```bash
# If the test script was created by CI
bash .devcontainer/test-in-container.sh

# Or run individual tests
uv --version
docker compose exec postgres pg_isready -U postgres
PGPASSWORD=password psql -h postgres -U postgres -d postgres -c "SELECT steadytext_version();"
```

## Troubleshooting

### PostgreSQL Image Not Found

If you see an error about `steadytext_devcontainer-postgres:latest` not found:

1. Build it locally:
   ```bash
   docker build -t steadytext_devcontainer-postgres:latest ./pg_steadytext
   ```

2. Or use the build override:
   ```bash
   docker compose -f .devcontainer/docker-compose.yml -f .devcontainer/docker-compose.build.yml up
   ```

### PostgreSQL Not Ready

If PostgreSQL doesn't become ready:

1. Check container logs:
   ```bash
   docker logs pg_steadytext_db
   ```

2. Ensure the extension built correctly:
   ```bash
   docker exec pg_steadytext_db ls -la /usr/share/postgresql/17/extension/pg_steadytext*
   ```

3. Try rebuilding with fallback models:
   ```bash
   docker build --build-arg STEADYTEXT_USE_FALLBACK_MODEL=true -t steadytext_devcontainer-postgres:latest ./pg_steadytext
   ```

### CI Failures

If GitHub Actions fail:

1. Check the workflow logs for specific error messages
2. Try running the same commands locally
3. Ensure all file paths are correct (CI runs from repository root)
4. Verify Docker BuildKit is enabled: `export DOCKER_BUILDKIT=1`

## Environment Variables

Important environment variables for CI and testing:

- `STEADYTEXT_USE_FALLBACK_MODEL=true` - Use known working models
- `STEADYTEXT_USE_MINI_MODELS=true` - Use smaller models for faster testing
- `DOCKER_BUILDKIT=1` - Enable BuildKit for better caching
- `COMPOSE_DOCKER_CLI_BUILD=1` - Use Docker CLI for compose builds

## Caching Strategy

The CI workflows use multiple caching layers:

1. **Docker Layer Cache**: Via BuildKit inline cache
2. **GitHub Actions Cache**: Using `type=gha` cache backend
3. **GitHub Container Registry**: For sharing built images between runs
4. **UV Cache**: Python package caching (when using cache-dependency-glob)

## Security Notes

- The PostgreSQL password is hardcoded as "password" for development only
- GitHub Actions use `GITHUB_TOKEN` for registry authentication (automatic)
- Sensitive operations require appropriate repository permissions

## Future Improvements

- [ ] Add matrix testing for multiple PostgreSQL versions
- [ ] Implement automated performance benchmarks
- [ ] Add security scanning for containers
- [ ] Create separate staging/production configurations
- [ ] Add integration tests for all extension functions
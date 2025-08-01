#!/bin/bash
# Build script for devcontainer with BuildKit enabled

set -e

echo "Building devcontainer with BuildKit caching..."

# Ensure BuildKit is enabled
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Create cache directories if they don't exist
mkdir -p /tmp/.buildx-cache
mkdir -p /tmp/.buildx-cache-pg

# Build using docker-compose with BuildKit
cd "$(dirname "$0")"
docker compose build --no-cache=false "$@"

echo "Build complete!"
echo ""
echo "To rebuild without cache, run: docker compose build --no-cache"
echo "To clear BuildKit cache, run: docker builder prune"
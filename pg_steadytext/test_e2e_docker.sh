#!/bin/bash
# End-to-end test script for pg_steadytext using Docker PostgreSQL images
# This script builds the extension in a Docker container and runs tests

set -euo pipefail

# Configuration
DOCKER_IMAGE="${DOCKER_IMAGE:-postgres:18}"
HOST_PORT="${HOST_PORT:-5432}"
STEADYTEXT_USE_MINI_MODELS="${STEADYTEXT_USE_MINI_MODELS:-true}"
CONTAINER_NAME="pg_steadytext_e2e_test"
TEST_SCRIPT="test_integration_localhost.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
VERBOSE=false
RUN_PGTAP=false
BENCHMARK=false
KEEP_CONTAINER=false

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -v, --verbose       Enable verbose output"
    echo "  --pgtap             Run pgTAP tests"
    echo "  --benchmark         Run performance benchmarks"
    echo "  --keep-container    Keep container running after tests"
    echo "  --help              Show this help message"
    exit 1
}

# Process arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --pgtap)
            RUN_PGTAP=true
            shift
            ;;
        --benchmark)
            BENCHMARK=true
            shift
            ;;
        --keep-container)
            KEEP_CONTAINER=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

log() {
    echo -e "$1"
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "$1"
    fi
}

# Cleanup function
cleanup() {
    if [ "$KEEP_CONTAINER" = false ]; then
        log "\n${BLUE}Cleaning up...${NC}"
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    else
        log "\n${YELLOW}Container kept running:${NC} $CONTAINER_NAME"
        log "To connect: docker exec -it $CONTAINER_NAME psql -U postgres"
        log "To clean up: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    log "${BLUE}=== pg_steadytext End-to-End Docker Test ===${NC}"
    log "Docker image: $DOCKER_IMAGE"
    log "Host port: $HOST_PORT"
    log "Mini models: $STEADYTEXT_USE_MINI_MODELS"
    
    # Stop and remove existing container
    log_verbose "\n${BLUE}Removing existing container if any...${NC}"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    
    # Start PostgreSQL container
    log "\n${BLUE}Starting PostgreSQL container...${NC}"
    docker run -d \
        --name "$CONTAINER_NAME" \
        -e POSTGRES_PASSWORD=postgres \
        -e STEADYTEXT_USE_MINI_MODELS="$STEADYTEXT_USE_MINI_MODELS" \
        -p "${HOST_PORT}:5432" \
        "$DOCKER_IMAGE"
    
    # Wait for PostgreSQL to be ready
    log "Waiting for PostgreSQL to be ready..."
    for i in {1..30}; do
        if docker exec "$CONTAINER_NAME" pg_isready -U postgres &>/dev/null; then
            log "${GREEN}✓${NC} PostgreSQL is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            log "${RED}Error:${NC} PostgreSQL failed to start"
            exit 1
        fi
        sleep 1
    done

    # Detect server major version for package selection
    PG_MAJOR=$(docker exec -u postgres "$CONTAINER_NAME" psql -tAc "SELECT current_setting('server_version_num')::int / 10000" | tr -d '[:space:]')
    if [[ -z "$PG_MAJOR" || ! "$PG_MAJOR" =~ ^[0-9]+$ ]]; then
        log "${RED}Error:${NC} Could not detect PostgreSQL major version"
        exit 1
    fi
    log "Detected PostgreSQL major version: $PG_MAJOR"
    
    # Install required system packages
    log "\n${BLUE}Installing system dependencies...${NC}"
    docker exec -u root "$CONTAINER_NAME" apt-get update -qq
    docker exec -u root "$CONTAINER_NAME" apt-get install -y -qq \
        build-essential \
        postgresql-server-dev-${PG_MAJOR} \
        postgresql-plpython3-${PG_MAJOR} \
        postgresql-${PG_MAJOR}-pgvector \
        python3-dev \
        python3-pip \
        git \
        bc
    
    # Install Python packages
    log "\n${BLUE}Installing Python packages...${NC}"
    docker exec -u root "$CONTAINER_NAME" pip3 install --quiet \
        --break-system-packages \
        steadytext \
        pyzmq \
        numpy
    
    # Copy extension source to container
    log "\n${BLUE}Copying extension source to container...${NC}"
    docker exec -u root "$CONTAINER_NAME" mkdir -p /tmp/pg_steadytext
    tar --exclude=".git" --exclude=".claude" -cf - . | \
        docker exec -i "$CONTAINER_NAME" tar -xf - -C /tmp/pg_steadytext
    docker exec -u root "$CONTAINER_NAME" chown -R postgres:postgres /tmp/pg_steadytext
    
    # Build and install the extension
    log "\n${BLUE}Building and installing extension...${NC}"
    docker exec -u postgres "$CONTAINER_NAME" bash -c "
        cd /tmp/pg_steadytext/pg_steadytext && 
        make clean && 
        make
    "
    docker exec -u root "$CONTAINER_NAME" bash -c "
        cd /tmp/pg_steadytext/pg_steadytext &&
        make install
    "
    
    # Verify installation
    log "\n${BLUE}Verifying installation...${NC}"
    create_output=$(docker exec -u postgres "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -tAc "
        CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;
        CREATE EXTENSION IF NOT EXISTS vector CASCADE;
        CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;
    " 2>&1) || true
    if echo "$create_output" | grep -q "ERROR\\|FATAL"; then
        log "${RED}Error:${NC} Failed to create extension"
        log "$create_output"
        exit 1
    else
        log "${GREEN}✓${NC} Extension created successfully"
    fi
    
    # Check extension version
    version=$(docker exec -u postgres "$CONTAINER_NAME" psql -tAc "SELECT steadytext_version()")
    log "${GREEN}✓${NC} Extension version: $version"
    
    # Install pgTAP if requested
    if [ "$RUN_PGTAP" = true ]; then
        log "\n${BLUE}Installing pgTAP...${NC}"
        docker exec -u root "$CONTAINER_NAME" apt-get install -y -qq postgresql-${PG_MAJOR}-pgtap
    fi
    
    # Run integration tests
    log "\n${BLUE}Running integration tests...${NC}"
    
    # Build test command
    TEST_CMD="export STEADYTEXT_USE_MINI_MODELS=$STEADYTEXT_USE_MINI_MODELS && cd /tmp/pg_steadytext/pg_steadytext && ./test_integration_localhost.sh"
    if [ "$VERBOSE" = true ]; then
        TEST_CMD="$TEST_CMD -v"
    fi
    if [ "$RUN_PGTAP" = true ]; then
        TEST_CMD="$TEST_CMD --pgtap"
    fi
    if [ "$BENCHMARK" = true ]; then
        TEST_CMD="$TEST_CMD --benchmark"
    fi
    
    # Run tests
    if docker exec -u postgres "$CONTAINER_NAME" bash -c "$TEST_CMD"; then
        log "\n${GREEN}✓ All tests passed!${NC}"
        exit_code=0
    else
        log "\n${RED}✗ Some tests failed${NC}"
        exit_code=1
    fi
    
    if [ "$exit_code" -eq 0 ]; then
        # Quick smoke test
        log "\n${BLUE}Running quick smoke test...${NC}"
        result=$(docker exec -u postgres "$CONTAINER_NAME" bash -lc \
            "export STEADYTEXT_USE_MINI_MODELS=$STEADYTEXT_USE_MINI_MODELS && timeout 180 psql -tAc \"SELECT steadytext_generate('Hello Docker!', 10)\"" 2>/dev/null || true)
        if [ -n "$result" ]; then
            log "${GREEN}✓${NC} Text generation works: ${result:0:50}..."
        else
            log "${RED}✗${NC} Text generation failed or timed out"
            exit_code=1
        fi

        # Test daemon status function shape
        daemon_status_ok=$(docker exec -u postgres "$CONTAINER_NAME" psql -tAc \
            "SELECT COALESCE((SELECT status FROM steadytext_daemon_status() LIMIT 1), '') <> ''")
        if [ "$daemon_status_ok" = "t" ]; then
            daemon_status=$(docker exec -u postgres "$CONTAINER_NAME" psql -tAc \
                "SELECT status FROM steadytext_daemon_status() LIMIT 1")
            log "${GREEN}✓${NC} Daemon status check works (status: $daemon_status)"
        else
            log "${RED}✗${NC} Daemon status check failed"
            exit_code=1
        fi
    else
        log "\n${YELLOW}Skipping smoke and daemon checks due to earlier integration failures.${NC}"
    fi
    
    # Show container logs if verbose
    if [ "$VERBOSE" = true ]; then
        log "\n${BLUE}Container logs:${NC}"
        docker logs "$CONTAINER_NAME" | tail -20
    fi
    
    return $exit_code
}

# Run main function
main

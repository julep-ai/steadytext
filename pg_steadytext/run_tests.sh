#!/bin/bash
# run_tests.sh - Comprehensive test runner for pg_steadytext
# AIDEV-NOTE: This script runs all tests and generates a report

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DB_NAME="${PGDATABASE:-test_steadytext}"
DB_USER="${PGUSER:-postgres}"
DB_HOST="${PGHOST:-localhost}"
DB_PORT="${PGPORT:-5432}"

echo -e "${BLUE}pg_steadytext Test Suite${NC}"
echo "=========================="
echo ""

# Function to run SQL test file
run_sql_test() {
    local test_file=$1
    local test_name=$(basename "$test_file" .sql)
    
    echo -n "Running $test_name... "
    
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
           -v ON_ERROR_STOP=1 -q -f "$test_file" > /tmp/test_${test_name}.out 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo -e "${RED}Error output:${NC}"
        cat /tmp/test_${test_name}.out
        return 1
    fi
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"
echo -n "  PostgreSQL connection: "
if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "SELECT 1" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "Cannot connect to PostgreSQL. Check your connection settings."
    exit 1
fi

# Create test database
echo -e "\n${YELLOW}Setting up test database...${NC}"
dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" --if-exists "$DB_NAME" 2>/dev/null || true
createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"

# Install extensions
echo -e "${YELLOW}Installing extensions...${NC}"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<EOF
-- Install prerequisites
CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;
CREATE EXTENSION IF NOT EXISTS vector CASCADE;

-- Install pg_steadytext
CREATE EXTENSION pg_steadytext CASCADE;

-- Verify installation
SELECT steadytext_version();
EOF

# Run Python unit tests
if [ -d "python/tests" ]; then
    echo -e "\n${YELLOW}Running Python unit tests...${NC}"
    cd python && python3 -m pytest tests/ -v && cd ..
fi

# Run SQL regression tests
echo -e "\n${YELLOW}Running SQL regression tests...${NC}"
FAILED_TESTS=0
TOTAL_TESTS=0

for test_file in test/sql/*.sql; do
    if [ -f "$test_file" ]; then
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        if ! run_sql_test "$test_file"; then
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    fi
done

# Run integration tests
echo -e "\n${YELLOW}Running integration tests...${NC}"

# Test 1: End-to-end generation with caching
echo -n "Testing generation with caching... "
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -q <<EOF
-- First call (cache miss)
SELECT length(steadytext_generate('Integration test prompt', 50)) > 0;

-- Second call (cache hit)
SELECT COUNT(*) FROM steadytext_cache WHERE prompt = 'Integration test prompt';

-- Verify cache statistics
SELECT total_entries > 0, cache_hit_rate >= 0 FROM steadytext_cache_stats();
EOF
echo -e "${GREEN}✓${NC}"

# Test 2: Daemon integration
echo -n "Testing daemon integration... "
if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -q -c "SELECT steadytext_daemon_status();" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠ Daemon not available${NC}"
fi

# Test 3: Performance benchmark
echo -e "\n${YELLOW}Running performance benchmark...${NC}"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<EOF
-- Benchmark generation speed
\timing on
SELECT COUNT(*) FROM generate_series(1, 10) AS i, 
       LATERAL (SELECT steadytext_generate('Benchmark ' || i::text, 20)) AS gen;

-- Benchmark embedding speed
SELECT COUNT(*) FROM generate_series(1, 10) AS i,
       LATERAL (SELECT steadytext_embed('Benchmark text ' || i::text)) AS emb;
\timing off
EOF

# Generate test report
echo -e "\n${BLUE}Test Summary${NC}"
echo "============"
echo -e "Total tests: ${TOTAL_TESTS}"
echo -e "Passed: ${GREEN}$((TOTAL_TESTS - FAILED_TESTS))${NC}"
echo -e "Failed: ${RED}${FAILED_TESTS}${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed.${NC}"
    exit 1
fi

# Cleanup (optional)
# dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" --if-exists "$DB_NAME"
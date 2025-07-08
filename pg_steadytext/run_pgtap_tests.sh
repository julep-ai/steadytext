#!/bin/bash
# run_pgtap_tests.sh - Run pgTAP tests for pg_steadytext
# AIDEV-NOTE: This script runs pgTAP tests and provides TAP output

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

# Parse command line arguments
VERBOSE=false
TAP_OUTPUT=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--tap)
            TAP_OUTPUT=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -v, --verbose    Show verbose output"
            echo "  -t, --tap        Output raw TAP format (for CI)"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$TAP_OUTPUT" = false ]; then
    echo -e "${BLUE}pg_steadytext pgTAP Test Suite${NC}"
    echo "=============================="
    echo ""
fi

# Function to run pgTAP test file
run_pgtap_test() {
    local test_file=$1
    local test_name=$(basename "$test_file" .sql)
    
    if [ "$TAP_OUTPUT" = true ]; then
        # Raw TAP output for CI
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
             -v ON_ERROR_STOP=1 -q -f "$test_file"
    else
        # Pretty output for humans
        echo -n "Running $test_name... "
        
        if output=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
                        -v ON_ERROR_STOP=1 -q -f "$test_file" 2>&1); then
            # Check if all tests passed
            if echo "$output" | grep -q "# Looks like you failed" || \
               echo "$output" | grep -q "^not ok" || \
               echo "$output" | grep -q "FAILED" || \
               echo "$output" | grep -q "ERROR:"; then
                echo -e "${RED}✗ FAILED${NC}"
                if [ "$VERBOSE" = true ]; then
                    echo "$output"
                fi
                return 1
            else
                echo -e "${GREEN}✓ PASSED${NC}"
                if [ "$VERBOSE" = true ]; then
                    echo "$output"
                fi
                return 0
            fi
        else
            echo -e "${RED}✗ ERROR${NC}"
            echo -e "${RED}Error output:${NC}"
            echo "$output"
            return 1
        fi
    fi
}

# Check prerequisites
if [ "$TAP_OUTPUT" = false ]; then
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    echo -n "  PostgreSQL connection: "
fi
if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -c "SELECT 1" >/dev/null 2>&1; then
    [ "$TAP_OUTPUT" = false ] && echo -e "${GREEN}✓${NC}"
else
    [ "$TAP_OUTPUT" = false ] && echo -e "${RED}✗${NC}"
    echo "Cannot connect to PostgreSQL. Check your connection settings." >&2
    exit 1
fi

# Create test database
if [ "$TAP_OUTPUT" = false ]; then
    echo -e "\n${YELLOW}Setting up test database...${NC}"
fi
dropdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" --if-exists "$DB_NAME" 2>/dev/null || true
createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"

# Install extensions
if [ "$TAP_OUTPUT" = false ]; then
    echo -e "${YELLOW}Installing extensions...${NC}"
fi
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" <<EOF >/dev/null 2>&1
-- Install prerequisites
CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;
CREATE EXTENSION IF NOT EXISTS vector CASCADE;

-- Install pgTAP
CREATE EXTENSION IF NOT EXISTS pgtap CASCADE;

-- Install pg_steadytext
CREATE EXTENSION pg_steadytext CASCADE;

-- Verify installation
SELECT steadytext_version();
EOF

# Run pgTAP tests
if [ "$TAP_OUTPUT" = false ]; then
    echo -e "\n${YELLOW}Running pgTAP tests...${NC}"
fi

FAILED_TESTS=0
TOTAL_TESTS=0

# Check if we have pgTAP test files
if ls test/pgtap/*.sql >/dev/null 2>&1; then
    for test_file in test/pgtap/*.sql; do
        if [ -f "$test_file" ]; then
            TOTAL_TESTS=$((TOTAL_TESTS + 1))
            if ! run_pgtap_test "$test_file"; then
                FAILED_TESTS=$((FAILED_TESTS + 1))
            fi
        fi
    done
else
    echo "No pgTAP tests found in test/pgtap/ directory" >&2
    exit 1
fi

# Summary (only for non-TAP output)
if [ "$TAP_OUTPUT" = false ]; then
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
fi

# For TAP output, exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    exit 0
else
    exit 1
fi

# AIDEV-NOTE: This script provides two modes:
# 1. Human-readable output (default) with colors and summaries
# 2. TAP output mode (-t flag) for CI integration
# The TAP mode outputs raw Test Anything Protocol format which can be
# consumed by prove, Jenkins TAP plugin, and other TAP consumers
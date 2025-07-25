#!/bin/bash
# Integration test script for pg_steadytext extension
# Connects from localhost to a PostgreSQL instance (typically in Docker)
# Usage: ./test_integration_localhost.sh [options]

set -euo pipefail

# Default connection parameters
DEFAULT_HOST="localhost"
DEFAULT_PORT="5432"
DEFAULT_USER="postgres"
DEFAULT_PASSWORD="postgres"
DEFAULT_DB="postgres"

# Test configuration
TEST_DB="test_steadytext_integration"
VERBOSE=false
RUN_PGTAP=false
TAP_FORMAT=false
BENCHMARK=false

# Statistics
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --host HOST          PostgreSQL host (default: $DEFAULT_HOST)"
    echo "  -p, --port PORT          PostgreSQL port (default: $DEFAULT_PORT)"
    echo "  -U, --user USER          PostgreSQL user (default: $DEFAULT_USER)"
    echo "  -W, --password PASSWORD  PostgreSQL password (default: $DEFAULT_PASSWORD)"
    echo "  -d, --database DATABASE  PostgreSQL database (default: $DEFAULT_DB)"
    echo "  -v, --verbose            Enable verbose output"
    echo "  --pgtap                  Run pgTAP tests"
    echo "  --tap                    Output in TAP format (for CI)"
    echo "  --benchmark              Run performance benchmarks"
    echo "  --help                   Show this help message"
    exit 1
}

# Process arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            PGHOST="$2"
            shift 2
            ;;
        -p|--port)
            PGPORT="$2"
            shift 2
            ;;
        -U|--user)
            PGUSER="$2"
            shift 2
            ;;
        -W|--password)
            PGPASSWORD="$2"
            shift 2
            ;;
        -d|--database)
            PGDATABASE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --pgtap)
            RUN_PGTAP=true
            shift
            ;;
        --tap)
            TAP_FORMAT=true
            shift
            ;;
        --benchmark)
            BENCHMARK=true
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

# Set connection parameters from environment or defaults
export PGHOST="${PGHOST:-$DEFAULT_HOST}"
export PGPORT="${PGPORT:-$DEFAULT_PORT}"
export PGUSER="${PGUSER:-$DEFAULT_USER}"
export PGPASSWORD="${PGPASSWORD:-$DEFAULT_PASSWORD}"
export PGDATABASE="${PGDATABASE:-$DEFAULT_DB}"

# Helper functions
log() {
    if [ "$TAP_FORMAT" = true ]; then
        return
    fi
    echo -e "$1"
}

log_verbose() {
    if [ "$VERBOSE" = true ] && [ "$TAP_FORMAT" = false ]; then
        echo -e "$1"
    fi
}

run_sql() {
    local sql="$1"
    local db="${2:-$PGDATABASE}"
    if [ "$VERBOSE" = true ]; then
        log_verbose "${BLUE}SQL:${NC} $sql"
    fi
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$db" -tAc "$sql"
}

run_sql_file() {
    local file="$1"
    local db="${2:-$PGDATABASE}"
    if [ "$VERBOSE" = true ]; then
        log_verbose "${BLUE}Running SQL file:${NC} $file"
    fi
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$db" -f "$file"
}

test_passed() {
    local test_name="$1"
    ((TESTS_PASSED++))
    if [ "$TAP_FORMAT" = true ]; then
        echo "ok $TESTS_RUN - $test_name"
    else
        log "${GREEN}✓${NC} $test_name"
    fi
}

test_failed() {
    local test_name="$1"
    local error="$2"
    ((TESTS_FAILED++))
    if [ "$TAP_FORMAT" = true ]; then
        echo "not ok $TESTS_RUN - $test_name"
        echo "# Error: $error"
    else
        log "${RED}✗${NC} $test_name"
        log "  ${RED}Error:${NC} $error"
    fi
}

test_skipped() {
    local test_name="$1"
    local reason="$2"
    ((TESTS_SKIPPED++))
    if [ "$TAP_FORMAT" = true ]; then
        echo "ok $TESTS_RUN - $test_name # SKIP $reason"
    else
        log "${YELLOW}⊘${NC} $test_name (skipped: $reason)"
    fi
}

run_test() {
    local test_name="$1"
    local expected="$2"
    local sql="$3"
    ((TESTS_RUN++))
    
    log_verbose "\n${BLUE}Running test:${NC} $test_name"
    
    local result
    if result=$(run_sql "$sql" "$TEST_DB" 2>&1); then
        if [[ "$result" == *"$expected"* ]]; then
            test_passed "$test_name"
        else
            test_failed "$test_name" "Expected '$expected', got '$result'"
        fi
    else
        test_failed "$test_name" "$result"
    fi
}

# Check prerequisites
check_prerequisites() {
    log "\n${BLUE}Checking prerequisites...${NC}"
    
    # Check psql availability
    if ! command -v psql &> /dev/null; then
        log "${RED}Error:${NC} psql command not found. Please install PostgreSQL client."
        exit 1
    fi
    
    # Test connection
    if ! run_sql "SELECT 1" &> /dev/null; then
        log "${RED}Error:${NC} Cannot connect to PostgreSQL at $PGHOST:$PGPORT"
        log "Please check your connection parameters and ensure PostgreSQL is running."
        exit 1
    fi
    
    log "${GREEN}✓${NC} Prerequisites satisfied"
}

# Create test database
setup_test_db() {
    log "\n${BLUE}Setting up test database...${NC}"
    
    # Drop existing test database if exists
    run_sql "DROP DATABASE IF EXISTS $TEST_DB" &> /dev/null || true
    
    # Create test database
    if ! run_sql "CREATE DATABASE $TEST_DB"; then
        log "${RED}Error:${NC} Failed to create test database"
        exit 1
    fi
    
    # Install required extensions
    log_verbose "Installing required extensions..."
    run_sql "CREATE EXTENSION IF NOT EXISTS plpython3u" "$TEST_DB"
    run_sql "CREATE EXTENSION IF NOT EXISTS pg_steadytext" "$TEST_DB"
    
    # Check if pgvector is available and install it
    if run_sql "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'" | grep -q 1; then
        run_sql "CREATE EXTENSION IF NOT EXISTS vector" "$TEST_DB"
    else
        log "${YELLOW}Warning:${NC} pgvector extension not available, some tests will be skipped"
    fi
    
    log "${GREEN}✓${NC} Test database ready"
}

# Clean up test database
cleanup_test_db() {
    log_verbose "\n${BLUE}Cleaning up...${NC}"
    run_sql "DROP DATABASE IF EXISTS $TEST_DB" &> /dev/null || true
}

# Initialize Python environment in test database
init_python_env() {
    log_verbose "Initializing Python environment..."
    if ! run_sql "SELECT _steadytext_init_python()" "$TEST_DB" &> /dev/null; then
        log "${YELLOW}Warning:${NC} Python initialization had issues, some tests may fail"
    fi
}

# Main test execution
main() {
    if [ "$TAP_FORMAT" = true ]; then
        echo "TAP version 13"
    fi
    
    log "${BLUE}=== pg_steadytext Integration Tests ===${NC}"
    log "Host: $PGHOST:$PGPORT"
    log "User: $PGUSER"
    log "Database: $PGDATABASE"
    
    # Check prerequisites
    check_prerequisites
    
    # Set up test database
    setup_test_db
    
    # Initialize Python environment
    init_python_env
    
    # Trap to ensure cleanup
    trap cleanup_test_db EXIT
    
    # Start testing
    log "\n${BLUE}Running tests...${NC}"
    
    # Basic functionality tests
    log "\n${BLUE}=== Basic Functionality Tests ===${NC}"
    
    # Test extension version
    run_test "Extension version check" \
        "1.4.1" \
        "SELECT steadytext_version()"
    
    # Test basic text generation
    run_test "Basic text generation" \
        "Hello" \
        "SELECT LEFT(steadytext_generate('Hello world', 10), 5)"
    
    # Test that generation is deterministic
    local gen1=$(run_sql "SELECT steadytext_generate('Test prompt', 50)" "$TEST_DB")
    local gen2=$(run_sql "SELECT steadytext_generate('Test prompt', 50)" "$TEST_DB")
    if [ "$gen1" = "$gen2" ]; then
        test_passed "Generation determinism"
    else
        test_failed "Generation determinism" "Results differ: '$gen1' vs '$gen2'"
    fi
    ((TESTS_RUN++))
    
    # Test embedding generation
    if run_sql "SELECT 1 FROM pg_extension WHERE extname = 'vector'" "$TEST_DB" | grep -q 1; then
        run_test "Basic embedding generation" \
            "1024" \
            "SELECT array_length(steadytext_embed('Test text')::float[], 1)"
        
        # Test embedding determinism
        local emb1=$(run_sql "SELECT steadytext_embed('Test embedding')::text" "$TEST_DB")
        local emb2=$(run_sql "SELECT steadytext_embed('Test embedding')::text" "$TEST_DB")
        if [ "$emb1" = "$emb2" ]; then
            test_passed "Embedding determinism"
        else
            test_failed "Embedding determinism" "Embeddings differ"
        fi
        ((TESTS_RUN++))
        
        # Test embedding normalization
        run_test "Embedding normalization" \
            "1" \
            "WITH e AS (SELECT steadytext_embed('Normalize test')::float[] as embedding)
             SELECT ROUND(sqrt(sum(v*v))::numeric, 2)::text FROM e, unnest(embedding) v"
    else
        test_skipped "Basic embedding generation" "pgvector not installed"
        test_skipped "Embedding determinism" "pgvector not installed"
        test_skipped "Embedding normalization" "pgvector not installed"
        ((TESTS_RUN+=3))
    fi
    
    # Test with NULL inputs
    run_test "NULL input handling for generate" \
        "error" \
        "SELECT steadytext_generate(NULL, 10)"
    
    run_test "NULL input handling for embed" \
        "error" \
        "SELECT steadytext_embed(NULL)"
    
    # Test with empty inputs
    run_test "Empty input for generate" \
        "" \
        "SELECT LENGTH(steadytext_generate('', 10))::text"
    
    # Test with special characters
    run_test "Special characters in generation" \
        "t" \
        "SELECT steadytext_generate(E'Test\nwith\nnewlines\tand\ttabs', 10) IS NOT NULL"
    
    # Test max_tokens parameter
    local short_gen=$(run_sql "SELECT LENGTH(steadytext_generate('Generate text', 10))" "$TEST_DB")
    local long_gen=$(run_sql "SELECT LENGTH(steadytext_generate('Generate text', 100))" "$TEST_DB")
    if [ "$long_gen" -gt "$short_gen" ]; then
        test_passed "max_tokens parameter works"
    else
        test_failed "max_tokens parameter works" "Short: $short_gen, Long: $long_gen"
    fi
    ((TESTS_RUN++))
    
    # Test daemon status
    run_test "Daemon status check" \
        "t" \
        "SELECT (steadytext_daemon_status()).daemon_available IS NOT NULL"
    
    # Test Python initialization
    run_test "Python environment initialized" \
        "t" \
        "SELECT _steadytext_is_initialized()"
    
    # Cache functionality tests
    log "\n${BLUE}=== Cache Functionality Tests ===${NC}"
    
    # Clear cache first
    run_sql "SELECT steadytext_cache_clear()" "$TEST_DB" &> /dev/null
    
    # Test cache miss and hit
    local first_call=$(run_sql "SELECT steadytext_generate('Cache test prompt', 20)" "$TEST_DB")
    local cache_size_after_first=$(run_sql "SELECT (steadytext_cache_stats()).total_entries" "$TEST_DB")
    
    # Second call should hit cache
    local second_call=$(run_sql "SELECT steadytext_generate('Cache test prompt', 20)" "$TEST_DB")
    local cache_size_after_second=$(run_sql "SELECT (steadytext_cache_stats()).total_entries" "$TEST_DB")
    
    if [ "$first_call" = "$second_call" ] && [ "$cache_size_after_first" = "$cache_size_after_second" ]; then
        test_passed "Cache hit for identical requests"
    else
        test_failed "Cache hit for identical requests" "Cache size changed or results differ"
    fi
    ((TESTS_RUN++))
    
    # Test cache stats
    run_test "Cache statistics available" \
        "t" \
        "SELECT (steadytext_cache_stats()).total_entries >= 0"
    
    # Test cache clear
    run_sql "SELECT steadytext_cache_clear()" "$TEST_DB"
    run_test "Cache clear works" \
        "0" \
        "SELECT (steadytext_cache_stats()).total_entries"
    
    # Test cache with different parameters
    run_sql "SELECT steadytext_generate('Test A', 10)" "$TEST_DB"
    run_sql "SELECT steadytext_generate('Test A', 20)" "$TEST_DB"
    run_sql "SELECT steadytext_generate('Test B', 10)" "$TEST_DB"
    
    run_test "Cache differentiates by parameters" \
        "3" \
        "SELECT (steadytext_cache_stats()).total_entries"
    
    # Test cache eviction settings
    run_test "Cache eviction settings exist" \
        "t" \
        "SELECT current_setting('pg_steadytext.cache_max_entries')::int > 0"
    
    # Test extended cache stats
    run_test "Extended cache statistics" \
        "t" \
        "SELECT (steadytext_cache_stats_extended()).avg_access_count >= 0"
    
    # Test cache usage analysis
    run_test "Cache usage analysis" \
        "t" \
        "SELECT COUNT(*) >= 0 FROM steadytext_cache_analyze_usage()"
    
    # Test manual cache eviction
    local entries_before=$(run_sql "SELECT (steadytext_cache_stats()).total_entries" "$TEST_DB")
    if [ "$entries_before" -gt 0 ]; then
        run_test "Manual cache eviction" \
            "t" \
            "SELECT (steadytext_cache_evict_by_age(1, '1 day'::interval)).evicted_count >= 0"
    else
        test_skipped "Manual cache eviction" "No cache entries to evict"
        ((TESTS_RUN++))
    fi
    
    # Test cache for embeddings
    if run_sql "SELECT 1 FROM pg_extension WHERE extname = 'vector'" "$TEST_DB" | grep -q 1; then
        run_sql "SELECT steadytext_cache_clear()" "$TEST_DB"
        run_sql "SELECT steadytext_embed('Cache embedding test')" "$TEST_DB"
        local embed_cache_size=$(run_sql "SELECT (steadytext_cache_stats()).total_entries" "$TEST_DB")
        
        run_test "Embedding cache works" \
            "1" \
            "SELECT $embed_cache_size"
    else
        test_skipped "Embedding cache works" "pgvector not installed"
        ((TESTS_RUN++))
    fi
    
    # Async queue tests
    log "\n${BLUE}=== Async Queue Tests ===${NC}"
    
    # Clear any existing queue entries
    run_sql "DELETE FROM steadytext_queue" "$TEST_DB" &> /dev/null
    
    # Test async generation request
    local async_id=$(run_sql "SELECT steadytext_generate_async('Async test', 20)" "$TEST_DB")
    if [[ "$async_id" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
        test_passed "Async generation returns UUID"
    else
        test_failed "Async generation returns UUID" "Got: $async_id"
    fi
    ((TESTS_RUN++))
    
    # Test async status check
    run_test "Async request in queue" \
        "pending" \
        "SELECT status FROM steadytext_queue WHERE request_id = '$async_id'::uuid"
    
    # Test async embed request
    if run_sql "SELECT 1 FROM pg_extension WHERE extname = 'vector'" "$TEST_DB" | grep -q 1; then
        local embed_async_id=$(run_sql "SELECT steadytext_embed_async('Async embed test')" "$TEST_DB")
        run_test "Async embed returns UUID" \
            "t" \
            "SELECT '$embed_async_id'::uuid IS NOT NULL"
    else
        test_skipped "Async embed returns UUID" "pgvector not installed"
        ((TESTS_RUN++))
    fi
    
    # Test batch async operations
    local batch_result=$(run_sql "SELECT array_length(steadytext_generate_batch_async(ARRAY['Test 1', 'Test 2', 'Test 3'], 10), 1)" "$TEST_DB")
    run_test "Batch async generation" \
        "3" \
        "SELECT $batch_result"
    
    # Test async structured generation
    run_test "Async JSON generation" \
        "t" \
        "SELECT steadytext_generate_json_async('Generate JSON', '{\"type\": \"string\"}'::jsonb)::uuid IS NOT NULL"
    
    run_test "Async regex generation" \
        "t" \
        "SELECT steadytext_generate_regex_async('Phone number', '\\d{3}-\\d{3}-\\d{4}')::uuid IS NOT NULL"
    
    run_test "Async choice generation" \
        "t" \
        "SELECT steadytext_generate_choice_async('Choose one', ARRAY['yes', 'no', 'maybe'])::uuid IS NOT NULL"
    
    # Test queue statistics
    run_test "Queue has entries" \
        "t" \
        "SELECT COUNT(*) > 0 FROM steadytext_queue"
    
    # Test priority handling
    local high_priority_id=$(run_sql "SELECT steadytext_generate_async('High priority', 10, priority := 10)" "$TEST_DB")
    local low_priority_id=$(run_sql "SELECT steadytext_generate_async('Low priority', 10, priority := 1)" "$TEST_DB")
    
    run_test "Priority values set correctly" \
        "t" \
        "SELECT (SELECT priority FROM steadytext_queue WHERE request_id = '$high_priority_id'::uuid) > 
                (SELECT priority FROM steadytext_queue WHERE request_id = '$low_priority_id'::uuid)"
    
    # Test cancel functionality
    local cancel_id=$(run_sql "SELECT steadytext_generate_async('To be cancelled', 10)" "$TEST_DB")
    run_test "Cancel async request" \
        "t" \
        "SELECT steadytext_cancel_async('$cancel_id'::uuid)"
    
    run_test "Cancelled request status" \
        "cancelled" \
        "SELECT status FROM steadytext_queue WHERE request_id = '$cancel_id'::uuid"
    
    # Test check_async function
    run_test "Check async status function" \
        "t" \
        "SELECT (steadytext_check_async('$async_id'::uuid)).status IS NOT NULL"
    
    # Test batch status check
    local batch_ids=$(run_sql "SELECT ARRAY[steadytext_generate_async('Batch 1', 10), steadytext_generate_async('Batch 2', 10)]::uuid[]" "$TEST_DB")
    run_test "Batch status check" \
        "t" \
        "SELECT COUNT(*) = 2 FROM steadytext_check_async_batch($batch_ids)"
    
    # Note about worker
    if [ "$VERBOSE" = true ]; then
        log_verbose "${YELLOW}Note:${NC} Async requests will remain 'pending' without a running worker process"
        log_verbose "To process async requests, run: python3 /path/to/pg_steadytext/python/worker.py"
    fi
    
    # Structured generation tests
    log "\n${BLUE}=== Structured Generation Tests ===${NC}"
    
    # Test JSON generation with schema
    local json_schema='{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}'
    local json_result=$(run_sql "SELECT steadytext_generate_json('Create a person', '$json_schema'::jsonb)" "$TEST_DB" 2>&1)
    
    if [[ "$json_result" == *"{"* ]] && [[ "$json_result" == *"}"* ]]; then
        test_passed "JSON generation with schema"
    else
        test_failed "JSON generation with schema" "Invalid JSON output: $json_result"
    fi
    ((TESTS_RUN++))
    
    # Test JSON validation
    run_test "Generated JSON is valid" \
        "t" \
        "SELECT (steadytext_generate_json('Create data', '{\"type\": \"string\"}'::jsonb))::jsonb IS NOT NULL"
    
    # Test regex generation
    local phone_regex='\\d{3}-\\d{3}-\\d{4}'
    local phone_result=$(run_sql "SELECT steadytext_generate_regex('My phone is', '$phone_regex')" "$TEST_DB")
    
    if [[ "$phone_result" =~ [0-9]{3}-[0-9]{3}-[0-9]{4} ]]; then
        test_passed "Regex generation matches pattern"
    else
        test_failed "Regex generation matches pattern" "Got: $phone_result"
    fi
    ((TESTS_RUN++))
    
    # Test choice generation
    local choices="ARRAY['red', 'green', 'blue']"
    local choice_result=$(run_sql "SELECT steadytext_generate_choice('Pick a color', $choices)" "$TEST_DB")
    
    if [[ "$choice_result" == "red" ]] || [[ "$choice_result" == "green" ]] || [[ "$choice_result" == "blue" ]]; then
        test_passed "Choice generation returns valid option"
    else
        test_failed "Choice generation returns valid option" "Got: $choice_result"
    fi
    ((TESTS_RUN++))
    
    # Test structured generation determinism
    local json1=$(run_sql "SELECT steadytext_generate_json('Test', '{\"type\": \"string\"}'::jsonb)" "$TEST_DB")
    local json2=$(run_sql "SELECT steadytext_generate_json('Test', '{\"type\": \"string\"}'::jsonb)" "$TEST_DB")
    
    if [ "$json1" = "$json2" ]; then
        test_passed "Structured generation is deterministic"
    else
        test_failed "Structured generation is deterministic" "Results differ"
    fi
    ((TESTS_RUN++))
    
    # Test complex JSON schema
    local complex_schema='{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"}
            }
        }
    }'
    run_test "Complex JSON schema generation" \
        "t" \
        "SELECT (steadytext_generate_json('Create list', '$complex_schema'::jsonb))::jsonb IS NOT NULL"
    
    # Test regex with different patterns
    run_test "Email regex pattern" \
        "t" \
        "SELECT steadytext_generate_regex('Email:', '[a-z]+@[a-z]+\\.[a-z]+') ~ '^[a-z]+@[a-z]+\\.[a-z]+$'"
    
    run_test "Date regex pattern" \
        "t" \
        "SELECT steadytext_generate_regex('Date:', '\\d{4}-\\d{2}-\\d{2}') ~ '^\\d{4}-\\d{2}-\\d{2}$'"
    
    # Test choice with single option
    run_test "Single choice option" \
        "only_option" \
        "SELECT steadytext_generate_choice('Choose:', ARRAY['only_option'])"
    
    # Test structured generation with caching
    run_sql "SELECT steadytext_cache_clear()" "$TEST_DB"
    local cache_before=$(run_sql "SELECT (steadytext_cache_stats()).total_entries" "$TEST_DB")
    run_sql "SELECT steadytext_generate_json('Cached JSON', '{\"type\": \"string\"}'::jsonb)" "$TEST_DB"
    local cache_after=$(run_sql "SELECT (steadytext_cache_stats()).total_entries" "$TEST_DB")
    
    if [ "$cache_after" -gt "$cache_before" ]; then
        test_passed "Structured generation uses cache"
    else
        test_failed "Structured generation uses cache" "Cache not updated"
    fi
    ((TESTS_RUN++))
    
    # Test error handling for invalid schemas
    local invalid_result=$(run_sql "SELECT steadytext_generate_json('Test', '{\"invalid\": \"schema\"}'::jsonb)" "$TEST_DB" 2>&1)
    if [[ "$invalid_result" == *"error"* ]] || [[ "$invalid_result" == *"ERROR"* ]]; then
        test_passed "Invalid schema handling"
    else
        test_failed "Invalid schema handling" "Should have failed with invalid schema"
    fi
    ((TESTS_RUN++))
    
    # Test NULL handling in structured functions
    run_test "NULL prompt in JSON generation" \
        "error" \
        "SELECT steadytext_generate_json(NULL, '{\"type\": \"string\"}'::jsonb)"
    
    run_test "NULL schema in JSON generation" \
        "error" \
        "SELECT steadytext_generate_json('Test', NULL::jsonb)"
    
    run_test "NULL choices in choice generation" \
        "error" \
        "SELECT steadytext_generate_choice('Test', NULL::text[])"
    
    # pgTAP tests (if requested)
    if [ "$RUN_PGTAP" = true ]; then
        log "\n${BLUE}=== Running pgTAP Tests ===${NC}"
        
        # Check if pgTAP is installed
        if ! run_sql "SELECT 1 FROM pg_available_extensions WHERE name = 'pgtap'" | grep -q 1; then
            log "${YELLOW}Warning:${NC} pgTAP extension not available, skipping pgTAP tests"
        else
            # Install pgTAP in test database
            run_sql "CREATE EXTENSION IF NOT EXISTS pgtap" "$TEST_DB"
            
            # Find pgTAP test files
            SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
            PGTAP_DIR="$SCRIPT_DIR/test/pgtap"
            
            if [ -d "$PGTAP_DIR" ]; then
                log_verbose "Found pgTAP test directory: $PGTAP_DIR"
                
                # Run each pgTAP test file
                for test_file in "$PGTAP_DIR"/*.sql; do
                    if [ -f "$test_file" ]; then
                        test_name=$(basename "$test_file" .sql)
                        log_verbose "\nRunning pgTAP test: $test_name"
                        
                        if [ "$TAP_FORMAT" = true ]; then
                            # Run in TAP format
                            psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$TEST_DB" \
                                 -XAtq -f "$test_file" 2>&1
                        else
                            # Run with pretty output
                            result=$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$TEST_DB" \
                                         -XAtq -f "$test_file" 2>&1)
                            
                            # Check if tests passed
                            if echo "$result" | grep -q "^ok"; then
                                # Count passed tests
                                passed=$(echo "$result" | grep -c "^ok" || true)
                                log "${GREEN}✓${NC} pgTAP test $test_name: $passed tests passed"
                            else
                                log "${RED}✗${NC} pgTAP test $test_name failed"
                                if [ "$VERBOSE" = true ]; then
                                    echo "$result"
                                fi
                            fi
                        fi
                    fi
                done
            else
                log "${YELLOW}Warning:${NC} pgTAP test directory not found: $PGTAP_DIR"
            fi
        fi
    fi
    
    # Performance benchmarks (if requested)
    if [ "$BENCHMARK" = true ]; then
        log "\n${BLUE}=== Performance Benchmarks ===${NC}"
        
        # Benchmark configuration
        ITERATIONS=10
        PROMPT_LENGTHS=(10 50 100 500)
        
        # Generation benchmarks
        log_verbose "\nBenchmarking text generation..."
        for length in "${PROMPT_LENGTHS[@]}"; do
            # Generate a prompt of specified length
            prompt=$(printf 'word %.0s' $(seq 1 $length))
            
            # Time multiple iterations
            start_time=$(date +%s.%N)
            for i in $(seq 1 $ITERATIONS); do
                run_sql "SELECT LENGTH(steadytext_generate('$prompt', 50))" "$TEST_DB" > /dev/null
            done
            end_time=$(date +%s.%N)
            
            # Calculate average time
            total_time=$(echo "$end_time - $start_time" | bc)
            avg_time=$(echo "scale=3; $total_time / $ITERATIONS" | bc)
            
            log "${GREEN}✓${NC} Generation (${length} words): ${avg_time}s avg over $ITERATIONS runs"
        done
        
        # Cache performance
        log_verbose "\nBenchmarking cache performance..."
        run_sql "SELECT steadytext_cache_clear()" "$TEST_DB"
        
        # First run (cache miss)
        start_time=$(date +%s.%N)
        run_sql "SELECT steadytext_generate('Cache benchmark test', 100)" "$TEST_DB" > /dev/null
        end_time=$(date +%s.%N)
        cache_miss_time=$(echo "$end_time - $start_time" | bc)
        
        # Second run (cache hit)
        start_time=$(date +%s.%N)
        run_sql "SELECT steadytext_generate('Cache benchmark test', 100)" "$TEST_DB" > /dev/null
        end_time=$(date +%s.%N)
        cache_hit_time=$(echo "$end_time - $start_time" | bc)
        
        # Calculate speedup
        if (( $(echo "$cache_hit_time > 0" | bc -l) )); then
            speedup=$(echo "scale=2; $cache_miss_time / $cache_hit_time" | bc)
            log "${GREEN}✓${NC} Cache speedup: ${speedup}x (miss: ${cache_miss_time}s, hit: ${cache_hit_time}s)"
        fi
        
        # Embedding benchmarks (if available)
        if run_sql "SELECT 1 FROM pg_extension WHERE extname = 'vector'" "$TEST_DB" | grep -q 1; then
            log_verbose "\nBenchmarking embeddings..."
            
            start_time=$(date +%s.%N)
            for i in $(seq 1 $ITERATIONS); do
                run_sql "SELECT array_length(steadytext_embed('Benchmark text $i')::float[], 1)" "$TEST_DB" > /dev/null
            done
            end_time=$(date +%s.%N)
            
            total_time=$(echo "$end_time - $start_time" | bc)
            avg_time=$(echo "scale=3; $total_time / $ITERATIONS" | bc)
            
            log "${GREEN}✓${NC} Embedding generation: ${avg_time}s avg over $ITERATIONS runs"
        fi
        
        # Structured generation benchmarks
        log_verbose "\nBenchmarking structured generation..."
        
        # JSON generation
        start_time=$(date +%s.%N)
        for i in $(seq 1 $ITERATIONS); do
            run_sql "SELECT LENGTH(steadytext_generate_json('Test $i', '{\"type\": \"string\"}'::jsonb))" "$TEST_DB" > /dev/null
        done
        end_time=$(date +%s.%N)
        
        total_time=$(echo "$end_time - $start_time" | bc)
        avg_time=$(echo "scale=3; $total_time / $ITERATIONS" | bc)
        
        log "${GREEN}✓${NC} JSON generation: ${avg_time}s avg over $ITERATIONS runs"
        
        # Concurrent request simulation
        if command -v parallel &> /dev/null; then
            log_verbose "\nBenchmarking concurrent requests..."
            
            start_time=$(date +%s.%N)
            seq 1 10 | parallel -j 5 "psql -h $PGHOST -p $PGPORT -U $PGUSER -d $TEST_DB -tAc \"SELECT LENGTH(steadytext_generate('Concurrent test {}', 20))\" > /dev/null"
            end_time=$(date +%s.%N)
            
            total_time=$(echo "$end_time - $start_time" | bc)
            log "${GREEN}✓${NC} 10 concurrent requests (5 parallel): ${total_time}s total"
        else
            log_verbose "${YELLOW}Note:${NC} Install GNU parallel for concurrent request benchmarks"
        fi
        
        # Memory usage estimate
        if [ "$VERBOSE" = true ]; then
            cache_size=$(run_sql "SELECT pg_size_pretty(SUM(pg_column_size(response))::bigint) FROM steadytext_cache" "$TEST_DB" 2>/dev/null || echo "0 bytes")
            log_verbose "\nCache memory usage: $cache_size"
        fi
    fi
    
    # Summary
    if [ "$TAP_FORMAT" = false ]; then
        log "\n${BLUE}=== Test Summary ===${NC}"
        log "Total tests: $TESTS_RUN"
        log "${GREEN}Passed:${NC} $TESTS_PASSED"
        log "${RED}Failed:${NC} $TESTS_FAILED"
        log "${YELLOW}Skipped:${NC} $TESTS_SKIPPED"
        
        if [ $TESTS_FAILED -eq 0 ]; then
            log "\n${GREEN}All tests passed!${NC}"
            exit 0
        else
            log "\n${RED}Some tests failed.${NC}"
            exit 1
        fi
    else
        echo "1..$TESTS_RUN"
        if [ $TESTS_FAILED -eq 0 ]; then
            exit 0
        else
            exit 1
        fi
    fi
}

# Run main function
main
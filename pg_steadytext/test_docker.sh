#!/bin/bash
# Test script for pg_steadytext Docker installation
# AIDEV-NOTE: This script verifies that the PostgreSQL extension works correctly in Docker

set -e

echo "Building Docker image..."
docker build -t pg_steadytext_test .

echo "Removing any existing test container..."
docker rm -f pg_steadytext_test 2>/dev/null || true

echo "Starting test container..."
docker run -d \
    -p 5432:5432 \
    --name pg_steadytext_test \
    -e POSTGRES_PASSWORD=postgres \
    pg_steadytext_test

echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker exec pg_steadytext_test pg_isready -U postgres >/dev/null 2>&1; then
        echo "PostgreSQL is ready!"
        break
    fi
    echo -n "."
    sleep 1
done

echo "Waiting for extensions to be created..."
sleep 5

echo -e "\n=== Testing pg_steadytext extension ==="

echo -e "\n1. Checking extension version:"
docker exec pg_steadytext_test psql -U postgres -t -c "SELECT steadytext_version();"

echo -e "\n2. Testing Python initialization:"
docker exec pg_steadytext_test psql -U postgres -c "SELECT _steadytext_init_python();"

echo -e "\n3. Testing text generation:"
docker exec pg_steadytext_test psql -U postgres -t -c "SELECT steadytext_generate('Write a haiku about PostgreSQL');"

echo -e "\n4. Testing embedding generation:"
docker exec pg_steadytext_test psql -U postgres -t -c "SELECT encode(steadytext_embed('PostgreSQL is awesome')::bytea, 'base64');" | head -n 1

echo -e "\n5. Checking daemon status:"
docker exec pg_steadytext_test psql -U postgres -c "SELECT * FROM steadytext_daemon_status();"

echo -e "\n6. Testing cache functionality:"
docker exec pg_steadytext_test psql -U postgres -c "
    -- Generate twice to test cache
    SELECT steadytext_generate('Test cache', 10, true) AS first_gen;
    SELECT steadytext_generate('Test cache', 10, true) AS second_gen;
    -- Check cache stats
    SELECT * FROM steadytext_cache_stats();
"

echo -e "\n=== All tests completed successfully! ==="

echo -e "\nTo interact with the container:"
echo "  docker exec -it pg_steadytext_test psql -U postgres"

echo -e "\nTo stop and remove the test container:"
echo "  docker rm -f pg_steadytext_test"
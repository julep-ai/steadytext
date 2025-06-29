#!/bin/bash
# Test script showing pg_steadytext working correctly

echo "=== Testing pg_steadytext extension ==="

# In a single session, init and then generate
docker exec -i pg_steadytext_final psql -U postgres <<EOF
-- Initialize Python environment
SELECT _steadytext_init_python();

-- Now generate text
SELECT steadytext_generate('Write a haiku about PostgreSQL');
EOF

echo -e "\n=== Testing with cache ==="
docker exec -i pg_steadytext_final psql -U postgres <<EOF
-- Re-init for this session
SELECT _steadytext_init_python();

-- Generate same text twice to test cache
SELECT steadytext_generate('Test cache', 10, true) AS first_gen;
SELECT steadytext_generate('Test cache', 10, true) AS second_gen;

-- Check cache stats
SELECT * FROM steadytext_cache_stats();
EOF

echo -e "\n=== Success! ==="
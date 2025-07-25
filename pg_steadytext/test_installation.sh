#!/bin/bash
# test_installation.sh - Test script for pg_steadytext installation
# AIDEV-NOTE: This script verifies that the extension is properly installed

set -e

echo "=== pg_steadytext Installation Test ==="
echo ""

# Check if PostgreSQL is running
echo -n "1. Checking PostgreSQL status... "
if pg_isready >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAILED"
    echo "   PostgreSQL is not running or not accessible"
    exit 1
fi

# Get PostgreSQL version and lib directory
PG_VERSION=$(pg_config --version | awk '{print $2}' | cut -d. -f1)
PG_LIBDIR=$(pg_config --pkglibdir)
echo "2. PostgreSQL version: $PG_VERSION"
echo "   Library directory: $PG_LIBDIR"

# Check if extension files are installed
echo ""
echo "3. Checking extension files:"
echo -n "   - SQL files... "
if ls $PG_LIBDIR/../share/postgresql/extension/pg_steadytext*.sql >/dev/null 2>&1; then
    echo "OK"
else
    echo "NOT FOUND"
fi

echo -n "   - Control file... "
if [ -f "$PG_LIBDIR/../share/postgresql/extension/pg_steadytext.control" ]; then
    echo "OK"
else
    echo "NOT FOUND"
fi

echo -n "   - Python modules... "
if [ -d "$PG_LIBDIR/pg_steadytext/python" ]; then
    echo "OK ($(ls -1 $PG_LIBDIR/pg_steadytext/python/*.py 2>/dev/null | wc -l) files)"
else
    echo "NOT FOUND"
fi

echo -n "   - Site-packages... "
if [ -d "$PG_LIBDIR/pg_steadytext/site-packages" ]; then
    echo "OK"
    # Check if packages are installed
    if [ -d "$PG_LIBDIR/pg_steadytext/site-packages/steadytext" ]; then
        echo "     - steadytext: INSTALLED"
    else
        echo "     - steadytext: NOT FOUND"
    fi
    if [ -d "$PG_LIBDIR/pg_steadytext/site-packages/zmq" ]; then
        echo "     - pyzmq: INSTALLED"
    else
        echo "     - pyzmq: NOT FOUND"
    fi
    if [ -d "$PG_LIBDIR/pg_steadytext/site-packages/numpy" ]; then
        echo "     - numpy: INSTALLED"
    else
        echo "     - numpy: NOT FOUND"
    fi
else
    echo "NOT FOUND"
fi

# Check Python packages in system
echo ""
echo "4. Checking system Python packages:"
for pkg in steadytext zmq numpy; do
    echo -n "   - $pkg... "
    if python3 -c "import $pkg" 2>/dev/null; then
        echo "OK"
    else
        echo "NOT FOUND"
    fi
done

# Test in PostgreSQL
echo ""
echo "5. Testing PostgreSQL extension:"

# Use PGDATABASE if set, otherwise default to postgres
DB="${PGDATABASE:-postgres}"

echo -n "   - Creating test database... "
createdb pg_steadytext_test 2>/dev/null || echo "Already exists"

echo -n "   - Creating extensions... "
psql -d pg_steadytext_test -c "CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;" >/dev/null 2>&1 && \
psql -d pg_steadytext_test -c "CREATE EXTENSION IF NOT EXISTS vector CASCADE;" >/dev/null 2>&1 && \
psql -d pg_steadytext_test -c "CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;" >/dev/null 2>&1 && \
echo "OK" || echo "FAILED"

echo -n "   - Testing version function... "
VERSION=$(psql -d pg_steadytext_test -t -c "SELECT steadytext_version();" 2>&1)
if [ $? -eq 0 ]; then
    echo "OK (version: $(echo $VERSION | tr -d ' '))"
else
    echo "FAILED"
    echo "     Error: $VERSION"
fi

echo -n "   - Testing text generation... "
RESULT=$(psql -d pg_steadytext_test -t -c "SELECT steadytext_generate('Hello test', 10);" 2>&1)
if [ $? -eq 0 ]; then
    echo "OK"
    echo "     Result: $(echo $RESULT | head -c 50)..."
else
    echo "FAILED"
    echo "     Error: $RESULT"
fi

echo -n "   - Testing embedding generation... "
RESULT=$(psql -d pg_steadytext_test -t -c "SELECT length(steadytext_embed('Test')::text);" 2>&1)
if [ $? -eq 0 ]; then
    echo "OK (embedding length: $(echo $RESULT | tr -d ' '))"
else
    echo "FAILED"
    echo "     Error: $RESULT"
fi

# Cleanup
echo ""
echo -n "6. Cleaning up test database... "
dropdb pg_steadytext_test 2>/dev/null && echo "OK" || echo "FAILED"

echo ""
echo "=== Installation test complete ==="
echo ""
echo "If you see any failures above, please check:"
echo "1. Run 'sudo make install' from the pg_steadytext directory"
echo "2. Ensure PostgreSQL has the required extensions (plpython3u, vector)"
echo "3. Check PostgreSQL logs for detailed error messages"
echo ""
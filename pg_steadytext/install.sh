#!/bin/bash
# install.sh - Quick installation script for pg_steadytext
# AIDEV-NOTE: This script automates the installation process for testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo "Installing pg_steadytext PostgreSQL extension..."
echo "============================================="

# Check if we're in the right directory
if [ ! -f "pg_steadytext.control" ]; then
    echo -e "${RED}Error: Please run this script from the pg_steadytext directory${NC}"
    exit 1
fi

# Check for required commands
echo -e "\n${YELLOW}Checking system requirements...${NC}"
echo -n "  PostgreSQL (pg_config): "
if command -v pg_config >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} $(pg_config --version)"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "    Install PostgreSQL development packages:"
    echo "    Ubuntu/Debian: sudo apt-get install postgresql-server-dev-all"
    echo "    RHEL/CentOS: sudo yum install postgresql-devel"
    echo "    macOS: brew install postgresql"
    exit 1
fi

echo -n "  Python 3: "
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"
else
    echo -e "${RED}✗ Not found${NC}"
    exit 1
fi

echo -n "  Make: "
if command -v make >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    exit 1
fi

# Check PostgreSQL extensions
echo -e "\n${YELLOW}Checking PostgreSQL extensions...${NC}"
PG_VERSION=$(pg_config --version | grep -oE '[0-9]+' | head -1)

echo -n "  plpython3u: "
if [ -f "/usr/lib/postgresql/$PG_VERSION/lib/plpython3.so" ] || \
   [ -f "/usr/local/lib/postgresql/plpython3.so" ] || \
   [ -f "$(pg_config --pkglibdir)/plpython3.so" ]; then
    echo -e "${GREEN}✓${NC} Available"
else
    echo -e "${YELLOW}⚠${NC} Not found - install postgresql-plpython3-$PG_VERSION"
    echo "    Ubuntu/Debian: sudo apt-get install postgresql-plpython3-$PG_VERSION"
    echo "    RHEL/CentOS: sudo yum install postgresql$PG_VERSION-plpython3"
fi

echo -n "  pgvector: "
if [ -f "$(pg_config --pkglibdir)/vector.so" ]; then
    echo -e "${GREEN}✓${NC} Available"
else
    echo -e "${YELLOW}⚠${NC} Not found - install pgvector"
    echo "    See: https://github.com/pgvector/pgvector#installation"
fi

# Check Python dependencies
echo -e "\n${YELLOW}Checking Python dependencies...${NC}"
MISSING_DEPS=0

for package in steadytext pyzmq numpy; do
    echo -n "  $package: "
    if python3 -c "import $package" 2>/dev/null; then
        VERSION=$(python3 -c "import $package; print(getattr($package, '__version__', 'installed'))" 2>/dev/null || echo "installed")
        echo -e "${GREEN}✓${NC} $VERSION"
    else
        echo -e "${RED}✗ Not installed${NC}"
        MISSING_DEPS=1
    fi
done

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "\n${YELLOW}Installing missing Python dependencies...${NC}"
    pip3 install steadytext pyzmq numpy || {
        echo -e "${RED}Error: Failed to install dependencies${NC}"
        echo "Try running with sudo: sudo pip3 install steadytext pyzmq numpy"
        exit 1
    }
fi

# Get PostgreSQL directories
PG_LIBDIR=$(pg_config --pkglibdir)
PG_SHAREDIR=$(pg_config --sharedir)
PG_DOCDIR=$(pg_config --docdir)

echo -e "\n${YELLOW}PostgreSQL directories:${NC}"
echo "  Library: $PG_LIBDIR"
echo "  Share: $PG_SHAREDIR"
echo "  Docs: $PG_DOCDIR"

# Build the extension
echo -e "\n${YELLOW}Building extension...${NC}"
make clean 2>/dev/null || true
make || {
    echo -e "${RED}Error: Build failed${NC}"
    exit 1
}

# Install the extension
echo -e "\n${YELLOW}Installing extension (may require sudo)...${NC}"
if [ -w "$PG_LIBDIR" ] && [ -w "$PG_SHAREDIR" ]; then
    make install
else
    echo "Need sudo permissions to install to PostgreSQL directories"
    sudo make install
fi || {
    echo -e "${RED}Error: Installation failed${NC}"
    exit 1
}

# Create Python module directory and install
echo -e "\n${YELLOW}Installing Python modules...${NC}"
PYTHON_DIR="$PG_LIBDIR/pg_steadytext/python"
if [ -w "$PG_LIBDIR" ]; then
    mkdir -p "$PYTHON_DIR"
else
    sudo mkdir -p "$PYTHON_DIR"
fi

# Copy Python modules
for module in python/*.py; do
    if [ -w "$PYTHON_DIR" ]; then
        cp "$module" "$PYTHON_DIR/"
    else
        sudo cp "$module" "$PYTHON_DIR/"
    fi
done

echo -e "\n${GREEN}✓ pg_steadytext installed successfully!${NC}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Connect to PostgreSQL as superuser"
echo "2. Create required extensions:"
echo "   CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;"
echo "   CREATE EXTENSION IF NOT EXISTS vector CASCADE;"
echo "3. Create pg_steadytext extension:"
echo "   CREATE EXTENSION pg_steadytext CASCADE;"
echo ""
echo "${YELLOW}Quick test:${NC}"
echo "   SELECT steadytext_version();"
echo "   SELECT steadytext_generate('Hello world');"
echo "   SELECT steadytext_embed('Test text');"
echo ""
echo "${YELLOW}Start daemon (optional):${NC}"
echo "   SELECT steadytext_daemon_start();"
echo "   SELECT * FROM steadytext_daemon_status();"

# Check if we can test the installation
if command -v psql >/dev/null 2>&1; then
    echo -e "\n${YELLOW}Would you like to test the installation now? (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Enter PostgreSQL connection parameters:"
        read -p "  Database name [postgres]: " DB_NAME
        DB_NAME=${DB_NAME:-postgres}
        read -p "  Username [postgres]: " DB_USER
        DB_USER=${DB_USER:-postgres}
        
        echo -e "\n${YELLOW}Running installation test...${NC}"
        psql -U "$DB_USER" -d "$DB_NAME" <<EOF
-- Create extensions
CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;
CREATE EXTENSION IF NOT EXISTS vector CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;

-- Test functions
SELECT steadytext_version();
SELECT length(steadytext_generate('Hello world', 10)) > 0 AS generation_works;
SELECT vector_dims(steadytext_embed('Test')) = 1024 AS embedding_works;
EOF
    fi
fi
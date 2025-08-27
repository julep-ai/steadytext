#!/bin/bash
# AIDEV-NOTE: Simple helper script to rebuild pg_steadytext by copying files
# This works with the existing container setup without needing RW mounts

set -e

# Color output for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔨 Rebuilding pg_steadytext extension (copy method)...${NC}"

# Source directory
SRC_DIR="/workspace/pg_steadytext"

# Check if source exists
if [ ! -d "$SRC_DIR" ]; then
    echo -e "${RED}❌ Source directory not found: $SRC_DIR${NC}"
    exit 1
fi

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "pg_steadytext_db"; then
    echo -e "${YELLOW}Starting postgres container...${NC}"
    docker compose up -d postgres
    sleep 5
fi

# Copy all necessary files to the container
echo -e "${YELLOW}Copying extension files to container...${NC}"
docker exec pg_steadytext_db mkdir -p /tmp/pg_steadytext_build

# Copy source files
docker cp "$SRC_DIR/Makefile" pg_steadytext_db:/tmp/pg_steadytext_build/
docker cp "$SRC_DIR/pg_steadytext.control" pg_steadytext_db:/tmp/pg_steadytext_build/
docker cp "$SRC_DIR/sql" pg_steadytext_db:/tmp/pg_steadytext_build/
docker cp "$SRC_DIR/python" pg_steadytext_db:/tmp/pg_steadytext_build/
docker cp "$SRC_DIR/test" pg_steadytext_db:/tmp/pg_steadytext_build/

# Build and install inside container
echo -e "${YELLOW}Building extension inside container...${NC}"
docker exec pg_steadytext_db bash -c '
    cd /tmp/pg_steadytext_build && \
    make clean && \
    make install
' || {
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
}

# Reinstall the extension in the database
echo -e "${YELLOW}Reinstalling extension in database...${NC}"
docker exec pg_steadytext_db psql -U postgres -d postgres <<EOF
DROP EXTENSION IF EXISTS pg_steadytext CASCADE;
CREATE EXTENSION pg_steadytext;
EOF

# Run a quick test to verify installation
echo -e "${YELLOW}Testing installation...${NC}"
VERSION=$(docker exec pg_steadytext_db psql -U postgres -d postgres -tAc "SELECT steadytext_version();" 2>/dev/null)
if [ -n "$VERSION" ]; then
    echo -e "${GREEN}✅ Extension successfully installed! Version: $VERSION${NC}"
else
    echo -e "${RED}❌ Extension test failed${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 Extension rebuild complete!${NC}"
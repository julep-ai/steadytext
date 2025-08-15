#!/bin/bash
set -e

echo "Setting up Steadytext development environment..."

# Ensure we're in the workspace directory
cd /workspace

# Wait for workspace to be mounted and check if pyproject.toml exists
echo "Checking workspace mount..."
for i in {1..10}; do
    if [ -f "pyproject.toml" ]; then
        echo "Workspace is ready!"
        break
    fi
    echo "Waiting for workspace to be mounted..."
    sleep 1
done

if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in /workspace"
    echo "Current directory: $(pwd)"
    echo "Directory contents:"
    ls -la
    exit 1
fi

# Install node utils
echo "Installing node utils..."
npm install -g claude-code-inject claude-code-generic-hooks pyright

# Install aria2 and pgtap
sudo apt install -yq aria2 pgtap direnv

# Install Python dependencies
echo "Installing Python dependencies with uv..."
uv sync --all-extras
uv run pip install -e .

# Check if pg_steadytext exists before trying to install it
if [ -d "pg_steadytext" ]; then
    uv run pip install -e pg_steadytext/
else
    echo "Warning: pg_steadytext directory not found, skipping its installation"
fi

# Check if PostgreSQL is needed and set up using Docker
if command -v docker &> /dev/null; then
    echo "Setting up PostgreSQL using Docker..."
    
    # Check if pg_steadytext directory exists
    if [ -d "pg_steadytext" ]; then
        # Navigate to pg_steadytext directory
        cd pg_steadytext
        
        # Start PostgreSQL container
        docker compose up -d postgres
        
        # Wait for PostgreSQL to be ready
        echo "Waiting for PostgreSQL to be ready..."
        for i in {1..30}; do
            if docker compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
                echo "PostgreSQL is ready!"
                break
            fi
            echo -n "."
            sleep 1
        done
        
        cd ..
    else
        echo "pg_steadytext directory not found, skipping PostgreSQL setup"
    fi
else
    echo "Docker not available. Skipping PostgreSQL setup."
    echo "You can manually set up PostgreSQL or use an external instance."
fi

echo "Development environment setup complete!" 
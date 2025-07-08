#!/bin/bash
set -e
set -x

echo 'Initializing pg_steadytext extension...'

# AIDEV-NOTE: Check if using fallback model
if [ "$STEADYTEXT_USE_FALLBACK_MODEL" = "true" ]; then
    echo "Using fallback model (Qwen) instead of default (Gemma-3n)"
fi

# Wait for PostgreSQL to be ready
until pg_isready -U "$POSTGRES_USER"; do
    echo 'Waiting for PostgreSQL to start...'
    sleep 1
done

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create required extensions
    CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;
    CREATE EXTENSION IF NOT EXISTS vector CASCADE;
    CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;
    
    -- Create pgTAP for testing (optional, but useful for development)
    -- AIDEV-NOTE: pgTAP provides TAP testing framework for PostgreSQL
    CREATE EXTENSION IF NOT EXISTS pgtap CASCADE;

    -- AIDEV-NOTE: Initialization is now done on-demand in functions

    -- Verify installation
    SELECT steadytext_version();
    
    -- Verify pgTAP is available
    SELECT pgtap_version();
EOSQL

## # Start SteadyText daemon in background
## echo 'Starting SteadyText daemon...'
## st daemon start || echo 'Warning: Failed to start SteadyText daemon'

echo 'pg_steadytext initialization complete!'

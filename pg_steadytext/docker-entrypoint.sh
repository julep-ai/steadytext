#!/bin/bash
set -e
set -x

echo 'Initializing pg_steadytext extension...'

# AIDEV-NOTE: Check if using fallback model
if [ "$STEADYTEXT_USE_FALLBACK_MODEL" = "true" ]; then
    echo "Using fallback model (Qwen) instead of default (Gemma-3n)"
fi

# AIDEV-NOTE: Configure TimescaleDB in postgresql.conf
# TimescaleDB requires shared_preload_libraries to be set
# The Omnigres base image already sets omni--0.2.11.so, so we need to append
echo 'Configuring TimescaleDB...'

# Wait for PostgreSQL to be ready
until pg_isready -U "$POSTGRES_USER"; do
    echo 'Waiting for PostgreSQL to start...'
    sleep 1
done

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create required extensions
    CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;
    CREATE EXTENSION IF NOT EXISTS vector CASCADE;
    
    -- Try to create TimescaleDB extension if available
    -- AIDEV-NOTE: TimescaleDB is optional - requires manual shared_preload_libraries configuration
    -- AIDEV-NOTE: To enable: ALTER SYSTEM SET shared_preload_libraries = 'omni--0.2.11.so', 'timescaledb';
    DO \$\$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb') THEN
            BEGIN
                CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
                RAISE NOTICE 'TimescaleDB extension installed successfully';
            EXCEPTION
                WHEN OTHERS THEN
                    RAISE NOTICE 'TimescaleDB available but not configured - requires shared_preload_libraries setup';
            END;
        END IF;
    END\$\$;
    
    -- Create pg_steadytext extension
    CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;
    
    -- Create pgTAP for testing (optional, but useful for development)
    -- AIDEV-NOTE: pgTAP provides TAP testing framework for PostgreSQL
    CREATE EXTENSION IF NOT EXISTS pgtap CASCADE;

    -- AIDEV-NOTE: Initialization is now done on-demand in functions

    -- Verify installation
    SELECT steadytext_version();
    
    -- Verify pgTAP is available
    SELECT pgtap_version();
    
    -- Verify TimescaleDB is available
    SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb';
EOSQL

## # Start SteadyText daemon in background
## echo 'Starting SteadyText daemon...'
## st daemon start || echo 'Warning: Failed to start SteadyText daemon'

echo 'pg_steadytext initialization complete!'

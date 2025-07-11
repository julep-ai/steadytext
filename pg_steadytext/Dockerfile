# Dockerfile for pg_steadytext PostgreSQL extension
# AIDEV-NOTE: This creates a PostgreSQL image with pg_steadytext pre-installed
# Optimized for build caching - dependencies first, source code last

FROM postgres:17

# Build arguments for model selection
# AIDEV-NOTE: Use STEADYTEXT_USE_FALLBACK_MODEL=true to use known working models
ARG STEADYTEXT_USE_FALLBACK_MODEL=false
ENV STEADYTEXT_USE_FALLBACK_MODEL=${STEADYTEXT_USE_FALLBACK_MODEL}

# Install build dependencies
# AIDEV-NOTE: This layer is cached and rarely changes
# AIDEV-NOTE: pgTAP added for PostgreSQL TAP testing framework
RUN apt-get update && apt-get install -y \
    postgresql-server-dev-17 \
    postgresql-plpython3-17 \
    postgresql-17-pgvector \
    postgresql-17-pgtap \
    python3-pip \
    python3-dev \
    make \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# AIDEV-NOTE: This layer is cached when requirements don't change
RUN pip3 install --break-system-packages \
    steadytext>=2.1.0 \
    pyzmq>=22.0.0 \
    numpy>=1.20.0

# Pre-download models to catch compatibility issues early
# AIDEV-NOTE: This ensures model issues are caught at build time, not runtime
RUN python3 -c "import os; os.environ['STEADYTEXT_USE_FALLBACK_MODEL'] = os.environ.get('STEADYTEXT_USE_FALLBACK_MODEL', 'false'); \
    from steadytext.models.cache import ensure_generation_model_cached, ensure_embedding_model_cached; \
    print(f'Using fallback model: {os.environ[\"STEADYTEXT_USE_FALLBACK_MODEL\"]}'); \
    print('Pre-downloading generation model...'); \
    ensure_generation_model_cached(); \
    print('Pre-downloading embedding model...'); \
    ensure_embedding_model_cached(); \
    print('Models downloaded successfully')" || \
    (echo "WARNING: Model pre-download failed. Models will be downloaded on first use." && true)

# Create working directory
WORKDIR /tmp/pg_steadytext

# Copy only the files needed for building first
# AIDEV-NOTE: This allows caching the build environment
COPY Makefile pg_steadytext.control META.json ./

# Create directories for source files
RUN mkdir -p sql python

# Copy SQL files
# AIDEV-NOTE: SQL files change less frequently than Python files
COPY sql/ ./sql/

# Copy Python files
# AIDEV-NOTE: Python files are most likely to change
COPY python/ ./python/

# Copy diagnostic script (optional - skip if not available)
# COPY diagnose_pg_model.py /usr/local/bin/diagnose_pg_model
# RUN chmod +x /usr/local/bin/diagnose_pg_model || true

# Build and install the extension
# AIDEV-NOTE: This layer rebuilds only when source files change
RUN make install

# Copy the entrypoint script separately
# AIDEV-NOTE: This ensures the entrypoint can be updated without rebuilding everything
COPY docker-entrypoint.sh /docker-entrypoint-initdb.d/01-init-pg-steadytext.sh
RUN chmod +x /docker-entrypoint-initdb.d/01-init-pg-steadytext.sh

# Clean up build directory to reduce image size
RUN rm -rf /tmp/pg_steadytext

# Expose PostgreSQL port
EXPOSE 5432

# Set default environment variables
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_DB=postgres

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
  CMD pg_isready -U postgres || exit 1

# AIDEV-NOTE: Usage:
# Standard build (may fail with gemma-3n models):
#   docker build -t pg_steadytext .
#
# Build with fallback model (recommended if gemma-3n fails):
#   docker build --build-arg STEADYTEXT_USE_FALLBACK_MODEL=true -t pg_steadytext .
#
# Run container:
#   docker run -d -p 5432:5432 --name pg_steadytext pg_steadytext
#
# Or run with fallback model environment variable:
#   docker run -d -p 5432:5432 -e STEADYTEXT_USE_FALLBACK_MODEL=true --name pg_steadytext pg_steadytext
#
# Test generation:
#   docker exec -it pg_steadytext psql -U postgres -c "SELECT steadytext_generate('Hello Docker!');"
#
# Run diagnostics if issues occur:
#   docker exec -it pg_steadytext /usr/local/bin/diagnose_pg_model
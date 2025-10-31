# syntax=docker/dockerfile:1
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
# AIDEV-NOTE: TimescaleDB package installed for optional time-series testing
# AIDEV-NOTE: TimescaleDB package installed and preloaded by default
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    # Add TimescaleDB repository
    apt-get install -y gnupg postgresql-common apt-transport-https lsb-release wget && \
    echo "deb https://packagecloud.io/timescale/timescaledb/debian/ $(lsb_release -c -s) main" > /etc/apt/sources.list.d/timescaledb.list && \
    wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | gpg --dearmor -o /etc/apt/trusted.gpg.d/timescaledb.gpg && \
    apt-get update && apt-get install -y \
    postgresql-server-dev-17 \
    postgresql-plpython3-17 \
    postgresql-17-pgvector \
    postgresql-17-pgtap \
    timescaledb-2-postgresql-17 \
    python3-pip \
    python3-dev \
    make \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configure PostgreSQL defaults for TimescaleDB
# AIDEV-NOTE: Ensure new databases preload TimescaleDB library
RUN for conf in /usr/share/postgresql/postgresql.conf.sample /usr/share/postgresql/17/postgresql.conf.sample; do \
        if [ -f "$conf" ]; then \
            sed -ri "s/^#?(shared_preload_libraries)\s*=.*/\1 = 'timescaledb'    # (change requires restart)/" "$conf"; \
        fi; \
    done

# Install Python dependencies from local source (parent directory)
# AIDEV-NOTE: Build context must include repo root so ../steadytext is available
COPY .. /opt/repo
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages -e /opt/repo && \
    pip3 install --break-system-packages \
    pyzmq>=22.0.0 \
    numpy>=1.20.0

# Create working directory
WORKDIR /tmp/pg_steadytext

# Copy only the files needed for building first
# AIDEV-NOTE: This allows caching the build environment
COPY pg_steadytext/Makefile pg_steadytext/pg_steadytext.control pg_steadytext/META.json ./

# Create directories for source files
RUN mkdir -p sql python

# Copy SQL files
# AIDEV-NOTE: SQL files change less frequently than Python files
COPY pg_steadytext/sql/ ./sql/

# Copy Python files
# AIDEV-NOTE: Python files are most likely to change
COPY pg_steadytext/python/ ./python/

# Build and install the extension
# AIDEV-NOTE: This layer rebuilds only when source files change
RUN make install

# Install PGMQ extension
# AIDEV-NOTE: Using cache mount for git operations
RUN --mount=type=cache,target=/tmp/git-cache \
    cd /tmp/git-cache && \
    if [ ! -d pgmq ]; then git clone https://github.com/pgmq/pgmq.git; fi && \
    cd pgmq && git pull && \
    cd pgmq-extension && \
    make clean && make && \
    make install

# Copy the entrypoint script separately
# AIDEV-NOTE: This ensures the entrypoint can be updated without rebuilding everything
COPY pg_steadytext/docker-entrypoint.sh /docker-entrypoint-initdb.d/01-init-pg-steadytext.sh
RUN chmod +x /docker-entrypoint-initdb.d/01-init-pg-steadytext.sh

# Clean up build directory to reduce image size
RUN rm -rf /tmp/pg_steadytext

# Expose PostgreSQL port
EXPOSE 5432

# Set default environment variables
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=password
ENV POSTGRES_DB=postgres

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
  CMD pg_isready -U postgres || exit 1

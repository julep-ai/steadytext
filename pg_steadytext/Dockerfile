# Dockerfile for pg_steadytext PostgreSQL extension
# AIDEV-NOTE: This creates a PostgreSQL image with pg_steadytext pre-installed

FROM postgres:16

# Install build dependencies
RUN apt-get update && apt-get install -y \
    postgresql-server-dev-16 \
    postgresql-plpython3-16 \
    postgresql-16-pgvector \
    python3-pip \
    python3-dev \
    make \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --break-system-packages \
    steadytext>=1.3.0 \
    pyzmq>=22.0.0 \
    numpy>=1.20.0

# Copy extension files
WORKDIR /tmp/pg_steadytext
COPY . .

# Build and install the extension
RUN make install

# Create initialization script
RUN echo "#!/bin/bash\n\
set -e\n\
\n\
psql -v ON_ERROR_STOP=1 --username \"\$POSTGRES_USER\" --dbname \"\$POSTGRES_DB\" <<-EOSQL\n\
    -- Create required extensions\n\
    CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;\n\
    CREATE EXTENSION IF NOT EXISTS pgvector CASCADE;\n\
    CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;\n\
    \n\
    -- Verify installation\n\
    SELECT steadytext_version();\n\
EOSQL\n\
\n\
# Start SteadyText daemon in background\n\
st daemon start || true\n" > /docker-entrypoint-initdb.d/01-init-pg-steadytext.sh

RUN chmod +x /docker-entrypoint-initdb.d/01-init-pg-steadytext.sh

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
# docker build -t pg_steadytext .
# docker run -d -p 5432:5432 --name pg_steadytext pg_steadytext
# docker exec -it pg_steadytext psql -U postgres -c "SELECT steadytext_generate('Hello Docker!');"
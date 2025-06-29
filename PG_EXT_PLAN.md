GOAL: I want to wrap this library into a postgres extension that uses a postgres cache table with lru / lfu stats stored. this way any generations are both deterministic and cache-able for small everyday tasks. the generation and embedding should be done in a separate, non-blocking background worker.

POSSIBLE EXTENSIONS TO USE:
- pgmq
- omni_python
- omni_worker
- omni_vfs
- pgvector

PLAN:
# Detailed Plan: Building pg_steadytext Extension

## 🎯 Project Overview

**pg_steadytext** will be a PostgreSQL extension that wraps the SteadyText library, providing deterministic text generation and embeddings with PostgreSQL-backed cache and queue management. The extension will leverage SteadyText's existing daemon architecture (ZeroMQ-based) and frecency cache system while adding PostgreSQL-specific features like ACID guarantees, SQL querying capabilities, and integration with pgvector for semantic search.

### Key Principles
- **Leverage existing architecture**: Use SteadyText's daemon instead of reimplementing
- **PostgreSQL-native**: Expose functionality through SQL functions and views
- **Production-ready**: Include security, monitoring, and resource management from day one
- **Zero-downtime upgrades**: Support schema migrations and backward compatibility
- **Easy installation**: One-command setup with automatic dependency handling

## 📦 Easy Installation

### 🚀 Quick Start (5 minutes)

```bash
# For most users - single command installation
curl -sSL https://pg-steadytext.io/install.sh | bash

# Or using Docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=mysecret steadytext/postgres:16

# Connect and use immediately
psql -U postgres -c "SELECT steadytext_generate('Hello world');"
```

### One-Command Installation

```bash
# Install via pgxn (PostgreSQL Extension Network)
pgxn install pg_steadytext

# Or via apt/yum for packaged distributions
sudo apt install postgresql-16-steadytext  # Debian/Ubuntu
sudo yum install pg_steadytext16          # RHEL/CentOS
```

### Docker Installation

```dockerfile
# Dockerfile for PostgreSQL with pg_steadytext
FROM postgres:16

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    postgresql-16-python3 \
    postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*

# Install SteadyText and pg_steadytext
RUN pip3 install steadytext
RUN pgxn install pg_steadytext

# Auto-create extension on database initialization
COPY --chown=postgres:postgres init-steadytext.sql /docker-entrypoint-initdb.d/
```

```sql
-- init-steadytext.sql
CREATE EXTENSION IF NOT EXISTS plpython3u;
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS pg_steadytext;

-- Start SteadyText daemon automatically
SELECT steadytext_daemon_start();
```

### Manual Installation

```bash
# 1. Install prerequisites
sudo apt install postgresql-16-python3 postgresql-16-pgvector python3-pip

# 2. Install Python package
pip3 install steadytext

# 3. Clone and install extension
git clone https://github.com/yourusername/pg_steadytext
cd pg_steadytext
make && sudo make install

# 4. In PostgreSQL
CREATE EXTENSION pg_steadytext CASCADE;
```

### Omnigres Bundle Installation

For users of Omnigres extensions:

```sql
-- Install all required Omnigres extensions first
CREATE EXTENSION IF NOT EXISTS omni_python CASCADE;
CREATE EXTENSION IF NOT EXISTS omni_worker CASCADE;
CREATE EXTENSION IF NOT EXISTS omni_vfs CASCADE;
CREATE EXTENSION IF NOT EXISTS pgmq CASCADE;
CREATE EXTENSION IF NOT EXISTS pgvector CASCADE;

-- Then install pg_steadytext
CREATE EXTENSION pg_steadytext;

-- Verify installation
SELECT steadytext_version();
```

### Cloud Provider Installation

#### AWS RDS
```sql
-- Using AWS RDS custom extensions
CALL rds_add_extension('pg_steadytext');
CREATE EXTENSION pg_steadytext;
```

#### Google Cloud SQL
```sql
-- Request extension via Cloud Console, then:
CREATE EXTENSION pg_steadytext;
```

#### Azure Database for PostgreSQL
```sql
-- Enable in portal under "Extensions", then:
CREATE EXTENSION pg_steadytext;
```

### Post-Installation Setup

```sql
-- 1. Configure resource limits
SELECT steadytext_config_set('max_memory_mb', '2048');
SELECT steadytext_config_set('max_concurrent_tasks', '10');
SELECT steadytext_config_set('cache_size_mb', '500');

-- 2. Start the daemon (if not auto-started)
SELECT steadytext_daemon_start();

-- 3. Verify installation
SELECT steadytext_generate('Hello world');

-- 4. Create indexes for better performance
CREATE INDEX idx_steadytext_cache_key ON steadytext_cache(cache_key);
CREATE INDEX idx_steadytext_cache_frecency ON steadytext_cache(frecency_score DESC);
CREATE INDEX idx_steadytext_queue_status ON steadytext_queue(status, created_at);

-- 5. Set up monitoring (optional)
SELECT steadytext_monitoring_enable();
```

### Package Structure

The extension will be packaged with:

```
pg_steadytext/
├── pg_steadytext.control          # Extension metadata
├── sql/
│   ├── pg_steadytext--1.0.0.sql  # Initial schema
│   ├── pg_steadytext--1.0.0--1.1.0.sql  # Upgrade scripts
│   └── uninstall_pg_steadytext.sql
├── python/
│   ├── __init__.py
│   ├── daemon_connector.py        # SteadyText integration
│   ├── cache_manager.py           # Cache sync logic
│   └── security.py                # Input validation
├── expected/                      # Test expected outputs
├── sql/                          # Test SQL scripts
├── Makefile
├── README.md
└── META.json                     # PGXN metadata
```

### Dependency Management

```makefile
# Makefile
EXTENSION = pg_steadytext
DATA = sql/pg_steadytext--1.0.0.sql
REGRESS = basic cache queue security performance

# Python module installation
PYTHON_MODULES = daemon_connector cache_manager security

# Dependencies check
check-deps:
	@echo "Checking dependencies..."
	@python3 -c "import steadytext" || (echo "ERROR: steadytext not installed. Run: pip3 install steadytext" && exit 1)
	@psql -c "SELECT 1 FROM pg_available_extensions WHERE name = 'plpython3u'" || (echo "ERROR: plpython3u not available" && exit 1)
	@psql -c "SELECT 1 FROM pg_available_extensions WHERE name = 'pgvector'" || (echo "ERROR: pgvector not available" && exit 1)

install: check-deps
	$(MAKE) -f Makefile.pgxs install
```

### Automated Setup Script

```bash
#!/bin/bash
# setup-pg-steadytext.sh

set -e

echo "🚀 Setting up pg_steadytext..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v apt-get &> /dev/null; then
        PKG_MANAGER="apt"
        POSTGRES_PKG="postgresql-16-python3 postgresql-16-pgvector"
    elif command -v yum &> /dev/null; then
        PKG_MANAGER="yum"
        POSTGRES_PKG="postgresql16-plpython3 pgvector_16"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PKG_MANAGER="brew"
    POSTGRES_PKG="postgresql@16"
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
case $PKG_MANAGER in
    apt)
        sudo apt-get update
        sudo apt-get install -y $POSTGRES_PKG python3-pip
        ;;
    yum)
        sudo yum install -y $POSTGRES_PKG python3-pip
        ;;
    brew)
        brew install $POSTGRES_PKG python@3
        ;;
esac

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install steadytext

# Install extension
echo "🔧 Installing PostgreSQL extension..."
if command -v pgxn &> /dev/null; then
    pgxn install pg_steadytext
else
    git clone https://github.com/yourusername/pg_steadytext /tmp/pg_steadytext
    cd /tmp/pg_steadytext
    make && sudo make install
fi

# Create extension in database
echo "🗄️ Creating extension in database..."
sudo -u postgres psql <<EOF
CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;
CREATE EXTENSION IF NOT EXISTS pgvector CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_steadytext CASCADE;

-- Start daemon
SELECT steadytext_daemon_start();

-- Verify
SELECT steadytext_generate('Installation test');
EOF

echo "✅ pg_steadytext successfully installed!"
echo "📖 Documentation: https://github.com/yourusername/pg_steadytext"
```

### Uninstallation

```sql
-- Graceful uninstallation
SELECT steadytext_daemon_stop();
DROP EXTENSION pg_steadytext CASCADE;

-- Clean up any remaining objects
DROP TABLE IF EXISTS steadytext_cache CASCADE;
DROP TABLE IF EXISTS steadytext_queue CASCADE;
DROP TABLE IF EXISTS steadytext_config CASCADE;
```

## 📦 Packaging and Distribution

### PGXN Package

The extension will be distributed via PGXN (PostgreSQL Extension Network) for easy installation:

```json
// META.json for PGXN
{
  "name": "pg_steadytext",
  "abstract": "Deterministic text generation and embeddings for PostgreSQL",
  "description": "PostgreSQL extension wrapping SteadyText library for deterministic AI text generation with caching",
  "version": "1.0.0",
  "maintainer": "Your Name <you@example.com>",
  "license": "postgresql",
  "provides": {
    "pg_steadytext": {
      "abstract": "Deterministic text generation functions",
      "file": "sql/pg_steadytext--1.0.0.sql",
      "docfile": "doc/pg_steadytext.md",
      "version": "1.0.0"
    }
  },
  "prereqs": {
    "runtime": {
      "requires": {
        "PostgreSQL": "16.0.0",
        "plpython3u": "0",
        "pgvector": "0.5.0"
      },
      "recommends": {
        "omni_python": "0",
        "omni_worker": "0",
        "pgmq": "0"
      }
    }
  },
  "resources": {
    "homepage": "https://github.com/yourusername/pg_steadytext",
    "bugtracker": {
      "web": "https://github.com/yourusername/pg_steadytext/issues"
    },
    "repository": {
      "url": "git://github.com/yourusername/pg_steadytext.git",
      "web": "https://github.com/yourusername/pg_steadytext",
      "type": "git"
    }
  },
  "generated_by": "pg_steadytext developer",
  "meta-spec": {
    "version": "1.0.0",
    "url": "http://pgxn.org/meta/spec.txt"
  },
  "tags": [
    "ai",
    "text generation",
    "embeddings",
    "nlp",
    "machine learning",
    "deterministic"
  ]
}
```

### Binary Packages

Pre-built packages for major platforms:

```yaml
# .github/workflows/release.yml
name: Release Packages

on:
  push:
    tags:
      - 'v*'

jobs:
  build-packages:
    strategy:
      matrix:
        include:
          - os: ubuntu-22.04
            pkg_type: deb
            postgres: [14, 15, 16]
          - os: ubuntu-20.04
            pkg_type: deb
            postgres: [14, 15, 16]
          - os: centos-8
            pkg_type: rpm
            postgres: [14, 15, 16]

    steps:
      - name: Build Package
        run: |
          make package PKG_TYPE=${{ matrix.pkg_type }} PG_VERSION=${{ matrix.postgres }}

      - name: Upload to Package Repository
        run: |
          case ${{ matrix.pkg_type }} in
            deb)
              deb-s3 upload --bucket=pg-steadytext-apt dist/*.deb
              ;;
            rpm)
              aws s3 cp dist/*.rpm s3://pg-steadytext-yum/
              ;;
          esac
```

### Container Images

Pre-built Docker images with everything configured:

```dockerfile
# docker/Dockerfile.postgres
ARG PG_VERSION=16
FROM postgres:${PG_VERSION}

LABEL maintainer="pg_steadytext maintainers"
LABEL description="PostgreSQL ${PG_VERSION} with pg_steadytext extension"

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-server-dev-${PG_MAJOR} \
    python3-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir steadytext

# Install required extensions
RUN apt-get update && apt-get install -y \
    postgresql-${PG_MAJOR}-python3 \
    postgresql-${PG_MAJOR}-pgvector \
    && rm -rf /var/lib/apt/lists/*

# Build and install pg_steadytext
WORKDIR /tmp
RUN git clone https://github.com/yourusername/pg_steadytext.git \
    && cd pg_steadytext \
    && make && make install \
    && cd .. && rm -rf pg_steadytext

# Add initialization script
COPY docker-entrypoint-initdb.d/* /docker-entrypoint-initdb.d/

# Add health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD pg_isready -U postgres && \
        psql -U postgres -c "SELECT steadytext_daemon_status();" || exit 1
```

### Kubernetes Operator

For cloud-native deployments:

```yaml
# kubernetes/pg-steadytext-operator.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pg-steadytext-operator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: pg-steadytext-operator
  template:
    metadata:
      labels:
        app: pg-steadytext-operator
    spec:
      containers:
      - name: operator
        image: pg-steadytext/operator:latest
        env:
        - name: WATCH_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
---
# Custom Resource Definition
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: steadytextclusters.pg-steadytext.io
spec:
  group: pg-steadytext.io
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              postgresVersion:
                type: string
                enum: ["14", "15", "16"]
              replicas:
                type: integer
                minimum: 1
              steadytextConfig:
                type: object
                properties:
                  maxMemoryMB:
                    type: integer
                  maxConcurrentTasks:
                    type: integer
                  cacheSizeMB:
                    type: integer
```

### Homebrew Formula

For macOS users:

```ruby
# homebrew-tap/Formula/pg_steadytext.rb
class PgSteadytext < Formula
  desc "PostgreSQL extension for deterministic text generation"
  homepage "https://github.com/yourusername/pg_steadytext"
  url "https://github.com/yourusername/pg_steadytext/archive/v1.0.0.tar.gz"
  sha256 "abc123..."
  license "PostgreSQL"

  depends_on "postgresql@16"
  depends_on "python@3.11"
  depends_on "pgvector"

  def install
    system "pip3", "install", "steadytext"

    ENV["PG_CONFIG"] = Formula["postgresql@16"].bin/"pg_config"
    system "make"
    system "make", "install", "DESTDIR=#{prefix}"
  end

  def post_install
    puts <<~EOS
      To use pg_steadytext, run:
        CREATE EXTENSION pg_steadytext;

      For more information:
        https://github.com/yourusername/pg_steadytext
    EOS
  end

  test do
    pg_version = Formula["postgresql@16"].version.major
    pg_ctl = Formula["postgresql@16"].bin/"pg_ctl"
    psql = Formula["postgresql@16"].bin/"psql"
    port = free_port

    system pg_ctl, "initdb", "-D", testpath/"test"
    (testpath/"test/postgresql.conf").write <<~EOS, mode: "a+"
      port = #{port}
    EOS
    system pg_ctl, "start", "-D", testpath/"test", "-l", testpath/"log"

    begin
      system psql, "-p", port.to_s, "-c", "CREATE EXTENSION pg_steadytext;", "postgres"
      output = shell_output("#{psql} -p #{port} -c \"SELECT steadytext_generate('test');\" postgres")
      assert_match /test/, output
    ensure
      system pg_ctl, "stop", "-D", testpath/"test"
    end
  end
end
```

## 🏗️ Integration with SteadyText

### SteadyText Daemon Integration

The extension will communicate with SteadyText's existing ZeroMQ daemon rather than implementing a new one:

```python
# daemon_connector.py
import zmq
import json
from steadytext.daemon.protocol import Request, Response
from steadytext.daemon import use_daemon

class SteadyTextConnector:
    """PostgreSQL-friendly wrapper for SteadyText daemon communication"""

    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self._ensure_daemon_running()

    def _ensure_daemon_running(self):
        """Start daemon if not already running"""
        try:
            # Check if daemon is responsive
            with use_daemon():
                pass
        except Exception:
            # Start daemon in background
            import subprocess
            subprocess.Popen(['st', 'daemon', 'start'])

    def generate(self, prompt, **kwargs):
        """Generate text using daemon with automatic fallback"""
        from steadytext import generate
        return generate(prompt, **kwargs)

    def embed(self, text):
        """Generate embedding using daemon with automatic fallback"""
        from steadytext import embed
        return embed(text)
```

### Cache Architecture

We'll extend SteadyText's SQLite cache with PostgreSQL-specific features:

1. **Mirror cache entries** in PostgreSQL for SQL querying
2. **Sync mechanism** between SQLite and PostgreSQL caches
3. **PostgreSQL as authoritative source** for cache eviction policies
4. **Leverage SteadyText's cache keys** for consistency

## 📋 Architecture Components

### 1. **Core Tables Structure**

```sql
-- Cache table that mirrors and extends SteadyText's SQLite cache
CREATE TABLE steadytext_cache (
    id SERIAL PRIMARY KEY,
    cache_key TEXT UNIQUE NOT NULL,  -- Matches SteadyText's cache key generation
    prompt TEXT NOT NULL,
    response TEXT,
    embedding VECTOR(1024),  -- For embedding cache

    -- Frecency statistics (synced with SteadyText's cache)
    access_count INT DEFAULT 1,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- SteadyText integration metadata
    steadytext_cache_hit BOOLEAN DEFAULT FALSE,  -- Whether this came from ST's cache
    model_name TEXT NOT NULL DEFAULT 'qwen3-1.7b',  -- Model used (supports switching)
    model_size TEXT CHECK (model_size IN ('small', 'medium', 'large')),
    thinking_mode BOOLEAN DEFAULT FALSE,  -- Whether thinking mode was enabled
    eos_string TEXT,  -- Custom end-of-sequence string if used

    -- Generation parameters
    generation_params JSONB,  -- temperature, max_tokens, seed, etc.
    response_size INT,
    generation_time_ms INT,  -- Time taken to generate (if not cached)

    -- Frecency score (computed column for efficient eviction)
    frecency_score FLOAT GENERATED ALWAYS AS (
        access_count * exp(-extract(epoch from (NOW() - last_accessed)) / 86400.0)
    ) STORED,

    INDEX idx_cache_key USING hash(cache_key),
    INDEX idx_frecency_score (frecency_score DESC),
    INDEX idx_last_accessed (last_accessed),
    INDEX idx_model_prompt (model_name, prompt text_pattern_ops)
);

-- Request queue with priority and resource management
CREATE TABLE steadytext_queue (
    id SERIAL PRIMARY KEY,
    request_id UUID DEFAULT gen_random_uuid(),
    request_type TEXT CHECK (request_type IN ('generate', 'embed', 'batch_embed')),

    -- Request data
    prompt TEXT,  -- For single requests
    prompts TEXT[],  -- For batch requests
    params JSONB,  -- Model params, thinking_mode, etc.

    -- Priority and scheduling
    priority INT DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    user_id TEXT,  -- For rate limiting per user
    session_id TEXT,  -- For request grouping

    -- Status tracking
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    result TEXT,
    results TEXT[],  -- For batch results
    embedding VECTOR(1024),
    embeddings VECTOR(1024)[],  -- For batch embeddings
    error TEXT,

    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    processing_time_ms INT,

    -- Resource tracking
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    daemon_endpoint TEXT,  -- Which daemon instance handled this

    INDEX idx_status_priority_created (status, priority DESC, created_at),
    INDEX idx_user_created (user_id, created_at DESC),
    INDEX idx_session (session_id)
);

-- FAISS index metadata
CREATE TABLE steadytext_indexes (
    id SERIAL PRIMARY KEY,
    index_name TEXT UNIQUE NOT NULL,
    index_data BYTEA,  -- Serialized FAISS index
    document_count INT,
    dimension INT DEFAULT 1024,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Configuration and resource limits
CREATE TABLE steadytext_config (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by TEXT DEFAULT current_user
);

-- Rate limiting per user
CREATE TABLE steadytext_rate_limits (
    user_id TEXT PRIMARY KEY,
    requests_per_minute INT DEFAULT 60,
    requests_per_hour INT DEFAULT 1000,
    requests_per_day INT DEFAULT 10000,
    current_minute_count INT DEFAULT 0,
    current_hour_count INT DEFAULT 0,
    current_day_count INT DEFAULT 0,
    last_reset_minute TIMESTAMPTZ DEFAULT NOW(),
    last_reset_hour TIMESTAMPTZ DEFAULT NOW(),
    last_reset_day TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit log for security and debugging
CREATE TABLE steadytext_audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id TEXT DEFAULT current_user,
    action TEXT NOT NULL,
    request_id UUID,
    details JSONB,
    ip_address INET,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,

    INDEX idx_audit_timestamp (timestamp DESC),
    INDEX idx_audit_user (user_id, timestamp DESC)
);

-- Daemon health monitoring
CREATE TABLE steadytext_daemon_health (
    daemon_id TEXT PRIMARY KEY,
    endpoint TEXT NOT NULL,
    last_heartbeat TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'healthy' CHECK (status IN ('healthy', 'unhealthy', 'starting', 'stopping')),
    version TEXT,
    model_loaded TEXT[],
    memory_usage_mb INT,
    active_connections INT,
    total_requests BIGINT DEFAULT 0,
    error_count INT DEFAULT 0,
    avg_response_time_ms INT
);
```

### 2. **Python Components Structure**

```
/python-files/
├── requirements.txt          # steadytext, psycopg2, numpy, faiss-cpu
├── pg_steadytext/
│   ├── __init__.py
│   ├── daemon_connector.py   # SteadyText daemon integration
│   ├── cache_sync.py         # Sync between SQLite and PostgreSQL caches
│   ├── security.py           # Input validation and rate limiting
│   ├── generation.py         # Text generation with queue management
│   ├── embeddings.py         # Embedding generation and batch processing
│   ├── worker.py             # Background worker for async processing
│   ├── health.py             # Daemon health monitoring
│   ├── metrics.py            # Performance metrics collection
│   └── config.py             # Configuration management
```

### 3. **Core Functions**

```sql
-- Synchronous generation with cache
CREATE FUNCTION steadytext_generate(
    prompt TEXT,
    max_tokens INT DEFAULT 512,
    use_cache BOOLEAN DEFAULT TRUE
) RETURNS TEXT;

-- Async generation (returns request_id)
CREATE FUNCTION steadytext_generate_async(
    prompt TEXT,
    max_tokens INT DEFAULT 512
) RETURNS UUID;

-- Check async result
CREATE FUNCTION steadytext_get_result(request_id UUID)
RETURNS TABLE(status TEXT, result TEXT, error TEXT);

-- Streaming generation
CREATE FUNCTION steadytext_generate_stream(
    prompt TEXT,
    max_tokens INT DEFAULT 512
) RETURNS SETOF TEXT;

-- Embeddings
CREATE FUNCTION steadytext_embed(
    text TEXT,
    use_cache BOOLEAN DEFAULT TRUE
) RETURNS VECTOR(1024);

-- Batch embeddings
CREATE FUNCTION steadytext_embed_batch(
    texts TEXT[]
) RETURNS TABLE(text TEXT, embedding VECTOR(1024));

-- Cache management
CREATE FUNCTION steadytext_cache_stats()
RETURNS TABLE(
    total_entries INT,
    total_size_mb FLOAT,
    avg_access_count FLOAT,
    cache_hit_rate FLOAT
);

CREATE FUNCTION steadytext_cache_evict(
    max_entries INT DEFAULT NULL,
    max_size_mb FLOAT DEFAULT NULL
) RETURNS INT;  -- Number of entries evicted

-- FAISS index operations
CREATE FUNCTION steadytext_index_create(
    index_name TEXT,
    documents TEXT[]
) RETURNS VOID;

CREATE FUNCTION steadytext_index_search(
    index_name TEXT,
    query TEXT,
    top_k INT DEFAULT 5
) RETURNS TABLE(document TEXT, similarity FLOAT);
```

## 🔧 Implementation Details

### Phase 1: Basic Infrastructure (Week 1)

1. **Set up Omnigres environment**
   ```sql
   CREATE EXTENSION IF NOT EXISTS omni_python CASCADE;
   CREATE EXTENSION IF NOT EXISTS omni_worker CASCADE;
   CREATE EXTENSION IF NOT EXISTS omni_vfs CASCADE;
   CREATE EXTENSION IF NOT EXISTS vector CASCADE;
   ```

2. **Create cache management in Python**
   ```python
   # cache.py
   import hashlib
   import json
   from datetime import datetime
   import plpy

   class FrecencyCache:
       def __init__(self, table_name='steadytext_cache'):
           self.table_name = table_name

       def get_cache_key(self, prompt, params=None):
           """Generate deterministic cache key"""
           data = {'prompt': prompt, 'params': params or {}}
           return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

       def get(self, prompt, params=None):
           """Retrieve from cache and update access stats"""
           cache_key = self.get_cache_key(prompt, params)

           # Update access count and last_accessed atomically
           result = plpy.execute(f"""
               UPDATE {self.table_name}
               SET access_count = access_count + 1,
                   last_accessed = NOW()
               WHERE cache_key = %s
               RETURNING response, embedding
           """, [cache_key])

           if result:
               return result[0]['response'], result[0]['embedding']
           return None, None

       def set(self, prompt, response, embedding=None, params=None, model_version='1.0'):
           """Store in cache"""
           cache_key = self.get_cache_key(prompt, params)

           plpy.execute(f"""
               INSERT INTO {self.table_name}
               (cache_key, prompt, response, embedding, model_version, generation_params)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (cache_key)
               DO UPDATE SET
                   response = EXCLUDED.response,
                   embedding = EXCLUDED.embedding,
                   access_count = {self.table_name}.access_count + 1,
                   last_accessed = NOW()
           """, [cache_key, prompt, response, embedding, model_version, json.dumps(params)])
   ```

3. **Implement model loading**
   ```python
   # models.py
   import os
   import steadytext
   from threading import Lock

   class ModelManager:
       _instance = None
       _lock = Lock()

       def __new__(cls):
           if cls._instance is None:
               with cls._lock:
                   if cls._instance is None:
                       cls._instance = super().__new__(cls)
                       cls._instance.initialized = False
           return cls._instance

       def initialize(self):
           if not self.initialized:
               # Preload models on first use
               steadytext.preload_models(verbose=True)
               self.initialized = True

       def generate(self, prompt, **kwargs):
           self.initialize()
           return steadytext.generate(prompt, **kwargs)

       def embed(self, text):
           self.initialize()
           return steadytext.embed(text)
   ```

### Phase 2: Core Functions (Week 2)

1. **Implement generation functions**
   ```python
   # generation.py
   from omni_python import pg
   from .cache import FrecencyCache
   from .models import ModelManager

   @pg
   def steadytext_generate(prompt: str, max_tokens: int = 512, use_cache: bool = True) -> str:
       cache = FrecencyCache()
       model = ModelManager()

       if use_cache:
           cached_response, _ = cache.get(prompt, {'max_tokens': max_tokens})
           if cached_response:
               return cached_response

       # Generate new response
       response = model.generate(prompt, max_tokens=max_tokens)

       # Store in cache
       if use_cache:
           cache.set(prompt, response, params={'max_tokens': max_tokens})

       return response

   @pg
   def steadytext_generate_stream(prompt: str, max_tokens: int = 512):
       """Generator function for streaming"""
       model = ModelManager()

       for token in steadytext.generate_iter(prompt):
           yield token
   ```

2. **Implement embedding functions**
   ```python
   # embeddings.py
   import numpy as np
   from omni_python import pg

   @pg
   def steadytext_embed(text: str, use_cache: bool = True):
       cache = FrecencyCache()
       model = ModelManager()

       if use_cache:
           _, cached_embedding = cache.get(text)
           if cached_embedding is not None:
               return cached_embedding

       # Generate embedding
       embedding = model.embed(text)

       # Convert numpy array to PostgreSQL vector
       embedding_list = embedding.tolist()

       if use_cache:
           cache.set(text, None, embedding_list)

       return embedding_list
   ```

### Phase 3: Background Worker (Week 3)

1. **Implement worker for async processing**
   ```python
   # worker.py
   import json
   import traceback
   from datetime import datetime
   import plpy
   from omni_python import pg

   @pg
   def process_steadytext_queue():
       """Background worker to process queue"""
       while True:
           # Get next pending task
           result = plpy.execute("""
               UPDATE steadytext_queue
               SET status = 'processing',
                   started_at = NOW()
               WHERE id = (
                   SELECT id FROM steadytext_queue
                   WHERE status = 'pending'
                   ORDER BY created_at
                   FOR UPDATE SKIP LOCKED
                   LIMIT 1
               )
               RETURNING *
           """)

           if not result:
               # No tasks, sleep
               plpy.execute("SELECT pg_sleep(1)")
               continue

           task = result[0]

           try:
               if task['request_type'] == 'generate':
                   response = steadytext_generate(
                       task['prompt'],
                       **(task['params'] or {})
                   )

                   plpy.execute("""
                       UPDATE steadytext_queue
                       SET status = 'completed',
                           result = %s,
                           completed_at = NOW()
                       WHERE id = %s
                   """, [response, task['id']])

               elif task['request_type'] == 'embed':
                   embedding = steadytext_embed(task['prompt'])

                   plpy.execute("""
                       UPDATE steadytext_queue
                       SET status = 'completed',
                           embedding = %s,
                           completed_at = NOW()
                       WHERE id = %s
                   """, [embedding, task['id']])

           except Exception as e:
               plpy.execute("""
                   UPDATE steadytext_queue
                   SET status = 'failed',
                       error = %s,
                       completed_at = NOW()
                   WHERE id = %s
               """, [str(e), task['id']])
   ```

2. **Set up omni_worker**
   ```sql
   -- Register the worker
   SELECT omni_worker.register_worker(
       'steadytext_processor',
       'process_steadytext_queue'
   );

   -- Start workers (configure number based on load)
   SELECT omni_worker.start_workers('steadytext_processor', 4);
   ```

### Phase 4: Advanced Features (Week 4)

1. **Implement cache eviction**
   ```python
   @pg
   def steadytext_cache_evict(max_entries: int = None, max_size_mb: float = None) -> int:
       """Evict least frecent entries"""

       # Get current stats
       stats = plpy.execute("""
           SELECT
               COUNT(*) as total_entries,
               SUM(LENGTH(response) + COALESCE(LENGTH(embedding::text), 0)) / 1048576.0 as total_size_mb
           FROM steadytext_cache
       """)[0]

       entries_to_evict = 0

       if max_entries and stats['total_entries'] > max_entries:
           entries_to_evict = stats['total_entries'] - max_entries

       if max_size_mb and stats['total_size_mb'] > max_size_mb:
           # Estimate entries to evict based on average size
           avg_size = stats['total_size_mb'] / stats['total_entries']
           size_based_eviction = int((stats['total_size_mb'] - max_size_mb) / avg_size)
           entries_to_evict = max(entries_to_evict, size_based_eviction)

       if entries_to_evict > 0:
           # Delete entries with lowest frecency scores
           result = plpy.execute("""
               DELETE FROM steadytext_cache
               WHERE id IN (
                   SELECT id FROM steadytext_cache
                   ORDER BY frecency_score ASC
                   LIMIT %s
               )
           """, [entries_to_evict])

           return entries_to_evict

       return 0
   ```

2. **FAISS integration**
   ```python
   # faiss_integration.py
   import faiss
   import numpy as np
   import pickle

   @pg
   def steadytext_index_create(index_name: str, documents: list):
       """Create FAISS index from documents"""
       model = ModelManager()

       # Generate embeddings for all documents
       embeddings = []
       for doc in documents:
           emb = model.embed(doc)
           embeddings.append(emb)

       # Create FAISS index
       dimension = 1024
       index = faiss.IndexFlatL2(dimension)

       # Add vectors to index
       vectors = np.array(embeddings).astype('float32')
       index.add(vectors)

       # Serialize index
       index_bytes = pickle.dumps(index)

       # Store in database
       plpy.execute("""
           INSERT INTO steadytext_indexes (index_name, index_data, document_count)
           VALUES (%s, %s, %s)
           ON CONFLICT (index_name)
           DO UPDATE SET
               index_data = EXCLUDED.index_data,
               document_count = EXCLUDED.document_count,
               updated_at = NOW()
       """, [index_name, index_bytes, len(documents)])
   ```

## 🔐 Security Considerations

### 1. **Input Validation and Sanitization**

```python
# security.py
import re
import plpy
from typing import Optional, Dict, Any

class SecurityValidator:
    """Comprehensive input validation and sanitization"""

    # Maximum lengths to prevent DoS
    MAX_PROMPT_LENGTH = 10000
    MAX_BATCH_SIZE = 100
    MAX_INDEX_NAME_LENGTH = 63  # PostgreSQL identifier limit

    @staticmethod
    def validate_prompt(prompt: str) -> str:
        """Validate and sanitize text prompts"""
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        if len(prompt) > SecurityValidator.MAX_PROMPT_LENGTH:
            raise ValueError(f"Prompt exceeds maximum length of {SecurityValidator.MAX_PROMPT_LENGTH}")

        # Remove null bytes and control characters
        cleaned = prompt.replace('\0', '').strip()

        # Check for SQL injection patterns (defense in depth)
        sql_patterns = [
            r";\s*(DROP|DELETE|UPDATE|INSERT|CREATE|ALTER)",
            r"--\s*$",
            r"/\*.*\*/",
            r"'\s*OR\s*'1'\s*=\s*'1"
        ]

        for pattern in sql_patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                raise ValueError("Potentially malicious input detected")

        return cleaned

    @staticmethod
    def validate_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model generation parameters"""
        allowed_params = {
            'max_tokens': (int, 1, 4096),
            'temperature': (float, 0.0, 2.0),
            'seed': (int, 0, 2**31 - 1),
            'thinking_mode': (bool, None, None),
            'model': (str, None, None),
            'size': (str, None, None)
        }

        validated = {}
        for key, value in params.items():
            if key not in allowed_params:
                continue  # Silently ignore unknown params

            expected_type, min_val, max_val = allowed_params[key]

            if not isinstance(value, expected_type):
                raise ValueError(f"Parameter {key} must be of type {expected_type.__name__}")

            if min_val is not None and value < min_val:
                raise ValueError(f"Parameter {key} must be >= {min_val}")

            if max_val is not None and value > max_val:
                raise ValueError(f"Parameter {key} must be <= {max_val}")

            if key == 'model' and value not in ['qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b', 'qwen2.5-0.5b', 'qwen2.5-7b']:
                raise ValueError(f"Unknown model: {value}")

            if key == 'size' and value not in ['small', 'medium', 'large']:
                raise ValueError(f"Invalid size: {value}")

            validated[key] = value

        return validated
```

### 2. **Rate Limiting Implementation**

```python
@pg
def check_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded rate limits"""

    # Reset counters if needed
    plpy.execute("""
        UPDATE steadytext_rate_limits
        SET current_minute_count = CASE
                WHEN EXTRACT(EPOCH FROM (NOW() - last_reset_minute)) >= 60 THEN 0
                ELSE current_minute_count
            END,
            last_reset_minute = CASE
                WHEN EXTRACT(EPOCH FROM (NOW() - last_reset_minute)) >= 60 THEN NOW()
                ELSE last_reset_minute
            END,
            current_hour_count = CASE
                WHEN EXTRACT(EPOCH FROM (NOW() - last_reset_hour)) >= 3600 THEN 0
                ELSE current_hour_count
            END,
            last_reset_hour = CASE
                WHEN EXTRACT(EPOCH FROM (NOW() - last_reset_hour)) >= 3600 THEN NOW()
                ELSE last_reset_hour
            END,
            current_day_count = CASE
                WHEN EXTRACT(EPOCH FROM (NOW() - last_reset_day)) >= 86400 THEN 0
                ELSE current_day_count
            END,
            last_reset_day = CASE
                WHEN EXTRACT(EPOCH FROM (NOW() - last_reset_day)) >= 86400 THEN NOW()
                ELSE last_reset_day
            END
        WHERE user_id = %s
    """, [user_id])

    # Check limits
    result = plpy.execute("""
        SELECT
            current_minute_count >= requests_per_minute AS minute_exceeded,
            current_hour_count >= requests_per_hour AS hour_exceeded,
            current_day_count >= requests_per_day AS day_exceeded
        FROM steadytext_rate_limits
        WHERE user_id = %s
    """, [user_id])

    if not result:
        # Create default limits for new user
        plpy.execute("""
            INSERT INTO steadytext_rate_limits (user_id)
            VALUES (%s)
            ON CONFLICT (user_id) DO NOTHING
        """, [user_id])
        return True

    row = result[0]
    if row['minute_exceeded'] or row['hour_exceeded'] or row['day_exceeded']:
        return False

    # Increment counters
    plpy.execute("""
        UPDATE steadytext_rate_limits
        SET current_minute_count = current_minute_count + 1,
            current_hour_count = current_hour_count + 1,
            current_day_count = current_day_count + 1
        WHERE user_id = %s
    """, [user_id])

    return True
```

### 3. **Access Control and Permissions**

```sql
-- Create roles for different access levels
CREATE ROLE steadytext_reader;
CREATE ROLE steadytext_user;
CREATE ROLE steadytext_admin;

-- Reader: Can only query results
GRANT SELECT ON steadytext_cache TO steadytext_reader;
GRANT SELECT ON steadytext_queue TO steadytext_reader;
GRANT EXECUTE ON FUNCTION steadytext_cache_stats() TO steadytext_reader;

-- User: Can generate text and embeddings
GRANT steadytext_reader TO steadytext_user;
GRANT INSERT ON steadytext_queue TO steadytext_user;
GRANT EXECUTE ON FUNCTION steadytext_generate(TEXT, INT, BOOLEAN) TO steadytext_user;
GRANT EXECUTE ON FUNCTION steadytext_generate_async(TEXT, INT) TO steadytext_user;
GRANT EXECUTE ON FUNCTION steadytext_embed(TEXT, BOOLEAN) TO steadytext_user;

-- Admin: Full access
GRANT steadytext_user TO steadytext_admin;
GRANT ALL ON ALL TABLES IN SCHEMA public TO steadytext_admin;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO steadytext_admin;

-- Row-level security for multi-tenant environments
ALTER TABLE steadytext_queue ENABLE ROW LEVEL SECURITY;

CREATE POLICY queue_user_policy ON steadytext_queue
    FOR ALL
    TO steadytext_user
    USING (user_id = current_user)
    WITH CHECK (user_id = current_user);
```

### 4. **Audit Logging**

```python
def log_audit(action: str, request_id: Optional[str] = None,
              details: Optional[Dict] = None, success: bool = True,
              error: Optional[str] = None):
    """Log security-relevant events"""
    plpy.execute("""
        INSERT INTO steadytext_audit_log
        (action, request_id, details, success, error_message, ip_address)
        VALUES (%s, %s, %s, %s, %s, inet_client_addr())
    """, [action, request_id, json.dumps(details), success, error])
```

### 5. **Secure Communication**

```python
class SecureDaemonConnector(SteadyTextConnector):
    """Enhanced connector with security features"""

    def __init__(self, host='localhost', port=5555, auth_token=None):
        super().__init__(host, port)
        self.auth_token = auth_token or os.environ.get('STEADYTEXT_AUTH_TOKEN')

    def _secure_request(self, request_type: str, **kwargs):
        """Send authenticated request to daemon"""
        # Add authentication token to request
        if self.auth_token:
            kwargs['auth_token'] = self.auth_token

        # Add request signature for integrity
        kwargs['timestamp'] = datetime.utcnow().isoformat()
        kwargs['signature'] = self._generate_signature(kwargs)

        return self._send_request(request_type, **kwargs)

    def _generate_signature(self, data: Dict) -> str:
        """Generate HMAC signature for request integrity"""
        if not self.auth_token:
            return ""

        message = json.dumps(data, sort_keys=True)
        return hmac.new(
            self.auth_token.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
```

## 🚀 Deployment Strategy

### 1. **Docker-based deployment**
```dockerfile
FROM ghcr.io/omnigres/omnigres-17:latest

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Copy extension files
COPY python-files/ /python-files/
COPY sql/ /sql/

# Install extension
RUN psql -U omnigres -d omnigres -f /sql/install.sql
```

### 2. **Installation script**
```sql
-- install.sql
CREATE EXTENSION IF NOT EXISTS omni_python CASCADE;
CREATE EXTENSION IF NOT EXISTS omni_worker CASCADE;
CREATE EXTENSION IF NOT EXISTS vector CASCADE;

-- Create schema
CREATE SCHEMA IF NOT EXISTS steadytext;

-- Load Python modules
SELECT omni_python.load_module('steadytext_wrapper', '/python-files/steadytext_wrapper');

-- Create tables
\i tables.sql

-- Create functions
\i functions.sql

-- Configure cache settings
INSERT INTO steadytext_config (key, value) VALUES
    ('cache_max_entries', '10000'::jsonb),
    ('cache_max_size_mb', '500'::jsonb),
    ('worker_count', '4'::jsonb)
ON CONFLICT (key) DO NOTHING;

-- Start background workers
SELECT omni_worker.start_workers('steadytext_processor', 4);
```

## 📊 Performance Optimizations

1. **Connection pooling** - Use pgBouncer for worker connections
2. **Batch processing** - Process multiple embeddings in single call
3. **Partial indexes** - Index only frequently accessed cache entries
4. **Table partitioning** - Partition queue table by date for easier cleanup
5. **Vacuum strategy** - Regular vacuum on high-churn tables

## 🔍 Enhanced Monitoring & Observability

### 1. **Prometheus Metrics Integration**

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps

# Define metrics
request_counter = Counter(
    'steadytext_requests_total',
    'Total number of requests',
    ['method', 'status', 'model']
)

request_duration = Histogram(
    'steadytext_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'model'],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

cache_hit_rate = Gauge(
    'steadytext_cache_hit_rate',
    'Cache hit rate percentage'
)

queue_size = Gauge(
    'steadytext_queue_size',
    'Current queue size by status',
    ['status']
)

daemon_health = Gauge(
    'steadytext_daemon_health',
    'Daemon health status (1=healthy, 0=unhealthy)',
    ['daemon_id']
)

model_info = Info(
    'steadytext_model',
    'Information about loaded models'
)

memory_usage = Gauge(
    'steadytext_memory_usage_bytes',
    'Memory usage in bytes',
    ['component']
)

def track_metrics(method='unknown', model='default'):
    """Decorator to track function metrics"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                request_counter.labels(method=method, status=status, model=model).inc()
                request_duration.labels(method=method, model=model).observe(duration)

        return wrapper
    return decorator

# Usage in functions
@pg
@track_metrics(method='generate', model='qwen3-1.7b')
def steadytext_generate_with_metrics(prompt: str, **kwargs):
    return steadytext_generate(prompt, **kwargs)
```

### 2. **OpenTelemetry Tracing**

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
import plpy

# Initialize tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer("pg_steadytext")

# Configure OTLP exporter
otlp_exporter = OTLPSpanExporter(
    endpoint="localhost:4317",
    insecure=True
)

span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument PostgreSQL
Psycopg2Instrumentor().instrument()

@pg
def traced_generate(prompt: str, **kwargs):
    """Generate with distributed tracing"""
    with tracer.start_as_current_span("steadytext.generate") as span:
        span.set_attribute("prompt.length", len(prompt))
        span.set_attribute("model", kwargs.get('model', 'default'))

        # Check cache
        with tracer.start_as_current_span("cache.lookup"):
            cache_result = check_cache(prompt, kwargs)
            if cache_result:
                span.set_attribute("cache.hit", True)
                return cache_result

        # Generate new result
        with tracer.start_as_current_span("daemon.generate"):
            result = generate_via_daemon(prompt, **kwargs)

        # Store in cache
        with tracer.start_as_current_span("cache.store"):
            store_in_cache(prompt, result, kwargs)

        span.set_attribute("cache.hit", False)
        span.set_attribute("result.length", len(result))
        return result
```

### 3. **Comprehensive Monitoring Views**

```sql
-- Real-time performance dashboard
CREATE VIEW steadytext_performance_dashboard AS
WITH recent_requests AS (
    SELECT * FROM steadytext_queue
    WHERE created_at > NOW() - INTERVAL '1 hour'
)
SELECT
    -- Request metrics
    (SELECT json_build_object(
        'total_requests_1h', COUNT(*),
        'completed_requests_1h', COUNT(*) FILTER (WHERE status = 'completed'),
        'failed_requests_1h', COUNT(*) FILTER (WHERE status = 'failed'),
        'avg_latency_ms', AVG(EXTRACT(EPOCH FROM (completed_at - created_at)) * 1000)
            FILTER (WHERE status = 'completed'),
        'p50_latency_ms', PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (completed_at - created_at)) * 1000
        ) FILTER (WHERE status = 'completed'),
        'p95_latency_ms', PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (completed_at - created_at)) * 1000
        ) FILTER (WHERE status = 'completed'),
        'p99_latency_ms', PERCENTILE_CONT(0.99) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (completed_at - created_at)) * 1000
        ) FILTER (WHERE status = 'completed')
    ) FROM recent_requests) as request_metrics,

    -- Cache performance
    (SELECT json_build_object(
        'cache_size_mb', SUM(LENGTH(response) + COALESCE(LENGTH(embedding::text), 0)) / 1048576.0,
        'cache_entries', COUNT(*),
        'cache_hit_rate',
            CASE
                WHEN SUM(access_count) > 0
                THEN (SUM(access_count) - COUNT(*))::FLOAT / SUM(access_count) * 100
                ELSE 0
            END,
        'most_accessed_prompts', (
            SELECT json_agg(json_build_object(
                'prompt', LEFT(prompt, 50),
                'access_count', access_count
            ))
            FROM (
                SELECT prompt, access_count
                FROM steadytext_cache
                ORDER BY access_count DESC
                LIMIT 5
            ) top_prompts
        )
    ) FROM steadytext_cache) as cache_metrics,

    -- Model usage
    (SELECT json_build_object(
        'models_used', array_agg(DISTINCT model_name),
        'model_distribution', json_object_agg(model_name, count)
    ) FROM (
        SELECT model_name, COUNT(*) as count
        FROM steadytext_cache
        WHERE created_at > NOW() - INTERVAL '1 hour'
        GROUP BY model_name
    ) model_stats) as model_metrics,

    -- System health
    (SELECT json_build_object(
        'healthy_daemons', COUNT(*) FILTER (WHERE status = 'healthy'),
        'total_daemons', COUNT(*),
        'avg_daemon_memory_mb', AVG(memory_usage_mb),
        'total_daemon_requests', SUM(total_requests)
    ) FROM steadytext_daemon_health) as system_health;

-- Detailed error tracking
CREATE VIEW steadytext_error_analysis AS
SELECT
    DATE_TRUNC('hour', completed_at) as error_hour,
    error,
    COUNT(*) as error_count,
    array_agg(DISTINCT user_id) as affected_users,
    MIN(completed_at) as first_occurrence,
    MAX(completed_at) as last_occurrence
FROM steadytext_queue
WHERE status = 'failed'
  AND completed_at > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', completed_at), error
ORDER BY error_hour DESC, error_count DESC;
```

### 4. **Health Check Endpoints**

```python
@pg
def steadytext_health_check() -> Dict[str, Any]:
    """Comprehensive health check for monitoring systems"""

    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {}
    }

    # Check daemon connectivity
    try:
        with daemon_pool.get_connection(timeout=1.0) as conn:
            conn.generate("health check", max_tokens=1)
        health_status['checks']['daemon'] = {'status': 'pass', 'latency_ms': 0}
    except Exception as e:
        health_status['checks']['daemon'] = {'status': 'fail', 'error': str(e)}
        health_status['status'] = 'degraded'

    # Check database
    try:
        result = plpy.execute("SELECT 1")
        health_status['checks']['database'] = {'status': 'pass'}
    except Exception as e:
        health_status['checks']['database'] = {'status': 'fail', 'error': str(e)}
        health_status['status'] = 'unhealthy'

    # Check queue depth
    queue_depth = plpy.execute("""
        SELECT COUNT(*) as pending FROM steadytext_queue
        WHERE status = 'pending'
    """)[0]['pending']

    if queue_depth > ResourceLimits.MAX_QUEUE_SIZE * 0.9:
        health_status['checks']['queue'] = {
            'status': 'warn',
            'queue_depth': queue_depth,
            'threshold': ResourceLimits.MAX_QUEUE_SIZE
        }
        if health_status['status'] == 'healthy':
            health_status['status'] = 'degraded'
    else:
        health_status['checks']['queue'] = {'status': 'pass', 'queue_depth': queue_depth}

    # Check memory usage
    memory = check_memory_usage()
    if memory['total_mb'] > ResourceLimits.MAX_TOTAL_MEMORY_MB * 0.9:
        health_status['checks']['memory'] = {
            'status': 'warn',
            'usage_mb': memory['total_mb'],
            'threshold_mb': ResourceLimits.MAX_TOTAL_MEMORY_MB
        }
        if health_status['status'] == 'healthy':
            health_status['status'] = 'degraded'
    else:
        health_status['checks']['memory'] = {'status': 'pass', 'usage_mb': memory['total_mb']}

    return health_status
```

### 5. **Logging Infrastructure**

```python
# logging_config.py
import logging
import json
from pythonjsonlogger import jsonlogger

# Configure structured logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    fmt='%(timestamp)s %(level)s %(name)s %(message)s',
    rename_fields={'timestamp': '@timestamp'}
)
logHandler.setFormatter(formatter)

logger = logging.getLogger('pg_steadytext')
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

class QueryLogger:
    """Log all queries with performance metrics"""

    @staticmethod
    def log_query(func_name: str, params: Dict, duration_ms: float,
                  result_size: int, cache_hit: bool = False):
        logger.info("query_executed", extra={
            'function': func_name,
            'parameters': params,
            'duration_ms': duration_ms,
            'result_size': result_size,
            'cache_hit': cache_hit,
            'user': plpy.execute("SELECT current_user")[0]['current_user'],
            'database': plpy.execute("SELECT current_database()")[0]['current_database']
        })

    @staticmethod
    def log_error(func_name: str, error: Exception, params: Dict):
        logger.error("query_error", extra={
            'function': func_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'parameters': params,
            'traceback': traceback.format_exc()
        })
```

### 6. **Grafana Dashboard Configuration**

```json
{
  "dashboard": {
    "title": "pg_steadytext Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(steadytext_requests_total[5m])",
            "legendFormat": "{{method}} - {{status}}"
          }
        ]
      },
      {
        "title": "Latency Percentiles",
        "targets": [
          {
            "expr": "histogram_quantile(0.5, steadytext_request_duration_seconds_bucket)",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, steadytext_request_duration_seconds_bucket)",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, steadytext_request_duration_seconds_bucket)",
            "legendFormat": "p99"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "steadytext_cache_hit_rate",
            "legendFormat": "Hit Rate %"
          }
        ]
      },
      {
        "title": "Queue Depth",
        "targets": [
          {
            "expr": "steadytext_queue_size",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "steadytext_memory_usage_bytes / 1024 / 1024",
            "legendFormat": "{{component}} MB"
          }
        ]
      }
    ]
  }
}
```

### 7. **Alerting Rules**

```yaml
# prometheus/alerts.yml
groups:
  - name: pg_steadytext
    rules:
      - alert: HighErrorRate
        expr: rate(steadytext_requests_total{status="error"}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: SlowRequests
        expr: histogram_quantile(0.95, steadytext_request_duration_seconds_bucket) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Slow request latency"
          description: "95th percentile latency is {{ $value }} seconds"

      - alert: LowCacheHitRate
        expr: steadytext_cache_hit_rate < 50
        for: 15m
        labels:
          severity: info
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value }}%"

      - alert: QueueBacklog
        expr: steadytext_queue_size{status="pending"} > 1000
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Large queue backlog"
          description: "{{ $value }} requests pending in queue"

      - alert: DaemonUnhealthy
        expr: steadytext_daemon_health == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Daemon unhealthy"
          description: "Daemon {{ $labels.daemon_id }} is unhealthy"
```

## 💾 Resource Management

### 1. **Memory Management**

```python
# config.py
class ResourceLimits:
    """Central configuration for resource limits"""

    # Memory limits
    MAX_MEMORY_PER_MODEL_MB = 4096  # 4GB per model
    MAX_TOTAL_MEMORY_MB = 16384     # 16GB total for all models
    MAX_CACHE_MEMORY_MB = 2048      # 2GB for cache data

    # Queue limits
    MAX_QUEUE_SIZE = 10000
    MAX_PENDING_PER_USER = 100
    MAX_BATCH_SIZE = 100

    # Connection limits
    MAX_DAEMON_CONNECTIONS = 10
    CONNECTION_TIMEOUT_MS = 5000
    REQUEST_TIMEOUT_MS = 300000  # 5 minutes

@pg
def check_memory_usage() -> Dict[str, float]:
    """Monitor memory usage across components"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    # Get model memory usage from daemon
    daemon_memory = plpy.execute("""
        SELECT SUM(memory_usage_mb) as total_mb
        FROM steadytext_daemon_health
        WHERE status = 'healthy'
    """)[0]['total_mb'] or 0

    # Get cache memory usage
    cache_memory = plpy.execute("""
        SELECT
            SUM(LENGTH(response) + COALESCE(LENGTH(embedding::text), 0)) / 1048576.0 as cache_mb
        FROM steadytext_cache
    """)[0]['cache_mb'] or 0

    return {
        'postgres_mb': memory_info.rss / 1048576.0,
        'daemon_mb': daemon_memory,
        'cache_mb': cache_memory,
        'total_mb': memory_info.rss / 1048576.0 + daemon_memory
    }

@pg
def enforce_memory_limits():
    """Enforce memory limits by evicting cache or refusing requests"""
    usage = check_memory_usage()

    if usage['cache_mb'] > ResourceLimits.MAX_CACHE_MEMORY_MB:
        # Trigger cache eviction
        evicted = steadytext_cache_evict(
            max_size_mb=ResourceLimits.MAX_CACHE_MEMORY_MB * 0.8  # Keep 80% after eviction
        )
        log_audit('cache_eviction', details={'evicted_count': evicted, 'reason': 'memory_limit'})

    if usage['total_mb'] > ResourceLimits.MAX_TOTAL_MEMORY_MB:
        raise Exception("System memory limit exceeded. Please retry later.")
```

### 2. **Queue Management**

```python
@pg
def manage_queue_size():
    """Prevent queue from growing unbounded"""

    # Check queue size
    queue_stats = plpy.execute("""
        SELECT
            COUNT(*) as total_count,
            COUNT(*) FILTER (WHERE status = 'pending') as pending_count,
            COUNT(*) FILTER (WHERE created_at < NOW() - INTERVAL '1 hour' AND status = 'pending') as stale_count
        FROM steadytext_queue
    """)[0]

    # Cancel stale requests
    if queue_stats['stale_count'] > 0:
        plpy.execute("""
            UPDATE steadytext_queue
            SET status = 'cancelled',
                error = 'Request timed out',
                completed_at = NOW()
            WHERE created_at < NOW() - INTERVAL '1 hour'
            AND status = 'pending'
        """)

    # Reject new requests if queue is full
    if queue_stats['pending_count'] >= ResourceLimits.MAX_QUEUE_SIZE:
        raise Exception("Queue is full. Please retry later.")

    # Clean up old completed requests
    plpy.execute("""
        DELETE FROM steadytext_queue
        WHERE completed_at < NOW() - INTERVAL '7 days'
        AND status IN ('completed', 'failed', 'cancelled')
    """)
```

### 3. **Connection Pool Management**

```python
import queue
import threading
from contextlib import contextmanager

class DaemonConnectionPool:
    """Connection pool for daemon communication"""

    def __init__(self, max_connections=10):
        self.max_connections = max_connections
        self.pool = queue.Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        """Create initial connections"""
        for _ in range(self.max_connections):
            conn = SteadyTextConnector()
            self.pool.put(conn)

    @contextmanager
    def get_connection(self, timeout=5.0):
        """Get connection from pool with timeout"""
        conn = None
        try:
            conn = self.pool.get(timeout=timeout)
            yield conn
        except queue.Empty:
            raise Exception("Connection pool exhausted")
        finally:
            if conn:
                # Check if connection is still healthy
                if self._is_healthy(conn):
                    self.pool.put(conn)
                else:
                    # Replace with new connection
                    try:
                        new_conn = SteadyTextConnector()
                        self.pool.put(new_conn)
                    except:
                        pass  # Pool will shrink if we can't create new connections

    def _is_healthy(self, conn):
        """Check if connection is still usable"""
        try:
            # Send ping to daemon
            conn.generate("", max_tokens=1)  # Minimal request
            return True
        except:
            return False

# Global connection pool
daemon_pool = DaemonConnectionPool(max_connections=ResourceLimits.MAX_DAEMON_CONNECTIONS)
```

### 4. **Automatic Resource Cleanup**

```sql
-- Scheduled cleanup job
CREATE OR REPLACE FUNCTION steadytext_cleanup_job()
RETURNS void AS $$
BEGIN
    -- Clean old queue entries
    DELETE FROM steadytext_queue
    WHERE completed_at < NOW() - INTERVAL '7 days';

    -- Clean old audit logs
    DELETE FROM steadytext_audit_log
    WHERE timestamp < NOW() - INTERVAL '30 days';

    -- Reset stale daemon health entries
    UPDATE steadytext_daemon_health
    SET status = 'unhealthy'
    WHERE last_heartbeat < NOW() - INTERVAL '5 minutes';

    -- Evict least used cache entries if needed
    PERFORM steadytext_cache_evict(
        max_entries := (SELECT value->>'cache_max_entries' FROM steadytext_config WHERE key = 'cache_max_entries')::int
    );
END;
$$ LANGUAGE plpgsql;

-- Schedule with pg_cron (if available)
SELECT cron.schedule('steadytext-cleanup', '0 * * * *', 'SELECT steadytext_cleanup_job()');
```

### 5. **Resource Monitoring Dashboard**

```sql
CREATE VIEW steadytext_resource_dashboard AS
SELECT
    -- Memory usage
    (SELECT json_build_object(
        'postgres_mb', pg_database_size(current_database()) / 1048576.0,
        'cache_mb', (SELECT SUM(LENGTH(response) + COALESCE(LENGTH(embedding::text), 0)) / 1048576.0 FROM steadytext_cache),
        'models_loaded', (SELECT array_agg(DISTINCT model_name) FROM steadytext_cache WHERE created_at > NOW() - INTERVAL '1 hour')
    )) as memory_stats,

    -- Queue stats
    (SELECT json_build_object(
        'pending', COUNT(*) FILTER (WHERE status = 'pending'),
        'processing', COUNT(*) FILTER (WHERE status = 'processing'),
        'completed_1h', COUNT(*) FILTER (WHERE status = 'completed' AND completed_at > NOW() - INTERVAL '1 hour'),
        'failed_1h', COUNT(*) FILTER (WHERE status = 'failed' AND completed_at > NOW() - INTERVAL '1 hour'),
        'avg_wait_time_ms', AVG(EXTRACT(EPOCH FROM (started_at - created_at)) * 1000) FILTER (WHERE started_at IS NOT NULL)
    ) FROM steadytext_queue) as queue_stats,

    -- Daemon health
    (SELECT json_build_object(
        'healthy_daemons', COUNT(*) FILTER (WHERE status = 'healthy'),
        'total_daemons', COUNT(*),
        'total_requests', SUM(total_requests),
        'avg_response_time_ms', AVG(avg_response_time_ms)
    ) FROM steadytext_daemon_health) as daemon_stats,

    -- Rate limit stats
    (SELECT json_build_object(
        'users_at_limit', COUNT(*) FILTER (WHERE
            current_minute_count >= requests_per_minute OR
            current_hour_count >= requests_per_hour OR
            current_day_count >= requests_per_day
        ),
        'total_users', COUNT(*)
    ) FROM steadytext_rate_limits) as rate_limit_stats;
```

## 🧪 Comprehensive Testing Strategy

### 1. **Unit Testing with pgTAP**

```sql
-- Install pgTAP for PostgreSQL testing
CREATE EXTENSION IF NOT EXISTS pgtap;

-- tests/unit/test_cache.sql
BEGIN;
SELECT plan(10);

-- Test cache key generation
SELECT is(
    steadytext_cache_key('Hello world', '{"max_tokens": 100}'::jsonb),
    steadytext_cache_key('Hello world', '{"max_tokens": 100}'::jsonb),
    'Cache key generation is deterministic'
);

-- Test prompt validation
SELECT throws_ok(
    $$ SELECT validate_prompt(NULL) $$,
    'Prompt must be a non-empty string',
    'NULL prompt throws error'
);

SELECT throws_ok(
    $$ SELECT validate_prompt(repeat('x', 10001)) $$,
    'Prompt exceeds maximum length',
    'Oversized prompt throws error'
);

-- Test rate limiting
SELECT ok(
    check_rate_limit('test_user'),
    'First request passes rate limit'
);

-- Test cache operations
INSERT INTO steadytext_cache (cache_key, prompt, response, model_name)
VALUES ('test_key', 'test prompt', 'test response', 'qwen3-1.7b');

SELECT is(
    (SELECT response FROM steadytext_cache WHERE cache_key = 'test_key'),
    'test response',
    'Cache retrieval works'
);

SELECT finish();
ROLLBACK;
```

### 2. **Integration Testing**

```python
# tests/integration/test_generation.py
import pytest
import psycopg2
from steadytext import generate, embed

class TestPgSteadyTextIntegration:

    @pytest.fixture
    def db_conn(self):
        conn = psycopg2.connect(
            host="localhost",
            database="test_db",
            user="test_user",
            password="test_pass"
        )
        yield conn
        conn.close()

    def test_generate_function(self, db_conn):
        """Test PostgreSQL generate function"""
        cursor = db_conn.cursor()

        # Test synchronous generation
        cursor.execute("SELECT steadytext_generate('Hello world', 100, true)")
        result = cursor.fetchone()[0]

        # Verify it matches direct library call
        direct_result = generate("Hello world", max_tokens=100)
        assert result == direct_result

    def test_async_generation(self, db_conn):
        """Test async generation with queue"""
        cursor = db_conn.cursor()

        # Submit async request
        cursor.execute("SELECT steadytext_generate_async('Test prompt', 50)")
        request_id = cursor.fetchone()[0]

        # Wait for completion
        import time
        for _ in range(30):  # 30 second timeout
            cursor.execute(
                "SELECT status, result FROM steadytext_queue WHERE request_id = %s",
                (request_id,)
            )
            status, result = cursor.fetchone()
            if status == 'completed':
                assert result is not None
                break
            time.sleep(1)
        else:
            pytest.fail("Async generation timed out")

    def test_batch_embeddings(self, db_conn):
        """Test batch embedding generation"""
        cursor = db_conn.cursor()

        texts = ['Hello', 'World', 'Test']
        cursor.execute(
            "SELECT * FROM steadytext_embed_batch(%s)",
            (texts,)
        )

        results = cursor.fetchall()
        assert len(results) == 3
        for text, embedding in results:
            assert len(embedding) == 1024  # Verify dimension
```

### 3. **Performance and Load Testing**

```python
# tests/performance/test_load.py
import asyncio
import aiohttp
import time
from statistics import mean, stdev

class LoadTester:

    def __init__(self, base_url, num_users=100, requests_per_user=10):
        self.base_url = base_url
        self.num_users = num_users
        self.requests_per_user = requests_per_user
        self.response_times = []

    async def simulate_user(self, session, user_id):
        """Simulate a single user making requests"""
        for i in range(self.requests_per_user):
            start_time = time.time()

            prompt = f"User {user_id} request {i}"
            payload = {
                "prompt": prompt,
                "max_tokens": 100
            }

            try:
                async with session.post(
                    f"{self.base_url}/generate",
                    json=payload
                ) as response:
                    await response.json()

                response_time = time.time() - start_time
                self.response_times.append(response_time)

            except Exception as e:
                print(f"Request failed: {e}")

    async def run_load_test(self):
        """Run concurrent load test"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.simulate_user(session, user_id)
                for user_id in range(self.num_users)
            ]
            await asyncio.gather(*tasks)

        # Calculate statistics
        return {
            "total_requests": len(self.response_times),
            "avg_response_time": mean(self.response_times),
            "std_dev": stdev(self.response_times),
            "p95": sorted(self.response_times)[int(len(self.response_times) * 0.95)],
            "p99": sorted(self.response_times)[int(len(self.response_times) * 0.99)]
        }

# Run load test
async def test_load():
    tester = LoadTester("http://localhost:8080", num_users=100)
    results = await tester.run_load_test()

    # Assert performance requirements
    assert results["avg_response_time"] < 1.0  # Average under 1 second
    assert results["p99"] < 5.0  # 99th percentile under 5 seconds
```

### 4. **Chaos Testing**

```python
# tests/chaos/test_resilience.py
import random
import subprocess
import time

class ChaosTest:

    def test_daemon_failure_recovery(self):
        """Test system behavior when daemon crashes"""

        # Kill daemon process
        subprocess.run(["st", "daemon", "stop", "--force"])

        # Try to generate text (should fallback or restart)
        start_time = time.time()
        result = steadytext_generate("Test after daemon crash")
        elapsed = time.time() - start_time

        assert result is not None
        assert elapsed < 10  # Should recover within 10 seconds

    def test_database_connection_loss(self):
        """Test handling of database connection failures"""

        # Simulate connection pool exhaustion
        connections = []
        try:
            for _ in range(200):  # Exceed max connections
                conn = psycopg2.connect(**db_config)
                connections.append(conn)
        except psycopg2.OperationalError:
            pass  # Expected

        # System should still handle requests gracefully
        result = check_system_health()
        assert result["status"] != "critical"

        # Cleanup
        for conn in connections:
            conn.close()

    def test_memory_pressure(self):
        """Test behavior under memory pressure"""

        # Generate many large requests
        large_prompts = [
            "x" * 5000 for _ in range(100)
        ]

        results = []
        for prompt in large_prompts:
            try:
                result = steadytext_generate_async(prompt)
                results.append(result)
            except Exception as e:
                # Should fail gracefully
                assert "memory limit" in str(e).lower()

        # Check that some requests succeeded
        assert len(results) > 0
```

### 5. **Determinism Testing**

```python
# tests/determinism/test_consistency.py
def test_generation_determinism():
    """Verify identical inputs produce identical outputs"""

    prompt = "Explain quantum computing"
    params = {"max_tokens": 200, "seed": 42}

    # Generate multiple times
    results = []
    for _ in range(10):
        result = steadytext_generate(prompt, **params)
        results.append(result)

    # All results should be identical
    assert all(r == results[0] for r in results)

def test_cross_instance_determinism():
    """Test determinism across different daemon instances"""

    # Start two daemons on different ports
    daemon1 = start_daemon(port=5555)
    daemon2 = start_daemon(port=5556)

    prompt = "Hello world"

    result1 = generate_with_daemon(daemon1, prompt)
    result2 = generate_with_daemon(daemon2, prompt)

    assert result1 == result2

    daemon1.stop()
    daemon2.stop()
```

### 6. **Security Testing**

```python
# tests/security/test_injection.py
def test_sql_injection_prevention():
    """Test SQL injection attack prevention"""

    malicious_prompts = [
        "'; DROP TABLE steadytext_cache; --",
        "' OR '1'='1",
        "'; INSERT INTO steadytext_cache VALUES ('hack', 'hacked'); --"
    ]

    for prompt in malicious_prompts:
        with pytest.raises(ValueError, match="Potentially malicious"):
            validate_prompt(prompt)

def test_rate_limit_enforcement():
    """Test rate limiting prevents abuse"""

    user_id = "test_attacker"

    # Reset rate limits
    reset_rate_limits(user_id)

    # Try to exceed rate limit
    success_count = 0
    for i in range(100):  # Try 100 requests rapidly
        if check_rate_limit(user_id):
            success_count += 1

    # Should be limited
    assert success_count < 70  # Less than per-minute limit
```

### 7. **Test Automation and CI/CD**

```yaml
# .github/workflows/test.yml
name: pg_steadytext tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: ghcr.io/omnigres/omnigres-17:latest
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt

    - name: Install pgTAP
      run: |
        sudo apt-get install postgresql-14-pgtap

    - name: Run unit tests
      run: |
        pg_prove -d test_db tests/unit/*.sql

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v

    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark

    - name: Run security tests
      run: |
        pytest tests/security/ -v

    - name: Generate coverage report
      run: |
        coverage run -m pytest
        coverage xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 🔄 Migration Strategy

### 1. **Version Management**

```sql
-- Extension versioning table
CREATE TABLE steadytext_version (
    version TEXT PRIMARY KEY,
    installed_at TIMESTAMPTZ DEFAULT NOW(),
    upgraded_from TEXT,
    notes TEXT
);

-- Migration history
CREATE TABLE steadytext_migrations (
    id SERIAL PRIMARY KEY,
    version TEXT NOT NULL,
    migration_name TEXT NOT NULL,
    executed_at TIMESTAMPTZ DEFAULT NOW(),
    execution_time_ms INT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    UNIQUE(version, migration_name)
);
```

### 2. **Upgrade Scripts**

```sql
-- Example: Upgrade from 1.0.0 to 1.1.0
-- File: updates/steadytext--1.0.0--1.1.0.sql

-- Add new columns with defaults (non-breaking)
ALTER TABLE steadytext_cache
ADD COLUMN IF NOT EXISTS generation_time_ms INT,
ADD COLUMN IF NOT EXISTS thinking_mode BOOLEAN DEFAULT FALSE;

-- Add new indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cache_model_thinking
ON steadytext_cache(model_name, thinking_mode);

-- Migrate existing data
UPDATE steadytext_cache
SET thinking_mode = (generation_params->>'thinking_mode')::boolean
WHERE generation_params ? 'thinking_mode';

-- Update version
INSERT INTO steadytext_version (version, upgraded_from, notes)
VALUES ('1.1.0', '1.0.0', 'Added thinking mode support');
```

### 3. **Backward Compatibility**

```python
# version_compat.py
from functools import wraps
import plpy

class VersionCompat:
    """Handle backward compatibility across versions"""

    @staticmethod
    def get_current_version():
        result = plpy.execute("""
            SELECT version FROM steadytext_version
            ORDER BY installed_at DESC LIMIT 1
        """)
        return result[0]['version'] if result else '1.0.0'

    @staticmethod
    def supports_feature(feature_name: str) -> bool:
        """Check if current version supports a feature"""
        version = VersionCompat.get_current_version()

        feature_map = {
            'thinking_mode': '1.1.0',
            'batch_embeddings': '1.2.0',
            'priority_queue': '1.3.0',
            'model_switching': '1.0.0'
        }

        required_version = feature_map.get(feature_name, '999.0.0')
        return version >= required_version

def version_check(min_version: str):
    """Decorator to check version requirements"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current = VersionCompat.get_current_version()
            if current < min_version:
                raise Exception(f"Function {func.__name__} requires version {min_version} or higher (current: {current})")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage example
@pg
@version_check('1.1.0')
def steadytext_generate_with_thinking(prompt: str, thinking_mode: bool = True):
    """Generate with thinking mode (requires v1.1.0+)"""
    return steadytext_generate(prompt, thinking_mode=thinking_mode)
```

### 4. **Zero-Downtime Migration Process**

```sql
-- Pre-migration checks
CREATE OR REPLACE FUNCTION steadytext_pre_migration_check(target_version TEXT)
RETURNS TABLE(check_name TEXT, status TEXT, details TEXT) AS $$
BEGIN
    -- Check disk space
    RETURN QUERY
    SELECT 'disk_space'::TEXT,
           CASE WHEN pg_database_size(current_database()) < 1000000000000
                THEN 'PASS' ELSE 'WARN' END,
           'Database size: ' || pg_size_pretty(pg_database_size(current_database()));

    -- Check active connections
    RETURN QUERY
    SELECT 'active_connections'::TEXT,
           CASE WHEN count(*) < 100 THEN 'PASS' ELSE 'WARN' END,
           'Active connections: ' || count(*)::TEXT
    FROM pg_stat_activity
    WHERE state = 'active';

    -- Check queue status
    RETURN QUERY
    SELECT 'queue_status'::TEXT,
           CASE WHEN count(*) = 0 THEN 'PASS' ELSE 'WARN' END,
           'Pending requests: ' || count(*)::TEXT
    FROM steadytext_queue
    WHERE status IN ('pending', 'processing');

    -- Check daemon health
    RETURN QUERY
    SELECT 'daemon_health'::TEXT,
           CASE WHEN count(*) > 0 THEN 'PASS' ELSE 'FAIL' END,
           'Healthy daemons: ' || count(*)::TEXT
    FROM steadytext_daemon_health
    WHERE status = 'healthy';
END;
$$ LANGUAGE plpgsql;

-- Migration wrapper with rollback support
CREATE OR REPLACE FUNCTION steadytext_migrate(target_version TEXT)
RETURNS void AS $$
DECLARE
    current_version TEXT;
    migration_start TIMESTAMPTZ;
    migration_id INT;
BEGIN
    -- Get current version
    SELECT version INTO current_version
    FROM steadytext_version
    ORDER BY installed_at DESC LIMIT 1;

    IF current_version >= target_version THEN
        RAISE NOTICE 'Already at version % or higher', target_version;
        RETURN;
    END IF;

    -- Start migration
    migration_start := NOW();

    BEGIN
        -- Run pre-checks
        IF EXISTS (
            SELECT 1 FROM steadytext_pre_migration_check(target_version)
            WHERE status = 'FAIL'
        ) THEN
            RAISE EXCEPTION 'Pre-migration checks failed';
        END IF;

        -- Execute migration script
        EXECUTE format('ALTER EXTENSION pg_steadytext UPDATE TO %L', target_version);

        -- Record successful migration
        INSERT INTO steadytext_migrations
        (version, migration_name, execution_time_ms)
        VALUES (
            target_version,
            format('%s -> %s', current_version, target_version),
            EXTRACT(EPOCH FROM (NOW() - migration_start)) * 1000
        );

        RAISE NOTICE 'Successfully migrated from % to %', current_version, target_version;

    EXCEPTION WHEN OTHERS THEN
        -- Record failed migration
        INSERT INTO steadytext_migrations
        (version, migration_name, success, error_message)
        VALUES (
            target_version,
            format('%s -> %s', current_version, target_version),
            FALSE,
            SQLERRM
        );

        -- Re-raise the exception
        RAISE;
    END;
END;
$$ LANGUAGE plpgsql;
```

### 5. **Data Migration Utilities**

```python
@pg
def migrate_cache_format(from_version: str, to_version: str):
    """Migrate cache entries between format versions"""

    if from_version == '1.0.0' and to_version == '1.1.0':
        # Add thinking_mode field to existing entries
        plpy.execute("""
            UPDATE steadytext_cache
            SET generation_params =
                CASE
                    WHEN generation_params IS NULL THEN '{"thinking_mode": false}'::jsonb
                    ELSE generation_params || '{"thinking_mode": false}'::jsonb
                END
            WHERE generation_params IS NULL
               OR NOT (generation_params ? 'thinking_mode')
        """)

    elif from_version == '1.1.0' and to_version == '1.2.0':
        # Example: Migrate to new embedding format
        plpy.execute("""
            -- Create temporary table with new format
            CREATE TEMP TABLE cache_migration AS
            SELECT
                id,
                cache_key,
                prompt,
                response,
                embedding,
                -- Transform data as needed
                jsonb_build_object(
                    'access_count', access_count,
                    'last_accessed', last_accessed,
                    'version', '1.2.0'
                ) as metadata
            FROM steadytext_cache;

            -- Apply migration
            UPDATE steadytext_cache c
            SET metadata = m.metadata
            FROM cache_migration m
            WHERE c.id = m.id;
        """)
```

### 6. **Rollback Strategy**

```sql
-- Rollback function
CREATE OR REPLACE FUNCTION steadytext_rollback(to_version TEXT)
RETURNS void AS $$
DECLARE
    rollback_script TEXT;
BEGIN
    -- Check if rollback script exists
    SELECT content INTO rollback_script
    FROM steadytext_rollback_scripts
    WHERE from_version = (SELECT version FROM steadytext_version ORDER BY installed_at DESC LIMIT 1)
      AND to_version = to_version;

    IF rollback_script IS NULL THEN
        RAISE EXCEPTION 'No rollback script found for version %', to_version;
    END IF;

    -- Execute rollback
    EXECUTE rollback_script;

    -- Update version
    UPDATE steadytext_version
    SET version = to_version,
        notes = 'Rolled back from ' || version
    WHERE version = (SELECT version FROM steadytext_version ORDER BY installed_at DESC LIMIT 1);

    RAISE NOTICE 'Successfully rolled back to version %', to_version;
END;
$$ LANGUAGE plpgsql;

-- Store rollback scripts
CREATE TABLE steadytext_rollback_scripts (
    from_version TEXT,
    to_version TEXT,
    content TEXT,
    PRIMARY KEY (from_version, to_version)
);
```

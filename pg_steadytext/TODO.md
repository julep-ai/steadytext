# pg_steadytext Implementation TODO

This file contains detailed implementation tasks for the pg_steadytext PostgreSQL extension.

## UPDATE: Async Functions Completed (v1.1.0)

AIDEV-NOTE: The following async functions have been implemented:
- ✅ `steadytext_generate_async` - Queue text generation
- ✅ `steadytext_embed_async` - Queue embedding generation  
- ✅ `steadytext_generate_json_async` - Queue JSON generation with schema
- ✅ `steadytext_generate_regex_async` - Queue regex-constrained generation
- ✅ `steadytext_generate_choice_async` - Queue choice-constrained generation
- ✅ `steadytext_generate_batch_async` - Queue multiple generation requests
- ✅ `steadytext_embed_batch_async` - Queue multiple embedding requests
- ✅ `steadytext_check_async` - Check async request status
- ✅ `steadytext_get_async_result` - Get result with timeout
- ✅ `steadytext_cancel_async` - Cancel pending request
- ✅ `steadytext_check_async_batch` - Check multiple request statuses
- ✅ Background worker (`python/worker.py`) processes all request types

## Phase 1: Basic Infrastructure Setup (Week 1)

### 1.1 Set up PostgreSQL Extensions [Complexity: Medium, Uncertainty: Low]

#### 1.1.1 Install plpython3u extension and verify Python 3.11+ support
```sql
-- Ubuntu/Debian installation
sudo apt-get install postgresql-plpython3-16  # Replace 16 with your PG version

-- Create extension (requires superuser)
CREATE EXTENSION plpython3u;

-- Test Python version
CREATE OR REPLACE FUNCTION check_python_version() 
RETURNS text AS $$
    import sys
    return f"Python {sys.version}"
$$ LANGUAGE plpython3u;

SELECT check_python_version();
```

#### 1.1.2 Install pgvector for embeddings
```sql
-- Installation
sudo apt install postgresql-16-pgvector
CREATE EXTENSION vector;

-- Verify with test table
CREATE TABLE test_embeddings (
    id SERIAL PRIMARY KEY,
    embedding vector(1024)  -- SteadyText uses 1024-dim embeddings
);
```

#### 1.1.3 Install Omnigres extensions (omni_python, omni_worker, omni_vfs)
```sql
-- For Docker setup
docker run --name omnigres \
  -e POSTGRES_PASSWORD=omnigres \
  -e POSTGRES_USER=omnigres \
  -e POSTGRES_DB=omnigres \
  -v $(pwd)/python:/python \
  -p 5432:5432 --rm ghcr.io/omnigres/omnigres-17:latest

-- Or install extensions
CREATE EXTENSION IF NOT EXISTS omni_python CASCADE;
CREATE EXTENSION IF NOT EXISTS omni_worker CASCADE;
CREATE EXTENSION IF NOT EXISTS omni_vfs CASCADE;
```

### 1.2 Create cache management system [Complexity: High, Uncertainty: Medium]

#### 1.2.1 Design FrecencyCache class matching SteadyText's implementation
```python
# python/cache_manager.py
class FrecencyCache:
    """
    Frecency = Frequency + Recency
    Score = access_count * (1 / (1 + time_since_last_access))
    """
    def calculate_frecency_score(self, access_count, last_accessed):
        time_weight = 1 / (1 + (current_time - last_accessed).total_seconds() / 3600)
        return access_count * time_weight
```

#### 1.2.2 Create cache synchronization between PostgreSQL and SQLite
```python
# AIDEV-NOTE: SteadyText uses SQLite cache at ~/.cache/steadytext/caches/
# Need to sync with PostgreSQL steadytext_cache table
import sqlite3
from pathlib import Path

STEADYTEXT_CACHE_PATH = Path.home() / ".cache" / "steadytext" / "caches" / "generation_cache.db"

def sync_from_sqlite():
    """Sync SteadyText's SQLite cache to PostgreSQL"""
    conn = sqlite3.connect(STEADYTEXT_CACHE_PATH)
    # Implementation here
```

### 1.3 Model loading infrastructure [Complexity: Medium, Uncertainty: Low]

#### 1.3.1 Implement singleton pattern for model instances
```python
# python/model_loader.py
class ModelSingleton:
    _instances = {}
    
    @classmethod
    def get_model(cls, model_type='generation'):
        if model_type not in cls._instances:
            # Load model using SteadyText's loader
            from steadytext.models.loader import load_model
            cls._instances[model_type] = load_model(model_type)
        return cls._instances[model_type]
```

### 1.4 Database schema creation [Complexity: Low, Uncertainty: Low]

#### 1.4.1 Create steadytext_cache table with proper indexes
```sql
CREATE TABLE steadytext_cache (
    id BIGSERIAL PRIMARY KEY,
    cache_key TEXT UNIQUE NOT NULL,
    prompt TEXT NOT NULL,
    generated_text TEXT,
    embedding vector(1024),  -- Using pgvector
    model_name TEXT NOT NULL DEFAULT 'qwen3-1.7b',
    max_tokens INT,
    temperature FLOAT DEFAULT 0.0,
    
    -- Frecency tracking
    access_count INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    frecency_score FLOAT GENERATED ALWAYS AS (
        access_count * (1.0 / (1.0 + EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 3600.0))
    ) STORED,
    
    -- Metadata
    cache_size_bytes INT,
    generation_time_ms INT
);

-- Indexes for performance
CREATE INDEX idx_cache_key ON steadytext_cache(cache_key);
CREATE INDEX idx_frecency ON steadytext_cache(frecency_score DESC);
CREATE INDEX idx_embedding ON steadytext_cache USING ivfflat (embedding vector_cosine_ops);
```

## Phase 2: Core Functions Implementation (Week 2)

### 2.1 Text generation functions [Complexity: High, Uncertainty: Medium]

#### 2.1.1 Implement steadytext_generate() synchronous function
```sql
CREATE OR REPLACE FUNCTION steadytext_generate(
    prompt TEXT,
    max_tokens INT DEFAULT 512,
    use_cache BOOLEAN DEFAULT TRUE,
    thinking_mode BOOLEAN DEFAULT FALSE
) RETURNS TEXT AS $$
    import json
    from steadytext import generate
    
    # Check cache first
    if use_cache:
        cache_key = plpy.execute(
            "SELECT cache_key FROM steadytext_cache WHERE prompt = $1 AND max_tokens = $2",
            [prompt, max_tokens]
        )
        if cache_key:
            return cache_key[0]['generated_text']
    
    # Generate with SteadyText
    result = generate(prompt, max_tokens=max_tokens, thinking_mode=thinking_mode)
    
    # Store in cache
    if use_cache:
        plpy.execute("""
            INSERT INTO steadytext_cache (cache_key, prompt, generated_text, max_tokens)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (cache_key) DO UPDATE
            SET access_count = steadytext_cache.access_count + 1,
                last_accessed = NOW()
        """, [cache_key, prompt, result, max_tokens])
    
    return result
$$ LANGUAGE plpython3u;
```

#### 2.1.2 Implement streaming generation (SETOF returns)
```sql
CREATE OR REPLACE FUNCTION steadytext_generate_stream(
    prompt TEXT,
    max_tokens INT DEFAULT 512
) RETURNS SETOF TEXT AS $$
    from steadytext import generate_iter
    
    for token in generate_iter(prompt, max_tokens=max_tokens):
        yield token
$$ LANGUAGE plpython3u;
```

### 2.2 Embedding functions [Complexity: Medium, Uncertainty: Low]

#### 2.2.1 Implement steadytext_embed() for single embeddings
```sql
CREATE OR REPLACE FUNCTION steadytext_embed(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT TRUE
) RETURNS vector AS $$
    from steadytext import embed
    import numpy as np
    
    # Generate embedding
    embedding = embed(text_input)  # Returns 1024-dim numpy array
    
    # Convert to PostgreSQL array format
    return embedding.tolist()
$$ LANGUAGE plpython3u;

-- Usage example
SELECT content, embedding <=> steadytext_embed('search query') AS distance
FROM steadytext_cache
WHERE embedding IS NOT NULL
ORDER BY distance
LIMIT 5;
```

### 2.3 SteadyText daemon integration [Complexity: High, Uncertainty: High]

#### 2.3.1 Implement ZeroMQ client connection to daemon
```python
# python/daemon_connector.py
import zmq
import json
from typing import Optional

class SteadyTextDaemonClient:
    def __init__(self, host='localhost', port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        request = {
            'type': 'generate',
            'prompt': prompt,
            'max_tokens': max_tokens,
            'seed': 42  # Deterministic
        }
        self.socket.send_json(request)
        response = self.socket.recv_json()
        return response['text']
```

#### 2.3.2 Handle daemon lifecycle (start/stop/restart)
```sql
CREATE OR REPLACE FUNCTION steadytext_daemon_start() RETURNS VOID AS $$
    import subprocess
    import os
    
    # Check if daemon is already running
    try:
        subprocess.run(['st', 'daemon', 'status'], check=True, capture_output=True)
        plpy.notice("Daemon already running")
    except:
        # Start daemon
        subprocess.run(['st', 'daemon', 'start'], check=True)
        plpy.notice("Daemon started successfully")
$$ LANGUAGE plpython3u;
```

## Phase 3: Background Worker & Async Processing (Week 3) ✅ COMPLETED

### 3.1 Queue system implementation [Complexity: High, Uncertainty: Medium] ✅ COMPLETED

#### 3.1.1 Design queue schema with priority and status tracking ✅ COMPLETED
```sql
CREATE TABLE steadytext_queue (
    id BIGSERIAL PRIMARY KEY,
    request_type TEXT NOT NULL CHECK (request_type IN ('generate', 'embed', 'batch')),
    request_data JSONB NOT NULL,
    priority INT DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    result_data JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    user_id TEXT,
    session_id TEXT
);

-- Indexes for queue processing
CREATE INDEX idx_queue_status_priority ON steadytext_queue(status, priority DESC, created_at);
CREATE INDEX idx_queue_user ON steadytext_queue(user_id, created_at DESC);
```

#### 3.1.2 Implement async generation function ✅ COMPLETED
```sql
CREATE OR REPLACE FUNCTION steadytext_generate_async(
    prompt TEXT,
    max_tokens INT DEFAULT 512,
    priority INT DEFAULT 5,
    user_id TEXT DEFAULT NULL
) RETURNS BIGINT AS $$
DECLARE
    queue_id BIGINT;
BEGIN
    INSERT INTO steadytext_queue (request_type, request_data, priority, user_id)
    VALUES ('generate', jsonb_build_object(
        'prompt', prompt,
        'max_tokens', max_tokens
    ), priority, user_id)
    RETURNING id INTO queue_id;
    
    RETURN queue_id;
END;
$$ LANGUAGE plpgsql;
```

### 3.2 Background worker process with omni_worker [Complexity: High, Uncertainty: High] ✅ COMPLETED (without omni_worker)

#### 3.2.1 Create worker using omni_worker framework ✅ COMPLETED
AIDEV-NOTE: Implemented as standalone Python worker (worker.py) that polls the queue table using psycopg2
```python
# python/worker.py
# AIDEV-NOTE: omni_worker provides background job processing
import time
from steadytext import generate, embed

def process_queue_item(item):
    """Process a single queue item"""
    if item['request_type'] == 'generate':
        result = generate(
            item['request_data']['prompt'],
            max_tokens=item['request_data'].get('max_tokens', 512)
        )
        return {'text': result}
    elif item['request_type'] == 'embed':
        result = embed(item['request_data']['text'])
        return {'embedding': result.tolist()}
```

## Phase 4: Production Features (Week 4)

### 4.1 pgvector integration (instead of FAISS) [Complexity: Medium, Uncertainty: Low]

#### 4.1.1 Create semantic search functions using pgvector
```sql
-- Function to find similar texts
CREATE OR REPLACE FUNCTION steadytext_semantic_search(
    query_text TEXT,
    limit_results INT DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.7
) RETURNS TABLE (
    content TEXT,
    similarity FLOAT,
    generated_text TEXT
) AS $$
DECLARE
    query_embedding vector;
BEGIN
    -- Generate embedding for query
    query_embedding := steadytext_embed(query_text);
    
    -- Search using pgvector
    RETURN QUERY
    SELECT 
        c.prompt as content,
        1 - (c.embedding <=> query_embedding) as similarity,
        c.generated_text
    FROM steadytext_cache c
    WHERE c.embedding IS NOT NULL
    AND 1 - (c.embedding <=> query_embedding) > similarity_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;

-- Create appropriate index
CREATE INDEX idx_embedding_ivfflat 
ON steadytext_cache USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- Adjust lists based on data size
```

### 4.2 Security hardening [Complexity: Medium, Uncertainty: Low]

#### 4.2.1 Implement comprehensive input validation
```python
# python/security.py
import re
from typing import Optional

class InputValidator:
    MAX_PROMPT_LENGTH = 10000
    MAX_TOKENS_LIMIT = 4096
    
    @staticmethod
    def validate_prompt(prompt: str) -> tuple[bool, Optional[str]]:
        if not prompt or not prompt.strip():
            return False, "Prompt cannot be empty"
        if len(prompt) > InputValidator.MAX_PROMPT_LENGTH:
            return False, f"Prompt exceeds maximum length of {InputValidator.MAX_PROMPT_LENGTH}"
        # Check for potential injection patterns
        if re.search(r'<script|javascript:|onerror=', prompt, re.IGNORECASE):
            return False, "Prompt contains potentially malicious content"
        return True, None
```

#### 4.2.2 Add rate limiting per user/session
```sql
CREATE TABLE steadytext_rate_limits (
    user_id TEXT PRIMARY KEY,
    requests_per_minute INT DEFAULT 10,
    requests_per_hour INT DEFAULT 100,
    current_minute_count INT DEFAULT 0,
    current_hour_count INT DEFAULT 0,
    last_minute_reset TIMESTAMPTZ DEFAULT NOW(),
    last_hour_reset TIMESTAMPTZ DEFAULT NOW()
);

-- Rate limiting function
CREATE OR REPLACE FUNCTION check_rate_limit(p_user_id TEXT) RETURNS BOOLEAN AS $$
-- Implementation here
$$ LANGUAGE plpgsql;
```

## Phase 5: Testing & Quality Assurance (Throughout)

### 5.1 Unit testing [Complexity: Low, Uncertainty: Low] ✅ COMPLETED

#### 5.1.1 Set up pgTAP testing framework ✅ COMPLETED
AIDEV-NOTE: pgTAP has been integrated as of v1.0.3
- Created comprehensive test suite in test/pgtap/
- Added run_pgtap_tests.sh script for test execution
- Updated Makefile with test-pgtap targets
- Added CI workflow for automated testing
- Converted existing tests to pgTAP format

```sql
-- pgTAP is now installed via Dockerfile and docker-entrypoint.sh
CREATE EXTENSION IF NOT EXISTS pgtap CASCADE;

-- Test suite includes:
-- 00_setup.sql - pgTAP verification
-- 01_basic.sql - Core functionality
-- 02_embeddings.sql - Vector operations
-- 03_async.sql - Queue operations  
-- 04_structured.sql - JSON/regex/choice
-- 05_cache_daemon.sql - Cache and daemon

-- Run tests with:
-- make test-pgtap
-- make test-pgtap-verbose
-- make test-pgtap-tap (for CI)
```

## Phase 6: Documentation & Deployment

### 6.1 Documentation [Complexity: Low, Uncertainty: Low]

#### 6.1.1 Create comprehensive README with examples
- Installation guide with all dependencies
- Basic usage examples
- Configuration options
- Troubleshooting guide

### 6.2 PGXN Package Creation [Complexity: Medium, Uncertainty: Low]

#### 6.2.1 Update META.json for PGXN distribution
- Already created in initial setup
- Need to test with pgxn client before publishing

---

## Later Stage Items (Can be deferred)

The following items are marked for future implementation:

### Cloud-Specific Features
- [ ] AWS RDS compatibility testing
- [ ] Google Cloud SQL integration
- [ ] Azure Database for PostgreSQL support

### Platform-Specific Packaging
- [ ] Homebrew formula for macOS
- [ ] APT/YUM repository setup
- [ ] Windows installer

### Advanced Monitoring
- [ ] Prometheus metrics exporter
- [ ] OpenTelemetry tracing
- [ ] Grafana dashboard templates

### Performance Optimization (Secondary Goal)
- [ ] Query optimization and profiling
- [ ] Connection pooling improvements
- [ ] Prepared statement caching
- [ ] Result caching strategies

---

## Notes

- **AIDEV-NOTE**: Performance optimization is a secondary goal - focus on correctness and functionality first
- **AIDEV-NOTE**: Use pgvector instead of FAISS for simplicity and better PostgreSQL integration
- **AIDEV-NOTE**: Leverage existing SteadyText daemon architecture rather than reimplementing
- **AIDEV-TODO**: Add integration tests for daemon failover scenarios
- **AIDEV-TODO**: Implement cache size limits and eviction policies
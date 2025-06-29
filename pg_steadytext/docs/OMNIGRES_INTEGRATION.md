# Omnigres Integration Guide for pg_steadytext

## Overview

While pg_steadytext works with standard PostgreSQL using `plpython3u`, it can leverage Omnigres extensions for enhanced functionality:

- **omni_python**: Advanced Python integration with better package management
- **omni_worker**: Background job processing for async operations
- **omni_vfs**: Virtual file system for secure cache access

## Installation

### Option 1: Docker (Recommended)

```bash
# Use the omnigres service from docker-compose
docker-compose up -d omnigres

# Connect to Omnigres
docker-compose exec omnigres psql -U omnigres -d omnigres_dev
```

### Option 2: Manual Installation

```sql
-- Install Omnigres extensions
CREATE EXTENSION IF NOT EXISTS omni_python CASCADE;
CREATE EXTENSION IF NOT EXISTS omni_worker CASCADE;
CREATE EXTENSION IF NOT EXISTS omni_vfs CASCADE;

-- Then install pg_steadytext
CREATE EXTENSION pg_steadytext CASCADE;
```

## Using omni_python

### Enhanced Python Environment

```sql
-- Configure Python packages with omni_python
SELECT omni_python.install_package('steadytext');
SELECT omni_python.install_package('pyzmq');
SELECT omni_python.install_package('numpy');

-- Create functions with better error handling
CREATE OR REPLACE FUNCTION steadytext_generate_omnigres(
    prompt TEXT,
    max_tokens INT DEFAULT 512
)
RETURNS TEXT
LANGUAGE plpython3u
AS $$
import omni_python
from steadytext import generate

# omni_python provides better module isolation
with omni_python.context():
    return generate(prompt, max_tokens=max_tokens)
$$;
```

### Package Management

```sql
-- List installed packages
SELECT * FROM omni_python.packages;

-- Update packages
SELECT omni_python.update_package('steadytext');
```

## Using omni_worker

### Background Processing

```python
# python/worker_omnigres.py
"""
Worker implementation using omni_worker
AIDEV-NOTE: This provides better job management than custom implementation
"""

import omni_worker
from steadytext import generate, embed

@omni_worker.task
def process_generation(prompt: str, max_tokens: int = 512) -> str:
    """Process text generation in background"""
    return generate(prompt, max_tokens=max_tokens)

@omni_worker.task
def process_embedding(text: str) -> list[float]:
    """Process embedding generation in background"""
    embedding = embed(text)
    return embedding.tolist()

# Register tasks
omni_worker.register(process_generation)
omni_worker.register(process_embedding)
```

### Create Worker Tables

```sql
-- Create worker-backed async functions
CREATE OR REPLACE FUNCTION steadytext_generate_async_omnigres(
    prompt TEXT,
    max_tokens INT DEFAULT 512
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    job_id UUID;
BEGIN
    -- Use omni_worker to queue the job
    SELECT omni_worker.queue_job(
        'process_generation',
        jsonb_build_object(
            'prompt', prompt,
            'max_tokens', max_tokens
        )
    ) INTO job_id;
    
    RETURN job_id;
END;
$$;

-- Check job status
CREATE OR REPLACE FUNCTION steadytext_check_job_omnigres(job_id UUID)
RETURNS TABLE(
    status TEXT,
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
)
LANGUAGE sql
AS $$
    SELECT 
        status,
        result,
        error_message,
        created_at,
        completed_at
    FROM omni_worker.jobs
    WHERE id = job_id;
$$;
```

## Using omni_vfs

### Secure Cache Access

```sql
-- Mount SteadyText cache directory
SELECT omni_vfs.mount(
    'steadytext_cache',
    'file://' || current_setting('home') || '/.cache/steadytext'
);

-- Read cache files securely
CREATE OR REPLACE FUNCTION read_steadytext_cache_file(filename TEXT)
RETURNS TEXT
LANGUAGE sql
AS $$
    SELECT omni_vfs.read_text('steadytext_cache/' || filename);
$$;

-- List cache contents
CREATE OR REPLACE FUNCTION list_steadytext_cache()
RETURNS TABLE(name TEXT, size BIGINT, modified TIMESTAMPTZ)
LANGUAGE sql
AS $$
    SELECT name, size, modified 
    FROM omni_vfs.list('steadytext_cache/');
$$;
```

### Cache Synchronization

```python
# python/cache_sync_omnigres.py
"""
Cache synchronization using omni_vfs
AIDEV-NOTE: More secure than direct file access
"""

import omni_vfs
import sqlite3
import json

def sync_cache_from_vfs():
    """Sync SteadyText SQLite cache using VFS"""
    # Read cache database through VFS
    cache_data = omni_vfs.read_binary('steadytext_cache/caches/generation_cache.db')
    
    # Process in memory
    conn = sqlite3.connect(':memory:')
    conn.executescript(cache_data)
    
    # Extract entries
    cursor = conn.execute('SELECT key, value FROM cache')
    for key, value in cursor:
        # Sync to PostgreSQL
        plpy.execute("""
            INSERT INTO steadytext_cache (cache_key, response)
            VALUES ($1, $2)
            ON CONFLICT (cache_key) DO UPDATE
            SET response = $2, last_accessed = NOW()
        """, [key, value])
```

## Migration from plpython3u

To migrate existing pg_steadytext installation to use Omnigres:

1. **Install Omnigres extensions** alongside existing setup
2. **Update Python functions** to use `omni_python.context()`
3. **Migrate async operations** to `omni_worker`
4. **Replace file operations** with `omni_vfs`

Example migration:

```sql
-- Before (plpython3u)
CREATE FUNCTION example() RETURNS TEXT
LANGUAGE plpython3u AS $$
import steadytext
return steadytext.generate("Hello")
$$;

-- After (with omni_python)
CREATE FUNCTION example() RETURNS TEXT
LANGUAGE plpython3u AS $$
import omni_python
with omni_python.context():
    import steadytext
    return steadytext.generate("Hello")
$$;
```

## Benefits of Omnigres

1. **Better Python Package Management**: Isolated environments, easier updates
2. **Robust Background Processing**: Built-in job queue with monitoring
3. **Secure File Access**: VFS prevents direct filesystem access
4. **Enhanced Monitoring**: Better visibility into Python execution
5. **Cloud-Ready**: Designed for containerized environments

## Performance Considerations

- omni_python adds slight overhead (~1-2ms per call)
- omni_worker is more efficient than custom polling
- omni_vfs may be slower than direct file access but more secure

## Troubleshooting

### Check Omnigres Installation
```sql
SELECT extname, extversion 
FROM pg_extension 
WHERE extname LIKE 'omni_%';
```

### Debug Python Environment
```sql
SELECT omni_python.version();
SELECT omni_python.path();
```

### Monitor Worker Jobs
```sql
SELECT * FROM omni_worker.jobs 
WHERE created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC;
```

---

**AIDEV-NOTE**: Omnigres integration is optional but recommended for production deployments requiring advanced Python integration and background processing capabilities.
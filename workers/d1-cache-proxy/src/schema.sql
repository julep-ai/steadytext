-- Schema for SteadyText cache tables in D1
-- Each cache instance gets its own table with a prefix

-- Example table structure (created dynamically with cache_name prefix)
-- CREATE TABLE IF NOT EXISTS {cache_name}_cache (
--     key TEXT PRIMARY KEY,
--     value TEXT NOT NULL,
--     frequency INTEGER DEFAULT 1,
--     last_access INTEGER NOT NULL,
--     size INTEGER NOT NULL,
--     created_at INTEGER NOT NULL
-- );

-- Index for frecency-based eviction
-- CREATE INDEX IF NOT EXISTS idx_{cache_name}_frecency 
-- ON {cache_name}_cache(frequency DESC, last_access DESC);

-- Metadata table for tracking cache configurations
CREATE TABLE IF NOT EXISTS cache_metadata (
    cache_name TEXT PRIMARY KEY,
    max_size_bytes INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    last_updated INTEGER NOT NULL
);
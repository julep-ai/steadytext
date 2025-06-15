# SQLite-based Concurrent Disk-Backed Frecency Cache

## Overview
Replace the current pickle-based disk cache with SQLite to enable safe concurrent access from multiple processes while maintaining the same API and frecency eviction algorithm.

## Motivation
Current implementation issues:
- File-level race conditions with pickle files
- No cross-process synchronization
- Potential data corruption with concurrent writes
- Single shared cache file for all processes

## Design Goals
1. **API Compatibility**: Keep the exact same public interface
2. **Zero New Dependencies**: Use Python's built-in `sqlite3`
3. **Performance**: Maintain or improve current performance
4. **Concurrency**: Support multiple readers/writers safely
5. **Cross-Platform**: Work identically on Linux, macOS, Windows

## Technical Design

### Database Schema
```sql
-- Main cache table
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL,           -- Pickled Python object
    frequency INTEGER DEFAULT 1,    -- Access frequency
    last_access INTEGER NOT NULL,   -- Unix timestamp * 1000000 for microsecond precision
    size INTEGER NOT NULL          -- Size of pickled value in bytes
);

-- Index for efficient frecency-based eviction
CREATE INDEX IF NOT EXISTS idx_frecency ON cache(frequency ASC, last_access ASC);

-- Metadata table for cache configuration
CREATE TABLE IF NOT EXISTS metadata (
    property TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

### Key Implementation Details

#### 1. Connection Management
```python
# Use WAL mode for better concurrency
connection.execute("PRAGMA journal_mode=WAL")
# Ensure durability
connection.execute("PRAGMA synchronous=NORMAL")
# Set reasonable timeout for concurrent access
connection.execute("PRAGMA busy_timeout=5000")  # 5 seconds
```

#### 2. Atomic Operations
- All cache operations wrapped in transactions
- Use `INSERT OR REPLACE` for set operations
- Prepared statements for performance

#### 3. Size Management
```python
def _evict_until_size_ok(self):
    """Evict entries until total cache size is under limit."""
    with self._get_connection() as conn:
        # Get current total size
        total_size = conn.execute("SELECT SUM(size) FROM cache").fetchone()[0]
        
        if total_size > self.max_size_bytes:
            # Calculate target (80% of limit)
            target_size = int(self.max_size_bytes * 0.8)
            
            # Evict entries with lowest frecency score
            conn.execute("""
                DELETE FROM cache 
                WHERE key IN (
                    SELECT key FROM cache 
                    ORDER BY frequency ASC, last_access ASC
                    LIMIT (
                        SELECT COUNT(*) FROM cache 
                        WHERE (
                            SELECT SUM(size) FROM cache c2 
                            WHERE c2.frequency < cache.frequency 
                               OR (c2.frequency = cache.frequency 
                                   AND c2.last_access <= cache.last_access)
                        ) > ?
                    )
                )
            """, (self.max_size_bytes - target_size,))
```

#### 4. Migration Path
- Check for existing pickle cache file on initialization
- One-time migration to SQLite format
- Delete pickle file after successful migration

## Implementation Plan

### Phase 1: Core Implementation
1. Create new `SQLiteDiskBackedFrecencyCache` class
2. Implement all base methods (get, set, clear)
3. Add connection pooling and error handling
4. Implement size-based eviction

### Phase 2: Migration
1. Add migration logic from pickle to SQLite
2. Handle version differences
3. Ensure backward compatibility

### Phase 3: Testing
1. Unit tests for all operations
2. Concurrent access tests
3. Performance benchmarks
4. Migration tests

### Phase 4: Integration
1. Update `DiskBackedFrecencyCache` to use SQLite backend
2. Update environment variable handling
3. Update documentation

## File Structure
```
steadytext/
├── disk_backed_frecency_cache.py  # Keep as wrapper
├── sqlite_cache_backend.py        # New SQLite implementation
└── tests/
    ├── test_disk_backed_frecency_cache.py  # Existing tests
    └── test_concurrent_cache.py            # New concurrent tests
```

## Testing Strategy

### Concurrent Access Tests
```python
def test_concurrent_writes():
    """Test multiple processes writing simultaneously."""
    # Use multiprocessing.Pool
    # Each process writes unique keys
    # Verify all writes succeed
    
def test_concurrent_read_write():
    """Test reading while writing from different processes."""
    # One process continuously writes
    # Multiple processes read
    # Verify consistency
    
def test_eviction_under_concurrent_load():
    """Test size-based eviction with concurrent access."""
    # Multiple processes fill cache beyond limit
    # Verify proper eviction
    # Check total size constraint
```

### Performance Tests
- Compare pickle vs SQLite performance
- Measure overhead of concurrent access
- Test with various cache sizes

## Configuration

### Environment Variables (unchanged)
- `STEADYTEXT_GENERATION_CACHE_CAPACITY`: Max entries (default: 256)
- `STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB`: Max size in MB (default: 50.0)
- `STEADYTEXT_EMBEDDING_CACHE_CAPACITY`: Max entries (default: 512)
- `STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB`: Max size in MB (default: 100.0)

### New Internal Settings
- WAL checkpoint interval
- Connection pool size
- Busy timeout duration

## Rollback Plan
1. Keep pickle implementation available
2. Feature flag for SQLite vs pickle backend
3. Automatic fallback on SQLite errors

## Success Criteria
1. All existing tests pass
2. No API changes required
3. Concurrent access works reliably
4. Performance within 10% of current implementation
5. Zero data corruption under stress testing

## Timeline
- Phase 1: 2-3 hours (core implementation)
- Phase 2: 1 hour (migration logic)
- Phase 3: 2-3 hours (comprehensive testing)
- Phase 4: 1 hour (integration and cleanup)

Total: ~7-8 hours of implementation work

## Future Enhancements (out of scope)
- Distributed caching across machines
- Cache statistics and monitoring
- TTL-based expiration
- Compression for large values
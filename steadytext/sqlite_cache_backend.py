# AIDEV-NOTE: An SQLite-based, concurrent, disk-backed frecency cache that uses WAL mode for optimal performance and has a __len__ method for compatibility with the cache manager.
from __future__ import annotations

import pickle
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .frecency_cache import FrecencyCache
    from .utils import get_cache_dir, logger
except ImportError:
    # For direct testing - these would fail with mypy but work when running directly
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from frecency_cache import FrecencyCache  # type: ignore[import-not-found,no-redef]
    from utils import get_cache_dir, logger  # type: ignore[import-not-found,no-redef]


class SQLiteDiskBackedFrecencyCache(FrecencyCache):
    """SQLite-based disk-backed frecency cache with concurrent access support.

    Features:
    - Thread-safe and process-safe operations using SQLite
    - WAL mode for optimal concurrent read/write performance
    - Automatic migration from pickle-based cache files
    - Size-based eviction using frecency algorithm
    - Atomic operations with proper error handling
    """

    def __init__(
        self,
        capacity: int = 128,
        cache_name: str = "frecency_cache",
        max_size_mb: float = 100.0,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize SQLite disk-backed frecency cache.

        Args:
            capacity: Maximum number of entries (for compatibility, not enforced)
            cache_name: Name for the cache database (without extension)
            max_size_mb: Maximum cache file size in megabytes
            cache_dir: Directory for cache file (defaults to steadytext cache dir)
        """
        # AIDEV-NOTE: Don't call super().__init__ - we manage our own state
        self.capacity = capacity
        self.cache_name = cache_name
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)

        # AIDEV-NOTE: Use the existing cache directory structure
        if cache_dir is None:
            self.cache_dir = get_cache_dir().parent / "caches"
        else:
            self.cache_dir = cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_file = self.cache_dir / f"{cache_name}.db"
        self.pickle_file = self.cache_dir / f"{cache_name}.pkl"  # For migration

        # AIDEV-NOTE: Thread-local storage for database connections
        self._local = threading.local()
        self._db_lock = threading.Lock()

        # Initialize database and migrate if needed
        try:
            self._init_database()
            self._migrate_from_pickle()
        except sqlite3.DatabaseError as e:
            if "file is not a database" in str(e) and self.db_file.exists():
                # Handle corrupted database on initialization
                backup_file = self.db_file.with_suffix(f".corrupted.{int(time.time())}")
                try:
                    self.db_file.rename(backup_file)
                    logger.info(f"Moved corrupted database to {backup_file}")
                    # Force reconnection
                    self._local.connection = None
                    # Try again with fresh database
                    self._init_database()
                    self._migrate_from_pickle()
                except Exception as rename_error:
                    logger.error(
                        f"Failed to handle corrupted database: {rename_error}",
                        exc_info=True,
                    )
                    raise
            else:
                raise

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection with proper configuration."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    conn = sqlite3.connect(
                        str(self.db_file),
                        timeout=10.0,  # Increased timeout for concurrent access
                        check_same_thread=False,
                        isolation_level="DEFERRED",  # Use explicit transactions for control
                    )

                    # AIDEV-NOTE: WAL (Write-Ahead Logging) mode is crucial for concurrent performance, as it allows readers to continue reading while a writer is modifying the database.
                    conn.execute("PRAGMA journal_mode=WAL")
                    # Use NORMAL synchronous mode for better performance in multi-thread scenarios
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute(
                        "PRAGMA busy_timeout=30000"
                    )  # 30 seconds for heavy concurrent loads
                    conn.execute("PRAGMA cache_size=10000")  # 10MB page cache
                    # Use smaller autocheckpoint for better concurrency
                    conn.execute(
                        "PRAGMA wal_autocheckpoint=100"
                    )  # Checkpoint every 100 pages
                    # Enable read_uncommitted for better read concurrency
                    conn.execute("PRAGMA read_uncommitted=1")

                    # Test the connection
                    conn.execute("SELECT 1")

                    self._local.connection = conn
                    break

                except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
                    if "database disk image is malformed" in str(
                        e
                    ) or "database is locked" in str(e):
                        if attempt < max_retries - 1:
                            # Wait before retry
                            time.sleep(0.1 * (attempt + 1))
                            continue

                    # AIDEV-NOTE: Handle corrupted database files
                    logger.warning(f"Database error on attempt {attempt + 1}: {e}")
                    if self.db_file.exists() and "malformed" in str(e):
                        # Move corrupted file out of the way
                        backup_file = self.db_file.with_suffix(
                            f".corrupted.{int(time.time())}"
                        )
                        try:
                            self.db_file.rename(backup_file)
                            logger.info(f"Moved corrupted database to {backup_file}")
                        except Exception:
                            # Another process might have already moved it
                            pass

                    if attempt == max_retries - 1:
                        # Final attempt: create new connection with fresh database
                        conn = sqlite3.connect(
                            str(self.db_file),
                            timeout=10.0,
                            check_same_thread=False,
                            isolation_level="DEFERRED",
                        )

                        # Configure new connection
                        conn.execute("PRAGMA journal_mode=WAL")
                        conn.execute("PRAGMA synchronous=FULL")
                        conn.execute("PRAGMA busy_timeout=10000")
                        conn.execute("PRAGMA cache_size=10000")
                        conn.execute("PRAGMA wal_autocheckpoint=1000")

                        self._local.connection = conn
                        # Reinitialize database schema
                        self._init_database()

        return self._local.connection

    @contextmanager
    def _transaction(self, max_retries=3):
        """Context manager for database transactions with retry logic."""
        for attempt in range(max_retries):
            conn = self._get_connection()
            try:
                # AIDEV-NOTE: Use BEGIN IMMEDIATE for better concurrent handling
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
                return  # Success, exit retry loop

            except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    # Rollback might fail if connection is broken
                    pass

                # Check if it's a transient error that should be retried
                if (
                    "database is locked" in str(e) or "disk I/O error" in str(e)
                ) and attempt < max_retries - 1:
                    # Wait with exponential backoff before retry
                    wait_time = 0.1 * (2**attempt)
                    time.sleep(wait_time)
                    # Force reconnection for fresh state
                    self._local.connection = None
                    continue

                # Non-retryable error or max retries reached
                if "database disk image is malformed" in str(e):
                    # Force reconnection on next access
                    self._local.connection = None

                logger.error(
                    f"Database transaction failed after {attempt + 1} attempts: {e}",
                    exc_info=True,
                )
                raise

    def _init_database(self) -> None:
        """Initialize SQLite database with required schema."""
        conn = self._get_connection()

        # AIDEV-NOTE: Create main cache table with frecency metadata
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_access INTEGER NOT NULL,
                size INTEGER NOT NULL
            )
        """
        )

        # AIDEV-NOTE: Index for efficient frecency-based eviction
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_frecency 
            ON cache(frequency ASC, last_access ASC)
        """
        )

        # AIDEV-NOTE: Metadata table for cache configuration
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                property TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """
        )

        conn.commit()

        # Store cache configuration
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (property, value) VALUES (?, ?)",
                ("max_size_bytes", str(self.max_size_bytes)),
            )

    def _migrate_from_pickle(self) -> None:
        """One-time migration from pickle-based cache to SQLite."""
        if not self.pickle_file.exists():
            return

        logger.info(f"Migrating pickle cache from {self.pickle_file} to SQLite")

        try:
            # Load pickle data
            with self.pickle_file.open("rb") as f:
                data = pickle.load(f)

            if not isinstance(data, dict) or "data" not in data or "meta" not in data:
                logger.warning("Invalid pickle cache format, skipping migration")
                return

            # Migrate to SQLite
            current_time = int(time.time() * 1000000)  # Microsecond precision

            with self._transaction() as conn:
                for key, value in data["data"].items():
                    if key in data["meta"]:
                        frequency, last_access = data["meta"][key]
                        # Convert old counter to timestamp-like value
                        adjusted_access = current_time - (10000000 - last_access)
                    else:
                        frequency = 1
                        adjusted_access = current_time

                    pickled_value = pickle.dumps(
                        value, protocol=pickle.HIGHEST_PROTOCOL
                    )

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cache 
                        (key, value, frequency, last_access, size)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            self._serialize_key(key),
                            pickled_value,
                            frequency,
                            adjusted_access,
                            len(pickled_value),
                        ),
                    )

            # Remove pickle file after successful migration
            self.pickle_file.unlink()
            logger.info(f"Successfully migrated {len(data['data'])} entries to SQLite")

        except Exception as e:
            logger.error(f"Failed to migrate pickle cache: {e}")

    def _get_current_time_us(self) -> int:
        """Get current time in microseconds for precise ordering."""
        return int(time.time() * 1000000)

    def _serialize_key(self, key: Any) -> str:
        """Serialize cache key to string, handling special characters and complex types."""
        # AIDEV-NOTE: Robust key serialization is necessary because cache keys can be complex, and SQLite requires a simple string as a primary key.
        if isinstance(key, str):
            # For string keys, use base64 encoding to handle special characters safely
            import base64

            return base64.b64encode(key.encode("utf-8")).decode("ascii")
        elif isinstance(key, (tuple, list)):
            # For tuple/list keys, serialize each element and join
            import base64

            serialized_parts = []
            for part in key:
                if isinstance(part, str):
                    encoded = base64.b64encode(part.encode("utf-8")).decode("ascii")
                else:
                    encoded = base64.b64encode(str(part).encode("utf-8")).decode(
                        "ascii"
                    )
                serialized_parts.append(encoded)
            return "::".join(serialized_parts)
        else:
            # For other types, convert to string and encode
            import base64

            return base64.b64encode(str(key).encode("utf-8")).decode("ascii")

    def _force_checkpoint(self) -> None:
        """Force WAL checkpoint to ensure data is written to main database file."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = self._get_connection()
                # Try TRUNCATE first for maximum durability
                result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if result and result[0] == 0:  # 0 means success
                    return
                # Fall back to FULL if TRUNCATE didn't work
                result = conn.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
                if result and result[0] == 0:
                    return
                # Final fallback to PASSIVE
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                return
            except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    logger.warning(f"Failed to checkpoint WAL: {e}")
                    break

    def _get_total_size(self) -> int:
        """Get total size of all cached values in bytes."""
        conn = self._get_connection()
        result = conn.execute("SELECT COALESCE(SUM(size), 0) FROM cache").fetchone()
        return result[0] if result else 0

    def _evict_until_size_ok(self) -> None:
        """Evict entries until total cache size is under limit."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._transaction() as conn:
                    # Get current size within the transaction to avoid race conditions
                    result = conn.execute(
                        "SELECT COALESCE(SUM(size), 0) FROM cache"
                    ).fetchone()
                    total_size = result[0] if result else 0

                    if total_size <= self.max_size_bytes:
                        return

                    # AIDEV-NOTE: Target 80% of max size to avoid frequent evictions
                    target_size = int(self.max_size_bytes * 0.8)
                    size_to_remove = total_size - target_size

                    # AIDEV-NOTE: Improved frecency calculation that combines frequency and recency
                    # Calculate score as frequency / time_since_last_access (higher = keep)
                    current_time = self._get_current_time_us()
                    cursor = conn.execute(
                        """
                        SELECT key, size, 
                               frequency * 1.0 / (1 + (? - last_access) / 1000000.0) as frecency_score
                        FROM cache 
                        ORDER BY frecency_score ASC, size DESC
                    """,
                        (current_time,),
                    )

                    removed_size = 0
                    keys_to_remove = []

                    for key, size, score in cursor:
                        keys_to_remove.append(key)
                        removed_size += size
                        if removed_size >= size_to_remove:
                            break

                    # Remove selected entries in the same transaction
                    if keys_to_remove:
                        placeholders = ",".join("?" * len(keys_to_remove))
                        conn.execute(
                            f"DELETE FROM cache WHERE key IN ({placeholders})",
                            keys_to_remove,
                        )

                        logger.info(
                            f"Evicted {len(keys_to_remove)} entries to free {removed_size} bytes"
                        )

                # Force checkpoint after eviction to ensure visibility
                self._force_checkpoint()

                return  # Success, exit retry loop

            except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(
                        f"Database locked during eviction, retrying... (attempt {attempt + 1})"
                    )
                    time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    logger.error(f"Failed to evict cache entries: {e}")
                    break

    def get(self, key: Any) -> Any | None:
        """Get value from cache, updating frecency metadata."""
        try:
            # AIDEV-NOTE: Get current entry and update frecency in single transaction
            with self._transaction() as conn:
                result = conn.execute(
                    "SELECT value, frequency FROM cache WHERE key = ?",
                    (self._serialize_key(key),),
                ).fetchone()

                if result is None:
                    return None

                pickled_value, frequency = result

                # Update frecency metadata
                new_frequency = frequency + 1
                new_access_time = self._get_current_time_us()

                conn.execute(
                    """
                    UPDATE cache 
                    SET frequency = ?, last_access = ?
                    WHERE key = ?
                """,
                    (new_frequency, new_access_time, self._serialize_key(key)),
                )

                # Deserialize value
                return pickle.loads(pickled_value)

        except Exception as e:
            logger.error(f"Failed to get cache entry for key {key}: {e}")
            return None

    def set(self, key: Any, value: Any) -> None:
        """Set value in cache with automatic eviction if needed."""
        try:
            pickled_value = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            value_size = len(pickled_value)
            current_time = self._get_current_time_us()

            with self._transaction() as conn:
                # Check if key already exists
                existing = conn.execute(
                    "SELECT frequency FROM cache WHERE key = ?",
                    (self._serialize_key(key),),
                ).fetchone()

                if existing:
                    # Update existing entry
                    new_frequency = existing[0] + 1
                    conn.execute(
                        """
                        UPDATE cache 
                        SET value = ?, frequency = ?, last_access = ?, size = ?
                        WHERE key = ?
                    """,
                        (
                            pickled_value,
                            new_frequency,
                            current_time,
                            value_size,
                            self._serialize_key(key),
                        ),
                    )
                else:
                    # Insert new entry
                    conn.execute(
                        """
                        INSERT INTO cache (key, value, frequency, last_access, size)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            self._serialize_key(key),
                            pickled_value,
                            1,
                            current_time,
                            value_size,
                        ),
                    )

            # Check if eviction is needed
            if self._get_total_size() > self.max_size_bytes:
                self._evict_until_size_ok()

            # AIDEV-NOTE: Force WAL checkpoint after write for better multi-process visibility
            # This ensures data is immediately visible to other processes/threads
            # Use TRUNCATE mode to ensure all data is written to main database file
            self._force_checkpoint()

        except Exception as e:
            logger.error(f"Failed to set cache entry for key {key}: {e}")

    def clear(self) -> None:
        """Clear all cache entries and reset database."""
        try:
            with self._transaction() as conn:
                conn.execute("DELETE FROM cache")

            logger.info("Cleared all cache entries")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def sync(self) -> None:
        """Explicitly sync cache to disk with full WAL checkpoint for multi-process safety."""
        try:
            # Use our centralized checkpoint method
            self._force_checkpoint()
            logger.debug("Synchronized cache to disk")
        except Exception as e:
            logger.error(f"Failed to sync cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring and debugging."""
        try:
            conn = self._get_connection()

            # Get basic stats
            stats = conn.execute(
                """
                SELECT 
                    COUNT(*) as entry_count,
                    COALESCE(SUM(size), 0) as total_size,
                    COALESCE(AVG(frequency), 0) as avg_frequency,
                    COALESCE(MAX(frequency), 0) as max_frequency
                FROM cache
            """
            ).fetchone()

            return {
                "entry_count": stats[0],
                "total_size_bytes": stats[1],
                "total_size_mb": stats[1] / (1024 * 1024),
                "avg_frequency": stats[2],
                "max_frequency": stats[3],
                "max_size_bytes": self.max_size_bytes,
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization": (
                    stats[1] / self.max_size_bytes if self.max_size_bytes > 0 else 0
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    def __len__(self) -> int:
        """Return number of entries in cache."""
        try:
            conn = self._get_connection()
            result = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get cache length: {e}")
            return 0

    def delete(self, key: Any) -> None:
        """Delete a single entry from the cache."""
        try:
            with self._transaction() as conn:
                conn.execute(
                    "DELETE FROM cache WHERE key = ?", (self._serialize_key(key),)
                )
        except Exception as e:
            logger.error(f"Failed to delete cache entry for key {key}: {e}")

    def __del__(self):
        """Clean up database connections with proper shutdown sequence."""
        try:
            if (
                hasattr(self, "_local")
                and hasattr(self._local, "connection")
                and self._local.connection
            ):
                try:
                    # Ensure any pending writes are checkpointed
                    self._local.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    pass

                try:
                    self._local.connection.close()
                except Exception:
                    pass
        except Exception:
            pass

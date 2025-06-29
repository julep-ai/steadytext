# AIDEV-SECTION: CACHE_MANAGER
"""
cache_manager.py - Cache synchronization between SteadyText and PostgreSQL

AIDEV-NOTE: This module manages the cache synchronization between SteadyText's
SQLite-based frecency cache and PostgreSQL's cache table. It ensures consistency
and provides PostgreSQL-specific cache management features.
"""

import hashlib
import json
import logging
from typing import Optional, Dict, Any, cast
import numpy as np

# AIDEV-NOTE: We'll use plpy for PostgreSQL interaction when running inside PostgreSQL
# For testing outside PostgreSQL, we'll use a mock
try:
    import plpy  # type: ignore

    IN_POSTGRES = True
except ImportError:
    IN_POSTGRES = False

    # Mock plpy for testing
    class MockPlpy:
        def execute(self, query, args=None):
            return []

        def prepare(self, query, types=None):
            return lambda *args: []

        def notice(self, msg):
            logging.info(msg)

        def warning(self, msg):
            logging.warning(msg)

        def error(self, msg):
            raise Exception(msg)

    plpy = MockPlpy()

# Configure logging
logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages cache synchronization between SteadyText and PostgreSQL.

    AIDEV-NOTE: This class provides methods to:
    1. Generate cache keys compatible with SteadyText
    2. Read from PostgreSQL cache with frecency updates
    3. Write to PostgreSQL cache
    4. Sync with SteadyText's SQLite cache (future feature)
    5. Implement frecency-based eviction
    """

    def __init__(self, table_name: str = "steadytext_cache"):
        """
        Initialize the cache manager.

        Args:
            table_name: Name of the PostgreSQL cache table
        """
        # AIDEV-NOTE: Validate table name to prevent SQL injection
        # Only allow alphanumeric characters and underscores
        if not table_name or not table_name.replace("_", "").isalnum():
            raise ValueError(
                f"Invalid table name: {table_name}. Only alphanumeric characters and underscores are allowed."
            )

        # Additional security: ensure table name doesn't contain SQL keywords
        sql_keywords = {
            "drop",
            "delete",
            "insert",
            "update",
            "select",
            "alter",
            "create",
            "truncate",
        }
        if table_name.lower() in sql_keywords:
            raise ValueError(f"Table name cannot be a SQL keyword: {table_name}")

        self.table_name = table_name

        # AIDEV-NOTE: Prepare commonly used queries for better performance
        if IN_POSTGRES:
            self._prepare_statements()

    def _prepare_statements(self):
        """Prepare PostgreSQL statements for repeated use."""
        # AIDEV-NOTE: Prepared statements improve performance for repeated queries
        # AIDEV-NOTE: SQL injection vulnerability fixed - table_name is now validated
        # in __init__ to only allow alphanumeric characters and underscores
        self.get_plan = plpy.prepare(
            f"""
            UPDATE {self.table_name}
            SET access_count = access_count + 1,
                last_accessed = NOW()
            WHERE cache_key = $1
            RETURNING response, embedding
        """,
            ["text"],
        )

        self.insert_plan = plpy.prepare(
            f"""
            INSERT INTO {self.table_name}
            (cache_key, prompt, response, embedding, model_name, seed, generation_params)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (cache_key)
            DO UPDATE SET
                response = EXCLUDED.response,
                embedding = EXCLUDED.embedding,
                access_count = {self.table_name}.access_count + 1,
                last_accessed = NOW()
        """,
            ["text", "text", "text", "text", "text", "int", "jsonb"],
        )

        self.evict_plan = plpy.prepare(
            f"""
            DELETE FROM {self.table_name}
            WHERE id IN (
                SELECT id FROM {self.table_name}
                ORDER BY frecency_score ASC
                LIMIT $1
            )
            RETURNING id
        """,
            ["int"],
        )

    def generate_cache_key(
        self, prompt: str, params: Optional[Dict[str, Any]] = None, key_prefix: str = ""
    ) -> str:
        """
        Generate a cache key compatible with SteadyText's cache key format.

        AIDEV-NOTE: Updated to match SteadyText's simple format from utils.py:
        - For generation: "{prompt}" or "{prompt}::EOS::{eos_string}"
        - For embeddings: Uses SHA256 hash of "embed:{text}"
        - No longer includes other parameters in the key

        Args:
            prompt: The input prompt
            params: Generation parameters (only eos_string is used for generation)
            key_prefix: Optional prefix for the key (e.g., "embed:" for embeddings)

        Returns:
            Cache key string (plain text for generation, SHA256 for embeddings)
        """
        # For embeddings, use SHA256 hash
        if key_prefix == "embed:":
            # AIDEV-NOTE: Embeddings still use SHA256 for consistency
            key_string = f"{key_prefix}{prompt}"
            return hashlib.sha256(key_string.encode()).hexdigest()

        # For generation, match SteadyText's simple format
        if params and "eos_string" in params and params["eos_string"] != "[EOS]":
            return f"{prompt}::EOS::{params['eos_string']}"
        else:
            return prompt

    def get_cached_generation(
        self, prompt: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Retrieve cached text generation from PostgreSQL.

        AIDEV-NOTE: Updates access count and last_accessed timestamp atomically
        to maintain accurate frecency scores.

        Args:
            prompt: The input prompt
            params: Generation parameters

        Returns:
            Cached response text or None if not found
        """
        if not IN_POSTGRES:
            return None

        cache_key = self.generate_cache_key(prompt, params)

        try:
            result = plpy.execute(self.get_plan, [cache_key])

            if result and len(result) > 0 and result[0]["response"]:
                return result[0]["response"]

        except Exception as e:
            plpy.warning(f"Cache retrieval error: {e}")

        return None

    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding from PostgreSQL.

        AIDEV-NOTE: Embeddings use "embed:" prefix in cache key to separate
        from text generation cache entries.

        Args:
            text: The input text

        Returns:
            Cached embedding as numpy array or None if not found
        """
        if not IN_POSTGRES:
            return None

        cache_key = self.generate_cache_key(text, key_prefix="embed:")

        try:
            result = plpy.execute(self.get_plan, [cache_key])

            if result and len(result) > 0 and result[0]["embedding"]:
                # AIDEV-NOTE: pgvector returns embeddings as strings like "[1,2,3]"
                embedding_str = result[0]["embedding"]
                if isinstance(embedding_str, str):
                    # Parse the vector string
                    embedding_list = json.loads(embedding_str)
                    return np.array(embedding_list, dtype=np.float32)
                else:
                    # Already a list
                    return np.array(embedding_str, dtype=np.float32)

        except Exception as e:
            plpy.warning(f"Embedding cache retrieval error: {e}")

        return None

    def cache_generation(
        self,
        prompt: str,
        response: str,
        params: Optional[Dict[str, Any]] = None,
        model_name: str = "gemma-3n",
    ) -> bool:
        """
        Store text generation in PostgreSQL cache.

        AIDEV-NOTE: Uses INSERT ... ON CONFLICT to handle concurrent inserts
        gracefully. Updates access count if key already exists.

        Args:
            prompt: The input prompt
            response: Generated text
            params: Generation parameters
            model_name: Model used for generation

        Returns:
            True if cached successfully, False otherwise
        """
        if not IN_POSTGRES:
            return False

        cache_key = self.generate_cache_key(prompt, params)

        try:
            plpy.execute(
                self.insert_plan,
                [
                    cache_key,
                    prompt,
                    response,
                    None,  # No embedding for text generation
                    model_name,
                    params.get("seed", 42) if params else 42,  # seed value
                    json.dumps(params or {}),
                ],
            )

            return True

        except Exception as e:
            plpy.warning(f"Cache storage error: {e}")
            return False

    def cache_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
        model_name: str = "qwen3-embedding",
    ) -> bool:
        """
        Store embedding in PostgreSQL cache.

        AIDEV-NOTE: Converts numpy array to PostgreSQL vector format.
        Uses "embed:" prefix for cache key.

        Args:
            text: The input text
            embedding: Embedding vector as numpy array
            model_name: Model used for embedding

        Returns:
            True if cached successfully, False otherwise
        """
        if not IN_POSTGRES:
            return False

        cache_key = self.generate_cache_key(text, params, key_prefix="embed:")

        try:
            # Convert numpy array to PostgreSQL vector format
            embedding_list = embedding.tolist()
            vector_str = "[" + ",".join(map(str, embedding_list)) + "]"

            plpy.execute(
                self.insert_plan,
                [
                    cache_key,
                    text,
                    None,  # No text response for embeddings
                    vector_str,
                    model_name,
                    params.get("seed", 42) if params else 42,  # seed value
                    json.dumps(params or {}),
                ],
            )

            return True

        except Exception as e:
            plpy.warning(f"Embedding cache storage error: {e}")
            return False

    def evict_least_frecent(self, count: int = 1) -> int:
        """
        Evict least frecent cache entries.

        AIDEV-NOTE: Uses the computed frecency_score column for efficient
        eviction. This matches SteadyText's frecency cache behavior.

        Args:
            count: Number of entries to evict

        Returns:
            Number of entries actually evicted
        """
        if not IN_POSTGRES:
            return 0

        try:
            result = plpy.execute(self.evict_plan, [count])
            return len(result)

        except Exception as e:
            plpy.warning(f"Cache eviction error: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        AIDEV-NOTE: Provides insights into cache performance and usage.

        Returns:
            Dictionary with cache statistics
        """
        if not IN_POSTGRES:
            return {"error": "Not running in PostgreSQL"}

        try:
            stats_query = f"""
                SELECT 
                    COUNT(*) as total_entries,
                    COUNT(DISTINCT prompt) as unique_prompts,
                    AVG(access_count) as avg_access_count,
                    MAX(access_count) as max_access_count,
                    SUM(CASE WHEN access_count > 1 THEN 1 ELSE 0 END)::FLOAT / 
                        NULLIF(COUNT(*), 0) as cache_hit_rate,
                    pg_size_pretty(
                        SUM(pg_column_size(response) + 
                            COALESCE(pg_column_size(embedding), 0))
                    ) as total_size,
                    MIN(created_at) as oldest_entry,
                    MAX(created_at) as newest_entry,
                    AVG(frecency_score) as avg_frecency_score
                FROM {self.table_name}
            """

            result = plpy.execute(stats_query)

            if result and len(result) > 0:
                stats = dict(result[0])
                # Convert timestamps to strings for JSON serialization
                if stats.get("oldest_entry"):
                    stats["oldest_entry"] = str(stats["oldest_entry"])
                if stats.get("newest_entry"):
                    stats["newest_entry"] = str(stats["newest_entry"])
                return stats

        except Exception as e:
            plpy.warning(f"Error getting cache stats: {e}")

        return {"error": "Failed to get cache stats"}

    def sync_with_steadytext(self):
        """
        Sync PostgreSQL cache with SteadyText's SQLite cache.

        AIDEV-NOTE: This is a placeholder for future implementation.
        It would read from SteadyText's SQLite cache and import
        entries into PostgreSQL for unified querying.

        AIDEV-SECTION: CACHE_SYNC_STRATEGY
        The cache synchronization strategy should:
        1. Connect to SteadyText's SQLite cache database
        2. Use matching cache key format (SHA256) for consistency
        3. Import entries that don't exist in PostgreSQL
        4. Preserve frecency statistics (access_count, last_accessed)
        5. Handle cache eviction consistently between systems
        6. Optionally bidirectional sync (PostgreSQL -> SQLite)

        Future implementation would:
        1. Connect to SteadyText's SQLite cache
        2. Read entries not in PostgreSQL
        3. Import them with preserved frecency stats
        4. Optionally sync deletions
        """
        # AIDEV-TODO: Implement cache synchronization with SteadyText
        plpy.notice("Cache synchronization with SteadyText not yet implemented")
        pass


# AIDEV-NOTE: Module-level convenience functions for PostgreSQL integration
_default_cache_manager: Optional[CacheManager] = None


def get_default_cache_manager() -> CacheManager:
    """Get or create the default cache manager instance."""
    global _default_cache_manager
    if _default_cache_manager is None:
        _default_cache_manager = CacheManager()
    assert _default_cache_manager is not None
    return cast(CacheManager, _default_cache_manager)

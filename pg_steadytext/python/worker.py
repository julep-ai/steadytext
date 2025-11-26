"""
Background worker for pg_steadytext queue processing
AIDEV-NOTE: This worker processes async generation and embedding requests
"""

import time
import logging
import signal
import json
from typing import Optional, Dict, Any

import psycopg2  # type: ignore
from psycopg2.extras import RealDictCursor  # type: ignore

# AIDEV-NOTE: Input validation is the application's responsibility
# The worker trusts that queue entries have already been validated

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pg_steadytext.worker")


class QueueWorker:
    """
    Worker for processing pg_steadytext queue items
    AIDEV-NOTE: Polls the steadytext_queue table and processes pending requests
    """

    def __init__(self, db_config: Dict[str, Any], poll_interval: int = 1):
        self.db_config = db_config
        self.poll_interval = poll_interval
        self.running = False

    def connect_db(self):
        """Create database connection"""
        return psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)

    def process_generation(self, request_data: Dict[str, Any]) -> str:
        """Process a text generation request"""
        from steadytext import generate

        prompt = request_data["prompt"]
        max_tokens = request_data.get("params", {}).get("max_tokens", 512)

        # Generate text using steadytext directly
        return generate(prompt, max_new_tokens=max_tokens)

    def process_embedding(self, request_data: Dict[str, Any]) -> list:
        """Process an embedding request"""
        from steadytext import embed

        text = request_data["prompt"]

        # Generate embedding using steadytext directly
        embedding = embed(text)

        if embedding is None:
            # Ensure deterministic fallback to zero vector for error cases
            from steadytext.utils import EMBEDDING_DIMENSION

            return [0.0] * EMBEDDING_DIMENSION

        return embedding.tolist()

    def process_generation_json(self, request_data: Dict[str, Any]) -> str:
        """Process a JSON generation request
        AIDEV-NOTE: Added in v1.1.0 for async structured generation support
        """
        from steadytext import generate_json

        prompt = request_data["prompt"]
        params = request_data.get("params", {})
        schema = params.get("schema", {})
        max_tokens = params.get("max_tokens", 512)
        seed = params.get("seed", 42)

        # Generate JSON using steadytext directly
        return generate_json(prompt, schema, max_tokens=max_tokens, seed=seed)

    def process_generation_regex(self, request_data: Dict[str, Any]) -> str:
        """Process a regex-constrained generation request
        AIDEV-NOTE: Added in v1.1.0 for async structured generation support
        """
        from steadytext import generate_regex

        prompt = request_data["prompt"]
        params = request_data.get("params", {})
        pattern = params.get("pattern", "")
        max_tokens = params.get("max_tokens", 512)
        seed = params.get("seed", 42)

        # Generate text matching regex using steadytext directly
        return generate_regex(prompt, pattern, max_tokens=max_tokens, seed=seed)

    def process_generation_choice(self, request_data: Dict[str, Any]) -> str:
        """Process a choice-constrained generation request
        AIDEV-NOTE: Added in v1.1.0 for async structured generation support
        """
        from steadytext import generate_choice

        prompt = request_data["prompt"]
        params = request_data.get("params", {})
        choices = params.get("choices", [])
        seed = params.get("seed", 42)

        # Generate choice using steadytext directly
        return generate_choice(prompt, choices, seed=seed)

    def process_rerank(self, request_data: Dict[str, Any]) -> list:
        """Process a rerank request
        AIDEV-NOTE: Added in v1.3.0 for async rerank support
        """
        from steadytext import rerank

        params = request_data.get("params", {})
        query = params.get("query", "")
        documents = params.get("documents", [])
        task = params.get(
            "task",
            "Given a web search query, retrieve relevant passages that answer the query",
        )
        return_scores = params.get("return_scores", True)
        seed = params.get("seed", 42)

        # Rerank using steadytext directly
        result = rerank(
            query=query,
            documents=documents,
            task=task,
            return_scores=return_scores,
            seed=seed,
        )

        # Convert result to JSON-serializable format
        if return_scores:
            # Result is list of (document, score) tuples
            return [[doc, float(score)] for doc, score in result]
        else:
            # Result is just list of documents
            return result

    def process_queue_item(self, item: Dict[str, Any]) -> None:
        """Process a single queue item"""
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                # Update status to processing
                cur.execute(
                    """
                    UPDATE steadytext_queue 
                    SET status = 'processing',
                        started_at = NOW()
                    WHERE id = %s
                """,
                    (item["id"],),
                )
                conn.commit()

                # Process based on request type
                start_time = time.time()
                try:
                    if item["request_type"] == "generate":
                        result = self.process_generation(item)
                        cur.execute(
                            """
                            UPDATE steadytext_queue 
                            SET status = 'completed',
                                result = %s,
                                completed_at = NOW(),
                                processing_time_ms = %s
                            WHERE id = %s
                        """,
                            (
                                result,
                                int((time.time() - start_time) * 1000),
                                item["id"],
                            ),
                        )

                    elif item["request_type"] == "embed":
                        embedding = self.process_embedding(item)
                        cur.execute(
                            """
                            UPDATE steadytext_queue 
                            SET status = 'completed',
                                embedding = %s::vector,
                                completed_at = NOW(),
                                processing_time_ms = %s
                            WHERE id = %s
                        """,
                            (
                                embedding,
                                int((time.time() - start_time) * 1000),
                                item["id"],
                            ),
                        )

                    # AIDEV-NOTE: Added structured generation types in v1.1.0
                    elif item["request_type"] == "generate_json":
                        result = self.process_generation_json(item)
                        cur.execute(
                            """
                            UPDATE steadytext_queue 
                            SET status = 'completed',
                                result = %s,
                                completed_at = NOW(),
                                processing_time_ms = %s
                            WHERE id = %s
                        """,
                            (
                                result,
                                int((time.time() - start_time) * 1000),
                                item["id"],
                            ),
                        )

                    elif item["request_type"] == "generate_regex":
                        result = self.process_generation_regex(item)
                        cur.execute(
                            """
                            UPDATE steadytext_queue 
                            SET status = 'completed',
                                result = %s,
                                completed_at = NOW(),
                                processing_time_ms = %s
                            WHERE id = %s
                        """,
                            (
                                result,
                                int((time.time() - start_time) * 1000),
                                item["id"],
                            ),
                        )

                    elif item["request_type"] == "generate_choice":
                        result = self.process_generation_choice(item)
                        cur.execute(
                            """
                            UPDATE steadytext_queue 
                            SET status = 'completed',
                                result = %s,
                                completed_at = NOW(),
                                processing_time_ms = %s
                            WHERE id = %s
                        """,
                            (
                                result,
                                int((time.time() - start_time) * 1000),
                                item["id"],
                            ),
                        )

                    elif item["request_type"] == "rerank":
                        result = self.process_rerank(item)
                        cur.execute(
                            """
                            UPDATE steadytext_queue 
                            SET status = 'completed',
                                result = %s::jsonb,
                                completed_at = NOW(),
                                processing_time_ms = %s
                            WHERE id = %s
                        """,
                            (
                                json.dumps(result),  # Convert to JSON string
                                int((time.time() - start_time) * 1000),
                                item["id"],
                            ),
                        )

                    else:
                        raise ValueError(
                            f"Unknown request type: {item['request_type']}"
                        )

                    conn.commit()
                    logger.info(f"Successfully processed request {item['request_id']}")

                except Exception as e:
                    # Update with error
                    cur.execute(
                        """
                        UPDATE steadytext_queue 
                        SET status = 'failed',
                            error = %s,
                            completed_at = NOW(),
                            retry_count = retry_count + 1
                        WHERE id = %s
                    """,
                        (str(e), item["id"]),
                    )
                    conn.commit()
                    logger.error(f"Failed to process request {item['request_id']}: {e}")

        finally:
            conn.close()

    def poll_queue(self) -> Optional[Dict[str, Any]]:
        """Poll for pending queue items"""
        conn = self.connect_db()
        try:
            with conn.cursor() as cur:
                # Get next pending item
                cur.execute("""
                    SELECT * FROM steadytext_queue
                    WHERE status = 'pending'
                    AND retry_count < max_retries
                    ORDER BY created_at
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                """)
                return cur.fetchone()
        finally:
            conn.close()

    def run(self):
        """Main worker loop"""
        logger.info("Starting pg_steadytext queue worker")
        self.running = True

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        while self.running:
            try:
                # Poll for work
                item = self.poll_queue()
                if item:
                    self.process_queue_item(item)
                else:
                    # No work, sleep
                    time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(self.poll_interval)

        logger.info("Worker stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False


def main():
    """Main entry point for worker"""
    # Parse database connection from environment or arguments
    import os

    db_config = {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": int(os.environ.get("PGPORT", 5432)),
        "database": os.environ.get("PGDATABASE", "postgres"),
        "user": os.environ.get("PGUSER", "postgres"),
        "password": os.environ.get("PGPASSWORD", ""),
    }

    # Create and run worker
    worker = QueueWorker(db_config)
    worker.run()


if __name__ == "__main__":
    main()

# AIDEV-NOTE: To run the worker:
# python worker.py
# Or with environment variables:
# PGHOST=localhost PGUSER=postgres PGPASSWORD=password python worker.py

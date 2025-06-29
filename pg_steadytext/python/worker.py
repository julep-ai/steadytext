"""
Background worker for pg_steadytext queue processing
AIDEV-NOTE: This worker processes async generation and embedding requests
"""

import time
import logging
import signal
from typing import Optional, Dict, Any

import psycopg2  # type: ignore
from psycopg2.extras import RealDictCursor  # type: ignore

from .daemon_connector import SteadyTextConnector
from .cache_manager import CacheManager
from .security import SecurityValidator

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
        self.daemon_client = SteadyTextConnector()
        self.cache_manager = CacheManager()
        self.validator = SecurityValidator()

    def connect_db(self):
        """Create database connection"""
        return psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)

    def process_generation(self, request_data: Dict[str, Any]) -> str:
        """Process a text generation request"""
        prompt = request_data["prompt"]
        max_tokens = request_data.get("params", {}).get("max_tokens", 512)
        # thinking_mode removed - not supported by SteadyText

        # Validate input
        is_valid, error_msg = SecurityValidator.validate_prompt(prompt)
        if not is_valid:
            raise ValueError(error_msg)

        # Generate text using daemon connector (handles fallback automatically)
        return self.daemon_client.generate(prompt, max_new_tokens=max_tokens)

    def process_embedding(self, request_data: Dict[str, Any]) -> list:
        """Process an embedding request"""
        text = request_data["prompt"]

        # Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Generate embedding
        if self.daemon_client.is_daemon_running():
            embedding = self.daemon_client.embed(text)
        else:
            # Fallback to direct generation
            from steadytext import embed

            embedding = embed(text)

        return embedding.tolist()

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

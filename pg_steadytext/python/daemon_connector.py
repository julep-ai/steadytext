# AIDEV-SECTION: DAEMON_CONNECTOR
"""
daemon_connector.py - PostgreSQL-friendly wrapper for SteadyText daemon communication

AIDEV-NOTE: This module provides the bridge between PostgreSQL and SteadyText's ZeroMQ daemon.
It handles automatic daemon startup, connection management, and fallback to direct generation.
"""

import time
import subprocess
import logging
from typing import List, Optional, cast
import numpy as np

# AIDEV-NOTE: Import SteadyText components - these should be available if steadytext is installed
try:
    from steadytext import generate, embed, generate_iter
    from steadytext.daemon import use_daemon

    STEADYTEXT_AVAILABLE = True
except ImportError as e:
    STEADYTEXT_AVAILABLE = False
    import sys

    logging.warning(
        f"SteadyText not available - extension will use fallback mode. Error: {e}"
    )
    logging.warning(f"Python path: {sys.path}")
    logging.warning("Install SteadyText with: pip3 install steadytext")

# Configure logging
logger = logging.getLogger(__name__)


class SteadyTextConnector:
    """
    PostgreSQL-friendly wrapper for SteadyText daemon communication.

    AIDEV-NOTE: This class provides a stable interface for PostgreSQL functions
    to interact with SteadyText, handling daemon lifecycle and fallbacks.
    """

    def __init__(
        self, host: str = "localhost", port: int = 5555, auto_start: bool = True
    ):
        """
        Initialize the SteadyText connector.

        Args:
            host: Daemon host address
            port: Daemon port number
            auto_start: Whether to auto-start daemon if not running
        """
        # AIDEV-NOTE: Validate host parameter to prevent injection attacks
        if not host:
            raise ValueError("Host cannot be empty")

        # Allow alphanumeric, dots, hyphens, and underscores (for hostnames and IPs)
        import re

        if not re.match(r"^[a-zA-Z0-9._-]+$", host):
            raise ValueError(
                f"Invalid host: {host}. Only alphanumeric characters, dots, hyphens, and underscores are allowed."
            )

        # Basic IP address validation (both IPv4 and simple hostname)
        if host.count(".") > 0:  # Might be an IP
            parts = host.split(".")
            if len(parts) == 4:  # IPv4 format
                try:
                    for part in parts:
                        num = int(part)
                        if num < 0 or num > 255:
                            raise ValueError(f"Invalid IP address: {host}")
                except ValueError:
                    pass  # Not an IP, might be hostname

        # Validate port parameter
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValueError(
                f"Invalid port: {port}. Port must be an integer between 1 and 65535."
            )

        self.host = host
        self.port = port
        self.auto_start = auto_start
        self.daemon_endpoint = f"tcp://{host}:{port}"

        # AIDEV-NOTE: Check if daemon is running and optionally start it
        if auto_start:
            self._ensure_daemon_running()

    def _ensure_daemon_running(self) -> bool:
        """
        Ensure the SteadyText daemon is running, starting it if necessary.

        AIDEV-NOTE: This method tries to connect to the daemon and starts it
        if the connection fails. It uses the SteadyText CLI for daemon management.

        Returns:
            True if daemon is running or was started successfully, False otherwise
        """
        if not STEADYTEXT_AVAILABLE:
            logger.error("SteadyText not available - cannot start daemon")
            return False

        try:
            # Try to use daemon context manager first
            with use_daemon():
                # Try a simple generation to verify daemon is responding
                test_result = generate("test", max_new_tokens=1)
                if test_result:
                    return True
        except Exception as e:
            logger.info(f"Daemon not responding at {self.daemon_endpoint}: {e}")

            if self.auto_start:
                logger.info("Attempting to start SteadyText daemon...")
                return self._start_daemon()

        return False

    def _start_daemon(self) -> bool:
        """
        Start the SteadyText daemon using the CLI.

        AIDEV-NOTE: Uses subprocess to run 'st daemon start' command.
        Waits briefly for daemon to become available.

        Returns:
            True if daemon started successfully, False otherwise
        """
        try:
            # Start daemon using SteadyText CLI
            result = subprocess.run(
                [
                    "st",
                    "daemon",
                    "start",
                    "--host",
                    self.host,
                    "--port",
                    str(self.port),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.error(f"Failed to start daemon: {result.stderr}")
                return False

            # Wait for daemon to become available
            for i in range(10):  # Try for up to 5 seconds
                time.sleep(0.5)
                try:
                    with use_daemon():
                        test_result = generate("test", max_new_tokens=1)
                        if test_result:
                            logger.info("SteadyText daemon started successfully")
                            return True
                except Exception:
                    continue

            logger.error("Daemon started but not responding")
            return False

        except subprocess.TimeoutExpired:
            logger.error("Timeout starting daemon")
            return False
        except Exception as e:
            logger.error(f"Error starting daemon: {e}")
            return False

    def is_daemon_running(self) -> bool:
        """
        Check if the SteadyText daemon is currently running.

        AIDEV-NOTE: This method checks if the daemon is responsive by attempting
        a simple operation. Used by worker.py to determine daemon availability.

        Returns:
            True if daemon is running and responsive, False otherwise
        """
        if not STEADYTEXT_AVAILABLE:
            return False

        try:
            # Try to use daemon context manager
            with use_daemon():
                # Try a simple generation to verify daemon is responding
                test_result = generate("test", max_new_tokens=1)
                return test_result is not None
        except Exception:
            # Any exception means daemon is not running
            return False

    def check_health(self) -> dict:
        """
        Get detailed health status of the daemon.

        AIDEV-NOTE: Returns a dictionary with health information.
        This method is referenced in the SQL file for daemon status checking.

        Returns:
            Dictionary with health status information
        """
        health_info = {
            "status": "unhealthy",
            "endpoint": self.daemon_endpoint,
            "steadytext_available": STEADYTEXT_AVAILABLE,
        }

        if self.is_daemon_running():
            health_info["status"] = "healthy"

        return health_info

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate text using SteadyText with automatic fallback.

        AIDEV-NOTE: This method tries to use the daemon first, then falls back
        to direct generation if daemon is unavailable. This ensures the PostgreSQL
        extension always returns a result.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate (legacy parameter name)
            max_new_tokens: Maximum tokens to generate (SteadyText parameter name)
            **kwargs: Additional generation parameters

        Returns:
            Generated text string
        """
        # AIDEV-NOTE: Handle both parameter names for compatibility
        if max_new_tokens is None:
            max_new_tokens = max_tokens or 512

        if not STEADYTEXT_AVAILABLE:
            # Return deterministic fallback if SteadyText not available
            return self._fallback_generate(prompt, max_new_tokens)

        try:
            # Try using daemon first
            with use_daemon():
                result = generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
                # Handle both str and tuple returns
                if isinstance(result, tuple):
                    return cast(str, result[0])  # First element is always the text
                else:
                    return cast(str, result)
        except Exception as e:
            logger.warning(f"Daemon generation failed: {e}, using direct generation")

            # Fall back to direct generation
            try:
                result = generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
                # Handle both str and tuple returns
                if isinstance(result, tuple):
                    return cast(str, result[0])  # First element is always the text
                else:
                    return cast(str, result)
            except Exception as e2:
                logger.error(f"Direct generation also failed: {e2}")
                return self._fallback_generate(prompt, max_new_tokens)

    def generate_stream(self, prompt: str, max_tokens: int = 512, **kwargs):
        """
        Generate text in streaming mode.

        AIDEV-NOTE: Yields tokens as they are generated. Falls back to
        chunked output if streaming not available.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Yields:
            Text tokens as they are generated
        """
        if not STEADYTEXT_AVAILABLE:
            # Yield fallback in chunks
            result = self._fallback_generate(prompt, max_tokens)
            for word in result.split():
                yield word + " "
            return

        try:
            # Try streaming with daemon
            with use_daemon():
                for token in generate_iter(
                    prompt,
                    max_new_tokens=max_tokens,
                    **kwargs,
                ):
                    yield token
        except Exception as e:
            logger.warning(f"Daemon streaming failed: {e}, using direct streaming")

            # Fall back to direct streaming
            try:
                for token in generate_iter(
                    prompt,
                    max_new_tokens=max_tokens,
                    **kwargs,
                ):
                    yield token
            except Exception as e2:
                logger.error(f"Direct streaming also failed: {e2}")
                # Yield fallback in chunks
                result = self._fallback_generate(prompt, max_tokens)
                for word in result.split():
                    yield word + " "

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using SteadyText.

        AIDEV-NOTE: Returns a 1024-dimensional normalized embedding vector.
        Falls back to zero vector if generation fails.

        Args:
            text: Input text to embed

        Returns:
            1024-dimensional numpy array
        """
        if not STEADYTEXT_AVAILABLE:
            # Return zero vector as fallback
            return np.zeros(1024, dtype=np.float32)

        try:
            # Try using daemon first
            with use_daemon():
                result = embed(text)
                if result is not None:
                    return result
                # If None, fall through to return zero vector
        except Exception as e:
            logger.warning(f"Daemon embedding failed: {e}, using direct embedding")

            # Fall back to direct embedding
            try:
                result = embed(text)
                if result is not None:
                    return result
                # If None, fall through to return zero vector
            except Exception as e2:
                logger.error(f"Direct embedding also failed: {e2}")

        # Return zero vector as fallback
        return np.zeros(1024, dtype=np.float32)

    def _fallback_generate(self, prompt: str, max_tokens: int) -> str:
        """
        Deterministic fallback text generation.

        AIDEV-NOTE: This provides a predictable output when SteadyText
        is unavailable, ensuring the PostgreSQL extension never errors.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens (used to limit output)

        Returns:
            Deterministic text based on prompt
        """
        # Use hash of prompt for deterministic output
        hash_val = hash(prompt) % 1000

        templates = [
            f"Generated response for prompt (hash: {hash_val}): {prompt[:50]}...",
            f"SteadyText fallback output #{hash_val} for input: {prompt[:50]}...",
            f"Deterministic response {hash_val}: Processing '{prompt[:50]}...'",
        ]

        # Select template based on hash
        template = templates[hash_val % len(templates)]

        # Limit to approximate token count (assuming ~4 chars per token)
        max_chars = max_tokens * 4
        return template[:max_chars]


# AIDEV-NOTE: Module-level convenience functions for PostgreSQL integration
_default_connector: Optional[SteadyTextConnector] = None


def get_default_connector() -> SteadyTextConnector:
    """Get or create the default connector instance."""
    global _default_connector
    if _default_connector is None:
        _default_connector = SteadyTextConnector()
    assert _default_connector is not None
    return cast(SteadyTextConnector, _default_connector)


def pg_generate(prompt: str, max_tokens: int = 512, **kwargs) -> str:
    """PostgreSQL-friendly wrapper for text generation."""
    connector = get_default_connector()
    return connector.generate(prompt, max_tokens, **kwargs)


def pg_embed(text: str) -> List[float]:
    """PostgreSQL-friendly wrapper for embedding generation."""
    connector = get_default_connector()
    embedding = connector.embed(text)
    return embedding.tolist()  # Convert to list for PostgreSQL

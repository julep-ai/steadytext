"""
ZeroMQ client implementation for SteadyText daemon.

AIDEV-NOTE: This client provides transparent access to the daemon server, falling back to direct model loading if the daemon is unavailable.
"""

import os
import contextlib
import threading
import time
from typing import Any, Dict, Optional, Union, Tuple, Iterator, Type, List
import numpy as np

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore[assignment]

from ..utils import logger, DEFAULT_SEED
from .protocol import (
    Request,
    Response,
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    STREAM_END_MARKER,
)


class DaemonClient:
    """Client for communicating with SteadyText daemon server.

    AIDEV-NOTE: Implements automatic fallback to direct model loading when the daemon is unavailable and caches connection failures.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout_ms: Optional[int] = None,
    ):
        if zmq is None:
            logger.warning("pyzmq not available, daemon client disabled")
            self.available = False
            return

        env_host = os.environ.get("STEADYTEXT_DAEMON_HOST", DEFAULT_DAEMON_HOST)
        self.host: str = host if host is not None else env_host
        self.port = port or int(
            os.environ.get("STEADYTEXT_DAEMON_PORT", str(DEFAULT_DAEMON_PORT))
        )
        # AIDEV-NOTE: Read timeout from environment at runtime, not import time
        # This allows tests to set shorter timeouts before imports happen
        if timeout_ms is None:
            timeout_ms = int(os.environ.get("STEADYTEXT_DAEMON_TIMEOUT_MS", "30000"))
        self.timeout_ms = timeout_ms
        self.context: Optional[Any] = None  # zmq.Context when connected
        self.socket: Optional[Any] = None  # zmq.Socket when connected
        self.available = True
        self._connected = False

        # AIDEV-NOTE: Caching connection failures prevents the client from repeatedly trying to connect to a downed daemon in a tight loop.
        self._last_failed_time: Optional[float] = None
        self._failure_cache_duration = float(
            os.environ.get("STEADYTEXT_DAEMON_FAILURE_CACHE_SECONDS", "5")
        )
        # AIDEV-NOTE: Allow disabling failure cache for development/debugging
        self._disable_failure_cache = (
            os.environ.get("STEADYTEXT_DISABLE_FAILURE_CACHE") == "1"
        )

    def connect(self) -> bool:
        """Connect to the daemon server.

        Returns:
            True if connection successful, False otherwise.
        """
        if not self.available:
            return False

        if self._connected:
            return True

        # AIDEV-NOTE: Check if we recently failed to connect (unless cache is disabled)
        if not self._disable_failure_cache and self._last_failed_time is not None:
            time_since_failure = time.time() - self._last_failed_time
            if time_since_failure < self._failure_cache_duration:
                # Still within failure cache window, don't try again
                logger.debug(
                    f"Daemon connection cached as failed, retrying in {self._failure_cache_duration - time_since_failure:.1f}s"
                )
                return False

        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.LINGER, 0)

            connect_address = f"tcp://{self.host}:{self.port}"
            self.socket.connect(connect_address)

            # Test connection with ping
            logger.debug(f"Testing connection to daemon at {connect_address}")
            if self.ping():
                self._connected = True
                self._last_failed_time = None  # Clear failure cache on success
                logger.info(f"Connected to SteadyText daemon at {connect_address}")
                return True
            else:
                self._last_failed_time = time.time()  # Cache failure time
                logger.debug(f"Daemon ping failed at {connect_address}")
                self.disconnect()
                return False

        except Exception as e:
            logger.debug(f"Failed to connect to daemon: {e}")
            self._last_failed_time = time.time()  # Cache failure time
            self.disconnect()
            return False

    def disconnect(self):
        """Disconnect from the daemon server."""
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None
        self._connected = False

    def clear_failure_cache(self):
        """Clear the connection failure cache to force immediate retry."""
        self._last_failed_time = None
        logger.debug("Daemon connection failure cache cleared")

    def ping(self) -> bool:
        """Check if daemon is responsive."""
        if not self.socket:
            return False
        try:
            request = Request(method="ping", params={})
            self.socket.send(request.to_json().encode())
            response_data = self.socket.recv()
            response = Response.from_json(response_data)
            return response.result == "pong" and response.error is None
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        return_logprobs: bool = False,
        eos_string: str = "[EOS]",
        model: Optional[str] = None,
        model_repo: Optional[str] = None,
        model_filename: Optional[str] = None,
        size: Optional[str] = None,
        seed: int = DEFAULT_SEED,
        max_new_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        schema: Optional[Union[Dict[str, Any], Type, object]] = None,
        regex: Optional[str] = None,
        choices: Optional[List[str]] = None,
    ) -> Union[str, Tuple[str, Optional[Dict[str, Any]]], None, Tuple[None, None]]:
        """Generate text via daemon."""
        if not self.connect():
            # AIDEV-NOTE: Fallback to direct generation handled by caller
            raise ConnectionError("Daemon not available")

        assert self.socket is not None  # Type guard for mypy
        try:
            params = {
                "prompt": prompt,
                "return_logprobs": return_logprobs,
                "eos_string": eos_string,
                "model": model,
                "model_repo": model_repo,
                "model_filename": model_filename,
                "size": size,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "response_format": response_format,
                "schema": schema,
                "regex": regex,
                "choices": choices,
            }

            request = Request(method="generate", params=params)
            self.socket.send(request.to_json().encode())
            response_data = self.socket.recv()
            response = Response.from_json(response_data)

            if response.error:
                raise RuntimeError(f"Daemon error: {response.error}")

            # AIDEV-NOTE: Return response result directly - server already formats correctly
            # For logprobs requests, server returns {"text": "...", "logprobs": [...]},
            # for regular requests, server returns the text string directly
            return response.result

        except Exception as e:
            if (
                zmq
                and hasattr(zmq, "error")
                and hasattr(zmq.error, "Again")
                and isinstance(e, zmq.error.Again)
            ):
                logger.warning("Daemon request timed out")
                raise ConnectionError("Daemon request timed out")
            else:
                logger.error(f"Daemon generate error: {e}")
                raise

    def generate_iter(
        self,
        prompt: str,
        eos_string: str = "[EOS]",
        include_logprobs: bool = False,
        model: Optional[str] = None,
        model_repo: Optional[str] = None,
        model_filename: Optional[str] = None,
        size: Optional[str] = None,
        seed: int = DEFAULT_SEED,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """Generate text iteratively via daemon.

        AIDEV-NOTE: The streaming implementation receives multiple responses from the server and yields tokens as they arrive.
        """
        if not self.connect():
            raise ConnectionError("Daemon not available")

        assert self.socket is not None  # Type guard for mypy
        try:
            params = {
                "prompt": prompt,
                "eos_string": eos_string,
                "include_logprobs": include_logprobs,
                "model": model,
                "model_repo": model_repo,
                "model_filename": model_filename,
                "size": size,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
            }

            request = Request(method="generate_iter", params=params)
            self.socket.send(request.to_json().encode())

            # AIDEV-NOTE: Receive streaming responses until end marker
            while True:
                response_data = self.socket.recv()
                response = Response.from_json(response_data)

                if response.error:
                    raise RuntimeError(f"Daemon error: {response.error}")

                # Extract token from response - response.result should be {"token": "..."}
                if isinstance(response.result, dict) and "token" in response.result:
                    token_data = response.result["token"]
                else:
                    token_data = response.result

                if token_data == STREAM_END_MARKER:
                    break  # Don't send ACK for end marker

                # For normal streaming, yield just the token string
                if not include_logprobs:
                    yield token_data
                else:
                    # For logprobs, yield the token data dict (contains token + logprobs)
                    yield token_data

                # Send acknowledgment for next token
                self.socket.send(b"ACK")

        except Exception as e:
            if (
                zmq
                and hasattr(zmq, "error")
                and hasattr(zmq.error, "Again")
                and isinstance(e, zmq.error.Again)
            ):
                logger.warning("Daemon streaming request timed out")
                raise ConnectionError("Daemon request timed out")
            else:
                logger.error(f"Daemon generate_iter error: {e}")
                raise

    def embed(self, text_input: Any, seed: int = DEFAULT_SEED) -> np.ndarray:
        """Generate embeddings via daemon."""
        if not self.connect():
            raise ConnectionError("Daemon not available")

        assert self.socket is not None  # Type guard for mypy
        try:
            params = {"text_input": text_input, "seed": seed}
            request = Request(method="embed", params=params)
            self.socket.send(request.to_json().encode())
            response_data = self.socket.recv()
            response = Response.from_json(response_data)

            if response.error:
                raise RuntimeError(f"Daemon error: {response.error}")

            # AIDEV-NOTE: Convert list back to numpy array
            return np.array(response.result, dtype=np.float32)

        except Exception as e:
            if (
                zmq
                and hasattr(zmq, "error")
                and hasattr(zmq.error, "Again")
                and isinstance(e, zmq.error.Again)
            ):
                logger.warning("Daemon request timed out")
                raise ConnectionError("Daemon request timed out")
            else:
                logger.error(f"Daemon embed error: {e}")
                raise

    def rerank(
        self,
        query: str,
        documents: Union[str, List[str]],
        task: str = "Given a web search query, retrieve relevant passages that answer the query",
        return_scores: bool = True,
        seed: int = DEFAULT_SEED,
    ) -> Union[List[Tuple[str, float]], List[str]]:
        """Rerank documents via daemon.

        AIDEV-NOTE: Sends reranking request to daemon server which uses
        the Qwen3-Reranker model to score query-document pairs.
        """
        if not self.connect():
            raise ConnectionError("Daemon not available")

        assert self.socket is not None  # Type guard for mypy
        try:
            params = {
                "query": query,
                "documents": documents,
                "task": task,
                "return_scores": return_scores,
                "seed": seed,
            }
            request = Request(method="rerank", params=params)
            self.socket.send(request.to_json().encode())
            response_data = self.socket.recv()
            response = Response.from_json(response_data)

            if response.error:
                raise RuntimeError(f"Daemon error: {response.error}")

            # AIDEV-NOTE: Result is already properly formatted from the server
            # Either List[Tuple[str, float]] or List[str] depending on return_scores
            return response.result

        except Exception as e:
            if (
                zmq
                and hasattr(zmq, "error")
                and hasattr(zmq.error, "Again")
                and isinstance(e, zmq.error.Again)
            ):
                logger.warning("Daemon request timed out")
                raise ConnectionError("Daemon request timed out")
            else:
                logger.error(f"Daemon rerank error: {e}")
                raise

    def shutdown(self) -> bool:
        """Request daemon shutdown."""
        if not self.connect():
            return False

        assert self.socket is not None  # Type guard for mypy
        try:
            request = Request(method="shutdown", params={})
            self.socket.send(request.to_json().encode())
            response_data = self.socket.recv()
            response = Response.from_json(response_data)
            return response.error is None
        except Exception:
            return False
        finally:
            self.disconnect()


# AIDEV-NOTE: Global client instance for SDK use with thread safety
_daemon_client = None
_daemon_client_lock = threading.Lock()


def get_daemon_client() -> Optional[DaemonClient]:
    """Get or create the global daemon client instance (thread-safe)."""
    global _daemon_client
    if _daemon_client is None:
        with _daemon_client_lock:
            # Double-check pattern for thread safety
            if _daemon_client is None:
                _daemon_client = DaemonClient()
    return _daemon_client


@contextlib.contextmanager
def use_daemon(
    host: Optional[str] = None, port: Optional[int] = None, required: bool = False
):
    """Context manager for using daemon within a scope.

    Args:
        host: Daemon host (defaults to STEADYTEXT_DAEMON_HOST env var or localhost)
        port: Daemon port (defaults to STEADYTEXT_DAEMON_PORT env var or 5555)
        required: If True, raise exception if daemon is not available

    Example:
        with use_daemon():
            # All generate/embed calls will try to use daemon first
            text = generate("Hello world")
    """
    client = DaemonClient(host=host, port=port)
    connected = client.connect()

    if required and not connected:
        raise RuntimeError("Daemon connection required but not available")

    # AIDEV-NOTE: Force daemon usage within this context (disable fallback)
    old_disable_val = os.environ.get("STEADYTEXT_DISABLE_DAEMON")
    if connected:
        # Ensure daemon is enabled within this context
        if "STEADYTEXT_DISABLE_DAEMON" in os.environ:
            del os.environ["STEADYTEXT_DISABLE_DAEMON"]
        os.environ["STEADYTEXT_DAEMON_HOST"] = client.host
        os.environ["STEADYTEXT_DAEMON_PORT"] = str(client.port)

    try:
        yield client if connected else None
    finally:
        # Restore original disable state
        if old_disable_val is not None:
            os.environ["STEADYTEXT_DISABLE_DAEMON"] = old_disable_val
        elif "STEADYTEXT_DISABLE_DAEMON" in os.environ:
            del os.environ["STEADYTEXT_DISABLE_DAEMON"]
        client.disconnect()

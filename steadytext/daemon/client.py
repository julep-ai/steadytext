"""
ZeroMQ client implementation for SteadyText daemon.

AIDEV-NOTE: This client provides transparent access to the daemon server,
falling back to direct model loading if the daemon is unavailable.
"""

import os
import contextlib
import threading
from typing import Any, Dict, Optional, Union, Tuple, Iterator
import numpy as np

try:
    import zmq
except ImportError:
    zmq = None

from ..utils import logger
from .protocol import (
    Request,
    Response,
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    REQUEST_TIMEOUT_MS,
    STREAM_END_MARKER,
)


class DaemonClient:
    """Client for communicating with SteadyText daemon server.

    AIDEV-NOTE: Implements automatic fallback to direct model loading when daemon
    is unavailable. All methods match the signature of the main API functions.
    """

    def __init__(
        self, host: str = None, port: int = None, timeout_ms: int = REQUEST_TIMEOUT_MS
    ):
        if zmq is None:
            logger.warning("pyzmq not available, daemon client disabled")
            self.available = False
            return

        self.host = host or os.environ.get(
            "STEADYTEXT_DAEMON_HOST", DEFAULT_DAEMON_HOST
        )
        self.port = port or int(
            os.environ.get("STEADYTEXT_DAEMON_PORT", str(DEFAULT_DAEMON_PORT))
        )
        self.timeout_ms = timeout_ms
        self.context = None
        self.socket = None
        self.available = True
        self._connected = False

    def connect(self) -> bool:
        """Connect to the daemon server.

        Returns:
            True if connection successful, False otherwise.
        """
        if not self.available:
            return False

        if self._connected:
            return True

        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.LINGER, 0)

            connect_address = f"tcp://{self.host}:{self.port}"
            self.socket.connect(connect_address)

            # Test connection with ping
            if self.ping():
                self._connected = True
                logger.info(f"Connected to SteadyText daemon at {connect_address}")
                return True
            else:
                self.disconnect()
                return False

        except Exception as e:
            logger.debug(f"Failed to connect to daemon: {e}")
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

    def ping(self) -> bool:
        """Check if daemon is responsive."""
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
    ) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]:
        """Generate text via daemon."""
        if not self.connect():
            # AIDEV-NOTE: Fallback to direct generation handled by caller
            raise ConnectionError("Daemon not available")

        try:
            params = {
                "prompt": prompt,
                "return_logprobs": return_logprobs,
                "eos_string": eos_string,
                "model": model,
                "model_repo": model_repo,
                "model_filename": model_filename,
                "size": size,
            }

            request = Request(method="generate", params=params)
            self.socket.send(request.to_json().encode())
            response_data = self.socket.recv()
            response = Response.from_json(response_data)

            if response.error:
                raise RuntimeError(f"Daemon error: {response.error}")

            # AIDEV-NOTE: Handle different return formats
            if return_logprobs and isinstance(response.result, dict):
                return (response.result["text"], response.result.get("logprobs"))
            return response.result

        except zmq.error.Again:
            logger.warning("Daemon request timed out")
            raise ConnectionError("Daemon request timed out")
        except Exception as e:
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
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """Generate text iteratively via daemon.

        AIDEV-NOTE: Streaming implementation receives multiple responses from server
        and yields tokens as they arrive.
        """
        if not self.connect():
            raise ConnectionError("Daemon not available")

        try:
            params = {
                "prompt": prompt,
                "eos_string": eos_string,
                "include_logprobs": include_logprobs,
                "model": model,
                "model_repo": model_repo,
                "model_filename": model_filename,
                "size": size,
            }

            request = Request(method="generate_iter", params=params)
            self.socket.send(request.to_json().encode())

            # AIDEV-NOTE: Receive streaming responses until end marker
            while True:
                response_data = self.socket.recv()
                response = Response.from_json(response_data)

                if response.error:
                    raise RuntimeError(f"Daemon error: {response.error}")

                token_data = response.result.get("token")
                if token_data == STREAM_END_MARKER:
                    break

                yield token_data

                # Send acknowledgment for next token
                self.socket.send(b"ACK")

        except zmq.error.Again:
            logger.warning("Daemon streaming request timed out")
            raise ConnectionError("Daemon request timed out")
        except Exception as e:
            logger.error(f"Daemon generate_iter error: {e}")
            raise

    def embed(self, text_input: Any) -> np.ndarray:
        """Generate embeddings via daemon."""
        if not self.connect():
            raise ConnectionError("Daemon not available")

        try:
            params = {"text_input": text_input}
            request = Request(method="embed", params=params)
            self.socket.send(request.to_json().encode())
            response_data = self.socket.recv()
            response = Response.from_json(response_data)

            if response.error:
                raise RuntimeError(f"Daemon error: {response.error}")

            # AIDEV-NOTE: Convert list back to numpy array
            return np.array(response.result, dtype=np.float32)

        except zmq.error.Again:
            logger.warning("Daemon request timed out")
            raise ConnectionError("Daemon request timed out")
        except Exception as e:
            logger.error(f"Daemon embed error: {e}")
            raise

    def shutdown(self) -> bool:
        """Request daemon shutdown."""
        if not self.connect():
            return False

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
def use_daemon(host: str = None, port: int = None, required: bool = False):
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

    # AIDEV-NOTE: Set environment variable to signal daemon usage
    old_val = os.environ.get("STEADYTEXT_USE_DAEMON")
    if connected:
        os.environ["STEADYTEXT_USE_DAEMON"] = "1"
        os.environ["STEADYTEXT_DAEMON_HOST"] = client.host
        os.environ["STEADYTEXT_DAEMON_PORT"] = str(client.port)

    try:
        yield client if connected else None
    finally:
        if old_val is None and "STEADYTEXT_USE_DAEMON" in os.environ:
            del os.environ["STEADYTEXT_USE_DAEMON"]
        elif old_val is not None:
            os.environ["STEADYTEXT_USE_DAEMON"] = old_val
        client.disconnect()

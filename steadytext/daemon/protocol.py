"""
Protocol definitions for daemon client-server communication.

AIDEV-NOTE: Uses JSON serialization over ZeroMQ for simplicity and debugging.
All messages include an ID for request-response matching.
"""

import os
import json
import uuid
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class Request:
    """Request message structure."""

    method: str  # "generate", "generate_iter", "embed", "rerank", "ping", "shutdown"
    params: Dict[str, Any]
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_json(self) -> str:
        """Serialize request to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: Union[str, bytes]) -> "Request":
        """Deserialize request from JSON string."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        obj = json.loads(data)
        return cls(**obj)


@dataclass
class Response:
    """Response message structure."""

    id: str
    result: Any
    error: Optional[str] = None

    def to_json(self) -> str:
        """Serialize response to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: Union[str, bytes]) -> "Response":
        """Deserialize response from JSON string."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        obj = json.loads(data)
        return cls(**obj)


@dataclass
class ErrorResponse(Response):
    """Error response convenience class."""

    def __init__(self, id: str, error: str):
        super().__init__(id=id, result=None, error=error)


# AIDEV-NOTE: Protocol constants for consistent communication
DEFAULT_DAEMON_PORT = 5555
DEFAULT_DAEMON_HOST = "localhost"
# AIDEV-NOTE: Reasonable timeout for daemon operations including model loading
# Can be overridden via STEADYTEXT_DAEMON_TIMEOUT_MS environment variable
REQUEST_TIMEOUT_MS = int(
    os.environ.get("STEADYTEXT_DAEMON_TIMEOUT_MS", "30000")
)  # 30 seconds default for reliable operation
STREAM_END_MARKER = "##STREAM_END##"

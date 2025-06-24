"""
SteadyText daemon module for persistent model serving via ZeroMQ.

AIDEV-NOTE: This module implements a daemon server that keeps models loaded
in memory and serves requests via ZeroMQ, avoiding repeated model loading overhead.
"""

from .client import DaemonClient, use_daemon
from .protocol import Request, Response, ErrorResponse

__all__ = ["DaemonClient", "use_daemon", "Request", "Response", "ErrorResponse"]

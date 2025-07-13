"""
ZeroMQ server implementation for SteadyText daemon.

AIDEV-NOTE: This server keeps models loaded in memory and serves generation/embedding requests via a ZeroMQ REP socket. It implements graceful shutdown and error handling.
"""

import sys
import signal
import logging
from typing import Dict, Any, Iterator, Optional

try:
    import zmq
except ImportError:
    # Only exit when running as main module, not during imports
    if __name__ == "__main__":
        print(
            "Error: pyzmq is required for daemon mode. Install with: pip install pyzmq",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        zmq = None  # type: ignore[assignment,unreachable]

from ..core.generator import DeterministicGenerator
from ..core.embedder import core_embed
from ..core.reranker import core_rerank
from ..cache_manager import get_generation_cache
from ..utils import (
    logger,
    set_deterministic_environment,
    DEFAULT_SEED,
    generate_cache_key,
    should_use_cache_for_generation,
    should_use_cache_for_streaming,
)
from .protocol import (
    Request,
    Response,
    ErrorResponse,
    DEFAULT_DAEMON_PORT,
    STREAM_END_MARKER,
)


# AIDEV-NOTE: The server maintains singleton instances of the generator and embedder and is integrated with the centralized cache system.
class DaemonServer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_DAEMON_PORT,
        preload_models: bool = True,
        size: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.running = False
        self.context = None
        self.socket = None
        self.size = size  # AIDEV-NOTE: Model size to preload (small, large)

        # AIDEV-NOTE: Model instances are created once and reused for all requests
        self.generator: Optional[DeterministicGenerator] = None

        # Set deterministic environment
        set_deterministic_environment(DEFAULT_SEED)

        if preload_models:
            self._preload_models()

    def _preload_models(self):
        """Preload models to ensure they're ready for requests."""
        logger.info("Preloading models...")
        try:
            # AIDEV-NOTE: Use the public preload_models function with size parameter
            from .. import preload_models

            preload_models(verbose=True, size=self.size)

            # Create generator instance after preloading
            self.generator = DeterministicGenerator()
            logger.info(f"Generator ready with {self.size or 'default'} model")
        except Exception as e:
            logger.error(f"Failed to load generator model: {e}")
            self.generator = None

        # Embedder is loaded on-demand by create_embedding

    def _handle_generate(self, params: Dict[str, Any]) -> Any:
        """Handle text generation request.

        AIDEV-NOTE: Now uses centralized cache for consistent caching behavior
        between daemon and direct access modes.
        """
        if self.generator is None:
            self.generator = DeterministicGenerator()

        prompt = params.get("prompt", "")
        return_logprobs = params.get("return_logprobs", False)
        eos_string = params.get("eos_string", "[EOS]")
        model = params.get("model")
        model_repo = params.get("model_repo")
        model_filename = params.get("model_filename")
        size = params.get("size")
        seed = params.get("seed", DEFAULT_SEED)
        max_new_tokens = params.get("max_new_tokens")

        # AIDEV-NOTE: Check cache first for non-logprobs requests using default model
        # This mirrors the caching logic in core/generator.py
        if should_use_cache_for_generation(return_logprobs, model_repo, model_filename):
            cache_key = generate_cache_key(prompt, eos_string)
            cached = get_generation_cache().get(cache_key)
            if cached is not None:
                logger.debug(f"Daemon: Cache hit for prompt: {str(prompt)[:50]}...")
                return cached

        result = self.generator.generate(
            prompt=prompt,
            return_logprobs=return_logprobs,
            eos_string=eos_string,
            model=model,
            model_repo=model_repo,
            model_filename=model_filename,
            size=size,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )

        # AIDEV-NOTE: Cache the result for non-logprobs requests using default model
        # This ensures cache consistency between daemon and direct access
        if should_use_cache_for_generation(return_logprobs, model_repo, model_filename):
            cache_key = generate_cache_key(prompt, eos_string)
            # Extract text from result if it's a tuple (shouldn't be for non-logprobs, but safety check)
            text_to_cache = result[0] if isinstance(result, tuple) else result
            get_generation_cache().set(cache_key, text_to_cache)
            logger.debug(f"Daemon: Cached result for prompt: {str(prompt)[:50]}...")

        # AIDEV-NOTE: Handle tuple return for logprobs
        if return_logprobs and isinstance(result, tuple):
            return {"text": result[0], "logprobs": result[1]}
        return result

    def _handle_generate_iter(self, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Handle streaming text generation request.

        AIDEV-NOTE: Streaming is handled by yielding multiple responses with the same ID. If a result is cached, streaming is simulated by yielding words from the cached result.
        """
        if self.generator is None:
            self.generator = DeterministicGenerator()

        prompt = params.get("prompt", "")
        eos_string = params.get("eos_string", "[EOS]")
        include_logprobs = params.get("include_logprobs", False)
        model = params.get("model")
        model_repo = params.get("model_repo")
        model_filename = params.get("model_filename")
        size = params.get("size")
        seed = params.get("seed", DEFAULT_SEED)
        max_new_tokens = params.get("max_new_tokens")

        # AIDEV-NOTE: Check cache for non-logprobs streaming requests using default model
        # If cached, simulate streaming by yielding words from cached result
        if should_use_cache_for_streaming(
            include_logprobs, model, model_repo, model_filename, size
        ):
            cache_key = generate_cache_key(prompt, eos_string)
            cached = get_generation_cache().get(cache_key)
            if cached is not None:
                logger.debug(
                    f"Daemon streaming: Cache hit for prompt: {str(prompt)[:50]}..."
                )
                # Simulate streaming by yielding cached text in chunks
                # AIDEV-NOTE: Use same chunking logic as live streaming to ensure consistency
                words = cached.split()
                char_index = 0
                for i, word in enumerate(words):
                    # Find the word in the original text to preserve exact spacing
                    word_start = cached.find(word, char_index)
                    if word_start > char_index:
                        # Yield any whitespace before the word
                        yield cached[char_index:word_start]
                    # Yield the word
                    yield word
                    char_index = word_start + len(word)

                # Yield any remaining content (trailing whitespace)
                if char_index < len(cached):
                    yield cached[char_index:]
                return

        # No cache hit or logprobs requested - use actual streaming
        # AIDEV-NOTE: Collect tokens during streaming to populate cache after completion
        collected_tokens = []
        for token in self.generator.generate_iter(
            prompt=prompt,
            eos_string=eos_string,
            include_logprobs=include_logprobs,
            model=model,
            model_repo=model_repo,
            model_filename=model_filename,
            size=size,
            seed=seed,
            max_new_tokens=max_new_tokens,
        ):
            # For consistency, always yield just the token - main loop will wrap it
            # Token is already a dict if include_logprobs=True, otherwise it's a string
            yield token

            # Collect tokens for caching (only for non-logprobs requests)
            if not include_logprobs:
                # Token is a string when not including logprobs
                collected_tokens.append(token)

        # AIDEV-NOTE: After streaming completes, populate cache for eligible requests
        # Only cache for default model, non-logprobs requests
        if (
            not include_logprobs
            and model is None
            and model_repo is None
            and model_filename is None
            and size is None
            and collected_tokens
        ):
            # Join collected tokens to form complete text
            complete_text = "".join(collected_tokens)

            cache_key = generate_cache_key(prompt, eos_string)
            get_generation_cache().set(cache_key, complete_text)
            logger.debug(
                f"Daemon streaming: Cached result for prompt: {str(prompt)[:50]}..."
            )

    def _handle_embed(self, params: Dict[str, Any]) -> Any:
        """Handle embedding generation request."""
        text_input = params.get("text_input", "")
        seed = params.get("seed", DEFAULT_SEED)
        embedding = core_embed(text_input, seed=seed)

        # AIDEV-NOTE: Convert numpy array to list for JSON serialization
        return embedding.tolist()

    def _handle_rerank(self, params: Dict[str, Any]) -> Any:
        """Handle reranking request.

        AIDEV-NOTE: Reranking results are cached based on query-document-task tuples.
        """
        query = params.get("query", "")
        documents = params.get("documents", [])
        task = params.get(
            "task",
            "Given a web search query, retrieve relevant passages that answer the query",
        )
        return_scores = params.get("return_scores", True)
        seed = params.get("seed", DEFAULT_SEED)

        result = core_rerank(
            query=query,
            documents=documents,
            task=task,
            return_scores=return_scores,
            seed=seed,
        )

        # AIDEV-NOTE: Result is already JSON-serializable (list of tuples or list of strings)
        return result

    def _create_error_response(
        self, request_id: str, error: Exception
    ) -> ErrorResponse:
        """Create a standardized error response.

        AIDEV-NOTE: Centralized error response creation to ensure consistency
        across all error handling paths in the daemon server.
        """
        error_msg = str(error)
        logger.error(f"Error handling request {request_id}: {error_msg}", exc_info=True)
        return ErrorResponse(id=request_id, error=error_msg)  # type: ignore[call-arg]

    def _handle_request(self, request: Request) -> Optional[Response]:
        """Process a single request and return response."""
        try:
            if request.method == "ping":
                assert request.id is not None  # Set in Request.__post_init__
                return Response(id=request.id, result="pong")  # type: ignore[call-arg]
            elif request.method == "shutdown":
                self.running = False
                assert request.id is not None  # Set in Request.__post_init__
                return Response(id=request.id, result="shutdown initiated")  # type: ignore[call-arg]
            elif request.method == "generate":
                result = self._handle_generate(request.params)
                assert request.id is not None  # Set in Request.__post_init__
                return Response(id=request.id, result=result)  # type: ignore[call-arg]
            elif request.method == "generate_iter":
                # AIDEV-NOTE: For streaming, we'll send multiple responses
                # This is handled differently in the main loop
                return None  # Signal to handle streaming
            elif request.method == "embed":
                result = self._handle_embed(request.params)
                assert request.id is not None  # Set in Request.__post_init__
                return Response(id=request.id, result=result)  # type: ignore[call-arg]
            elif request.method == "rerank":
                result = self._handle_rerank(request.params)
                assert request.id is not None  # Set in Request.__post_init__
                return Response(id=request.id, result=result)  # type: ignore[call-arg]
            else:
                assert request.id is not None  # Set in Request.__post_init__
                return ErrorResponse(  # type: ignore[call-arg]
                    id=request.id, error=f"Unknown method: {request.method}"
                )
        except Exception as e:
            assert request.id is not None  # Set in Request.__post_init__
            return self._create_error_response(request.id, e)

    def run(self):
        """Run the daemon server."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        bind_address = f"tcp://{self.host}:{self.port}"
        try:
            self.socket.bind(bind_address)
            logger.info(f"SteadyText daemon listening on {bind_address}")
        except zmq.error.ZMQError as e:  # type: ignore[attr-defined]
            logger.error(f"Failed to bind to {bind_address}: {e}")
            return

        self.running = True

        # AIDEV-NOTE: Set up signal handlers for graceful shutdown (only works in main thread)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            # Signal handlers only work in main thread - ignore this error for test threads
            logger.debug("Could not set signal handlers - running in thread")

        while self.running:
            try:
                # Wait for request with timeout to allow checking self.running
                if self.socket.poll(1000):  # 1 second timeout
                    message = self.socket.recv()
                    request = Request.from_json(message)

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Received request: {request.method} (id: {request.id})"
                        )

                    # AIDEV-NOTE: Streaming is implemented via a request-response loop. The client sends an ACK for each token received.
                    if request.method == "generate_iter":
                        try:
                            for token in self._handle_generate_iter(request.params):
                                # Send each token as a separate response
                                response = Response(
                                    request.id,  # type: ignore[arg-type]
                                    {"token": token},
                                )
                                self.socket.send(response.to_json().encode())

                                # Wait for client acknowledgment before sending next token
                                ack = self.socket.recv()
                                if ack != b"ACK":
                                    logger.warning(f"Unexpected acknowledgment: {ack}")
                                    break

                            # Send end-of-stream marker
                            response = Response(
                                request.id,  # type: ignore[arg-type]
                                {"token": STREAM_END_MARKER},
                            )
                            self.socket.send(response.to_json().encode())
                        except Exception as e:
                            error_response = self._create_error_response(request.id, e)  # type: ignore[arg-type]
                            self.socket.send(error_response.to_json().encode())
                    else:
                        # Normal request-response handling
                        response = self._handle_request(request)
                        if response:
                            self.socket.send(response.to_json().encode())

            except zmq.error.Again:  # type: ignore[attr-defined]
                # Timeout, continue to check self.running
                continue
            except Exception as e:
                # Try to send error response if possible
                try:
                    if "request" in locals():
                        error_response = self._create_error_response(request.id, e)  # type: ignore[arg-type]
                        self.socket.send(error_response.to_json().encode())
                    else:
                        # Log error when no request context is available
                        logger.error(
                            f"Server error without request context: {e}", exc_info=True
                        )
                except Exception:
                    pass

        logger.info("Daemon server shutting down...")
        self.cleanup()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def cleanup(self):
        """Clean up resources."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        logger.info("Daemon server stopped")


def main():
    """Entry point for running the daemon server directly."""
    import argparse

    parser = argparse.ArgumentParser(description="SteadyText daemon server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_DAEMON_PORT, help="Port to bind to"
    )
    parser.add_argument(
        "--no-preload", action="store_true", help="Don't preload models on startup"
    )
    parser.add_argument(
        "--size",
        choices=["small", "large"],
        help="Model size to preload (small=2B, large=4B)",
    )

    args = parser.parse_args()

    server = DaemonServer(
        host=args.host,
        port=args.port,
        preload_models=not args.no_preload,
        size=args.size,
    )
    server.run()


if __name__ == "__main__":
    main()

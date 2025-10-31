"""OpenRouter provider for unified access to multiple AI models.

AIDEV-ANCHOR: OpenRouter provider implementation
AIDEV-NOTE: OpenRouter provides unified API access to models from Anthropic, OpenAI, Meta, etc.
Uses the OpenAI-compatible ChatCompletion API for text generation and provides embedding support via OpenRouter endpoints.
"""

import os
import json
import time
from typing import Optional, Iterator, Dict, Any, List, Union
import logging
import numpy as np
import requests

from .base import RemoteModelProvider
from .openrouter_config import OpenRouterConfig
from .openrouter_errors import (
    OpenRouterError,
    OpenRouterAuthError,
    OpenRouterModelError,
    OpenRouterConnectionError,
    map_http_error_to_exception,
    should_retry_error,
)
from .openrouter_responses import OpenRouterResponseParser

logger = logging.getLogger("steadytext.providers.openrouter")

# AIDEV-NOTE: Import httpx only when needed to avoid forcing dependency
_httpx_module = None


def _get_httpx():
    """Lazy import of httpx module."""
    global _httpx_module
    if _httpx_module is None:
        try:
            import httpx

            _httpx_module = httpx
        except ImportError:
            logger.error("httpx library not installed. Install with: pip install httpx")
            return None
    return _httpx_module


# AIDEV-NOTE: Provide a lazy import handle for the OpenAI-compatible client so tests can mock
_openai_module = None


def _get_openai():
    """Lazy import of the OpenAI-compatible client (used by OpenRouter).

    Tests patch this function to provide a mocked client.
    """
    global _openai_module
    if _openai_module is None:
        try:
            import openai  # type: ignore

            _openai_module = openai
        except Exception as e:
            logger.warning(f"OpenAI client unavailable: {e}")
            _openai_module = None
    return _openai_module


class OpenRouterProvider(RemoteModelProvider):
    """OpenRouter model provider with unified access to multiple AI models.

    AIDEV-ANCHOR: OpenRouter provider class
    AIDEV-NOTE: Provides access to models from Anthropic, OpenAI, Meta, Google, etc.
    through OpenRouter's unified API. Uses best-effort determinism via seed parameters.
    """

    def __init__(
        self, api_key: Optional[str] = None, model: str = "anthropic/claude-3.5-sonnet"
    ):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var
            model: Model to use in OpenRouter format (provider/model-name)
        """
        if api_key is not None:
            api_key_stripped = api_key.strip()
            if not api_key_stripped:
                raise ValueError("OpenRouter API key cannot be empty or whitespace")
            api_key = api_key_stripped

        super().__init__(api_key)
        self.model = model

        # Try to get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
            if self.api_key:
                self.api_key = self.api_key.strip()

        # Validate model format according to contract tests
        if "/" not in self.model:
            raise ValueError("Invalid model format")

        # Initialize configuration
        self.config = OpenRouterConfig(api_key=self.api_key, model=self.model)

        # Initialize lazily used clients/cache
        self._client = None
        self._cached_models: Optional[List[str]] = None
        self._availability_cache: Optional[bool] = None

    @property
    def provider_name(self) -> str:
        """Return the provider name for display."""
        return "OpenRouter"

    def is_available(self) -> bool:
        """Check if OpenRouter provider is available.

        AIDEV-ANCHOR: OpenRouter availability check
        Validates API key and tests connectivity to OpenRouter service.
        """
        if self._availability_cache is not None:
            return self._availability_cache

        if not self._ensure_api_key_loaded():
            self._availability_cache = False
            return False

        try:
            headers = self.config.get_headers()
            url = f"{self.config.base_url}/models"
            # Use requests per contract tests, with (connect, read) timeout tuple
            response = requests.get(url, headers=headers, timeout=self.config.timeout)
            if response.status_code == 200:
                self._availability_cache = True
            elif response.status_code == 401:
                logger.warning("OpenRouter API key authentication failed")
                self._availability_cache = False
            else:
                logger.warning(f"OpenRouter API returned status {response.status_code}")
                self._availability_cache = False
        except Exception as e:
            logger.warning(f"OpenRouter availability check failed: {e}")
            self._availability_cache = False

        return self._availability_cache

    def _ensure_api_key_loaded(self) -> bool:
        """Ensure we have a non-empty API key cached locally and in config."""
        if self.api_key and str(self.api_key).strip():
            return True

        env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if env_key:
            self.api_key = env_key
            try:
                self.config.api_key = env_key
            except Exception:
                # Config may not exist yet during initialization
                pass
            return True

        return False

    def _require_api_key(self) -> str:
        """Return the resolved API key or raise an auth error."""
        if self._ensure_api_key_loaded():
            return str(self.api_key)
        raise OpenRouterAuthError("OpenRouter API key is missing")

    def _get_client(self):
        """Get or create httpx client.

        AIDEV-ANCHOR: HTTP client management
        Creates and reuses httpx client with proper timeouts and connection pooling.
        """
        if self._client is None:
            httpx = _get_httpx()
            if httpx is None:
                raise RuntimeError("httpx library not available")
            # Configure timeout explicitly with connect/read values
            connect_timeout, read_timeout = self.config.timeout
            self._client = httpx.Client(
                timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout)
            )
        return self._client

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Any:
        """Make HTTP request to OpenRouter API with error handling and retries (httpx path)."""
        client = self._get_client()
        headers = self.config.get_headers()
        url = f"{self.config.base_url}{endpoint}"

        for attempt in range(self.config.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = client.get(url, headers=headers)
                elif method.upper() == "POST":
                    if stream:
                        response = client.stream(
                            "POST", url, json=data, headers=headers
                        )
                        return response
                    else:
                        response = client.post(url, json=data, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if 200 <= response.status_code < 300:
                    if stream:
                        return response
                    return response.json()

                try:
                    error_data = response.json()
                except Exception:
                    error_data = {"error": {"message": response.text}}

                exception = map_http_error_to_exception(
                    response.status_code, error_data
                )

                if attempt < self.config.max_retries and should_retry_error(exception):
                    delay = self.config.get_retry_delay(attempt)
                    logger.warning(
                        f"OpenRouter request failed (attempt {attempt + 1}), retrying in {delay}s: {exception}"
                    )
                    time.sleep(delay)
                    continue

                raise exception

            except (ConnectionError, TimeoutError) as e:
                if attempt < self.config.max_retries:
                    delay = self.config.get_retry_delay(attempt)
                    logger.warning(
                        f"OpenRouter connection failed (attempt {attempt + 1}), retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                    continue
                raise OpenRouterConnectionError(
                    f"Connection failed after {self.config.max_retries} retries: {e}"
                )

        raise OpenRouterError("Request failed after all retry attempts")

    def _get_default_headers_for_client(self) -> Dict[str, str]:
        """Headers to attach to OpenAI-compatible clients pointing at OpenRouter."""
        return {
            "HTTP-Referer": "https://github.com/julep-ai/steadytext",
            "X-Title": "SteadyText",
            "User-Agent": "steadytext/python-client",
        }

    def _prepare_embedding_inputs(self, text: Union[str, List[str]]) -> List[str]:
        """Normalize embedding inputs into a list of strings."""
        if isinstance(text, str):
            return [text]

        if isinstance(text, list):
            for idx, item in enumerate(text):
                if not isinstance(item, str):
                    raise TypeError(
                        f"Embedding input at index {idx} must be a string, got {type(item).__name__}"
                    )
            return list(text)

        raise TypeError("Embedding input must be a string or list of strings")

    def _normalize_openai_model_name(self, model: str) -> str:
        """Convert OpenRouter model identifier to OpenAI client format."""
        if "/" in model:
            provider, model_name = model.split("/", 1)
            if provider == "openai":
                return model_name
        return model

    def _deterministic_embedding_fallback(self, count: int) -> np.ndarray:
        """Return deterministic zero embeddings for fallback scenarios."""
        if count <= 0:
            return np.empty((0, 0), dtype=np.float32)
        return np.zeros((count, 1024), dtype=np.float32)

    def _embed_with_openai_client(
        self,
        openai_module,
        texts: List[str],
        model: str,
        normalize: bool,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Embed texts using the OpenAI-compatible client exposed by OpenRouter."""
        client = openai_module.OpenAI(
            api_key=self.api_key,
            base_url=self.config.base_url,
            default_headers=self._get_default_headers_for_client(),
        )

        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("model", self._normalize_openai_model_name(model))
        call_kwargs.setdefault("input", texts)

        response = client.embeddings.create(**call_kwargs)
        embeddings = [
            np.array(item.embedding, dtype=np.float32) for item in response.data
        ]
        if not embeddings:
            return None

        embedding_matrix = np.array(embeddings, dtype=np.float32)
        if embedding_matrix.ndim == 1:
            embedding_matrix = embedding_matrix.reshape(1, -1)

        if normalize and embedding_matrix.size > 0:
            for i in range(embedding_matrix.shape[0]):
                norm = np.linalg.norm(embedding_matrix[i])
                if norm > 0:
                    embedding_matrix[i] = embedding_matrix[i] / norm

        return embedding_matrix

    def _embed_with_requests(
        self,
        texts: List[str],
        model: str,
        normalize: bool,
        **kwargs,
    ) -> np.ndarray:
        """Embed texts using direct REST calls to OpenRouter."""
        url = f"{self.config.base_url}/embeddings"
        headers = self.config.get_headers()
        payload: Dict[str, Any] = {
            "model": model,
            "input": texts,
        }
        payload.update(kwargs)

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout,
            )
        except Exception as exc:
            logger.warning(f"OpenRouter embeddings request failed: {exc}")
            return self._deterministic_embedding_fallback(len(texts))

        if 200 <= response.status_code < 300:
            try:
                response_data = response.json()
            except ValueError as exc:
                raise OpenRouterError(f"Invalid embedding response: {exc}") from exc

            data_items = (
                response_data.get("data", []) if isinstance(response_data, dict) else []
            )
            if not data_items:
                return self._deterministic_embedding_fallback(len(texts))

            embeddings: List[List[float]] = []
            for item in data_items:
                embedding_values = (
                    item.get("embedding") if isinstance(item, dict) else None
                )
                if not isinstance(embedding_values, list):
                    raise OpenRouterError("Embedding response missing 'embedding' list")
                embeddings.append([float(value) for value in embedding_values])

            embeddings_np = np.array(embeddings, dtype=np.float32)
            if embeddings_np.ndim == 1:
                embeddings_np = embeddings_np.reshape(1, -1)

            if normalize and embeddings_np.size > 0:
                for i in range(embeddings_np.shape[0]):
                    norm = np.linalg.norm(embeddings_np[i])
                    if norm > 0:
                        embeddings_np[i] = embeddings_np[i] / norm

            return embeddings_np

        try:
            err_json = response.json()
        except Exception:
            err_json = {"error": {"message": response.text}}
        raise map_http_error_to_exception(response.status_code, err_json)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
        schema: Optional[Union[Dict[str, Any], type, object]] = None,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """Generate text using OpenRouter API.

        AIDEV-ANCHOR: Text generation method
        Uses OpenRouter's ChatCompletion API with seed for best-effort determinism.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            seed: Seed for best-effort determinism
            temperature: Temperature for sampling (0.0 = deterministic)
            response_format: Response format specification
            schema: JSON schema for structured output
            stream: If True, return an iterator (SSE streaming)
            **kwargs: Additional OpenRouter-specific parameters

        Returns:
            Generated text

        Raises:
            OpenRouterError: On API errors (auth/model/etc.)
        """
        self._issue_warning()

        self._require_api_key()

        # Parameter validation per contract tests
        if not (0.0 <= float(temperature) <= 2.0):
            raise ValueError("Temperature must be between 0 and 2")
        if "top_p" in kwargs:
            top_p = float(kwargs["top_p"])  # type: ignore[arg-type]
            if not (0.0 <= top_p <= 1.0):
                raise ValueError("top_p must be between 0 and 1")
        if "max_tokens" in kwargs:
            if int(kwargs["max_tokens"]) <= 0:
                raise ValueError("max_tokens must be positive")
        if max_new_tokens is not None and int(max_new_tokens) <= 0:
            raise ValueError("max_tokens must be positive")

        # Build request payload
        messages = [{"role": "user", "content": prompt}]

        # Handle structured output
        if response_format or schema:
            if schema and not response_format:
                response_format = {"type": "json_object"}

            if schema:
                schema_str = (
                    json.dumps(schema) if isinstance(schema, dict) else str(schema)
                )
                system_message = f"You must respond with valid JSON that adheres to this schema: {schema_str}"
                messages.insert(0, {"role": "system", "content": system_message})

        # Build parameters
        params: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
        }

        # Add optional parameters
        if max_new_tokens is not None:
            params["max_tokens"] = max_new_tokens
        if response_format is not None:
            params["response_format"] = response_format

        # Add additional kwargs
        params.update(kwargs)

        url = f"{self.config.base_url}/chat/completions"
        headers = self.config.get_headers()

        # Streaming mode returns an iterator over SSE chunks
        if stream:
            stream_payload = dict(params)
            stream_payload["stream"] = True
            try:
                resp = requests.post(
                    url,
                    json=stream_payload,
                    headers=headers,
                    stream=True,
                )
            except Exception:
                # Deterministic fallback: return an iterator yielding nothing
                # Fall back to non-streaming
                try:
                    nonstream_payload = dict(params)
                    nonstream_payload["stream"] = False
                    nonstream = requests.post(
                        url,
                        json=nonstream_payload,
                        headers=headers,
                    )
                    if 200 <= nonstream.status_code < 300:
                        data = nonstream.json()
                        text = OpenRouterResponseParser.parse_chat_completion(data)
                        return iter([text])  # type: ignore[return-value]
                except Exception:
                    pass
                return iter(())  # type: ignore[return-value]

            if 200 <= resp.status_code < 300:

                def _iter():
                    yielded = False
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        try:
                            line = (
                                raw.decode("utf-8")
                                if isinstance(raw, (bytes, bytearray))
                                else str(raw)
                            )
                        except Exception:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                content = (
                                    OpenRouterResponseParser.parse_streaming_chunk(
                                        chunk
                                    )
                                )
                                if content:
                                    yielded = True
                                    yield content
                            except Exception:
                                continue
                    # Fallback if nothing yielded
                    if not yielded:
                        try:
                            nonstream_payload = dict(params)
                            nonstream_payload["stream"] = False
                            nonstream = requests.post(
                                url,
                                json=nonstream_payload,
                                headers=headers,
                            )
                            if 200 <= nonstream.status_code < 300:
                                data = nonstream.json()
                                text = OpenRouterResponseParser.parse_chat_completion(
                                    data
                                )
                                yield text
                        except Exception:
                            return

                return _iter()  # type: ignore[return-value]

            # Map error status to exceptions
            try:
                err_json = resp.json()
            except Exception:
                err_json = {"error": {"message": resp.text}}
            raise map_http_error_to_exception(resp.status_code, err_json)

        # Non-streaming request
        try:
            resp = requests.post(url, json=params, headers=headers)
        except Exception:
            # Deterministic fallback on network failure
            return prompt

        if 200 <= resp.status_code < 300:
            data = resp.json()
            return OpenRouterResponseParser.parse_chat_completion(data)

        try:
            err_json = resp.json()
        except Exception:
            err_json = {"error": {"message": resp.text}}
        raise map_http_error_to_exception(resp.status_code, err_json)

    def generate_iter(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        temperature: float = 0.0,
        **kwargs,
    ) -> Iterator[str]:
        """Generate text iteratively using OpenRouter streaming.

        AIDEV-ANCHOR: Streaming generation method
        Uses OpenRouter's streaming ChatCompletion API for real-time response generation.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            seed: Seed for best-effort determinism
            temperature: Temperature for sampling
            **kwargs: Additional OpenRouter-specific parameters

        Yields:
            Generated text chunks

        Raises:
            OpenRouterError: On API errors (with graceful empty fallback on client failure)
        """
        self._issue_warning()

        self._require_api_key()

        openai = _get_openai()
        if openai is None:
            # Fallback to non-streaming generate when OpenAI client is unavailable
            try:
                fallback_text = self.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    temperature=temperature,
                    **kwargs,
                )
            except Exception:
                return
            else:
                if fallback_text:
                    yield fallback_text
                return

        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.config.base_url,
                default_headers=self._get_default_headers_for_client(),
            )
        except Exception:
            try:
                fallback_text = self.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    temperature=temperature,
                    **kwargs,
                )
            except Exception:
                return
            else:
                if fallback_text:
                    yield fallback_text
                return

        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "seed": seed,
            "stream": True,
        }
        if max_new_tokens is not None:
            call_kwargs["max_tokens"] = max_new_tokens
        call_kwargs.update(kwargs)

        try:
            stream = client.chat.completions.create(**call_kwargs)
        except Exception:
            try:
                fallback_text = self.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    temperature=temperature,
                    **kwargs,
                )
            except Exception:
                return
            else:
                if fallback_text:
                    yield fallback_text
                return

        yielded_any = False
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yielded_any = True
                    yield content
            except Exception:
                continue

        if not yielded_any:
            try:
                fallback_text = self.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    temperature=temperature,
                    **kwargs,
                )
            except Exception:
                return
            else:
                if fallback_text:
                    yield fallback_text

    def embed(
        self,
        text: Union[str, List[str]],
        seed: int = 42,
        model: Optional[str] = None,
        normalize: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Generate embeddings using OpenRouter.

        Args:
            text: Text or list of texts to embed
            seed: Seed parameter (currently unused, reserved for future determinism support)
            model: Optional override for embedding model
            normalize: Whether to L2-normalize embeddings in REST fallback path
            **kwargs: Additional OpenRouter-specific parameters

        Returns:
            NumPy array containing embeddings, or None on unrecoverable error
        """
        del seed  # Seed currently unused for remote embeddings
        self._issue_warning()

        if not self._ensure_api_key_loaded():
            if isinstance(text, list):
                return self._deterministic_embedding_fallback(len(text))
            return self._deterministic_embedding_fallback(1)

        self._require_api_key()

        if text is None:
            raise ValueError("Embedding input cannot be None")

        texts = self._prepare_embedding_inputs(text)
        if len(texts) == 0:
            return np.empty((0, 0), dtype=np.float32)

        embedding_model = model or self.model

        openai_module = _get_openai()
        if openai_module is not None:
            try:
                result = self._embed_with_openai_client(
                    openai_module,
                    texts,
                    embedding_model,
                    normalize,
                    **kwargs,
                )
                if result is not None:
                    return result
            except Exception as exc:
                logger.error(
                    f"OpenRouter OpenAI-compatible embedding call failed: {exc}"
                )
                return None

        rest_kwargs = dict(kwargs)
        return self._embed_with_requests(
            texts, embedding_model, normalize, **rest_kwargs
        )

    def get_supported_models(self) -> List[str]:
        """Get list of supported OpenRouter models.

        AIDEV-ANCHOR: Model listing method
        Retrieves available models from OpenRouter API with caching.

        Returns:
            List of model names in OpenRouter format (provider/model-name)
        """
        if self._cached_models is not None:
            return list(self._cached_models)

        try:
            url = f"{self.config.base_url}/models"
            headers = self.config.get_headers()
            resp = requests.get(url, headers=headers)
            if not (200 <= resp.status_code < 300):
                return []
            response_data = resp.json()

            if "data" in response_data:
                models: List[str] = []
                for model_info in response_data["data"]:
                    model_id = model_info.get("id")
                    if isinstance(model_id, str) and "/" in model_id:
                        parts = model_id.split("/")
                        if len(parts) >= 2 and all(part.strip() for part in parts):
                            models.append(model_id)
                models = sorted(models)
                self._cached_models = models
                return list(models)
            else:
                logger.warning("Unexpected response format from OpenRouter models API")
                return []

        except Exception as e:
            logger.warning(f"Failed to get OpenRouter models: {e}")
            return []

    def get_supported_embedding_models(self) -> List[str]:
        """Return the subset of models that appear to support embeddings."""
        models = self.get_supported_models()
        if not models:
            return []

        return [model for model in models if self.config.is_embedding_model(model)]

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific OpenRouter model.

        Args:
            model: Model name in OpenRouter format

        Returns:
            Dictionary with model information (pricing, capabilities, etc.)

        Raises:
            OpenRouterModelError: If model not found
        """
        try:
            url = f"{self.config.base_url}/models"
            headers = self.config.get_headers()
            resp = requests.get(url, headers=headers)
            if not (200 <= resp.status_code < 300):
                try:
                    err_json = resp.json()
                except Exception:
                    err_json = {"error": {"message": resp.text}}
                raise map_http_error_to_exception(resp.status_code, err_json)

            response_data = resp.json()
            if "data" in response_data:
                for model_info in response_data["data"]:
                    if model_info.get("id") == model:
                        return model_info
                raise OpenRouterModelError(f"Model not found: {model}")
            else:
                raise OpenRouterError(
                    "Unexpected response format from OpenRouter models API"
                )

        except OpenRouterError:
            raise
        except Exception as e:
            raise OpenRouterError(f"Failed to get model info: {e}")

    def supports_embeddings(self) -> bool:
        """OpenRouter supports embeddings via the remote API."""
        return True

    def _is_valid_api_key_format(self, api_key: str) -> bool:
        """Validate OpenRouter API key format.

        OpenRouter keys start with 'sk-or-' followed by alphanumeric chars.

        Args:
            api_key: API key to validate

        Returns:
            True if format appears valid
        """
        if not api_key or not api_key.strip():
            return False

        # OpenRouter keys start with 'sk-or-' and are at least 20 characters
        # AIDEV-NOTE: This is a basic check, actual key validation happens on API call
        return api_key.startswith("sk-or-") and len(api_key) >= 20

    def __del__(self):
        """Clean up HTTP client on destruction."""
        if hasattr(self, "_client") and self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass  # Ignore cleanup errors

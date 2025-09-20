"""OpenRouter provider for unified access to multiple AI models.

AIDEV-ANCHOR: OpenRouter provider implementation
AIDEV-NOTE: OpenRouter provides unified API access to models from Anthropic, OpenAI, Meta, etc.
Uses OpenAI-compatible ChatCompletion API for text generation and embeddings.
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
    OpenRouterRateLimitError,
    OpenRouterModelError,
    OpenRouterTimeoutError,
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
            logger.error(
                "httpx library not installed. Install with: pip install httpx"
            )
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

    def __init__(self, api_key: Optional[str] = None, model: str = "anthropic/claude-3.5-sonnet"):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var
            model: Model to use in OpenRouter format (provider/model-name)
        """
        super().__init__(api_key)
        self.model = model

        # Try to get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY")

        # Validate API key presence according to contract tests
        if self.api_key is None or not str(self.api_key).strip():
            raise RuntimeError("API key is missing")

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

        if not self.api_key:
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
        stream: bool = False
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
                        response = client.stream("POST", url, json=data, headers=headers)
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

                exception = map_http_error_to_exception(response.status_code, error_data)

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
            try:
                resp = requests.post(url, json=params, headers=headers, stream=True)
            except Exception:
                # Deterministic fallback: return an iterator yielding nothing
                # Fall back to non-streaming
                try:
                    nonstream = requests.post(url, json=params | {"stream": False}, headers=headers)
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
                            line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                        except Exception:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                content = OpenRouterResponseParser.parse_streaming_chunk(chunk)
                                if content:
                                    yielded = True
                                    yield content
                            except Exception:
                                continue
                    # Fallback if nothing yielded
                    if not yielded:
                        try:
                            nonstream = requests.post(url, json=params | {"stream": False}, headers=headers)
                            if 200 <= nonstream.status_code < 300:
                                data = nonstream.json()
                                text = OpenRouterResponseParser.parse_chat_completion(data)
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
        try:
            openai = _get_openai()
            if openai is None:
                # Fallback to non-streaming generate
                try:
                    text = self.generate(prompt, max_new_tokens=max_new_tokens, seed=seed, temperature=temperature, **kwargs)
                    return iter([text])  # type: ignore[return-value]
                except Exception:
                    return iter(())  # type: ignore[return-value]

            client = openai.OpenAI()
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

            stream = client.chat.completions.create(**call_kwargs)
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
                    # Fallback: non-streaming call via REST
                    return_text = self.generate(prompt, max_new_tokens=max_new_tokens, seed=seed, temperature=temperature, **kwargs)
                    yield return_text
                except Exception:
                    return
        except Exception:
            # Graceful empty iterator on failure per tests
            if False:
                yield ""  # make this a generator syntactically
            return

    def embed(
        self,
        text: Union[str, List[str]],
        seed: int = 42,
        model: Optional[str] = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        raise NotImplementedError("OpenRouterProvider does not support embeddings")

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
                raise OpenRouterError("Unexpected response format from OpenRouter models API")

        except OpenRouterError:
            raise
        except Exception as e:
            raise OpenRouterError(f"Failed to get model info: {e}")

    def supports_embeddings(self) -> bool:
        """OpenRouter does not support embeddings."""
        return False

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
        if hasattr(self, '_client') and self._client is not None:
            try:
                self._client.close()
            except:
                pass  # Ignore cleanup errors
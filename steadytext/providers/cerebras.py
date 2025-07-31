"""Cerebras provider for unsafe mode.

AIDEV-NOTE: Cerebras provides a seed parameter similar to OpenAI
for best-effort determinism with their cloud inference API.
Uses OpenAI client with custom base URL.
"""

import os
from typing import Optional, Iterator, Dict, Any, List, Union
import logging

from .base import RemoteModelProvider

logger = logging.getLogger("steadytext.providers.cerebras")

# AIDEV-NOTE: Import OpenAI only when needed
_openai_module = None


def _get_openai():
    """Lazy import of openai module."""
    global _openai_module
    if _openai_module is None:
        try:
            import openai

            _openai_module = openai
        except ImportError:
            logger.error(
                "OpenAI library not installed. Install with: pip install openai"
            )
            return None
    return _openai_module


class CerebrasProvider(RemoteModelProvider):
    """Cerebras model provider with seed support.

    AIDEV-NOTE: Cerebras inference API provides seed parameter for reproducibility.
    Uses OpenAI-compatible API at api.cerebras.ai.
    """

    API_BASE = "https://api.cerebras.ai/v1"

    def __init__(self, api_key: Optional[str] = None, model: str = "llama3.1-8b"):
        """Initialize Cerebras provider.

        Args:
            api_key: Cerebras API key. If None, uses CEREBRAS_API_KEY env var
            model: Model to use
        """
        super().__init__(api_key)
        self.model = model

        # Try to get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get("CEREBRAS_API_KEY")

        self._client = None

    @property
    def provider_name(self) -> str:
        return f"Cerebras ({self.model})"

    def is_available(self) -> bool:
        """Check if Cerebras is available."""
        if not self.api_key:
            return False

        openai = _get_openai()
        if openai is None:
            return False

        # AIDEV-NOTE: No static model checking - let provider handle it
        return True

    def _get_client(self):
        """Get or create OpenAI client with Cerebras base URL."""
        if self._client is None:
            openai = _get_openai()
            if openai is None:
                raise RuntimeError("OpenAI library not available")
            self._client = openai.OpenAI(base_url=self.API_BASE, api_key=self.api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
        schema: Optional[Union[Dict[str, Any], type, object]] = None,
        **kwargs,
    ) -> str:
        """Generate text using Cerebras with seed for determinism."""
        self._issue_warning()

        if not self.is_available():
            raise RuntimeError(
                "Cerebras provider not available. Install OpenAI client with: pip install openai"
            )

        client = self._get_client()

        # Handle structured output
        if response_format or schema:
            # Convert schema to response_format if needed
            if schema and not response_format:
                response_format = {"type": "json_object"}

            # Create system message instructing JSON output
            if schema:
                import json

                schema_str = (
                    json.dumps(schema) if isinstance(schema, dict) else str(schema)
                )
                system_message = f"You must respond with valid JSON that adheres to this schema: {schema_str}"
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens or 512,
                temperature=temperature,
                seed=seed,  # For reproducibility
                response_format=response_format,
                **kwargs,
            )
        else:
            # Regular generation
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens or 512,
                temperature=temperature,
                seed=seed,  # For reproducibility
                **kwargs,
            )

        return response.choices[0].message.content or ""

    def generate_iter(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        temperature: float = 0.0,
        **kwargs,
    ) -> Iterator[str]:
        """Generate text iteratively using Cerebras streaming."""
        self._issue_warning()

        if not self.is_available():
            raise RuntimeError(
                "Cerebras provider not available. Install OpenAI client with: pip install openai"
            )

        client = self._get_client()

        stream = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens or 512,
            temperature=temperature,
            seed=seed,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        # AIDEV-NOTE: Return empty list to let provider handle model validation
        return []

    def _is_valid_api_key_format(self, api_key: str) -> bool:
        """Validate Cerebras API key format.

        Cerebras keys are typically long alphanumeric strings.

        Args:
            api_key: API key to validate

        Returns:
            True if format appears valid
        """
        if not api_key or not api_key.strip():
            return False

        # Basic check - should be reasonably long
        # AIDEV-NOTE: Cerebras key format may vary, this is a basic sanity check
        clean_key = api_key.strip()
        return len(clean_key) >= 20

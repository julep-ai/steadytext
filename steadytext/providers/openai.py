"""OpenAI provider for unsafe mode.

AIDEV-NOTE: OpenAI provides a seed parameter for "best-effort" determinism
but explicitly states it's not guaranteed across all conditions.
"""

import os
from typing import Optional, Iterator, Dict, Any, List, Union
import logging

from .base import RemoteModelProvider

logger = logging.getLogger("steadytext.providers.openai")

# AIDEV-NOTE: Import OpenAI only when needed to avoid forcing dependency
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


class OpenAIProvider(RemoteModelProvider):
    """OpenAI model provider with seed support.

    AIDEV-NOTE: OpenAI's seed parameter provides best-effort determinism.
    From their docs: "While we make best efforts to ensure determinism,
    it is not guaranteed."
    """

    # Models that support the seed parameter (as of 2024)
    SEED_SUPPORTED_MODELS = [
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
    ]

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var
            model: Model to use (must support seed parameter)
        """
        super().__init__(api_key)
        self.model = model

        # Try to get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        # Initialize client lazily
        self._client = None

    @property
    def provider_name(self) -> str:
        return f"OpenAI ({self.model})"

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        if not self.api_key:
            return False

        openai = _get_openai()
        if openai is None:
            return False

        # AIDEV-NOTE: No static model checking - let provider handle it
        return True

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            openai = _get_openai()
            if openai is None:
                raise RuntimeError("OpenAI library not available")
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def _get_max_tokens_param_name(self) -> str:
        """Get the correct parameter name for max tokens based on model type."""
        # FIXME: This is a temporary heuristic - we should maintain a proper list
        # of reasoning models or query the API for model capabilities
        # Check if this is a reasoning model (o1 series)
        if self.model.startswith("o"):
            return "max_completion_tokens"
        return "max_tokens"

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
        """Generate text using OpenAI with seed for best-effort determinism."""
        self._issue_warning()

        if not self.is_available():
            raise RuntimeError(
                "OpenAI provider not available. Install with: pip install openai"
            )

        client = self._get_client()

        # AIDEV-NOTE: temperature=0 + seed provides maximum determinism possible
        # FIXME: Reasoning models only support temperature=1.0
        # This should be handled more elegantly in the future
        if self.model.startswith("o"):
            actual_temperature = 1.0
        else:
            actual_temperature = temperature

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

            # Build parameters dict, excluding None values
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": actual_temperature,
                "seed": seed,  # Best-effort determinism
                "response_format": response_format,
            }

            # Only include max_tokens if it's not None
            if max_new_tokens is not None:
                param_name = self._get_max_tokens_param_name()
                params[param_name] = max_new_tokens

            # Add any additional kwargs
            params.update(kwargs)

            response = client.chat.completions.create(**params)
        else:
            # Regular generation
            # Build parameters dict, excluding None values
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": actual_temperature,
                "seed": seed,  # Best-effort determinism
            }

            # Only include max_tokens if it's not None
            if max_new_tokens is not None:
                param_name = self._get_max_tokens_param_name()
                params[param_name] = max_new_tokens

            # Add any additional kwargs
            params.update(kwargs)

            response = client.chat.completions.create(**params)

        return response.choices[0].message.content or ""

    def generate_iter(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        temperature: float = 0.0,
        **kwargs,
    ) -> Iterator[str]:
        """Generate text iteratively using OpenAI streaming."""
        self._issue_warning()

        if not self.is_available():
            raise RuntimeError(
                "OpenAI provider not available. Install with: pip install openai"
            )

        client = self._get_client()

        # FIXME: Reasoning models only support temperature=1.0
        # This should be handled more elegantly in the future
        if self.model.startswith("o"):
            actual_temperature = 1.0
        else:
            actual_temperature = temperature

        # Build parameters dict, excluding None values
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": actual_temperature,
            "seed": seed,  # Best-effort determinism
            "stream": True,
        }

        # Only include max_tokens if it's not None
        if max_new_tokens is not None:
            param_name = self._get_max_tokens_param_name()
            params[param_name] = max_new_tokens

        # Add any additional kwargs
        params.update(kwargs)

        stream = client.chat.completions.create(**params)

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def get_supported_models(self) -> List[str]:
        """Get list of models that support seed parameter."""
        # AIDEV-NOTE: Return empty list to let provider handle model validation
        return []

    def _is_valid_api_key_format(self, api_key: str) -> bool:
        """Validate OpenAI API key format.

        OpenAI keys typically start with 'sk-' followed by alphanumeric chars.

        Args:
            api_key: API key to validate

        Returns:
            True if format appears valid
        """
        if not api_key or not api_key.strip():
            return False

        # Basic format check - OpenAI keys start with 'sk-'
        # AIDEV-NOTE: This is a basic check, actual key validation happens on API call
        return api_key.startswith("sk-") and len(api_key) > 10

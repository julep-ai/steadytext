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
            
        # Check if model supports seed
        if self.model not in self.SEED_SUPPORTED_MODELS:
            logger.error(
                f"Model {self.model} does not support seed parameter. "
                f"Supported models: {', '.join(self.SEED_SUPPORTED_MODELS)}"
            )
            return False
            
        return True
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            openai = _get_openai()
            if openai is None:
                raise RuntimeError("OpenAI library not available")
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Generate text using OpenAI with seed for best-effort determinism."""
        self._issue_warning()
        
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available")
        
        client = self._get_client()
        
        # AIDEV-NOTE: temperature=0 + seed provides maximum determinism possible
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,  # Best-effort determinism
            **kwargs
        )
        
        return response.choices[0].message.content or ""
    
    def generate_iter(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        temperature: float = 0.0,
        **kwargs
    ) -> Iterator[str]:
        """Generate text iteratively using OpenAI streaming."""
        self._issue_warning()
        
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available")
        
        client = self._get_client()
        
        stream = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,  # Best-effort determinism
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    def get_supported_models(self) -> List[str]:
        """Get list of models that support seed parameter."""
        return self.SEED_SUPPORTED_MODELS.copy()
    
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
        return api_key.startswith('sk-') and len(api_key) > 10
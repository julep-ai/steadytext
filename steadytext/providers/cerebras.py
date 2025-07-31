"""Cerebras provider for unsafe mode.

AIDEV-NOTE: Cerebras provides a seed parameter similar to OpenAI
for best-effort determinism with their cloud inference API.
"""

import os
from typing import Optional, Iterator, Dict, Any, List
import logging
import json

from .base import RemoteModelProvider

logger = logging.getLogger("steadytext.providers.cerebras")

# AIDEV-NOTE: Import httpx only when needed
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


class CerebrasProvider(RemoteModelProvider):
    """Cerebras model provider with seed support.
    
    AIDEV-NOTE: Cerebras inference API provides seed parameter for reproducibility.
    Uses their cloud API at api.cerebras.ai.
    """
    
    # Available models on Cerebras Cloud
    SUPPORTED_MODELS = [
        "llama3.1-8b",
        "llama3.1-70b", 
        "llama3-8b",
        "llama3-70b",
    ]
    
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
        
        httpx = _get_httpx()
        if httpx is None:
            return False
            
        if self.model not in self.SUPPORTED_MODELS:
            logger.error(
                f"Model {self.model} not supported. "
                f"Supported models: {', '.join(self.SUPPORTED_MODELS)}"
            )
            return False
            
        return True
    
    def _get_client(self):
        """Get or create httpx client."""
        if self._client is None:
            httpx = _get_httpx()
            if httpx is None:
                raise RuntimeError("httpx library not available")
            self._client = httpx.Client(
                base_url=self.API_BASE,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0
            )
        return self._client
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Generate text using Cerebras with seed for determinism."""
        self._issue_warning()
        
        if not self.is_available():
            raise RuntimeError("Cerebras provider not available")
        
        client = self._get_client()
        
        # AIDEV-NOTE: Cerebras uses OpenAI-compatible API format
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens or 512,
            "temperature": temperature,
            "seed": seed,  # For reproducibility
            **kwargs
        }
        
        response = client.post("/chat/completions", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"] or ""
    
    def generate_iter(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        temperature: float = 0.0,
        **kwargs
    ) -> Iterator[str]:
        """Generate text iteratively using Cerebras streaming."""
        self._issue_warning()
        
        if not self.is_available():
            raise RuntimeError("Cerebras provider not available")
        
        httpx = _get_httpx()
        if httpx is None:
            raise RuntimeError("httpx not available")
        
        # AIDEV-NOTE: Use streaming with httpx
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens or 512,
            "temperature": temperature,
            "seed": seed,
            "stream": True,
            **kwargs
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        with httpx.stream(
            "POST",
            f"{self.API_BASE}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60.0
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming response: {data_str}")
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models."""
        return self.SUPPORTED_MODELS.copy()
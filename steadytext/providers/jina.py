"""Jina AI provider for unsafe mode embeddings.

AIDEV-NOTE: Jina AI provides multilingual embeddings with high-quality semantic understanding.
Unlike OpenAI, they don't provide seed parameters, so determinism is not guaranteed.
"""

import os
from typing import Optional, List, Union, Iterator
import logging
import numpy as np

from .base import RemoteModelProvider

logger = logging.getLogger("steadytext.providers.jina")

# AIDEV-NOTE: Import requests only when needed to avoid forcing dependency
_requests_module = None


def _get_requests():
    """Lazy import of requests module."""
    global _requests_module
    if _requests_module is None:
        try:
            import requests

            _requests_module = requests
        except ImportError:
            logger.error(
                "Requests library not installed. Install with: pip install requests"
            )
            return None
    return _requests_module


class JinaProvider(RemoteModelProvider):
    """Jina AI embedding provider.

    AIDEV-NOTE: Jina AI provides high-quality multilingual embeddings but doesn't support
    seed parameters, so embeddings are not deterministic.
    """

    # Available embedding models
    EMBEDDING_MODELS = [
        "jina-embeddings-v3",
        "jina-embeddings-v2-base-en",
        "jina-embeddings-v2-base-de",
        "jina-embeddings-v2-base-es",
        "jina-embeddings-v2-base-code",
        "jina-embeddings-v2-base-zh",
        "jina-clip-v1",
        "jina-colbert-v2",
    ]

    # Model dimensions (for reference and validation)
    MODEL_DIMENSIONS = {
        "jina-embeddings-v3": 1024,  # Can be configured with dimensions parameter
        "jina-embeddings-v2-base-en": 768,
        "jina-embeddings-v2-base-de": 768,
        "jina-embeddings-v2-base-es": 768,
        "jina-embeddings-v2-base-code": 768,
        "jina-embeddings-v2-base-zh": 768,
        "jina-clip-v1": 768,
        "jina-colbert-v2": 128,
    }

    # API endpoint
    API_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(
        self, api_key: Optional[str] = None, model: str = "jina-embeddings-v3"
    ):
        """Initialize Jina provider.

        Args:
            api_key: Jina API key. If None, uses JINA_API_KEY env var
            model: Embedding model to use (default: jina-embeddings-v3)
        """
        super().__init__(api_key)
        self.model = model

        # Try to get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get("JINA_API_KEY")

    @property
    def provider_name(self) -> str:
        return f"Jina ({self.model})"

    def is_available(self) -> bool:
        """Check if Jina is available."""
        if not self.api_key:
            return False

        requests = _get_requests()
        if requests is None:
            return False

        return True

    def embed(
        self,
        text: Union[str, List[str]],
        seed: int = 42,
        task: Optional[str] = None,
        dimensions: Optional[int] = None,
        late_chunking: bool = False,
        embedding_type: str = "float",
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Generate embeddings using Jina AI.

        Args:
            text: Text or list of texts to embed
            seed: Ignored - Jina doesn't support seeded embeddings
            task: Task for the embeddings (e.g., "retrieval.query", "retrieval.passage", "text-matching", "classification", "separation")
            dimensions: Output dimensions (only for jina-embeddings-v3)
            late_chunking: Enable late chunking for better context understanding
            embedding_type: Type of embedding ("float", "binary", "ubinary")
            **kwargs: Additional Jina-specific parameters

        Returns:
            Numpy array of embeddings or None if error
        """
        self._issue_warning()

        if seed != 42:
            logger.warning(
                "Jina does not support seeded embeddings. "
                "Seed parameter will be ignored."
            )

        if not self.is_available():
            raise RuntimeError(
                "Jina provider not available. Install requests with: pip install requests"
            )

        requests = _get_requests()
        if requests is None:
            raise RuntimeError("Requests library not available")

        # Convert single string to list for batch processing
        texts = [text] if isinstance(text, str) else text

        # Build request payload
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_type": embedding_type,
        }

        # Add optional parameters
        if task:
            payload["task"] = task
        if dimensions and self.model == "jina-embeddings-v3":
            payload["dimensions"] = dimensions
        if late_chunking:
            payload["late_chunking"] = late_chunking

        # Add any additional kwargs
        payload.update(kwargs)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            # Call Jina API
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )

            # Check for errors
            response.raise_for_status()

            # Parse response
            result = response.json()

            # Extract embeddings from response
            # AIDEV-NOTE: Jina returns embeddings in the OpenAI-compatible format
            embeddings = []
            for item in result.get("data", []):
                embedding = item.get("embedding", [])
                embeddings.append(embedding)

            if not embeddings:
                logger.error("No embeddings returned from Jina API")
                return None

            # Convert to numpy array
            embeddings_np = np.array(embeddings, dtype=np.float32)

            # Normalize each embedding (L2 normalization)
            # AIDEV-NOTE: Remote embeddings need to be normalized to match SteadyText behavior
            for i in range(embeddings_np.shape[0]):
                norm = np.linalg.norm(embeddings_np[i])
                if norm > 0:
                    embeddings_np[i] = embeddings_np[i] / norm

            # If single text was input, return single embedding
            if isinstance(text, str):
                return embeddings_np[0]

            # For multiple texts, return average embedding (matching SteadyText behavior)
            # AIDEV-NOTE: SteadyText averages batch embeddings then normalizes
            avg_embedding = np.mean(embeddings_np, axis=0)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            return avg_embedding

        except Exception as e:
            logger.error(f"Jina embedding generation failed: {e}")
            return None

    def supports_embeddings(self) -> bool:
        """Jina specializes in embeddings."""
        return True

    def supports_streaming(self) -> bool:
        """Jina doesn't support streaming (embeddings only)."""
        return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        **kwargs,
    ) -> str:
        """Jina doesn't support text generation."""
        raise NotImplementedError(
            "Jina is an embedding-only provider. Use 'embed' method for embeddings."
        )

    def generate_iter(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        **kwargs,
    ) -> Iterator[str]:
        """Jina doesn't support text generation."""
        raise NotImplementedError(
            "Jina is an embedding-only provider. Use 'embed' method for embeddings."
        )

    def get_supported_models(self) -> List[str]:
        """Get list of supported embedding models."""
        return self.EMBEDDING_MODELS

    def _is_valid_api_key_format(self, api_key: str) -> bool:
        """Validate Jina API key format.

        Jina keys typically start with 'jina_' followed by alphanumeric chars.

        Args:
            api_key: API key to validate

        Returns:
            True if format appears valid
        """
        if not api_key or not api_key.strip():
            return False

        # Basic format check - Jina keys start with 'jina_'
        # AIDEV-NOTE: This is a basic check, actual key validation happens on API call
        return api_key.startswith("jina_") and len(api_key) > 10

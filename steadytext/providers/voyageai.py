"""VoyageAI provider for unsafe mode embeddings.

AIDEV-NOTE: VoyageAI specializes in high-quality embeddings with semantic understanding.
Unlike OpenAI, they don't provide seed parameters, so determinism is not guaranteed.
"""

import os
from typing import Optional, List, Union, Iterator
import logging
import numpy as np

from .base import RemoteModelProvider

logger = logging.getLogger("steadytext.providers.voyageai")

# AIDEV-NOTE: Import VoyageAI only when needed to avoid forcing dependency
_voyageai_module = None


def _get_voyageai():
    """Lazy import of voyageai module."""
    global _voyageai_module
    if _voyageai_module is None:
        try:
            import voyageai

            _voyageai_module = voyageai
        except ImportError:
            logger.error(
                "VoyageAI library not installed. Install with: pip install voyageai"
            )
            return None
    return _voyageai_module


class VoyageAIProvider(RemoteModelProvider):
    """VoyageAI embedding provider.

    AIDEV-NOTE: VoyageAI provides high-quality embeddings but doesn't support
    seed parameters, so embeddings are not deterministic.
    """

    # Available embedding models
    EMBEDDING_MODELS = [
        "voyage-3",
        "voyage-3-lite",
        "voyage-code-3",
        "voyage-finance-2",
        "voyage-law-2",
        "voyage-multilingual-2",
        "voyage-large-2-instruct",
        "voyage-large-2",
        "voyage-2",
        "voyage-lite-02-instruct",
    ]

    # Model dimensions (for reference and validation)
    MODEL_DIMENSIONS = {
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-code-3": 1024,
        "voyage-finance-2": 1024,
        "voyage-law-2": 1024,
        "voyage-multilingual-2": 1024,
        "voyage-large-2-instruct": 1024,
        "voyage-large-2": 1536,
        "voyage-2": 1024,
        "voyage-lite-02-instruct": 1024,
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "voyage-3-lite"):
        """Initialize VoyageAI provider.

        Args:
            api_key: VoyageAI API key. If None, uses VOYAGE_API_KEY env var
            model: Embedding model to use (default: voyage-3-lite)
        """
        super().__init__(api_key)
        self.model = model

        # Try to get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get("VOYAGE_API_KEY")

        # Initialize client lazily
        self._client = None

    @property
    def provider_name(self) -> str:
        return f"VoyageAI ({self.model})"

    def is_available(self) -> bool:
        """Check if VoyageAI is available."""
        if not self.api_key:
            return False

        voyageai = _get_voyageai()
        if voyageai is None:
            return False

        return True

    def _get_client(self):
        """Get or create VoyageAI client."""
        if self._client is None:
            voyageai = _get_voyageai()
            if voyageai is None:
                raise RuntimeError("VoyageAI library not available")
            self._client = voyageai.Client(api_key=self.api_key)
        return self._client

    def embed(
        self,
        text: Union[str, List[str]],
        seed: int = 42,
        input_type: str = "document",
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Generate embeddings using VoyageAI.

        Args:
            text: Text or list of texts to embed
            seed: Ignored - VoyageAI doesn't support seeded embeddings
            input_type: Type of input ("document" or "query")
            **kwargs: Additional VoyageAI-specific parameters

        Returns:
            Numpy array of embeddings or None if error
        """
        self._issue_warning()

        if seed != 42:
            logger.warning(
                "VoyageAI does not support seeded embeddings. "
                "Seed parameter will be ignored."
            )

        if not self.is_available():
            raise RuntimeError(
                "VoyageAI provider not available. Install with: pip install voyageai"
            )

        client = self._get_client()

        # Convert single string to list for batch processing
        texts = [text] if isinstance(text, str) else text

        try:
            # Call VoyageAI API
            result = client.embed(
                texts=texts, model=self.model, input_type=input_type, **kwargs
            )

            # Extract embeddings from response
            embeddings = result.embeddings

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
            logger.error(f"VoyageAI embedding generation failed: {e}")
            return None

    def supports_embeddings(self) -> bool:
        """VoyageAI specializes in embeddings."""
        return True

    def supports_streaming(self) -> bool:
        """VoyageAI doesn't support streaming (embeddings only)."""
        return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        **kwargs,
    ) -> str:
        """VoyageAI doesn't support text generation."""
        raise NotImplementedError(
            "VoyageAI is an embedding-only provider. Use 'embed' method for embeddings."
        )

    def generate_iter(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        **kwargs,
    ) -> Iterator[str]:
        """VoyageAI doesn't support text generation."""
        raise NotImplementedError(
            "VoyageAI is an embedding-only provider. Use 'embed' method for embeddings."
        )

    def get_supported_models(self) -> List[str]:
        """Get list of supported embedding models."""
        return self.EMBEDDING_MODELS

    def _is_valid_api_key_format(self, api_key: str) -> bool:
        """Validate VoyageAI API key format.

        VoyageAI keys typically start with 'pa-' followed by alphanumeric chars.

        Args:
            api_key: API key to validate

        Returns:
            True if format appears valid
        """
        if not api_key or not api_key.strip():
            return False

        # Basic format check - VoyageAI keys start with 'pa-'
        # AIDEV-NOTE: This is a basic check, actual key validation happens on API call
        return api_key.startswith("pa-") and len(api_key) > 10

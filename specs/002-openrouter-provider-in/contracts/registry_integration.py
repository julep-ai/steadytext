"""Contract for OpenRouter integration with provider registry.

This contract defines how OpenRouter provider should integrate with the
existing provider registry system in steadytext.providers.registry.
"""

from typing import Dict, List, Type, Optional, Tuple
from abc import ABC, abstractmethod


class ProviderRegistryContract(ABC):
    """Contract for provider registry integration."""

    @abstractmethod
    def add_openrouter_to_registry(
        self, registry: Dict[str, Type], provider_class: Type
    ) -> Dict[str, Type]:
        """Add OpenRouter provider to PROVIDER_REGISTRY.

        Args:
            registry: Current PROVIDER_REGISTRY dictionary
            provider_class: OpenRouterProvider class

        Returns:
            Updated registry with OpenRouter added

        Expected behavior:
        - Add "openrouter" key mapping to OpenRouterProvider class
        - Maintain existing providers (openai, cerebras, voyageai, jina)
        - Preserve registry immutability patterns
        """
        pass

    @abstractmethod
    def validate_openrouter_api_key(self, api_key: Optional[str]) -> str:
        """Validate OpenRouter API key in get_provider function.

        Args:
            api_key: API key to validate (None means read from env)

        Returns:
            Valid API key string

        Raises:
            RuntimeError: If API key is missing or invalid format

        Expected behavior:
        - Check OPENROUTER_API_KEY environment variable if api_key is None
        - Validate key format (starts with "sk-or-")
        - Provide clear error message if missing
        - Follow same pattern as existing providers
        """
        pass

    @abstractmethod
    def parse_openrouter_model(self, model: str) -> Tuple[str, str]:
        """Parse OpenRouter model string in parse_remote_model function.

        Args:
            model: Model string like "openrouter:anthropic/claude-3.5-sonnet"

        Returns:
            Tuple of ("openrouter", "anthropic/claude-3.5-sonnet")

        Raises:
            ValueError: If model format is invalid

        Expected behavior:
        - Handle "openrouter:" prefix correctly
        - Preserve OpenRouter's provider/model format after colon
        - Validate OpenRouter model format (must contain "/")
        - Integrate with existing is_remote_model() logic
        """
        pass

    @abstractmethod
    def get_openrouter_provider_instance(
        self, model: str, api_key: Optional[str] = None
    ):
        """Create OpenRouter provider instance in get_provider function.

        Args:
            model: Full model string "openrouter:provider/model"
            api_key: Optional API key override

        Returns:
            Configured OpenRouterProvider instance

        Expected behavior:
        - Extract model name after "openrouter:" prefix
        - Pass resolved API key from environment or parameter
        - Call OpenRouterProvider(api_key=key, model=model_name)
        - Verify provider.is_available() before returning
        - Follow same error handling as existing providers
        """
        pass

    @abstractmethod
    def update_provider_listing(self, providers: List[str]) -> List[str]:
        """Update list_providers to include OpenRouter.

        Args:
            providers: Current list of provider names

        Returns:
            Updated list including "openrouter"

        Expected behavior:
        - Add "openrouter" to the list
        - Maintain alphabetical ordering if applicable
        - Preserve existing provider names
        """
        pass

    @abstractmethod
    def update_remote_models_listing(
        self, models: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Update list_remote_models to include OpenRouter models.

        Args:
            models: Current provider -> model list mapping

        Returns:
            Updated mapping with OpenRouter models

        Expected behavior:
        - Add "openrouter" key with list of available models
        - Handle API errors gracefully (empty list on failure)
        - Cache model list for performance if possible
        - Follow same error handling as existing providers
        """
        pass


class UnsafeModeContract(ABC):
    """Contract for unsafe mode integration."""

    @abstractmethod
    def check_unsafe_mode_for_openrouter(self) -> bool:
        """Verify unsafe mode is enabled for OpenRouter usage.

        Returns:
            True if unsafe mode is enabled

        Expected behavior:
        - Use existing is_unsafe_mode_enabled() function
        - Same check as other remote providers
        - No OpenRouter-specific unsafe mode logic needed
        """
        pass

    @abstractmethod
    def handle_unsafe_mode_error(self) -> RuntimeError:
        """Create appropriate error when unsafe mode is disabled.

        Returns:
            RuntimeError with helpful message

        Expected behavior:
        - Use same error message format as existing providers
        - Mention STEADYTEXT_UNSAFE_MODE=true requirement
        - Include warning about determinism
        - Reference OpenRouter in the error context
        """
        pass


class ApiKeyValidationContract(ABC):
    """Contract for API key validation patterns."""

    @abstractmethod
    def validate_openrouter_key_format(self, api_key: str) -> bool:
        """Validate OpenRouter API key format.

        Args:
            api_key: API key to validate

        Returns:
            True if format is valid

        Expected format:
        - Starts with "sk-or-"
        - Minimum length of 20 characters
        - Contains only alphanumeric, hyphens, underscores
        """
        pass

    @abstractmethod
    def get_openrouter_env_key(self) -> Optional[str]:
        """Get OpenRouter API key from environment.

        Returns:
            API key from OPENROUTER_API_KEY or None

        Expected behavior:
        - Read from os.environ.get("OPENROUTER_API_KEY")
        - Return None if not set or empty
        - No validation in this function (validation happens separately)
        """
        pass

    @abstractmethod
    def generate_key_error_message(self, provider: str) -> str:
        """Generate helpful error message for missing API key.

        Args:
            provider: Provider name ("openrouter")

        Returns:
            Formatted error message

        Expected format:
        - Mention specific environment variable (OPENROUTER_API_KEY)
        - Include link to get API key if available
        - Follow same format as existing providers
        - Be actionable and specific
        """
        pass


# Integration points with existing code
class RegistryIntegrationPoints:
    """Expected changes to existing registry.py file."""

    PROVIDER_REGISTRY_UPDATE = """
    # Add to imports
    from .openrouter import OpenRouterProvider

    # Add to PROVIDER_REGISTRY
    PROVIDER_REGISTRY: Dict[str, Type[RemoteModelProvider]] = {
        "openai": OpenAIProvider,
        "cerebras": CerebrasProvider,
        "voyageai": VoyageAIProvider,
        "jina": JinaProvider,
        "openrouter": OpenRouterProvider,  # NEW
    }
    """

    GET_PROVIDER_UPDATE = """
    # Add to get_provider() function after existing elif blocks
    elif provider_name == "openrouter":
        actual_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not actual_key:
            raise RuntimeError(
                f"Provider {provider_name} is not available. Check API key and dependencies."
            )
    """

    PROVIDER_CONSTRUCTOR_UPDATE = """
    # Add to provider constructor logic
    if provider_name in ["openai", "cerebras", "voyageai", "jina", "openrouter"]:
        provider = provider_class(api_key=actual_key, model=model_name)
    """

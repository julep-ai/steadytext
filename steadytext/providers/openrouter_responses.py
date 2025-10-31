"""OpenRouter API response parsing classes.

AIDEV-ANCHOR: OpenRouter response models
This module defines Pydantic models for parsing OpenRouter API responses,
ensuring type safety and validation for both chat completion and embedding responses.
"""

from typing import List, Optional, Dict, Any, Iterator
import numpy as np
from pydantic import BaseModel, Field, validator

from .openrouter_errors import OpenRouterError


class Usage(BaseModel):
    """Token usage tracking information.

    AIDEV-NOTE: Tracks token consumption for billing and monitoring.
    Total tokens should equal prompt_tokens + completion_tokens.
    """

    prompt_tokens: int = Field(..., ge=0, description="Input tokens consumed")
    completion_tokens: int = Field(..., ge=0, description="Output tokens generated")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")

    @validator("total_tokens")
    def validate_total_tokens(cls, v, values):
        """Validate that total tokens equals sum of prompt and completion tokens."""
        prompt_tokens = values.get("prompt_tokens", 0)
        completion_tokens = values.get("completion_tokens", 0)
        expected_total = prompt_tokens + completion_tokens

        # Allow for small discrepancies due to API variations
        if abs(v - expected_total) > 1:
            raise ValueError(
                f"total_tokens ({v}) should equal prompt_tokens ({prompt_tokens}) + "
                f"completion_tokens ({completion_tokens}) = {expected_total}"
            )
        return v


class Message(BaseModel):
    """Message content structure for chat completions.

    AIDEV-NOTE: Represents the generated message content from the assistant.
    """

    role: str = Field(..., description="Message role (typically 'assistant')")
    content: str = Field(..., description="Generated text content")

    @validator("role")
    def validate_role(cls, v):
        """Validate message role is expected value."""
        if v not in ["assistant", "user", "system"]:
            raise ValueError(f"Invalid role: {v}")
        return v


class Choice(BaseModel):
    """Single generation choice in OpenRouter chat completion response.

    AIDEV-NOTE: Represents one possible completion from the model.
    OpenRouter typically returns a single choice.
    """

    index: int = Field(..., ge=0, description="Choice index")
    message: Message = Field(..., description="Generated message content")
    finish_reason: str = Field(..., description="Reason for completion")

    @validator("finish_reason")
    def validate_finish_reason(cls, v):
        """Validate finish reason is a known value."""
        valid_reasons = [
            "stop",
            "length",
            "error",
            "content_filter",
            "function_call",
            "tool_calls",
        ]
        if v not in valid_reasons:
            # Log warning but don't fail - OpenRouter might add new reasons
            import logging

            logger = logging.getLogger("steadytext.providers.openrouter")
            logger.warning(f"Unknown finish_reason: {v}")
        return v


class EmbeddingData(BaseModel):
    """Embedding vector data for single text input.

    AIDEV-NOTE: Contains the actual embedding vector and metadata.
    All embedding values must be finite numbers.
    """

    object: str = Field(..., description="Object type (should be 'embedding')")
    index: int = Field(..., ge=0, description="Embedding index")
    embedding: List[float] = Field(..., description="Vector embedding values")

    @validator("object")
    def validate_object_type(cls, v):
        """Validate object type is 'embedding'."""
        if v != "embedding":
            raise ValueError(f"Expected object type 'embedding', got '{v}'")
        return v

    @validator("embedding")
    def validate_embedding_values(cls, v):
        """Validate embedding contains finite numbers."""
        if not v:
            raise ValueError("Embedding vector cannot be empty")

        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Embedding value at index {i} is not a number: {val}")
            if not np.isfinite(val):
                raise ValueError(f"Embedding value at index {i} is not finite: {val}")

        return v


class ChatCompletionResponse(BaseModel):
    """OpenRouter chat completion API response.

    AIDEV-ANCHOR: Chat completion response model
    Parses and validates the full response structure from OpenRouter's
    chat completion API endpoint.
    """

    id: str = Field(..., description="Response identifier")
    object: str = Field(..., description="Response type")
    created: int = Field(..., gt=0, description="Unix timestamp")
    model: str = Field(..., description="Model used for generation")
    choices: List[Choice] = Field(..., description="Generation choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")

    @validator("object")
    def validate_object_type(cls, v):
        """Validate response object type."""
        if v != "chat.completion":
            raise ValueError(f"Expected object type 'chat.completion', got '{v}'")
        return v

    @validator("choices")
    def validate_choices_not_empty(cls, v):
        """Validate that choices list is not empty."""
        if not v:
            raise ValueError("Choices list cannot be empty")
        return v

    def extract_text_content(self) -> str:
        """Extract the generated text content from the response.

        Returns:
            Generated text content from the first choice

        Raises:
            OpenRouterError: If no valid content is found
        """
        if not self.choices:
            raise OpenRouterError("No choices in response")

        choice = self.choices[0]
        content = choice.message.content

        if content is None:
            raise OpenRouterError("No content in response message")

        return content

    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage information as dictionary.

        Returns:
            Dictionary with token usage stats, empty dict if not available
        """
        if not self.usage:
            return {}

        return {
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
        }


class EmbeddingResponse(BaseModel):
    """OpenRouter embedding API response.

    AIDEV-ANCHOR: Embedding response model
    Parses and validates the full response structure from OpenRouter's
    embedding API endpoint.
    """

    object: str = Field(..., description="Response type")
    data: List[EmbeddingData] = Field(..., description="Embedding data")
    model: str = Field(..., description="Model used for embeddings")
    usage: Optional[Usage] = Field(None, description="Token usage information")

    @validator("object")
    def validate_object_type(cls, v):
        """Validate response object type."""
        if v not in ["list", "embedding"]:
            raise ValueError(f"Expected object type 'list' or 'embedding', got '{v}'")
        return v

    @validator("data")
    def validate_data_not_empty(cls, v):
        """Validate that data list is not empty."""
        if not v:
            raise ValueError("Embedding data list cannot be empty")
        return v

    def extract_embeddings(self) -> np.ndarray:
        """Extract embedding vectors as NumPy array.

        Returns:
            NumPy array of embeddings (2D for multiple inputs, 1D for single input)

        Raises:
            OpenRouterError: If no valid embeddings are found
        """
        if not self.data:
            raise OpenRouterError("No embedding data in response")

        embeddings = []
        for item in self.data:
            embeddings.append(item.embedding)

        # Convert to numpy array with float32 dtype for consistency
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # AIDEV-NOTE: Normalize each embedding (L2 normalization) to match SteadyText behavior
        for i in range(embeddings_np.shape[0]):
            norm = np.linalg.norm(embeddings_np[i])
            if norm > 0:
                embeddings_np[i] = embeddings_np[i] / norm

        # Return single embedding if only one input
        if embeddings_np.shape[0] == 1:
            return embeddings_np[0]

        return embeddings_np

    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage information as dictionary.

        Returns:
            Dictionary with token usage stats, empty dict if not available
        """
        if not self.usage:
            return {}

        return {
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
        }


class OpenRouterResponseParser:
    """Utility class for parsing OpenRouter API responses.

    AIDEV-ANCHOR: Response parser utility
    Provides high-level methods to parse different types of OpenRouter
    API responses with proper error handling and validation.
    """

    @staticmethod
    def parse_chat_completion(response_data: Dict[str, Any]) -> str:
        """Parse chat completion response to extract generated text.

        Args:
            response_data: Raw OpenRouter API response dictionary

        Returns:
            Generated text content

        Raises:
            OpenRouterError: If response format is invalid or parsing fails
        """
        try:
            response = ChatCompletionResponse(**response_data)
            return response.extract_text_content()
        except Exception as e:
            raise OpenRouterError(f"Failed to parse chat completion response: {e}")

    @staticmethod
    def parse_embedding_response(response_data: Dict[str, Any]) -> np.ndarray:
        """Parse embedding response to extract vectors.

        Args:
            response_data: Raw OpenRouter API response dictionary

        Returns:
            NumPy array of embedding vectors

        Raises:
            OpenRouterError: If response format is invalid or parsing fails
        """
        try:
            response = EmbeddingResponse(**response_data)
            return response.extract_embeddings()
        except Exception as e:
            raise OpenRouterError(f"Failed to parse embedding response: {e}")

    @staticmethod
    def parse_streaming_chunk(chunk_data: Dict[str, Any]) -> Optional[str]:
        """Parse a single streaming chunk from chat completion.

        Args:
            chunk_data: Raw streaming chunk data

        Returns:
            Text content from chunk, or None if no content

        Raises:
            OpenRouterError: If chunk format is invalid
        """
        try:
            # Streaming chunks have similar structure but with delta instead of message
            if "choices" not in chunk_data or not chunk_data["choices"]:
                return None

            choice = chunk_data["choices"][0]
            if "delta" not in choice:
                return None

            delta = choice["delta"]
            if "content" not in delta:
                return None

            return delta["content"]
        except Exception as e:
            raise OpenRouterError(f"Failed to parse streaming chunk: {e}")

    @staticmethod
    def extract_finish_reason(response_data: Dict[str, Any]) -> Optional[str]:
        """Extract finish reason from response.

        Args:
            response_data: Raw response data

        Returns:
            Finish reason string, or None if not available
        """
        try:
            if "choices" not in response_data or not response_data["choices"]:
                return None

            choice = response_data["choices"][0]
            return choice.get("finish_reason")
        except Exception:
            return None

    @staticmethod
    def extract_usage(response_data: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage information from response.

        Args:
            response_data: Raw response data

        Returns:
            Dictionary with token usage stats, empty if not available
        """
        try:
            if "usage" not in response_data:
                return {}

            usage_data = response_data["usage"]
            return {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }
        except Exception:
            return {}

    @staticmethod
    def parse_streaming_response(stream_response) -> Iterator[str]:
        """Parse streaming response from OpenRouter API.

        Args:
            stream_response: httpx streaming response object

        Yields:
            Text content chunks from the stream

        Raises:
            OpenRouterError: If streaming fails or chunk parsing fails
        """
        import json

        try:
            from .openrouter_errors import OpenRouterError
        except ImportError:
            # Handle case where module isn't fully available
            class OpenRouterError(Exception):
                pass

        try:
            with stream_response as response:
                for line in response.iter_lines():
                    if not line:
                        continue

                    # OpenRouter uses Server-Sent Events format
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        # Check for end-of-stream marker
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            content = OpenRouterResponseParser.parse_streaming_chunk(
                                chunk_data
                            )
                            if content is not None:
                                yield content
                        except json.JSONDecodeError:
                            # Skip malformed JSON chunks
                            continue
                        except OpenRouterError:
                            # Skip chunks that can't be parsed
                            continue

        except Exception as e:
            raise OpenRouterError(f"Streaming response parsing failed: {e}")

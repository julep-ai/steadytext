# AIDEV-NOTE: Core text generation module with deterministic fallback.
# Features:
# - Implements both model-based generation and a hash-based deterministic fallback.
# - Integrates stop sequences (DEFAULT_STOP_SEQUENCES).
# - Ensures determinism using DEFAULT_SEED.
# - Supports streaming token generation via generate_iter().
# - Allows dynamic model switching at runtime.
# - Defers environment setup to prevent pytest collection hangs.

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator, cast, Type

from ..cache_manager import get_generation_cache
from ..models.loader import get_generator_model_instance
from ..utils import set_deterministic_environment  # Assuming this is in utils.py
from ..utils import (
    DEFAULT_SEED,
    DEFAULT_STOP_SEQUENCES,
    GENERATION_MAX_NEW_TOKENS,
    LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC,
    logger,
    resolve_model_params,
    generate_cache_key,
    should_use_cache_for_generation,
    should_use_cache_for_streaming,
    validate_seed,
    get_optimal_context_window,
)
from ..exceptions import ContextLengthExceededError

# AIDEV-NOTE: Removed module-level set_deterministic_environment call to prevent
# hanging during pytest collection. This is now called lazily when needed.


# AIDEV-NOTE: Uses centralized cache manager for consistency. Fallback results are cached.


# AIDEV-NOTE: Main generator class with model caching, error handling, and dynamic model switching.
class DeterministicGenerator:
    def __init__(self) -> None:
        # AIDEV-NOTE: Deterministic environment is set when the generator is created to prevent pytest collection hangs.
        set_deterministic_environment(DEFAULT_SEED)

        self.model = None
        self._logits_enabled = False
        self._current_model_key = "default::default"
        # AIDEV-NOTE: Initialize with default context window to prevent None errors during validation
        self._context_window = (
            get_optimal_context_window()
        )  # Will be updated when model is loaded
        # Load model without logits_all initially
        # AIDEV-NOTE: Skip model loading if STEADYTEXT_SKIP_MODEL_LOAD is set
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") != "1":
            self._load_model(enable_logits=False)

    def _load_model(
        self,
        enable_logits: bool = False,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        force_reload: bool = False,
    ):
        """Load or reload the model with specific logits configuration.

        AIDEV-NOTE: Supports loading custom models and respects STEADYTEXT_SKIP_MODEL_LOAD.
        """
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            logger.debug(
                "_load_model: STEADYTEXT_SKIP_MODEL_LOAD=1, skipping model load"
            )
            self.model = None
            return

        self.model = get_generator_model_instance(
            force_reload=force_reload,
            enable_logits=enable_logits,
            repo_id=repo_id,
            filename=filename,
        )
        self._logits_enabled = enable_logits
        self._current_model_key = f"{repo_id or 'default'}::{filename or 'default'}"

        # AIDEV-NOTE: Store context window size from loaded model
        if self.model is not None:
            # Get actual context window from model
            if hasattr(self.model, "n_ctx"):
                try:
                    ctx_value = self.model.n_ctx()
                    # AIDEV-NOTE: Ensure we get a valid integer, not a Mock or other object
                    # This handles cases where tests use mock models that return Mock objects
                    if isinstance(ctx_value, int) and ctx_value > 0:
                        self._context_window = ctx_value
                    else:
                        raise ValueError(f"Invalid context window value: {ctx_value}")
                except (TypeError, ValueError, AttributeError) as e:
                    # AIDEV-NOTE: Fallback to default context window if model doesn't provide valid n_ctx
                    # This ensures tests with mock models still work properly
                    logger.warning(
                        f"Failed to get context window from model: {e}. Using default."
                    )
                    self._context_window = get_optimal_context_window(
                        model_name=None, model_repo=repo_id
                    )
            else:
                # Fallback to calculating it
                model_name = None
                # Try to resolve model name for context window calculation
                from ..utils import (
                    MODEL_REGISTRY,
                    GENERATION_MODEL_REPO,
                    GENERATION_MODEL_FILENAME,
                )

                for name, config in MODEL_REGISTRY.items():
                    if (
                        repo_id is None
                        and filename is None
                        and config.get("repo") == GENERATION_MODEL_REPO
                        and config.get("filename") == GENERATION_MODEL_FILENAME
                    ) or (
                        config.get("repo") == repo_id
                        and config.get("filename") == filename
                    ):
                        model_name = name
                        break
                self._context_window = get_optimal_context_window(
                    model_name=model_name, model_repo=repo_id or GENERATION_MODEL_REPO
                )
        else:
            logger.error(
                f"DeterministicGenerator: Model instance is None after attempting to load {self._current_model_key}."
            )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer.

        AIDEV-NOTE: This method provides token counting for input validation.
        Falls back to character-based estimation if tokenizer is not available.

        Args:
            text: Input text to count tokens for

        Returns:
            Number of tokens in the text
        """
        if self.model is None:
            # Fallback: estimate ~4 characters per token
            return len(text) // 4

        try:
            # Use model's tokenizer if available
            if hasattr(self.model, "tokenize"):
                tokens = self.model.tokenize(text.encode("utf-8"))
                return len(tokens)
            else:
                # Fallback estimation
                return len(text) // 4
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}. Using fallback estimation.")
            return len(text) // 4

    def _validate_input_length(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> None:
        """Validate that input prompt fits within context window.

        AIDEV-NOTE: Validates input length and raises ContextLengthExceededError
        if the input would exceed the model's context window, leaving room for
        the response tokens.

        Args:
            prompt: Input prompt to validate
            max_new_tokens: Maximum tokens to generate (for reserving space)

        Raises:
            ContextLengthExceededError: If input exceeds available context
        """
        if self._context_window is None:
            # If we don't know the context window, we can't validate
            return

        # Count input tokens
        input_tokens = self._count_tokens(prompt)

        # Reserve space for output
        output_reserve = max_new_tokens or GENERATION_MAX_NEW_TOKENS

        # Calculate available tokens for input (leave 10% margin for safety)
        safety_margin = int(self._context_window * 0.1)
        available_tokens = self._context_window - output_reserve - safety_margin

        if input_tokens > available_tokens:
            raise ContextLengthExceededError(
                input_tokens=input_tokens,
                max_tokens=available_tokens,
                input_text=prompt,
                message=(
                    f"Input is too long: {input_tokens} tokens. "
                    f"Maximum allowed: {available_tokens} tokens "
                    f"(context window: {self._context_window}, "
                    f"reserved for output: {output_reserve}, "
                    f"safety margin: {safety_margin})"
                ),
            )

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        return_logprobs: bool = False,
        eos_string: str = "[EOS]",
        model: Optional[str] = None,
        model_repo: Optional[str] = None,
        model_filename: Optional[str] = None,
        size: Optional[str] = None,
        seed: int = DEFAULT_SEED,
        response_format: Optional[Dict[str, Any]] = None,
        schema: Optional[Union[Dict[str, Any], Type, object]] = None,
        regex: Optional[str] = None,
        choices: Optional[List[str]] = None,
    ) -> Union[str, Tuple[str, Optional[Dict[str, Any]]], None, Tuple[None, None]]:
        """Generate text with optional model switching and structured output.

        Args:
            prompt: Input text prompt
            return_logprobs: Whether to return token log probabilities
            eos_string: End-of-sequence string ("[EOS]" uses model defaults)
            model: Model name from registry (e.g., "qwen2.5-3b")
            model_repo: Custom Hugging Face repository ID
            model_filename: Custom model filename
            size: Size identifier ("small", "large")
            seed: Seed for deterministic generation
            response_format: Dict specifying output format (e.g., {"type": "json_object"})
            schema: JSON schema, Pydantic model, or Python type for structured output
            regex: Regular expression pattern for output format
            choices: List of allowed string choices for output

        AIDEV-NOTE: Model switching parameters allow using different models without restarting.
        AIDEV-NOTE: Structured output parameters delegate to the structured generator when provided.
        """
        validate_seed(seed)
        set_deterministic_environment(seed)

        # AIDEV-NOTE: Check if structured output is requested
        is_structured = any([response_format, schema, regex, choices])

        # Handle structured generation
        if is_structured:
            # Structured generation doesn't support logprobs
            if return_logprobs:
                logger.warning(
                    "Structured generation does not support logprobs. Ignoring return_logprobs=True."
                )

            # Import here to avoid circular dependencies
            from .structured import (
                generate_json,
                generate_regex,
                generate_choice,
            )

            # Determine which structured generator to use
            if schema is not None or (
                response_format and response_format.get("type") == "json_object"
            ):
                # JSON generation
                if schema is None:
                    # If only response_format is provided, generate generic JSON
                    schema = dict  # Generic dict type
                result = generate_json(
                    prompt=prompt,
                    schema=schema,
                    max_tokens=max_new_tokens or GENERATION_MAX_NEW_TOKENS,
                    seed=seed,
                )
                return (result, None) if return_logprobs else result
            elif regex is not None:
                # Regex pattern matching
                result = generate_regex(
                    prompt=prompt,
                    pattern=regex,
                    max_tokens=max_new_tokens or GENERATION_MAX_NEW_TOKENS,
                    seed=seed,
                )
                return (result, None) if return_logprobs else result
            elif choices is not None:
                # Multiple choice
                result = generate_choice(
                    prompt=prompt,
                    choices=choices,
                    max_tokens=max_new_tokens or GENERATION_MAX_NEW_TOKENS,
                    seed=seed,
                )
                return (result, None) if return_logprobs else result

        # Resolve model parameters
        repo_id: Optional[str] = None
        filename: Optional[str] = None

        if model or model_repo or model_filename or size:
            try:
                repo_id, filename = resolve_model_params(
                    model, model_repo, model_filename, size
                )
            except ValueError as e:
                logger.error(f"Invalid model specification: {e}")
                fallback = _deterministic_fallback_generate(prompt, seed)
                return (fallback, None) if return_logprobs else fallback

        # Handle caching only for non-logprobs requests and default model
        if should_use_cache_for_generation(return_logprobs, repo_id, filename):
            cache_key = generate_cache_key(prompt, eos_string)
            cached = get_generation_cache().get(cache_key)
            if cached is not None:
                return cached

        if not isinstance(prompt, str):
            logger.error(
                f"DeterministicGenerator.generate: Invalid prompt type: {type(prompt)}. Expected str."
            )
            return (None, None) if return_logprobs else None

        # Check if we need to load a different model
        model_key = f"{repo_id or 'default'}::{filename or 'default'}"
        needs_different_model = model_key != self._current_model_key

        # Load appropriate model if needed
        if needs_different_model or (return_logprobs and not self._logits_enabled):
            logger.info(f"Loading model {model_key} with logits={return_logprobs}")
            self._load_model(
                enable_logits=return_logprobs,
                repo_id=repo_id,
                filename=filename,
                force_reload=False,  # Use cache if available
            )

        # AIDEV-NOTE: Return None if model is not loaded instead of using fallback
        skip_model_load = os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1"
        if self.model is None or skip_model_load:
            if skip_model_load:
                logger.error(
                    "DeterministicGenerator.generate: STEADYTEXT_SKIP_MODEL_LOAD=1. "
                    "Model loading is disabled."
                )
            else:
                logger.error(
                    "DeterministicGenerator.generate: Model not loaded. "
                    "Cannot generate text."
                )
            # Return None to indicate failure
            return (None, None) if return_logprobs else None

        if not prompt or not prompt.strip():  # Check after ensuring prompt is a string
            logger.error(
                "DeterministicGenerator.generate: Empty or whitespace-only "
                "prompt received. Cannot generate from empty prompt."
            )
            # Return None for empty/whitespace prompts
            return (None, None) if return_logprobs else None

        # AIDEV-NOTE: Validate input length before generation
        try:
            self._validate_input_length(prompt, max_new_tokens)
        except ContextLengthExceededError as e:
            logger.error(f"Input validation failed: {e}")
            # Re-raise the exception to notify the caller
            raise

        try:
            # AIDEV-NOTE: Reset model cache before generation to ensure deterministic
            # behavior across multiple calls with the same seed
            if hasattr(self.model, "reset"):
                self.model.reset()

            final_prompt = prompt

            sampling_params = {**LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC}
            # AIDEV-NOTE: Handle custom eos_string by adding it to the stop sequences.
            if eos_string == "[EOS]":
                sampling_params["stop"] = DEFAULT_STOP_SEQUENCES
            else:
                # Combine default stop sequences with custom eos_string
                sampling_params["stop"] = DEFAULT_STOP_SEQUENCES + [eos_string]
            # Always use DEFAULT_SEED for determinism
            sampling_params["seed"] = seed

            if return_logprobs:
                # Request logprobs for each generated token
                sampling_params["logprobs"] = GENERATION_MAX_NEW_TOKENS

            if max_new_tokens is not None:
                sampling_params["max_tokens"] = max_new_tokens

            # AIDEV-NOTE: Use create_chat_completion for model interaction.
            output: Dict[str, Any] = self.model.create_chat_completion(
                messages=[{"role": "user", "content": final_prompt}], **sampling_params
            )

            generated_text = ""
            logprobs = None
            if output and "choices" in output and len(output["choices"]) > 0:
                choice = output["choices"][0]
                # AIDEV-NOTE: The model response structure for chat completion may vary.
                # Check for 'text' or 'message.content'.
                if "text" in choice and choice["text"] is not None:  # noqa E501
                    generated_text = choice["text"].strip()  # noqa E501
                elif (
                    "message" in choice
                    and "content" in choice["message"]
                    and choice["message"]["content"] is not None
                ):
                    generated_text = choice["message"]["content"].strip()
                if return_logprobs:
                    logprobs = choice.get("logprobs", None)

            if not generated_text:
                logger.warning(
                    f"DeterministicGenerator.generate: Model returned empty or "
                    f"whitespace-only text for prompt: '{prompt[:50]}...'"
                )

            # Only cache non-logprobs results for default model
            if should_use_cache_for_generation(return_logprobs, repo_id, filename):
                cache_key = generate_cache_key(prompt, eos_string)
                get_generation_cache().set(cache_key, generated_text)

            return (generated_text, logprobs) if return_logprobs else generated_text

        except Exception as e:
            logger.error(
                f"DeterministicGenerator.generate: Error during text generation "
                f"for prompt '{prompt[:50]}...': {e}",
                exc_info=True,
            )
            fallback_output = ""
            return (fallback_output, None) if return_logprobs else fallback_output

    def generate_iter(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        eos_string: str = "[EOS]",
        include_logprobs: bool = False,
        model: Optional[str] = None,
        model_repo: Optional[str] = None,
        model_filename: Optional[str] = None,
        size: Optional[str] = None,
        seed: int = DEFAULT_SEED,
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """Generate text iteratively, yielding tokens as they are produced.

        AIDEV-NOTE: Streaming generation that falls back to word-by-word yielding from the deterministic fallback.

        Args:
            prompt: The input prompt to generate from
            eos_string: Custom end-of-sequence string. "[EOS]" means use model's default.
            include_logprobs: If True, yield dicts with token and logprob info
            model: Model name from registry (e.g., "qwen2.5-3b")
            model_repo: Custom Hugging Face repository ID
            model_filename: Custom model filename
            size: Size identifier ("small", "large")
            seed: Seed for deterministic generation


        """
        validate_seed(seed)
        set_deterministic_environment(seed)
        if not isinstance(prompt, str):
            logger.error(
                f"DeterministicGenerator.generate_iter: Invalid prompt type: {type(prompt)}. Expected str."
            )
            # Return early for invalid input
            return

        # AIDEV-NOTE: Check cache first for non-logprobs requests using default model
        # This ensures streaming benefits from caching like non-streaming mode
        if should_use_cache_for_streaming(
            include_logprobs, model, model_repo, model_filename, size
        ):
            cache_key = generate_cache_key(prompt, eos_string)
            cached = get_generation_cache().get(cache_key)
            if cached is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"DeterministicGenerator.generate_iter: Cache hit for prompt: {str(prompt)[:50]}..."
                    )
                # Simulate streaming by yielding cached text in chunks
                # AIDEV-NOTE: Simulate streaming from cache using the same chunking logic as live streaming for consistency.
                words = cached.split()
                char_index = 0
                for i, word in enumerate(words):
                    # Find the word in the original text to preserve exact spacing
                    word_start = cached.find(word, char_index)
                    if word_start > char_index:
                        # Yield any whitespace before the word
                        yield cached[char_index:word_start]
                    # Yield the word
                    yield word
                    char_index = word_start + len(word)

                # Yield any remaining content (trailing whitespace)
                if char_index < len(cached):
                    yield cached[char_index:]
                return

        # Resolve model parameters
        repo_id: Optional[str] = None
        filename: Optional[str] = None

        if model or model_repo or model_filename or size:
            try:
                repo_id, filename = resolve_model_params(
                    model, model_repo, model_filename, size
                )
            except ValueError as e:
                logger.error(f"Invalid model specification: {e}")
                # Yield words from fallback
                fallback_text = _deterministic_fallback_generate(prompt, seed)
                words = fallback_text.split()
                for i, word in enumerate(words):
                    if include_logprobs:
                        # AIDEV-NOTE: Fallback returns None logprobs for compatibility
                        yield {
                            "token": word + (" " if i < len(words) - 1 else ""),
                            "logprobs": None,
                        }
                    else:
                        yield word + (" " if i < len(words) - 1 else "")

                # Cache fallback result for non-logprobs requests with default model
                if should_use_cache_for_streaming(
                    include_logprobs, model, model_repo, model_filename, size
                ):
                    cache_key = generate_cache_key(prompt, eos_string)
                    get_generation_cache().set(cache_key, fallback_text)
                return

        # Check if we need to load a different model
        model_key = f"{repo_id or 'default'}::{filename or 'default'}"
        needs_different_model = model_key != self._current_model_key

        # Load appropriate model if needed
        if needs_different_model or (include_logprobs and not self._logits_enabled):
            logger.info(f"Loading model {model_key} with logits={include_logprobs}")
            self._load_model(
                enable_logits=include_logprobs,
                repo_id=repo_id,
                filename=filename,
                force_reload=False,
            )

        # AIDEV-NOTE: Return early if model is not loaded
        skip_model_load = os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1"
        if self.model is None or skip_model_load:
            if skip_model_load:
                logger.error(
                    "DeterministicGenerator.generate_iter: STEADYTEXT_SKIP_MODEL_LOAD=1. "
                    "Model loading is disabled."
                )
            else:
                logger.error(
                    "DeterministicGenerator.generate_iter: Model not loaded. "
                    "Cannot generate text."
                )
            # Return empty iterator
            return

        if not prompt or not prompt.strip():
            logger.error(
                "DeterministicGenerator.generate_iter: Empty or whitespace-only "
                "prompt received. Cannot generate from empty prompt."
            )
            # Return empty iterator
            return

        # AIDEV-NOTE: Validate input length before streaming generation
        try:
            self._validate_input_length(prompt, max_new_tokens)
        except ContextLengthExceededError as e:
            logger.error(f"Input validation failed: {e}")
            # Re-raise the exception to notify the caller
            raise

        try:
            # AIDEV-NOTE: Reset model cache before generation
            if hasattr(self.model, "reset"):
                self.model.reset()

            final_prompt = prompt

            sampling_params = {**LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC}
            # AIDEV-NOTE: Handle custom eos_string for streaming generation
            if eos_string == "[EOS]":
                sampling_params["stop"] = DEFAULT_STOP_SEQUENCES
            else:
                sampling_params["stop"] = DEFAULT_STOP_SEQUENCES + [eos_string]
            sampling_params["seed"] = seed
            sampling_params["stream"] = True  # Enable streaming

            if include_logprobs:
                # Request logprobs for streaming
                sampling_params["logprobs"] = GENERATION_MAX_NEW_TOKENS

            if max_new_tokens is not None:
                sampling_params["max_tokens"] = max_new_tokens

            # AIDEV-NOTE: Streaming API returns an iterator of partial outputs
            stream = self.model.create_chat_completion(
                messages=[{"role": "user", "content": final_prompt}], **sampling_params
            )

            # AIDEV-NOTE: Collect tokens for processing and caching
            should_cache = (
                not include_logprobs
                and model is None
                and model_repo is None
                and model_filename is None
                and size is None
            )

            # AIDEV-NOTE: For non-logprobs requests, yield tokens immediately and handle caching after.
            if not include_logprobs:
                # AIDEV-NOTE: The `stream` iterator is consumed here. We need to handle caching
                # of the complete text after the loop.
                full_text_list = []
                for chunk in stream:
                    token = None
                    if chunk and "choices" in chunk and len(chunk["choices"]) > 0:
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})

                        if "content" in delta and delta["content"]:
                            token = delta["content"]
                        elif "text" in choice and choice["text"]:
                            token = choice["text"]

                        if token is not None:
                            full_text_list.append(token)
                            yield token

                # Join the collected tokens to form the complete text for caching
                complete_text = "".join(full_text_list)

                # Cache the full response if eligible
                if should_cache and complete_text:
                    cache_key = generate_cache_key(prompt, eos_string)
                    get_generation_cache().set(cache_key, complete_text)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"DeterministicGenerator.generate_iter: Cached result for prompt: {str(prompt)[:50]}..."
                        )

            else:
                # For logprobs requests, yield tokens immediately without cleaning
                for chunk in stream:
                    if chunk and "choices" in chunk and len(chunk["choices"]) > 0:
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})

                        # Yield dict with token and logprob info when requested
                        token_info = {}
                        if "content" in delta and delta["content"]:
                            token_info["token"] = delta["content"]
                        elif "text" in choice and choice["text"]:
                            token_info["token"] = choice["text"]

                        if "logprobs" in choice:
                            token_info["logprobs"] = choice["logprobs"]

                        if "token" in token_info:
                            yield token_info

        except Exception as e:
            logger.error(
                f"DeterministicGenerator.generate_iter: Error during streaming generation "
                f"for prompt '{prompt[:50]}...': {e}",
                exc_info=True,
            )
            # On error, don't yield anything further


# AIDEV-NOTE: Export validation function for use in structured generation
def _validate_input_length(
    model, prompt: str, max_new_tokens: Optional[int] = None
) -> None:
    """Validate input length against model's context window.

    This is a module-level function to allow use from structured.py.
    """
    generator = _get_generator_instance()
    if model is not None:
        # Temporarily set the model for validation
        old_model = generator.model
        generator.model = model
        try:
            generator._validate_input_length(prompt, max_new_tokens)
        finally:
            generator.model = old_model
    else:
        generator._validate_input_length(prompt, max_new_tokens)


# AIDEV-NOTE: DEPRECATED - The deterministic fallback generator has been disabled.
# The system now returns None when models are unavailable instead of generating
# deterministic but meaningless text. This change was made because the fallback
# was causing more confusion than it was solving.
#
# Original implementation preserved below for reference:
# A complex, hash-based fallback generator for deterministic output when the model is unavailable.
def _deterministic_fallback_generate(prompt: str, seed: int = DEFAULT_SEED) -> str:
    # Ensure prompt_for_hash is always a string, even if original prompt was not.
    if not isinstance(prompt, str) or not prompt.strip():
        prompt_for_hash = f"invalid_prompt_type_or_empty:{type(prompt).__name__}"
        logger.warning(
            f"Fallback generator: Invalid or empty prompt type received "
            f"({type(prompt).__name__}). Using placeholder for hash: "
            f"'{prompt_for_hash}'"
        )
    else:
        prompt_for_hash = prompt

    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "and",
        "a",
        "in",
        "it",
        "is",
        "to",
        "that",
        "this",
        "was",
        "for",
        "on",
        "at",
        "as",
        "by",
        "an",
        "be",
        "with",
        "if",
        "then",
        "else",
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliett",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
        "x-ray",
        "yankee",
        "zulu",
        "error",
        "fallback",
        "deterministic",
        "output",
        "generated",
        "text",
        "response",
        "steady",
        "system",
        "mode",
        "token",
        "sequence",
        "placeholder",
        "content",
        "reliable",
        "consistent",
        "predictable",
        "algorithmic",
        "data",
        "model",
        "layer",
    ]

    hasher = hashlib.sha256(prompt_for_hash.encode("utf-8"))
    hex_digest = hasher.hexdigest()  # Example: '50d858e0985ecc7f60418aaf0cc5ab58...'

    seed1 = int(hex_digest[:8], 16) ^ seed
    seed2 = int(hex_digest[8:16], 16) ^ seed
    seed3 = int(hex_digest[16:24], 16) ^ seed

    try:
        max_tokens_target = GENERATION_MAX_NEW_TOKENS
    except NameError:
        max_tokens_target = 100
        logger.warning("GENERATION_MAX_NEW_TOKENS not found, fallback using 100.")

    num_words_to_generate = (seed3 % 21) + (max_tokens_target - 10)
    num_words_to_generate = max(1, num_words_to_generate)

    fallback_text_parts: List[str] = []

    current_seed = seed1
    for i in range(num_words_to_generate):
        index_val = (current_seed >> (i % 16)) ^ (seed2 + i)
        index = index_val % len(words)
        fallback_text_parts.append(words[index])

        current_seed = (current_seed * 1664525 + seed2 + 1013904223 + i) & 0xFFFFFFFF
        seed2 = (seed2 * 22695477 + current_seed + 1 + i) & 0xFFFFFFFF

    return " ".join(fallback_text_parts)


# AIDEV-NOTE: DEPRECATED - See note above _deterministic_fallback_generate
def _deterministic_fallback_generate_iter(
    prompt: str, seed: int = DEFAULT_SEED
) -> Iterator[str]:
    """DEPRECATED: Iterative version of deterministic fallback that yields words one by one.

    AIDEV-NOTE: Used by generate_iter when the model is unavailable. Yields the same output as _deterministic_fallback_generate but word by word.
    """
    fallback_text = _deterministic_fallback_generate(prompt, seed)
    for word in fallback_text.split():
        yield word + " "


# AIDEV-NOTE: A module-level singleton generator instance for backward compatibility, made lazy to prevent model loading during import.
_generator_instance: Optional[DeterministicGenerator] = None


def _get_generator_instance() -> DeterministicGenerator:
    """Get or create the singleton generator instance.

    AIDEV-NOTE: Lazy initialization prevents model loading at import time, fixing pytest collection hangs.
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = DeterministicGenerator()
    # AIDEV-NOTE: Cast since we know it's not None after initialization
    return cast(DeterministicGenerator, _generator_instance)


def core_generate(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    return_logprobs: bool = False,
    eos_string: str = "[EOS]",
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None,
    seed: int = DEFAULT_SEED,
    response_format: Optional[Dict[str, Any]] = None,
    schema: Optional[Union[Dict[str, Any], Type, object]] = None,
    regex: Optional[str] = None,
    choices: Optional[List[str]] = None,
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]], None, Tuple[None, None]]:
    """Generate text deterministically with optional model switching and structured output.

    This is the main public API for text generation. It maintains backward
    compatibility while adding support for dynamic model switching and structured output.

    Args:
        prompt: Input text prompt
        max_new_tokens: The maximum number of new tokens to generate.
        return_logprobs: Whether to return token log probabilities
        eos_string: End-of-sequence string ("[EOS]" uses model defaults)
        model: Model name from registry (e.g., "gemma-3n-2b", "gemma-3n-4b")
        model_repo: Custom Hugging Face repository ID
        model_filename: Custom model filename
        size: Size identifier ("small", "large")
        seed: Seed for deterministic generation
        response_format: Dict specifying output format (e.g., {"type": "json_object"})
        schema: JSON schema, Pydantic model, or Python type for structured output
        regex: Regular expression pattern for output format
        choices: List of allowed string choices for output

    Returns:
        Generated text string, or tuple of (text, logprobs) if return_logprobs=True

    Examples:
        # Use default model (large/4B)
        text = generate("Hello, world!")

        # Use size parameter
        text = generate("Quick response", size="small")  # Uses Gemma-3n-2B
        text = generate("Complex task", size="large")    # Uses Gemma-3n-4B

        # Use a model from the registry
        text = generate("Explain quantum computing", model="gemma-3n-2b")

        # Use a custom model
        text = generate(
            "Write a poem",
            model_repo="ggml-org/gemma-3n-E4B-it-GGUF",
            model_filename="gemma-3n-E4B-it-Q8_0.gguf"
        )

        # Generate JSON with schema
        from pydantic import BaseModel
        class Person(BaseModel):
            name: str
            age: int

        result = generate("Create a person", schema=Person)
        # Returns: "Let me create...<json-output>{"name": "John", "age": 30}</json-output>"

        # Generate with regex pattern
        phone = generate("My phone number is", regex=r"\d{3}-\d{3}-\d{4}")

        # Generate with choices
        answer = generate("Is Python good?", choices=["yes", "no", "maybe"])

    AIDEV-NOTE: Model switching allows using different models without changing environment variables. Models are cached after the first load.
    AIDEV-NOTE: Structured output parameters enable JSON, regex, and choice-constrained generation.
    """
    return _get_generator_instance().generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        return_logprobs=return_logprobs,
        eos_string=eos_string,
        model=model,
        model_repo=model_repo,
        model_filename=model_filename,
        size=size,
        seed=seed,
        response_format=response_format,
        schema=schema,
        regex=regex,
        choices=choices,
    )


def core_generate_iter(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    eos_string: str = "[EOS]",
    include_logprobs: bool = False,
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None,
    seed: int = DEFAULT_SEED,
) -> Iterator[Union[str, Dict[str, Any]]]:
    """Generate text iteratively with optional model switching.

    Yields tokens as they are generated, enabling real-time streaming output.

    Args:
        prompt: Input text prompt
        max_new_tokens: The maximum number of new tokens to generate.
        eos_string: End-of-sequence string ("[EOS]" uses model defaults)
        include_logprobs: Whether to include log probabilities in output
        model: Model name from registry (e.g., "gemma-3n-2b")
        model_repo: Custom Hugging Face repository ID
        model_filename: Custom model filename
        size: Size identifier ("small", "large")
        seed: Seed for deterministic generation

    Yields:
        String tokens, or dicts with 'token' and 'logprobs' if include_logprobs=True

    AIDEV-NOTE: Streaming generation with model switching support. Falls back to word-by-word yielding from the deterministic fallback.
    """
    return _get_generator_instance().generate_iter(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        eos_string=eos_string,
        include_logprobs=include_logprobs,
        model=model,
        model_repo=model_repo,
        model_filename=model_filename,
        size=size,
        seed=seed,
    )

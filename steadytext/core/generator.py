# AIDEV-NOTE: Core text generation module with deterministic fallback mechanism
# Implements both model-based generation and hash-based deterministic fallback
# AIDEV-NOTE: Fixed fallback behavior - generator now calls
# _deterministic_fallback_generate() when model is None
# AIDEV-NOTE: Added stop sequences integration - DEFAULT_STOP_SEQUENCES
# are now passed to model calls
# AIDEV-NOTE: Fixed determinism issue - now always uses DEFAULT_SEED when
# no explicit seed is provided to ensure consistent outputs
# AIDEV-NOTE: Added generate_iter() method for streaming token generation
# with graceful fallback to word-by-word yielding when model unavailable
# AIDEV-NOTE: Added dynamic model switching (v1.0.0) - generate() and generate_iter()
# now accept model parameters to switch between different models at runtime while
# maintaining deterministic outputs for each model

import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator

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
)

# Ensure environment is set for determinism when this module is loaded
set_deterministic_environment(DEFAULT_SEED)


# AIDEV-NOTE: Use centralized cache manager for consistent caching across daemon and direct access
# AIDEV-NOTE: Cache is now shared between all components and properly centralized
# AIDEV-QUESTION: Should fallback results be cached, or only model-generated ones?


# AIDEV-NOTE: Main generator class with model instance caching and error handling
# AIDEV-NOTE: Extended to support dynamic model switching via generate() parameters
class DeterministicGenerator:
    def __init__(self):
        self.model = None
        self._logits_enabled = False
        self._current_model_key = "default::default"
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

        AIDEV-NOTE: Now supports loading custom models via repo_id and filename.
        AIDEV-NOTE: Respects STEADYTEXT_SKIP_MODEL_LOAD for test environments.
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
        if self.model is None:
            logger.error(
                f"DeterministicGenerator: Model instance is None after attempting to load {self._current_model_key}."
            )

    def generate(
        self,
        prompt: str,
        return_logprobs: bool = False,
        eos_string: str = "[EOS]",
        model: Optional[str] = None,
        model_repo: Optional[str] = None,
        model_filename: Optional[str] = None,
        size: Optional[str] = None,
        thinking_mode: bool = False,
    ) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]:
        """Generate text with optional model switching.

        Args:
            prompt: Input text prompt
            return_logprobs: Whether to return token log probabilities
            eos_string: End-of-sequence string ("[EOS]" uses model defaults)
            model: Model name from registry (e.g., "qwen2.5-3b")
            model_repo: Custom Hugging Face repository ID
            model_filename: Custom model filename
            size: Size identifier ("small", "medium", "large")
            thinking_mode: Enable Qwen3 thinking mode (default: False appends /no_think, True appends /think)

        AIDEV-NOTE: Model switching parameters allow using different models without
        restarting. Precedence: model_repo/model_filename > model > size > env vars > defaults.
        AIDEV-NOTE: For Qwen3 models, thinking_mode=False (default) appends '/no_think' to prevent
        verbose reasoning output. Set thinking_mode=True to append '/think' and see the model's thinking process.
        """
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
                fallback = _deterministic_fallback_generate(prompt)
                return (fallback, None) if return_logprobs else fallback

        # Handle caching only for non-logprobs requests and default model
        if not return_logprobs and repo_id is None and filename is None:
            prompt_str = prompt if isinstance(prompt, str) else str(prompt)
            # Include eos_string in cache key if it's not the default
            cache_key = (
                prompt_str
                if eos_string == "[EOS]"
                else f"{prompt_str}::EOS::{eos_string}"
            )
            cached = get_generation_cache().get(cache_key)
            if cached is not None:
                return cached

        if not isinstance(prompt, str):
            logger.error(
                f"DeterministicGenerator.generate: Invalid prompt type: {type(prompt)}. Expected str. Using fallback."
            )
            # Pass string representation to fallback
            fallback = _deterministic_fallback_generate(str(prompt))
            return (fallback, None) if return_logprobs else fallback

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

        # AIDEV-NOTE: This is where the fallback to _deterministic_fallback_generate occurs if the model isn't loaded.
        # Also check for STEADYTEXT_SKIP_MODEL_LOAD environment variable
        if self.model is None or os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
                logger.debug(
                    "DeterministicGenerator.generate: STEADYTEXT_SKIP_MODEL_LOAD=1. "
                    "Using fallback generator."
                )
            else:
                logger.warning(
                    "DeterministicGenerator.generate: Model not loaded. "
                    "Using fallback generator."
                )
            fallback = _deterministic_fallback_generate(prompt)

            # Cache fallback result for non-logprobs requests and default model
            if not return_logprobs and repo_id is None and filename is None:
                prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                # Include eos_string in cache key if it's not the default
                cache_key = (
                    prompt_str
                    if eos_string == "[EOS]"
                    else f"{prompt_str}::EOS::{eos_string}"
                )
                get_generation_cache().set(cache_key, fallback)

            return (fallback, None) if return_logprobs else fallback

        if not prompt or not prompt.strip():  # Check after ensuring prompt is a string
            logger.warning(
                "DeterministicGenerator.generate: Empty or whitespace-only "
                "prompt received. Using fallback generator."
            )
            # Call fallback for empty/whitespace
            fallback = _deterministic_fallback_generate(prompt)

            # Cache fallback result for non-logprobs requests and default model
            if not return_logprobs and repo_id is None and filename is None:
                prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                # Include eos_string in cache key if it's not the default
                cache_key = (
                    prompt_str
                    if eos_string == "[EOS]"
                    else f"{prompt_str}::EOS::{eos_string}"
                )
                get_generation_cache().set(cache_key, fallback)

            return (fallback, None) if return_logprobs else fallback

        try:
            # AIDEV-NOTE: Reset model cache before generation to ensure deterministic
            # behavior across multiple calls with the same seed
            if hasattr(self.model, "reset"):
                self.model.reset()

            # AIDEV-NOTE: Add /think to prompt for Qwen3 thinking mode when enabled,
            # otherwise add /no_think to disable thinking mode (default)
            final_prompt = prompt
            if thinking_mode:
                final_prompt = prompt + " /think"
            else:
                final_prompt = prompt + " /no_think"

            sampling_params = {**LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC}
            # AIDEV-NOTE: Handle custom eos_string - if it's "[EOS]", use default stop sequences
            # Otherwise, add the custom eos_string to stop sequences
            if eos_string == "[EOS]":
                sampling_params["stop"] = DEFAULT_STOP_SEQUENCES
            else:
                # Combine default stop sequences with custom eos_string
                sampling_params["stop"] = DEFAULT_STOP_SEQUENCES + [eos_string]
            # Always use DEFAULT_SEED for determinism
            sampling_params["seed"] = DEFAULT_SEED

            if return_logprobs:
                # Request logprobs for each generated token
                sampling_params["logprobs"] = GENERATION_MAX_NEW_TOKENS

            # AIDEV-NOTE: Use create_chat_completion for model interaction.
            output: Dict[str, Any] = self.model.create_chat_completion(
                messages=[{"role": "user", "content": final_prompt}], **sampling_params
            )

            generated_text = ""
            logprobs = None
            if output and "choices" in output and len(output["choices"]) > 0:
                choice = output["choices"][0]
                # AIDEV-NOTE: Model response structure for chat completion may vary.
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
                    logprobs = choice.get("logprobs")

            if not generated_text:
                logger.warning(
                    f"DeterministicGenerator.generate: Model returned empty or "
                    f"whitespace-only text for prompt: '{prompt[:50]}...'"
                )

            # AIDEV-NOTE: Strip empty or whitespace-only <think></think> tags from output
            # This handles cases where think tags contain only whitespace/newlines
            generated_text = re.sub(
                r"<think>\s*</think>\s*",
                "",
                generated_text,
                flags=re.MULTILINE | re.DOTALL,
            )

            # Only cache non-logprobs results for default model
            if not return_logprobs and repo_id is None and filename is None:
                prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                # Include eos_string in cache key if it's not the default
                cache_key = (
                    prompt_str
                    if eos_string == "[EOS]"
                    else f"{prompt_str}::EOS::{eos_string}"
                )
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
        eos_string: str = "[EOS]",
        include_logprobs: bool = False,
        model: Optional[str] = None,
        model_repo: Optional[str] = None,
        model_filename: Optional[str] = None,
        size: Optional[str] = None,
        thinking_mode: bool = False,
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """Generate text iteratively, yielding tokens as they are produced.

        AIDEV-NOTE: Streaming generation for real-time output. Falls back to
        yielding words from deterministic fallback when model unavailable.

        Args:
            prompt: The input prompt to generate from
            eos_string: Custom end-of-sequence string. "[EOS]" means use model's default.
            include_logprobs: If True, yield dicts with token and logprob info
            model: Model name from registry (e.g., "qwen2.5-3b")
            model_repo: Custom Hugging Face repository ID
            model_filename: Custom model filename
            size: Size identifier ("small", "medium", "large")
            thinking_mode: Enable Qwen3 thinking mode (default: False appends /no_think, True appends /think)

        AIDEV-NOTE: When running in pytest, collecting many tokens (>400) can cause
        hanging due to interaction between streaming API and pytest's output capture.
        Tests should limit token collection to avoid this issue. Works fine outside pytest.
        """
        if not isinstance(prompt, str):
            logger.error(
                f"DeterministicGenerator.generate_iter: Invalid prompt type: {type(prompt)}. Expected str. Using fallback."
            )
            # Yield words from fallback
            fallback_text = _deterministic_fallback_generate(str(prompt))
            words = fallback_text.split()
            for i, word in enumerate(words):
                if include_logprobs:
                    # AIDEV-NOTE: Fallback returns None logprobs for compatibility
                    yield {"token": word + (" " if i < len(words) - 1 else ""), "logprobs": None}
                else:
                    yield word + (" " if i < len(words) - 1 else "")
            
            # Cache fallback result for non-logprobs requests with default model
            if not include_logprobs and model is None and model_repo is None and model_filename is None and size is None:
                from ..cache_manager import get_generation_cache
                prompt_str = str(prompt)
                cache_key = prompt_str if eos_string == "[EOS]" else f"{prompt_str}::EOS::{eos_string}"
                get_generation_cache().set(cache_key, fallback_text)
            return

        # AIDEV-NOTE: Check cache first for non-logprobs requests using default model
        # This ensures streaming benefits from caching like non-streaming mode
        if (
            not include_logprobs
            and model is None
            and model_repo is None
            and model_filename is None
            and size is None
        ):
            from ..cache_manager import get_generation_cache

            prompt_str = prompt if isinstance(prompt, str) else str(prompt)
            cache_key = (
                prompt_str
                if eos_string == "[EOS]"
                else f"{prompt_str}::EOS::{eos_string}"
            )
            cached = get_generation_cache().get(cache_key)
            if cached is not None:
                logger.debug(
                    f"DeterministicGenerator.generate_iter: Cache hit for prompt: {prompt_str[:50]}..."
                )
                # Simulate streaming by yielding cached text in chunks
                # AIDEV-NOTE: Use same chunking logic as live streaming to ensure consistency
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
                fallback_text = _deterministic_fallback_generate(prompt)
                words = fallback_text.split()
                for i, word in enumerate(words):
                    if include_logprobs:
                        # AIDEV-NOTE: Fallback returns None logprobs for compatibility
                        yield {"token": word + (" " if i < len(words) - 1 else ""), "logprobs": None}
                    else:
                        yield word + (" " if i < len(words) - 1 else "")
                
                # Cache fallback result for non-logprobs requests with default model
                if not include_logprobs and model is None and model_repo is None and model_filename is None and size is None:
                    from ..cache_manager import get_generation_cache
                    prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                    cache_key = prompt_str if eos_string == "[EOS]" else f"{prompt_str}::EOS::{eos_string}"
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

        # AIDEV-NOTE: Check if model is loaded, fallback to word-by-word generation if not
        # Also check for STEADYTEXT_SKIP_MODEL_LOAD environment variable
        if self.model is None or os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
                logger.debug(
                    "DeterministicGenerator.generate_iter: STEADYTEXT_SKIP_MODEL_LOAD=1. "
                    "Using fallback generator."
                )
            else:
                logger.warning(
                    "DeterministicGenerator.generate_iter: Model not loaded. "
                    "Using fallback generator."
                )
            fallback_text = _deterministic_fallback_generate(prompt)
            words = fallback_text.split()
            for i, word in enumerate(words):
                if include_logprobs:
                    # AIDEV-NOTE: Fallback returns None logprobs for compatibility
                    yield {"token": word + (" " if i < len(words) - 1 else ""), "logprobs": None}
                else:
                    yield word + (" " if i < len(words) - 1 else "")
            
            # Cache fallback result for non-logprobs requests with default model
            if not include_logprobs and model is None and model_repo is None and model_filename is None and size is None:
                from ..cache_manager import get_generation_cache
                prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                cache_key = prompt_str if eos_string == "[EOS]" else f"{prompt_str}::EOS::{eos_string}"
                get_generation_cache().set(cache_key, fallback_text)
            return

        if not prompt or not prompt.strip():
            logger.warning(
                "DeterministicGenerator.generate_iter: Empty or whitespace-only "
                "prompt received. Using fallback generator."
            )
            fallback_text = _deterministic_fallback_generate(prompt)
            words = fallback_text.split()
            for i, word in enumerate(words):
                if include_logprobs:
                    # AIDEV-NOTE: Fallback returns None logprobs for compatibility
                    yield {"token": word + (" " if i < len(words) - 1 else ""), "logprobs": None}
                else:
                    yield word + (" " if i < len(words) - 1 else "")
            
            # Cache fallback result for non-logprobs requests with default model
            if not include_logprobs and model is None and model_repo is None and model_filename is None and size is None:
                from ..cache_manager import get_generation_cache
                prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                cache_key = prompt_str if eos_string == "[EOS]" else f"{prompt_str}::EOS::{eos_string}"
                get_generation_cache().set(cache_key, fallback_text)
            return

        try:
            # AIDEV-NOTE: Reset model cache before generation
            if hasattr(self.model, "reset"):
                self.model.reset()

            # AIDEV-NOTE: Add /think to prompt for Qwen3 thinking mode when enabled,
            # otherwise add /no_think to disable thinking mode (default)
            final_prompt = prompt
            if thinking_mode:
                final_prompt = prompt + " /think"
            else:
                final_prompt = prompt + " /no_think"

            sampling_params = {**LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC}
            # AIDEV-NOTE: Handle custom eos_string for streaming generation
            if eos_string == "[EOS]":
                sampling_params["stop"] = DEFAULT_STOP_SEQUENCES
            else:
                sampling_params["stop"] = DEFAULT_STOP_SEQUENCES + [eos_string]
            sampling_params["seed"] = DEFAULT_SEED
            sampling_params["stream"] = True  # Enable streaming

            if include_logprobs:
                # Request logprobs for streaming
                sampling_params["logprobs"] = GENERATION_MAX_NEW_TOKENS

            # AIDEV-NOTE: Streaming API returns an iterator of partial outputs
            stream = self.model.create_chat_completion(
                messages=[{"role": "user", "content": final_prompt}], **sampling_params
            )

            # AIDEV-NOTE: Collect tokens for processing and caching
            collected_tokens = []
            should_cache = (
                not include_logprobs
                and model is None
                and model_repo is None
                and model_filename is None
                and size is None
            )

            # AIDEV-NOTE: For non-logprobs requests, collect all tokens first to enable cleaning
            # This ensures consistency with non-streaming generate() function
            if not include_logprobs:
                for chunk in stream:
                    if chunk and "choices" in chunk and len(chunk["choices"]) > 0:
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})

                        # Normal string token collection
                        token = None
                        if "content" in delta and delta["content"]:
                            token = delta["content"]
                        elif "text" in choice and choice["text"]:
                            token = choice["text"]

                        if token is not None:
                            collected_tokens.append(token)

                # Apply same think tag cleaning as non-streaming generate()
                complete_text = "".join(collected_tokens)
                cleaned_text = re.sub(
                    r"<think>\s*</think>\s*",
                    "",
                    complete_text,
                    flags=re.MULTILINE | re.DOTALL,
                )

                # Re-yield cleaned text in chunks to preserve exact content
                if cleaned_text:
                    # Yield in word-sized chunks to simulate token streaming
                    words = cleaned_text.split()
                    char_index = 0
                    for i, word in enumerate(words):
                        # Find the word in the original text to preserve exact spacing
                        word_start = cleaned_text.find(word, char_index)
                        if word_start > char_index:
                            # Yield any whitespace before the word
                            yield cleaned_text[char_index:word_start]
                        # Yield the word
                        yield word
                        char_index = word_start + len(word)

                    # Yield any remaining content (trailing whitespace)
                    if char_index < len(cleaned_text):
                        yield cleaned_text[char_index:]

                # Cache the cleaned text if eligible
                if should_cache and cleaned_text:
                    from ..cache_manager import get_generation_cache

                    prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                    cache_key = (
                        prompt_str
                        if eos_string == "[EOS]"
                        else f"{prompt_str}::EOS::{eos_string}"
                    )
                    get_generation_cache().set(cache_key, cleaned_text)
                    logger.debug(
                        f"DeterministicGenerator.generate_iter: Cached result for prompt: {prompt_str[:50]}..."
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


# AIDEV-NOTE: Complex hash-based fallback generation algorithm for
# deterministic output when model is unavailable - uses multiple hash seeds
# for word selection
# AIDEV-NOTE: This is the hash-based fallback mechanism.
def _deterministic_fallback_generate(prompt: str) -> str:
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

    seed1 = int(hex_digest[:8], 16)
    seed2 = int(hex_digest[8:16], 16)
    seed3 = int(hex_digest[16:24], 16)

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


def _deterministic_fallback_generate_iter(prompt: str) -> Iterator[str]:
    """Iterative version of deterministic fallback that yields words one by one.

    AIDEV-NOTE: Used by generate_iter when model is unavailable. Yields the same
    deterministic output as _deterministic_fallback_generate but word by word.
    """
    fallback_text = _deterministic_fallback_generate(prompt)
    for word in fallback_text.split():
        yield word + " "


# AIDEV-NOTE: Module-level singleton generator instance for backward compatibility
_generator_instance = DeterministicGenerator()


def generate(
    prompt: str,
    return_logprobs: bool = False,
    eos_string: str = "[EOS]",
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None,
    thinking_mode: bool = False,
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]:
    """Generate text deterministically with optional model switching.

    This is the main public API for text generation. It maintains backward
    compatibility while adding support for dynamic model switching.

    Args:
        prompt: Input text prompt
        return_logprobs: Whether to return token log probabilities
        eos_string: End-of-sequence string ("[EOS]" uses model defaults)
        model: Model name from registry (e.g., "qwen2.5-3b", "qwen3-8b")
        model_repo: Custom Hugging Face repository ID (e.g., "Qwen/Qwen2.5-3B-Instruct-GGUF")
        model_filename: Custom model filename (e.g., "qwen2.5-3b-instruct-q8_0.gguf")
        size: Size identifier ("small", "medium", "large") - maps to Qwen3 0.6B/1.7B/4B models
        thinking_mode: Enable Qwen3 thinking mode (default: False appends /no_think)

    Returns:
        Generated text string, or tuple of (text, logprobs) if return_logprobs=True

    Examples:
        # Use default model (medium/1.7B)
        text = generate("Hello, world!")

        # Use size parameter
        text = generate("Quick response", size="small")  # Uses Qwen3-0.6B
        text = generate("Complex task", size="large")    # Uses Qwen3-4B

        # Use a model from the registry
        text = generate("Explain quantum computing", model="qwen2.5-3b")

        # Use a custom model
        text = generate(
            "Write a poem",
            model_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
            model_filename="qwen2.5-7b-instruct-q8_0.gguf"
        )

    AIDEV-NOTE: Model switching allows using different models without changing
    environment variables. Models are cached after first load for efficiency.
    The size parameter provides convenient access to Qwen3 models of different sizes.
    """
    return _generator_instance.generate(
        prompt=prompt,
        return_logprobs=return_logprobs,
        eos_string=eos_string,
        model=model,
        model_repo=model_repo,
        model_filename=model_filename,
        size=size,
        thinking_mode=thinking_mode,
    )


def generate_iter(
    prompt: str,
    eos_string: str = "[EOS]",
    include_logprobs: bool = False,
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None,
    thinking_mode: bool = False,
) -> Iterator[Union[str, Dict[str, Any]]]:
    """Generate text iteratively with optional model switching.

    Yields tokens as they are generated, enabling real-time streaming output.

    Args:
        prompt: Input text prompt
        eos_string: End-of-sequence string ("[EOS]" uses model defaults)
        include_logprobs: Whether to include log probabilities in output
        model: Model name from registry (e.g., "qwen2.5-3b")
        model_repo: Custom Hugging Face repository ID
        model_filename: Custom model filename
        size: Size identifier ("small", "medium", "large") - maps to Qwen3 0.6B/1.7B/4B models
        thinking_mode: Enable Qwen3 thinking mode (default: False appends /no_think)

    Yields:
        String tokens, or dicts with 'token' and 'logprobs' if include_logprobs=True

    AIDEV-NOTE: Streaming generation with model switching support. Falls back
    to word-by-word yielding from deterministic fallback when model unavailable.
    """
    return _generator_instance.generate_iter(
        prompt=prompt,
        eos_string=eos_string,
        include_logprobs=include_logprobs,
        model=model,
        model_repo=model_repo,
        model_filename=model_filename,
        size=size,
        thinking_mode=thinking_mode,
    )

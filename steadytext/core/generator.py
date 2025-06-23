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

import hashlib
import os as _os
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator

from ..disk_backed_frecency_cache import DiskBackedFrecencyCache
from ..models.loader import get_generator_model_instance
from ..utils import set_deterministic_environment  # Assuming this is in utils.py
from ..utils import (
    DEFAULT_SEED,
    DEFAULT_STOP_SEQUENCES,
    GENERATION_MAX_NEW_TOKENS,
    LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC,
    logger,
)

# Ensure environment is set for determinism when this module is loaded
set_deterministic_environment(DEFAULT_SEED)


# AIDEV-NOTE: Disk-backed frecency cache for generated text results
# AIDEV-NOTE: Persistent cache for successful generation outputs with configurable limits
# AIDEV-QUESTION: Should fallback results be cached, or only model-generated ones?
# AIDEV-NOTE: Cache capacity and size can be configured via environment variables:
# - STEADYTEXT_GENERATION_CACHE_CAPACITY (default: 256)
# - STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB (default: 50.0)
_generation_cache = DiskBackedFrecencyCache(
    capacity=int(_os.environ.get("STEADYTEXT_GENERATION_CACHE_CAPACITY", "256")),
    cache_name="generation_cache",
    max_size_mb=float(
        _os.environ.get("STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB", "50.0")
    ),
)


# AIDEV-NOTE: Main generator class with model instance caching and error handling
class DeterministicGenerator:
    def __init__(self):
        self.model = None
        self._logits_enabled = False
        # Load model without logits_all initially
        self._load_model(enable_logits=False)

    def _load_model(self, enable_logits: bool = False):
        """Load or reload the model with specific logits configuration."""
        self.model = get_generator_model_instance(
            force_reload=True, enable_logits=enable_logits
        )
        self._logits_enabled = enable_logits
        if self.model is None:
            logger.error(
                "DeterministicGenerator: Model instance is None after attempting to load."
            )

    def generate(
        self, prompt: str, return_logprobs: bool = False, eos_string: str = "[EOS]"
    ) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]:
        # Handle caching only for non-logprobs requests
        if not return_logprobs:
            prompt_str = prompt if isinstance(prompt, str) else str(prompt)
            # Include eos_string in cache key if it's not the default
            cache_key = (
                prompt_str
                if eos_string == "[EOS]"
                else f"{prompt_str}::EOS::{eos_string}"
            )
            cached = _generation_cache.get(cache_key)
            if cached is not None:
                return cached

        if not isinstance(prompt, str):
            logger.error(
                f"DeterministicGenerator.generate: Invalid prompt type: {type(prompt)}. Expected str. Using fallback."
            )
            # Pass string representation to fallback
            fallback = _deterministic_fallback_generate(str(prompt))
            return (fallback, None) if return_logprobs else fallback

        # Reload model if logprobs requested but not enabled
        if return_logprobs and not self._logits_enabled:
            logger.info("Reloading model with logits support for logprobs generation.")
            self._load_model(enable_logits=True)
        elif not return_logprobs and self._logits_enabled:
            # Optionally reload without logits to save memory/performance
            # For now, we'll keep the model loaded with logits if already enabled
            pass

        # AIDEV-NOTE: This is where the fallback to _deterministic_fallback_generate occurs if the model isn't loaded.
        if self.model is None:
            logger.warning(
                "DeterministicGenerator.generate: Model not loaded. "
                "Using fallback generator."
            )
            fallback = _deterministic_fallback_generate(prompt)
            return (fallback, None) if return_logprobs else fallback

        if not prompt or not prompt.strip():  # Check after ensuring prompt is a string
            logger.warning(
                "DeterministicGenerator.generate: Empty or whitespace-only "
                "prompt received. Using fallback generator."
            )
            # Call fallback for empty/whitespace
            fallback = _deterministic_fallback_generate(prompt)
            return (fallback, None) if return_logprobs else fallback

        try:
            # AIDEV-NOTE: Reset model cache before generation to ensure deterministic
            # behavior across multiple calls with the same seed
            if hasattr(self.model, "reset"):
                self.model.reset()

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
                messages=[{"role": "user", "content": prompt}], **sampling_params
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

            # Only cache non-logprobs results
            if not return_logprobs:
                prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                # Include eos_string in cache key if it's not the default
                cache_key = (
                    prompt_str
                    if eos_string == "[EOS]"
                    else f"{prompt_str}::EOS::{eos_string}"
                )
                _generation_cache.set(cache_key, generated_text)

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
        self, prompt: str, eos_string: str = "[EOS]", include_logprobs: bool = False
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """Generate text iteratively, yielding tokens as they are produced.

        AIDEV-NOTE: Streaming generation for real-time output. Falls back to
        yielding words from deterministic fallback when model unavailable.

        Args:
            prompt: The input prompt to generate from
            eos_string: Custom end-of-sequence string. "[EOS]" means use model's default.
            include_logprobs: If True, yield dicts with token and logprob info

        AIDEV-TODO: Investigate potential hanging issues in pytest when multiple
        generate_iter calls are made in the same session. Works fine in isolation
        but may have state/threading issues in test environment.
        """
        if not isinstance(prompt, str):
            logger.error(
                f"DeterministicGenerator.generate_iter: Invalid prompt type: {type(prompt)}. Expected str. Using fallback."
            )
            # Yield words from fallback
            fallback_text = _deterministic_fallback_generate(str(prompt))
            for word in fallback_text.split():
                yield word + " "
            return

        # Reload model if logprobs requested but not enabled
        if include_logprobs and not self._logits_enabled:
            logger.info("Reloading model with logits support for logprobs generation.")
            self._load_model(enable_logits=True)

        # AIDEV-NOTE: Check if model is loaded, fallback to word-by-word generation if not
        if self.model is None:
            logger.warning(
                "DeterministicGenerator.generate_iter: Model not loaded. "
                "Using fallback generator."
            )
            fallback_text = _deterministic_fallback_generate(prompt)
            for word in fallback_text.split():
                yield word + " "
            return

        if not prompt or not prompt.strip():
            logger.warning(
                "DeterministicGenerator.generate_iter: Empty or whitespace-only "
                "prompt received. Using fallback generator."
            )
            fallback_text = _deterministic_fallback_generate(prompt)
            for word in fallback_text.split():
                yield word + " "
            return

        try:
            # AIDEV-NOTE: Reset model cache before generation
            if hasattr(self.model, "reset"):
                self.model.reset()

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
                messages=[{"role": "user", "content": prompt}], **sampling_params
            )

            for chunk in stream:
                if chunk and "choices" in chunk and len(chunk["choices"]) > 0:
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})

                    # AIDEV-NOTE: Streaming responses use 'delta' for incremental content
                    if include_logprobs:
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
                    else:
                        # Normal string yielding
                        if "content" in delta and delta["content"]:
                            yield delta["content"]
                        elif "text" in choice and choice["text"]:
                            yield choice["text"]

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

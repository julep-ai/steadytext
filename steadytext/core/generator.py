# AIDEV-NOTE: Core text generation module with deterministic fallback mechanism
# Implements both model-based generation and hash-based deterministic fallback
# AIDEV-NOTE: Fixed fallback behavior - generator now calls
# _deterministic_fallback_generate() when model is None
# AIDEV-NOTE: Added stop sequences integration - DEFAULT_STOP_SEQUENCES
# are now passed to model calls

import hashlib
from ..models.loader import get_generator_model_instance
from ..utils import (
    logger,
    LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC,
    GENERATION_MAX_NEW_TOKENS,
    DEFAULT_SEED,
    DEFAULT_STOP_SEQUENCES,
    set_deterministic_environment,  # Assuming this is in utils.py
)
from typing import List, Dict, Any, Optional

# Ensure environment is set for determinism when this module is loaded
set_deterministic_environment(DEFAULT_SEED)


# AIDEV-NOTE: Main generator class with model instance caching and error handling
class DeterministicGenerator:
    def __init__(self):
        self.model = get_generator_model_instance()
        if self.model is None:
            logger.error(
                "DeterministicGenerator: Model instance is None after attempting to load."
            )

    def generate(
        self, prompt: str, seed: Optional[int] = None, return_logprobs: bool = False
    ) -> Any:
        if not isinstance(prompt, str):
            logger.error(
                f"DeterministicGenerator.generate: Invalid prompt type: "
                f"{type(prompt)}. Expected str. Using fallback."
            )
            # Pass string representation to fallback
            fallback = _deterministic_fallback_generate(str(prompt))
            return (fallback, None) if return_logprobs else fallback

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
            sampling_params = {**LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC}
            sampling_params["stop"] = DEFAULT_STOP_SEQUENCES
            if seed is not None:
                sampling_params["seed"] = seed
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

            return (generated_text, logprobs) if return_logprobs else generated_text

        except Exception as e:
            logger.error(
                f"DeterministicGenerator.generate: Error during text generation "
                f"for prompt '{prompt[:50]}...': {e}",
                exc_info=True,
            )
            fallback_output = ""
            return (fallback_output, None) if return_logprobs else fallback_output


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

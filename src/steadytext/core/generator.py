import hashlib
from ..models.loader import get_generator_model_instance
from ..utils import (
    logger,
    LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC,
    GENERATION_MAX_NEW_TOKENS,
    DEFAULT_SEED,
    set_deterministic_environment # Assuming this is in utils.py
)
from typing import List, Dict, Any

# Ensure environment is set for determinism when this module is loaded
set_deterministic_environment(DEFAULT_SEED)

def _apply_qwen_chat_template(prompt: str) -> str:
    system_prompt_content = "You are a helpful assistant."
    return (
        f"<|im_start|>system\n{system_prompt_content}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

class DeterministicGenerator:
    def __init__(self):
        self.model = get_generator_model_instance()
        if self.model is None:
            logger.error("DeterministicGenerator: Model instance is None after attempting to load.")

    def generate(self, prompt: str) -> str:
        if self.model is None:
            logger.warning("DeterministicGenerator.generate: Model not loaded. Returning empty string from core generator.")
            return ""

        if not prompt or not prompt.strip():
            logger.warning("DeterministicGenerator.generate: Empty or whitespace-only prompt received. Returning empty string from core generator.")
            return ""

        formatted_prompt = _apply_qwen_chat_template(prompt)

        try:
            sampling_params = {**LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC}

            output: Dict[str, Any] = self.model(
                formatted_prompt,
                **sampling_params
            )

            generated_text = ""
            if output and "choices" in output and len(output["choices"]) > 0:
                choice = output["choices"][0]
                if "text" in choice and choice["text"] is not None:
                    generated_text = choice["text"].strip()

            if not generated_text:
                logger.warning(f"DeterministicGenerator.generate: Model returned empty or whitespace-only text for prompt: '{prompt[:50]}...'")

            return generated_text

        except Exception as e:
            logger.error(f"DeterministicGenerator.generate: Error during text generation for prompt '{prompt[:50]}...': {e}", exc_info=True)
            return ""

def _deterministic_fallback_generate(prompt: str) -> str:
    if not prompt or not prompt.strip():
        prompt_for_hash = "empty_prompt_placeholder_for_hash"
        logger.warning(f"Fallback generator: Empty prompt received, using placeholder for hash: '{prompt_for_hash}'")
    else:
        prompt_for_hash = prompt

    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "and", "a", "in", "it", "is", "to", "that", "this", "was", "for",
        "on", "at", "as", "by", "an", "be", "with", "if", "then", "else",
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliett", "kilo", "lima", "mike", "november",
        "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
        "victor", "whiskey", "x-ray", "yankee", "zulu", "error", "fallback",
        "deterministic", "output", "generated", "text", "response", "steady",
        "system", "mode", "token", "sequence", "placeholder", "content", "reliable",
        "consistent", "predictable", "algorithmic", "data", "model", "layer"
    ]

    hasher = hashlib.sha256(prompt_for_hash.encode('utf-8'))
    hex_digest = hasher.hexdigest()

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

# AIDEV-NOTE: Core reranking module for scoring query-document relevance using Qwen3-Reranker model.
# Features:
# - Deterministic scoring of query-document pairs
# - Batch processing for efficiency
# - Caching support for reranking results
# - Fallback to simple similarity scoring when model unavailable
# - Uses yes/no token logits for binary relevance judgments

import os
from typing import List, Optional, Tuple, Union

import numpy as np

from ..cache_manager import get_cache_manager
from ..models.loader import get_generator_model_instance
from ..utils import (
    DEFAULT_SEED,
    RERANKING_MODEL_REPO,
    RERANKING_MODEL_FILENAME,
    logger,
    set_deterministic_environment,
    validate_seed,
    get_optimal_context_window,
)
from ..exceptions import ContextLengthExceededError

# AIDEV-NOTE: Get the reranking cache from the cache manager
# This will be created on-demand when first accessed
# AIDEV-TODO: Consider adding support for cross-encoder models like ms-marco-MiniLM
# AIDEV-TODO: Add support for batch reranking with optimized model inference


# AIDEV-NOTE: Format the instruction for the reranker model
# Based on the pattern provided in the issue
def _format_reranking_instruction(
    task: str, query: str, document: str
) -> Tuple[str, int, int]:
    """Format query and document for reranking model.

    AIDEV-NOTE: Follows the specific format required by Qwen3-Reranker model
    with system prompt and special tokens.

    Args:
        task: Task description for the reranking (e.g., "Given a web search query, retrieve relevant passages")
        query: The query text
        document: The document text to score

    Returns:
        Tuple of (formatted_prompt, yes_token_pos, no_token_pos)
    """
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    # Format the instruction
    instruction = f"Instruct: {task}\nQuery: {query}\nDocument: {document}\n"
    full_prompt = prefix + instruction + suffix

    # Calculate token positions for yes/no (they should be at the end after the thinking)
    # AIDEV-NOTE: The model will generate either "yes" or "no" as the final token
    # We'll extract logits at the position right before these tokens
    yes_token_pos = -1  # Last token position for "yes"
    no_token_pos = -1  # Last token position for "no"

    return full_prompt, yes_token_pos, no_token_pos


# AIDEV-NOTE: Simple similarity fallback for when model is not available
def _fallback_rerank_score(
    query: str, document: str, seed: int = DEFAULT_SEED
) -> float:
    """Compute a deterministic fallback score based on word overlap.

    AIDEV-NOTE: This provides a simple but deterministic fallback when the model
    cannot be loaded. Uses normalized word overlap as a basic relevance measure.

    Args:
        query: Query text
        document: Document text
        seed: Seed for any randomness (not used but kept for consistency)

    Returns:
        Score between 0.0 and 1.0
    """
    # Simple word overlap score
    query_words = set(query.lower().split())
    doc_words = set(document.lower().split())

    if not query_words:
        return 0.0

    overlap = len(query_words.intersection(doc_words))
    score = overlap / len(query_words)

    return min(1.0, score)


# AIDEV-NOTE: Main reranker class following the pattern of DeterministicGenerator
class DeterministicReranker:
    def __init__(self) -> None:
        # AIDEV-NOTE: Set deterministic environment on initialization
        set_deterministic_environment(DEFAULT_SEED)

        self.model = None
        self._current_model_key = f"{RERANKING_MODEL_REPO}::{RERANKING_MODEL_FILENAME}"
        self._context_window = get_optimal_context_window(
            model_name="qwen3-reranker-4b"
        )
        self._yes_token_id: Optional[int] = None
        self._no_token_id: Optional[int] = None

        # AIDEV-NOTE: Skip model loading if STEADYTEXT_SKIP_MODEL_LOAD is set
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") != "1":
            self._load_model()

    def _load_model(
        self,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        force_reload: bool = False,
    ):
        """Load the reranking model.

        AIDEV-NOTE: Uses the centralized model loader with reranking-specific parameters.
        """
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            logger.debug(
                "_load_model: STEADYTEXT_SKIP_MODEL_LOAD=1, skipping model load"
            )
            self.model = None
            return

        # Use default reranking model if not specified
        if repo_id is None:
            repo_id = RERANKING_MODEL_REPO
        if filename is None:
            filename = RERANKING_MODEL_FILENAME

        # AIDEV-NOTE: Load with logits enabled for extracting yes/no token probabilities
        self.model = get_generator_model_instance(
            force_reload=force_reload,
            enable_logits=True,  # Need logits for yes/no scoring
            repo_id=repo_id,
            filename=filename,
        )

        self._current_model_key = f"{repo_id}::{filename}"

        # AIDEV-NOTE: Get token IDs for "yes" and "no" if model loaded successfully
        if self.model is not None:
            try:
                # Try to get tokenizer and convert tokens to IDs
                if hasattr(self.model, "tokenize"):
                    # Tokenize "yes" and "no" to get their IDs
                    yes_tokens = self.model.tokenize(
                        "yes".encode("utf-8"), add_bos=False
                    )
                    no_tokens = self.model.tokenize("no".encode("utf-8"), add_bos=False)

                    if yes_tokens and no_tokens:
                        self._yes_token_id = yes_tokens[0]
                        self._no_token_id = no_tokens[0]
                        logger.info(
                            f"Reranker initialized with yes_token_id={self._yes_token_id}, "
                            f"no_token_id={self._no_token_id}"
                        )
                    else:
                        logger.warning("Could not get token IDs for 'yes' and 'no'")
            except Exception as e:
                logger.warning(f"Error getting yes/no token IDs: {e}")
        else:
            logger.error(
                f"DeterministicReranker: Model instance is None after attempting to load {self._current_model_key}."
            )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer.

        AIDEV-NOTE: Reuses the same token counting logic as the generator.
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

    def _validate_input_length(self, prompt: str) -> None:
        """Validate that input prompt fits within context window.

        AIDEV-NOTE: Ensures the formatted reranking prompt fits within the model's context.
        """
        if self._context_window is None:
            # If we don't know the context window, we can't validate
            return

        # Count input tokens
        input_tokens = self._count_tokens(prompt)

        # For reranking, we only need space for yes/no output (minimal)
        output_reserve = 10  # Just need space for "yes" or "no"

        # Calculate available tokens for input (leave 10% margin for safety)
        safety_margin = int(self._context_window * 0.1)
        available_tokens = self._context_window - output_reserve - safety_margin

        if input_tokens > available_tokens:
            raise ContextLengthExceededError(
                input_tokens=input_tokens,
                max_tokens=available_tokens,
                input_text=prompt,
                message=(
                    f"Reranking input is too long: {input_tokens} tokens. "
                    f"Maximum allowed: {available_tokens} tokens "
                    f"(context window: {self._context_window}, "
                    f"reserved for output: {output_reserve}, "
                    f"safety margin: {safety_margin})"
                ),
            )

    def rerank(
        self,
        query: str,
        documents: Union[str, List[str]],
        task: str = "Given a web search query, retrieve relevant passages that answer the query",
        return_scores: bool = True,
        seed: int = DEFAULT_SEED,
    ) -> Union[List[Tuple[str, float]], List[str]]:
        """Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: Single document or list of documents to rerank
            task: Task description for the reranking
            return_scores: If True, return (document, score) tuples; if False, just return sorted documents
            seed: Random seed for determinism

        Returns:
            If return_scores=True: List of (document, score) tuples sorted by score descending
            If return_scores=False: List of documents sorted by relevance descending

        AIDEV-NOTE: Returns empty list on errors to maintain "Never Fails" principle.
        """
        validate_seed(seed)
        set_deterministic_environment(seed)

        # Normalize input to list
        if isinstance(documents, str):
            documents = [documents]

        if not documents:
            return []

        # Check cache
        cache_manager = get_cache_manager()
        cache = cache_manager.get_reranking_cache()

        results = []

        for doc in documents:
            # Create cache key
            cache_key = (query, doc, task)

            # Try to get from cache
            score = None
            if cache is not None:
                try:
                    cached = cache.get(cache_key)
                    if cached is not None:
                        score = cached
                        logger.debug(
                            f"Reranking cache hit for query='{query[:50]}...', doc='{doc[:50]}...'"
                        )
                except Exception as e:
                    logger.warning(f"Error accessing reranking cache: {e}")

            # If not in cache, compute score
            if score is None:
                if (
                    self.model is None
                    or self._yes_token_id is None
                    or self._no_token_id is None
                ):
                    # Use fallback scoring
                    logger.debug(
                        "Using fallback reranking due to missing model or token IDs"
                    )
                    score = _fallback_rerank_score(query, doc, seed)
                else:
                    # Format the prompt
                    prompt, _, _ = _format_reranking_instruction(task, query, doc)

                    # Validate input length
                    try:
                        self._validate_input_length(prompt)
                    except ContextLengthExceededError as e:
                        logger.warning(f"Document too long for reranking: {e}")
                        score = 0.0  # Assign low score to documents that don't fit
                        results.append((doc, score))
                        continue

                    try:
                        # AIDEV-NOTE: Generate with the model to get logits
                        # We use max_tokens=1 since we only need the first token (yes/no)
                        response = self.model(
                            prompt,
                            max_tokens=1,
                            temperature=0.0,
                            seed=seed,
                            logprobs=True,
                            top_logprobs=100,  # Get top 100 to ensure we capture yes/no
                        )

                        if response and "choices" in response and response["choices"]:
                            choice = response["choices"][0]
                            if "logprobs" in choice and choice["logprobs"]:
                                # Get the logprobs for the generated token
                                token_logprobs = choice["logprobs"]["top_logprobs"][0]

                                # Find logprobs for yes/no tokens
                                yes_logprob = -float("inf")
                                no_logprob = -float("inf")

                                for token_id, logprob in token_logprobs.items():
                                    if int(token_id) == self._yes_token_id:
                                        yes_logprob = logprob
                                    elif int(token_id) == self._no_token_id:
                                        no_logprob = logprob

                                # Convert log probabilities to probabilities
                                yes_prob = np.exp(yes_logprob)
                                no_prob = np.exp(no_logprob)

                                # Normalize to get a score between 0 and 1
                                total_prob = yes_prob + no_prob
                                if total_prob > 0:
                                    score = yes_prob / total_prob
                                else:
                                    score = 0.5  # Default to neutral if both are 0

                                logger.debug(
                                    f"Reranking score: {score:.3f} "
                                    f"(yes_prob={yes_prob:.3f}, no_prob={no_prob:.3f})"
                                )
                            else:
                                logger.warning(
                                    "No logprobs in model response, using fallback"
                                )
                                score = _fallback_rerank_score(query, doc, seed)
                        else:
                            logger.warning(
                                "Invalid model response format, using fallback"
                            )
                            score = _fallback_rerank_score(query, doc, seed)

                    except Exception as e:
                        logger.error(f"Error during reranking: {e}")
                        score = _fallback_rerank_score(query, doc, seed)

                # Cache the result
                if cache is not None and score is not None:
                    try:
                        cache.put(cache_key, score)
                    except Exception as e:
                        logger.warning(f"Error caching reranking result: {e}")

            results.append((doc, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Return based on return_scores flag
        if return_scores:
            return results
        else:
            return [doc for doc, _ in results]


# AIDEV-NOTE: Global instance for convenient access
_reranker_instance: Optional[DeterministicReranker] = None


def get_reranker() -> DeterministicReranker:
    """Get or create the global reranker instance.

    AIDEV-NOTE: Follows the singleton pattern used by other components.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = DeterministicReranker()
    return _reranker_instance


# AIDEV-NOTE: Core reranking function that wraps the reranker class
def core_rerank(
    query: str,
    documents: Union[str, List[str]],
    task: str = "Given a web search query, retrieve relevant passages that answer the query",
    return_scores: bool = True,
    seed: int = DEFAULT_SEED,
) -> Union[List[Tuple[str, float]], List[str]]:
    """Core reranking function.

    See DeterministicReranker.rerank for documentation.

    AIDEV-NOTE: This provides a functional interface to the reranker,
    similar to how core_generate and core_embed work.
    """
    reranker = get_reranker()
    return reranker.rerank(
        query=query,
        documents=documents,
        task=task,
        return_scores=return_scores,
        seed=seed,
    )

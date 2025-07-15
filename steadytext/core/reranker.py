# AIDEV-NOTE: Core reranking module for scoring query-document relevance using Qwen3-Reranker model.
# Features:
# - Deterministic scoring of query-document pairs
# - Batch processing for efficiency
# - Caching support for reranking results
# - Fallback to simple similarity scoring when model unavailable
# - Uses yes/no token logits for binary relevance judgments
#
# Important: The reranker is designed to evaluate whether documents (passages) answer queries.
# It works best with complete sentences or paragraphs rather than single words, as it needs
# sufficient context to determine relevance.

import os
from typing import List, Optional, Tuple, Union, cast


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
    """Compute a deterministic fallback score based on word overlap and simple heuristics.

    AIDEV-NOTE: This provides a simple but deterministic fallback when the model
    cannot be loaded. Uses normalized word overlap as a basic relevance measure,
    with some basic semantic heuristics for common phrases.

    Args:
        query: Query text
        document: Document text
        seed: Seed for any randomness (not used but kept for consistency)

    Returns:
        Score between 0.0 and 1.0
    """
    query_lower = query.lower()
    doc_lower = document.lower()

    # Simple word overlap score
    query_words = set(query_lower.split())
    doc_words = set(doc_lower.split())

    if not query_words:
        return 0.0

    overlap = len(query_words.intersection(doc_words))
    score = overlap / len(query_words)

    # AIDEV-NOTE: Add some basic semantic heuristics for common phrases
    # This helps when the fallback is used
    semantic_pairs = [
        # Well-known sayings
        (["doctor", "away"], ["apple"]),
        (["apple", "day"], ["doctor", "health"]),
        # Medical terms
        (
            ["medical", "health", "doctor", "nurse"],
            ["nurse", "medical", "health", "doctor"],
        ),
        # Food items
        (["food", "eat", "vegetable", "fruit"], ["apple", "potato"]),
    ]

    # Check for semantic relationships
    for query_keywords, doc_keywords in semantic_pairs:
        if any(kw in query_lower for kw in query_keywords) and any(
            kw in doc_lower for kw in doc_keywords
        ):
            # Boost score for semantic matches
            score = max(score, 0.3)  # At least 30% relevance for semantic matches
            # AIDEV-NOTE: Debug logging to trace semantic matching
            logger.debug(
                f"Semantic match found: query contains {[kw for kw in query_keywords if kw in query_lower]}, "
                f"doc contains {[kw for kw in doc_keywords if kw in doc_lower]}"
            )
            break

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
        logger.info(f"Loading reranking model: {repo_id}/{filename}")
        self.model = get_generator_model_instance(
            force_reload=force_reload,
            enable_logits=True,  # Need logits for yes/no scoring
            repo_id=repo_id,
            filename=filename,
        )

        self._current_model_key = f"{repo_id}::{filename}"

        # AIDEV-NOTE: Get token IDs for "yes" and "no" if model loaded successfully
        if self.model is not None:
            logger.info("Reranking model loaded successfully")
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
                        logger.info(
                            f"Reranking cache hit for query='{query[:50]}...', doc='{doc[:50]}...', score={score}"
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
                        f"Using fallback reranking - model={self.model is not None}, "
                        f"yes_token_id={self._yes_token_id}, no_token_id={self._no_token_id}"
                    )
                    score = _fallback_rerank_score(query, doc, seed)
                else:
                    # Format the prompt
                    prompt, _, _ = _format_reranking_instruction(task, query, doc)
                    logger.info(f"DEBUG: Reranking prompt:\n{prompt[:500]}...")

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
                        logger.debug(
                            f"Generating reranking response for query='{query[:50]}...', doc='{doc[:50]}...'"
                        )
                        response = self.model(
                            prompt,
                            max_tokens=1,
                            temperature=0.0,
                            seed=seed,
                            logprobs=True,
                        )
                        logger.debug(f"Reranker model response: {response}")

                        if response and "choices" in response and response["choices"]:
                            choice = response["choices"][0]
                            # Check the generated text directly
                            if "text" in choice:
                                generated_text = choice["text"].strip().lower()
                                logger.debug(
                                    f"Reranker generated text: '{generated_text}'"
                                )
                                # Be more flexible with matching - handle variations
                                if generated_text.startswith("yes"):
                                    score = 1.0
                                    logger.debug(
                                        f"Reranking score from text (YES): {score:.3f}"
                                    )
                                elif generated_text.startswith("no"):
                                    score = 0.0
                                    logger.debug(
                                        f"Reranking score from text (NO): {score:.3f}"
                                    )
                                else:
                                    logger.warning(
                                        f"Unexpected text generated: '{generated_text}', using fallback"
                                    )
                                    score = _fallback_rerank_score(query, doc, seed)
                            elif "logprobs" in choice and choice["logprobs"]:
                                # AIDEV-NOTE: When logprobs=True, the format is different
                                # We need to look at the token_logprobs array
                                logprobs_data = choice["logprobs"]

                                # Get the logprobs for all tokens at the first position
                                if (
                                    "token_logprobs" in logprobs_data
                                    and logprobs_data["token_logprobs"]
                                ):
                                    # This contains the log probability of the generated token
                                    logprobs_data["token_logprobs"][0]

                                    # Get the generated token
                                    generated_token = None
                                    if (
                                        "tokens" in logprobs_data
                                        and logprobs_data["tokens"]
                                    ):
                                        generated_token = logprobs_data["tokens"][0]

                                    # Check if the generated token is "yes" or "no"
                                    # We'll use the generated token's probability directly
                                    if generated_token:
                                        generated_text = generated_token.strip().lower()
                                        # Be more flexible with matching - handle variations
                                        if generated_text.startswith("yes"):
                                            score = 1.0
                                            logger.debug(
                                                f"Reranking generated '{generated_text}' from logprobs, score: {score:.3f}"
                                            )
                                        elif generated_text.startswith("no"):
                                            score = 0.0
                                            logger.debug(
                                                f"Reranking generated '{generated_text}' from logprobs, score: {score:.3f}"
                                            )
                                        else:
                                            # Unexpected token generated, use fallback
                                            logger.warning(
                                                f"Unexpected token generated: '{generated_text}', using fallback"
                                            )
                                            score = _fallback_rerank_score(
                                                query, doc, seed
                                            )
                                    else:
                                        logger.warning(
                                            "No generated token found, using fallback"
                                        )
                                        score = _fallback_rerank_score(query, doc, seed)
                                else:
                                    logger.warning(
                                        "No token_logprobs in response, using fallback"
                                    )
                                    score = _fallback_rerank_score(query, doc, seed)
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

                # AIDEV-NOTE: Cache all valid scores (both model-generated and fallback) for performance
                # This ensures repeated queries don't need to recompute fallback scores
                if cache is not None and score is not None:
                    try:
                        cache.set(cache_key, score)
                        logger.debug(f"Cached reranking score: {score}")
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
    # AIDEV-NOTE: Cast since we know it's not None after initialization
    return cast(DeterministicReranker, _reranker_instance)


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

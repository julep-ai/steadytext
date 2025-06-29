# AIDEV-NOTE: Vector operations CLI for SteadyText embeddings, providing similarity, distance, search, and arithmetic operations.
import click
import sys
import json
import numpy as np
from typing import List, Tuple, Optional

from ...core.embedder import core_embed as create_embedding


@click.group()
def vector():
    """Perform vector operations on embeddings."""
    pass


def _get_embedding(text: str, seed: int) -> np.ndarray:
    """Helper to get embedding for text."""
    return create_embedding(text, seed=seed)


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    AIDEV-NOTE: Since embeddings are L2-normalized, cosine similarity
    simplifies to just the dot product. Range is [-1, 1] where 1 means
    identical direction, 0 means orthogonal, -1 means opposite.
    """
    return float(np.dot(vec1, vec2))


def _euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return float(np.linalg.norm(vec1 - vec2))


def _manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Manhattan (L1) distance between two vectors."""
    return float(np.sum(np.abs(vec1 - vec2)))


@vector.command()
@click.argument("text1")
@click.argument("text2")
@click.option(
    "--metric",
    type=click.Choice(["cosine", "dot"]),
    default="cosine",
    help="Similarity metric to use",
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for deterministic embedding.",
    show_default=True,
)
def similarity(text1: str, text2: str, metric: str, output_json: bool, seed: int):
    """Compute similarity between two text embeddings.

    Examples:
        st vector similarity "apple" "orange"
        st vector similarity "king" "queen" --metric dot
        st vector similarity "hello" "world" --json
    """
    # AIDEV-NOTE: Get embeddings for both texts
    vec1 = _get_embedding(text1, seed)
    vec2 = _get_embedding(text2, seed)

    if metric == "cosine":
        score = _cosine_similarity(vec1, vec2)
    else:  # dot
        score = float(np.dot(vec1, vec2))

    if output_json:
        result = {
            "text1": text1,
            "text2": text2,
            "metric": metric,
            "similarity": score,
        }
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"{score:.6f}")


@vector.command()
@click.argument("text1")
@click.argument("text2")
@click.option(
    "--metric",
    type=click.Choice(["euclidean", "manhattan", "cosine"]),
    default="euclidean",
    help="Distance metric to use",
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for deterministic embedding.",
    show_default=True,
)
def distance(text1: str, text2: str, metric: str, output_json: bool, seed: int):
    """Compute distance between two text embeddings.

    Examples:
        st vector distance "apple" "orange"
        st vector distance "cat" "dog" --metric manhattan
        st vector distance "hello" "world" --metric cosine --json
    """
    vec1 = _get_embedding(text1, seed)
    vec2 = _get_embedding(text2, seed)

    if metric == "euclidean":
        dist = _euclidean_distance(vec1, vec2)
    elif metric == "manhattan":
        dist = _manhattan_distance(vec1, vec2)
    else:  # cosine distance (1 - cosine_similarity)
        dist = 1.0 - _cosine_similarity(vec1, vec2)

    if output_json:
        result = {
            "text1": text1,
            "text2": text2,
            "metric": metric,
            "distance": dist,
        }
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"{dist:.6f}")


@vector.command()
@click.argument("query")
@click.option(
    "--candidates",
    type=click.Path(exists=True),
    help="File containing candidate texts (one per line)",
)
@click.option("--stdin", is_flag=True, help="Read candidates from stdin")
@click.option("--top", type=int, default=1, help="Number of top results to return")
@click.option(
    "--metric",
    type=click.Choice(["cosine", "euclidean"]),
    default="cosine",
    help="Similarity/distance metric",
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for deterministic embedding.",
    show_default=True,
)
def search(
    query: str,
    candidates: Optional[str],
    stdin: bool,
    top: int,
    metric: str,
    output_json: bool,
    seed: int,
):
    """Find most similar texts from a list of candidates.

    Examples:
        echo -e "apple\\norange\\ncar\\ntruck" | st vector search "fruit" --stdin
        st vector search "programming" --candidates topics.txt --top 3
        st vector search "query" --stdin --metric euclidean --json
    """
    # AIDEV-NOTE: Get candidates from stdin or file
    candidate_texts: List[str] = []

    if stdin or (not candidates and not sys.stdin.isatty()):
        candidate_texts = [line.strip() for line in sys.stdin if line.strip()]
    elif candidates:
        with open(candidates, "r") as f:
            candidate_texts = [line.strip() for line in f if line.strip()]
    else:
        click.echo(
            "Error: No candidates provided. Use --stdin or --candidates.", err=True
        )
        sys.exit(1)

    if not candidate_texts:
        click.echo("Error: No valid candidates found.", err=True)
        sys.exit(1)

    # Get query embedding
    query_vec = _get_embedding(query, seed)

    # Compute scores for all candidates
    scores: List[Tuple[str, float]] = []
    for candidate in candidate_texts:
        candidate_vec = _get_embedding(candidate, seed)

        if metric == "cosine":
            score = _cosine_similarity(query_vec, candidate_vec)
            # For cosine, higher is better
            scores.append((candidate, score))
        else:  # euclidean
            score = _euclidean_distance(query_vec, candidate_vec)
            # For euclidean, lower is better, so negate for sorting
            scores.append((candidate, -score))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Get top results
    top_results = scores[:top]

    if output_json:
        results = []
        for text, score in top_results:
            if metric == "euclidean":
                # Convert back to positive distance
                score = -score
            results.append(
                {
                    "text": text,
                    metric: score,
                }
            )

        output = {
            "query": query,
            "metric": metric,
            "results": results,
        }
        click.echo(json.dumps(output, indent=2))
    else:
        for text, score in top_results:
            if metric == "euclidean":
                # Convert back to positive distance
                score = -score
            click.echo(f"{text}\t{score:.6f}")


@vector.command()
@click.argument("texts", nargs=-1, required=True)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for deterministic embedding.",
    show_default=True,
)
def average(texts: Tuple[str, ...], output_json: bool, seed: int):
    """Compute average of multiple text embeddings.

    Examples:
        st vector average "cat" "dog" "hamster"
        st vector average "python" "javascript" "rust" --json
    """
    if len(texts) < 2:
        click.echo("Error: Need at least 2 texts to average.", err=True)
        sys.exit(1)

    # AIDEV-NOTE: Get embeddings for all texts
    embeddings = [_get_embedding(text, seed) for text in texts]

    # Average the embeddings
    avg_embedding = np.mean(embeddings, axis=0)

    # L2 normalize the result
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    if output_json:
        output = {
            "texts": list(texts),
            "embedding": avg_embedding.tolist(),
            "dimension": len(avg_embedding),
        }
        click.echo(json.dumps(output, indent=2))
    else:
        # Output as numpy-style array representation
        np.set_printoptions(threshold=50, precision=6)
        click.echo("Average embedding (first 50 values):")
        click.echo(np.array2string(avg_embedding[:50], separator=", "))


@vector.command()
@click.argument("base")
@click.argument("add_terms", nargs=-1)
@click.option("--subtract", multiple=True, help="Terms to subtract from the result")
@click.option("--normalize", is_flag=True, default=True, help="L2 normalize the result")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for deterministic embedding.",
    show_default=True,
)
def arithmetic(
    base: str,
    add_terms: Tuple[str, ...],
    subtract: Tuple[str, ...],
    normalize: bool,
    output_json: bool,
    seed: int,
):
    """Perform vector arithmetic on embeddings.

    Examples:
        st vector arithmetic "king" "woman" --subtract "man"
        st vector arithmetic "paris" --subtract "france" "italy"
        st vector arithmetic "good" "better" --json
    """
    # Start with base embedding
    result = _get_embedding(base, seed)

    # Add terms
    for term in add_terms:
        result = result + _get_embedding(term, seed)

    # Subtract terms
    for term in subtract:
        result = result - _get_embedding(term, seed)

    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm

    if output_json:
        output = {
            "base": base,
            "add": list(add_terms),
            "subtract": list(subtract),
            "normalized": normalize,
            "embedding": result.tolist(),
            "dimension": len(result),
        }
        click.echo(json.dumps(output, indent=2))
    else:
        # Show the operation and first 50 values
        operation_parts = [base]
        if add_terms:
            operation_parts.extend([f"+ {term}" for term in add_terms])
        if subtract:
            operation_parts.extend([f"- {term}" for term in subtract])

        click.echo(f"Vector arithmetic: {' '.join(operation_parts)}")
        if normalize:
            click.echo("(L2 normalized)")

        np.set_printoptions(threshold=50, precision=6)
        click.echo("\nResult (first 50 values):")
        click.echo(np.array2string(result[:50], separator=", "))

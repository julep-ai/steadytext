import click
import json
import sys
from typing import List


# AIDEV-NOTE: CLI command for reranking documents based on query relevance
# Uses the Qwen3-Reranker model to score query-document pairs
@click.command()
@click.argument("query")
@click.argument("documents", nargs=-1)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--scores/--no-scores", default=True, help="Include relevance scores in output"
)
@click.option(
    "--task",
    default="Given a web search query, retrieve relevant passages that answer the query",
    help="Task description for reranking",
)
@click.option(
    "--top-k",
    type=int,
    default=None,
    help="Return only top K documents",
)
@click.option(
    "--file",
    "doc_file",
    type=click.Path(exists=True),
    help="Read documents from file (one per line)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for deterministic reranking",
    show_default=True,
)
def rerank(query, documents, output_json, scores, task, top_k, doc_file, seed):
    """Rerank documents by relevance to a query.

    The QUERY is the search query to rank documents against.

    DOCUMENTS can be provided as arguments or read from stdin/file.

    Examples:
        st rerank "What is Python?" "Python is a programming language" "Snakes are reptiles"

        st rerank "climate change" --file documents.txt

        echo -e "Doc 1\\nDoc 2\\nDoc 3" | st rerank "my query"

        st rerank "medical symptoms" "doc1" "doc2" --task "Find relevant medical information"

        st rerank "search query" "doc1" "doc2" "doc3" --top-k 2 --json
    """
    import time
    from ... import rerank as do_rerank

    # Collect documents from various sources
    doc_list: List[str] = []

    # From arguments
    if documents:
        doc_list.extend(documents)

    # From file
    if doc_file:
        with open(doc_file, "r") as f:
            file_docs = [line.strip() for line in f if line.strip()]
            doc_list.extend(file_docs)

    # From stdin if no documents provided yet
    if not doc_list and not sys.stdin.isatty():
        stdin_docs = [line.strip() for line in sys.stdin if line.strip()]
        doc_list.extend(stdin_docs)

    if not doc_list:
        click.echo(
            "Error: No documents provided. Use 'st rerank --help' for usage.", err=True
        )
        sys.exit(1)

    # Perform reranking
    start_time = time.time()
    results = do_rerank(
        query=query,
        documents=doc_list,
        task=task,
        return_scores=scores,
        seed=seed,
    )
    elapsed_time = time.time() - start_time

    # Apply top-k filtering if requested
    if top_k and top_k > 0:
        results = results[:top_k]

    # Format output
    if output_json:
        # JSON output with metadata
        if scores:
            # Results are List[Tuple[str, float]]
            ranked_docs = [
                {"document": doc, "score": score, "rank": i + 1}
                for i, (doc, score) in enumerate(results)
            ]
        else:
            # Results are List[str]
            ranked_docs = [
                {"document": doc, "rank": i + 1} for i, doc in enumerate(results)
            ]

        output = {
            "query": query,
            "task": task,
            "documents": ranked_docs,
            "model": "Qwen3-Reranker-4B",
            "total_documents": len(doc_list),
            "returned_documents": len(results),
            "time_taken": elapsed_time,
        }
        click.echo(json.dumps(output, indent=2))
    else:
        # Simple text output
        if scores:
            # Output with scores
            for i, (doc, score) in enumerate(results):
                click.echo(f"{i + 1}. [{score:.4f}] {doc}")
        else:
            # Output without scores
            for i, doc in enumerate(results):
                click.echo(f"{i + 1}. {doc}")

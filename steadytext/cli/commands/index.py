# AIDEV-NOTE: Index command for creating and managing FAISS indices
# Uses chonkie for deterministic text chunking and faiss-cpu for vector storage

import click
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import faiss
except ImportError:
    click.echo("Error: faiss-cpu not installed. Run: pip install faiss-cpu", err=True)
    raise

try:
    from chonkie import TokenChunker
except ImportError:
    click.echo("Error: chonkie not installed. Run: pip install chonkie", err=True)
    raise

from ...core.embedder import core_embed as create_embedding
from ...core.generator import DeterministicGenerator
from ...utils import get_cache_dir, logger
from ...disk_backed_frecency_cache import DiskBackedFrecencyCache

# AIDEV-NOTE: Cache for index search results to ensure deterministic retrieval
_index_search_cache = DiskBackedFrecencyCache(
    capacity=256,
    cache_name="index_search_cache",
    max_size_mb=50.0,
)


def _get_indices_dir() -> Path:
    """Get the directory for storing FAISS indices."""
    indices_dir = get_cache_dir() / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    return indices_dir


def _hash_file_content(file_path: Path) -> str:
    """Generate deterministic hash of file content."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _chunk_text_deterministically(
    text: str, max_chunk_size: int = 512, chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Chunk text using chonkie with deterministic tokenization.

    AIDEV-NOTE: We need to use the same tokenizer as the model for consistency.
    For now, we'll use a simple character-based chunking as fallback.
    """
    try:
        # Try to use the model's tokenizer for accurate token counting
        generator = DeterministicGenerator()
        if hasattr(generator, "model") and generator.model is not None:
            # Use model's tokenizer if available
            chunker = TokenChunker(
                tokenizer=generator.model.tokenizer(),
                chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = chunker.chunk(text)
            return [
                {"text": chunk.text, "start": chunk.start_index, "end": chunk.end_index}
                for chunk in chunks
            ]
    except Exception as e:
        logger.warning(f"Could not use model tokenizer for chunking: {e}")

    # AIDEV-NOTE: Improved offline tokenizer-based chunking using word tokenization
    # This provides better accuracy than character-based approximation
    return _word_based_chunking(text, max_chunk_size, chunk_overlap)


def _word_based_chunking(
    text: str, max_chunk_size: int = 512, chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Word-based chunking that approximates tokenization without requiring a model.

    AIDEV-NOTE: Uses simple word splitting with punctuation handling.
    Estimates tokens as: words + punctuation marks + 10% for subword tokens.
    """
    import re

    # Split text into words and punctuation, preserving positions
    token_pattern = re.compile(r"\w+|[^\w\s]")
    tokens = []
    for match in token_pattern.finditer(text):
        tokens.append(
            {
                "text": match.group(),
                "start": int(match.start()),
                "end": int(match.end()),
            }
        )

    if not tokens:
        return [{"text": text, "start": 0, "end": len(text)}]

    chunks = []
    chunk_start_idx = 0

    while chunk_start_idx < len(tokens):
        # Determine chunk end based on token count
        chunk_end_idx = min(chunk_start_idx + max_chunk_size, len(tokens))

        # Adjust for subword tokens (add 10% buffer)
        adjusted_size = int(max_chunk_size * 0.9)
        if chunk_end_idx - chunk_start_idx > adjusted_size:
            chunk_end_idx = chunk_start_idx + adjusted_size

        # Extract chunk text and positions
        chunk_tokens = tokens[chunk_start_idx:chunk_end_idx]
        if chunk_tokens:
            chunk_text_start: int = chunk_tokens[0]["start"]
            chunk_text_end: int = chunk_tokens[-1]["end"]
            chunk_text = text[chunk_text_start:chunk_text_end]

            chunks.append(
                {"text": chunk_text, "start": chunk_text_start, "end": chunk_text_end}
            )

        # Move to next chunk with overlap
        if chunk_end_idx >= len(tokens):
            # Last chunk, we're done
            break
        chunk_start_idx = max(chunk_end_idx - chunk_overlap, chunk_start_idx + 1)

    return chunks


@click.group()
def index():
    """Manage FAISS indices for vector search."""
    pass


@index.command()
@click.argument("input_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default="default.faiss", help="Output index filename")
@click.option("--chunk-size", default=512, help="Maximum tokens per chunk")
@click.option("--chunk-overlap", default=50, help="Token overlap between chunks")
@click.option("--force", is_flag=True, help="Overwrite existing index")
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for deterministic embedding.",
    show_default=True,
)
def create(
    input_files: Tuple[str, ...],
    output: str,
    chunk_size: int,
    chunk_overlap: int,
    force: bool,
    seed: int,
):
    """Create a FAISS index from text files.

    Examples:
        st index create document.txt --output my_index.faiss
        st index create *.txt --output project.faiss --chunk-size 256
    """
    indices_dir = _get_indices_dir()
    index_path = indices_dir / output
    meta_path = indices_dir / f"{output}.meta"

    if index_path.exists() and not force:
        click.echo(
            f"Error: Index {index_path} already exists. Use --force to overwrite.",
            err=True,
        )
        return

    all_chunks = []
    file_metadata = {}

    # Process each input file
    for file_path in input_files:
        path = Path(file_path)
        click.echo(f"Processing {path}...")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            click.echo(f"Error reading {path}: {e}", err=True)
            continue

        # Generate file hash for tracking
        file_hash = _hash_file_content(path)
        file_metadata[str(path)] = {
            "hash": file_hash,
            "size": len(content),
            "path": str(path.absolute()),
        }

        # Chunk the text
        chunks = _chunk_text_deterministically(content, chunk_size, chunk_overlap)

        # Add source file info to each chunk
        for chunk in chunks:
            chunk["source_file"] = str(path)
            chunk["file_hash"] = file_hash
            all_chunks.append(chunk)

        click.echo(f"  Created {len(chunks)} chunks")

    if not all_chunks:
        click.echo("Error: No chunks created from input files.", err=True)
        return

    click.echo(f"\nTotal chunks: {len(all_chunks)}")
    click.echo("Generating embeddings...")

    # Generate embeddings for all chunks
    embeddings = []
    for i, chunk in enumerate(all_chunks):
        if i % 10 == 0:
            click.echo(f"  Processing chunk {i + 1}/{len(all_chunks)}...", err=True)

        embedding = create_embedding(chunk["text"], seed=seed)
        embeddings.append(embedding)

    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Create FAISS index
    # AIDEV-NOTE: Using IndexFlatL2 for exact search (deterministic)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    # AIDEV-NOTE: Add embeddings to index - FAISS expects float32 arrays
    index.add(embeddings_array)  # type: ignore[call-arg]

    # Save index
    faiss.write_index(index, str(index_path))

    # Save metadata
    metadata = {
        "version": "1.0",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(all_chunks),
        "dimension": dimension,
        "chunks": all_chunks,
        "files": file_metadata,
    }

    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    click.echo("\nIndex created successfully:")
    click.echo(f"  Index file: {index_path}")
    click.echo(f"  Metadata: {meta_path}")
    click.echo(f"  Chunks: {len(all_chunks)}")
    click.echo(f"  Dimension: {dimension}")


@index.command()
@click.argument("index_file", type=click.Path(exists=True))
def info(index_file: str):
    """Show information about a FAISS index.

    Example:
        st index info my_index.faiss
    """
    index_path = Path(index_file)
    if not index_path.is_absolute():
        index_path = _get_indices_dir() / index_file

    meta_path = index_path.parent / f"{index_path.name}.meta"

    if not index_path.exists():
        click.echo(f"Error: Index file {index_path} not found.", err=True)
        return

    if not meta_path.exists():
        click.echo(f"Error: Metadata file {meta_path} not found.", err=True)
        return

    # Load metadata
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Load index to get stats
    index = faiss.read_index(str(index_path))

    click.echo("Index Information:")
    click.echo(f"  Path: {index_path}")
    click.echo(f"  Version: {metadata.get('version', 'unknown')}")
    click.echo(f"  Chunks: {metadata['num_chunks']}")
    click.echo(f"  Dimension: {metadata['dimension']}")
    click.echo(f"  Chunk size: {metadata['chunk_size']} tokens")
    click.echo(f"  Chunk overlap: {metadata['chunk_overlap']} tokens")
    click.echo(f"  Index size: {index.ntotal} vectors")
    click.echo("\nSource files:")

    for file_path, file_info in metadata.get("files", {}).items():
        click.echo(f"  - {file_path}")
        click.echo(f"    Hash: {file_info['hash'][:16]}...")
        click.echo(f"    Size: {file_info['size']:,} bytes")


@index.command()
@click.argument("index_file", type=click.Path(exists=True))
@click.argument("query")
@click.option("--top-k", "-k", default=3, help="Number of results to return")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for deterministic embedding.",
    show_default=True,
)
def search(index_file: str, query: str, top_k: int, output_json: bool, seed: int):
    """Search for similar chunks in a FAISS index.

    Examples:
        st index search my_index.faiss "how to install"
        st index search default.faiss "configuration" --top-k 5 --json
    """
    index_path = Path(index_file)
    if not index_path.is_absolute():
        index_path = _get_indices_dir() / index_file

    meta_path = index_path.parent / f"{index_path.name}.meta"

    if not index_path.exists():
        click.echo(f"Error: Index file {index_path} not found.", err=True)
        return

    if not meta_path.exists():
        click.echo(f"Error: Metadata file {meta_path} not found.", err=True)
        return

    # Check cache first
    cache_key = (str(index_path), query, top_k)
    cached_result = _index_search_cache.get(cache_key)
    if cached_result is not None:
        if output_json:
            click.echo(json.dumps(cached_result, indent=2))
        else:
            for i, result in enumerate(cached_result["results"]):
                click.echo(f"\n--- Result {i + 1} (score: {result['score']:.4f}) ---")
                click.echo(f"Source: {result['source_file']}")
                click.echo(f"Text: {result['text'][:200]}...")
        return

    # Load index and metadata
    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Generate query embedding
    query_embedding = create_embedding(query, seed=seed)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    # Search
    distances, indices = index.search(query_embedding, top_k)

    # Build results
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0 or idx >= len(metadata["chunks"]):
            continue

        chunk = metadata["chunks"][idx]
        results.append(
            {
                "rank": i + 1,
                "score": float(dist),
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "start": chunk["start"],
                "end": chunk["end"],
            }
        )

    output = {"query": query, "top_k": top_k, "results": results}

    # Cache the result
    _index_search_cache.set(cache_key, output)

    if output_json:
        click.echo(json.dumps(output, indent=2))
    else:
        for i, result in enumerate(results):
            click.echo(f"\n--- Result {i + 1} (score: {result['score']:.4f}) ---")
            click.echo(f"Source: {result['source_file']}")
            click.echo(f"Text: {result['text'][:200]}...")


# AIDEV-NOTE: Helper function to get default index path
def get_default_index_path() -> Optional[Path]:
    """Get the path to the default index if it exists."""
    default_path = _get_indices_dir() / "default.faiss"
    if default_path.exists():
        return default_path
    return None


# AIDEV-NOTE: Helper function to search index for context
def search_index_for_context(
    query: str, index_path: Optional[Path] = None, top_k: int = 3, seed: int = 42
) -> List[str]:
    """
    Search FAISS index and return top chunks as context.
    Returns empty list if no index available or on error.
    """
    if index_path is None:
        index_path = get_default_index_path()

    if index_path is None or not index_path.exists():
        return []

    meta_path = index_path.parent / f"{index_path.name}.meta"
    if not meta_path.exists():
        return []

    # Check cache
    cache_key = (str(index_path), query, top_k)
    cached_result = _index_search_cache.get(cache_key)

    if cached_result is not None:
        return [r["text"] for r in cached_result["results"]]

    try:
        # Load index and metadata
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        # Generate query embedding
        query_embedding = create_embedding(query, seed=seed)
        query_embedding = np.array([query_embedding], dtype=np.float32)

        # Search
        distances, indices = index.search(query_embedding, top_k)

        # Extract chunks
        chunks = []
        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(metadata["chunks"]):
                continue

            chunk = metadata["chunks"][idx]
            chunks.append(chunk["text"])
            results.append(
                {
                    "rank": len(results) + 1,
                    "score": float(dist),
                    "text": chunk["text"],
                    "source_file": chunk["source_file"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                }
            )

        # Cache the result
        _index_search_cache.set(
            cache_key, {"query": query, "top_k": top_k, "results": results}
        )

        return chunks

    except Exception as e:
        logger.error(f"Error searching index: {e}")
        return []

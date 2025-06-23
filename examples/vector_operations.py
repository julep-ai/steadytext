#!/usr/bin/env python3
"""Example: Vector operations with SteadyText embeddings

This example demonstrates various vector operations available through the CLI.
All operations are deterministic - running them multiple times produces identical results.
"""

import subprocess
import json
import tempfile
import os


def run_command(cmd):
    """Run a shell command and return the output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def demo_similarity():
    """Demonstrate cosine similarity between texts."""
    print("=== Cosine Similarity ===")

    # Similar texts
    result = run_command(
        'st vector similarity "machine learning" "artificial intelligence"'
    )
    print(f"ML vs AI similarity: {result}")

    # Different texts
    result = run_command('st vector similarity "cat" "quantum physics"')
    print(f"Cat vs Physics similarity: {result}")

    # Get JSON output
    result = run_command('st vector similarity "Python" "programming" --json')
    data = json.loads(result)
    print(f"\nJSON output: similarity = {data['similarity']:.4f}")


def demo_distance():
    """Demonstrate distance calculations."""
    print("\n=== Distance Metrics ===")

    # Euclidean distance
    result = run_command('st vector distance "hello" "world" --metric euclidean')
    print(f"Euclidean distance: {result}")

    # Manhattan distance
    result = run_command('st vector distance "hello" "world" --metric manhattan')
    print(f"Manhattan distance: {result}")

    # Cosine distance
    result = run_command('st vector distance "hello" "world" --metric cosine')
    print(f"Cosine distance: {result}")


def demo_search():
    """Demonstrate similarity search."""
    print("\n=== Similarity Search ===")

    # Create temporary files with different content
    with tempfile.TemporaryDirectory() as tmpdir:
        files = {
            "python.txt": "Python is a high-level programming language",
            "java.txt": "Java is an object-oriented programming language",
            "cooking.txt": "Cooking is the art of preparing food",
            "music.txt": "Music is organized sound that creates emotion",
        }

        for filename, content in files.items():
            with open(os.path.join(tmpdir, filename), "w") as f:
                f.write(content)

        # Search for programming-related content
        result = run_command(
            f'st vector search "software development" {tmpdir}/*.txt --json'
        )
        data = json.loads(result)

        print("Query: 'software development'")
        for result in data["results"]:
            print(f"  {result['file']}: {result['similarity']:.4f}")


def demo_average():
    """Demonstrate embedding averaging."""
    print("\n=== Embedding Average ===")

    # Average similar concepts
    result = run_command('st vector average "cat" "dog" "bird" --json')
    data = json.loads(result)
    print(f"Average of pet animals: {data['dimension']}D vector")
    print(f"First 5 values: {data['embedding'][:5]}")


def demo_arithmetic():
    """Demonstrate vector arithmetic (king - man + woman = queen)."""
    print("\n=== Vector Arithmetic ===")

    # Classic word analogy
    result = run_command('st vector arithmetic "king" - "man" + "woman"')
    print("king - man + woman:")
    print(f"Result vector (first 10 values): {result}")

    # Another example
    result = run_command('st vector arithmetic "Paris" - "France" + "Japan" --json')
    data = json.loads(result)
    print("\nParis - France + Japan = vector similar to 'Tokyo'")
    print(f"Dimension: {data['dimension']}")


def demo_stdin():
    """Demonstrate stdin input."""
    print("\n=== Stdin Input ===")

    # Pipe text to similarity
    result = subprocess.run(
        'echo "neural networks" | st vector similarity - "deep learning"',
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"Piped similarity: {result}")


if __name__ == "__main__":
    print("SteadyText Vector Operations Examples")
    print("=====================================\n")

    print("Note: All operations are deterministic - run this script")
    print("multiple times and you'll get identical results!\n")

    demo_similarity()
    demo_distance()
    demo_search()
    demo_average()
    demo_arithmetic()
    demo_stdin()

    print("\n\nTip: Use --json flag for structured output in your applications!")

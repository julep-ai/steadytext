#!/usr/bin/env python3
"""Example: FAISS Index Management with SteadyText

This example demonstrates how to create and use FAISS indices for
deterministic document retrieval and context-enhanced generation.
"""

import subprocess
import json
import tempfile
import os


def run_command(cmd):
    """Run a shell command and return the output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout.strip()


def create_sample_documents(tmpdir):
    """Create sample documents for indexing."""
    documents = {
        "python_intro.txt": """
Python Programming Introduction

Python is a high-level, interpreted programming language known for its simplicity 
and readability. It was created by Guido van Rossum and first released in 1991.

Key features:
- Easy to learn syntax
- Dynamic typing
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python is used for web development, data science, AI, automation, and more.
""",
        "python_setup.txt": """
Installing Python

To install Python on your system:

1. Windows: Download installer from python.org
2. macOS: Use Homebrew: brew install python3
3. Linux: Use package manager: apt-get install python3

After installation, verify with: python --version

Setting up virtual environments:
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\\Scripts\\activate
""",
        "python_basics.txt": """
Python Basic Syntax

Variables and Data Types:
- Numbers: int, float, complex
- Strings: text enclosed in quotes
- Lists: ordered, mutable sequences
- Dictionaries: key-value pairs
- Sets: unordered, unique elements

Control Flow:
- if/elif/else statements
- for loops
- while loops
- break and continue

Functions:
def greet(name):
    return f"Hello, {name}!"
""",
        "machine_learning.txt": """
Machine Learning with Python

Python is the most popular language for machine learning due to its rich 
ecosystem of libraries:

1. NumPy: Numerical computing
2. Pandas: Data manipulation
3. Scikit-learn: Traditional ML algorithms
4. TensorFlow/PyTorch: Deep learning
5. Matplotlib/Seaborn: Data visualization

Common ML tasks:
- Classification
- Regression
- Clustering
- Dimensionality reduction
""",
    }

    for filename, content in documents.items():
        with open(os.path.join(tmpdir, filename), "w") as f:
            f.write(content)

    return list(documents.keys())


def demo_index_creation():
    """Demonstrate creating a FAISS index."""
    print("=== Creating FAISS Index ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample documents
        files = create_sample_documents(tmpdir)
        print(f"Created {len(files)} sample documents")

        # Create index
        index_path = os.path.join(tmpdir, "python_docs.faiss")
        cmd = f"st index create {tmpdir}/*.txt --output python_docs.faiss --chunk-size 256"
        output = run_command(cmd)
        print(output)

        # Get index info
        print("\n=== Index Information ===")
        info_output = run_command("st index info python_docs.faiss")
        print(info_output)

        return index_path


def demo_index_search():
    """Demonstrate searching a FAISS index."""
    print("\n=== Searching Index ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create documents and index
        create_sample_documents(tmpdir)
        run_command(f"st index create {tmpdir}/*.txt --output docs.faiss")

        # Search queries
        queries = [
            "how to install Python",
            "what are Python data types",
            "machine learning libraries",
            "virtual environment setup",
        ]

        for query in queries:
            print(f"\nQuery: '{query}'")
            result = run_command(
                f'st index search docs.faiss "{query}" --top-k 2 --json'
            )
            data = json.loads(result)

            for i, res in enumerate(data["results"]):
                print(f"  Result {i + 1} (score: {res['score']:.4f})")
                print(f"  Source: {res['source_file']}")
                print(f"  Text: {res['text'][:100]}...")


def demo_context_generation():
    """Demonstrate context-enhanced generation with index."""
    print("\n=== Context-Enhanced Generation ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create documents and index
        create_sample_documents(tmpdir)
        run_command(f"st index create {tmpdir}/*.txt --output knowledge.faiss")

        # Generate without context
        print("\nGeneration WITHOUT index:")
        result = run_command(
            'st generate "What libraries are used for machine learning?" --no-index'
        )
        print(f"Response: {result[:200]}...")

        # Generate with context
        print("\nGeneration WITH index:")
        result = run_command(
            'st generate "What libraries are used for machine learning?" --index-file knowledge.faiss'
        )
        print(f"Response: {result[:200]}...")

        # Show what context was used
        print("\nContext chunks used:")
        search_result = run_command(
            'st index search knowledge.faiss "machine learning libraries" --top-k 3 --json'
        )
        data = json.loads(search_result)
        for i, res in enumerate(data["results"]):
            print(f"  Chunk {i + 1}: {res['text'][:80]}...")


def demo_incremental_index():
    """Demonstrate updating an index with new documents."""
    print("\n=== Incremental Index Updates ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create initial documents
        with open(os.path.join(tmpdir, "doc1.txt"), "w") as f:
            f.write("Initial document about Python basics")

        # Create initial index
        run_command(f"st index create {tmpdir}/doc1.txt --output my_index.faiss")

        # Add more documents
        with open(os.path.join(tmpdir, "doc2.txt"), "w") as f:
            f.write("Advanced Python topics and best practices")

        # Recreate index with all documents (using --force)
        run_command(f"st index create {tmpdir}/*.txt --output my_index.faiss --force")
        print("Index updated with new documents")

        # Verify the update
        info = run_command("st index info my_index.faiss")
        print(info)


def demo_default_index():
    """Demonstrate using default index."""
    print("\n=== Default Index Usage ===")

    # Create a default.faiss index
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "readme.txt"), "w") as f:
            f.write("This is a README file explaining how to use this software.")

        run_command(f"st index create {tmpdir}/readme.txt --output default.faiss")

        # Now generation will automatically use default.faiss
        print("Generation automatically uses default.faiss when available:")
        result = run_command('st generate "How do I use this software?"')
        print(f"Response: {result[:150]}...")


if __name__ == "__main__":
    print("SteadyText Index Management Examples")
    print("====================================\n")

    print("Note: FAISS indices provide deterministic document retrieval")
    print("for context-enhanced generation (RAG pattern).\n")

    demo_index_creation()
    demo_index_search()
    demo_context_generation()
    demo_incremental_index()
    demo_default_index()

    print("\n\nTips:")
    print("- Create a default.faiss index for automatic context retrieval")
    print("- Use --chunk-size to control granularity (default: 512 tokens)")
    print("- Use --top-k to control how many context chunks to retrieve")
    print("- All operations are deterministic for reproducible RAG!")

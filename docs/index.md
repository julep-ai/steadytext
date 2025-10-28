# SteadyText Documentation

Deterministic AI for app developers **and** database teams. Same inputs, same outputs—across Python and Postgres.

!!! important "Versioning"
    Releases follow `yyyy.mm.dd`. The current docs track the **2025.10.xx** train covering both the Python SDK and the `pg_steadytext` extension.

---

## Choose Your Track

=== "Python Library"
Time-to-first-deterministic-result in minutes. Install the SDK, run your first `steadytext.generate`, and wire deterministic workflows into apps or CI.

- Start with the [Python Quick Start](quick-start.md)
- Follow the tutorials grouped under the Python Library pillar.
- Dive into the [API Reference](api/index.md) and operations playbooks

=== "Postgres Extension"
Bring deterministic generation, embeddings, and reranking into SQL. Install the extension, connect to the daemon, and build pipelines inside Postgres.

- Begin with the [Postgres Quick Start](postgresql-extension.md)
- Follow the tutorials grouped under the Postgres Extension pillar.
- Explore the [Function Catalog](postgresql-extension-reference.md) and operations guides

---

## Core Platform at a Glance

- **Daemon-backed services** keep models hot and provide shared caching across SDK and SQL.
- **Deterministic execution** guarantees reproducible outputs with seed control and graceful fallbacks.
- **Structured generation & reranking** work the same in Python and SQL, powered by a shared model registry.
- **Vector indexing** via FAISS integrates with CLI helpers and Postgres search functions.

Read the [Core Platform Hub](architecture.md) to learn how the pillars connect and when to choose each surface.

---

## Why Teams Ship with SteadyText

- **Predictable CI** – deterministic responses eliminate flaky AI-based tests.
- **Hybrid deployment** – mix Python services with SQL automation without managing separate model stacks.
- **Timeboxed iteration** – date-based versions and changelog callouts make upgrades deliberate.
- **Open ecosystem** – models, cache backends, and integrations are swappable via configuration.

See [Benchmarks](benchmarks.md) for performance data and [FAQ](faq.md) for quick answers.

---

## What’s New

- **Twin Pillars navigation**: mirrored quick starts and tutorials for Python and Postgres.
- **Expanded Postgres coverage**: async workflows, reranking, and prompt registry guidance.
- **Operations consolidation**: deployment, caching, and unsafe-mode docs share a single home.

Check [Version History](version_history.md) and [Changelog (GitHub)](https://github.com/julep-ai/steadytext/blob/main/CHANGELOG.md) for release specifics.

---

## Need Help?

- Operations questions → [Operations & Integrations](deployment.md)
- Community support → [Contributing Guide](contributing.md) & Slack/Discord
- Found a gap? Open an issue or ping the docs team with anchor references.

You're now ready to pick a track and explore. Happy shipping!

---

## 📦 Installation & Models

Install stable release:

```bash
# Using UV (recommended - 10-100x faster)
uv add steadytext

# Or using pip
pip install steadytext
```

### Models

**Current models (v2025.8.17+)**:

* Generation (Small): `Qwen3-4B-Instruct` (3.9GB) - High-quality 4B parameter model (default)
* Generation (Large): `Qwen3-30B-A3B-Instruct` (12GB) - Advanced 30B parameter model
* Embeddings: `Jina-v4-Text-Retrieval` (1.2GB) - State-of-the-art 2048-dim embeddings (truncated to 1024)
* Reranking: `Qwen3-Reranker-4B` (3.5GB) - Document reranking model

!!! note "Version Stability"
    Each major version will use a fixed set of models only, so that only forced upgrades from pip will change the models (and the deterministic output)

---

## 🎯 Use Cases

!!! success "Perfect for"
    * **Testing AI features**: Reliable asserts that never flake
    * **Deterministic CLI tooling**: Consistent outputs for automation  
    * **Reproducible documentation**: Examples that always work
    * **Offline/dev/staging environments**: No API keys needed
    * **Semantic caching and embedding search**: Fast similarity matching

!!! warning "Not ideal for"
    * Creative or conversational tasks
    * Latest knowledge queries  
    * Large-scale chatbot deployments

---

## 📋 Examples

Use SteadyText in tests or CLI tools for consistent, reproducible results:

```python
# Testing with reliable assertions
def test_ai_function():
    result = my_ai_function("test input")
    expected = steadytext.generate("expected output for 'test input'")
    assert result == expected  # No flakes!

# CLI tools with consistent outputs
import click

@click.command()
def ai_tool(prompt):
    print(steadytext.generate(prompt))
```

📂 **[More examples →](examples/index.md)**

---

## 🔍 API Overview

```python
# Text generation
steadytext.generate(prompt: str) -> str
steadytext.generate(prompt, return_logprobs=True)
steadytext.generate(prompt, schema=MyModel)  # Structured output

# Streaming generation
steadytext.generate_iter(prompt: str)

# Document reranking
steadytext.rerank(query: str, documents: List[str]) -> List[Tuple[str, float]]

# Embeddings
steadytext.embed(text: str | List[str]) -> np.ndarray

# Model preloading
steadytext.preload_models(verbose=True)
```

📚 **[Full API Documentation →](api/index.md)**

---

## 🔧 Configuration

Control caching behavior via environment variables:

```bash
# Generation cache (default: 256 entries, 50MB)
export STEADYTEXT_GENERATION_CACHE_CAPACITY=256
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=50

# Embedding cache (default: 512 entries, 100MB)
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=512
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=100
```

---

## 🤝 Contributing

Contributions are welcome! See [Contributing Guide](contributing.md) for guidelines.

---

## 📄 License

* **Code**: MIT
* **Models**: MIT (Qwen3)

---

Built with ❤️ for developers tired of flaky AI tests.

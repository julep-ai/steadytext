# Why SteadyText

SteadyText brings deterministic AI to both application code and SQL, so teams can automate with confidence. This page highlights the problems we solve, the principles behind the platform, and the scenarios where the twin pillars shine.

---

## The Pain We Eliminate

| Symptom | Traditional AI stack | With SteadyText |
|---------|---------------------|-----------------|
| Flaky tests | Same prompt returns different answers; CI breaks unpredictably | Seeds and caching guarantee identical outputs |
| Operational blind spots | Models run behind opaque APIs | Local daemon makes behavior observable and auditable |
| Split stacks | Python teams and database teams manage separate tooling | Shared models and cache power both pillars |
| Compliance risk | Data leaves infra for hosted APIs | Everything runs where your data already lives |

---

## Core Principles

1. **Deterministic by default** – Seeds propagate through every code path, including streaming and embeddings.  
2. **Never fail** – If a model is unavailable, functions return `NULL`/`None` so workflows can handle it explicitly.  
3. **Local-first** – No external API calls; caches and models live with your workloads.  
4. **Twin Pillars** – Python SDK and Postgres extension share daemons, caches, and structured generation features.  
5. **Extensible** – Swap models, cache backends, and integrations without rewriting pipelines.

---

## How the Pieces Fit

- **Daemon & cache**: A ZeroMQ service keeps models warm and exposes a shared frecency cache. Both pillars use it, so embeddings generated in Python can be reused in SQL and vice versa.
- **Deterministic engine**: Generators, embedders, rerankers, and structured output parsers live in the core layer (see [Architecture Overview](architecture.md)).
- **Surface adapters**:  
  - Python exposes `steadytext.generate`, `embed`, `rerank`, and structured helpers.  
  - Postgres exposes `steadytext_generate`, `steadytext_embed`, `steadytext_rerank`, plus async and prompt-registry SQL functions.
- **Operations hub**: Deployment, unsafe-mode overrides, and integration recipes are documented once and linked from both journeys.

---

## When to Use Which Pillar

| Scenario | Start Here | Why |
|----------|------------|-----|
| CI/CD gating, developer tools | [Python Quick Start](quick-start.md) | Fast local iteration, direct SDK access |
| In-database automation, analytics | [Postgres Quick Start](postgresql-extension.md) | Run AI functions inside SQL transactions |
| Hybrid app + SQL workflows | Both | Share seeds, caches, and reranking between services |

---

## Common Outcomes

- **QA teams** record seeds for every generation and plug deterministic checks into pytest and pgTap.  
- **Data engineers** run ETL pipelines that enrich records via SQL functions without leaving the warehouse.  
- **Product teams** ship deterministic content flows (emails, support responses, documentation) with versioned prompts.  
- **Ops/SRE** teams keep a single daemon and cache fleet, monitor health via SQL, and roll out updates with predictable behavior.

---

## Performance Snapshot

- Generation latency: typically \<100 ms after cache warm-up  
- Embeddings: 1024-d vectors with deterministic L2 normalization  
- Cache hit rate: 90%+ in production workloads with frecency eviction  
- Resource footprint: 2–4 GB RAM for default model set; configurable via cache settings

---

## Next Steps

- Dive into the [Core Platform Hub](architecture.md) for diagrams and internals.
- Jump to the [Python](quick-start.md) or [Postgres](postgresql-extension.md) quick starts.
- Review [Benchmarks](benchmarks.md) and [Vector Indexing](vector-indexing.md) to plan production rollouts.

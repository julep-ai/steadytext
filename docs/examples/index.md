# Examples Hub

Find runnable scenarios that demonstrate SteadyText in real projects. Pick the surface you’re building on, then drill into verticals or cross-cutting recipes.

---

## Python Library

| Topic | What you’ll learn | Jump in |
|-------|-------------------|---------|
| Getting deterministic outputs | First-generation patterns, streaming, embeddings | [Basic Usage](basic-usage.md) |
| Controlling seeds | Reproducible experimentation and CI diffs | [Custom Seeds](custom-seeds.md) |
| Developer tooling | Build deterministic CLIs and automation | [Tooling](tooling.md) |
| Data & pipelines | Batch ETL/ELT integration with the daemon | [Data Pipelines](data-pipelines.md) |
| Vertical solutions | Content, logs, customer analytics | [Content Management](content-management.md), [Log Analysis](log-analysis.md), [Customer Intelligence](customer-intelligence.md) |
| Quality gates | Deterministic regression checks | [Testing with AI](testing.md) |

---

## Postgres Extension

| Topic | What you’ll learn | Jump in |
|-------|-------------------|---------|
| Install & integrate | Wire SQL functions into existing schemas | [Integration Overview](postgresql-integration.md) |
| Analytical workloads | Embeddings + reranking for BI dashboards | [PostgreSQL Analytics](postgresql-analytics.md) |
| Content + commerce | Build deterministic editorial & commerce flows | [PostgreSQL Blog CMS](postgresql-blog-cms.md), [PostgreSQL E-commerce](postgresql-ecommerce.md) |
| Search & retrieval | Semantic search, query expansion, result shaping | [PostgreSQL Search](postgresql-search.md) |
| Real-time apps | LISTEN/NOTIFY, async jobs, background workers | [PostgreSQL Realtime](postgresql-realtime.md) |

---

## Shared Recipes

- Caching & performance tuning → [Caching](caching.md), [Performance Tuning](performance-tuning.md)
- Daemon operations → [Daemon Usage](daemon-usage.md)
- Error handling patterns → [Error Handling](error-handling.md)
- Tool stack integration → [Shell Integration](../shell-integration.md), [Vector Indexing](../vector-indexing.md)

---

## How to Use These Examples

1. **Clone the repo**  
   ```bash
   git clone https://github.com/julep-ai/steadytext.git
   cd steadytext/examples
   ```
2. **Pick your surface** — follow the Python or Postgres table above.
3. **Run the code** — every script is deterministic; outputs match the docs.
4. **Adapt & link back** — reuse the patterns and add `AIDEV-REF` comments if you extend them.

!!! tip "Same input → same output"
    Determinism underpins every example. When experimenting, change seeds intentionally and record them so teammates and CI runs can reproduce your outputs precisely.

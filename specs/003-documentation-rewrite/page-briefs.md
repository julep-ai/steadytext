## Core Platform Hub
- **Purpose**: Introduce the shared foundation that powers both Python and Postgres experiences, directing readers to the right surface-specific journey.
- **Primary Audience**: Product evaluators, architects, and senior developers comparing surfaces.
- **Key Questions Answered**: What is SteadyText's core architecture? How do the two pillars relate? Where should I start?
- **Related Assets**: README landing hero, architecture diagram assets, benchmarks summary, concept glossary.
- **Migration Notes**: Pull forward essentials from `concepts.md`, `architecture.md`, and `why-steadytext.md`; set redirect plan once new landing pages ship.

### Architecture Overview
- **Purpose**: Summarize the service layers, daemon architecture, and how models are orchestrated across Python and Postgres.
- **Primary Audience**: Technical leaders and engineers planning deployments.
- **Key Questions Answered**: How is compute shared? What does the daemon provide? What are the extension touchpoints?
- **Related Assets**: `architecture.md`, daemon diagrams, cache backend docs.
- **Migration Notes**: Reuse relevant paragraphs from current `architecture.md`; ensure diagrams map to new nav anchors.

### Deterministic Concepts
- **Purpose**: Explain SteadyText's deterministic guarantees and seed mechanics for both pillars.
- **Primary Audience**: Developers evaluating reliability and QA teams verifying repeatability.
- **Key Questions Answered**: How do seeds flow? What happens when models are unavailable? How do fallbacks behave?
- **Related Assets**: `concepts.md`, deterministic generator code references, examples/custom-seeds.md.
- **Migration Notes**: Merge content from `concepts.md` and `why-steadytext.md`; update anchor references accordingly.

### Structured Generation
- **Purpose**: Highlight structured output capabilities that apply equally to Python SDK and SQL functions.
- **Primary Audience**: Builders implementing schema-bound outputs or deterministic workflows.
- **Key Questions Answered**: Which structured options exist? How do I validate outputs? Can Postgres enforce schemas?
- **Related Assets**: `structured-generation.md`, `docs/examples/tooling.md`, SQL structured guide.
- **Migration Notes**: Harmonize terminology between SDK and extension guides; set cross-links to both quick starts.

### Model Switching
- **Purpose**: Describe how users choose between standard, large, and mini models across both pillars.
- **Primary Audience**: Developers optimizing cost/performance and ops teams setting defaults.
- **Key Questions Answered**: Which models are available? How do I toggle mini mode? How does the daemon respond?
- **Related Assets**: `model-switching.md`, environment variable reference, CLI flags.
- **Migration Notes**: Consolidate guidance from `model-switching.md` and scattered FAQ entries.

### Vector Indexing
- **Purpose**: Provide a high-level decision guide for creating, querying, and integrating FAISS indexes.
- **Primary Audience**: Developers building retrieval workflows from Python or SQL.
- **Key Questions Answered**: When should I index? How does Postgres consume indexes? What CLI helpers exist?
- **Related Assets**: `vector-indexing.md`, CLI command docs, integration recipes.
- **Migration Notes**: Merge overlapping instructions from examples and CLI docs; maintain CLI cross-links.

### Reranking Overview
- **Purpose**: Explain reranking capabilities available via SDK and SQL interfaces.
- **Primary Audience**: Retrieval engineers and data scientists tuning relevance.
- **Key Questions Answered**: Which reranker models are supported? How do I call them from SQL? How do I validate outputs?
- **Related Assets**: `reranking.md`, Postgres reranking guide, benchmarks snapshots.
- **Migration Notes**: Unify `reranking.md` with Postgres reranking doc; create consistent anchor references.

### EOS String Implementation
- **Purpose**: Document the end-of-sequence handling strategy and why it matters for deterministic outputs.
- **Primary Audience**: Engineers debugging edge cases and contributors reviewing generation internals.
- **Key Questions Answered**: How do EOS tokens differ by model? How does fallback behave? What are the extension implications?
- **Related Assets**: `eos-string-implementation.md`, generator code anchors, SQL structured docs.
- **Migration Notes**: Keep deep-dive content but expose a summary for casual readers; ensure cross-links into internals.

### Benchmarks
- **Purpose**: Provide a human-readable summary of benchmark methodology and highlight key results.
- **Primary Audience**: Decision makers evaluating SteadyText performance.
- **Key Questions Answered**: How fast and accurate is SteadyText? What datasets and metrics are used? Where are raw reports stored?
- **Related Assets**: `benchmarks.md`, `benchmarks/results/` appendices, product roadmap notes.
- **Migration Notes**: Maintain links to raw Markdown reports; add anchor references for future automated updates.

### FAQ
- **Purpose**: Organize common questions by theme (platform, Python, Postgres, operations) and reduce support load.
- **Primary Audience**: New adopters and support teams.
- **Key Questions Answered**: Common installation issues, feature availability, roadmap signals.
- **Related Assets**: `faq.md`, troubleshooting indices, release notes.
- **Migration Notes**: Clean redundant Q&A currently scattered across pages; enforce tag-based categorization.

### Examples Hub
- **Purpose**: Act as a directory guiding users to Python or Postgres-focused examples.
- **Primary Audience**: Builders seeking applied tutorials by use case.
- **Key Questions Answered**: Which example fits my scenario? Is there a SQL equivalent? How do I adapt tutorials?
- **Related Assets**: All pages under `docs/examples/`, new quick start sections, tutorial templates.
- **Migration Notes**: Replace existing `examples/index.md` with dual-track pointers; add consistent metadata blocks.

## Python Library Journey

1. **Python Quick Start** (`docs/quick-start.md`)
   - **Objective**: Install SteadyText and run first deterministic AI generation with custom seeds.
   - **Key takeaway**: Installation via uv/pip, basic `generate()` and `embed()` calls with seed parameters for reproducible variations.
   - **Dependencies**: Python 3.10+, 4GB+ RAM, 5-15GB disk space for models.

2. **Basic Usage** (`docs/examples/basic-usage.md`)
   - **Objective**: Master fundamental SteadyText operations including generation, streaming, embeddings, and structured output.
   - **Key takeaway**: Deterministic guarantees (same input = same output), L2-normalized embeddings, JSON/regex/choice constraints for structured generation.
   - **Dependencies**: Core steadytext library, optional pydantic for schemas.

3. **Custom Seeds** (`docs/examples/custom-seeds.md`)
   - **Objective**: Use custom seeds to generate reproducible variations for A/B testing, research, and content diversification.
   - **Key takeaway**: Seed management patterns (ranges, scheduling, conditional strategies), reproducibility frameworks for scientific workflows.
   - **Dependencies**: Basic usage patterns, understanding of deterministic generation.

4. **Content Management** (`docs/examples/content-management.md`)
   - **Objective**: Build AI-powered content management systems with auto-generation, A/B testing, and moderation pipelines.
   - **Key takeaway**: SQL-based content workflows (product descriptions, SEO metadata), real-time moderation, personalized content per user segment.
   - **Dependencies**: PostgreSQL + pg_steadytext extension, basic SQL knowledge.

5. **Customer Intelligence** (`docs/examples/customer-intelligence.md`)
   - **Objective**: Transform customer data into actionable insights with sentiment analysis, churn prediction, and personalized recommendations.
   - **Key takeaway**: Real-time review analysis, AI-powered segmentation, automated retention strategies via SQL triggers.
   - **Dependencies**: PostgreSQL + pg_steadytext, customer/interaction tables, understanding of analytics workflows.

6. **Data Pipelines** (`docs/examples/data-pipelines.md`)
   - **Objective**: Create intelligent ETL pipelines that enrich, transform, and analyze data with AI in PostgreSQL.
   - **Key takeaway**: Real-time enrichment triggers, batch processing patterns, quality monitoring with AI validation, automated reporting.
   - **Dependencies**: PostgreSQL + pg_steadytext, optional pg_cron for scheduling, TimescaleDB for time-series.

7. **Log Analysis** (`docs/examples/log-analysis.md`)
   - **Objective**: Transform logs from noise into insights with AI-powered summarization, threat detection, and anomaly recognition.
   - **Key takeaway**: Continuous aggregates for hourly summaries, security risk classification, pattern recognition with statistical baselines.
   - **Dependencies**: PostgreSQL + pg_steadytext, optional TimescaleDB for continuous aggregates, pg_cron for automation.

8. **Testing with AI** (`docs/examples/testing.md`)
   - **Objective**: Build reliable, non-flaky AI tests using SteadyText's deterministic outputs for mocking and validation.
   - **Key takeaway**: Deterministic assertion patterns, reproducible fuzz testing, mock AI services with state management.
   - **Dependencies**: Python testing framework (unittest/pytest), steadytext library, understanding of test patterns.

## Python Library Journey
1. **Python Quick Start** – orient newcomers to installation and the twin-pillar decision.
   - Key takeaway: Installs the SDK, highlights when to reach for Postgres.
   - Dependencies: README landing instructions, future dedicated quick-start page.
2. **Basic Usage** – guide through first deterministic generation call.
   - Key takeaway: Shows default generator pattern and result interpretation.
   - Dependencies: Requires seed explanation and CLI prerequisites.
3. **Custom Seeds** – explain seed control for reproducibility.
   - Key takeaway: Demonstrates deterministic workflows and comparison testing.
   - Dependencies: Links to deterministic concepts reference.
4. **Content Management** – showcase structured prompts for editorial pipelines.
   - Key takeaway: Applies structured generation to CMS-like use cases.
   - Dependencies: Structured generation guide, prompt registry references.
5. **Customer Intelligence** – apply embeddings and retrieval to customer data sets.
   - Key takeaway: Demonstrates analytic workflows and reranking tie-ins.
   - Dependencies: Vector indexing guide, reranking overview.
6. **Data Pipelines** – integrate SteadyText within ETL/ELT jobs.
   - Key takeaway: Highlights batch processing patterns and daemon usage.
   - Dependencies: Daemon operations guide, caching recipes.
7. **Log Analysis** – analyze operational logs for anomaly detection.
   - Key takeaway: Shows combining embeddings with deterministic classification.
   - Dependencies: Performance tuning doc, structured output references.
8. **Testing with AI** – describe how to integrate deterministic checks into CI.
   - Key takeaway: Positions AI-based tests as complements to unit tests.
   - Dependencies: Testing frameworks reference, deterministic concept link.

## Postgres Extension Journey
1. **Postgres Quick Start** – install the extension, connect to the daemon, and issue the first SQL generation.
   - Key takeaway: Validates end-to-end install with minimal SQL snippets.
   - Dependencies: `pg_steadytext/INSTALL.md`, daemon status docs.
2. **Integration Overview** – explain how to wire SteadyText SQL functions into existing schemas.
   - Key takeaway: Provides migration path for pgvector or custom SQL wrappers.
   - Dependencies: `examples/postgresql-integration.md`, prompt registry guide.
3. **Analytics Playbook** – demonstrate analytical aggregations with embeddings and reranking.
   - Key takeaway: Shows how to combine embeddings, reranking, and SQL joins.
   - Dependencies: Vector indexing, reranking overview, analytics example.
4. **Blog & CMS** – build editorial workflows entirely in SQL.
   - Key takeaway: Applies structured generation and prompt registry within content tables.
   - Dependencies: Structured generation guide, prompt registry doc.
5. **E-commerce Blueprint** – showcase personalization and search inside transactional schemas.
   - Key takeaway: Combines embeddings with product catalog filters and reranking.
   - Dependencies: `examples/postgresql-ecommerce.md`, function catalog reference.
6. **Semantic Search** – deliver search and retrieval patterns with SQL-first tooling.
   - Key takeaway: Demonstrates query expansion, reranking, and result shaping.
   - Dependencies: `examples/postgresql-search.md`, vector indexing guide.
7. **Real-time Apps** – leverage LISTEN/NOTIFY and async jobs for live experiences.
   - Key takeaway: Highlights async functions, queue usage, and safety considerations.
   - Dependencies: `postgresql-extension-async.md`, async functions doc.
8. **Advanced Workflows** – cover prompt registry, structured outputs, and advanced SQL composition.
   - Key takeaway: Shows how to orchestrate sophisticated pipelines entirely in SQL.
   - Dependencies: `postgresql-extension-advanced.md`, prompt registry doc, structured generation guide.
9. **Document Reranking** – apply reranker functions to improve retrieval quality.
   - Key takeaway: Walks through reranking SQL functions and evaluation tips.
   - Dependencies: `postgresql-extension-reranking.md`, benchmarks snapshots.
10. **Operations & Troubleshooting** – direct operators to deployment, monitoring, and recovery guidance.
   - Key takeaway: Points to operations runbooks and known-issue mitigations.
   - Dependencies: `postgresql-extension-ai.md`, `postgresql-extension-troubleshooting.md`, deployment guides.

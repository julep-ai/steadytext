# Plan: Documentation Rewrite (Python & Postgres Parity)

**Branch**: `003-documentation-rewrite` • **Date**: 2025-10-28 • **Spec**: `/specs/003-documentation-rewrite/spec.md`

## Scope Snapshot
Reorganize SteadyText docs into the Twin Pillars structure that surfaces Python Library and Postgres Extension equally, consolidate overlapping material, and stage a phased migration without breaking existing workflows or builds.

## Workstreams
- **Information Architecture & Navigation** – Update `mkdocs.yml`, define section landing pages, ensure anchors and breadcrumbs reflect Python/Postgres parity.
- **Content Inventory & Page Briefs** – Map every current doc (root, `docs/`, `pg_steadytext/docs/`, `benchmarks/`, etc.) to new destinations with keep/merge/rewrite decisions.
- **Core Content Refresh** – Draft new Home, Core Platform hub, mirrored Quick Starts, and align tutorials/reference stubs for both tracks.
- **Operations & QA** – Plan redirects/backlinks, document anchor conventions, run `uv run poe docs-build` and link checks each milestone.

## Dependencies
- Finalized spec approval.
- MkDocs configuration and deployment pipeline readiness.
- Contributor guidelines for anchor comments and review gates.

## Risks & Mitigations
- **Link rot during migration** → Maintain inventory with redirect plan, stage updates in feature branch, run link checker.
- **Scope creep into product changes** → Keep non-goals visible, gate adds via review checklist.
- **Voice inconsistency between tracks** → Establish shared copy deck/terminology and peer-review cross-track pages.
- **Timeline slip due to volume** → Sequence deliverables into milestones, enable parallel work on inventory and drafting.

## Deliverables
- Updated `mkdocs.yml` IA skeleton and landing page wireframes.
- Content inventory matrix with actions per document.
- Page briefs and outline docs for both Python and Postgres tracks.
- Anchor/comment guidelines addendum for maintainers.
- Migration checklist including testing steps and redirect strategy.

## Progress Notes (2025-10-28)
- Implemented Twin Pillars navigation skeleton in `mkdocs.yml` placing Python Library and Postgres Extension as parallel top-level sections.
- Legacy `quick-start.md` temporarily services the Python quick start slot; new dedicated pages will replace it during drafting tasks.
- Postgres tutorials currently point to existing guides to preserve site build until rewritten content lands.
- Operations & Integrations pillar now aggregates deployment, migration, and performance docs pending consolidation workstreams.
- First-draft home, Python quick start, Postgres quick start, examples hub, FAQ, and benchmarks pages rewritten to reflect parity narrative and link unification.

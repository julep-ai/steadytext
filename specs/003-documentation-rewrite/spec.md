# Feature Spec: Documentation Rewrite (Python & Postgres Parity)

**Feature Branch**: `003-documentation-rewrite` • **Created**: 2025-10-28 • **Status**: Draft

## Summary
Rewrite the SteadyText documentation to elevate the PostgreSQL extension alongside the Python library, delivering a dual-track information architecture that gives teams clear, mirrored journeys for building with either surface while sharing core concepts and operations material.

## Goals
- Establish a "Twin Pillars" navigation that gives Python and Postgres equal visibility from the landing page through quick starts, tutorials, and references.
- Consolidate and refresh content so each topic has a single authoritative home with clear cross-links instead of redundant pages.
- Provide role-oriented guidance for evaluators, implementers, and operators while keeping maintenance overhead manageable.
- Ship a spec-driven migration plan that lets us rewrite and launch the new docs without breaking existing links or workflows.

## Non-Goals
- Redesign the MkDocs theme or visual styling beyond navigation updates.
- Produce new product features or APIs; scope is limited to documentation structure and content.
- Automate reference generation pipelines (e.g., SQL doc extraction) beyond scoping the work.

## User Scenario
A database engineer lands on the docs homepage, immediately sees equal-weight entry points for "Postgres Extension" and "Python Library," follows the Postgres quick start to install the extension, references the SQL function catalog, and jumps to operations guidance—all without detouring through Python-specific pages. Meanwhile, an application developer can take the parallel Python quick start path and reach architecture details under the shared Core Platform section when needed.

## Functional Requirements
1. **FR-001:** Publish a revised MkDocs navigation reflecting the Twin Pillars IA with top-level sections: Home, Core Platform, Python Library, Postgres Extension, Operations & Integrations, Contribute & Changelog.
2. **FR-002:** Create updated landing content that orients readers to both tracks and links into mirrored quick starts.
3. **FR-003:** Produce page briefs for every existing doc (current path → new destination) with an action of keep, merge, rewrite, or retire.
4. **FR-004:** Capture migration sequencing, review checkpoints, and build/test steps in the plan so the rewrite can execute in phases without breaking `poe docs-build`.
5. **FR-005:** Document anchor-comment expectations and cross-linking patterns to keep the reorganized docs maintainable.

## Dependencies
- Existing specs processes and templates in `specs/templates/` for planning, tasks, and research docs.
- MkDocs configuration (`mkdocs.yml`) and deployment pipeline to publish the reorganized site.
- Contributor guidelines (`CONTRIBUTING.md`, `CLAUDE.md`) for anchor usage, wording style, and review workflow.

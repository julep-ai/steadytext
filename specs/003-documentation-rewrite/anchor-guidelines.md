# Anchor Comment Addendum

## Purpose
Supplement existing CLAUDE.md rules with doc-specific anchor practices for the rewrite.

## Required Anchors
- Place an `AIDEV-ANCHOR:` comment above every new landing page section (Home, Core Platform, Python, Postgres).
- Maintain one anchor per ~50 lines for long tutorials or references.
- Use descriptive labels under 60 characters, mirroring nav titles when possible.

## Cross-Linking
- Reference anchors with `AIDEV-REF:` including file path and anchor label.
- When mirroring content between Python and Postgres journeys, add reciprocal refs at the top of each page.
- Document anchor updates in PR descriptions for reviewer traceability.

## Maintenance Notes
- Run the anchor-comment-manager agent before merges touching multiple docs.
- Audit anchors quarterly to confirm they align with current nav structure.
- Avoid deleting legacy anchors until redirects and references are updated.

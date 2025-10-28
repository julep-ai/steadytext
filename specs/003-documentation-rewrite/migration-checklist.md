# Migration Checklist

## Redirect & URL Strategy
- TODO: inventory existing URLs
- TODO: define redirects for retired pages
- TODO: update sitemap references

## Build & Verification Steps
- TODO: run `uv run poe docs-build`
- TODO: run link checker (command TBD)
- TODO: smoke-test search and nav tabs

## Communication Plan
- TODO: draft launch notes for README and release
- TODO: notify contributors via Slack/Discord
- TODO: update changelog entry

## Risk Log
- TODO: document dependency on mkdocs-material version
- TODO: confirm external embeds still resolve
- TODO: monitor analytics for navigation drop-offs

## Operations & Integrations Consolidation
- TODO: map duplicated deployment docs into new Operations pillar
- TODO: consolidate caching content (cache-backends.md, examples/caching.md, daemon usage)
- TODO: define integration tagging scheme (Python, Postgres, Shared)
- TODO: outline redirect list for deprecated operations pages

## Phased Implementation Schedule
1. **Milestone 1 – IA Scaffold**: Update `mkdocs.yml`, land landing-page placeholders, verify `uv run poe docs-build` passes.
2. **Milestone 2 – Core Platform & Python Track**: Publish refreshed Core Platform pages and Python journey; rerun docs build and manual nav QA.
3. **Milestone 3 – Postgres Track**: Release Postgres journey pages, function reference updates, and integration redirects; run link checker (command TBD).
4. **Milestone 4 – Operations & Launch**: Finalize migration redirects, update README/CHANGELOG, announce via community channels; monitor analytics for two weeks.

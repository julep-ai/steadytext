# Tasks: [FEATURE NAME]

**Input**: plan.md, research.md, data-model.md, contracts/

## How to Use
- Number tasks as `T001`, `T002`, …
- Mark `[P]` only when tasks touch different files and have no ordering needs.
- Keep descriptions action-oriented and include file paths.
- Tests must precede implementation.

## Task Outline (example structure)

- **T001** Capture/finish research notes (`specs/[feature]/research.md`).
- **T002** Update design artifacts (`data-model.md`, `contracts/`, `quickstart.md`).
- **T003 [P]** Add failing tests in `tests/...`.
- **T004** Implement feature code in `src/...` or `mcp_server/...`.
- **T005 [P]** Wire integrations/background jobs.
- **T006** Update docs/runbooks.
- **T007** Cleanup + verify feature flags, run full test suite.

Feel free to adjust ordering or add more tasks to match the plan; the checklist above is a starting point, not a straitjacket.

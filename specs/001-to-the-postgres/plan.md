# Implementation Plan: PostgreSQL Prompt Registry

**Branch**: `001-to-the-postgres` | **Date**: 2025-09-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/workspace/specs/001-to-the-postgres/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, or `GEMINI.md` for Gemini CLI).
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Add a lightweight prompt registry to the pg_steadytext PostgreSQL extension that stores Jinja-based prompt templates with versioning support and provides simple functions for rendering them with variable substitution. The system will maintain immutable version history and validate templates on entry.

## Technical Context
**Language/Version**: PL/Python3u with PostgreSQL 17
**Primary Dependencies**: Jinja2 (Python templating library)
**Storage**: PostgreSQL tables within extension schema
**Testing**: pgTAP test framework
**Target Platform**: PostgreSQL extension (pg_steadytext)
**Project Type**: single - PostgreSQL extension enhancement
**Performance Goals**: Sub-millisecond template rendering for typical prompts (<10KB)
**Constraints**: Must be lightweight, simple to use, maintain backward compatibility with existing extension
**Scale/Scope**: Support hundreds of prompts with multiple versions each

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 1 (pg_steadytext extension enhancement)
- Using framework directly? Yes - Jinja2 for templating
- Single data model? Yes - prompt registry tables only
- Avoiding patterns? Yes - no unnecessary abstractions

**Architecture**:
- EVERY feature as library? N/A - Extension functions in PostgreSQL
- Libraries listed: N/A - Extension context
- CLI per library: N/A - SQL functions instead
- Library docs: Will document SQL functions in extension

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? Yes - pgTAP tests first
- Git commits show tests before implementation? Yes
- Order: Contract→Integration→E2E→Unit strictly followed? Yes, adapted for SQL
- Real dependencies used? Yes - actual PostgreSQL instance
- Integration tests for: new tables, functions, version tracking
- FORBIDDEN: Implementation before test, skipping RED phase

**Observability**:
- Structured logging included? N/A - PostgreSQL logs
- Frontend logs → backend? N/A - Database extension
- Error context sufficient? Yes - detailed SQL error messages

**Versioning**:
- Version number assigned? Yes - 2025.9.6 following date-based versioning
- BUILD increments on every change? Yes
- Breaking changes handled? No breaking changes - additive only

## Project Structure

### Documentation (this feature)
```
specs/001-to-the-postgres/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
pg_steadytext/
├── sql/
│   └── pg_steadytext--2025.9.6.sql    # New version with prompt registry
├── python/
│   └── prompt_registry.py             # Jinja2 rendering logic
├── test/
│   └── pgtap/
│       └── 17_prompt_registry.sql     # pgTAP tests for prompt registry
└── CLAUDE.md                          # Updated with prompt registry notes
```

**Structure Decision**: Single project - PostgreSQL extension enhancement

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - How to handle Jinja2 dependency in PL/Python3u environment
   - Best practices for template validation in PostgreSQL
   - Optimal schema design for version tracking

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Prompt template entity with slug, content, required variables
   - Prompt version entity with version number, created timestamp
   - Relationship: one-to-many between prompt and versions

2. **Generate API contracts** from functional requirements:
   - `steadytext_prompt_create(slug, template, required_vars[], metadata)` → UUID
   - `steadytext_prompt_update(slug, template, required_vars[], metadata)` → UUID
   - `steadytext_prompt_get(slug, version?)` → template record
   - `steadytext_prompt_render(slug, variables, version?)` → rendered text
   - `steadytext_prompt_list()` → list of prompts with latest version
   - `steadytext_prompt_versions(slug)` → list of versions

3. **Generate contract tests** from contracts:
   - Test for each function's input/output
   - Tests must fail initially (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Create prompt → retrieve → render with variables
   - Update prompt → verify new version created
   - Render with missing variables → error
   - Invalid Jinja syntax → validation error

5. **Update agent file incrementally** (O(1) operation):
   - Add prompt registry notes to CLAUDE.md
   - Document Jinja2 handling pattern
   - Keep under 150 lines for token efficiency

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md update

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each SQL function → pgTAP test task [P]
- Each entity → table creation task [P]
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:
- TDD order: Tests before implementation
- Dependency order: Tables before functions before tests
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 15-20 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
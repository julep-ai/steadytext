# Implementation Plan: OpenRouter Provider Support

**Branch**: `002-openrouter-provider-in` | **Date**: 2025-09-19 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-openrouter-provider-in/spec.md`

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
Add OpenRouter provider support to SteadyText's remote model provider system, enabling developers to access OpenRouter's unified API for multiple AI models through the existing "provider:model" pattern. Implementation follows existing provider patterns (OpenAI, Cerebras) with OPENROUTER_API_KEY environment variable authentication.

## Technical Context
**Language/Version**: Python 3.10-3.13 (requires-python >=3.10,<3.14)
**Primary Dependencies**: llama-cpp-python, numpy, openai (for existing providers), httpx/requests for HTTP API calls
**Storage**: N/A (stateless provider, API key from environment)
**Testing**: pytest (existing test framework in codebase)
**Target Platform**: Cross-platform Python package (Linux, macOS, Windows)
**Project Type**: single (library extension to existing steadytext package)
**Performance Goals**: Match existing provider performance, API call latency dependent on OpenRouter service
**Constraints**: Must follow existing provider patterns, requires unsafe_mode=True, deterministic fallback behavior
**Scale/Scope**: Single provider class addition, ~200-300 LOC, follows existing 4-provider pattern

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 1 (extending existing steadytext package)
- Using framework directly? ✓ (direct HTTP calls to OpenRouter API, no wrapper frameworks)
- Single data model? ✓ (reuses existing provider model pattern)
- Avoiding patterns? ✓ (follows existing provider registry pattern, no additional complexity)

**Architecture**:
- EVERY feature as library? ✓ (OpenRouter provider as part of steadytext.providers library)
- Libraries listed: steadytext.providers (remote model provider registry and implementations)
- CLI per library: ✓ (existing CLI supports all providers via "provider:model" syntax)
- Library docs: llms.txt format planned? ✓ (follows existing provider documentation pattern)

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? ✓ (tests written first, must fail, then implement)
- Git commits show tests before implementation? ✓ (will follow TDD approach)
- Order: Contract→Integration→E2E→Unit strictly followed? ✓ (provider contract tests → integration with registry → unit tests)
- Real dependencies used? ✓ (actual OpenRouter API calls in integration tests)
- Integration tests for: new libraries, contract changes, shared schemas? ✓ (provider registry integration)
- FORBIDDEN: Implementation before test, skipping RED phase ✓ (will strictly follow)

**Observability**:
- Structured logging included? ✓ (follows existing provider logging patterns)
- Frontend logs → backend? N/A (library package, no frontend)
- Error context sufficient? ✓ (detailed error messages for API failures, key validation)

**Versioning**:
- Version number assigned? ✓ (2025.8.27 → 2025.8.28 or next date-based version)
- BUILD increments on every change? ✓ (date-based versioning for each release)
- Breaking changes handled? ✓ (purely additive change, no breaking changes to existing API)

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure]
```

**Structure Decision**: Option 1 (Single project) - Adding provider to existing steadytext package structure

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

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
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `/scripts/update-agent-context.sh [claude|gemini|copilot]` for your AI assistant
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P]
- Each user story → integration test task
- Implementation tasks to make tests pass

**OpenRouter-Specific Task Strategy**:
- Provider contract tests → test_openrouter_provider_contract.py [P]
- Registry integration tests → test_openrouter_registry_integration.py [P]
- Error handling tests → test_openrouter_errors.py [P]
- CLI integration tests → test_openrouter_cli.py
- OpenRouter provider implementation → steadytext/providers/openrouter.py
- Registry updates → steadytext/providers/registry.py modifications
- Documentation updates → README.md, docs/unsafe-mode.md

**Ordering Strategy**:
- TDD order: Contract tests → Integration tests → Unit tests → Implementation
- Dependency order: Base classes → Provider → Registry integration → CLI support
- Mark [P] for parallel execution (independent test files)

**Estimated Output**: 15-20 numbered, ordered tasks in tasks.md focusing on provider pattern extension

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
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) - research.md created
- [x] Phase 1: Design complete (/plan command) - data-model.md, contracts/, quickstart.md, CLAUDE.md updated
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented (No deviations found)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
# Tasks: PostgreSQL Prompt Registry

**Input**: Design documents from `/workspace/specs/001-to-the-postgres/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: DB, middleware, logging
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- PostgreSQL extension paths: `pg_steadytext/sql/`, `pg_steadytext/python/`, `pg_steadytext/test/pgtap/`
- Version-based SQL files following pg_steadytext pattern
- Python modules in `pg_steadytext/python/`

## Phase 3.1: Setup
- [ ] T001 Update Makefile to include Jinja2 dependency in install-python-deps target
- [ ] T002 Create migration script pg_steadytext/sql/pg_steadytext--2025.8.26--2025.9.6.sql
- [ ] T003 Create main version file pg_steadytext/sql/pg_steadytext--2025.9.6.sql with full schema
- [ ] T004 [P] Update pg_steadytext.control with new version 2025.9.6

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (pgTAP)
- [ ] T005 [P] Test steadytext_prompt_create function in pg_steadytext/test/pgtap/17_prompt_registry.sql
- [ ] T006 [P] Test steadytext_prompt_update function in pg_steadytext/test/pgtap/17_prompt_registry.sql
- [ ] T007 [P] Test steadytext_prompt_get function in pg_steadytext/test/pgtap/17_prompt_registry.sql
- [ ] T008 [P] Test steadytext_prompt_render function in pg_steadytext/test/pgtap/17_prompt_registry.sql
- [ ] T009 [P] Test steadytext_prompt_list function in pg_steadytext/test/pgtap/17_prompt_registry.sql
- [ ] T010 [P] Test steadytext_prompt_versions function in pg_steadytext/test/pgtap/17_prompt_registry.sql
- [ ] T011 [P] Test steadytext_prompt_delete function in pg_steadytext/test/pgtap/17_prompt_registry.sql

### Integration Tests
- [ ] T012 Test complete prompt lifecycle (create→update→render) in pg_steadytext/test/pgtap/17_prompt_registry.sql
- [ ] T013 Test version tracking across multiple updates in pg_steadytext/test/pgtap/17_prompt_registry.sql
- [ ] T014 Test Jinja2 template validation on invalid syntax in pg_steadytext/test/pgtap/17_prompt_registry.sql
- [ ] T015 Test missing required variables error handling in pg_steadytext/test/pgtap/17_prompt_registry.sql

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Database Schema
- [ ] T016 Create steadytext_prompts table in pg_steadytext/sql/pg_steadytext--2025.9.6.sql
- [ ] T017 Create steadytext_prompt_versions table in pg_steadytext/sql/pg_steadytext--2025.9.6.sql
- [ ] T018 Add indexes and constraints for prompt tables in pg_steadytext/sql/pg_steadytext--2025.9.6.sql

### Python Module
- [ ] T019 [P] Create prompt_registry.py with Jinja2 template validation in pg_steadytext/python/prompt_registry.py
- [ ] T020 Implement _validate_jinja2_template function in pg_steadytext/python/prompt_registry.py
- [ ] T021 Implement template rendering with variable substitution in pg_steadytext/python/prompt_registry.py
- [ ] T022 Add template caching mechanism using GD in pg_steadytext/python/prompt_registry.py

### SQL Functions
- [ ] T023 Implement steadytext_prompt_create function in pg_steadytext/sql/pg_steadytext--2025.9.6.sql
- [ ] T024 Implement steadytext_prompt_update function in pg_steadytext/sql/pg_steadytext--2025.9.6.sql
- [ ] T025 Implement steadytext_prompt_get function in pg_steadytext/sql/pg_steadytext--2025.9.6.sql
- [ ] T026 Implement steadytext_prompt_render function in pg_steadytext/sql/pg_steadytext--2025.9.6.sql
- [ ] T027 Implement steadytext_prompt_list function in pg_steadytext/sql/pg_steadytext--2025.9.6.sql
- [ ] T028 Implement steadytext_prompt_versions function in pg_steadytext/sql/pg_steadytext--2025.9.6.sql
- [ ] T029 Implement steadytext_prompt_delete function in pg_steadytext/sql/pg_steadytext--2025.9.6.sql

### Helper Functions
- [ ] T030 Implement _get_next_version helper function in pg_steadytext/sql/pg_steadytext--2025.9.6.sql
- [ ] T031 Add dynamic schema resolution for all prompt functions in pg_steadytext/sql/pg_steadytext--2025.9.6.sql

### Aliases
- [ ] T032 Create st_* short aliases for all prompt functions in pg_steadytext/sql/pg_steadytext--2025.9.6.sql

## Phase 3.4: Integration
- [ ] T033 Integrate prompt_registry.py with _steadytext_init_python function
- [ ] T034 Add prompt registry to extension initialization
- [ ] T035 Test with existing pg_steadytext daemon and cache infrastructure
- [ ] T036 Verify TimescaleDB compatibility with @extschema@ pattern

## Phase 3.5: Polish
- [ ] T037 [P] Add AIDEV-NOTE comments to prompt_registry.py
- [ ] T038 [P] Update pg_steadytext/CLAUDE.md with prompt registry documentation
- [ ] T039 [P] Add performance tests for template rendering < 1ms
- [ ] T040 [P] Create example prompts in quickstart documentation
- [ ] T041 Run complete pgTAP test suite with STEADYTEXT_USE_MINI_MODELS=true
- [ ] T042 Test installation and upgrade from version 2025.8.26

## Dependencies
- Setup (T001-T004) must complete first
- Tests (T005-T015) before any implementation (T016-T032)
- Database schema (T016-T018) before SQL functions (T023-T029)
- Python module (T019-T022) before SQL functions that use it
- Core implementation (T016-T032) before integration (T033-T036)
- Everything before polish (T037-T042)

## Parallel Example
```bash
# Phase 3.2 - Launch all pgTAP contract tests together:
Task: "Test steadytext_prompt_create function in pg_steadytext/test/pgtap/17_prompt_registry.sql"
Task: "Test steadytext_prompt_update function in pg_steadytext/test/pgtap/17_prompt_registry.sql"
Task: "Test steadytext_prompt_get function in pg_steadytext/test/pgtap/17_prompt_registry.sql"
Task: "Test steadytext_prompt_render function in pg_steadytext/test/pgtap/17_prompt_registry.sql"
Task: "Test steadytext_prompt_list function in pg_steadytext/test/pgtap/17_prompt_registry.sql"
Task: "Test steadytext_prompt_versions function in pg_steadytext/test/pgtap/17_prompt_registry.sql"
Task: "Test steadytext_prompt_delete function in pg_steadytext/test/pgtap/17_prompt_registry.sql"

# Phase 3.5 - Launch all documentation updates together:
Task: "Add AIDEV-NOTE comments to prompt_registry.py"
Task: "Update pg_steadytext/CLAUDE.md with prompt registry documentation"
Task: "Add performance tests for template rendering < 1ms"
Task: "Create example prompts in quickstart documentation"
```

## Notes
- All pgTAP tests go in single file (17_prompt_registry.sql) but test different functions
- Use dynamic schema resolution pattern from existing pg_steadytext code
- Follow DROP TABLE IF EXISTS pattern for extension tables (not CREATE TABLE IF NOT EXISTS)
- Jinja2 dependency handled via pip in Makefile (similar to SteadyText)
- Template caching in GD for session-level performance
- Maintain backward compatibility - this is an additive enhancement

## Validation Checklist
- [x] All 7 contract functions have test tasks
- [x] Both entities (prompts, versions) have table creation tasks
- [x] All SQL functions have implementation tasks
- [x] Integration scenarios from quickstart are covered
- [x] TDD order enforced (tests before implementation)
- [x] Parallel execution properly marked for independent tasks
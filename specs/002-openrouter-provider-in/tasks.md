# Tasks: OpenRouter Provider Support

**Input**: Design documents from `/specs/002-openrouter-provider-in/`
**Prerequisites**: plan.md (✓), research.md (✓), data-model.md (✓), contracts/ (✓)

## Execution Flow (main)
```
1. Load plan.md from feature directory ✓
   → Tech stack: Python 3.10-3.13, httpx/requests, pytest
   → Structure: Single project (library extension)
2. Load optional design documents: ✓
   → data-model.md: OpenRouterProvider, OpenRouterResponse, Error classes
   → contracts/: OpenRouterProviderContract, RegistryIntegrationContract
   → research.md: HTTP client decision, error handling strategy
3. Generate tasks by category: ✓
   → Setup: Dependencies, linting
   → Tests: Contract tests, integration tests
   → Core: Provider class, error classes, response parsing
   → Integration: Registry integration, CLI support
   → Polish: Unit tests, documentation updates
4. Apply task rules: ✓
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...) ✓
6. Generate dependency graph ✓
7. Create parallel execution examples ✓
8. Validate task completeness: ✓
   → All contracts have tests ✓
   → All entities have implementation ✓
   → All integration points covered ✓
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `steadytext/`, `tests/` at repository root
- Following existing SteadyText provider pattern

## Phase 3.1: Setup
- [ ] T001 Install httpx dependency for OpenRouter HTTP client in pyproject.toml
- [ ] T002 [P] Configure linting rules for new provider code following project standards

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests [P] - Independent Files
- [ ] T003 [P] Contract test OpenRouterProvider initialization in tests/test_openrouter_provider_contract.py
- [ ] T004 [P] Contract test OpenRouterProvider.is_available() in tests/test_openrouter_availability_contract.py
- [ ] T005 [P] Contract test OpenRouterProvider.generate() in tests/test_openrouter_generate_contract.py
- [ ] T006 [P] Contract test OpenRouterProvider.embed() in tests/test_openrouter_embed_contract.py
- [ ] T007 [P] Contract test OpenRouterProvider.get_supported_models() in tests/test_openrouter_models_contract.py

### Integration Tests [P] - Independent Files
- [ ] T008 [P] Integration test OpenRouter registry integration in tests/test_openrouter_registry_integration.py
- [ ] T009 [P] Integration test OpenRouter CLI support in tests/test_openrouter_cli_integration.py
- [ ] T010 [P] Integration test OpenRouter error handling and fallbacks in tests/test_openrouter_error_integration.py
- [ ] T011 [P] Integration test OpenRouter API key validation in tests/test_openrouter_auth_integration.py

### User Story Tests [P] - Independent Files
- [ ] T012 [P] Test basic text generation user scenario in tests/test_openrouter_generation_scenario.py
- [ ] T013 [P] Test streaming generation user scenario in tests/test_openrouter_streaming_scenario.py
- [ ] T014 [P] Test embedding generation user scenario in tests/test_openrouter_embedding_scenario.py
- [ ] T015 [P] Test error handling user scenario in tests/test_openrouter_error_scenario.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Base Error Classes [P] - Independent Files
- [ ] T016 [P] OpenRouter error hierarchy in steadytext/providers/openrouter_errors.py
- [ ] T017 [P] OpenRouter response parsing classes in steadytext/providers/openrouter_responses.py
- [ ] T018 [P] OpenRouter configuration class in steadytext/providers/openrouter_config.py

### Main Provider Implementation - Sequential (Same File)
- [ ] T019 OpenRouterProvider class foundation in steadytext/providers/openrouter.py
- [ ] T020 OpenRouterProvider.__init__() method with API key validation
- [ ] T021 OpenRouterProvider.is_available() method with API connectivity check
- [ ] T022 OpenRouterProvider._make_request() internal HTTP method
- [ ] T023 OpenRouterProvider.generate() method for text generation
- [ ] T024 OpenRouterProvider.embed() method for embeddings
- [ ] T025 OpenRouterProvider.get_supported_models() method
- [ ] T026 OpenRouterProvider streaming support implementation
- [ ] T027 OpenRouterProvider error handling and deterministic fallbacks

## Phase 3.4: Integration

### Registry Integration - Sequential (Same File)
- [ ] T028 Add OpenRouterProvider import to steadytext/providers/registry.py
- [ ] T029 Add "openrouter" entry to PROVIDER_REGISTRY in steadytext/providers/registry.py
- [ ] T030 Add OPENROUTER_API_KEY validation to get_provider() in steadytext/providers/registry.py
- [ ] T031 Update provider constructor logic for OpenRouter in steadytext/providers/registry.py

### Package Integration [P] - Independent Files
- [ ] T032 [P] Export OpenRouter classes in steadytext/providers/__init__.py
- [ ] T033 [P] Update main package imports for OpenRouter in steadytext/__init__.py

## Phase 3.5: Polish

### Unit Tests [P] - Independent Files
- [ ] T034 [P] Unit tests for OpenRouter error classes in tests/test_openrouter_errors_unit.py
- [ ] T035 [P] Unit tests for OpenRouter response parsing in tests/test_openrouter_responses_unit.py
- [ ] T036 [P] Unit tests for OpenRouter configuration in tests/test_openrouter_config_unit.py
- [ ] T037 [P] Unit tests for OpenRouter HTTP client logic in tests/test_openrouter_http_unit.py

### Documentation Updates [P] - Independent Files
- [ ] T038 [P] Update README.md with OpenRouter provider documentation
- [ ] T039 [P] Update docs/unsafe-mode.md with OpenRouter examples
- [ ] T040 [P] Add OpenRouter to provider list in docs/api/generation.md
- [ ] T041 [P] Update CLI help text to include OpenRouter examples

### Final Validation
- [ ] T042 Run existing test suite to ensure no regressions
- [ ] T043 Test OpenRouter provider with real API key (manual validation)
- [ ] T044 Execute quickstart.md scenarios to validate user experience

## Dependencies

### Critical Dependencies (TDD Enforcement)
- **All Tests (T003-T015) MUST complete before ANY implementation (T016-T027)**
- Tests must fail initially to prove TDD compliance

### Implementation Dependencies
- T016-T018 (base classes) before T019-T027 (main provider)
- T019 (foundation) blocks T020-T027 (methods)
- T028-T031 (registry) requires T019-T027 (provider) to be complete
- T032-T033 (package integration) requires T019-T031 to be complete

### Polish Dependencies
- T034-T037 (unit tests) after T016-T027 (implementation)
- T038-T041 (docs) can run parallel with implementation
- T042-T044 (validation) requires all implementation complete

## Parallel Execution Examples

### Contract Tests Launch (T003-T007)
```bash
# Launch all contract tests together:
Task: "Contract test OpenRouterProvider initialization in tests/test_openrouter_provider_contract.py"
Task: "Contract test OpenRouterProvider.is_available() in tests/test_openrouter_availability_contract.py"
Task: "Contract test OpenRouterProvider.generate() in tests/test_openrouter_generate_contract.py"
Task: "Contract test OpenRouterProvider.embed() in tests/test_openrouter_embed_contract.py"
Task: "Contract test OpenRouterProvider.get_supported_models() in tests/test_openrouter_models_contract.py"
```

### Integration Tests Launch (T008-T011)
```bash
# Launch all integration tests together:
Task: "Integration test OpenRouter registry integration in tests/test_openrouter_registry_integration.py"
Task: "Integration test OpenRouter CLI support in tests/test_openrouter_cli_integration.py"
Task: "Integration test OpenRouter error handling and fallbacks in tests/test_openrouter_error_integration.py"
Task: "Integration test OpenRouter API key validation in tests/test_openrouter_auth_integration.py"
```

### Base Classes Launch (T016-T018)
```bash
# Launch base class implementations together:
Task: "OpenRouter error hierarchy in steadytext/providers/openrouter_errors.py"
Task: "OpenRouter response parsing classes in steadytext/providers/openrouter_responses.py"
Task: "OpenRouter configuration class in steadytext/providers/openrouter_config.py"
```

### Documentation Updates Launch (T038-T041)
```bash
# Launch documentation updates together:
Task: "Update README.md with OpenRouter provider documentation"
Task: "Update docs/unsafe-mode.md with OpenRouter examples"
Task: "Add OpenRouter to provider list in docs/api/generation.md"
Task: "Update CLI help text to include OpenRouter examples"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing (critical for TDD)
- Commit after each task
- Follow existing SteadyText provider patterns (OpenAI, Cerebras as examples)
- All API calls must include deterministic fallback behavior
- Use existing unsafe mode patterns for consistency

## Task Generation Rules Applied

1. **From Contracts**:
   - OpenRouterProviderContract → T003-T007 (contract tests) [P]
   - RegistryIntegrationContract → T008, T028-T031 (registry integration)

2. **From Data Model**:
   - OpenRouterProvider → T019-T027 (main implementation)
   - OpenRouterResponse → T017 (response parsing) [P]
   - Error classes → T016 (error hierarchy) [P]
   - OpenRouterConfig → T018 (configuration) [P]

3. **From User Stories (quickstart.md)**:
   - Basic generation → T012 (generation scenario) [P]
   - Streaming → T013 (streaming scenario) [P]
   - Embeddings → T014 (embedding scenario) [P]
   - Error handling → T015 (error scenario) [P]

4. **Ordering Applied**:
   - Setup (T001-T002) → Tests (T003-T015) → Models (T016-T018) →
   - Services (T019-T027) → Integration (T028-T033) → Polish (T034-T044)

## Validation Checklist

- [x] All contracts have corresponding tests (OpenRouterProviderContract → T003-T007)
- [x] All entities have model tasks (OpenRouterProvider → T019-T027, Responses → T017, etc.)
- [x] All tests come before implementation (T003-T015 before T016-T027)
- [x] Parallel tasks truly independent (checked file paths)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Registry integration properly sequenced (T028-T031 after T019-T027)
- [x] CLI integration tested (T009, T032-T033)
- [x] Documentation coverage complete (T038-T041)

## Success Criteria
Upon completion of all tasks:
1. OpenRouter provider fully integrated into SteadyText
2. All existing tests continue to pass
3. New provider follows existing patterns (OpenAI, Cerebras)
4. Comprehensive test coverage for new functionality
5. Complete documentation for users
6. CLI support for "openrouter:" model prefix
7. Deterministic fallback behavior maintained
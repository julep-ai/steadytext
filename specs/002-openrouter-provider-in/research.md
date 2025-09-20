# Research: OpenRouter Provider Implementation

## OpenRouter API Analysis

### Decision: Use OpenRouter's ChatCompletion API
**Rationale**: OpenRouter provides OpenAI-compatible ChatCompletion API that supports both text generation and embeddings through a unified interface. This aligns with existing provider patterns in SteadyText.

**Alternatives considered**:
- Direct model-specific APIs: Rejected due to complexity of supporting multiple model formats
- Custom protocol: Rejected as OpenRouter already provides standardized interface

### Decision: HTTP Client Library
**Rationale**: Use `httpx` for async HTTP client capabilities, consistent with modern Python HTTP patterns. Fallback to `requests` if httpx unavailable.

**Alternatives considered**:
- `requests` only: Rejected due to lack of async support for future improvements
- `aiohttp`: Rejected to minimize new dependencies
- Built-in `urllib`: Rejected due to complexity and lack of modern features

### Decision: Authentication Pattern
**Rationale**: Follow existing pattern - API key from environment variable (OPENROUTER_API_KEY), with optional override in constructor. Consistent with OpenAI, Cerebras providers.

**Alternatives considered**:
- Config file approach: Rejected to maintain simplicity
- Multiple auth methods: Rejected to avoid complexity

## OpenRouter Model Format Analysis

### Decision: Support OpenRouter's Model Naming Convention
**Rationale**: OpenRouter uses format like "anthropic/claude-3.5-sonnet", "openai/gpt-4". SteadyText will use "openrouter:anthropic/claude-3.5-sonnet" pattern.

**Alternatives considered**:
- Flatten model names: Rejected as it loses provider context within OpenRouter
- Custom mapping: Rejected as it adds maintenance burden

## Error Handling Research

### Decision: OpenRouter-Specific Error Mapping
**Rationale**: Map OpenRouter HTTP error codes to appropriate SteadyText exceptions while maintaining deterministic fallback behavior.

**Key Error Cases**:
- 401: Invalid API key → RuntimeError with clear message
- 429: Rate limiting → Implement retry with exponential backoff, fallback to deterministic
- 404: Model not found → ValueError with available models suggestion
- 503: Service unavailable → Warning log, fallback to deterministic

**Alternatives considered**:
- Generic error handling: Rejected as it provides poor user experience
- No fallback: Rejected as it violates SteadyText's "Never Fails" principle

## Integration Points Research

### Decision: Minimal Registry Changes
**Rationale**: Add "openrouter" key to PROVIDER_REGISTRY, implement OpenRouterProvider class following RemoteModelProvider interface. No changes to core logic needed.

**Integration Requirements**:
1. `steadytext/providers/openrouter.py` - New provider implementation
2. `steadytext/providers/registry.py` - Add registry entry and API key validation
3. `steadytext/providers/__init__.py` - Export new provider
4. Tests following existing pattern in `tests/test_*_provider.py`

### Decision: Configuration Dependencies
**Rationale**: OpenRouter API is HTTP-based, only requires standard HTTP client. No model downloads or special dependencies beyond what's already available.

**Dependencies**:
- httpx or requests (already available in ecosystem)
- json (built-in)
- os, logging (built-in)

## Performance Considerations

### Decision: Connection Pooling and Timeouts
**Rationale**: Use httpx's built-in connection pooling. Set reasonable timeouts (30s connect, 120s read) to balance responsiveness with model processing time.

**Alternatives considered**:
- No timeout: Rejected due to hanging requests risk
- Very short timeout: Rejected as some models may need longer processing time

### Decision: Streaming Support
**Rationale**: Implement streaming support for text generation to match existing provider capabilities and improve user experience for long responses.

**Implementation**: Use OpenRouter's streaming API with Server-Sent Events (SSE) parsing.

## Testing Strategy Research

### Decision: Multi-Level Testing Approach
**Rationale**: Follow existing provider testing patterns with mock and integration tests.

**Test Levels**:
1. Unit tests with mocked HTTP responses
2. Integration tests with actual OpenRouter API (when API key available)
3. Registry integration tests
4. CLI integration tests

**Mock Strategy**: Mock httpx/requests responses for predictable unit testing, use pytest-httpx for clean mocking.

## Documentation Requirements

### Decision: Follow Existing Provider Documentation Pattern
**Rationale**: Document OpenRouter provider in same style as existing providers, update main docs to include OpenRouter in provider list.

**Documentation Updates Needed**:
- README.md: Add OpenRouter to provider list
- docs/unsafe-mode.md: Add OpenRouter examples
- Provider docstrings: Follow existing format
- CLI help: Automatically includes new provider

## Security Considerations

### Decision: Standard API Key Handling
**Rationale**: Follow same security practices as existing providers - environment variable storage, no key logging, validate key format.

**Security Measures**:
- API key validation before first request
- No logging of API keys or request/response content
- Clear error messages without exposing sensitive data
- Follow OpenRouter's security best practices

**Alternatives considered**:
- Key encryption: Rejected as overly complex for this use case
- Multiple auth methods: Rejected to maintain simplicity
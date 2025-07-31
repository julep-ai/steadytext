# pg_steadytext Changelog

All notable changes to the pg_steadytext PostgreSQL extension will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.4] - 2025-01-31

### Added
- **Unsafe Mode Support:** Added `unsafe_mode` parameter to all generation functions
  - New parameter allows explicit opt-in for remote AI models (containing ':' in model name)
  - Remote models like `openai:gpt-4o-mini` require `unsafe_mode := TRUE`
  - Local models work with any `unsafe_mode` setting
  - Security enhancement: prevents accidental use of non-deterministic remote models

### Changed
- Updated all SQL generation functions to accept `unsafe_mode BOOLEAN DEFAULT FALSE`
- Modified `daemon_connector.py` to propagate `unsafe_mode` parameter to SteadyText library
- Enhanced validation: models with ':' in name are rejected unless `unsafe_mode=TRUE`

### Security
- Remote model usage now requires explicit permission via `unsafe_mode` parameter
- Prevents unintentional loss of determinism when using remote AI providers

## [1.4.3] - 2025-01-30

### Fixed
- Fixed parameter name mismatch in direct generation fallback
  - Changed `max_tokens` to `max_new_tokens` when calling steadytext.generate() directly
  - This fixes the error: `generate() got an unexpected keyword argument 'max_tokens'`
  - Affects all SQL versions (1.4.0, 1.4.1, 1.4.2)
  - Note: Structured generation functions correctly use `max_tokens` as the daemon API expects

- Fixed JSON configuration value handling to prevent SQL syntax errors
  - Properly escape single quotes in JSON config values
  - Handle NULL values correctly when current_setting doesn't exist

### Added
- **Reranking Support:** Propagated reranking functions to all SQL version files
  - Added missing `pg_steadytext--1.3.0.sql` with complete reranking implementation
  - Updated queue table to support 'rerank' and 'batch_rerank' request types
  - Fixed header comments to reflect reranking capabilities

### Documentation
- Enhanced CLAUDE.md with detailed fix documentation
- Added AIDEV-NOTE comments explaining API inconsistencies

## [1.4.2] - 2025-01-25

### Fixed
- Fixed `AttributeError: 'SteadyTextConnector' object has no attribute 'start_daemon'` by adding public `start_daemon()` method to SteadyTextConnector class in daemon_connector.py
- This resolves compatibility issues between SQL files and the Python module where pg_steadytext--1.4.1.sql calls `connector.start_daemon()` but only the private `_start_daemon()` method existed

### Technical Details
- Added public wrapper method `start_daemon()` that calls the existing private `_start_daemon()` method
- No SQL changes required - this is a Python module fix only
- Maintains backward compatibility with existing installations

## [1.4.1] - Previous Release

### Changed
- Updated to use IMMUTABLE functions with read-only cache access
- Removed cache updates from IMMUTABLE functions to comply with PostgreSQL requirements
- Changed from frecency-based to age-based cache eviction

## [1.4.0] - Previous Release

### Added
- Automatic cache eviction using pg_cron
- Enhanced cache statistics and analysis functions
- Python package auto-installation in Makefile

### Changed
- Improved error messages for missing Python packages
- Enhanced Python path detection and module loading
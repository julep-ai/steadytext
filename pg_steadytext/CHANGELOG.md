# pg_steadytext Changelog

All notable changes to the pg_steadytext PostgreSQL extension will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Versioning Scheme

As of version 2025.8.16, pg_steadytext uses **date-based versioning** in the format `yyyy.mm.dd` (without zero-padding for month and day). This change was made to better reflect the rapid pace of model improvements and feature development, where the previous semantic versioning approach of tying major version bumps to model changes was no longer feasible.

- **Format:** `yyyy.mm.dd` (e.g., `2025.8.16`, `2025.12.3`)
- **Benefits:** Clear indication of release currency and maintenance activity
- **Rationale:** Supported models and features are evolving quickly, making date-based versioning more practical

## [2025.8.26] - 2025-08-26

### Fixed
- **Complete Schema Qualification:** Extended schema qualification to ALL functions that access extension tables
  - Added dynamic schema resolution using `plpy.execute()` and `plpy.quote_ident()` to:
    - `steadytext_daemon_start()` and `steadytext_daemon_stop()` - daemon control functions
    - `steadytext_generate_json()`, `steadytext_generate_regex()`, `steadytext_generate_choice()` - structured generation
    - `steadytext_summarize_text()` and `steadytext_summarize_finalize()` - summarization helpers
  - Updated all SQL alias functions (`st_daemon_start()`, `st_daemon_stop()`) to use `@extschema@` placeholder
  - Ensures complete compatibility with TimescaleDB continuous aggregates and non-standard search paths
  - Resolves all remaining issues from #95 and #100

### Changed
- **Migration Script:** Created comprehensive migration from v2025.8.17 to v2025.8.26
  - Includes all 17 function updates with proper schema qualification
  - Properly manages extension membership for replaced functions
  - Ensures smooth upgrade path for existing installations

### Technical Notes
- Functions now dynamically determine the extension schema at runtime
- PL/Python functions use `plpy.quote_ident()` for safe schema identifier quoting
- SQL alias functions use `@extschema@` which PostgreSQL replaces with the actual extension schema
- This pattern should be used for all future functions that access extension tables

## [2025.8.17] - 2025-08-17

### Added
- **Enhanced AI Summarization Functions:** Complete overhaul of summarization capabilities
  - Renamed all `ai_*` functions to `steadytext_*` with automatic `st_*` aliases for consistency
  - Added `model` and `unsafe_mode` parameters to all summarization functions for remote model support
  - Increased default `max_facts` from 5 to 10 for better fact extraction
  - Support for remote models like OpenAI GPT-4o-mini with explicit `unsafe_mode := TRUE`

### Fixed  
- **Schema Qualification for TimescaleDB:** Fixed issue #95 where functions failed in continuous aggregates
  - Added explicit schema qualification using `@extschema@` placeholder to all table references
  - Ensures functions work correctly regardless of search_path settings
  - Critical fix for TimescaleDB continuous aggregate compatibility
  
- **Python Scoping in Aggregate Functions:** Fixed NameError in PL/Python aggregate functions
  - Resolved issue where reassigning argument variables caused scoping problems
  - Fixed `steadytext_summarize` accumulate, combine, and finalize functions
  - Prevents "No data to summarize" errors reported in PR #100

### Changed
- Simplified `steadytext_extract_facts` function to prevent crashes
- Updated all migration scripts to use proper schema qualification pattern

## [2025.8.16] - 2025-08-16

### Changed
- Version bump with date-based versioning
- Updated all version references throughout the extension and documentation
- Aligned PostgreSQL extension version with main Python package

## [2025.8.15] - 2025-08-15

### Changed
- **BREAKING:** Switched from semantic versioning to date-based versioning (yyyy.mm.dd format)
- All version references updated throughout the extension
- Migration path provided from version 1.4.6

### Notes
- This is a version numbering change only - no functional changes to the extension
- Future releases will use the yyyy.mm.dd format
- The previous semantic versioning policy of "new major version for model changes" is discontinued

## [1.4.6] - 2025-08-14

### Added
- **Unsafe Mode Support for Embeddings:** Extended unsafe_mode support to embedding functions
  - Added `model` and `unsafe_mode` parameters to `steadytext_embed()`, `steadytext_embed_cached()`, and `steadytext_embed_async()`
  - Remote embedding models (e.g., `openai:text-embedding-3-small`) now supported with `unsafe_mode := TRUE`
  - Consistent with generation functions' unsafe_mode behavior
  - Updated all embedding-related aliases (`st_embed`, `st_embed_cached`, `st_embed_async`)

### Changed
- Modified `daemon_connector.py` embed() method to accept model and unsafe_mode parameters
- Enhanced cache key generation to include model name for proper caching of remote embeddings
- Skip daemon for remote embedding models to improve performance

### Security
- Remote embedding models require explicit `unsafe_mode=TRUE` parameter, consistent with generation functions
- Prevents accidental use of non-deterministic remote embedding providers

## [1.4.5] - 2025-08-02

### Changed
- **Version Bump:** Updated version number for maintenance release
- Updated SteadyText library dependency to >= 2.6.1

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
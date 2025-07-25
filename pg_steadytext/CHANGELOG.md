# pg_steadytext Changelog

All notable changes to the pg_steadytext PostgreSQL extension will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
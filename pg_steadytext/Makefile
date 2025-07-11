# Makefile for pg_steadytext PostgreSQL extension
# AIDEV-NOTE: This Makefile builds and installs the pg_steadytext extension

EXTENSION = pg_steadytext
EXTVERSION = 1.2.0

# SQL scripts
DATA = sql/$(EXTENSION)--$(EXTVERSION).sql
# AIDEV-NOTE: Do not use DATA_built here - it causes duplicate installation attempts

# Python modules to install
PYTHON_MODULES = python/__init__.py \
                 python/daemon_connector.py \
                 python/cache_manager.py \
                 python/security.py \
                 python/config.py \
                 python/worker.py

# Regression tests
REGRESS = basic cache queue daemon
REGRESS_OPTS = --inputdir=test --outputdir=test

# Python module installation directory
MODULE_PATHNAME = $(libdir)/$(EXTENSION)

# PostgreSQL extension build system
PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)

# AIDEV-NOTE: Override libdir to use PostgreSQL's pkglibdir for consistency
# This ensures Python modules are installed where PostgreSQL expects them
pkglibdir = $(shell $(PG_CONFIG) --pkglibdir)

include $(PGXS)

# AIDEV-NOTE: Custom targets for Python module installation
install: install-python check-deps

install-python:
	@echo "Installing Python modules for pg_steadytext..."
	@echo "pkglibdir: $(pkglibdir)"
	@echo "DESTDIR: $(DESTDIR)"
	@echo "Target directory: $(DESTDIR)$(pkglibdir)/$(EXTENSION)/python"
	@mkdir -p '$(DESTDIR)$(pkglibdir)/$(EXTENSION)/python'
	@for module in $(PYTHON_MODULES); do \
		echo "  Installing $$module to $(DESTDIR)$(pkglibdir)/$(EXTENSION)/python/"; \
		$(INSTALL_DATA) $$module '$(DESTDIR)$(pkglibdir)/$(EXTENSION)/python/'; \
	done
	@echo "Python modules installed to: $(DESTDIR)$(pkglibdir)/$(EXTENSION)/python/"

# Check dependencies before installation
check-deps:
	@echo "Checking dependencies..."
	@echo -n "  Python3: "
	@which python3 >/dev/null 2>&1 && echo "OK" || (echo "MISSING - Install python3"; exit 1)
	@echo -n "  plpython3u: "
	@$(PG_CONFIG) --version >/dev/null 2>&1 && echo "OK (checking at runtime)" || echo "WARNING"
	@echo -n "  vector (pgvector package): "
	@echo "OK (checking at runtime)"
	@echo -n "  SteadyText: "
	@python3 -c "import steadytext; print('OK')" 2>/dev/null || \
		(echo "MISSING - Run: pip3 install steadytext"; exit 1)

# Development targets
.PHONY: test
test: install
	@echo "Running regression tests..."
	$(pg_regress_installcheck) $(REGRESS_OPTS) $(REGRESS)

.PHONY: test-pgtap
test-pgtap: install
	@echo "Running pgTAP tests..."
	@if [ -x ./run_pgtap_tests.sh ]; then \
		./run_pgtap_tests.sh; \
	else \
		echo "ERROR: run_pgtap_tests.sh not found or not executable"; \
		exit 1; \
	fi

.PHONY: test-pgtap-verbose
test-pgtap-verbose: install
	@echo "Running pgTAP tests (verbose)..."
	@if [ -x ./run_pgtap_tests.sh ]; then \
		./run_pgtap_tests.sh --verbose; \
	else \
		echo "ERROR: run_pgtap_tests.sh not found or not executable"; \
		exit 1; \
	fi

.PHONY: test-pgtap-tap
test-pgtap-tap: install
	@if [ -x ./run_pgtap_tests.sh ]; then \
		./run_pgtap_tests.sh --tap; \
	else \
		echo "ERROR: run_pgtap_tests.sh not found or not executable"; \
		exit 1; \
	fi

.PHONY: test-all
test-all: test test-pgtap test-python
	@echo "All tests completed!"

.PHONY: test-python
test-python:
	@echo "Testing Python modules..."
	@cd python && python3 -m pytest -v

# Clean targets
clean: clean-python

clean-python:
	@echo "Cleaning Python build artifacts..."
	@find python -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find python -name "*.pyc" -delete 2>/dev/null || true
	@find python -name "*.pyo" -delete 2>/dev/null || true

# Distribution targets
.PHONY: dist
dist:
	@echo "Creating distribution package..."
	@mkdir -p dist
	@git archive --format=tar.gz --prefix=$(EXTENSION)-$(EXTVERSION)/ \
		-o dist/$(EXTENSION)-$(EXTVERSION).tar.gz HEAD

# Documentation
.PHONY: docs
docs:
	@echo "Building documentation..."
	@if [ -f README.md ]; then \
		pandoc -f markdown -t html -o README.html README.md; \
		echo "Documentation built: README.html"; \
	fi

# PGXN release
.PHONY: pgxn-release
pgxn-release: dist
	@echo "Preparing PGXN release..."
	@if [ ! -f META.json ]; then \
		echo "ERROR: META.json not found"; \
		exit 1; \
	fi
	@echo "Ready for PGXN upload: dist/$(EXTENSION)-$(EXTVERSION).tar.gz"

# Development helpers
.PHONY: dev-install
dev-install: install
	@echo "Installing for development..."
	@echo "CREATE EXTENSION IF NOT EXISTS plpython3u CASCADE;" | psql -U postgres
	@echo "CREATE EXTENSION IF NOT EXISTS vector CASCADE;" | psql -U postgres
	@echo "DROP EXTENSION IF EXISTS $(EXTENSION) CASCADE;" | psql -U postgres
	@echo "CREATE EXTENSION $(EXTENSION);" | psql -U postgres
	@echo "Development installation complete!"

.PHONY: dev-test
dev-test: dev-install
	@echo "Running development tests..."
	@psql -U postgres -c "SELECT steadytext_version();"
	@psql -U postgres -c "SELECT steadytext_daemon_status();"
	@psql -U postgres -c "SELECT steadytext_generate('Hello world');"

# Uninstall
.PHONY: uninstall-complete
uninstall-complete: uninstall
	@echo "Removing Python modules..."
	@rm -rf '$(DESTDIR)$(pkglibdir)/$(EXTENSION)'

# Help target
.PHONY: help
help:
	@echo "pg_steadytext Makefile targets:"
	@echo "  make              - Build the extension"
	@echo "  make install      - Install the extension"
	@echo "  make check-deps   - Check dependencies"
	@echo "  make test         - Run regression tests"
	@echo "  make test-pgtap   - Run pgTAP tests"
	@echo "  make test-pgtap-verbose - Run pgTAP tests with verbose output"
	@echo "  make test-pgtap-tap - Run pgTAP tests with TAP output (for CI)"
	@echo "  make test-all     - Run all tests (regression, pgTAP, Python)"
	@echo "  make test-python  - Run Python unit tests"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make dist         - Create distribution package"
	@echo "  make docs         - Build documentation"
	@echo "  make dev-install  - Install for development"
	@echo "  make dev-test     - Quick development test"
	@echo "  make help         - Show this help"

# AIDEV-NOTE: Default target
all: check-deps
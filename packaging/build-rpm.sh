#!/bin/bash
# Build RPM packages for pg_steadytext

set -e

# Read version from META.json
VERSION=$(python3 -c "import json; print(json.load(open('../pg_steadytext/META.json'))['version'])")
PYTHON_VERSION=$(python3 -c "import tomllib; print(tomllib.load(open('../pyproject.toml', 'rb'))['project']['version'])")

# PostgreSQL versions to build for
PG_VERSIONS=${PG_VERSIONS:-"14 15 16 17"}

echo "Building RPM packages for pg_steadytext v${VERSION}"
echo "Python package version: ${PYTHON_VERSION}"

# Create build directory
BUILD_DIR="build/rpm"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/BUILD"
mkdir -p "$BUILD_DIR/RPMS"
mkdir -p "$BUILD_DIR/SOURCES"
mkdir -p "$BUILD_DIR/SPECS"
mkdir -p "$BUILD_DIR/SRPMS"

for PG_VERSION in $PG_VERSIONS; do
    echo "Building for PostgreSQL ${PG_VERSION}..."
    
    # Create spec file
    cat > "${BUILD_DIR}/SPECS/pg_steadytext-pg${PG_VERSION}.spec" << EOF
%define pgmajorversion ${PG_VERSION}
%define pginstdir /usr/pgsql-%{pgmajorversion}
%define sname pg_steadytext

Name:           %{sname}_%{pgmajorversion}
Version:        ${VERSION}
Release:        1%{?dist}
Summary:        Deterministic AI text generation for PostgreSQL
License:        MIT
URL:            https://github.com/julep-ai/steadytext
BuildArch:      noarch

Requires:       postgresql%{pgmajorversion}-server
Requires:       postgresql%{pgmajorversion}-plpython3
Requires:       python3-pip
Requires:       python3-virtualenv

%description
pg_steadytext provides PostgreSQL functions for deterministic text generation
and embedding creation using the SteadyText Python library. Features include
generate(), embed(), and various async functions for non-blocking AI operations.

%prep
# Nothing to prep

%build
# Nothing to build

%install
rm -rf %{buildroot}

# Create directories
mkdir -p %{buildroot}%{pginstdir}/share/extension
mkdir -p %{buildroot}/opt/steadytext/python
mkdir -p %{buildroot}/usr/lib/systemd/system
mkdir -p %{buildroot}/usr/share/doc/%{name}

# Copy extension files
cp ${PWD}/../pg_steadytext/*.control %{buildroot}%{pginstdir}/share/extension/
cp ${PWD}/../pg_steadytext/sql/*.sql %{buildroot}%{pginstdir}/share/extension/

# Copy Python files
cp -r ${PWD}/../pg_steadytext/python/* %{buildroot}/opt/steadytext/python/

# Copy service file
cp ${PWD}/../pg_steadytext/pg_steadytext_worker.service %{buildroot}/usr/lib/systemd/system/

# Copy documentation
cp ${PWD}/../pg_steadytext/README.md %{buildroot}/usr/share/doc/%{name}/
cp ${PWD}/../LICENSE %{buildroot}/usr/share/doc/%{name}/

%files
%defattr(-,root,root,-)
%{pginstdir}/share/extension/*
/opt/steadytext/python/*
/usr/lib/systemd/system/pg_steadytext_worker.service
%doc /usr/share/doc/%{name}/*

%post
# Create virtual environment and install dependencies
python3 -m venv /opt/steadytext/venv
/opt/steadytext/venv/bin/pip install --upgrade pip
/opt/steadytext/venv/bin/pip install steadytext

# Set permissions
chown -R postgres:postgres /opt/steadytext

# Reload systemd and enable worker service
systemctl daemon-reload
systemctl enable pg_steadytext_worker.service

echo "pg_steadytext installed successfully."
echo "To complete setup:"
echo "1. Create extension: CREATE EXTENSION pg_steadytext;"
echo "2. Start worker: systemctl start pg_steadytext_worker"

%preun
# Stop and disable worker service
systemctl stop pg_steadytext_worker.service || true
systemctl disable pg_steadytext_worker.service || true

%changelog
* $(date "+%a %b %d %Y") SteadyText Maintainers <maintainers@steadytext.ai> - ${VERSION}-1
- Initial RPM release
EOF
    
    # Build the RPM
    rpmbuild --define "_topdir ${PWD}/${BUILD_DIR}" \
             --define "_builddir ${PWD}" \
             -bb "${BUILD_DIR}/SPECS/pg_steadytext-pg${PG_VERSION}.spec"
    
    echo "Built RPM for PostgreSQL ${PG_VERSION}"
done

echo "All RPM packages built successfully!"
echo "Packages are in: ${BUILD_DIR}/RPMS/"
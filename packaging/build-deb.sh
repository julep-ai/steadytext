#!/bin/bash
# Build Debian packages for pg_steadytext

set -e

# Read version from META.json
VERSION=$(python3 -c "
import json, sys
try:
    print(json.load(open('../pg_steadytext/META.json'))['version'])
except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
    print(f'Error reading version from META.json: {e}', file=sys.stderr)
    sys.exit(1)
") || exit 1

PYTHON_VERSION=$(python3 -c "
import tomllib, sys
try:
    print(tomllib.load(open('../pyproject.toml', 'rb'))['project']['version'])
except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError) as e:
    print(f'Error reading version from pyproject.toml: {e}', file=sys.stderr)
    sys.exit(1)
") || exit 1

# PostgreSQL versions to build for
PG_VERSIONS=${PG_VERSIONS:-"14 15 16 17"}

echo "Building Debian packages for pg_steadytext v${VERSION}"
echo "Python package version: ${PYTHON_VERSION}"

# Create build directory
BUILD_DIR="build/deb"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

for PG_VERSION in $PG_VERSIONS; do
    echo "Building for PostgreSQL ${PG_VERSION}..."
    
    # Package name follows Debian convention
    PKG_NAME="postgresql-${PG_VERSION}-pg-steadytext"
    PKG_DIR="${BUILD_DIR}/${PKG_NAME}_${VERSION}"
    
    # Create package structure
    mkdir -p "${PKG_DIR}/DEBIAN"
    mkdir -p "${PKG_DIR}/usr/share/postgresql/${PG_VERSION}/extension"
    mkdir -p "${PKG_DIR}/usr/lib/postgresql/${PG_VERSION}/lib"
    mkdir -p "${PKG_DIR}/usr/share/doc/${PKG_NAME}"
    mkdir -p "${PKG_DIR}/opt/steadytext/python"
    mkdir -p "${PKG_DIR}/etc/systemd/system"
    
    # Copy extension files
    cp ../pg_steadytext/*.control "${PKG_DIR}/usr/share/postgresql/${PG_VERSION}/extension/"
    cp ../pg_steadytext/sql/*.sql "${PKG_DIR}/usr/share/postgresql/${PG_VERSION}/extension/"
    
    # Copy Python files
    cp -r ../pg_steadytext/python/* "${PKG_DIR}/opt/steadytext/python/"
    
    # Copy service file
    cp ../pg_steadytext/pg_steadytext_worker.service "${PKG_DIR}/etc/systemd/system/"
    
    # Copy documentation
    cp ../pg_steadytext/README.md "${PKG_DIR}/usr/share/doc/${PKG_NAME}/"
    cp ../LICENSE "${PKG_DIR}/usr/share/doc/${PKG_NAME}/"
    
    # Create control file
    cat > "${PKG_DIR}/DEBIAN/control" << EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: database
Priority: optional
Architecture: all
Depends: postgresql-${PG_VERSION}, postgresql-plpython3-${PG_VERSION}, python3-pip, python3-venv
Maintainer: SteadyText Maintainers <maintainers@steadytext.ai>
Description: Deterministic AI text generation for PostgreSQL
 pg_steadytext provides PostgreSQL functions for deterministic text generation
 and embedding creation using the SteadyText Python library. Features include
 generate(), embed(), and various async functions for non-blocking AI operations.
Homepage: https://github.com/julep-ai/steadytext
EOF
    
    # Create postinst script
    cat > "${PKG_DIR}/DEBIAN/postinst" << 'EOF'
#!/bin/bash
set -e

# Create virtual environment and install dependencies
if ! python3 -m venv /opt/steadytext/venv; then
    echo "Failed to create virtual environment" >&2
    exit 1
fi
if ! /opt/steadytext/venv/bin/pip install --upgrade pip; then
    echo "Failed to upgrade pip" >&2
    exit 1
fi
if ! /opt/steadytext/venv/bin/pip install steadytext; then
    echo "Failed to install steadytext package" >&2
    exit 1
fi

# Set permissions
chown -R postgres:postgres /opt/steadytext

# Reload systemd and enable worker service
systemctl daemon-reload
systemctl enable pg_steadytext_worker.service

echo "pg_steadytext installed successfully."
echo "To complete setup:"
echo "1. Create extension: CREATE EXTENSION pg_steadytext;"
echo "2. Start worker: systemctl start pg_steadytext_worker"

exit 0
EOF
    chmod 755 "${PKG_DIR}/DEBIAN/postinst"
    
    # Create prerm script
    cat > "${PKG_DIR}/DEBIAN/prerm" << 'EOF'
#!/bin/bash
set -e

# Stop and disable worker service
systemctl stop pg_steadytext_worker.service || true
systemctl disable pg_steadytext_worker.service || true

exit 0
EOF
    chmod 755 "${PKG_DIR}/DEBIAN/prerm"
    
    # Build the package
    dpkg-deb --build "${PKG_DIR}" "${BUILD_DIR}/${PKG_NAME}_${VERSION}.deb"
    echo "Built: ${BUILD_DIR}/${PKG_NAME}_${VERSION}.deb"
done

echo "All Debian packages built successfully!"
echo "Packages are in: ${BUILD_DIR}/"
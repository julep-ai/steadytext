#!/bin/bash
# Prepare and upload pg_steadytext to PGXN

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

echo "Preparing pg_steadytext v${VERSION} for PGXN"

# Create build directory
BUILD_DIR="build/pgxn"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Create distribution directory
DIST_NAME="pg_steadytext-${VERSION}"
DIST_DIR="${BUILD_DIR}/${DIST_NAME}"
mkdir -p "$DIST_DIR"

# Copy all necessary files
cp -r ../pg_steadytext/* "$DIST_DIR/"
cp ../LICENSE "$DIST_DIR/"

# Ensure META.json is valid for PGXN
cd "$DIST_DIR"

# Validate META.json structure
python3 -c "
import json
import sys

with open('META.json', 'r') as f:
    meta = json.load(f)

# Required fields for PGXN
required = ['name', 'abstract', 'version', 'maintainer', 'license', 'provides']
missing = [field for field in required if field not in meta]

if missing:
    print(f'ERROR: META.json missing required fields: {missing}')
    sys.exit(1)

# Ensure provides section is correct
if 'pg_steadytext' not in meta.get('provides', {}):
    print('ERROR: META.json must have pg_steadytext in provides section')
    sys.exit(1)

print('META.json validation passed!')
"

# Create ZIP file for distribution
cd ..
zip -r "${DIST_NAME}.zip" "${DIST_NAME}"

echo "PGXN distribution package created: ${BUILD_DIR}/${DIST_NAME}.zip"
echo ""
echo "To upload to PGXN:"
echo "1. Create account at https://manager.pgxn.org/account/register"
echo "2. Authenticate: pgxn-client authenticate"
echo "3. Upload: pgxn-client upload ${BUILD_DIR}/${DIST_NAME}.zip"
echo ""
echo "Or manually upload at: https://manager.pgxn.org/upload"

# Generate Pigsty configuration
cat > "${BUILD_DIR}/pigsty-pg_steadytext.yaml" << EOF
# Pigsty extension configuration for pg_steadytext
# Add this to your pigsty.yml under pg_extensions

pg_extensions:
  - name: pg_steadytext
    version: ${VERSION}
    description: "Deterministic AI text generation for PostgreSQL"
    category: "AI/ML"
    url: "https://github.com/julep-ai/steadytext"
    requires:
      - plpython3u
    install_commands:
      - "pip3 install steadytext"
      - "systemctl enable pg_steadytext_worker"
    config_parameters:
      plpython3.python_path: "/opt/steadytext/venv/lib/python3.*/site-packages"
EOF

echo "Pigsty configuration generated: ${BUILD_DIR}/pigsty-pg_steadytext.yaml"
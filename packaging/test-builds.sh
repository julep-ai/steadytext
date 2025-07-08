#!/bin/bash
# Test built packages

set -e

echo "Testing built packages..."

# Test Debian packages
if [ -d "build/deb" ]; then
    echo "=== Testing Debian packages ==="
    for deb in build/deb/*.deb; do
        if [ -f "$deb" ]; then
            echo "Checking: $deb"
            dpkg-deb --info "$deb"
            dpkg-deb --contents "$deb" | head -20
            echo "---"
        fi
    done
fi

# Test RPM packages
if [ -d "build/rpm/RPMS" ]; then
    echo "=== Testing RPM packages ==="
    for rpm in build/rpm/RPMS/*/*.rpm; do
        if [ -f "$rpm" ]; then
            echo "Checking: $rpm"
            rpm -qpi "$rpm" 2>/dev/null || echo "Note: rpm command not available"
            echo "---"
        fi
    done
fi

# Test PGXN package
if ls build/pgxn/*.zip >/dev/null 2>&1; then
    echo "=== Testing PGXN package ==="
    for zip in build/pgxn/*.zip; do
        if [ -f "$zip" ]; then
            echo "Checking: $zip"
            unzip -l "$zip" | head -20
            echo "---"
        fi
    done
fi

echo "Package testing complete!"
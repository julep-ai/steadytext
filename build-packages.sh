#!/bin/bash
# Convenience script to build all packages

set -e

echo "Building SteadyText packages..."

# Change to packaging directory
cd packaging

# Make scripts executable
chmod +x *.sh

# Parse command line arguments
COMMAND=${1:-all}

case $COMMAND in
    all)
        echo "Building all packages..."
        ./build-deb.sh
        ./build-rpm.sh
        ./pgxn-upload.sh
        ;;
    deb)
        echo "Building Debian packages..."
        ./build-deb.sh
        ;;
    rpm)
        echo "Building RPM packages..."
        ./build-rpm.sh
        ;;
    pgxn)
        echo "Building PGXN package..."
        ./pgxn-upload.sh
        ;;
    test)
        echo "Testing built packages..."
        ./test-builds.sh
        ;;
    clean)
        echo "Cleaning build artifacts..."
        # Use safer cleanup with explicit path validation
        if [ -d "packaging/build" ]; then
            rm -rf packaging/build/
        fi
        ;;
    *)
        echo "Usage: $0 [all|deb|rpm|pgxn|test|clean]"
        echo ""
        echo "Commands:"
        echo "  all   - Build all package types (default)"
        echo "  deb   - Build Debian packages only"
        echo "  rpm   - Build RPM packages only" 
        echo "  pgxn  - Build PGXN package only"
        echo "  test  - Test built packages"
        echo "  clean - Remove build artifacts"
        exit 1
        ;;
esac

echo "Done!"
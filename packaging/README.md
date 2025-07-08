# SteadyText Packaging

This directory contains scripts to build distribution packages for pg_steadytext.

## Supported Package Types

- **Debian/Ubuntu** (.deb) - For apt-based systems
- **RHEL/Rocky/Fedora** (.rpm) - For yum/dnf-based systems
- **PGXN** - PostgreSQL Extension Network
- **Pigsty** - Pigsty PostgreSQL distribution

## Prerequisites

### For Debian packaging:
```bash
sudo apt-get install dpkg-dev debhelper python3-pip
```

### For RPM packaging:
```bash
sudo yum install rpm-build python3-pip
# or
sudo dnf install rpm-build python3-pip
```

### For PGXN:
```bash
pip install pgxn-client
```

## Building Packages

### Make scripts executable:
```bash
./make-executable.sh
```

### Build all packages:
```bash
cd packaging
./build-deb.sh    # Build Debian packages
./build-rpm.sh    # Build RPM packages
./pgxn-upload.sh  # Prepare PGXN package
```

### Build for specific PostgreSQL versions:
```bash
PG_VERSIONS="15 16" ./build-deb.sh
```

### Test built packages:
```bash
./test-builds.sh
```

## Package Contents

All packages include:
- PostgreSQL extension files (.control, .sql)
- Python support modules
- Systemd service for async worker
- Documentation

## Installation

See INSTALL_PACKAGES.md in the root directory for installation instructions.

## Versioning

Package versions are automatically read from:
- `pg_steadytext/META.json` - PostgreSQL extension version
- `pyproject.toml` - Python package version

## CI/CD Integration

GitHub Actions automatically builds packages on release tags. See `.github/workflows/build-packages.yml`.

## Directory Structure

```
build/
├── deb/                    # Debian packages
│   ├── postgresql-14-pg-steadytext_*.deb
│   ├── postgresql-15-pg-steadytext_*.deb
│   └── ...
├── rpm/                    # RPM packages
│   └── RPMS/
│       └── noarch/
│           ├── pg_steadytext_14-*.rpm
│           └── ...
└── pgxn/                   # PGXN distribution
    ├── pg_steadytext-*.zip
    └── pigsty-pg_steadytext.yaml
```
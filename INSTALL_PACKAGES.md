# Installing pg_steadytext from Packages

This guide covers installation of pg_steadytext using distribution packages.

## Quick Install

### Debian/Ubuntu

```bash
# Download the package for your PostgreSQL version
wget https://github.com/julep-ai/steadytext/releases/download/vX.Y.Z/postgresql-16-pg-steadytext_X.Y.Z.deb

# Install the package
sudo dpkg -i postgresql-16-pg-steadytext_X.Y.Z.deb

# Fix any dependency issues
sudo apt-get install -f

# Create the extension in your database
sudo -u postgres psql -c "CREATE EXTENSION pg_steadytext;"

# Start the async worker
sudo systemctl start pg_steadytext_worker
```

### RHEL/Rocky/Fedora

```bash
# Download the package for your PostgreSQL version
wget https://github.com/julep-ai/steadytext/releases/download/vX.Y.Z/pg_steadytext_16-X.Y.Z.rpm

# Install the package
sudo rpm -ivh pg_steadytext_16-X.Y.Z.rpm

# Create the extension in your database
sudo -u postgres psql -c "CREATE EXTENSION pg_steadytext;"

# Start the async worker
sudo systemctl start pg_steadytext_worker
```

## Installation via Package Managers

### PGXN (PostgreSQL Extension Network)

```bash
# Install pgxn client
pip install pgxn-client

# Install pg_steadytext
pgxn install pg_steadytext

# Or download and install manually
pgxn download pg_steadytext
pgxn load pg_steadytext
```

### Pigsty

If you're using Pigsty, add this to your `pigsty.yml`:

```yaml
pg_extensions:
  - name: pg_steadytext
    version: "1.2.0"  # Replace with desired version
```

Then run:
```bash
./pigsty pg_extension -e pg_steadytext
```

## Available Packages

Packages are available for:
- PostgreSQL 14, 15, 16, and 17
- Debian/Ubuntu (amd64)
- RHEL/Rocky/Fedora (x86_64)
- Any system via PGXN

## Verifying Installation

After installation, verify the extension is available:

```sql
-- Check available extensions
SELECT * FROM pg_available_extensions WHERE name = 'pg_steadytext';

-- Create the extension
CREATE EXTENSION pg_steadytext;

-- Test basic functionality
SELECT steadytext_generate('Hello world!');
SELECT steadytext_version();
```

## Post-Installation Configuration

### 1. Python Path Configuration

The extension needs access to the SteadyText Python package:

```sql
-- Check current Python path
SHOW plpython3.python_path;

-- The package installer should have set this automatically
-- If not, set it manually (adjust Python version as needed):
ALTER DATABASE your_db SET plpython3.python_path = '/opt/steadytext/venv/lib/python3.*/site-packages';
-- Or find the exact path:
-- ls /opt/steadytext/venv/lib/
```

### 2. Async Worker Service

For async functions, ensure the worker service is running:

```bash
# Check service status
sudo systemctl status pg_steadytext_worker

# Start the service
sudo systemctl start pg_steadytext_worker

# Enable on boot
sudo systemctl enable pg_steadytext_worker

# View logs
sudo journalctl -u pg_steadytext_worker -f
```

### 3. Model Downloads

Models are downloaded on first use. To pre-download:

```bash
# As postgres user
sudo -u postgres /opt/steadytext/venv/bin/python -c "
from steadytext import generate, embed
print('Downloading models...')
generate('test')
embed('test')
print('Models ready!')
"
```

## Troubleshooting

### Permission Errors

If you encounter permission errors:
```bash
# Ensure postgres owns the steadytext directory
sudo chown -R postgres:postgres /opt/steadytext
```

### Python Module Not Found

If the extension can't find the Python module:
```sql
-- Set the Python path explicitly (adjust Python version as needed)
ALTER DATABASE your_db SET plpython3.python_path = '/opt/steadytext/venv/lib/python3.*/site-packages';
-- Or find the exact path:
-- ls /opt/steadytext/venv/lib/

-- Reload configuration
SELECT pg_reload_conf();
```

### Worker Service Issues

If the async worker won't start:
```bash
# Check logs
sudo journalctl -u pg_steadytext_worker -n 50

# Test worker manually
sudo -u postgres /opt/steadytext/venv/bin/python /opt/steadytext/python/worker.py
```

## Uninstallation

### Debian/Ubuntu
```bash
# Remove the package
sudo apt-get remove postgresql-16-pg-steadytext

# Clean up
sudo rm -rf /opt/steadytext
```

### RHEL/Rocky/Fedora
```bash
# Remove the package
sudo rpm -e pg_steadytext_16

# Clean up
sudo rm -rf /opt/steadytext
```

## Building from Source

If packages aren't available for your system, see the main README for source installation instructions, or build packages yourself using the scripts in the `packaging/` directory.
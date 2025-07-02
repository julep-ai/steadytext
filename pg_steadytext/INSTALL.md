# pg_steadytext Installation Guide

This guide provides instructions for installing the pg_steadytext PostgreSQL extension.

## Prerequisites

- PostgreSQL 12 or later
- Python 3.8 or later
- `postgresql-plpython3` package for your PostgreSQL version
- `omni-python` extension for PostgreSQL Python integration (see https://docs.omnigres.org/quick_start/)
- pip (Python package manager)

## Installation Methods

### Method 1: Docker Installation (Recommended)

The easiest way to install pg_steadytext is using Docker. Here's a complete Dockerfile example:

```dockerfile
FROM postgres:17

# Install Python and required system packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    postgresql-plpython3-17 \
    git \
    make \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install omni-python extension
RUN apt-get install -y postgresql-17-omni-python || \
    (git clone https://github.com/omnigres/omnigres.git && \
     cd omnigres/extensions/omni_python && \
     make && make install)

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --break-system-packages \
      pgai[vectorizer-worker] \
      steadytext>=2.2.0 \
      pyzmq>=22.0.0 \
      numpy>=1.20.0

# Pre-download SteadyText models (optional but recommended)
# This ensures models are available at container startup
RUN python3 -c "import os; os.environ['STEADYTEXT_USE_FALLBACK_MODEL'] = os.environ.get('STEADYTEXT_USE_FALLBACK_MODEL', 'false'); \
    from steadytext.models.cache import ensure_generation_model_cached, ensure_embedding_model_cached; \
    print(f'Using fallback model: {os.environ[\"STEADYTEXT_USE_FALLBACK_MODEL\"]}'); \
    print('Pre-downloading generation model...'); \
    ensure_generation_model_cached(); \
    print('Pre-downloading embedding model...'); \
    ensure_embedding_model_cached(); \
    print('Models downloaded successfully')" || \
    (echo "WARNING: Model pre-download failed. Models will be downloaded on first use." && true)

# Install pg_steadytext extension
ADD https://github.com/julep-ai/steadytext.git#:pg_steadytext /tmp/pg_steadytext
RUN cd /tmp/pg_steadytext && make install

# Optional: Set up environment for the postgres user
USER postgres
ENV PYTHONPATH=/usr/local/lib/python3.11/dist-packages
```

Build and run the Docker container:

```bash
docker build -t pg_steadytext .
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres --name pg_steadytext pg_steadytext
```

### Method 2: Manual Installation

For manual installation on a running PostgreSQL server:

#### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y postgresql-plpython3-$(pg_config --version | awk '{print $2}' | sed 's/\..*//')
sudo apt-get install -y postgresql-$(pg_config --version | awk '{print $2}' | sed 's/\..*//')-omni-python || echo "omni-python not in apt, will install from source"
sudo apt-get install -y python3 python3-pip python3-dev git make gcc

# If omni-python is not available via apt, install it from source:
if ! dpkg -l | grep -q postgresql.*omni-python; then
    git clone https://github.com/omnigres/omnigres.git
    cd omnigres/extensions/omni_python
    make && sudo make install
    cd ../../..
fi
```

**RHEL/CentOS/Rocky:**
```bash
sudo yum install -y postgresql$(pg_config --version | awk '{print $2}' | sed 's/\..*//')-plpython3
sudo yum install -y python3 python3-pip python3-devel git make gcc

# Install omni-python from source (typically not in RHEL/CentOS repos)
git clone https://github.com/omnigres/omnigres.git
cd omnigres/extensions/omni_python
make && sudo make install
cd ../../..
```

**macOS (with Homebrew):**
```bash
brew install postgresql python@3.11
# Note: plpython3 is typically included with PostgreSQL on macOS

# Install omni-python from source
git clone https://github.com/omnigres/omnigres.git
cd omnigres/extensions/omni_python
make && sudo make install
cd ../../..
```

#### Step 2: Install Python Dependencies

Install the required Python packages:

```bash
# For system-wide installation (may require sudo)
pip3 install steadytext>=2.2.0 pyzmq>=22.0.0 numpy>=1.20.0

# For user installation
pip3 install --user steadytext>=2.2.0 pyzmq>=22.0.0 numpy>=1.20.0
```

**Note:** On some systems, you may need to use `--break-system-packages` flag:
```bash
pip3 install --break-system-packages steadytext>=2.2.0 pyzmq>=22.0.0 numpy>=1.20.0
```

#### Step 3: Download and Install pg_steadytext

```bash
# Clone the repository
git clone https://github.com/julep-ai/steadytext.git
cd steadytext/pg_steadytext

# Install the extension
sudo make install

# Or if you don't have sudo access, specify the PostgreSQL directory
make install PG_CONFIG=/path/to/pg_config
```

#### Step 4: Enable the Extension in PostgreSQL

Connect to your PostgreSQL database and run:

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS plpython3u;
CREATE EXTENSION IF NOT EXISTS omni_python;

-- Enable pg_steadytext
CREATE EXTENSION IF NOT EXISTS pg_steadytext;
```

### Method 3: Using pgxn (PostgreSQL Extension Network)

*Note: pg_steadytext is not yet available on pgxn. This section will be updated when it becomes available.*

## Verifying Installation

After installation, verify that the extension is working:

```sql
-- Check extension version
SELECT steadytext_version();

-- Test text generation
SELECT steadytext_generate('Hello, world!');

-- Test embedding generation
SELECT steadytext_embed('Test embedding');
```

## Configuration

### Environment Variables

The following environment variables can be set for the PostgreSQL server process:

- `STEADYTEXT_USE_FALLBACK_MODEL`: Set to 'true' to use fallback models
- `PYTHONPATH`: May need to be set to include the SteadyText installation directory

### PostgreSQL Configuration

Add to your `postgresql.conf`:

```conf
# Allow plpython3u to find Python packages
# Adjust the path based on your Python installation
plpython3.python_path = '/usr/local/lib/python3.11/dist-packages'
```

Or set it per-database:

```sql
ALTER DATABASE your_database SET plpython3.python_path TO '/usr/local/lib/python3.11/dist-packages';
```

## Troubleshooting

### Common Issues

1. **"could not load library plpython3"**
   - Ensure postgresql-plpython3 is installed for your PostgreSQL version
   - Check that the plpython3u extension is available: `SELECT * FROM pg_available_extensions WHERE name = 'plpython3u';`

2. **"extension \"omni_python\" does not exist"**
   - Ensure omni-python is installed (see installation steps above)
   - Check available extensions: `SELECT * FROM pg_available_extensions WHERE name = 'omni_python';`
   - If missing, install from source as shown in the installation instructions

3. **"ImportError: No module named steadytext"**
   - Verify SteadyText is installed: `python3 -c "import steadytext; print(steadytext.__version__)"`
   - Check PYTHONPATH configuration in PostgreSQL
   - Ensure the postgres user can access the Python packages

4. **"Model download failed"**
   - Ensure the PostgreSQL server has internet access
   - Check disk space in the model cache directory
   - Try pre-downloading models: `python3 -c "from steadytext import preload_models; preload_models(verbose=True)"`

5. **"Permission denied" errors**
   - Ensure the postgres user has write access to the model cache directory
   - Default cache location: `~postgres/.cache/steadytext/models/`

### Getting Help

If you encounter issues:

1. Check the PostgreSQL logs for detailed error messages
2. Verify all dependencies are correctly installed
3. Open an issue at https://github.com/julep-ai/steadytext/issues

## Docker Compose Example

For a complete setup with Docker Compose:

```yaml
version: '3.8'

services:
  postgres:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: steadytext_db
      PYTHONPATH: /usr/local/lib/python3.11/dist-packages
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - model_cache:/var/lib/postgresql/.cache/steadytext

volumes:
  postgres_data:
  model_cache:
```

This ensures model downloads persist across container restarts.
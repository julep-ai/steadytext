#!/bin/bash
# Build PostgreSQL with Python 3.13 support for pg_steadytext
# AIDEV-NOTE: This script builds PostgreSQL from source with Python 3.13 support

set -e

# Configuration
PG_VERSION="${PG_VERSION:-17.2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
BUILD_DIR="${BUILD_DIR:-/tmp/pg_build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local/pgsql}"
CORES=$(nproc)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building PostgreSQL ${PG_VERSION} with Python ${PYTHON_VERSION} support${NC}"

# Check if running as root (recommended for system-wide install)
if [[ $EUID -ne 0 ]]; then
   echo -e "${YELLOW}Warning: This script should be run as root for system-wide installation${NC}"
   echo "Continue anyway? (y/n)"
   read -r response
   if [[ ! "$response" =~ ^[Yy]$ ]]; then
       exit 1
   fi
fi

# Install build dependencies
echo -e "${GREEN}Installing build dependencies...${NC}"
apt-get update
apt-get install -y \
    build-essential \
    wget \
    libreadline-dev \
    zlib1g-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libicu-dev \
    liblz4-dev \
    libzstd-dev \
    pkg-config \
    uuid-dev \
    software-properties-common

# Install Python 3.13 if not present
if ! command -v python${PYTHON_VERSION} &> /dev/null; then
    echo -e "${GREEN}Installing Python ${PYTHON_VERSION}...${NC}"
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Download PostgreSQL source
if [ ! -f "postgresql-${PG_VERSION}.tar.gz" ]; then
    echo -e "${GREEN}Downloading PostgreSQL ${PG_VERSION} source...${NC}"
    wget "https://ftp.postgresql.org/pub/source/v${PG_VERSION}/postgresql-${PG_VERSION}.tar.gz"
fi

# Extract source
echo -e "${GREEN}Extracting source...${NC}"
tar -xzf "postgresql-${PG_VERSION}.tar.gz"
cd "postgresql-${PG_VERSION}"

# Configure PostgreSQL with Python 3.13
echo -e "${GREEN}Configuring PostgreSQL with Python ${PYTHON_VERSION}...${NC}"
./configure \
    --prefix="$INSTALL_PREFIX" \
    --with-python \
    --with-openssl \
    --with-libxml \
    --with-libxslt \
    --with-icu \
    --with-lz4 \
    --with-zstd \
    --with-uuid=e2fs \
    PYTHON="/usr/bin/python${PYTHON_VERSION}"

# Build PostgreSQL
echo -e "${GREEN}Building PostgreSQL (using ${CORES} cores)...${NC}"
make -j"$CORES"

# Build contrib modules
echo -e "${GREEN}Building contrib modules...${NC}"
cd contrib
make -j"$CORES"
cd ..

# Install PostgreSQL
echo -e "${GREEN}Installing PostgreSQL to ${INSTALL_PREFIX}...${NC}"
make install
cd contrib
make install
cd ..

# Create postgres user if it doesn't exist
if ! id "postgres" &>/dev/null; then
    echo -e "${GREEN}Creating postgres user...${NC}"
    useradd -r -s /bin/bash postgres
fi

# Set up environment
echo -e "${GREEN}Setting up environment...${NC}"
cat > /etc/profile.d/postgresql.sh << EOF
export PATH=${INSTALL_PREFIX}/bin:\$PATH
export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:\$LD_LIBRARY_PATH
EOF

# Create systemd service file
echo -e "${GREEN}Creating systemd service...${NC}"
cat > /etc/systemd/system/postgresql-custom.service << EOF
[Unit]
Description=PostgreSQL database server (custom build with Python ${PYTHON_VERSION})
After=network.target

[Service]
Type=notify
User=postgres
ExecStart=${INSTALL_PREFIX}/bin/postgres -D /var/lib/postgresql/data
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
KillSignal=SIGINT
TimeoutSec=0

[Install]
WantedBy=multi-user.target
EOF

# Install Python packages for PostgreSQL
echo -e "${GREEN}Installing Python packages for pg_steadytext...${NC}"
/usr/bin/python${PYTHON_VERSION} -m pip install \
    steadytext>=2.6.2 \
    pyzmq>=22.0.0 \
    numpy>=1.20.0

# Create data directory
mkdir -p /var/lib/postgresql/data
chown postgres:postgres /var/lib/postgresql/data

# Success message
echo -e "${GREEN}PostgreSQL ${PG_VERSION} with Python ${PYTHON_VERSION} has been successfully built!${NC}"
echo
echo "Next steps:"
echo "1. Source the environment: source /etc/profile.d/postgresql.sh"
echo "2. Initialize the database: sudo -u postgres ${INSTALL_PREFIX}/bin/initdb -D /var/lib/postgresql/data"
echo "3. Start PostgreSQL: sudo systemctl start postgresql-custom"
echo "4. Install pg_steadytext extension"
echo
echo -e "${YELLOW}To verify Python version in PostgreSQL:${NC}"
echo "sudo -u postgres psql -c \"DO \\\$\\\$ import sys; plpy.notice(f'Python: {sys.version}') \\\$\\\$ LANGUAGE plpython3u;\""
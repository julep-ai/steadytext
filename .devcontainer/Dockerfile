FROM docker.io/jfxs/devcontainer-uv:latest

# Install Node.js and other dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    libpq-dev \
    postgresql-client \
    make \
    curl \
    sudo \
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Docker CLI for Docker-in-Docker support
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Create writable cache directories
RUN mkdir -p /tmp/.uv-cache /tmp/.local && chmod -R 777 /tmp/.uv-cache /tmp/.local

# Set working directory
WORKDIR /workspace

# Ensure git and other tools are available
RUN apt-get update && apt-get install -y git && apt-get clean -y && rm -rf /var/lib/apt/lists/* 
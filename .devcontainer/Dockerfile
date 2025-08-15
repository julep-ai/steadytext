# syntax=docker/dockerfile:1
FROM docker.io/jfxs/devcontainer-uv:latest

# Install all system dependencies in a single layer with cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    libpq-dev \
    postgresql-client \
    make \
    curl \
    sudo \
    git \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    aria2 \
    direnv \
    pgtap \
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Create writable cache directories
RUN mkdir -p /tmp/.uv-cache /tmp/.local && chmod -R 777 /tmp/.uv-cache /tmp/.local

# Set working directory
WORKDIR /workspace

# Install node utils
RUN npm install -g claude-code-inject claude-code-generic-hooks pyright

# Add cache mount points for UV
ENV UV_CACHE_DIR=/tmp/.uv-cache
ENV PYTHONUSERBASE=/tmp/.local

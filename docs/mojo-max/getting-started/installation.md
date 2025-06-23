# Installation Guide

This guide covers installing Mojo and MAX on your system. The Community Edition supports Mac, Linux, and WSL environments.

## System Requirements

- **Operating System**: macOS, Linux, or Windows (via WSL)
- **Python**: 3.8 or later
- **Hardware**: Any x86_64 CPU; NVIDIA GPUs supported for acceleration

## Installation Methods

### 1. Using pip (Recommended)

```bash
# Create a virtual environment
python3 -m venv mojo-env
source mojo-env/bin/activate  # On Windows: mojo-env\Scripts\activate

# Install the modular package
pip install modular \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://modular.gateway.scarf.sh/simple/
```

### 2. Using uv (Fast Python Package Manager)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project and install
uv init my-project && cd my-project
uv venv && source .venv/bin/activate
uv pip install modular \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://modular.gateway.scarf.sh/simple/ \
  --index-strategy unsafe-best-match
```

### 3. Using conda

```bash
# Create conda environment
conda create -n mojo-env
conda activate mojo-env

# Install from Modular's conda channel
conda install -c conda-forge -c https://conda.modular.com/max/ modular
```

### 4. Using pixi

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | sh

# Create project
pixi init my-project -c https://conda.modular.com/max/ -c conda-forge
cd my-project

# Add modular package
pixi add "modular=25.4"
pixi shell
```

## Verify Installation

After installation, verify everything is working:

```bash
# Check Mojo version
mojo --version

# Check MAX CLI
max --version

# Run a simple Mojo program
echo 'def main(): print("Hello from Mojo!")' > hello.mojo
mojo run hello.mojo
```

## Environment Variables

Optional environment variables for configuration:

```bash
# Set custom cache directory (default: ~/.cache/modular)
export MODULAR_HOME=/path/to/cache

# Enable debug logging
export MODULAR_LOG_LEVEL=debug
```

## Docker Installation

For containerized deployments:

```bash
# Pull the MAX container
docker pull modular/max:latest

# Run with GPU support
docker run --gpus all -it modular/max:latest
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you've activated your virtual environment
2. **Permission denied**: Use `sudo` for system-wide installation (not recommended)
3. **GPU not detected**: Install NVIDIA drivers and CUDA toolkit
4. **SSL errors**: Update certificates or use `--trusted-host` flag

### Getting Help

- Community Discord: Join via [modular.com/community](https://modular.com/community)
- GitHub Issues: [github.com/modular/mojo/issues](https://github.com/modular/mojo/issues)
- Forum: [forum.modular.com](https://forum.modular.com)

## Next Steps

- Follow the [Quickstart Guide](quickstart.md) to run your first model
- Learn [Mojo Basics](../mojo-language/basics.md) for writing high-performance code
- Explore the [Model Repository](https://builds.modular.com) for pre-optimized models
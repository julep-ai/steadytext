# Mojo and MAX Documentation

This documentation covers the Modular platform's Community Edition, including the Mojo programming language and MAX inference server. All features documented here are available for free under the Community License.

## Overview

**Mojo** is a Python-compatible systems programming language that enables high-performance CPU and GPU programming without CUDA. It combines Python's ease of use with the performance of C++ and Rust.

**MAX** is a high-performance inference framework for serving AI models with an OpenAI-compatible API. It supports 500+ models and provides state-of-the-art performance on both CPUs and GPUs.

## Community Edition

The Community Edition is free forever and includes:
- Unlimited use on CPUs and NVIDIA GPUs
- Up to 8 discrete GPUs from other vendors (AMD, etc.)
- Free for both personal and commercial use
- Perpetual license guarantee (cannot be revoked)

For commercial use, Modular asks that you notify them at usage@modular.com so they can showcase your usage.

## Documentation Structure

### Getting Started
- [Installation](getting-started/installation.md) - Install Mojo and MAX
- [Quickstart](getting-started/quickstart.md) - Run your first model
- [Hello World](getting-started/hello-world.md) - Basic Mojo programs

### Mojo Language
- [Basics](mojo-language/basics.md) - Core language features
- [Python Interoperability](mojo-language/python-interop.md) - Using Python from Mojo
- [Calling Mojo from Python](mojo-language/calling-mojo-from-python.md) - Creating Python extensions
- [Functions](mojo-language/functions.md) - Function definitions and calls
- [Structs and Traits](mojo-language/structs-and-traits.md) - Type system
- [Metaprogramming](mojo-language/metaprogramming.md) - Compile-time programming

### MAX Inference
- [Overview](max-inference/overview.md) - MAX architecture and features
- [Serving Models](max-inference/serving-models.md) - Running inference servers
- [Offline Inference](max-inference/offline-inference.md) - Direct Python API
- [API Reference](max-inference/api-reference.md) - OpenAI-compatible endpoints
- [Deployment](max-inference/deployment.md) - Docker and production setup

### Community Edition
- [Licensing](community-edition/licensing.md) - License terms and usage
- [Limitations](community-edition/limitations.md) - Current limitations and roadmap

### Examples
- [Basic Examples](examples/basic-examples.md) - Simple Mojo programs
- [Python Integration](examples/python-integration.md) - Interop examples

## Key Features for Python Developers

1. **Drop-in Performance**: Write performance-critical code in Mojo and call it from Python
2. **Python Compatibility**: Import and use any Python library directly in Mojo
3. **Zero-Config Inference**: Deploy models with a single command
4. **OpenAI API Compatible**: Use existing client code with MAX endpoints

## Quick Links

- [Official Docs](https://docs.modular.com)
- [Model Repository](https://builds.modular.com)
- [Community Forum](https://forum.modular.com)
- [GitHub Examples](https://github.com/modular/modular)
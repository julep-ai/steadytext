# Version History

This document outlines the major versions of SteadyText and the key features introduced in each.

| Version | Key Features                                                                                                                            | Default Generation Model                               | Default Embedding Model                                | Python Versions |
| :------ | :-------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- | :----------------------------------------------------- | :-------------- |
| **2.x** | - **Daemon Mode**: Persistent model serving with ZeroMQ.<br>- **Gemma-3n Models**: Switched to `gemma-3n` for generation.<br>- **Thinking Mode Deprecated**: Removed thinking mode. | `unsloth/gemma-3n-E2B-it-GGUF` (gemma-3n-E2B-it-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | `>=3.10, <3.14` |
| **1.x** | - **Model Switching**: Added support for switching models via environment variables and a model registry.<br>- **Qwen3 Models**: Switched to `qwen3-1.7b` for generation.<br>- **Indexing**: Added support for FAISS indexing. | `Qwen/Qwen3-1.7B-GGUF` (Qwen3-1.7B-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | `>=3.10, <3.14` |
| **0.x** | - **Initial Release**: Deterministic text generation and embedding.                                                                      | `Qwen/Qwen1.5-0.5B-Chat-GGUF` (qwen1_5-0_5b-chat-q4_k_m.gguf) | `Qwen/Qwen1.5-0.5B-Chat-GGUF` (qwen1_5-0_5b-chat-q8_0.gguf) | `>=3.10`        |

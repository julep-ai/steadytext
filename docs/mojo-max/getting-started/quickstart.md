# Quickstart Guide

Get up and running with MAX inference in minutes. This guide shows you how to serve a model locally and make inference requests.

## Step 1: Set Up Your Project

```bash
# Create a project directory
mkdir max-quickstart && cd max-quickstart

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install modular openai \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://modular.gateway.scarf.sh/simple/
```

## Step 2: Run Offline Inference

Create a file `offline_inference.py`:

```python
from max.entrypoints.llm import LLM
from max.pipelines import PipelineConfig

def main():
    # Use a non-gated model (no HuggingFace token required)
    model_path = "modularai/Llama-3.1-8B-Instruct-GGUF"
    pipeline_config = PipelineConfig(model_path=model_path)
    llm = LLM(pipeline_config)

    prompts = [
        "In the beginning, there was",
        "I believe the meaning of life is",
        "The fastest way to learn python is",
    ]

    print("Generating responses...")
    responses = llm.generate(prompts, max_new_tokens=50)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        print(f"========== Response {i} ==========")
        print(prompt + response)
        print()

if __name__ == "__main__":
    main()
```

Run it:

```bash
python offline_inference.py
```

Expected output:
```
========== Response 0 ==========
In the beginning, there was darkness. Then came the light...

========== Response 1 ==========
I believe the meaning of life is to find purpose and happiness...

========== Response 2 ==========
The fastest way to learn python is through hands-on practice...
```

## Step 3: Start an Inference Server

Start a local endpoint with OpenAI-compatible API:

```bash
max serve --model-path=modularai/Llama-3.1-8B-Instruct-GGUF
```

Wait for the message:
```
Server ready on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 4: Send Requests to the Server

Create `chat_client.py`:

```python
from openai import OpenAI

# Configure client to use local endpoint
client = OpenAI(
    base_url="http://0.0.0.0:8000/v1",
    api_key="EMPTY",  # MAX doesn't require an API key
)

# Send a chat completion request
completion = client.chat.completions.create(
    model="modularai/Llama-3.1-8B-Instruct-GGUF",
    messages=[
        {
            "role": "user",
            "content": "Who won the world series in 2020?"
        },
    ],
)

print(completion.choices[0].message.content)
```

Run it (in a new terminal):

```bash
python chat_client.py
```

Expected output:
```
The Los Angeles Dodgers won the 2020 World Series. They defeated the Tampa Bay Rays 4 games to 2.
```

## Common Use Cases

### 1. Streaming Responses

```python
# Stream tokens as they're generated
stream = client.chat.completions.create(
    model="modularai/Llama-3.1-8B-Instruct-GGUF",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### 2. Custom Generation Parameters

```python
completion = client.chat.completions.create(
    model="modularai/Llama-3.1-8B-Instruct-GGUF",
    messages=[{"role": "user", "content": "Write a haiku"}],
    temperature=0.7,
    max_tokens=100,
    top_p=0.9,
)
```

### 3. Using Different Models

```bash
# Serve a different model
max serve --model-path=microsoft/Phi-3.5-mini-instruct

# Or use a larger model
max serve --model-path=meta-llama/Llama-3.1-70B-Instruct
```

## Server Options

```bash
# Change port
max serve --model-path=MODEL --port=8080

# Limit concurrent requests
max serve --model-path=MODEL --max-concurrent-requests=10

# Enable GPU acceleration (if available)
max serve --model-path=MODEL --device=cuda

# Use multiple GPUs
max serve --model-path=MODEL --tensor-parallel-size=2
```

## Performance Tips

1. **Model Format**: Use GGUF models for best performance/memory trade-off
2. **Batch Processing**: Send multiple prompts at once for better throughput
3. **Caching**: Models are cached locally after first download
4. **GPU Usage**: Automatically uses GPU if available

## Next Steps

- Explore [500+ supported models](https://builds.modular.com)
- Learn about [advanced serving features](../max-inference/serving-models.md)
- Try [Mojo for custom operators](../mojo-language/basics.md)
- Deploy with [Docker](../max-inference/deployment.md)

## Troubleshooting

### Model Download Issues
- Check internet connection
- Verify model name is correct
- Some models require HuggingFace token

### Out of Memory
- Use quantized models (Q4, Q8)
- Reduce batch size
- Use CPU offloading

### Slow Performance
- Enable GPU with `--device=cuda`
- Use smaller models for testing
- Check system resources with `htop`
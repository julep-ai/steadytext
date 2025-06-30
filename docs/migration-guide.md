# Migration Guide

This guide helps you migrate between different versions of SteadyText and from other text generation libraries.

## Table of Contents

- [Version Migration](#version-migration)
  - [v2.0.x to v2.1.x](#v20x-to-v21x)
  - [v1.x to v2.x](#v1x-to-v2x)
  - [v0.x to v1.x](#v0x-to-v1x)
- [Library Migration](#library-migration)
  - [From OpenAI](#from-openai)
  - [From Hugging Face](#from-hugging-face)
  - [From LangChain](#from-langchain)
  - [From Anthropic](#from-anthropic)
- [Breaking Changes](#breaking-changes)
- [Feature Mapping](#feature-mapping)
- [Code Examples](#code-examples)
- [Best Practices](#best-practices)

## Version Migration

### v2.0.x to v2.1.x

**Major Breaking Change**: Deterministic fallback behavior removed.

#### What Changed

Functions now return `None` instead of fallback text when models are unavailable:

```python
# v2.0.x behavior
result = steadytext.generate("Hello")
# Returns: "Hello. This is a deterministic..." (fallback text)

# v2.1.x behavior
result = steadytext.generate("Hello")
# Returns: None (when model not loaded)
```

#### Migration Steps

1. **Update error handling**:
   ```python
   # Old code (v2.0.x)
   text = steadytext.generate(prompt)
   # Always returned something
   
   # New code (v2.1.x)
   text = steadytext.generate(prompt)
   if text is None:
       # Handle model not available
       text = "Unable to generate text"
   ```

2. **Update embedding handling**:
   ```python
   # Old code (v2.0.x)
   embedding = steadytext.embed(text)
   # Always returned zero vector on failure
   
   # New code (v2.1.x)
   embedding = steadytext.embed(text)
   if embedding is None:
       # Handle model not available
       embedding = np.zeros(1024)
   ```

3. **Update tests**:
   ```python
   # Old test (v2.0.x)
   def test_generation():
       result = steadytext.generate("test")
       assert result.startswith("test. This is")
   
   # New test (v2.1.x)
   def test_generation():
       result = steadytext.generate("test")
       if result is not None:
           assert isinstance(result, str)
       else:
           # Model not available is acceptable
           pass
   ```

#### PostgreSQL Extension Changes

```sql
-- Old behavior (v2.0.x)
SELECT steadytext_generate('Hello');
-- Returns: 'Hello. This is a deterministic...'

-- New behavior (v2.1.x)
SELECT steadytext_generate('Hello');
-- Returns: NULL (when model not available)

-- Add NULL handling
SELECT COALESCE(
    steadytext_generate('Hello'),
    'Fallback text'
) AS result;
```

### v1.x to v2.x

**Major Changes**: 
- New models (GPT-2 → Gemma-3n)
- New embedding model (DistilBERT → Qwen3)
- Changed embedding dimensions (768 → 1024)

#### Model Changes

```python
# v1.x (GPT-2 based)
text = steadytext.generate("Hello")  # Used GPT-2

# v2.x (Gemma-3n based)
text = steadytext.generate("Hello")  # Uses Gemma-3n
text = steadytext.generate("Hello", model_size="large")  # 4B model
```

#### Embedding Dimension Changes

```python
# v1.x embeddings (768 dimensions)
embedding = steadytext.embed("text")
print(embedding.shape)  # (768,)

# v2.x embeddings (1024 dimensions)
embedding = steadytext.embed("text")
print(embedding.shape)  # (1024,)

# Migration for stored embeddings
def migrate_embeddings(old_embeddings):
    """Pad old embeddings to new size."""
    # Note: This is for compatibility only
    # Regenerate embeddings for best results
    padded = np.zeros((len(old_embeddings), 1024))
    padded[:, :768] = old_embeddings
    return padded
```

#### API Changes

```python
# v1.x
from steadytext import generate_text, create_embedding
text = generate_text("prompt")
emb = create_embedding("text")

# v2.x
import steadytext
text = steadytext.generate("prompt")
emb = steadytext.embed("text")
```

### v0.x to v1.x

**Major Changes**:
- Introduced daemon mode
- Added caching system
- New CLI structure

#### Function Name Changes

```python
# v0.x
from steadytext import steady_generate
result = steady_generate("Hello")

# v1.x
from steadytext import generate
result = generate("Hello")
```

#### CLI Changes

```bash
# v0.x
steadytext-generate "prompt"

# v1.x
st generate "prompt"
# or
echo "prompt" | st
```

## Library Migration

### From OpenAI

Migrate from OpenAI's API to SteadyText:

```python
# OpenAI code
import openai

openai.api_key = "sk-..."
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Hello world",
    max_tokens=100,
    temperature=0.7
)
text = response.choices[0].text

# SteadyText equivalent
import steadytext

text = steadytext.generate(
    "Hello world",
    max_new_tokens=100,
    seed=42  # For determinism
)
```

#### Key Differences

| Feature | OpenAI | SteadyText |
|---------|--------|------------|
| API Key | Required | Not needed |
| Cost | Per token | Free |
| Determinism | Optional | Default |
| Offline | No | Yes |
| Models | Cloud-based | Local |

#### Embedding Migration

```python
# OpenAI embeddings
response = openai.Embedding.create(
    model="text-embedding-ada-002",
    input="Hello world"
)
embedding = response['data'][0]['embedding']

# SteadyText embeddings
embedding = steadytext.embed("Hello world")
```

### From Hugging Face

Migrate from Transformers library:

```python
# Hugging Face code
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')
result = generator("Hello world", max_length=100)
text = result[0]['generated_text']

# SteadyText equivalent
import steadytext

text = steadytext.generate("Hello world", max_new_tokens=100)
```

#### Model Loading Comparison

```python
# Hugging Face (explicit loading)
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Complex inference code...

# SteadyText (automatic loading)
text = steadytext.generate("Hello")  # Models loaded automatically
```

#### Embedding Migration

```python
# Hugging Face embeddings
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

inputs = tokenizer("Hello world", return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
embedding = outputs.last_hidden_state.mean(dim=1).numpy()

# SteadyText embeddings
embedding = steadytext.embed("Hello world")
```

### From LangChain

Integrate SteadyText with LangChain:

```python
# LangChain with OpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0)
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a story about {topic}"
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("robots")

# LangChain with SteadyText
from langchain.llms.base import LLM
from typing import Optional, List

class SteadyTextLLM(LLM):
    seed: int = 42
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        result = steadytext.generate(prompt, seed=self.seed)
        return result if result else ""
    
    @property
    def _llm_type(self) -> str:
        return "steadytext"

# Use with LangChain
llm = SteadyTextLLM(seed=42)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("robots")
```

#### Embedding Integration

```python
# Custom SteadyText embeddings for LangChain
from langchain.embeddings.base import Embeddings

class SteadyTextEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [steadytext.embed(text).tolist() for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return steadytext.embed(text).tolist()

# Use with vector stores
from langchain.vectorstores import FAISS
embeddings = SteadyTextEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)
```

### From Anthropic

Migrate from Claude API:

```python
# Anthropic code
import anthropic

client = anthropic.Client(api_key="...")
response = client.completions.create(
    model="claude-2",
    prompt=f"{anthropic.HUMAN_PROMPT} Hello {anthropic.AI_PROMPT}",
    max_tokens_to_sample=100
)
text = response.completion

# SteadyText equivalent
import steadytext

text = steadytext.generate(
    "Hello",
    max_new_tokens=100
)
```

## Breaking Changes

### Summary of All Breaking Changes

| Version | Change | Impact | Migration |
|---------|--------|--------|-----------|
| v2.1.0 | Removed fallback generation | Functions return None | Add null checks |
| v2.0.0 | New models (Gemma-3n/Qwen3) | Different outputs | Regenerate content |
| v2.0.0 | Embedding dimensions (768→1024) | Incompatible vectors | Re-embed data |
| v1.0.0 | API restructure | Import changes | Update imports |

### Handling Breaking Changes

```python
def handle_breaking_changes():
    """Example of handling all breaking changes."""
    
    # Handle v2.1.0 None returns
    text = steadytext.generate("Hello")
    if text is None:
        text = "Fallback text"
    
    # Handle dimension changes
    try:
        old_embedding = load_old_embedding()  # 768 dims
        if len(old_embedding) == 768:
            # Regenerate with new model
            new_embedding = steadytext.embed(original_text)
    except Exception as e:
        print(f"Migration needed: {e}")
```

## Feature Mapping

### Generation Features

| Feature | Other Libraries | SteadyText |
|---------|----------------|------------|
| Basic generation | `model.generate()` | `steadytext.generate()` |
| Streaming | `stream=True` | `steadytext.generate_iter()` |
| Temperature | `temperature=0.7` | `seed=42` (deterministic) |
| Max length | `max_length=100` | `max_new_tokens=100` |
| Stop tokens | `stop=["\\n"]` | `eos_string="\\n"` |
| Batch | `model.generate(batch)` | List comprehension |

### Embedding Features

| Feature | Other Libraries | SteadyText |
|---------|----------------|------------|
| Create embedding | `model.encode()` | `steadytext.embed()` |
| Batch embeddings | `model.encode(list)` | List comprehension |
| Normalization | Manual | Automatic (L2) |
| Dimensions | Varies | Always 1024 |

## Code Examples

### Complete Migration Example

```python
# Full migration from OpenAI to SteadyText

# Old OpenAI-based application
class OpenAIApp:
    def __init__(self, api_key):
        openai.api_key = api_key
    
    def generate_content(self, prompt):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].text
    
    def create_embedding(self, text):
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']

# New SteadyText-based application
class SteadyTextApp:
    def __init__(self, seed=42):
        self.seed = seed
    
    def generate_content(self, prompt):
        result = steadytext.generate(
            prompt,
            max_new_tokens=200,
            seed=self.seed
        )
        return result if result else "Generation unavailable"
    
    def create_embedding(self, text):
        embedding = steadytext.embed(text, seed=self.seed)
        return embedding.tolist() if embedding is not None else [0] * 1024
```

### Database Migration

```python
# Migrate embeddings in PostgreSQL

import psycopg2
import steadytext
import numpy as np

def migrate_embeddings_to_v2():
    conn = psycopg2.connect("postgresql://...")
    cur = conn.cursor()
    
    # Get old embeddings
    cur.execute("SELECT id, text, embedding FROM documents WHERE version = 1")
    
    for doc_id, text, old_embedding in cur.fetchall():
        # Regenerate with new model
        new_embedding = steadytext.embed(text)
        
        if new_embedding is not None:
            # Update with new 1024-dim embedding
            cur.execute(
                "UPDATE documents SET embedding = %s, version = 2 WHERE id = %s",
                (new_embedding.tolist(), doc_id)
            )
    
    conn.commit()
    conn.close()
```

## Best Practices

### 1. Version Pinning

```toml
# pyproject.toml
[tool.poetry.dependencies]
steadytext = "^2.1.0"  # Allows 2.1.x updates

# Or strict pinning
steadytext = "2.1.0"  # Exact version
```

### 2. Gradual Migration

```python
class MigrationWrapper:
    """Wrapper to support both old and new behavior."""
    
    def __init__(self, use_new_version=True):
        self.use_new_version = use_new_version
    
    def generate(self, prompt):
        if self.use_new_version:
            # New v2.1.x behavior
            result = steadytext.generate(prompt)
            return result if result else "Fallback"
        else:
            # Simulate old behavior
            result = steadytext.generate(prompt)
            if result is None:
                return f"{prompt}. This is a deterministic fallback..."
            return result
```

### 3. Testing Migration

```python
import pytest

def test_migration_compatibility():
    """Test that migration handles all cases."""
    
    # Test None handling
    result = steadytext.generate("test")
    if result is None:
        # Ensure fallback works
        assert "fallback" in handle_none_result("test")
    
    # Test embedding dimensions
    embedding = steadytext.embed("test")
    if embedding is not None:
        assert embedding.shape == (1024,)
    
    # Test seed consistency
    if result is not None:
        result2 = steadytext.generate("test", seed=42)
        assert result == result2
```

### 4. Monitoring Migration

```python
import logging

logger = logging.getLogger(__name__)

def monitored_generate(prompt):
    """Generate with migration monitoring."""
    start_time = time.time()
    
    result = steadytext.generate(prompt)
    
    if result is None:
        logger.warning(
            "Generation returned None",
            extra={
                "prompt": prompt[:50],
                "duration": time.time() - start_time
            }
        )
        return "Migration fallback"
    
    logger.info(
        "Generation successful",
        extra={
            "prompt": prompt[:50],
            "duration": time.time() - start_time,
            "length": len(result)
        }
    )
    return result
```

### 5. Rollback Strategy

```python
class VersionedSteadyText:
    """Support multiple versions during migration."""
    
    def __init__(self, version="2.1"):
        self.version = version
    
    def generate(self, prompt, **kwargs):
        if self.version == "2.0":
            # Simulate old behavior
            result = steadytext.generate(prompt, **kwargs)
            if result is None:
                return self._fallback_generate(prompt)
            return result
        else:
            # New behavior
            return steadytext.generate(prompt, **kwargs)
    
    def _fallback_generate(self, prompt):
        """Simulate v2.0.x fallback."""
        return f"{prompt}. This is a deterministic fallback..."
```

## Migration Timeline

### Recommended Migration Path

1. **Week 1-2**: Update error handling for None returns
2. **Week 3-4**: Test in development environment
3. **Week 5-6**: Gradual rollout to staging
4. **Week 7-8**: Production deployment with monitoring
5. **Week 9+**: Remove compatibility wrappers

### Deprecation Schedule

- **v2.0.x**: Supported until December 2024
- **v1.x**: Security fixes only
- **v0.x**: No longer supported

## Getting Help

- **Migration Issues**: [GitHub Issues](https://github.com/diwank/steadytext/issues)
- **Documentation**: [Full docs](https://steadytext.readthedocs.io)
- **Community**: [Discussions](https://github.com/diwank/steadytext/discussions)

## Quick Reference Card

```python
# Check version
import steadytext
print(steadytext.__version__)

# Handle v2.1.x None returns
result = steadytext.generate("prompt")
text = result if result else "default"

# Check embedding dimensions
emb = steadytext.embed("text")
if emb is not None:
    assert emb.shape == (1024,)

# Use deterministic seeds
text1 = steadytext.generate("hi", seed=42)
text2 = steadytext.generate("hi", seed=42)
assert text1 == text2  # Always true
```
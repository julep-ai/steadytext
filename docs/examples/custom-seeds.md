# Custom Seeds Guide

Learn how to use custom seeds in SteadyText for reproducible variations in text generation and embeddings.

## Overview

SteadyText uses seeds to control randomness, allowing you to:
- Generate different outputs for the same prompt
- Ensure reproducible results across runs
- Create variations while maintaining determinism
- Control randomness in production systems

## Table of Contents

- [Understanding Seeds](#understanding-seeds)
- [Text Generation with Seeds](#text-generation-with-seeds)
- [Streaming with Seeds](#streaming-with-seeds)
- [Embeddings with Seeds](#embeddings-with-seeds)
- [Seed Strategies](#seed-strategies)
- [CLI Seed Usage](#cli-seed-usage)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Understanding Seeds

### What is a Seed?

A seed is an integer that initializes the random number generator. Same seed + same input = same output, always.

```python
import steadytext

# Default seed (42) - always same result
text1 = steadytext.generate("Hello world")
text2 = steadytext.generate("Hello world")
assert text1 == text2  # Always true

# Custom seeds - different results
text3 = steadytext.generate("Hello world", seed=123)
text4 = steadytext.generate("Hello world", seed=456)
assert text3 != text4  # Different seeds, different outputs
```

### Seed Behavior

- **Deterministic**: Same seed always produces same result
- **Independent**: Each operation uses its own seed
- **Cascading**: Seed affects all random choices in generation
- **Cross-platform**: Same seed works identically everywhere

## Basic Seed Usage

### Simple Text Generation

```python
import steadytext

# Default seed (42) - consistent across runs
text1 = steadytext.generate("Write a haiku about AI")
text2 = steadytext.generate("Write a haiku about AI")
assert text1 == text2  # Always identical

# Custom seed - reproducible but different from default
text3 = steadytext.generate("Write a haiku about AI", seed=123)
text4 = steadytext.generate("Write a haiku about AI", seed=123)
assert text3 == text4  # Same seed, same result
assert text1 != text3  # Different seeds, different results

print("Default seed result:", text1)
print("Custom seed result:", text3)
```

### Embedding Generation

```python
import numpy as np

# Default seed embeddings
emb1 = steadytext.embed("artificial intelligence")
emb2 = steadytext.embed("artificial intelligence")
assert np.array_equal(emb1, emb2)  # Identical

# Custom seed embeddings
emb3 = steadytext.embed("artificial intelligence", seed=456)
emb4 = steadytext.embed("artificial intelligence", seed=456)
assert np.array_equal(emb3, emb4)  # Same seed, same result
assert not np.array_equal(emb1, emb3)  # Different seeds, different embeddings

# Calculate similarity between different seed embeddings
similarity = np.dot(emb1, emb3)  # Cosine similarity (vectors are normalized)
print(f"Similarity between different seeds: {similarity:.3f}")
```

## Reproducible Research

### Research Workflow Example

```python
import steadytext
import json
from datetime import datetime

class ReproducibleResearch:
    def __init__(self, base_seed=42):
        self.base_seed = base_seed
        self.current_seed = base_seed
        self.results = []
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "base_seed": base_seed,
            "steadytext_version": "2.1.0+",
        }
    
    def generate_with_logging(self, prompt, **kwargs):
        """Generate text and log the result with seed information."""
        result = steadytext.generate(prompt, seed=self.current_seed, **kwargs)
        
        self.results.append({
            "seed": self.current_seed,
            "prompt": prompt,
            "result": result,
            "kwargs": kwargs,
            "timestamp": datetime.now().isoformat()
        })
        
        self.current_seed += 1  # Increment for next generation
        return result
    
    def embed_with_logging(self, text, **kwargs):
        """Generate embedding and log the result with seed information."""
        embedding = steadytext.embed(text, seed=self.current_seed, **kwargs)
        
        self.results.append({
            "seed": self.current_seed,
            "text": text,
            "embedding": embedding.tolist(),  # Convert numpy array to list
            "kwargs": kwargs,
            "timestamp": datetime.now().isoformat()
        })
        
        self.current_seed += 1
        return embedding
    
    def save_results(self, filename):
        """Save all results to a JSON file for reproducibility."""
        with open(filename, 'w') as f:
            json.dump({
                "metadata": self.metadata,
                "results": self.results
            }, f, indent=2)
    
    def load_and_verify(self, filename):
        """Load previous results and verify reproducibility."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        print("Verifying reproducibility...")
        for result in data["results"]:
            if "prompt" in result:  # Text generation
                regenerated = steadytext.generate(
                    result["prompt"], 
                    seed=result["seed"],
                    **result["kwargs"]
                )
                if regenerated == result["result"]:
                    print(f"✓ Seed {result['seed']}: Text generation verified")
                else:
                    print(f"✗ Seed {result['seed']}: Text generation FAILED")
            
            elif "text" in result:  # Embedding
                regenerated = steadytext.embed(
                    result["text"],
                    seed=result["seed"],
                    **result["kwargs"]
                )
                if np.allclose(regenerated, result["embedding"], atol=1e-6):\n                    print(f"✓ Seed {result['seed']}: Embedding verified")\n                else:\n                    print(f"✗ Seed {result['seed']}: Embedding FAILED")\n\n# Usage example\nresearch = ReproducibleResearch(base_seed=100)\n\n# Conduct research with automatic seed management\nresearch_prompts = [\n    "Explain the benefits of renewable energy",\n    "Describe the future of artificial intelligence",\n    "Summarize the importance of biodiversity"\n]\n\nfor prompt in research_prompts:\n    result = research.generate_with_logging(prompt, max_new_tokens=200)\n    print(f"Generated {len(result)} characters for: {prompt[:50]}...")\n\n# Generate embeddings for analysis\nembedding_texts = ["AI", "machine learning", "deep learning"]\nfor text in embedding_texts:\n    embedding = research.embed_with_logging(text)\n    print(f"Generated embedding for: {text}")\n\n# Save results for reproducibility\nresearch.save_results("research_results.json")\nprint("Results saved to research_results.json")\n\n# Later: verify reproducibility\nresearch.load_and_verify("research_results.json")\n```\n\n## A/B Testing with Seeds\n\n### Content Comparison Framework\n\n```python\nimport steadytext\nfrom typing import List, Dict, Any\n\nclass ABTester:\n    def __init__(self):\n        self.variants = {}\n    \n    def create_variants(self, prompt: str, variant_seeds: List[int], **kwargs) -> Dict[str, str]:\n        """Create multiple variants of the same prompt using different seeds."""\n        variants = {}\n        for i, seed in enumerate(variant_seeds):\n            variant_name = f"variant_{chr(65 + i)}"  # A, B, C, etc.\n            variants[variant_name] = steadytext.generate(\n                prompt, \n                seed=seed, \n                **kwargs\n            )\n        return variants\n    \n    def compare_variants(self, prompt: str, seeds: List[int], **kwargs) -> Dict[str, Any]:\n        """Generate and compare multiple variants."""\n        variants = self.create_variants(prompt, seeds, **kwargs)\n        \n        analysis = {\n            "prompt": prompt,\n            "seeds": seeds,\n            "variants": variants,\n            "stats": {\n                variant: {\n                    "length": len(text),\n                    "word_count": len(text.split()),\n                    "seed": seeds[i]\n                }\n                for i, (variant, text) in enumerate(variants.items())\n            }\n        }\n        \n        return analysis\n    \n    def batch_compare(self, prompts: List[str], seeds: List[int], **kwargs) -> List[Dict[str, Any]]:\n        """Compare variants for multiple prompts."""\n        return [self.compare_variants(prompt, seeds, **kwargs) for prompt in prompts]\n\n# Usage example\ntester = ABTester()\n\n# Define test variants with specific seeds\ntest_seeds = [100, 200, 300, 400, 500]\n\n# Single prompt A/B test\nresult = tester.compare_variants(\n    "Write a compelling product description for a smartwatch",\n    seeds=test_seeds[:3],  # Test 3 variants\n    max_new_tokens=150\n)\n\nprint("=== A/B Test Results ===")\nfor variant, text in result["variants"].items():\n    stats = result["stats"][variant]\n    print(f"\\n{variant.upper()} (seed {stats['seed']}):")\n    print(f"Length: {stats['length']} chars, {stats['word_count']} words")\n    print(f"Text: {text[:100]}...")\n\n# Batch testing for multiple prompts\nmarketing_prompts = [\n    "Create an email subject line for a summer sale",\n    "Write a social media post about a new product launch",\n    "Compose a customer testimonial request"\n]\n\nbatch_results = tester.batch_compare(\n    marketing_prompts, \n    seeds=[42, 123, 456],\n    max_new_tokens=100\n)\n\nprint("\\n=== Batch A/B Test Results ===")\nfor i, result in enumerate(batch_results):\n    print(f"\\nPrompt {i+1}: {result['prompt'][:50]}...")\n    for variant, text in result["variants"].items():\n        seed = result["stats"][variant]["seed"]\n        print(f"  {variant} (seed {seed}): {text[:80]}...")\n```\n\n### Email Campaign Testing\n\n```python\nimport steadytext\nimport random\n\ndef generate_email_variants(subject_base: str, body_base: str, num_variants: int = 5):\n    """Generate email variants for A/B testing."""\n    # Use consistent seed ranges for reproducibility\n    seeds = [1000 + i * 100 for i in range(num_variants)]\n    \n    variants = []\n    for i, seed in enumerate(seeds):\n        subject = steadytext.generate(\n            f"Create an engaging email subject line based on: {subject_base}",\n            seed=seed,\n            max_new_tokens=20\n        ).strip()\n        \n        body = steadytext.generate(\n            f"Write a compelling email body for: {body_base}",\n            seed=seed,\n            max_new_tokens=200\n        ).strip()\n        \n        variants.append({\n            "variant_id": f"V{i+1}",\n            "seed": seed,\n            "subject": subject,\n            "body": body\n        })\n    \n    return variants\n\n# Generate email campaign variants\nvariants = generate_email_variants(\n    subject_base="New product launch announcement",\n    body_base="Introducing our revolutionary AI-powered smartwatch with health monitoring"\n)\n\nprint("=== Email Campaign Variants ===")\nfor variant in variants:\n    print(f"\\n{variant['variant_id']} (seed {variant['seed']}):\")\n    print(f"Subject: {variant['subject']}")\n    print(f"Body: {variant['body'][:100]}...")\n```\n\n## Content Variations\n\n### Style and Tone Variations\n\n```python\nimport steadytext\n\ndef generate_style_variations(base_content: str, styles: Dict[str, int]):\n    \"\"\"Generate content in different styles using specific seeds.\"\"\"\n    variations = {}\n    \n    for style_name, seed in styles.items():\n        prompt = f"Rewrite the following content in a {style_name} style: {base_content}"\n        variation = steadytext.generate(\n            prompt,\n            seed=seed,\n            max_new_tokens=250\n        )\n        variations[style_name] = {\n            "seed": seed,\n            "content": variation\n        }\n    \n    return variations\n\n# Define styles with consistent seeds\nstyles = {\n    "professional": 2000,\n    "casual": 2100,\n    "technical": 2200,\n    "creative": 2300,\n    "humorous": 2400\n}\n\nbase_content = "Our new software helps businesses manage their data more efficiently."\n\nvariations = generate_style_variations(base_content, styles)\n\nprint("=== Style Variations ===")\nfor style, data in variations.items():\n    print(f"\\n{style.upper()} (seed {data['seed']}):")\n    print(data['content'])\n```\n\n### Multi-Language Content\n\n```python\nimport steadytext\n\ndef generate_multilingual_content(english_content: str, languages: Dict[str, int]):\n    \"\"\"Generate content adapted for different languages/cultures using seeds.\"\"\"\n    adaptations = {}\n    \n    for language, seed in languages.items():\n        prompt = f\"Adapt this content for {language} audience, keeping cultural context in mind: {english_content}\"\n        adaptation = steadytext.generate(\n            prompt,\n            seed=seed,\n            max_new_tokens=200\n        )\n        adaptations[language] = {\n            "seed": seed,\n            "content": adaptation\n        }\n    \n    return adaptations\n\n# Define languages with seeds\nlanguages = {\n    "Spanish": 3000,\n    "French": 3100,\n    "German": 3200,\n    "Japanese": 3300,\n    "Brazilian Portuguese": 3400\n}\n\nenglish_content = "Join our community of innovators and discover cutting-edge technology solutions."\n\nadaptations = generate_multilingual_content(english_content, languages)\n\nprint("=== Multilingual Adaptations ===")\nfor language, data in adaptations.items():\n    print(f"\\n{language} (seed {data['seed']}):")\n    print(data['content'])\n```\n\n## Embedding Experiments\n\n### Semantic Similarity Analysis\n\n```python\nimport steadytext\nimport numpy as np\nfrom sklearn.cluster import KMeans\nimport matplotlib.pyplot as plt\n\ndef analyze_embedding_variations(text: str, seeds: List[int]):\n    \"\"\"Analyze how different seeds affect embeddings for the same text.\"\"\"\n    embeddings = []\n    for seed in seeds:\n        emb = steadytext.embed(text, seed=seed)\n        embeddings.append(emb)\n    \n    embeddings = np.array(embeddings)\n    \n    # Calculate pairwise similarities\n    similarities = []\n    for i in range(len(embeddings)):\n        for j in range(i+1, len(embeddings)):\n            sim = np.dot(embeddings[i], embeddings[j])\n            similarities.append(sim)\n    \n    analysis = {\n        "text": text,\n        "seeds": seeds,\n        "embeddings": embeddings,\n        "pairwise_similarities": similarities,\n        "mean_similarity": np.mean(similarities),\n        "std_similarity": np.std(similarities),\n        "min_similarity": np.min(similarities),\n        "max_similarity": np.max(similarities)\n    }\n    \n    return analysis\n\n# Analyze embedding variations\ntest_text = "artificial intelligence and machine learning"\ntest_seeds = [4000, 4100, 4200, 4300, 4400]\n\nanalysis = analyze_embedding_variations(test_text, test_seeds)\n\nprint(f"=== Embedding Variation Analysis ===")\nprint(f"Text: {analysis['text']}")\nprint(f"Seeds tested: {analysis['seeds']}")\nprint(f"Mean similarity: {analysis['mean_similarity']:.4f}")\nprint(f"Std similarity: {analysis['std_similarity']:.4f}")\nprint(f"Range: {analysis['min_similarity']:.4f} - {analysis['max_similarity']:.4f}")\n\n# Detailed similarity matrix\nprint("\\nSimilarity Matrix:")\nembeddings = analysis['embeddings']\nfor i, seed_i in enumerate(test_seeds):\n    row = []\n    for j, seed_j in enumerate(test_seeds):\n        if i == j:\n            sim = 1.0\n        else:\n            sim = np.dot(embeddings[i], embeddings[j])\n        row.append(f"{sim:.3f}")\n    print(f"Seed {seed_i}: {' '.join(row)}")\n```\n\n### Domain-Specific Embedding Clusters\n\n```python\nimport steadytext\nimport numpy as np\nfrom sklearn.cluster import KMeans\nfrom collections import defaultdict\n\ndef create_domain_embeddings(domains: Dict[str, List[str]], seed_base: int = 5000):\n    \"\"\"Create embeddings for different domains using consistent seeding.\"\"\"\n    domain_embeddings = defaultdict(list)\n    \n    for domain, texts in domains.items():\n        domain_seed = seed_base + hash(domain) % 1000  # Consistent seed per domain\n        \n        for text in texts:\n            embedding = steadytext.embed(text, seed=domain_seed)\n            domain_embeddings[domain].append({\n                "text": text,\n                "embedding": embedding,\n                "seed": domain_seed\n            })\n    \n    return dict(domain_embeddings)\n\n# Define domain-specific texts\ndomains = {\n    "technology": [\n        "artificial intelligence",\n        "machine learning",\n        "deep learning",\n        "neural networks",\n        "computer vision"\n    ],\n    "healthcare": [\n        "medical diagnosis",\n        "patient care",\n        "clinical trials",\n        "pharmaceutical research",\n        "telemedicine"\n    ],\n    "finance": [\n        "investment strategy",\n        "risk management",\n        "financial planning",\n        "market analysis",\n        "portfolio optimization"\n    ]\n}\n\n# Generate embeddings\ndomain_embeddings = create_domain_embeddings(domains)\n\n# Analyze domain clustering\nall_embeddings = []\nall_labels = []\nall_texts = []\n\nfor domain, items in domain_embeddings.items():\n    for item in items:\n        all_embeddings.append(item['embedding'])\n        all_labels.append(domain)\n        all_texts.append(item['text'])\n\nall_embeddings = np.array(all_embeddings)\n\n# Perform clustering\nkmeans = KMeans(n_clusters=3, random_state=42)\ncluster_labels = kmeans.fit_predict(all_embeddings)\n\nprint("=== Domain Clustering Results ===")\nfor i, (text, true_domain, predicted_cluster) in enumerate(zip(all_texts, all_labels, cluster_labels)):\n    print(f"{text:25} | True: {true_domain:10} | Cluster: {predicted_cluster}")\n\n# Calculate clustering accuracy\nfrom sklearn.metrics import adjusted_rand_score\nlabel_to_int = {label: i for i, label in enumerate(set(all_labels))}\ntrue_labels_int = [label_to_int[label] for label in all_labels]\n\naccuracy = adjusted_rand_score(true_labels_int, cluster_labels)\nprint(f"\\nClustering accuracy (ARI): {accuracy:.3f}")\n```\n\n## CLI Workflows\n\n### Batch Processing Scripts\n\n```bash\n#!/bin/bash\n# batch_generation.sh - Generate content variants using CLI\n\n# Define seeds for different variants\nSEEDS=(1000 2000 3000 4000 5000)\nPROMPT="Write a brief product description for a smartwatch"\n\necho "=== Batch Generation with Different Seeds ==="\n\nfor i in "${!SEEDS[@]}"; do\n    seed=${SEEDS[$i]}\n    variant_name="variant_$(echo $i | tr '0-4' 'A-E')"  # A, B, C, D, E\n    \n    echo \"\"\n    echo \"$variant_name (seed $seed):\"\n    echo \"$PROMPT\" | st --seed $seed --max-new-tokens 100\n    echo \"---\"\ndone\n```\n\n```bash\n#!/bin/bash\n# embedding_comparison.sh - Compare embeddings with different seeds\n\nTEXT=\"artificial intelligence\"\nSEEDS=(6000 6100 6200)\n\necho \"=== Embedding Comparison ===\"\necho \"Text: $TEXT\"\n\nfor seed in \"${SEEDS[@]}\"; do\n    echo \"\"\n    echo \"Seed $seed:\"\n    st embed \"$TEXT\" --seed $seed --format json | jq '.[:5]'  # Show first 5 dimensions\ndone\n```\n\n### Reproducible Research Pipeline\n\n```bash\n#!/bin/bash\n# research_pipeline.sh - Complete research workflow with seeds\n\nRESEARCH_DIR=\"./research_$(date +%Y%m%d_%H%M%S)\"\nBASE_SEED=7000\n\nmkdir -p \"$RESEARCH_DIR\"\ncd \"$RESEARCH_DIR\"\n\necho \"=== Research Pipeline Started ===\" | tee research.log\necho \"Base seed: $BASE_SEED\" | tee -a research.log\necho \"Directory: $RESEARCH_DIR\" | tee -a research.log\n\n# Generate research questions\necho \"Generating research questions...\" | tee -a research.log\necho \"Generate 5 research questions about AI ethics\" | \\\n    st --seed $BASE_SEED --max-new-tokens 200 > questions.txt\n\n# Generate detailed explanations\necho \"Generating detailed explanations...\" | tee -a research.log\ncounter=0\nwhile IFS= read -r question; do\n    if [[ -n \"$question\" && \"$question\" != *\"Generate\"* ]]; then\n        seed=$((BASE_SEED + 100 + counter * 10))\n        echo \"Processing: $question (seed $seed)\" | tee -a research.log\n        echo \"$question\" | st --seed $seed --max-new-tokens 300 > \"explanation_$counter.txt\"\n        counter=$((counter + 1))\n    fi\ndone < questions.txt\n\n# Generate embeddings for analysis\necho \"Generating embeddings...\" | tee -a research.log\nfor file in explanation_*.txt; do\n    if [[ -f \"$file\" ]]; then\n        seed=$((BASE_SEED + 500))\n        echo \"Creating embedding for $file (seed $seed)\" | tee -a research.log\n        cat \"$file\" | st embed --seed $seed --format json > \"${file%.txt}_embedding.json\"\n    fi\ndone\n\necho \"Research pipeline completed. Results in: $RESEARCH_DIR\" | tee -a research.log\necho \"Files generated:\" | tee -a research.log\nls -la | tee -a research.log\n```\n\n## Advanced Patterns\n\n### Seed Scheduling and Management\n\n```python\nimport steadytext\nfrom typing import Iterator, List, Dict, Any\nimport hashlib\n\nclass SeedManager:\n    \"\"\"Advanced seed management for complex workflows.\"\"\"\n    \n    def __init__(self, base_seed: int = 42):\n        self.base_seed = base_seed\n        self.used_seeds = set()\n        self.seed_history = []\n    \n    def get_deterministic_seed(self, context: str) -> int:\n        \"\"\"Generate deterministic seed based on context string.\"\"\"\n        # Create reproducible seed from context\n        context_hash = hashlib.md5(context.encode()).hexdigest()\n        seed = self.base_seed + int(context_hash[:8], 16) % 10000\n        \n        self.used_seeds.add(seed)\n        self.seed_history.append({\n            "context": context,\n            "seed": seed,\n            "method": "deterministic"\n        })\n        \n        return seed\n    \n    def get_sequential_seed(self, increment: int = 1) -> int:\n        \"\"\"Get next seed in sequence.\"\"\"\n        seed = self.base_seed + len(self.seed_history) * increment\n        \n        self.used_seeds.add(seed)\n        self.seed_history.append({\n            "context": f"sequential_{len(self.seed_history)}\",\n            "seed": seed,\n            "method": "sequential"\n        })\n        \n        return seed\n    \n    def get_category_seed(self, category: str, item_id: int = 0) -> int:\n        \"\"\"Get seed for specific category and item.\"\"\"\n        category_base = hash(category) % 1000\n        seed = self.base_seed + category_base * 100 + item_id\n        \n        self.used_seeds.add(seed)\n        self.seed_history.append({\n            "context": f"{category}_{item_id}\",\n            "seed": seed,\n            "method": "category\",\n            \"category\": category,\n            \"item_id\": item_id\n        })\n        \n        return seed\n    \n    def generate_with_context(self, prompt: str, context: str, **kwargs) -> str:\n        \"\"\"Generate text with context-based seed.\"\"\"\n        seed = self.get_deterministic_seed(context)\n        return steadytext.generate(prompt, seed=seed, **kwargs)\n    \n    def embed_with_context(self, text: str, context: str, **kwargs):\n        \"\"\"Generate embedding with context-based seed.\"\"\"\n        seed = self.get_deterministic_seed(context)\n        return steadytext.embed(text, seed=seed, **kwargs)\n    \n    def export_seed_history(self) -> List[Dict[str, Any]]:\n        \"\"\"Export seed usage history for reproducibility.\"\"\"\n        return self.seed_history.copy()\n\n# Usage example\nmanager = SeedManager(base_seed=10000)\n\n# Context-based generation\ncontents = [\n    (\"Write a technical blog post about AI\", \"blog_technical_ai\"),\n    (\"Create a social media post about innovation\", \"social_innovation\"),\n    (\"Generate a product description\", \"product_smartwatch\")\n]\n\nresults = []\nfor prompt, context in contents:\n    result = manager.generate_with_context(\n        prompt, \n        context, \n        max_new_tokens=150\n    )\n    results.append({\n        \"context\": context,\n        \"prompt\": prompt,\n        \"result\": result\n    })\n\n# Category-based generation\ncategories = [\"marketing\", \"technical\", \"creative\"]\nfor category in categories:\n    for i in range(3):  # 3 items per category\n        seed = manager.get_category_seed(category, i)\n        prompt = f\"Write a {category} message about our new product\"\n        result = steadytext.generate(prompt, seed=seed, max_new_tokens=100)\n        print(f\"{category}_{i} (seed {seed}): {result[:50]}...\")\n\n# Export history for reproducibility\nhistory = manager.export_seed_history()\nprint(f\"\\nGenerated {len(history)} items with managed seeds\")\nfor entry in history[-5:]:  # Show last 5 entries\n    print(f\"Context: {entry['context']}, Seed: {entry['seed']}, Method: {entry['method']}\")\n```\n\n### Conditional Seed Strategies\n\n```python\nimport steadytext\nfrom enum import Enum\nfrom typing import Optional, Callable\n\nclass SeedStrategy(Enum):\n    DETERMINISTIC = \"deterministic\"  # Same input always gives same seed\n    SEQUENTIAL = \"sequential\"        # Incrementing seed sequence\n    RANDOM_BOUNDED = \"random_bounded\" # Random within bounds\n    CONTENT_BASED = \"content_based\"   # Seed based on content analysis\n\nclass ConditionalSeedGenerator:\n    \"\"\"Generate seeds based on content and context conditions.\"\"\"\n    \n    def __init__(self, base_seed: int = 42):\n        self.base_seed = base_seed\n        self.counters = {}\n    \n    def analyze_content(self, content: str) -> Dict[str, Any]:\n        \"\"\"Analyze content to determine appropriate seed strategy.\"\"\"\n        word_count = len(content.split())\n        has_technical_terms = any(term in content.lower() for term in \n                                ['algorithm', 'neural', 'machine', 'ai', 'data'])\n        has_creative_intent = any(term in content.lower() for term in \n                             ['story', 'creative', 'imagine', 'artistic'])\n        \n        return {\n            \"word_count\": word_count,\n            \"is_technical\": has_technical_terms,\n            \"is_creative\": has_creative_intent,\n            \"is_short\": word_count < 10,\n            \"is_long\": word_count > 50\n        }\n    \n    def determine_strategy(self, content: str, context: Optional[str] = None) -> SeedStrategy:\n        \"\"\"Determine best seed strategy based on content analysis.\"\"\"\n        analysis = self.analyze_content(content)\n        \n        if analysis[\"is_creative\"]:\n            return SeedStrategy.RANDOM_BOUNDED  # More variation for creative content\n        elif analysis[\"is_technical\"]:\n            return SeedStrategy.DETERMINISTIC   # Consistency for technical content\n        elif analysis[\"is_short\"]:\n            return SeedStrategy.CONTENT_BASED   # Content-based for short prompts\n        else:\n            return SeedStrategy.SEQUENTIAL      # Sequential for general content\n    \n    def generate_seed(self, content: str, strategy: Optional[SeedStrategy] = None, \n                     context: Optional[str] = None) -> int:\n        \"\"\"Generate seed using specified or determined strategy.\"\"\"\n        if strategy is None:\n            strategy = self.determine_strategy(content, context)\n        \n        if strategy == SeedStrategy.DETERMINISTIC:\n            # Hash-based deterministic seed\n            content_hash = hash(content) % 10000\n            return self.base_seed + content_hash\n        \n        elif strategy == SeedStrategy.SEQUENTIAL:\n            # Sequential counter per context\n            key = context or \"default\"\n            if key not in self.counters:\n                self.counters[key] = 0\n            self.counters[key] += 1\n            return self.base_seed + self.counters[key] * 100\n        \n        elif strategy == SeedStrategy.RANDOM_BOUNDED:\n            # Bounded random based on content\n            content_hash = abs(hash(content))\n            random_offset = content_hash % 1000\n            return self.base_seed + 5000 + random_offset\n        \n        elif strategy == SeedStrategy.CONTENT_BASED:\n            # Seed based on content characteristics\n            analysis = self.analyze_content(content)\n            seed_offset = (\n                analysis[\"word_count\"] * 10 +\n                (100 if analysis[\"is_technical\"] else 0) +\n                (200 if analysis[\"is_creative\"] else 0)\n            )\n            return self.base_seed + seed_offset\n        \n        return self.base_seed\n    \n    def smart_generate(self, content: str, context: Optional[str] = None, **kwargs) -> str:\n        \"\"\"Generate text with automatically chosen seed strategy.\"\"\"\n        strategy = self.determine_strategy(content, context)\n        seed = self.generate_seed(content, strategy, context)\n        \n        print(f\"Strategy: {strategy.value}, Seed: {seed}\")\n        return steadytext.generate(content, seed=seed, **kwargs)\n\n# Usage example\ngenerator = ConditionalSeedGenerator(base_seed=20000)\n\n# Test different content types\ntest_prompts = [\n    \"Write a creative story about a robot\",  # Should use RANDOM_BOUNDED\n    \"Explain the neural network algorithm\",  # Should use DETERMINISTIC\n    \"Hello\",                                 # Should use CONTENT_BASED\n    \"Generate a comprehensive technical report about machine learning applications in healthcare\"  # Should use SEQUENTIAL\n]\n\nprint(\"=== Conditional Seed Strategy Results ===\")\nfor prompt in test_prompts:\n    print(f\"\\nPrompt: {prompt}\")\n    result = generator.smart_generate(prompt, max_new_tokens=50)\n    print(f\"Result: {result[:80]}...\")\n```\n\n## Best Practices\n\n### 1. Documentation and Reproducibility\n\n```python\n# Always document your seeds\nCONSTANT_SEEDS = {\n    \"BASELINE_RESEARCH\": 42,\n    \"VARIATION_A\": 100,\n    \"VARIATION_B\": 200,\n    \"CREATIVE_CONTENT\": 300,\n    \"TECHNICAL_CONTENT\": 400,\n    \"PRODUCTION_DEFAULT\": 500\n}\n\n# Use descriptive seed values\ndef generate_with_purpose(prompt: str, purpose: str):\n    seed = CONSTANT_SEEDS.get(purpose.upper(), CONSTANT_SEEDS[\"BASELINE_RESEARCH\"])\n    return steadytext.generate(prompt, seed=seed)\n```\n\n### 2. Seed Range Management\n\n```python\n# Organize seeds by ranges to avoid conflicts\nSEED_RANGES = {\n    \"research\": (1000, 1999),\n    \"production\": (2000, 2999),\n    \"testing\": (3000, 3999),\n    \"experiments\": (4000, 4999),\n    \"benchmarks\": (5000, 5999)\n}\n\ndef get_range_seed(category: str, offset: int = 0) -> int:\n    if category not in SEED_RANGES:\n        raise ValueError(f\"Unknown category: {category}\")\n    \n    start, end = SEED_RANGES[category]\n    seed = start + offset\n    \n    if seed > end:\n        raise ValueError(f\"Seed {seed} exceeds range for {category} ({start}-{end})\")\n    \n    return seed\n```\n\n### 3. Testing and Validation\n\n```python\ndef validate_reproducibility(prompt: str, seed: int, iterations: int = 5):\n    \"\"\"Validate that a prompt+seed combination is truly reproducible.\"\"\"\n    results = []\n    for i in range(iterations):\n        result = steadytext.generate(prompt, seed=seed)\n        results.append(result)\n    \n    # Check if all results are identical\n    is_reproducible = all(result == results[0] for result in results)\n    \n    print(f\"Reproducibility test for seed {seed}: {'PASS' if is_reproducible else 'FAIL'}\")\n    if not is_reproducible:\n        print(\"Different results found:\")\n        for i, result in enumerate(results):\n            print(f\"  Iteration {i+1}: {result[:50]}...\")\n    \n    return is_reproducible\n\n# Test key seeds\nfor purpose, seed in CONSTANT_SEEDS.items():\n    validate_reproducibility(\"Test prompt for validation\", seed)\n```\n\nThis comprehensive guide demonstrates the power and flexibility of custom seeds in SteadyText. By using seeds strategically, you can achieve reproducible research, conduct effective A/B testing, generate controlled variations, and build robust content generation pipelines.\n
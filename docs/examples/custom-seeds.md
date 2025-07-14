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
  - [What is a Seed?](#what-is-a-seed)
  - [Seed Behavior](#seed-behavior)
- [Basic Seed Usage](#basic-seed-usage)
  - [Simple Text Generation](#simple-text-generation)
  - [Embedding Generation](#embedding-generation)
- [Reproducible Research](#reproducible-research)
  - [Research Workflow Example](#research-workflow-example)
- [A/B Testing with Seeds](#ab-testing-with-seeds)
  - [Content Comparison Framework](#content-comparison-framework)
  - [Email Campaign Testing](#email-campaign-testing)
- [Content Variations](#content-variations)
  - [Style and Tone Variations](#style-and-tone-variations)
  - [Multi-Language Content](#multi-language-content)
- [Embedding Experiments](#embedding-experiments)
  - [Semantic Similarity Analysis](#semantic-similarity-analysis)
  - [Domain-Specific Embedding Clusters](#domain-specific-embedding-clusters)
- [CLI Workflows](#cli-workflows)
  - [Batch Processing Scripts](#batch-processing-scripts)
  - [Reproducible Research Pipeline](#reproducible-research-pipeline)
- [Advanced Patterns](#advanced-patterns)
  - [Seed Scheduling and Management](#seed-scheduling-and-management)
  - [Conditional Seed Strategies](#conditional-seed-strategies)
- [Best Practices](#best-practices)

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
                if np.allclose(regenerated, result["embedding"], atol=1e-6):
                    print(f"✓ Seed {result['seed']}: Embedding verified")
                else:
                    print(f"✗ Seed {result['seed']}: Embedding FAILED")

# Usage example
research = ReproducibleResearch(base_seed=100)

# Conduct research with automatic seed management
research_prompts = [
    "Explain the benefits of renewable energy",
    "Describe the future of artificial intelligence",
    "Summarize the importance of biodiversity"
]

for prompt in research_prompts:
    result = research.generate_with_logging(prompt, max_new_tokens=200)
    print(f"Generated {len(result)} characters for: {prompt[:50]}...")

# Generate embeddings for analysis
embedding_texts = ["AI", "machine learning", "deep learning"]
for text in embedding_texts:
    embedding = research.embed_with_logging(text)
    print(f"Generated embedding for: {text}")

# Save results for reproducibility
research.save_results("research_results.json")
print("Results saved to research_results.json")

# Later: verify reproducibility
research.load_and_verify("research_results.json")
```

## A/B Testing with Seeds

A/B testing is a powerful technique for comparing different variations of content. With SteadyText's deterministic seeds, you can create reproducible variations for testing.

### Content Comparison Framework

Create a framework for systematic A/B testing of generated content.

```python
import steadytext
import json
from datetime import datetime

class ABTestFramework:
    def __init__(self, base_prompt, variations=5, base_seed=42):
        self.base_prompt = base_prompt
        self.variations = variations
        self.base_seed = base_seed
        self.results = []
    
    def generate_variations(self):
        """Generate multiple variations of content using different seeds."""
        for i in range(self.variations):
            seed = self.base_seed + i
            content = steadytext.generate(self.base_prompt, seed=seed)
            
            self.results.append({
                "variation_id": f"variant_{chr(65+i)}",  # A, B, C, etc.
                "seed": seed,
                "content": content,
                "metrics": {
                    "length": len(content),
                    "word_count": len(content.split()),
                    "timestamp": datetime.now().isoformat()
                }
            })
        
        return self.results
    
    def compare_variations(self):
        """Compare all generated variations."""
        print(f"Generated {len(self.results)} variations for: {self.base_prompt[:50]}...")
        print("-" * 80)
        
        for result in self.results:
            print(f"\n{result['variation_id']} (seed: {result['seed']}):")
            print(f"Length: {result['metrics']['length']} chars")
            print(f"Words: {result['metrics']['word_count']}")
            print(f"Preview: {result['content'][:100]}...")
    
    def save_test_results(self, filename):
        """Save A/B test results for analysis."""
        with open(filename, 'w') as f:
            json.dump({
                "test_config": {
                    "base_prompt": self.base_prompt,
                    "variations": self.variations,
                    "base_seed": self.base_seed
                },
                "results": self.results
            }, f, indent=2)

# Example usage
ab_test = ABTestFramework(
    base_prompt="Write a compelling email subject line for our new product launch",
    variations=3
)

ab_test.generate_variations()
ab_test.compare_variations()
ab_test.save_test_results("ab_test_results.json")
```

### Email Campaign Testing

Test different email variations with consistent seeding for reproducibility.

```python
import steadytext

class EmailCampaignTester:
    def __init__(self, campaign_name, target_audience):
        self.campaign_name = campaign_name
        self.target_audience = target_audience
        self.templates = {}
    
    def generate_email_variant(self, tone, seed):
        """Generate email content with specific tone and seed."""
        prompt = f"""Write a marketing email for {self.campaign_name} targeting {self.target_audience}.
        Tone: {tone}
        Include: subject line, greeting, body, and call-to-action."""
        
        return steadytext.generate(prompt, seed=seed, max_new_tokens=400)
    
    def create_campaign_variants(self):
        """Create multiple email variants with different tones."""
        tones = ["professional", "friendly", "urgent", "casual", "exclusive"]
        
        for i, tone in enumerate(tones):
            seed = 1000 + i  # Consistent seed for each tone
            self.templates[tone] = {
                "seed": seed,
                "content": self.generate_email_variant(tone, seed),
                "tone": tone
            }
        
        return self.templates
    
    def test_personalization(self, template_tone, customer_names):
        """Test personalization with consistent results."""
        base_template = self.templates[template_tone]
        personalized = []
        
        for i, name in enumerate(customer_names):
            # Use customer-specific seed for personalization
            customer_seed = base_template["seed"] + hash(name) % 1000
            
            prompt = f"Personalize this email for {name}: {base_template['content'][:200]}..."
            personalized_content = steadytext.generate(prompt, seed=customer_seed, max_new_tokens=100)
            
            personalized.append({
                "customer": name,
                "seed": customer_seed,
                "preview": personalized_content[:100] + "..."
            })
        
        return personalized

# Example usage
tester = EmailCampaignTester("Summer Sale 2024", "young professionals")
variants = tester.create_campaign_variants()

# Test personalization
customers = ["Alice Johnson", "Bob Smith", "Carol Davis"]
personalized = tester.test_personalization("friendly", customers)

for p in personalized:
    print(f"Email for {p['customer']} (seed: {p['seed']}):")
    print(p['preview'])
    print()
```

## Content Variations

Generate content in different styles, tones, and languages using seed-based variations.

### Style and Tone Variations

Use different seeds to generate content with various stylistic approaches.

```python
import steadytext

class StyleVariationGenerator:
    def __init__(self, base_content):
        self.base_content = base_content
        self.styles = {
            "formal": 2000,
            "casual": 2001,
            "technical": 2002,
            "creative": 2003,
            "minimalist": 2004
        }
    
    def generate_style_variant(self, style):
        """Generate content in a specific style."""
        if style not in self.styles:
            raise ValueError(f"Unknown style: {style}")
        
        seed = self.styles[style]
        prompt = f"Rewrite this in a {style} style: {self.base_content}"
        
        return steadytext.generate(prompt, seed=seed, max_new_tokens=300)
    
    def generate_all_styles(self):
        """Generate content in all available styles."""
        results = {}
        
        for style in self.styles:
            results[style] = {
                "seed": self.styles[style],
                "content": self.generate_style_variant(style)
            }
        
        return results
    
    def compare_lengths(self, results):
        """Compare the length of different style variants."""
        for style, data in results.items():
            word_count = len(data["content"].split())
            print(f"{style.capitalize()}: {word_count} words (seed: {data['seed']})")

# Example usage
base_text = "Our company provides innovative solutions for modern businesses."
generator = StyleVariationGenerator(base_text)

all_styles = generator.generate_all_styles()
generator.compare_lengths(all_styles)

# Show samples
for style, data in all_styles.items():
    print(f"\n{style.upper()} (seed: {data['seed']}):")
    print(data["content"][:150] + "...")
```

### Multi-Language Content

Adapt content for different languages and cultural contexts using seeds.

```python
import steadytext

class MultilingualContentGenerator:
    def __init__(self, source_content, source_language="English"):
        self.source_content = source_content
        self.source_language = source_language
        # Assign consistent seeds for each language
        self.language_seeds = {
            "Spanish": 3000,
            "French": 3001,
            "German": 3002,
            "Italian": 3003,
            "Portuguese": 3004,
            "Japanese": 3005,
            "Chinese": 3006
        }
    
    def translate_content(self, target_language):
        """Generate content adapted for target language."""
        if target_language not in self.language_seeds:
            raise ValueError(f"Unsupported language: {target_language}")
        
        seed = self.language_seeds[target_language]
        prompt = f"""Translate and culturally adapt this {self.source_language} content to {target_language}:
        
        {self.source_content}
        
        Maintain the tone and intent while making it natural for {target_language} speakers."""
        
        return steadytext.generate(prompt, seed=seed, max_new_tokens=400)
    
    def create_multilingual_set(self):
        """Create content in all supported languages."""
        translations = {
            self.source_language: {
                "seed": 2999,  # Original content seed
                "content": self.source_content
            }
        }
        
        for language in self.language_seeds:
            translations[language] = {
                "seed": self.language_seeds[language],
                "content": self.translate_content(language)
            }
        
        return translations
    
    def verify_consistency(self, language, expected_seed):
        """Verify that content generation is consistent for a language."""
        result1 = self.translate_content(language)
        result2 = self.translate_content(language)
        
        return result1 == result2  # Should be True due to same seed

# Example usage
content = "Welcome to our platform! We're excited to help you achieve your goals."
generator = MultilingualContentGenerator(content)

# Generate all translations
translations = generator.create_multilingual_set()

# Verify consistency
print("Consistency check:")
for lang in ["Spanish", "French", "German"]:
    is_consistent = generator.verify_consistency(lang, generator.language_seeds[lang])
    print(f"{lang}: {'✓' if is_consistent else '✗'}")
```

## Embedding Experiments

Explore how seeds affect embeddings and use them for various analysis tasks.

### Semantic Similarity Analysis

Analyze how different seeds affect the semantic representation of text.

```python
import steadytext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticAnalyzer:
    def __init__(self):
        self.embeddings = {}
    
    def analyze_seed_impact(self, text, seeds):
        """Analyze how different seeds affect embeddings of the same text."""
        results = []
        
        for seed in seeds:
            embedding = steadytext.embed(text, seed=seed)
            self.embeddings[f"{text}_seed{seed}"] = embedding
            results.append({
                "seed": seed,
                "embedding": embedding,
                "norm": np.linalg.norm(embedding)
            })
        
        # Calculate pairwise similarities
        embeddings_matrix = np.array([r["embedding"] for r in results])
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        return {
            "text": text,
            "seeds": seeds,
            "embeddings": results,
            "similarity_matrix": similarity_matrix
        }
    
    def compare_semantic_drift(self, texts, base_seed=42, num_seeds=5):
        """Compare how much embeddings drift across seeds for different texts."""
        drift_analysis = []
        
        for text in texts:
            seeds = [base_seed + i for i in range(num_seeds)]
            embeddings = []
            
            for seed in seeds:
                emb = steadytext.embed(text, seed=seed)
                embeddings.append(emb)
            
            # Calculate average embedding and deviations
            avg_embedding = np.mean(embeddings, axis=0)
            deviations = [np.linalg.norm(emb - avg_embedding) for emb in embeddings]
            
            drift_analysis.append({
                "text": text,
                "avg_deviation": np.mean(deviations),
                "max_deviation": np.max(deviations),
                "min_deviation": np.min(deviations)
            })
        
        return drift_analysis
    
    def find_stable_pairs(self, text1, text2, num_seeds=10):
        """Find seed pairs that maintain relative similarity."""
        base_similarity = np.dot(
            steadytext.embed(text1, seed=42),
            steadytext.embed(text2, seed=42)
        )
        
        stable_pairs = []
        
        for i in range(num_seeds):
            seed1 = 100 + i
            seed2 = 200 + i
            
            emb1 = steadytext.embed(text1, seed=seed1)
            emb2 = steadytext.embed(text2, seed=seed2)
            similarity = np.dot(emb1, emb2)
            
            if abs(similarity - base_similarity) < 0.05:  # Within 5% of base
                stable_pairs.append({
                    "seed_pair": (seed1, seed2),
                    "similarity": similarity,
                    "difference": similarity - base_similarity
                })
        
        return stable_pairs

# Example usage
analyzer = SemanticAnalyzer()

# Analyze seed impact
result = analyzer.analyze_seed_impact("artificial intelligence", seeds=[42, 123, 456, 789])
print(f"Similarity matrix for '{result['text']}':")
print(result["similarity_matrix"])

# Compare drift across different texts
texts = ["AI", "machine learning", "deep learning", "neural networks"]
drift = analyzer.compare_semantic_drift(texts)
for d in drift:
    print(f"{d['text']}: avg deviation = {d['avg_deviation']:.4f}")
```

### Domain-Specific Embedding Clusters

Create consistent embeddings for domain-specific text clustering.

```python
import steadytext
import numpy as np
from collections import defaultdict

class DomainEmbeddingManager:
    def __init__(self):
        # Assign seed ranges to different domains
        self.domain_seeds = {
            "medical": 5000,
            "legal": 5100,
            "technical": 5200,
            "financial": 5300,
            "educational": 5400
        }
        self.embeddings = defaultdict(dict)
    
    def embed_domain_text(self, text, domain):
        """Embed text using domain-specific seed."""
        if domain not in self.domain_seeds:
            raise ValueError(f"Unknown domain: {domain}")
        
        seed = self.domain_seeds[domain]
        embedding = steadytext.embed(text, seed=seed)
        
        self.embeddings[domain][text] = embedding
        return embedding
    
    def create_domain_clusters(self, domain, texts):
        """Create embeddings for multiple texts in a domain."""
        clusters = []
        
        for i, text in enumerate(texts):
            # Use domain seed + index for consistency within domain
            seed = self.domain_seeds[domain] + i
            embedding = steadytext.embed(text, seed=seed)
            
            clusters.append({
                "text": text,
                "embedding": embedding,
                "seed": seed
            })
        
        return clusters
    
    def cross_domain_similarity(self, text):
        """Compare how the same text is embedded across domains."""
        results = {}
        
        for domain in self.domain_seeds:
            embedding = self.embed_domain_text(text, domain)
            results[domain] = embedding
        
        # Calculate cross-domain similarities
        similarities = {}
        domains = list(results.keys())
        
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                d1, d2 = domains[i], domains[j]
                sim = np.dot(results[d1], results[d2])
                similarities[f"{d1}-{d2}"] = sim
        
        return similarities
    
    def find_domain_keywords(self, domain, candidate_words):
        """Find words that cluster well within a domain."""
        domain_embeddings = []
        
        for word in candidate_words:
            emb = self.embed_domain_text(word, domain)
            domain_embeddings.append(emb)
        
        # Calculate centroid
        centroid = np.mean(domain_embeddings, axis=0)
        
        # Find words closest to centroid
        distances = []
        for i, word in enumerate(candidate_words):
            dist = np.linalg.norm(domain_embeddings[i] - centroid)
            distances.append((word, dist))
        
        # Sort by distance (closest first)
        distances.sort(key=lambda x: x[1])
        
        return distances[:10]  # Top 10 domain keywords

# Example usage
manager = DomainEmbeddingManager()

# Create domain-specific clusters
medical_terms = ["diagnosis", "treatment", "patient", "symptoms", "medication"]
medical_clusters = manager.create_domain_clusters("medical", medical_terms)

legal_terms = ["contract", "litigation", "defendant", "jurisdiction", "statute"]
legal_clusters = manager.create_domain_clusters("legal", legal_terms)

# Analyze cross-domain similarity
similarities = manager.cross_domain_similarity("analysis")
print("Cross-domain similarities for 'analysis':")
for pair, sim in similarities.items():
    print(f"{pair}: {sim:.3f}")

# Find domain keywords
candidates = ["research", "study", "analysis", "report", "findings", "evidence", 
              "data", "results", "conclusion", "methodology"]
medical_keywords = manager.find_domain_keywords("medical", candidates)
print("\nTop medical domain keywords:")
for word, dist in medical_keywords[:5]:
    print(f"{word}: {dist:.3f}")
```

## CLI Workflows

Use SteadyText's CLI with custom seeds for batch processing and automation.

### Batch Processing Scripts

Create shell scripts for processing multiple items with different seeds.

```bash
#!/bin/bash
# batch_generate.sh - Generate multiple variations with different seeds

# Configuration
BASE_PROMPT="Write a product description for"
PRODUCTS=("laptop" "smartphone" "headphones" "smartwatch" "tablet")
BASE_SEED=1000

# Create output directory
mkdir -p output/product_descriptions

# Generate descriptions for each product with multiple seeds
for i in "${!PRODUCTS[@]}"; do
    product="${PRODUCTS[$i]}"
    
    # Generate 3 variations per product
    for variation in 0 1 2; do
        seed=$((BASE_SEED + i * 10 + variation))
        output_file="output/product_descriptions/${product}_v${variation}.txt"
        
        echo "Generating description for $product (seed: $seed)..."
        echo "$BASE_PROMPT $product" | st generate --seed $seed > "$output_file"
    done
done

# Generate comparison report
echo "Product Description Variations Report" > output/report.txt
echo "====================================" >> output/report.txt
echo "" >> output/report.txt

for product in "${PRODUCTS[@]}"; do
    echo "## $product" >> output/report.txt
    for v in 0 1 2; do
        echo "### Variation $v:" >> output/report.txt
        head -n 3 "output/product_descriptions/${product}_v${v}.txt" >> output/report.txt
        echo "" >> output/report.txt
    done
done
```

### Reproducible Research Pipeline

Build complete research workflows with seed management.

```python
#!/usr/bin/env python3
# research_pipeline.py - Reproducible research pipeline with SteadyText

import subprocess
import json
import hashlib
from datetime import datetime
from pathlib import Path

class ResearchPipeline:
    def __init__(self, project_name, base_seed=42):
        self.project_name = project_name
        self.base_seed = base_seed
        self.output_dir = Path(f"research_{project_name}")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize metadata
        self.metadata = {
            "project": project_name,
            "base_seed": base_seed,
            "start_time": datetime.now().isoformat(),
            "experiments": []
        }
    
    def run_experiment(self, name, prompts, seeds_per_prompt=3):
        """Run an experiment with multiple prompts and seeds."""
        experiment_data = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "prompts": [],
            "results": []
        }
        
        for prompt_idx, prompt in enumerate(prompts):
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            
            for seed_offset in range(seeds_per_prompt):
                seed = self.base_seed + prompt_idx * 100 + seed_offset
                
                # Run generation via CLI
                result = subprocess.run(
                    ["st", "generate", "--seed", str(seed), "--json"],
                    input=prompt,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    output = json.loads(result.stdout)
                    
                    experiment_data["results"].append({
                        "prompt": prompt,
                        "prompt_hash": prompt_hash,
                        "seed": seed,
                        "output": output["text"],
                        "metadata": output.get("metadata", {})
                    })
                else:
                    print(f"Error generating for seed {seed}: {result.stderr}")
        
        # Save experiment data
        exp_file = self.output_dir / f"experiment_{name}.json"
        with open(exp_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        self.metadata["experiments"].append(name)
        return experiment_data
    
    def generate_embeddings(self, texts, name="embeddings"):
        """Generate embeddings for a list of texts."""
        embeddings_data = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "embeddings": []
        }
        
        for idx, text in enumerate(texts):
            seed = self.base_seed + 10000 + idx
            
            # Run embedding via CLI
            result = subprocess.run(
                ["st", "embed", "--seed", str(seed), "--json"],
                input=text,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                output = json.loads(result.stdout)
                embeddings_data["embeddings"].append({
                    "text": text,
                    "seed": seed,
                    "embedding": output["embedding"][:10],  # Store first 10 dims
                    "shape": output["shape"]
                })
        
        # Save embeddings data
        emb_file = self.output_dir / f"embeddings_{name}.json"
        with open(emb_file, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        return embeddings_data
    
    def finalize(self):
        """Finalize the research pipeline and save metadata."""
        self.metadata["end_time"] = datetime.now().isoformat()
        
        # Save metadata
        meta_file = self.output_dir / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Create summary report
        report = [
            f"# Research Pipeline Report: {self.project_name}",
            f"Generated on: {self.metadata['end_time']}",
            f"Base seed: {self.metadata['base_seed']}",
            "",
            "## Experiments Conducted:",
            ""
        ]
        
        for exp in self.metadata["experiments"]:
            report.append(f"- {exp}")
        
        report_file = self.output_dir / "REPORT.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Research pipeline completed. Results in: {self.output_dir}")

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ResearchPipeline("climate_study", base_seed=2024)
    
    # Run text generation experiments
    climate_prompts = [
        "Explain the greenhouse effect in simple terms",
        "Describe renewable energy solutions",
        "What are the impacts of deforestation?"
    ]
    
    pipeline.run_experiment("climate_basics", climate_prompts)
    
    # Generate embeddings for key terms
    key_terms = [
        "climate change",
        "global warming",
        "carbon footprint",
        "sustainability",
        "renewable energy"
    ]
    
    pipeline.generate_embeddings(key_terms, "climate_terms")
    
    # Finalize and generate report
    pipeline.finalize()
```

## Advanced Patterns

Advanced techniques for seed management in complex applications.

### Seed Scheduling and Management

Implement sophisticated seed management for large-scale applications.

```python
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class SeedScheduler:
    def __init__(self, base_seed=42):
        self.base_seed = base_seed
        self.seed_registry = {}
        self.time_based_seeds = {}
        self.usage_stats = {}
    
    def register_task(self, task_name: str, seed_range: Tuple[int, int]):
        """Register a task with a specific seed range."""
        if task_name in self.seed_registry:
            raise ValueError(f"Task {task_name} already registered")
        
        start, end = seed_range
        # Check for overlaps
        for existing_task, (existing_start, existing_end) in self.seed_registry.items():
            if start <= existing_end and end >= existing_start:
                raise ValueError(f"Seed range overlaps with task {existing_task}")
        
        self.seed_registry[task_name] = seed_range
        self.usage_stats[task_name] = {"count": 0, "last_used": None}
    
    def get_task_seed(self, task_name: str, sub_id: str = None) -> int:
        """Get a seed for a specific task and optional sub-identifier."""
        if task_name not in self.seed_registry:
            raise ValueError(f"Task {task_name} not registered")
        
        start, end = self.seed_registry[task_name]
        
        if sub_id:
            # Hash the sub_id to get a consistent offset
            hash_val = int(hashlib.md5(sub_id.encode()).hexdigest(), 16)
            seed = start + (hash_val % (end - start))
        else:
            # Use sequential seeds
            count = self.usage_stats[task_name]["count"]
            seed = start + (count % (end - start))
            self.usage_stats[task_name]["count"] += 1
        
        self.usage_stats[task_name]["last_used"] = datetime.now()
        return seed
    
    def create_time_based_seed(self, task_name: str, interval: timedelta) -> int:
        """Create seeds that change based on time intervals."""
        current_time = datetime.now()
        
        if task_name in self.time_based_seeds:
            last_time, last_seed = self.time_based_seeds[task_name]
            if current_time - last_time < interval:
                return last_seed
        
        # Generate new seed for this time period
        time_bucket = int(current_time.timestamp() // interval.total_seconds())
        seed = self.get_task_seed(task_name, f"time_{time_bucket}")
        
        self.time_based_seeds[task_name] = (current_time, seed)
        return seed
    
    def get_user_seed(self, user_id: str, feature: str) -> int:
        """Get a consistent seed for a user-feature combination."""
        combined_id = f"{user_id}_{feature}"
        return self.get_task_seed("user_features", combined_id)
    
    def export_seed_map(self) -> Dict:
        """Export the current seed mapping for documentation."""
        return {
            "base_seed": self.base_seed,
            "registry": self.seed_registry,
            "usage_stats": {
                task: {
                    "count": stats["count"],
                    "last_used": stats["last_used"].isoformat() if stats["last_used"] else None
                }
                for task, stats in self.usage_stats.items()
            }
        }

# Example usage
scheduler = SeedScheduler(base_seed=1000)

# Register different tasks with non-overlapping seed ranges
scheduler.register_task("content_generation", (1000, 2000))
scheduler.register_task("embeddings", (2000, 3000))
scheduler.register_task("user_features", (3000, 4000))
scheduler.register_task("ab_testing", (4000, 5000))

# Get seeds for different purposes
content_seed = scheduler.get_task_seed("content_generation", "article_123")
embedding_seed = scheduler.get_task_seed("embeddings", "doc_456")
user_seed = scheduler.get_user_seed("user_789", "recommendations")

# Time-based seeds (changes every hour)
hourly_seed = scheduler.create_time_based_seed("ab_testing", timedelta(hours=1))

print(f"Content seed: {content_seed}")
print(f"Embedding seed: {embedding_seed}")
print(f"User seed: {user_seed}")
print(f"Hourly seed: {hourly_seed}")

# Export seed map for documentation
seed_map = scheduler.export_seed_map()
print("\nSeed Map:")
print(json.dumps(seed_map, indent=2))
```

### Conditional Seed Strategies

Use different seeding strategies based on content characteristics.

```python
import steadytext
import re
from enum import Enum
from typing import Optional

class ContentType(Enum):
    TECHNICAL = "technical"
    CREATIVE = "creative"
    BUSINESS = "business"
    CASUAL = "casual"
    ACADEMIC = "academic"

class ConditionalSeedStrategy:
    def __init__(self, base_seed=42):
        self.base_seed = base_seed
        
        # Define seed offsets for different content types
        self.content_type_offsets = {
            ContentType.TECHNICAL: 0,
            ContentType.CREATIVE: 1000,
            ContentType.BUSINESS: 2000,
            ContentType.CASUAL: 3000,
            ContentType.ACADEMIC: 4000
        }
        
        # Define seed modifiers for content characteristics
        self.modifiers = {
            "short": 0,
            "medium": 100,
            "long": 200,
            "formal": 0,
            "informal": 50,
            "urgent": 300,
            "evergreen": 400
        }
    
    def detect_content_type(self, text: str) -> ContentType:
        """Detect content type based on text characteristics."""
        text_lower = text.lower()
        
        # Simple heuristics for content type detection
        technical_keywords = ["algorithm", "function", "database", "api", "code"]
        creative_keywords = ["story", "imagine", "creative", "artistic", "design"]
        business_keywords = ["revenue", "market", "strategy", "customer", "roi"]
        academic_keywords = ["research", "study", "hypothesis", "analysis", "theory"]
        
        scores = {
            ContentType.TECHNICAL: sum(1 for kw in technical_keywords if kw in text_lower),
            ContentType.CREATIVE: sum(1 for kw in creative_keywords if kw in text_lower),
            ContentType.BUSINESS: sum(1 for kw in business_keywords if kw in text_lower),
            ContentType.ACADEMIC: sum(1 for kw in academic_keywords if kw in text_lower),
            ContentType.CASUAL: 1  # Default score
        }
        
        return max(scores, key=scores.get)
    
    def determine_length_category(self, text: str) -> str:
        """Determine if content should be short, medium, or long."""
        word_count = len(text.split())
        
        if word_count < 50:
            return "short"
        elif word_count < 200:
            return "medium"
        else:
            return "long"
    
    def determine_formality(self, text: str) -> str:
        """Determine if content should be formal or informal."""
        informal_indicators = ["you're", "don't", "can't", "won't", "!", "?"]
        informal_count = sum(1 for indicator in informal_indicators if indicator in text)
        
        return "informal" if informal_count > 2 else "formal"
    
    def calculate_seed(self, 
                      text: str, 
                      override_type: Optional[ContentType] = None,
                      urgency: bool = False,
                      evergreen: bool = False) -> int:
        """Calculate appropriate seed based on content characteristics."""
        # Determine content type
        content_type = override_type or self.detect_content_type(text)
        
        # Get base offset for content type
        seed = self.base_seed + self.content_type_offsets[content_type]
        
        # Add modifiers based on characteristics
        seed += self.modifiers[self.determine_length_category(text)]
        seed += self.modifiers[self.determine_formality(text)]
        
        if urgency:
            seed += self.modifiers["urgent"]
        elif evergreen:
            seed += self.modifiers["evergreen"]
        
        return seed
    
    def generate_with_strategy(self, 
                             prompt: str,
                             override_type: Optional[ContentType] = None,
                             **kwargs) -> str:
        """Generate content using conditional seed strategy."""
        seed = self.calculate_seed(prompt, override_type, 
                                 kwargs.get("urgency", False),
                                 kwargs.get("evergreen", False))
        
        # Remove our custom kwargs before passing to generate
        generate_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ["urgency", "evergreen"]}
        
        return steadytext.generate(prompt, seed=seed, **generate_kwargs)
    
    def batch_generate_variants(self, base_prompt: str) -> Dict[str, str]:
        """Generate variants for different content types."""
        variants = {}
        
        for content_type in ContentType:
            seed = self.calculate_seed(base_prompt, override_type=content_type)
            prompt = f"Write this in a {content_type.value} style: {base_prompt}"
            
            variants[content_type.value] = {
                "seed": seed,
                "content": steadytext.generate(prompt, seed=seed, max_new_tokens=200)
            }
        
        return variants

# Example usage
strategy = ConditionalSeedStrategy(base_seed=5000)

# Test content type detection and seed calculation
test_prompts = [
    "Explain how REST APIs work",
    "Write a creative story about the future",
    "Analyze market trends for Q4",
    "Hey, what's up with the weather today?",
    "Examine the hypothesis that climate change affects biodiversity"
]

for prompt in test_prompts:
    content_type = strategy.detect_content_type(prompt)
    seed = strategy.calculate_seed(prompt)
    print(f"Prompt: {prompt[:50]}...")
    print(f"Detected type: {content_type.value}, Seed: {seed}")
    print()

# Generate with strategy
technical_prompt = "Explain machine learning algorithms"
result = strategy.generate_with_strategy(
    technical_prompt,
    override_type=ContentType.TECHNICAL,
    max_new_tokens=150
)
print(f"Technical generation (seed: {strategy.calculate_seed(technical_prompt)}):")
print(result[:200] + "...")

# Generate variants for different styles
base_prompt = "Describe the benefits of cloud computing"
variants = strategy.batch_generate_variants(base_prompt)

print("\nContent variants:")
for style, data in variants.items():
    print(f"\n{style.upper()} (seed: {data['seed']}):")
    print(data['content'][:150] + "...")
```

## Best Practices

Follow these best practices to make the most of custom seeds in SteadyText.

### 1. Documentation and Reproducibility

Always document your seed choices and their purposes for future reference.

```python
# Good: Document seed usage
SEED_DOCUMENTATION = {
    "default": 42,
    "testing": {
        "unit_tests": 100,
        "integration_tests": 200,
        "performance_tests": 300
    },
    "production": {
        "content_generation": 1000,
        "embeddings": 2000,
        "personalization": 3000
    },
    "experiments": {
        "ab_test_2024_q1": 4000,
        "feature_rollout_v2": 5000
    }
}

# Create a seed manifest file
import json
with open("seeds.json", "w") as f:
    json.dump(SEED_DOCUMENTATION, f, indent=2)
```

### 2. Seed Range Management

Organize seeds into ranges to avoid conflicts and maintain clarity.

```python
class SeedRanges:
    # Reserve ranges for different purposes
    TESTING = range(0, 1000)
    DEVELOPMENT = range(1000, 2000)
    PRODUCTION = range(2000, 10000)
    USER_SPECIFIC = range(10000, 20000)
    TIME_BASED = range(20000, 30000)
    EXPERIMENTAL = range(30000, 40000)
    
    @staticmethod
    def validate_seed(seed, purpose):
        """Ensure seed is in correct range for its purpose."""
        ranges = {
            "test": SeedRanges.TESTING,
            "dev": SeedRanges.DEVELOPMENT,
            "prod": SeedRanges.PRODUCTION,
            "user": SeedRanges.USER_SPECIFIC,
            "time": SeedRanges.TIME_BASED,
            "exp": SeedRanges.EXPERIMENTAL
        }
        
        if purpose in ranges and seed in ranges[purpose]:
            return True
        return False
```

### 3. Testing and Validation

Regularly validate that your seed-based workflows remain reproducible.

```python
import steadytext
import hashlib

def validate_seed_reproducibility(test_cases):
    """Validate that seeds produce consistent results."""
    failures = []
    
    for test in test_cases:
        prompt = test["prompt"]
        seed = test["seed"]
        expected_hash = test.get("expected_hash")
        
        # Generate twice with same seed
        result1 = steadytext.generate(prompt, seed=seed)
        result2 = steadytext.generate(prompt, seed=seed)
        
        # Check consistency
        if result1 != result2:
            failures.append(f"Inconsistent results for seed {seed}")
        
        # Check against expected hash if provided
        if expected_hash:
            actual_hash = hashlib.md5(result1.encode()).hexdigest()
            if actual_hash != expected_hash:
                failures.append(f"Hash mismatch for seed {seed}")
    
    return len(failures) == 0, failures

# Test cases
test_cases = [
    {"prompt": "Hello", "seed": 42, "expected_hash": "abc123..."},
    {"prompt": "Test prompt", "seed": 100},
    {"prompt": "Another test", "seed": 200}
]

is_valid, errors = validate_seed_reproducibility(test_cases)
if not is_valid:
    print("Validation failed:", errors)
```

This comprehensive guide demonstrates the power and flexibility of custom seeds in SteadyText. By using seeds strategically, you can achieve reproducible research, conduct effective A/B testing, generate controlled variations, and build robust content generation pipelines.
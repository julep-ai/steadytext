#!/usr/bin/env python3
"""
Examples of content generation with SteadyText
"""

import hashlib
import steadytext


# ASCII Art generation
def cowsay(message):
    """Generate ASCII art of a cow saying something"""
    return steadytext.generate(f"Draw ASCII art of a cow saying: {message}")


# Game content generation
def generate_npc_dialogue(npc_name, player_level):
    """Generate consistent NPC dialogue based on player level"""
    return steadytext.generate(
        f"Fantasy RPG: NPC named '{npc_name}' greets a level {player_level} player. "
        f"One sentence of dialogue."
    )


# Mock data generation
def generate_product_review(product_id, stars):
    """Generate fake product reviews for testing"""
    return steadytext.generate(
        f"Write a {stars}-star review for product ID {product_id}. "
        f"Keep it under 50 words."
    )


def generate_user_bio(profession):
    """Generate professional bios"""
    return steadytext.generate(
        f"Write a professional bio for a {profession}. Two sentences, formal tone."
    )


# Documentation generation
def auto_document_function(func_name, params):
    """Generate docstrings for functions"""
    return steadytext.generate(
        f"Write a Python docstring for function {func_name}({params}). "
        f"Include description, parameters, and return value."
    )


# Story generation
def generate_story_chapter(book_id, chapter_num):
    """Generate consistent story chapters"""
    return steadytext.generate(
        f"Fantasy novel ID {book_id}, chapter {chapter_num}. "
        f"Write the opening paragraph."
    )


# Semantic cache keys using embeddings
def semantic_cache_key(query):
    """Generate cache keys based on semantic similarity"""
    embedding = steadytext.embed(query)
    return hashlib.sha256(embedding.tobytes()).hexdigest()


if __name__ == "__main__":
    print("=== Content Generation Examples ===\n")

    # ASCII art
    print("ASCII Cow:")
    print(cowsay("Hello World")[:200] + "...\n")

    # NPC dialogue
    npc_text = generate_npc_dialogue("Wizard Gandor", 5)
    print(f"NPC Dialogue: {npc_text}\n")

    # Product review
    review = generate_product_review("ABC123", 4)
    print(f"Product Review: {review}\n")

    # Professional bio
    bio = generate_user_bio("software engineer")
    print(f"Professional Bio: {bio}\n")

    # Auto documentation
    docstring = auto_document_function("calculate_total", "items, tax_rate")
    print(f"Generated Docstring:\n{docstring[:150]}...\n")

    # Story generation
    story = generate_story_chapter("fantasy_001", 3)
    print(f"Story Opening: {story[:100]}...\n")

    # Semantic caching
    key1 = semantic_cache_key("What is machine learning?")
    key2 = semantic_cache_key("Explain ML to me")
    key3 = semantic_cache_key("How to cook pasta")
    print(f"Cache key for 'What is machine learning?': {key1[:16]}...")
    print(f"Cache key for 'Explain ML to me': {key2[:16]}...")
    print(f"Cache key for 'How to cook pasta': {key3[:16]}...")
    print(
        f"Similar queries have {'similar' if key1[:8] == key2[:8] else 'different'} cache keys"
    )

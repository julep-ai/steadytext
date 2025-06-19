# Testing with AI

Learn how to use SteadyText to build reliable AI tests that never flake.

## The Problem with AI Testing

Traditional AI testing is challenging because:

- **Non-deterministic outputs**: Same input produces different results
- **Flaky tests**: Tests pass sometimes, fail others  
- **Hard to mock**: AI services are complex to replicate
- **Unpredictable behavior**: Edge cases are difficult to reproduce

SteadyText solves these by providing **deterministic AI outputs** - same input always produces the same result.

## Basic Test Patterns

### Deterministic Assertions

```python
import steadytext

def test_ai_code_generation():
    """Test that never flakes - same input, same output."""
    
    def my_ai_function(prompt):
        # Your actual AI function (GPT-4, Claude, etc.)
        # For testing, we compare against SteadyText
        return call_real_ai_service(prompt)
    
    prompt = "write a function to reverse a string"
    result = my_ai_function(prompt)
    expected = steadytext.generate(prompt)
    
    # This assertion is deterministic and reliable
    assert result.strip() == expected.strip()
```

### Embedding Similarity Tests

```python
import numpy as np

def test_document_similarity():
    """Test semantic similarity calculations."""
    
    def calculate_similarity(doc1, doc2):
        vec1 = steadytext.embed(doc1)
        vec2 = steadytext.embed(doc2)
        return np.dot(vec1, vec2)  # Already normalized
    
    # These similarities are always the same
    similarity = calculate_similarity(
        "machine learning algorithms",
        "artificial intelligence methods"
    )
    
    assert similarity > 0.7  # Reliable threshold
    assert similarity < 1.0  # Not identical documents
```

## Mock AI Services

### Simple Mock

```python
class MockAI:
    """Deterministic AI mock for testing."""
    
    def complete(self, prompt: str) -> str:
        return steadytext.generate(prompt)
    
    def embed(self, text: str) -> np.ndarray:
        return steadytext.embed(text)
    
    def chat(self, messages: list) -> str:
        # Convert chat format to single prompt
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" 
                           for msg in messages])
        return steadytext.generate(f"Chat response to: {prompt}")

# Usage in tests
def test_chat_functionality():
    ai = MockAI()
    response = ai.chat([
        {"role": "user", "content": "Hello"}
    ])
    
    # Response is always the same
    assert len(response) > 0
    assert "hello" in response.lower()
```

### Advanced Mock with State

```python
class StatefulMockAI:
    """Mock AI that maintains conversation state."""
    
    def __init__(self):
        self.conversation_history = []
    
    def chat(self, message: str) -> str:
        # Include history in prompt for context
        history = "\n".join(self.conversation_history[-5:])  # Last 5 messages
        full_prompt = f"History: {history}\nNew message: {message}"
        
        response = steadytext.generate(full_prompt)
        
        # Update history
        self.conversation_history.append(f"User: {message}")
        self.conversation_history.append(f"AI: {response}")
        
        return response

def test_conversation_flow():
    """Test multi-turn conversations."""
    ai = StatefulMockAI()
    
    response1 = ai.chat("What's the weather like?")
    response2 = ai.chat("What about tomorrow?")
    
    # Both responses are deterministic
    assert len(response1) > 0
    assert len(response2) > 0
    # Tomorrow's response considers the context
    assert response2 != response1
```

## Test Data Generation

### Reproducible Fixtures

```python
def generate_test_user(user_id: int) -> dict:
    """Generate consistent test user data."""
    return {
        "id": user_id,
        "name": steadytext.generate(f"Generate name for user {user_id}"),
        "bio": steadytext.generate(f"Write bio for user {user_id}"),
        "interests": steadytext.generate(f"List interests for user {user_id}"),
        "embedding": steadytext.embed(f"user {user_id} profile")
    }

def test_user_recommendation():
    """Test user recommendation system."""
    # Generate consistent test users
    users = [generate_test_user(i) for i in range(10)]
    
    # Test similarity calculations
    user1 = users[0]
    user2 = users[1]
    
    similarity = np.dot(user1["embedding"], user2["embedding"])
    
    # Similarity is always the same for these users
    assert isinstance(similarity, float)
    assert -1.0 <= similarity <= 1.0
```

### Fuzz Testing

```python
def generate_fuzz_input(test_name: str, iteration: int) -> str:
    """Generate reproducible fuzz test inputs."""
    seed_prompt = f"Generate test input for {test_name} iteration {iteration}"
    return steadytext.generate(seed_prompt)

def test_parser_robustness():
    """Fuzz test with reproducible inputs."""
    
    def parse_user_input(text):
        # Your parsing function
        return {"words": text.split(), "length": len(text)}
    
    # Generate 100 consistent fuzz inputs
    for i in range(100):
        fuzz_input = generate_fuzz_input("parser_test", i)
        
        try:
            result = parse_user_input(fuzz_input)
            assert isinstance(result, dict)
            assert "words" in result
            assert "length" in result
        except Exception as e:
            # Reproducible error case
            print(f"Fuzz input {i} caused error: {e}")
            print(f"Input was: {fuzz_input[:100]}...")
```

## Integration Testing

### API Testing

```python
import requests_mock

def test_ai_api_integration():
    """Test integration with AI API using deterministic responses."""
    
    with requests_mock.Mocker() as m:
        # Mock the AI API with deterministic responses
        def generate_response(request, context):
            prompt = request.json().get("prompt", "")
            return {"response": steadytext.generate(prompt)}
        
        m.post("https://api.ai-service.com/generate", json=generate_response)
        
        # Your actual API client code
        response = requests.post("https://api.ai-service.com/generate", 
                               json={"prompt": "Hello world"})
        
        # Response is always the same
        expected_text = steadytext.generate("Hello world")
        assert response.json()["response"] == expected_text
```

### Database Testing

```python
import sqlite3

def test_ai_content_storage():
    """Test storing AI-generated content in database."""
    
    # Create in-memory database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE content (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            generated_text TEXT,
            embedding BLOB
        )
    """)
    
    # Generate deterministic content
    prompt = "Write a short story about AI"
    text = steadytext.generate(prompt)
    embedding = steadytext.embed(text)
    
    # Store in database
    cursor.execute("""
        INSERT INTO content (prompt, generated_text, embedding) 
        VALUES (?, ?, ?)
    """, (prompt, text, embedding.tobytes()))
    
    # Verify storage
    cursor.execute("SELECT * FROM content WHERE id = 1")
    row = cursor.fetchone()
    
    assert row[1] == prompt
    assert row[2] == text
    assert len(row[3]) == 1024 * 4  # 1024 float32 values
    
    conn.close()
```

## Performance Testing

### Consistency Benchmarks

```python
import time

def test_generation_performance():
    """Test that generation performance is consistent."""
    
    prompt = "Explain machine learning in one paragraph"
    times = []
    
    # Warm up cache
    steadytext.generate(prompt)
    
    # Measure cached performance
    for _ in range(10):
        start = time.time()
        result = steadytext.generate(prompt)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    
    # Cached calls should be very fast
    assert avg_time < 0.1  # Less than 100ms
    
    # All results should be identical
    results = [steadytext.generate(prompt) for _ in range(5)]
    assert all(r == results[0] for r in results)
```

## Best Practices

!!! tip "Testing Guidelines"
    1. **Use deterministic prompts**: Keep test prompts simple and specific
    2. **Cache warmup**: Call functions once before timing tests
    3. **Mock external services**: Use SteadyText to replace real AI APIs
    4. **Test edge cases**: Generate consistent edge case inputs
    5. **Version pin**: Keep SteadyText version fixed for test stability

!!! warning "Limitations"
    - **Model changes**: Updates to SteadyText models will change outputs
    - **Creative tasks**: SteadyText is optimized for consistency, not creativity
    - **Context length**: Limited to model's context window

## Complete Example

```python
import unittest
import numpy as np
import steadytext

class TestAIFeatures(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ai = MockAI()
        self.test_prompts = [
            "Write a function to sort a list",
            "Explain what is machine learning",
            "Generate a product description"
        ]
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic."""
        for prompt in self.test_prompts:
            result1 = steadytext.generate(prompt)
            result2 = steadytext.generate(prompt)
            self.assertEqual(result1, result2)
    
    def test_embedding_consistency(self):
        """Test that embeddings are consistent."""
        text = "test embedding consistency"
        vec1 = steadytext.embed(text)
        vec2 = steadytext.embed(text)
        np.testing.assert_array_equal(vec1, vec2)
    
    def test_mock_ai_service(self):
        """Test mock AI service."""
        response = self.mock_ai.complete("Hello")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Response should be deterministic
        response2 = self.mock_ai.complete("Hello")
        self.assertEqual(response, response2)

if __name__ == "__main__":
    unittest.main()
```

This comprehensive testing approach ensures your AI features are reliable, reproducible, and maintainable.
# Python Integration Examples

This guide provides practical examples of integrating Mojo with Python code, focusing on real-world use cases and performance optimization patterns.

## Example 1: Accelerating NumPy Operations

### Problem
Python/NumPy code that's too slow for large arrays.

### Python Original
```python
import numpy as np

def compute_distances_python(points1, points2):
    """Compute pairwise Euclidean distances between two sets of points."""
    n1, d = points1.shape
    n2, _ = points2.shape
    distances = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            diff = points1[i] - points2[j]
            distances[i, j] = np.sqrt(np.sum(diff ** 2))
    
    return distances
```

### Mojo Acceleration
Create `fast_distances.mojo`:

```mojo
from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from math import sqrt
from memory import memset_zero
from os import abort

@export
fn PyInit_fast_distances() -> PythonObject:
    try:
        var m = PythonModuleBuilder("fast_distances")
        m.def_function[compute_distances]("compute_distances")
        return m.finalize()
    except e:
        return abort[PythonObject](str(e))

fn compute_distances(points1_obj: PythonObject, points2_obj: PythonObject) raises -> PythonObject:
    """Compute pairwise distances using Mojo for speed."""
    var np = Python.import_module("numpy")
    
    # Get dimensions
    var n1 = Int(points1_obj.shape[0])
    var n2 = Int(points2_obj.shape[0])
    var d = Int(points1_obj.shape[1])
    
    # Convert to Mojo arrays for fast processing
    var points1 = DynamicVector[Float64](n1 * d)
    var points2 = DynamicVector[Float64](n2 * d)
    
    # Copy data
    for i in range(n1 * d):
        points1.push_back(Float64(points1_obj.flat[i]))
    for i in range(n2 * d):
        points2.push_back(Float64(points2_obj.flat[i]))
    
    # Compute distances
    var distances = DynamicVector[Float64](n1 * n2)
    
    @parameter
    fn compute_row(i: Int):
        for j in range(n2):
            var sum_sq = 0.0
            for k in range(d):
                var diff = points1[i * d + k] - points2[j * d + k]
                sum_sq += diff * diff
            distances[i * n2 + j] = sqrt(sum_sq)
    
    # Parallel computation
    for i in range(n1):
        compute_row(i)
    
    # Convert back to NumPy
    var result = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            result[i, j] = distances[i * n2 + j]
    
    return result
```

### Usage Example
```python
import numpy as np
import max.mojo.importer
import sys
sys.path.insert(0, "")
import fast_distances

# Generate test data
points1 = np.random.randn(1000, 3)
points2 = np.random.randn(800, 3)

# Compare performance
import time

start = time.time()
dist_python = compute_distances_python(points1, points2)
python_time = time.time() - start

start = time.time()
dist_mojo = fast_distances.compute_distances(points1, points2)
mojo_time = time.time() - start

print(f"Python time: {python_time:.3f}s")
print(f"Mojo time: {mojo_time:.3f}s")
print(f"Speedup: {python_time/mojo_time:.1f}x")
print(f"Results match: {np.allclose(dist_python, dist_mojo)}")
```

## Example 2: String Processing Acceleration

### Problem
Text preprocessing that's bottlenecking your NLP pipeline.

### Mojo String Processor
```mojo
from python import Python, PythonObject
from python.bindings import PythonModuleBuilder

@export
fn PyInit_text_utils() -> PythonObject:
    try:
        var m = PythonModuleBuilder("text_utils")
        m.def_function[clean_text]("clean_text")
        m.def_function[tokenize_fast]("tokenize_fast")
        m.def_function[batch_normalize]("batch_normalize")
        return m.finalize()
    except e:
        return abort[PythonObject](str(e))

fn clean_text(text_obj: PythonObject) raises -> PythonObject:
    """Remove special characters and normalize whitespace."""
    var text = String(text_obj)
    var result = String()
    
    for i in range(len(text)):
        var c = text[i]
        if c.isalnum() or c.isspace():
            if c.isspace():
                # Normalize whitespace
                if result and result[-1] != ' ':
                    result += ' '
            else:
                result += c.lower() if c.isupper() else c
    
    return PythonObject(result.strip())

fn tokenize_fast(text_obj: PythonObject) raises -> PythonObject:
    """Fast word tokenization."""
    var text = String(text_obj)
    var tokens = List[String]()
    var current_token = String()
    
    for i in range(len(text)):
        var c = text[i]
        if c.isalnum():
            current_token += c
        else:
            if current_token:
                tokens.append(current_token)
                current_token = String()
    
    if current_token:
        tokens.append(current_token)
    
    # Convert to Python list
    var py_list = PythonObject([])
    for token in tokens:
        py_list.append(PythonObject(token[]))
    
    return py_list

fn batch_normalize(texts_obj: PythonObject) raises -> PythonObject:
    """Normalize a batch of texts in parallel."""
    var texts_len = len(texts_obj)
    var results = PythonObject([])
    
    for i in range(texts_len):
        var cleaned = clean_text(texts_obj[i])
        results.append(cleaned)
    
    return results
```

### Integration in Python Pipeline
```python
import text_utils

class FastTextPreprocessor:
    """Drop-in replacement for slow text preprocessing."""
    
    def __init__(self):
        self.tokenizer = text_utils.tokenize_fast
        self.cleaner = text_utils.clean_text
    
    def preprocess_batch(self, texts):
        """Process multiple texts efficiently."""
        # Clean all texts
        cleaned = text_utils.batch_normalize(texts)
        
        # Tokenize
        tokenized = []
        for text in cleaned:
            tokens = self.tokenizer(text)
            tokenized.append(tokens)
        
        return tokenized

# Usage
preprocessor = FastTextPreprocessor()
texts = ["Hello, World!", "This is a TEST.", "Mojo is FAST!!!"]
processed = preprocessor.preprocess_batch(texts)
print(processed)
```

## Example 3: Custom Data Structure

### Problem
Need a high-performance data structure not available in Python.

### Mojo Implementation
```mojo
struct RingBuffer[T: AnyType]:
    """High-performance ring buffer implementation."""
    var data: List[T]
    var capacity: Int
    var head: Int
    var tail: Int
    var size: Int
    
    fn __init__(out self, capacity: Int):
        self.capacity = capacity
        self.data = List[T](capacity=capacity)
        for _ in range(capacity):
            self.data.append(T())  # Pre-allocate
        self.head = 0
        self.tail = 0
        self.size = 0
    
    fn push(mut self, value: T) raises:
        if self.size == self.capacity:
            raise Error("Buffer is full")
        
        self.data[self.tail] = value
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1
    
    fn pop(mut self) raises -> T:
        if self.size == 0:
            raise Error("Buffer is empty")
        
        var value = self.data[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return value
    
    fn is_empty(self) -> Bool:
        return self.size == 0
    
    fn is_full(self) -> Bool:
        return self.size == self.capacity

# Python bindings
@export
fn PyInit_ring_buffer() -> PythonObject:
    try:
        var m = PythonModuleBuilder("ring_buffer")
        m.def_function[create_buffer]("create_buffer")
        m.def_function[push_item]("push")
        m.def_function[pop_item]("pop")
        m.def_function[get_size]("size")
        return m.finalize()
    except e:
        return abort[PythonObject](str(e))

var global_buffer: RingBuffer[Float64]

fn create_buffer(capacity_obj: PythonObject) raises -> PythonObject:
    var capacity = Int(capacity_obj)
    global_buffer = RingBuffer[Float64](capacity)
    return PythonObject(True)

fn push_item(value_obj: PythonObject) raises -> PythonObject:
    var value = Float64(value_obj)
    global_buffer.push(value)
    return PythonObject(None)

fn pop_item(dummy: PythonObject) raises -> PythonObject:
    return PythonObject(global_buffer.pop())

fn get_size(dummy: PythonObject) raises -> PythonObject:
    return PythonObject(global_buffer.size)
```

## Example 4: Integrating with SteadyText

Here's how you might accelerate SteadyText's core operations:

### Mojo-Accelerated Text Generation
```mojo
from python import Python, PythonObject
from random import random_si64

fn deterministic_token_selection(
    logits_obj: PythonObject, 
    seed_obj: PythonObject,
    temperature_obj: PythonObject
) raises -> PythonObject:
    """Fast deterministic token selection for text generation."""
    var np = Python.import_module("numpy")
    
    # Convert inputs
    var seed = Int(seed_obj)
    var temperature = Float64(temperature_obj)
    var logits_array = logits_obj
    var vocab_size = Int(len(logits_obj))
    
    # Apply temperature
    if temperature > 0:
        logits_array = logits_array / temperature
    
    # Compute softmax efficiently
    var max_logit = Float64(np.max(logits_array))
    var exp_sum = 0.0
    
    for i in range(vocab_size):
        exp_sum += exp(Float64(logits_array[i]) - max_logit)
    
    # Create probability distribution
    var probs = np.zeros(vocab_size)
    for i in range(vocab_size):
        probs[i] = exp(Float64(logits_array[i]) - max_logit) / exp_sum
    
    # Deterministic sampling using seed
    random.seed(seed)
    var cumsum = 0.0
    var rand_val = Float64(random_si64(0, 1000000)) / 1000000.0
    
    for i in range(vocab_size):
        cumsum += Float64(probs[i])
        if cumsum > rand_val:
            return PythonObject(i)
    
    return PythonObject(vocab_size - 1)
```

### Integration with SteadyText
```python
# In steadytext/core/generator.py
import max.mojo.importer
import sys
sys.path.insert(0, "")

try:
    import mojo_accelerators
    MOJO_AVAILABLE = True
except ImportError:
    MOJO_AVAILABLE = False

class DeterministicGenerator:
    def __init__(self, model_path, seed=42):
        self.seed = seed
        self.use_mojo = MOJO_AVAILABLE
        # ... existing init code ...
    
    def _sample_token(self, logits, temperature=1.0):
        """Sample token with optional Mojo acceleration."""
        if self.use_mojo:
            try:
                token_id = mojo_accelerators.deterministic_token_selection(
                    logits, self.seed, temperature
                )
                return int(token_id)
            except:
                # Fallback to Python implementation
                pass
        
        # Existing Python implementation
        return self._sample_token_python(logits, temperature)
```

## Performance Tips

### 1. Batch Operations
```mojo
fn process_batch(items_obj: PythonObject) raises -> PythonObject:
    """Process multiple items in one call to minimize overhead."""
    var n = len(items_obj)
    var results = PythonObject([])
    
    # Convert all at once
    var mojo_items = List[Float64](capacity=n)
    for i in range(n):
        mojo_items.append(Float64(items_obj[i]))
    
    # Process in Mojo
    for i in range(n):
        mojo_items[i] *= 2.0  # Example operation
    
    # Convert back
    for item in mojo_items:
        results.append(PythonObject(item[]))
    
    return results
```

### 2. Memory Reuse
```mojo
struct ProcessorState:
    """Reusable state to avoid allocations."""
    var buffer: List[Float64]
    var temp_storage: List[Float64]
    
    fn __init__(out self, max_size: Int):
        self.buffer = List[Float64](capacity=max_size)
        self.temp_storage = List[Float64](capacity=max_size)
    
    fn process(mut self, data: PythonObject) raises -> PythonObject:
        # Reuse allocated memory
        self.buffer.clear()
        # ... processing ...
```

### 3. Parallel Processing
```mojo
from algorithm import parallelize

fn parallel_operation(data_obj: PythonObject) raises -> PythonObject:
    var n = len(data_obj)
    var results = List[Float64](n)
    
    @parameter
    fn process_chunk(i: Int):
        # Process independent chunks in parallel
        results[i] = expensive_operation(Float64(data_obj[i]))
    
    parallelize[process_chunk](n)
    
    # Convert results
    var py_results = PythonObject([])
    for r in results:
        py_results.append(PythonObject(r[]))
    return py_results
```

## Debugging Integration Issues

### 1. Type Checking
```python
def safe_mojo_call(mojo_func, *args):
    """Wrapper with type checking and fallback."""
    try:
        # Validate inputs
        for arg in args:
            if not isinstance(arg, (int, float, str, list, np.ndarray)):
                raise TypeError(f"Unsupported type: {type(arg)}")
        
        # Call Mojo function
        return mojo_func(*args)
    except Exception as e:
        print(f"Mojo call failed: {e}")
        # Fallback to Python implementation
        return python_fallback(*args)
```

### 2. Performance Profiling
```python
import time
import functools

def profile_mojo(func):
    """Decorator to profile Mojo vs Python performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Time Mojo version
        start = time.perf_counter()
        mojo_result = func(*args, **kwargs)
        mojo_time = time.perf_counter() - start
        
        # Time Python equivalent
        python_func = globals().get(f"{func.__name__}_python")
        if python_func:
            start = time.perf_counter()
            python_result = python_func(*args, **kwargs)
            python_time = time.perf_counter() - start
            
            print(f"{func.__name__}:")
            print(f"  Mojo: {mojo_time:.6f}s")
            print(f"  Python: {python_time:.6f}s")
            print(f"  Speedup: {python_time/mojo_time:.2f}x")
        
        return mojo_result
    return wrapper
```

## Next Steps

- Explore [MAX Serving](../max-inference/serving-models.md) for model deployment
- Learn about [Mojo Metaprogramming](../mojo-language/metaprogramming.md)
- Check the [Community Examples](https://github.com/modular/modular-community)
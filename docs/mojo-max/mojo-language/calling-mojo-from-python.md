# Calling Mojo from Python

This guide explains how to create Mojo extensions that can be called from Python, allowing you to write performance-critical code in Mojo while keeping your Python application structure.

> **Beta Feature**: This functionality is in early development and the API may change significantly.

## Overview

Mojo can compile to Python extension modules (`.so` files) that Python can import like any other module. This enables you to:
- Write performance-critical functions in Mojo
- Keep your existing Python codebase
- Gradually migrate bottlenecks to Mojo

## Basic Example

### Project Structure
```
my_project/
├── main.py           # Python entry point
└── mojo_math.mojo    # Mojo module
```

### Step 1: Write Mojo Module

Create `mojo_math.mojo`:

```mojo
from python import PythonObject
from python.bindings import PythonModuleBuilder
import math
from os import abort

@export
fn PyInit_mojo_math() -> PythonObject:
    """Initialize the Python module."""
    try:
        var m = PythonModuleBuilder("mojo_math")
        
        # Register functions
        m.def_function[factorial]("factorial", docstring="Compute n!")
        m.def_function[fibonacci]("fibonacci", docstring="Compute nth Fibonacci number")
        m.def_function[is_prime]("is_prime", docstring="Check if n is prime")
        
        return m.finalize()
    except e:
        return abort[PythonObject](String("Error creating module: " + str(e)))

fn factorial(py_n: PythonObject) raises -> PythonObject:
    """Compute factorial of n."""
    var n = Int(py_n)
    if n < 0:
        raise Error("Factorial not defined for negative numbers")
    return PythonObject(math.factorial(n))

fn fibonacci(py_n: PythonObject) raises -> PythonObject:
    """Compute nth Fibonacci number."""
    var n = Int(py_n)
    if n <= 0:
        return PythonObject(0)
    elif n == 1:
        return PythonObject(1)
    
    var a = 0
    var b = 1
    for _ in range(2, n + 1):
        var temp = a + b
        a = b
        b = temp
    return PythonObject(b)

fn is_prime(py_n: PythonObject) raises -> PythonObject:
    """Check if n is prime."""
    var n = Int(py_n)
    if n <= 1:
        return PythonObject(False)
    if n <= 3:
        return PythonObject(True)
    if n % 2 == 0 or n % 3 == 0:
        return PythonObject(False)
    
    var i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return PythonObject(False)
        i += 6
    return PythonObject(True)
```

### Step 2: Use from Python

Create `main.py`:

```python
import max.mojo.importer
import sys

# Add current directory to Python path
sys.path.insert(0, "")

# Import the Mojo module
import mojo_math

# Use Mojo functions
print(f"10! = {mojo_math.factorial(10)}")
print(f"20th Fibonacci = {mojo_math.fibonacci(20)}")
print(f"Is 17 prime? {mojo_math.is_prime(17)}")

# Performance comparison
import time
import math as py_math

def python_factorial(n):
    return py_math.factorial(n)

# Time comparison
n = 20
start = time.time()
for _ in range(10000):
    python_factorial(n)
py_time = time.time() - start

start = time.time()
for _ in range(10000):
    mojo_math.factorial(n)
mojo_time = time.time() - start

print(f"\nPython time: {py_time:.4f}s")
print(f"Mojo time: {mojo_time:.4f}s")
print(f"Speedup: {py_time/mojo_time:.2f}x")
```

Run it:
```bash
python main.py
```

## Working with Custom Types

### Defining and Exposing Mojo Types

```mojo
from python import PythonObject
from python.bindings import PythonModuleBuilder

struct Vector2D(Movable, Defaultable, Representable):
    var x: Float64
    var y: Float64
    
    fn __init__(out self, x: Float64 = 0.0, y: Float64 = 0.0):
        self.x = x
        self.y = y
    
    fn __repr__(self) -> String:
        return "Vector2D(" + str(self.x) + ", " + str(self.y) + ")"
    
    fn magnitude(self) -> Float64:
        return (self.x * self.x + self.y * self.y).sqrt()
    
    fn dot(self, other: Vector2D) -> Float64:
        return self.x * other.x + self.y * other.y

@export
fn PyInit_vectors() -> PythonObject:
    try:
        var m = PythonModuleBuilder("vectors")
        
        # Register the type
        m.add_type[Vector2D]("Vector2D")
        
        # Register constructor function (workaround for __init__ limitation)
        m.def_function[create_vector]("create_vector")
        m.def_function[vector_magnitude]("magnitude")
        m.def_function[vector_dot]("dot")
        
        return m.finalize()
    except e:
        return abort[PythonObject](str(e))

fn create_vector(x_obj: PythonObject, y_obj: PythonObject) raises -> PythonObject:
    var x = Float64(x_obj)
    var y = Float64(y_obj)
    return PythonObject(alloc=Vector2D(x, y))

fn vector_magnitude(vec_obj: PythonObject) raises -> PythonObject:
    var vec = vec_obj.downcast_value_ptr[Vector2D]()
    return PythonObject(vec[].magnitude())

fn vector_dot(vec1_obj: PythonObject, vec2_obj: PythonObject) raises -> PythonObject:
    var vec1 = vec1_obj.downcast_value_ptr[Vector2D]()
    var vec2 = vec2_obj.downcast_value_ptr[Vector2D]()
    return PythonObject(vec1[].dot(vec2[]))
```

## Advanced Patterns

### Handling Multiple Arguments

For functions with more than 3 arguments, use variadic functions:

```mojo
@export
fn PyInit_advanced() -> PythonObject:
    try:
        var b = PythonModuleBuilder("advanced")
        b.def_py_function[sum_many]("sum_many")
        b.def_py_function[concat_strings]("concat_strings")
        return b.finalize()
    except e:
        return abort[PythonObject](str(e))

fn sum_many(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Sum any number of numeric arguments."""
    var total = 0.0
    for i in range(len(args)):
        total += Float64(args[i])
    return PythonObject(total)

fn concat_strings(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Concatenate any number of strings."""
    if len(args) == 0:
        return PythonObject("")
    
    var result = String(args[0])
    for i in range(1, len(args)):
        result += String(args[i])
    return PythonObject(result)
```

### Keyword Arguments Pattern

Since native keyword arguments aren't supported, use a dict pattern:

Python wrapper (`wrapper.py`):
```python
import mojo_module

def process_data(data, *, normalize=True, scale=1.0, offset=0.0):
    """Python wrapper that handles keyword arguments."""
    return mojo_module._process_data(data, {
        "normalize": normalize,
        "scale": scale,
        "offset": offset
    })
```

Mojo implementation:
```mojo
fn _process_data(data_obj: PythonObject, kwargs: PythonObject) raises -> PythonObject:
    var data = data_obj  # Process as needed
    
    # Extract keyword arguments with defaults
    var normalize = Bool(kwargs.get("normalize", PythonObject(True)))
    var scale = Float64(kwargs.get("scale", PythonObject(1.0)))
    var offset = Float64(kwargs.get("offset", PythonObject(0.0)))
    
    # Process data...
    return data
```

## Building Extension Modules

### Manual Build

Instead of using the auto-import, you can build modules manually:

```bash
# Build as shared library
mojo build mojo_math.mojo --emit shared-lib -o mojo_math.so

# Use directly in Python
python -c "import mojo_math; print(mojo_math.factorial(5))"
```

### Build Script

Create `build.py`:

```python
import subprocess
import sys
import os

def build_mojo_modules():
    modules = ["mojo_math.mojo", "vectors.mojo"]
    
    for module in modules:
        if os.path.exists(module):
            output = module.replace(".mojo", ".so")
            cmd = ["mojo", "build", module, "--emit", "shared-lib", "-o", output]
            
            print(f"Building {module}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error building {module}:")
                print(result.stderr)
                sys.exit(1)
            else:
                print(f"Successfully built {output}")

if __name__ == "__main__":
    build_mojo_modules()
```

## Performance Tips

### 1. Minimize Conversions

```mojo
# Inefficient: Multiple conversions
fn process_list_slow(py_list: PythonObject) raises -> PythonObject:
    var result = PythonObject([])
    for i in range(len(py_list)):
        var item = Int(py_list[i])  # Conversion on each iteration
        result.append(PythonObject(item * 2))
    return result

# Efficient: Batch conversion
fn process_list_fast(py_list: PythonObject) raises -> PythonObject:
    var size = len(py_list)
    var mojo_list = List[Int](capacity=size)
    
    # Convert once
    for i in range(size):
        mojo_list.append(Int(py_list[i]))
    
    # Process in Mojo
    for i in range(size):
        mojo_list[i] *= 2
    
    # Convert back once
    var result = PythonObject([])
    for item in mojo_list:
        result.append(PythonObject(item[]))
    return result
```

### 2. Use SIMD for Numerical Operations

```mojo
fn compute_distances(points1: PythonObject, points2: PythonObject) raises -> PythonObject:
    """Compute pairwise distances between two sets of 2D points."""
    var n1 = len(points1)
    var n2 = len(points2)
    
    # Pre-convert to Mojo arrays for SIMD
    # ... conversion code ...
    
    # Use SIMD for distance computation
    # ... SIMD implementation ...
    
    return result
```

## Debugging

### Enable Debug Output

```mojo
fn debug_function(obj: PythonObject) raises -> PythonObject:
    print("Type:", Python.type(obj))
    print("Value:", obj)
    print("Attributes:", Python.dir(obj))
    
    # Safe attribute access
    if Python.hasattr(obj, "shape"):
        print("Shape:", obj.shape)
    
    return obj
```

### Error Handling

```mojo
fn safe_operation(obj: PythonObject) raises -> PythonObject:
    try:
        # Attempt conversion
        var value = Int(obj)
        return PythonObject(value * 2)
    except:
        # Provide helpful error message
        raise Error("Expected integer, got " + str(Python.type(obj)))
```

## Current Limitations

1. **Maximum 3 arguments** for regular functions (use variadic for more)
2. **No native keyword arguments** (use dict pattern)
3. **No direct `__init__` binding** (use factory functions)
4. **No static methods** (use module-level functions)
5. **No properties** (use getter/setter functions)
6. **Limited automatic conversions** (implement manually)

## Best Practices

1. **Start Small**: Convert one function at a time
2. **Profile First**: Identify actual bottlenecks
3. **Minimize Boundary Crossing**: Batch operations
4. **Document Types**: Clear docstrings for Python users
5. **Provide Python Wrappers**: For better ergonomics
6. **Test Thoroughly**: Both Mojo and Python sides

## Next Steps

- Explore [MAX Inference](../max-inference/overview.md) for model serving
- Learn about [Performance Optimization](../examples/python-integration.md)
- Check [Community Examples](https://github.com/modular/modular-community)
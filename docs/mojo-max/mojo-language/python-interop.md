# Python Interoperability

Mojo provides seamless integration with Python, allowing you to use existing Python libraries and gradually migrate performance-critical code to Mojo.

## Importing Python Modules

Use the `Python` module to import any Python library:

```mojo
from python import Python

def main():
    # Import Python modules
    var np = Python.import_module("numpy")
    var pd = Python.import_module("pandas")
    var torch = Python.import_module("torch")
    
    # Use them like in Python
    var arr = np.arange(15).reshape(3, 5)
    print(arr)
    print(arr.shape)
```

## Working with PythonObject

All Python values in Mojo are represented as `PythonObject`:

```mojo
from python import Python, PythonObject

def python_operations():
    var py = Python.import_module("builtins")
    
    # Create Python objects
    var py_list = PythonObject([1, 2, 3, 4, 5])
    var py_dict = PythonObject({"name": "Alice", "age": 30})
    
    # Access attributes
    print(py_dict["name"])
    
    # Call methods
    var doubled = py_list.__mul__(2)
    print(doubled)  # [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    
    # Use Python built-ins
    var sum_result = py.sum(py_list)
    print(sum_result)  # 15
```

## Type Conversions

### Python to Mojo

Convert Python objects to Mojo types:

```mojo
def convert_from_python():
    var py_int = PythonObject(42)
    var py_float = PythonObject(3.14)
    var py_str = PythonObject("Hello")
    var py_bool = PythonObject(True)
    var py_list = PythonObject([1, 2, 3])
    
    # Convert to Mojo types
    var mojo_int = Int(py_int)
    var mojo_float = Float64(py_float)
    var mojo_str = String(py_str)
    var mojo_bool = Bool(py_bool)
    
    # Convert list elements
    var mojo_list = List[Int]()
    for i in range(len(py_list)):
        mojo_list.append(Int(py_list[i]))
```

### Mojo to Python

Convert Mojo values to Python objects:

```mojo
def convert_to_python():
    # Automatic conversion for basic types
    var py_obj1 = PythonObject(42)
    var py_obj2 = PythonObject(3.14)
    var py_obj3 = PythonObject("Hello Mojo")
    var py_obj4 = PythonObject(True)
    
    # Convert collections
    var mojo_list = List[Int](1, 2, 3, 4, 5)
    var py_list = PythonObject([])
    for item in mojo_list:
        py_list.append(PythonObject(item[]))
```

## Common Python Libraries

### NumPy Integration

```mojo
def numpy_example():
    var np = Python.import_module("numpy")
    
    # Create arrays
    var arr1 = np.array([1, 2, 3, 4, 5])
    var arr2 = np.linspace(0, 1, 10)
    var arr3 = np.random.randn(3, 3)
    
    # Operations
    var result = np.dot(arr3, arr3.T)
    var mean = np.mean(arr1)
    var std = np.std(arr1)
    
    print("Mean:", mean, "Std:", std)
    
    # Convert to Mojo for performance
    var size = Int(arr1.size)
    var sum = 0.0
    for i in range(size):
        sum += Float64(arr1[i])
    print("Mojo sum:", sum)
```

### Pandas Integration

```mojo
def pandas_example():
    var pd = Python.import_module("pandas")
    
    # Create DataFrame
    var data = PythonObject({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["NYC", "LA", "Chicago"]
    })
    var df = pd.DataFrame(data)
    
    # Operations
    print(df.head())
    print(df.describe())
    
    # Filter
    var adults = df[df["age"] > 25]
    print(adults)
```

### Matplotlib Integration

```mojo
def plotting_example():
    var plt = Python.import_module("matplotlib.pyplot")
    var np = Python.import_module("numpy")
    
    # Generate data
    var x = np.linspace(0, 10, 100)
    var y = np.sin(x)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sine Wave")
    plt.legend()
    plt.grid(True)
    plt.savefig("sine_wave.png")
    plt.close()
```

## Working with Python Classes

```mojo
def python_class_example():
    var py = Python.import_module("builtins")
    
    # Define a Python class using exec
    var code = """
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hello, I'm {self.name}"
    
    def birthday(self):
        self.age += 1
"""
    
    var globals = PythonObject({})
    py.exec(code, globals)
    
    # Use the class
    var Person = globals["Person"]
    var alice = Person("Alice", 30)
    
    print(alice.greet())
    alice.birthday()
    print("Age after birthday:", alice.age)
```

## Exception Handling

Handle Python exceptions in Mojo:

```mojo
def handle_python_errors():
    var py = Python.import_module("builtins")
    
    try:
        # This will raise a Python exception
        var result = py.int("not a number")
    except e:
        print("Python error:", e)
    
    # Check for None
    var maybe_none = PythonObject(None)
    if maybe_none.is_none():
        print("Got None value")
```

## Performance Considerations

### When to Use Python vs Mojo

```mojo
def performance_comparison():
    var np = Python.import_module("numpy")
    var time = Python.import_module("time")
    
    # Python/NumPy version
    var start = time.time()
    var py_arr = np.arange(1000000)
    var py_sum = np.sum(py_arr * py_arr)
    var py_time = time.time() - start
    
    # Mojo version
    start = time.time()
    var mojo_sum = 0
    for i in range(1000000):
        mojo_sum += i * i
    var mojo_time = time.time() - start
    
    print("Python/NumPy time:", py_time)
    print("Mojo time:", mojo_time)
    print("Speedup:", Float64(py_time) / Float64(mojo_time), "x")
```

### Best Practices

1. **Use Python for**: 
   - Existing libraries (pandas, scikit-learn, etc.)
   - Prototyping and experimentation
   - I/O operations and file handling

2. **Use Mojo for**:
   - Performance-critical loops
   - Numerical computations
   - Low-level operations

3. **Optimization Strategy**:
   - Start with Python
   - Profile to find bottlenecks
   - Rewrite hot paths in Mojo
   - Keep Python for everything else

## Advanced Patterns

### Python Context Managers

```mojo
def use_context_manager():
    var py = Python.import_module("builtins")
    
    # Using with statement equivalent
    var file = py.open("example.txt", "w")
    try:
        file.write("Hello from Mojo!")
    finally:
        file.close()
```

### Async Python Code

```mojo
def async_example():
    var asyncio = Python.import_module("asyncio")
    var aiohttp = Python.import_module("aiohttp")
    
    var code = """
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

result = asyncio.run(fetch_data('https://api.example.com/data'))
"""
    
    var globals = PythonObject({})
    globals["aiohttp"] = aiohttp
    globals["asyncio"] = asyncio
    
    Python.import_module("builtins").exec(code, globals)
    print(globals["result"])
```

## Debugging Tips

1. **Print PythonObject type**: 
   ```mojo
   print(type(py_obj))
   ```

2. **Check attributes**:
   ```mojo
   print(dir(py_obj))
   ```

3. **Handle missing attributes**:
   ```mojo
   if hasattr(py_obj, "some_attr"):
       var value = py_obj.some_attr
   ```

## Next Steps

- Learn [Calling Mojo from Python](calling-mojo-from-python.md)
- Explore [Performance Optimization](../examples/python-integration.md)
- See [Real-world Examples](../examples/basic-examples.md)
# Mojo Language Basics

Mojo is a Python-compatible systems programming language that combines Python's ease of use with the performance of C++ and Rust. This guide covers the essential Mojo syntax and concepts.

## Hello World

Every Mojo program needs a `main()` function as the entry point:

```mojo
def main():
    print("Hello, world!")
```

## Variables

Mojo supports both implicit and explicit variable declarations:

```mojo
def main():
    # Implicit declaration
    x = 10
    y = x * x
    
    # Explicit declaration with var
    var z: Int = 100
    var name: String = "Mojo"
    
    # Type is inferred if not specified
    var result = x + z  # Int type inferred
```

Variables in Mojo are statically typed - once a type is assigned, it cannot change:

```mojo
x = 10
x = "Hello"  # Error: Cannot convert "StringLiteral" to "Int"
```

## Functions

Mojo supports two function declaration styles:

### def Functions (Python-style)
```mojo
def greet(name: String) -> String:
    return "Hello, " + name + "!"

def add(x: Int, y: Int) -> Int:
    return x + y
```

### fn Functions (Strict mode)
```mojo
fn multiply(x: Int, y: Int) -> Int:
    return x * y

fn process(data: String) -> None:
    print("Processing:", data)
```

Key differences:
- `fn` functions have stricter error handling
- `fn` functions require explicit type annotations
- `fn` functions have immutable arguments by default

## Control Flow

### Conditionals
```mojo
def check_value(x: Int):
    if x > 0:
        print("Positive")
    elif x < 0:
        print("Negative")
    else:
        print("Zero")
```

### Loops
```mojo
def loop_examples():
    # For loop
    for i in range(5):
        print(i)
    
    # While loop
    var count = 0
    while count < 3:
        print("Count:", count)
        count += 1
```

## Structs

Structs are Mojo's primary way to define custom types:

```mojo
struct Point:
    var x: Float64
    var y: Float64
    
    fn __init__(out self, x: Float64, y: Float64):
        self.x = x
        self.y = y
    
    fn distance_to(self, other: Point) -> Float64:
        var dx = self.x - other.x
        var dy = self.y - other.y
        return (dx * dx + dy * dy).sqrt()

def use_point():
    var p1 = Point(0.0, 0.0)
    var p2 = Point(3.0, 4.0)
    print(p1.distance_to(p2))  # Prints 5.0
```

### Simplified Structs with Decorators

```mojo
@fieldwise_init
struct Person(Copyable, Movable):
    var name: String
    var age: Int
    
    def describe(self):
        print(self.name, "is", self.age, "years old")
```

## Traits

Traits define interfaces that structs can implement:

```mojo
trait Drawable:
    fn draw(self): ...

trait Movable:
    fn move_to(mut self, x: Int, y: Int): ...

struct Shape(Drawable, Movable):
    var x: Int
    var y: Int
    
    fn draw(self):
        print("Drawing at", self.x, self.y)
    
    fn move_to(mut self, x: Int, y: Int):
        self.x = x
        self.y = y
```

## Memory Management

Mojo provides fine-grained control over memory:

```mojo
fn memory_example():
    # Stack allocation
    var x = 42
    
    # Heap allocation with Pointer
    var ptr = Pointer[Int].alloc(1)
    ptr.store(100)
    print(ptr.load())
    ptr.free()
    
    # Reference semantics
    var original = String("Hello")
    var borrowed = original  # Copies reference, not data
```

## Error Handling

Mojo uses exceptions for error handling:

```mojo
fn divide(a: Int, b: Int) raises -> Int:
    if b == 0:
        raise Error("Division by zero")
    return a // b

def safe_divide():
    try:
        var result = divide(10, 0)
        print(result)
    except e:
        print("Error:", e)
```

## Collections

### Lists
```mojo
def list_example():
    var numbers = List[Int]()
    numbers.append(1)
    numbers.append(2)
    numbers.append(3)
    
    for num in numbers:
        print(num)
```

### Dictionaries
```mojo
def dict_example():
    var ages = Dict[String, Int]()
    ages["Alice"] = 30
    ages["Bob"] = 25
    
    print(ages["Alice"])
```

## Type Parameters

Mojo supports generic programming with type parameters:

```mojo
struct Container[T: AnyType]:
    var value: T
    
    fn __init__(out self, value: T):
        self.value = value
    
    fn get(self) -> T:
        return self.value

def use_container():
    var int_container = Container[Int](42)
    var str_container = Container[String]("Hello")
    
    print(int_container.get())
    print(str_container.get())
```

## SIMD and Vectorization

Mojo has built-in support for SIMD operations:

```mojo
from math import sqrt

fn simd_example():
    # Process 4 float32 values at once
    var vec = SIMD[DType.float32, 4](1.0, 4.0, 9.0, 16.0)
    var result = sqrt(vec)
    print(result)  # [1.0, 2.0, 3.0, 4.0]
```

## Compile-Time Programming

Use parameters for compile-time computation:

```mojo
fn repeat[count: Int](msg: String):
    @parameter
    for i in range(count):
        print(msg)

def use_repeat():
    repeat[3]("Hello")  # Unrolled at compile time
```

## Best Practices

1. **Use `fn` for performance-critical code**: Better optimization
2. **Prefer `var` for clarity**: Makes code more readable
3. **Use type annotations**: Helps catch errors early
4. **Leverage SIMD**: For numerical computations
5. **Profile before optimizing**: Mojo is already fast

## Common Patterns

### Builder Pattern
```mojo
struct Config:
    var host: String
    var port: Int
    var debug: Bool
    
    fn __init__(out self):
        self.host = "localhost"
        self.port = 8080
        self.debug = False
    
    fn with_host(mut self, host: String) -> Self:
        self.host = host
        return self
    
    fn with_port(mut self, port: Int) -> Self:
        self.port = port
        return self

def use_builder():
    var config = Config().with_host("0.0.0.0").with_port(3000)
```

### Resource Management
```mojo
struct File:
    var handle: Int
    
    fn __init__(out self, path: String):
        # Open file
        self.handle = 1  # Simplified
    
    fn __del__(owned self):
        # Automatically close file
        print("Closing file")
```

## Next Steps

- Learn about [Python Interoperability](python-interop.md)
- Explore [Advanced Functions](functions.md)
- Understand [Structs and Traits](structs-and-traits.md)
- Try [Metaprogramming](metaprogramming.md)
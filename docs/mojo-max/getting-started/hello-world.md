# Hello World in Mojo

This guide walks through creating your first Mojo programs, from simple Hello World to more advanced examples.

## Your First Mojo Program

### Basic Hello World

Create a file `hello.mojo`:

```mojo
def main():
    print("Hello, World!")
```

Run it:
```bash
mojo run hello.mojo
```

Output:
```
Hello, World!
```

### Using Functions

Create `greetings.mojo`:

```mojo
fn greet(name: String) -> String:
    return "Hello, " + name + "!"

def main():
    var message = greet("Mojo")
    print(message)
    
    # Multiple greetings
    var names = ["Alice", "Bob", "Charlie"]
    for name in names:
        print(greet(name))
```

## Interactive Mojo REPL

Start the Mojo REPL:
```bash
mojo repl
```

Try some commands:
```mojo
> print("Hello from REPL!")
Hello from REPL!

> var x = 42
> var y = x * 2
> print(y)
84

> fn square(n: Int) -> Int:
.     return n * n
.
> print(square(5))
25
```

Exit with `Ctrl+D` or `exit()`.

## Command Line Arguments

Create `args.mojo`:

```mojo
from sys import argv

def main():
    print("Program name:", argv()[0])
    
    var args = argv()
    if len(args) > 1:
        print("Hello,", args[1] + "!")
    else:
        print("Hello, World!")
        print("Usage:", args[0], "<name>")
```

Run it:
```bash
mojo run args.mojo Alice
```

Output:
```
Program name: args.mojo
Hello, Alice!
```

## Working with User Input

Create `interactive.mojo`:

```mojo
from python import Python

def main():
    var builtins = Python.import_module("builtins")
    
    # Get user input
    var name = builtins.input("What's your name? ")
    print("Nice to meet you,", name + "!")
    
    # Get a number
    var age_str = builtins.input("How old are you? ")
    try:
        var age = Int(String(age_str))
        print("In 10 years, you'll be", age + 10)
    except:
        print("That's not a valid number!")
```

## File I/O Example

Create `file_io.mojo`:

```mojo
from python import Python

def main():
    var builtins = Python.import_module("builtins")
    
    # Write to file
    print("Writing to file...")
    var file = builtins.open("hello.txt", "w")
    file.write("Hello from Mojo!\n")
    file.write("This is line 2.\n")
    file.close()
    
    # Read from file
    print("\nReading from file:")
    file = builtins.open("hello.txt", "r")
    var content = file.read()
    print(content)
    file.close()
```

## Simple Calculator

Create `calculator.mojo`:

```mojo
from python import Python

fn add(a: Float64, b: Float64) -> Float64:
    return a + b

fn subtract(a: Float64, b: Float64) -> Float64:
    return a - b

fn multiply(a: Float64, b: Float64) -> Float64:
    return a * b

fn divide(a: Float64, b: Float64) raises -> Float64:
    if b == 0:
        raise Error("Division by zero!")
    return a / b

def main():
    var builtins = Python.import_module("builtins")
    
    print("Simple Calculator")
    print("-" * 20)
    
    try:
        var num1_str = builtins.input("Enter first number: ")
        var num2_str = builtins.input("Enter second number: ")
        
        var num1 = Float64(String(num1_str))
        var num2 = Float64(String(num2_str))
        
        print("\nResults:")
        print(num1, "+", num2, "=", add(num1, num2))
        print(num1, "-", num2, "=", subtract(num1, num2))
        print(num1, "*", num2, "=", multiply(num1, num2))
        
        try:
            print(num1, "/", num2, "=", divide(num1, num2))
        except e:
            print("Division error:", e)
            
    except:
        print("Invalid input! Please enter numbers.")
```

## Working with Lists

Create `lists.mojo`:

```mojo
def main():
    # Create a list
    var numbers = List[Int]()
    
    # Add elements
    for i in range(5):
        numbers.append(i * i)
    
    # Print list
    print("Squares:", end=" ")
    for num in numbers:
        print(num[], end=" ")
    print()
    
    # Sum elements
    var total = 0
    for num in numbers:
        total += num[]
    print("Sum:", total)
    
    # List operations
    print("Length:", len(numbers))
    print("First:", numbers[0])
    print("Last:", numbers[len(numbers) - 1])
```

## Temperature Converter

Create `temperature.mojo`:

```mojo
fn celsius_to_fahrenheit(celsius: Float64) -> Float64:
    return (celsius * 9.0 / 5.0) + 32.0

fn fahrenheit_to_celsius(fahrenheit: Float64) -> Float64:
    return (fahrenheit - 32.0) * 5.0 / 9.0

fn celsius_to_kelvin(celsius: Float64) -> Float64:
    return celsius + 273.15

def main():
    print("Temperature Converter")
    print("=" * 25)
    
    # Example conversions
    var temp_c = 25.0
    print("\n", temp_c, "°C is:")
    print("  ", celsius_to_fahrenheit(temp_c), "°F")
    print("  ", celsius_to_kelvin(temp_c), "K")
    
    var temp_f = 77.0
    print("\n", temp_f, "°F is:")
    print("  ", fahrenheit_to_celsius(temp_f), "°C")
    
    # Temperature table
    print("\nConversion Table (C -> F):")
    print("C    F")
    print("-" * 10)
    for c in range(-10, 41, 10):
        var f = celsius_to_fahrenheit(Float64(c))
        print(c, "  ", f)
```

## FizzBuzz

Create `fizzbuzz.mojo`:

```mojo
def main():
    print("FizzBuzz from 1 to 30:")
    print("-" * 20)
    
    for i in range(1, 31):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)
```

## Prime Number Checker

Create `primes.mojo`:

```mojo
fn is_prime(n: Int) -> Bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    var i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True

def main():
    print("Prime numbers between 1 and 100:")
    
    var count = 0
    for n in range(2, 101):
        if is_prime(n):
            print(n, end=" ")
            count += 1
            if count % 10 == 0:
                print()  # New line every 10 primes
    
    print("\n\nTotal primes found:", count)
```

## Building Standalone Executables

You can compile Mojo programs to standalone executables:

```bash
# Build executable
mojo build hello.mojo -o hello

# Run executable
./hello
```

For optimized builds:
```bash
mojo build -O3 calculator.mojo -o calculator
```

## Next Steps

Now that you've written your first Mojo programs:

1. **Learn the Language**: Continue with [Mojo Basics](../mojo-language/basics.md)
2. **Try Python Integration**: See [Python Interop](../mojo-language/python-interop.md)
3. **Build Something**: Create a small project using Mojo
4. **Explore Performance**: Try optimizing numeric code
5. **Join the Community**: Share your projects on the [forum](https://forum.modular.com)

## Tips for Beginners

1. **Start Simple**: Don't try to optimize everything at first
2. **Use Python**: Import Python libraries when needed
3. **Read Errors**: Mojo's error messages are helpful
4. **Experiment**: Try different approaches in the REPL
5. **Ask Questions**: The community is friendly and helpful

Happy coding with Mojo! 🔥
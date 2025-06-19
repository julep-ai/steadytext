# CLI Tools & Tooling

Build deterministic command-line tools and development utilities with SteadyText.

## Why SteadyText for CLI Tools?

Traditional AI-powered CLI tools have problems:

- **Inconsistent outputs**: Same command gives different results
- **Unreliable automation**: Scripts break due to changing responses  
- **Hard to test**: Non-deterministic behavior makes testing difficult
- **User confusion**: Users expect consistent behavior from tools

SteadyText solves these with **deterministic outputs** - same input always produces the same result.

## Basic CLI Patterns

### Simple Command Tools

```python
import click
import steadytext

@click.command()
@click.argument('topic')
def motivate(topic):
    """Generate motivational quotes about any topic."""
    prompt = f"Write an inspiring quote about {topic}"
    quote = steadytext.generate(prompt)
    click.echo(f"💪 {quote}")

# Usage: python script.py programming
# Always generates the same quote for "programming"
```

### Error Code Explainer

```python
@click.command()
@click.argument('error_code')
def explain(error_code):
    """Convert error codes to friendly explanations."""
    prompt = f"Explain error {error_code} in simple, user-friendly terms"
    explanation = steadytext.generate(prompt)
    click.echo(f"🔍 {error_code}: {explanation}")

# Usage: python script.py ECONNREFUSED
# Always gives the same explanation for ECONNREFUSED
```

### Command Generator

```python
@click.command()
@click.argument('task')
def git_helper(task):
    """Generate git commands for common tasks."""
    prompt = f"Git command to {task}. Return only the command."
    command = steadytext.generate(prompt).strip()
    click.echo(f"💻 {command}")

# Usage: python script.py "undo last commit"
# Always suggests the same git command
```

## Development Tools

### Code Generation Helper

```python
import os
import click

@click.group()
def codegen():
    """Code generation CLI tool."""
    pass

@codegen.command()
@click.argument('function_name')
@click.argument('description')
@click.option('--language', '-l', default='python', help='Programming language')
def function(function_name, description, language):
    """Generate a function from description."""
    prompt = f"Write a {language} function named {function_name} that {description}"
    code = steadytext.generate(prompt)
    
    # Save to file
    ext = {'python': 'py', 'javascript': 'js', 'rust': 'rs'}.get(language, 'txt')
    filename = f"{function_name}.{ext}"
    
    with open(filename, 'w') as f:
        f.write(code)
    
    click.echo(f"✅ Generated {filename}")
    click.echo(f"📄 Preview:\n{code[:200]}...")

# Usage: python codegen.py function binary_search "search for item in sorted array"
```

### Documentation Generator

```python
@codegen.command()
@click.argument('project_name')
def readme(project_name):
    """Generate README.md for a project."""
    prompt = f"Write a comprehensive README.md for a project called {project_name}"
    readme_content = steadytext.generate(prompt)
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    click.echo("✅ Generated README.md")

# Usage: python codegen.py readme "my-awesome-project"
```

## Testing & QA Tools

### Test Case Generator

```python
@click.command()
@click.argument('function_description')
def test_cases(function_description):
    """Generate test cases for a function."""
    prompt = f"Generate 5 test cases for a function that {function_description}"
    cases = steadytext.generate(prompt)
    
    # Save to test file
    with open('test_cases.py', 'w') as f:
        f.write(f"# Test cases for: {function_description}\n")
        f.write(cases)
    
    click.echo("✅ Generated test_cases.py")
    click.echo(f"📋 Preview:\n{cases[:300]}...")

# Usage: python tool.py "calculates fibonacci numbers"
```

### Mock Data Generator

```python
@click.command()
@click.argument('data_type')
@click.option('--count', '-c', default=10, help='Number of items to generate')
def mockdata(data_type, count):
    """Generate mock data for testing."""
    items = []
    
    for i in range(count):
        prompt = f"Generate realistic {data_type} data item {i+1}"
        item = steadytext.generate(prompt)
        items.append(item.strip())
    
    # Output as JSON
    import json
    output = {data_type: items}
    
    with open(f'mock_{data_type}.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    click.echo(f"✅ Generated mock_{data_type}.json with {count} items")

# Usage: python tool.py user_profiles --count 20
```

## Content & Documentation Tools

### Commit Message Generator

```python
@click.command()
@click.argument('changes', nargs=-1)
def commit_msg(changes):
    """Generate commit messages from change descriptions."""
    change_list = " ".join(changes)
    prompt = f"Write a concise git commit message for: {change_list}"
    message = steadytext.generate(prompt).strip()
    
    click.echo(f"📝 Suggested commit message:")
    click.echo(f"   {message}")
    
    # Optionally copy to clipboard or commit directly
    if click.confirm("Use this commit message?"):
        os.system(f'git commit -m "{message}"')
        click.echo("✅ Committed!")

# Usage: python tool.py "added user authentication" "fixed login bug"
```

### API Documentation Generator

```python
@click.command()
@click.argument('api_endpoint')
@click.argument('description')
def api_docs(api_endpoint, description):
    """Generate API documentation for an endpoint."""
    prompt = f"""Generate API documentation for endpoint {api_endpoint} that {description}.
    Include: description, parameters, example request/response, error codes."""
    
    docs = steadytext.generate(prompt)
    
    # Save to markdown file
    safe_name = api_endpoint.replace('/', '_').replace('{', '').replace('}', '')
    filename = f"api_{safe_name}.md"
    
    with open(filename, 'w') as f:
        f.write(f"# {api_endpoint}\n\n")
        f.write(docs)
    
    click.echo(f"✅ Generated {filename}")

# Usage: python tool.py "/users/{id}" "returns user profile information"
```

## Automation & Scripting

### Configuration Generator

```python
@click.command()
@click.argument('service_name')
@click.option('--format', '-f', default='yaml', help='Config format (yaml, json, toml)')
def config(service_name, format):
    """Generate configuration files for services."""
    prompt = f"Generate a {format} configuration file for {service_name} service"
    config_content = steadytext.generate(prompt)
    
    ext = {'yaml': 'yml', 'json': 'json', 'toml': 'toml'}.get(format, 'txt')
    filename = f"{service_name}.{ext}"
    
    with open(filename, 'w') as f:
        f.write(config_content)
    
    click.echo(f"✅ Generated {filename}")

# Usage: python tool.py database --format yaml
```

### Script Template Generator

```python
@click.command()
@click.argument('script_type')
@click.argument('purpose')
def script_template(script_type, purpose):
    """Generate script templates for common tasks."""
    prompt = f"Generate a {script_type} script template for {purpose}"
    script = steadytext.generate(prompt)
    
    ext = {'bash': 'sh', 'python': 'py', 'powershell': 'ps1'}.get(script_type, 'txt')
    filename = f"template.{ext}"
    
    with open(filename, 'w') as f:
        f.write(script)
    
    # Make executable if shell script
    if ext == 'sh':
        os.chmod(filename, 0o755)
    
    click.echo(f"✅ Generated {filename}")

# Usage: python tool.py bash "automated deployment"
```

## Complete CLI Tool Example

```python
#!/usr/bin/env python3
"""
DevHelper - A deterministic development tool powered by SteadyText
"""

import os
import json
import click
import steadytext

@click.group()
@click.version_option()
def cli():
    """DevHelper - Deterministic development utilities."""
    pass

@cli.group()
def generate():
    """Code and content generation commands."""
    pass

@generate.command()
@click.argument('name')
@click.argument('description')
@click.option('--lang', '-l', default='python', help='Programming language')
def function(name, description, lang):
    """Generate a function from description."""
    prompt = f"Write a {lang} function named {name} that {description}"
    code = steadytext.generate(prompt)
    
    ext = {'python': 'py', 'javascript': 'js', 'rust': 'rs'}.get(lang, 'txt')
    filename = f"{name}.{ext}"
    
    with open(filename, 'w') as f:
        f.write(code)
    
    click.echo(f"✅ Generated {filename}")

@generate.command()
@click.argument('count', type=int)
@click.option('--type', '-t', default='user', help='Data type to generate')
def testdata(count, type):
    """Generate test data."""
    data = []
    
    for i in range(count):
        prompt = f"Generate realistic {type} test data item {i+1} as JSON"
        item = steadytext.generate(prompt)
        data.append(item.strip())
    
    output_file = f"test_{type}_data.json"
    with open(output_file, 'w') as f:
        json.dump({f"{type}_data": data}, f, indent=2)
    
    click.echo(f"✅ Generated {output_file} with {count} items")

@cli.command()
@click.argument('error_code')
def explain(error_code):
    """Explain error codes in friendly terms."""
    prompt = f"Explain error {error_code} in simple, user-friendly terms"
    explanation = steadytext.generate(prompt)
    click.echo(f"🔍 {error_code}:")
    click.echo(f"   {explanation}")

@cli.command()
@click.argument('task')
def git(task):
    """Generate git commands for tasks."""
    prompt = f"Git command to {task}. Return only the command."
    command = steadytext.generate(prompt).strip()
    click.echo(f"💻 {command}")
    
    if click.confirm("Execute this command?"):
        os.system(command)

if __name__ == '__main__':
    cli()
```

Save this as `devhelper.py` and use it:

```bash
# Generate a function
python devhelper.py generate function binary_search "search sorted array"

# Generate test data  
python devhelper.py generate testdata 10 --type user

# Explain error codes
python devhelper.py explain ECONNREFUSED

# Get git commands
python devhelper.py git "undo last commit but keep changes"
```

## Best Practices

!!! tip "CLI Tool Guidelines"
    1. **Keep prompts specific**: Clear, detailed prompts give better results
    2. **Add confirmation prompts**: For destructive operations, ask before executing
    3. **Save outputs to files**: Generate content to files for later use
    4. **Use consistent formatting**: Same input should always produce same output
    5. **Add help text**: Use Click's built-in help system

!!! success "Benefits of Deterministic CLI Tools"
    - **Reliable automation**: Scripts work consistently
    - **Easier testing**: Predictable outputs make testing simple
    - **User trust**: Users know what to expect
    - **Debugging**: Reproducible behavior makes issues easier to track
    - **Documentation**: Examples in docs always work

!!! warning "Considerations"
    - **Creative vs. Deterministic**: SteadyText prioritizes consistency over creativity
    - **Context limits**: Model has limited context window
    - **Update impacts**: SteadyText updates may change outputs (major versions only)

This approach creates reliable, testable CLI tools that users can depend on for consistent behavior.
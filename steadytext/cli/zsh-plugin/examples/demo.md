# SteadyText ZSH Plugin Demo

This document shows example interactions with the SteadyText ZSH plugins.

## Context-Aware Suggestions

### Example 1: Git Repository Context

```bash
$ pwd
/home/user/myproject

$ git status
On branch feature/new-api
Changes not staged for commit:
  modified:   src/api.py
  modified:   tests/test_api.py

$ # Press Ctrl-X Ctrl-S
$ git add src/api.py tests/test_api.py && git commit -m "Update API implementation"
```

The AI suggests the git add and commit command based on:
- Current directory is a git repo
- There are unstaged changes
- The files changed are related (api.py and its tests)

### Example 2: Python Virtual Environment

```bash
$ pwd
/home/user/django-app

$ ls
manage.py  requirements.txt  myapp/  venv/

$ # Press Ctrl-X Ctrl-S
$ source venv/bin/activate && python manage.py runserver
```

The AI recognizes:
- Django project structure (manage.py present)
- Virtual environment exists
- Likely next action is to activate venv and run server

### Example 3: Failed Command Recovery

```bash
$ python script.py --input data.csv
Traceback (most recent call last):
  File "script.py", line 10, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'

$ # Press Ctrl-X Ctrl-S
$ pip install pandas
```

The AI understands:
- Previous command failed with ModuleNotFoundError
- The missing module is 'pandas'
- Solution is to install it with pip

## Autosuggestions (As You Type)

### Example 1: Directory Navigation

```bash
$ cd ~/proj[TAB or →]
$ cd ~/projects/steadytext/  # Suggested completion
```

### Example 2: Docker Commands

```bash
$ docker ps -a | grep steady[TAB or →]
$ docker ps -a | grep steadytext  # Completes based on context
```

### Example 3: Project-Specific Commands

When in a Node.js project:
```bash
$ npm [TAB or →]
$ npm run dev  # Suggests common npm scripts
```

When in a Python project with Makefile:
```bash
$ make [TAB or →]
$ make test  # Suggests makefile targets
```

## Advanced Features

### Project Context File

Create `.steadytext-context` in your project:

```bash
$ cat .steadytext-context
This is a FastAPI project.
Common commands:
- uvicorn main:app --reload
- pytest tests/
- black . && isort .
```

Now suggestions will be project-aware:

```bash
$ # In the project directory, press Ctrl-X Ctrl-S
$ uvicorn main:app --reload
```

### Custom Context Patterns

The plugin recognizes patterns like:

1. **Error Recovery**: After a command fails, suggests fixes
2. **Workflow Continuation**: After `git add`, suggests `git commit`
3. **Environment Awareness**: Activates virtual envs before Python commands
4. **Build Tools**: Recognizes Makefile, package.json, Cargo.toml, etc.

### Performance Considerations

- First suggestion might take 100-200ms (model loading)
- Subsequent suggestions are faster due to caching
- Async mode prevents blocking your typing
- Cache persists across shell sessions

## Configuration Examples

### Fast Mode (Optimized for Speed)
```bash
export STEADYTEXT_SUGGEST_MODEL_SIZE="small"
export STEADYTEXT_SUGGEST_STRATEGY="mixed"  # Use history first
export STEADYTEXT_SUGGEST_ASYNC=1
export STEADYTEXT_SUGGEST_CACHE_SIZE=200
```

### Accurate Mode (Better Suggestions)
```bash
export STEADYTEXT_SUGGEST_MODEL_SIZE="large"
export STEADYTEXT_SUGGEST_STRATEGY="context"  # Always use AI
export STEADYTEXT_SUGGEST_MAX_CONTEXT=1000
```

### Minimal Mode (Privacy-Conscious)
```bash
export STEADYTEXT_SUGGEST_STRATEGY="history"  # No AI, just history
# Or disable completely:
export STEADYTEXT_SUGGEST_ENABLED=0
```
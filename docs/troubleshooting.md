# Troubleshooting Guide

Common issues and solutions for SteadyText.

## Installation Issues

### Model Download Problems

**Issue**: Models fail to download automatically

**Solutions**:
1. Enable model downloads:
   ```bash
   export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true
   ```

2. Check internet connection and firewall settings

3. Manually download models:
   ```bash
   st models preload --size small
   ```

4. Clear model cache and retry:
   ```bash
   st cache clear
   rm -rf ~/.cache/steadytext/models/
   ```

### Permission Errors

**Issue**: Permission denied errors when installing or running

**Solutions**:
1. Install in user directory:
   ```bash
   pip install --user steadytext
   ```

2. Use virtual environment:
   ```bash
   python -m venv steadytext-env
   source steadytext-env/bin/activate  # Linux/macOS
   # or
   steadytext-env\Scripts\activate  # Windows
   pip install steadytext
   ```

3. Fix cache directory permissions:
   ```bash
   sudo chown -R $USER ~/.cache/steadytext
   ```

## Runtime Issues

### Memory Problems

**Issue**: Out of memory errors or high memory usage

**Solutions**:
1. Reduce cache sizes:
   ```bash
   export STEADYTEXT_GENERATION_CACHE_CAPACITY=64
   export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=10.0
   export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=128
   export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=20.0
   ```

2. Use smaller models:
   ```python
   import steadytext
   text = steadytext.generate("prompt", size="small")
   ```

3. Disable daemon mode:
   ```bash
   export STEADYTEXT_DISABLE_DAEMON=true
   ```

4. Use memory cache backend:
   ```bash
   export STEADYTEXT_CACHE_BACKEND=memory
   ```

### Context Length Exceeded

**Issue**: `ContextLengthExceededError` when generating text

**Solutions**:
1. Reduce input length or split into smaller chunks

2. Increase context window:
   ```bash
   export STEADYTEXT_MAX_CONTEXT_WINDOW=8192
   ```

3. Use streaming generation for long outputs:
   ```python
   import steadytext
   for chunk in steadytext.generate_iter("long prompt"):
       print(chunk, end="", flush=True)
   ```

### Slow Performance

**Issue**: Generation or embedding is slow

**Solutions**:
1. Enable daemon mode (if not already enabled):
   ```bash
   st daemon start
   ```

2. Increase cache capacity:
   ```bash
   export STEADYTEXT_GENERATION_CACHE_CAPACITY=512
   export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=1024
   ```

3. Use smaller models for better speed:
   ```python
   text = steadytext.generate("prompt", size="small")
   ```

4. Preload models:
   ```bash
   st models preload --size small
   ```

## Daemon Issues

### Connection Problems

**Issue**: Cannot connect to daemon

**Solutions**:
1. Check if daemon is running:
   ```bash
   st daemon status
   ```

2. Start daemon:
   ```bash
   st daemon start
   ```

3. Check daemon configuration:
   ```bash
   export STEADYTEXT_DAEMON_HOST=localhost
   export STEADYTEXT_DAEMON_PORT=5557
   ```

4. Restart daemon:
   ```bash
   st daemon restart
   ```

### Daemon Crashes

**Issue**: Daemon process terminates unexpectedly

**Solutions**:
1. Check daemon logs:
   ```bash
   st daemon status --json
   ```

2. Start daemon in foreground for debugging:
   ```bash
   st daemon start --foreground
   ```

3. Clear daemon state:
   ```bash
   st daemon stop --force
   st cache clear
   st daemon start
   ```

## PostgreSQL Extension Issues

### Extension Not Loading

**Issue**: PostgreSQL extension fails to load

**Solutions**:
1. Check PostgreSQL logs for error messages

2. Ensure Python and SteadyText are properly installed:
   ```bash
   sudo -u postgres python3 -c "import steadytext; print('OK')"
   ```

3. Check plpython3u extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS plpython3u;
   ```

4. Reinstall extension:
   ```bash
   cd pg_steadytext
   make clean
   make install
   ```

### Python Path Issues

**Issue**: Python modules not found in PostgreSQL

**Solutions**:
1. Set Python path in PostgreSQL:
   ```sql
   ALTER SYSTEM SET plpython3.python_path = '/path/to/steadytext/venv/lib/python3.x/site-packages';
   SELECT pg_reload_conf();
   ```

2. Use virtual environment path:
   ```bash
   # Find the correct path
   python3 -c "import steadytext; print(steadytext.__file__)"
   ```

### Async Worker Issues

**Issue**: Async functions not working

**Solutions**:
1. Check worker status:
   ```sql
   SELECT * FROM steadytext_queue_status();
   ```

2. Restart worker:
   ```bash
   sudo systemctl restart steadytext-worker
   ```

3. Check worker logs:
   ```bash
   sudo journalctl -u steadytext-worker -f
   ```

## Cache Issues

### Cache Corruption

**Issue**: Cache returns invalid results

**Solutions**:
1. Clear all caches:
   ```bash
   st cache clear
   ```

2. Reset cache directory:
   ```bash
   rm -rf ~/.cache/steadytext/
   ```

3. Switch to memory backend temporarily:
   ```bash
   export STEADYTEXT_CACHE_BACKEND=memory
   ```

### Cache Size Problems

**Issue**: Cache files growing too large

**Solutions**:
1. Reduce cache limits:
   ```bash
   export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=25.0
   export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=50.0
   ```

2. Regular cache cleanup:
   ```bash
   st cache status  # Check current usage
   st cache clear   # Clear if needed
   ```

## Shell Integration Issues

### Completion Not Working

**Issue**: Tab completion not functioning

**Solutions**:
1. Reinstall completions:
   ```bash
   st completion --install
   ```

2. Restart shell or source configuration:
   ```bash
   source ~/.bashrc  # Bash
   source ~/.zshrc   # Zsh
   ```

3. Check completion installation:
   ```bash
   st completion --shell zsh  # Generate completion script
   ```

### ZSH Plugin Issues

**Issue**: ZSH suggestions not working

**Solutions**:
1. Check plugin configuration:
   ```bash
   echo $STEADYTEXT_SUGGEST_ENABLED
   ```

2. Enable suggestions:
   ```bash
   export STEADYTEXT_SUGGEST_ENABLED=1
   ```

3. Restart ZSH:
   ```bash
   exec zsh
   ```

## Development Issues

### Testing Problems

**Issue**: Tests fail or hang

**Solutions**:
1. Allow model downloads for tests:
   ```bash
   STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true python -m pytest
   ```

2. Use memory cache for tests:
   ```bash
   STEADYTEXT_CACHE_BACKEND=memory python -m pytest
   ```

3. Skip model-dependent tests:
   ```bash
   python -m pytest -k "not test_model"
   ```

### Import Errors

**Issue**: Cannot import steadytext modules

**Solutions**:
1. Check installation:
   ```bash
   pip list | grep steadytext
   ```

2. Reinstall in development mode:
   ```bash
   pip install -e .
   ```

3. Check Python path:
   ```bash
   python -c "import sys; print(sys.path)"
   ```

## Getting Help

If you continue to experience issues:

1. Check the [FAQ](faq.md) for common questions
2. Review the [Configuration Guide](api/configuration.md) for advanced settings
3. Open an issue on [GitHub](https://github.com/julep-ai/steadytext/issues)
4. Include error messages, system information, and configuration details

### Debug Information

To gather debug information:

```bash
# System information
uname -a
python --version
pip list | grep steadytext

# SteadyText status
st --version
st daemon status
st cache status

# Environment variables
env | grep STEADYTEXT
```

Include this information when reporting issues.
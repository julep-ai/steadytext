-- Migration from pg_steadytext 2025.10.30 to 2025.11.25
-- AIDEV-NOTE: Simplified Python layer - calls steadytext directly

\echo 'Upgrading pg_steadytext from 2025.10.30 to 2025.11.25'
\echo 'This upgrade simplifies the Python layer by removing wrapper modules'

-- Notify about changes
DO $$
BEGIN
    RAISE NOTICE 'pg_steadytext 2025.11.25: Simplified Python layer';
    RAISE NOTICE 'Removed: daemon_connector, cache_manager, config modules';
    RAISE NOTICE 'Functions now call steadytext library directly';
END $$;

-- Drop and recreate _steadytext_init_python
DROP FUNCTION IF EXISTS @extschema@._steadytext_init_python();

CREATE OR REPLACE FUNCTION _steadytext_init_python()
RETURNS void
LANGUAGE plpython3u
AS $c$
# AIDEV-NOTE: Initialize Python environment for pg_steadytext with enhanced error handling
import sys
import os
import site

# Get PostgreSQL lib directory with fallback
try:
    result = plpy.execute("SELECT setting FROM pg_settings WHERE name = 'pkglibdir'")
    if result and len(result) > 0 and result[0]['setting']:
        pg_lib_dir = result[0]['setting']
    else:
        # Fallback for Docker/Debian PostgreSQL 17
        pg_lib_dir = '/usr/lib/postgresql/17/lib'
        plpy.notice(f"Using fallback pkglibdir: {pg_lib_dir}")
except Exception as e:
    # Fallback for Docker/Debian PostgreSQL 17
    pg_lib_dir = '/usr/lib/postgresql/17/lib'
    plpy.notice(f"Error getting pkglibdir, using fallback: {pg_lib_dir}")

python_module_dir = os.path.join(pg_lib_dir, 'pg_steadytext', 'python')

# Save pg_lib_dir in GD for use in error messages
GD['pg_lib_dir'] = pg_lib_dir

# Verify the directory exists
if not os.path.exists(python_module_dir):
    plpy.error(f"Python module directory not found: {python_module_dir}")

# Add to Python path if not already there
if python_module_dir not in sys.path:
    sys.path.insert(0, python_module_dir)
    site.addsitedir(python_module_dir)  # Process .pth files if any

# AIDEV-NOTE: Add site-packages directory for locally installed Python packages
site_packages_dir = os.path.join(pg_lib_dir, 'pg_steadytext', 'site-packages')
if os.path.exists(site_packages_dir) and site_packages_dir not in sys.path:
    sys.path.insert(0, site_packages_dir)
    site.addsitedir(site_packages_dir)
    plpy.notice(f"Added site-packages to path: {site_packages_dir}")

# AIDEV-NOTE: Add common Python package locations
# These are common locations where pip might install packages
common_paths = [
    # User site-packages
    site.getusersitepackages(),
    # System-wide site-packages
    '/usr/local/lib/python3.10/dist-packages',
    '/usr/local/lib/python3.11/dist-packages',
    '/usr/local/lib/python3.12/dist-packages',
    '/usr/lib/python3/dist-packages',
    # Virtual environment (if activated)
    os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'),
]

for path in common_paths:
    if path and os.path.exists(path) and path not in sys.path:
        sys.path.append(path)

# Log Python path for debugging
plpy.notice(f"Python path: {sys.path}")
plpy.notice(f"Looking for modules in: {python_module_dir}")

# Check if directory exists
if not os.path.exists(python_module_dir):
    plpy.error(f"Python module directory does not exist: {python_module_dir}")

# List files in directory for debugging
try:
    files = os.listdir(python_module_dir)
    plpy.notice(f"Files in module directory: {files}")
except Exception as e:
    plpy.warning(f"Could not list module directory: {e}")

# Try to import required external packages first
required_packages = {
    'steadytext': 'SteadyText library',
    'zmq': 'ZeroMQ for daemon communication',
    'numpy': 'NumPy for embeddings'
}

missing_packages = []
for package, description in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(f"{package} ({description})")

if missing_packages:
    pg_lib_dir = GD.get('pg_lib_dir', '/usr/lib/postgresql/17/lib')
    site_packages_dir = os.path.join(pg_lib_dir, 'pg_steadytext', 'site-packages')

    error_msg = f"""
=================================================================
Missing required Python packages: {', '.join(missing_packages)}

The pg_steadytext extension requires these packages to function.

To fix this, run ONE of the following commands:

1. Install via make (recommended):
   cd /path/to/pg_steadytext
   sudo make install

2. Install to PostgreSQL's site-packages:
   sudo pip3 install --target={site_packages_dir} steadytext pyzmq numpy

3. Install system-wide:
   sudo pip3 install steadytext pyzmq numpy

4. Install to user directory:
   pip3 install --user steadytext pyzmq numpy

After installation, restart PostgreSQL and try again.
=================================================================
"""
    plpy.error(error_msg)

# Try to import our modules and cache them in GD
try:
    # Clear any previous module cache
    for key in list(GD.keys()):
        if key.startswith('module_'):
            del GD[key]

    # Import and cache modules
    import security
    import prompt_registry

    # Store modules in GD for reuse
    GD['module_security'] = security
    GD['module_prompt_registry'] = prompt_registry
    GD['steadytext_initialized'] = True

    plpy.notice(f"pg_steadytext Python environment initialized successfully from {python_module_dir}")
except ImportError as e:
    GD['steadytext_initialized'] = False
    pg_lib_dir = GD.get('pg_lib_dir', '/usr/lib/postgresql/17/lib')
    site_packages_dir = os.path.join(pg_lib_dir, 'pg_steadytext', 'site-packages')

    error_msg = f"""
=================================================================
Failed to import pg_steadytext modules from {python_module_dir}

Error: {str(e)}

This usually means the extension files are installed but Python
module files are missing or there's an import error.

To fix this:

1. Ensure all Python module files are present in:
   {python_module_dir}

2. Check that required packages are installed:
   sudo pip3 install --target={site_packages_dir} steadytext pyzmq numpy

3. Or reinstall the extension:
   cd /path/to/pg_steadytext
   sudo make install

After fixing, restart PostgreSQL and try again.
=================================================================
"""
    plpy.error(error_msg)
except Exception as e:
    GD['steadytext_initialized'] = False
    plpy.error(f"Unexpected error during initialization: {e}")
$c$;

-- Drop and recreate steadytext_generate
DROP FUNCTION IF EXISTS @extschema@.steadytext_generate(TEXT, INT, BOOLEAN, INT, TEXT, TEXT, TEXT, TEXT, TEXT, BOOLEAN);

CREATE OR REPLACE FUNCTION steadytext_generate(
    prompt TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    eos_string TEXT DEFAULT '[EOS]',
    model TEXT DEFAULT NULL,
    model_repo TEXT DEFAULT NULL,
    model_filename TEXT DEFAULT NULL,
    size TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE  -- IMMUTABLE: Intentional for query optimization despite daemon interaction
AS $c$
# AIDEV-NOTE: Main text generation function that calls steadytext directly
# v2025.8.15: Added support for eos_string, model, model_repo, model_filename, size parameters
# v2025.8.15: Added model parameter for remote model access
import json
import os
import subprocess
import os
import subprocess

def raise_p0001(message: str) -> None:
    plan = GD.get('steadytext_raise_plan')
    if plan is None:
        ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
        ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
        plan = plpy.prepare(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_raise_p0001($1)", ["text"])
        GD['steadytext_raise_plan'] = plan

    plpy.execute(plan, [message])

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Get configuration
# AIDEV-NOTE: Use extension schema instead of current_schema() for cross-schema compatibility
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
config_select_plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(ext_schema)}.steadytext_config WHERE key = $1", ["text"])

# Resolve max_tokens, using the provided value or fetching the default
resolved_max_tokens = max_tokens
if resolved_max_tokens is None:
    rv = plpy.execute(config_select_plan, ["default_max_tokens"])
    resolved_max_tokens = json.loads(rv[0]["value"]) if rv else 512

# Resolve seed, using the provided value or fetching the default
resolved_seed = seed
if resolved_seed is None:
    rv = plpy.execute(config_select_plan, ["default_seed"])
    resolved_seed = json.loads(rv[0]["value"]) if rv else 42

# Resolve daemon settings from configuration (preserve backward compatibility)
host_rv = plpy.execute(config_select_plan, ["daemon_host"])
daemon_host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(config_select_plan, ["daemon_port"])
daemon_port = json.loads(port_rv[0]["value"]) if port_rv else 5555

auto_start_rv = plpy.execute(config_select_plan, ["daemon_auto_start"])
daemon_auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

# Basic null/empty checks (validation is application's responsibility)
if prompt is None:
    raise_p0001("Prompt cannot be null")
if prompt.strip() == "":
    raise_p0001("Prompt cannot be empty")
if resolved_seed < 0:
    raise_p0001("seed must be non-negative")

# Validate unsafe_mode usage (keep this security check)
if not unsafe_mode and model and ':' in model:
    raise_p0001("Remote models (containing ':' ) require unsafe_mode=TRUE")
if unsafe_mode and not model:
    raise_p0001("unsafe_mode=TRUE requires a model parameter")

# Configure daemon endpoint for the steadytext client
os.environ["STEADYTEXT_DAEMON_HOST"] = str(daemon_host)
os.environ["STEADYTEXT_DAEMON_PORT"] = str(daemon_port)

# Best-effort auto-start for local daemon when enabled and using local models
is_remote_model = unsafe_mode and model and ':' in model
if daemon_auto_start and not is_remote_model:
    try:
        from steadytext.daemon.client import DaemonClient

        client = DaemonClient(host=daemon_host, port=daemon_port)
        if not client.connect():
            # Attempt to start the daemon; ignore failures to preserve determinism
            subprocess.run(
                [
                    'st', 'daemon', 'start',
                    '--host', str(daemon_host),
                    '--port', str(daemon_port)
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10
            )
    except Exception as e:
        plpy.warning(f"Daemon auto-start skipped: {e}")

# Check if we should use cache
if use_cache:
    # Generate cache key consistent with SteadyText format
    # Include eos_string in cache key if it's not the default
    if eos_string == '[EOS]':
        cache_key = prompt
    else:
        cache_key = f"{prompt}::EOS::{eos_string}"

    # Try to get from cache first
    # AIDEV-NOTE: Get schema dynamically at runtime for TimescaleDB continuous aggregates compatibility
    cache_plan = plpy.prepare(f"""
        SELECT response
        FROM {plpy.quote_ident(ext_schema)}.steadytext_cache
        WHERE cache_key = $1
    """, ["text"])

    cache_result = plpy.execute(cache_plan, [cache_key])
    if cache_result and cache_result[0]["response"]:
        plpy.notice(f"Cache hit for key: {cache_key[:8]}...")
        cached_value = cache_result[0]["response"]
        if isinstance(cached_value, (dict, list)):
            return json.dumps(cached_value)
        return cached_value

# Cache miss - generate new content
# Call steadytext directly
from steadytext import generate

kwargs = {"seed": resolved_seed}
if eos_string and eos_string != '[EOS]':
    kwargs["eos_string"] = eos_string
if model:
    kwargs["model"] = model
if model_repo:
    kwargs["model_repo"] = model_repo
if model_filename:
    kwargs["model_filename"] = model_filename
if size:
    kwargs["size"] = size
if unsafe_mode:
    kwargs["unsafe_mode"] = unsafe_mode

result = generate(prompt, max_new_tokens=resolved_max_tokens, **kwargs)
return result
$c$;

-- Drop and recreate steadytext_generate_stream
DROP FUNCTION IF EXISTS @extschema@.steadytext_generate_stream(TEXT, INT, BOOLEAN, INT, TEXT, TEXT, TEXT, TEXT, TEXT, BOOLEAN);

CREATE OR REPLACE FUNCTION steadytext_generate_stream(
    prompt TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    eos_string TEXT DEFAULT '[EOS]',
    model TEXT DEFAULT NULL,
    model_repo TEXT DEFAULT NULL,
    model_filename TEXT DEFAULT NULL,
    size TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS SETOF TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
import json
import os
import subprocess

# Initialize Python environment if necessary
if not GD.get('steadytext_initialized', False):
    ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Resolve defaults from configuration
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'

config_plan = GD.get('steadytext_config_plan_stream')
if config_plan is None:
    config_plan = plpy.prepare(f"SELECT value FROM {plpy.quote_ident(ext_schema)}.steadytext_config WHERE key = $1", ["text"])
    GD['steadytext_config_plan_stream'] = config_plan

resolved_max_tokens = max_tokens
if resolved_max_tokens is None:
    rv = plpy.execute(config_plan, ["default_max_tokens"])
    resolved_max_tokens = json.loads(rv[0]["value"]) if rv else 512

resolved_seed = seed
if resolved_seed is None:
    rv = plpy.execute(config_plan, ["default_seed"])
    resolved_seed = json.loads(rv[0]["value"]) if rv else 42

# Resolve daemon settings to honor database configuration
host_rv = plpy.execute(config_plan, ["daemon_host"])
daemon_host = json.loads(host_rv[0]["value"]) if host_rv else "localhost"

port_rv = plpy.execute(config_plan, ["daemon_port"])
daemon_port = json.loads(port_rv[0]["value"]) if port_rv else 5555

auto_start_rv = plpy.execute(config_plan, ["daemon_auto_start"])
daemon_auto_start = json.loads(auto_start_rv[0]["value"]) if auto_start_rv else True

# Basic null/empty checks
if prompt is None:
    plpy.error("Prompt cannot be null")
if prompt.strip() == "":
    plpy.error("Prompt cannot be empty")

# Validate unsafe_mode
if not unsafe_mode and model and ':' in model:
    plpy.error("Remote models (containing ':' ) require unsafe_mode=TRUE")

# Configure daemon endpoint for steadytext client
os.environ["STEADYTEXT_DAEMON_HOST"] = str(daemon_host)
os.environ["STEADYTEXT_DAEMON_PORT"] = str(daemon_port)

# Best-effort auto-start for local daemon (streaming)
is_remote_model = unsafe_mode and model and ':' in model
if daemon_auto_start and not is_remote_model:
    try:
        from steadytext.daemon.client import DaemonClient

        client = DaemonClient(host=daemon_host, port=daemon_port)
        if not client.connect():
            subprocess.run(
                [
                    'st', 'daemon', 'start',
                    '--host', str(daemon_host),
                    '--port', str(daemon_port)
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10
            )
    except Exception as e:
        plpy.warning(f"Daemon auto-start skipped: {e}")

# Call steadytext directly with streaming
from steadytext import generate

kwargs = {"seed": resolved_seed, "stream": True}
if eos_string and eos_string != '[EOS]':
    kwargs["eos_string"] = eos_string
if model:
    kwargs["model"] = model
if model_repo:
    kwargs["model_repo"] = model_repo
if model_filename:
    kwargs["model_filename"] = model_filename
if size:
    kwargs["size"] = size
if unsafe_mode:
    kwargs["unsafe_mode"] = unsafe_mode

# Generate and yield chunks
for chunk in generate(prompt, max_new_tokens=resolved_max_tokens, **kwargs):
    yield (chunk,)
$c$;

-- Drop and recreate steadytext_embed
DROP FUNCTION IF EXISTS @extschema@.steadytext_embed(TEXT, BOOLEAN, INT, TEXT, BOOLEAN);

CREATE OR REPLACE FUNCTION steadytext_embed(
    text_input TEXT,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    model TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS vector(1024)
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE LEAKPROOF  -- IMMUTABLE: Intentional for query optimization despite model interaction
AS $c$
# AIDEV-NOTE: Embedding function with remote model support via unsafe_mode
# Added in v2025.8.15 to support OpenAI and other remote embedding providers
# Uses runtime inspection to maintain backward compatibility with older steadytext versions

import json
import numpy as np

# Shared helper to raise SQLSTATE P0001
def raise_p0001(message: str) -> None:
    security = GD.get('module_security')
    if security and hasattr(security, 'raise_sqlstate'):
        security.raise_sqlstate(message, "P0001")
        return

    plan = GD.get('steadytext_raise_plan')
    if plan is None:
        ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
        ext_schema_local = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
        plan = plpy.prepare(f"SELECT {plpy.quote_ident(ext_schema_local)}._steadytext_raise_p0001($1)", ["text"])
        GD['steadytext_raise_plan'] = plan

    plpy.execute(plan, [message])

# Get extension schema for all subsequent queries
# AIDEV-NOTE: Define ext_schema at function level to avoid UnboundLocalError
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'

# Initialize Python environment if needed
try:
    if 'steadytext_initialized' not in GD:
        plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
except:
    pass

# Basic null/empty check (validation is application's responsibility)
if text_input is None:
    raise_p0001("Text cannot be null")
if text_input.strip() == "":
    raise_p0001("Text cannot be empty")

# Validate unsafe_mode for remote models
if not unsafe_mode and model and ':' in model:
    raise_p0001("Remote models (containing ':' ) require unsafe_mode=TRUE")

# Check cache if enabled (IMMUTABLE functions only read cache)
if use_cache:
    # Generate cache key including model if specified
    cache_key_parts = ['embed', text_input]
    if model:
        cache_key_parts.append(model)
    cache_key_input = ':'.join(cache_key_parts)

    import hashlib
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()

    # Try to get from cache (read-only for IMMUTABLE)
    # AIDEV-NOTE: Use extension schema for TimescaleDB continuous aggregates compatibility
    plan = plpy.prepare(
        f"SELECT embedding FROM {plpy.quote_ident(ext_schema)}.steadytext_cache WHERE cache_key = $1",
        ["text"]
    )
    rv = plpy.execute(plan, [cache_key])

    if rv and len(rv) > 0:
        cached_embedding = rv[0]['embedding']
        if cached_embedding:
            return cached_embedding

# AIDEV-NOTE: Call steadytext.embed() directly - it handles daemon fallback internally
try:
    from steadytext import embed as steadytext_embed
    import inspect

    embed_sig = inspect.signature(steadytext_embed)

    kwargs = {'seed': seed}
    if model and 'model' in embed_sig.parameters:
        kwargs['model'] = model
    if unsafe_mode and 'unsafe_mode' in embed_sig.parameters:
        kwargs['unsafe_mode'] = unsafe_mode

    result = steadytext_embed(text_input, **kwargs)

    if result is not None:
        if hasattr(result, 'tolist'):
            embedding_list = result.tolist()
        else:
            embedding_list = list(result)
        return embedding_list
except Exception as e:
    plpy.warning(f"Embedding generation failed: {e}")
    return [0.0] * 1024

# Fallback zero vector if no other return path executed
return [0.0] * 1024
$c$;

-- Drop and recreate steadytext_generate_json
DROP FUNCTION IF EXISTS @extschema@.steadytext_generate_json(TEXT, JSONB, INT, BOOLEAN, INT, BOOLEAN, TEXT);

CREATE OR REPLACE FUNCTION steadytext_generate_json(
    prompt TEXT,
    schema JSONB,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE,
    model TEXT DEFAULT NULL
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
# AIDEV-NOTE: Generate JSON that conforms to a schema using llama.cpp grammars
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
# v2025.8.15: Added model parameter for remote model access
import json
import hashlib

def raise_p0001(message: str) -> None:
    security = GD.get('module_security')
    if security and hasattr(security, 'raise_sqlstate'):
        security.raise_sqlstate(message, "P0001")
        return

    plan = GD.get('steadytext_raise_plan')
    if plan is None:
        ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
        ext_schema_local = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
        plan = plpy.prepare(f"SELECT {plpy.quote_ident(ext_schema_local)}._steadytext_raise_p0001($1)", ["text"])
        GD['steadytext_raise_plan'] = plan

    plpy.execute(plan, [message])

def ensure_json_string(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value)

    if value is None:
        return json.dumps({})

    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except json.JSONDecodeError:
            return json.dumps({})

    return json.dumps({})

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Basic null/empty checks
if prompt is None:
    raise_p0001("Prompt cannot be null")
if prompt.strip() == "":
    raise_p0001("Prompt cannot be empty")
if schema is None:
    raise_p0001("Schema cannot be null")

# Normalize schema: accept JSON dict/list or JSON string
schema_value = schema
if isinstance(schema_value, str):
    try:
        schema_value = json.loads(schema_value)
    except json.JSONDecodeError as e:
        raise_p0001(f"Invalid JSON schema: {e}")

# Validate unsafe_mode
if not unsafe_mode and model and ':' in model:
    raise_p0001("Remote models (containing ':' ) require unsafe_mode=TRUE")

# Call steadytext directly
from steadytext import generate_json

result = generate_json(
    prompt,
    schema_value,
    max_tokens=max_tokens,  # Note: NOT max_new_tokens for structured gen!
    seed=seed,
    model=model if model else None,
    unsafe_mode=unsafe_mode
)
return result
$c$;

-- Drop and recreate steadytext_generate_regex
DROP FUNCTION IF EXISTS @extschema@.steadytext_generate_regex(TEXT, TEXT, INT, BOOLEAN, INT, BOOLEAN, TEXT);

CREATE OR REPLACE FUNCTION steadytext_generate_regex(
    prompt TEXT,
    pattern TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE,
    model TEXT DEFAULT NULL
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
# AIDEV-NOTE: Generate text matching a regex pattern using llama.cpp grammars
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
# v2025.8.15: Added model parameter for remote model access
import json
import hashlib

def raise_p0001(message: str) -> None:
    security = GD.get('module_security')
    if security and hasattr(security, 'raise_sqlstate'):
        security.raise_sqlstate(message, "P0001")
        return

    plan = GD.get('steadytext_raise_plan')
    if plan is None:
        ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
        ext_schema_local = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
        plan = plpy.prepare(f"SELECT {plpy.quote_ident(ext_schema_local)}._steadytext_raise_p0001($1)", ["text"])
        GD['steadytext_raise_plan'] = plan

    plpy.execute(plan, [message])

def ensure_text(value):
    if value is None:
        return ""
    return str(value)

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Basic null/empty checks
if prompt is None:
    raise_p0001("Prompt cannot be null")
if prompt.strip() == "":
    raise_p0001("Prompt cannot be empty")
if pattern is None or pattern.strip() == "":
    raise_p0001("Pattern cannot be null or empty")

# Validate unsafe_mode
if not unsafe_mode and model and ':' in model:
    raise_p0001("Remote models (containing ':' ) require unsafe_mode=TRUE")

# Call steadytext directly
from steadytext import generate_regex

result = generate_regex(
    prompt,
    pattern,
    max_tokens=max_tokens,  # Note: NOT max_new_tokens!
    seed=seed,
    model=model if model else None,
    unsafe_mode=unsafe_mode
)
return result
$c$;

-- Drop and recreate steadytext_generate_choice
DROP FUNCTION IF EXISTS @extschema@.steadytext_generate_choice(TEXT, TEXT[], INT, BOOLEAN, INT, BOOLEAN, TEXT);

CREATE OR REPLACE FUNCTION steadytext_generate_choice(
    prompt TEXT,
    choices TEXT[],
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT TRUE,
    seed INT DEFAULT 42,
    unsafe_mode BOOLEAN DEFAULT FALSE,
    model TEXT DEFAULT NULL
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
# AIDEV-NOTE: Generate text constrained to one of the provided choices
# Fixed in v1.4.1 to use SELECT-only cache reads for true immutability
# v2025.8.15: Added model parameter for remote model access
import json
import hashlib

def raise_p0001(message: str) -> None:
    security = GD.get('module_security')
    if security and hasattr(security, 'raise_sqlstate'):
        security.raise_sqlstate(message, "P0001")
        return

    plan = GD.get('steadytext_raise_plan')
    if plan is None:
        ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
        ext_schema_local = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
        plan = plpy.prepare(f"SELECT {plpy.quote_ident(ext_schema_local)}._steadytext_raise_p0001($1)", ["text"])
        GD['steadytext_raise_plan'] = plan

    plpy.execute(plan, [message])

def ensure_choice(value, choices):
    if value in choices:
        return value
    if choices:
        return choices[0]
    return ""

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

# Basic null/empty checks
if prompt is None:
    raise_p0001("Prompt cannot be null")
if prompt.strip() == "":
    raise_p0001("Prompt cannot be empty")
if choices is None or len(choices) == 0:
    raise_p0001("Choices cannot be null or empty")
if len(choices) < 2:
    raise_p0001("Must provide at least 2 choices")

# Validate unsafe_mode
if not unsafe_mode and model and ':' in model:
    raise_p0001("Remote models (containing ':' ) require unsafe_mode=TRUE")

# Call steadytext directly
from steadytext import generate_choice

result = generate_choice(
    prompt,
    choices,
    max_tokens=max_tokens,  # Note: NOT max_new_tokens!
    seed=seed,
    model=model if model else None,
    unsafe_mode=unsafe_mode
)
return result
$c$;

-- Drop and recreate steadytext_rerank
DROP FUNCTION IF EXISTS @extschema@.steadytext_rerank(TEXT, TEXT[], TEXT, BOOLEAN, INT);

CREATE OR REPLACE FUNCTION steadytext_rerank(
    query text,
    documents text[],
    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    return_scores boolean DEFAULT true,
    seed integer DEFAULT 42
) RETURNS TABLE(document text, score float)
AS $c$
    import json
    import logging
    from typing import List, Tuple

    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    def raise_p0001(message: str) -> None:
        security = GD.get('module_security')
        if security and hasattr(security, 'raise_sqlstate'):
            security.raise_sqlstate(message, "P0001")
            return

        plan = GD.get('steadytext_raise_plan')
        if plan is None:
            ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
            ext_schema_local = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
            plan = plpy.prepare(f"SELECT {plpy.quote_ident(ext_schema_local)}._steadytext_raise_p0001($1)", ["text"])
            GD['steadytext_raise_plan'] = plan

        plpy.execute(plan, [message])

    def deterministic_rerank(documents: List[str], query_text: str, task_text: str) -> List[Tuple[str, float]]:
        query_terms = [term for term in query_text.lower().split() if term]
        task_terms = [term for term in (task_text or '').lower().split() if term]
        results: List[Tuple[str, float]] = []

        for idx, doc in enumerate(documents):
            doc_lower = doc.lower()
            score = 0.0

            for term in query_terms:
                if term in doc_lower:
                    score += 0.35

            for term in task_terms:
                if term in doc_lower:
                    score += 0.25

            score += max(0.05, 0.1 / (idx + 1))
            score = float(min(score, 1.0))
            results.append((doc, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results

    # Validate inputs
    if query is None:
        raise_p0001("Query cannot be null")

    if isinstance(query, str) and query.strip() == "":
        raise_p0001("Query cannot be empty")

    if documents is None:
        raise_p0001("Documents cannot be null")

    documents_list = list(documents)

    if len(documents_list) == 0:
        raise_p0001("Documents array cannot be empty")

    if any(doc is None for doc in documents_list):
        raise_p0001("Documents cannot contain null elements")

    # Check if initialized, if not, initialize now
    if not GD.get('steadytext_initialized', False):
        # Initialize on demand
        ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
        ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
        plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
        # Check again after initialization
        if not GD.get('steadytext_initialized', False):
            plpy.error("Failed to initialize pg_steadytext Python environment")

    try:
        # Call steadytext directly
        from steadytext import rerank

        results = rerank(
            query=query,
            documents=documents_list,
            task=task,
            return_scores=True,  # Always get scores for PostgreSQL
            seed=seed
        )

        if not results:
            results = deterministic_rerank(documents_list, query, task or '')

        normalized_results = []
        for item in results:
            if isinstance(item, dict):
                doc = item.get("document")
                raw_score = item.get("score")
            else:
                doc = item[0] if len(item) > 0 else None
                raw_score = item[1] if len(item) > 1 else None

            if doc is None:
                continue

            try:
                score_value = float(raw_score) if raw_score is not None else 0.0
            except (TypeError, ValueError):
                score_value = 0.0

            # Ensure scores stay within expected range and never hit zero
            score_value = max(0.01, min(score_value, 1.0))
            normalized_results.append(
                {"document": doc, "score": score_value if return_scores else None}
            )

        if not normalized_results:
            # Should be rare, but keep deterministic fallback as safety net
            return [
                {"document": doc, "score": score if return_scores else None}
                for doc, score in deterministic_rerank(documents_list, query, task or "")
            ]

        return normalized_results

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return [
            {"document": doc, "score": score if return_scores else None}
            for doc, score in deterministic_rerank(documents_list, query, task or '')
        ]
$c$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

-- Drop and recreate steadytext_rerank_batch
DROP FUNCTION IF EXISTS @extschema@.steadytext_rerank_batch(TEXT[], TEXT[], TEXT, BOOLEAN, INT);

CREATE OR REPLACE FUNCTION steadytext_rerank_batch(
    queries text[],
    documents text[],
    task text DEFAULT 'Given a web search query, retrieve relevant passages that answer the query',
    return_scores boolean DEFAULT true,
    seed integer DEFAULT 42
) RETURNS TABLE(query_index integer, document text, score float)
AS $c$
    import json
    import logging
    from typing import List, Tuple

    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not queries or len(queries) == 0:
        plpy.error("Queries cannot be empty")

    if not documents or len(documents) == 0:
        return []

    # Check if initialized, if not, initialize now
    if not GD.get('steadytext_initialized', False):
        # Initialize on demand
        ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
        ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
        plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
        # Check again after initialization
        if not GD.get('steadytext_initialized', False):
            plpy.error("Failed to initialize pg_steadytext Python environment")

    from steadytext import rerank

    all_results = []

    # Process each query
    for idx, query in enumerate(queries):
        try:
            # Call rerank for this query
            results = rerank(
                query=query,
                documents=list(documents),
                task=task,
                return_scores=True,
                seed=seed
            )

            # Add query index to results
            for doc, score in results:
                all_results.append({
                    "query_index": idx,
                    "document": doc,
                    "score": score
                })

        except Exception as e:
            logger.error(f"Reranking failed for query {idx}: {e}")
            # Continue with next query
            continue

    return all_results
$c$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

-- Drop and recreate steadytext_daemon_status
DROP FUNCTION IF EXISTS @extschema@.steadytext_daemon_status();

CREATE OR REPLACE FUNCTION steadytext_daemon_status()
RETURNS TABLE(
    daemon_id TEXT,
    status TEXT,
    endpoint TEXT,
    last_heartbeat TIMESTAMPTZ,
    uptime_seconds INT
)
LANGUAGE plpython3u
AS $c$
# AIDEV-NOTE: Check SteadyText daemon health status
import json

# Get extension schema first (always needed)
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'

# Check if initialized, if not, initialize now
if not GD.get('steadytext_initialized', False):
    # Initialize on demand
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    # Check again after initialization
    if not GD.get('steadytext_initialized', False):
        plpy.error("Failed to initialize pg_steadytext Python environment")

try:
    import subprocess
    result = subprocess.run(['st', 'daemon', 'status', '--json'], capture_output=True, text=True, timeout=5)

    if result.returncode == 0:
        status_data = json.loads(result.stdout)
        if status_data.get('running'):
            # Update health status to healthy
            update_plan = plpy.prepare(f"""
                UPDATE {plpy.quote_ident(ext_schema)}.steadytext_daemon_health
                SET status = 'healthy',
                    last_heartbeat = NOW(),
                    uptime_seconds = GREATEST(EXTRACT(EPOCH FROM (NOW() - COALESCE(last_heartbeat, NOW()))), 0)::INT
                WHERE daemon_id = 'default'
                RETURNING daemon_id, status, endpoint, last_heartbeat, uptime_seconds
            """)
            result = plpy.execute(update_plan)
            return result
        else:
            # Update health status to unhealthy (daemon not running)
            update_plan = plpy.prepare(f"""
                UPDATE {plpy.quote_ident(ext_schema)}.steadytext_daemon_health
                SET status = 'unhealthy',
                    last_heartbeat = NOW(),
                    uptime_seconds = 0
                WHERE daemon_id = 'default'
                RETURNING daemon_id, status, endpoint, last_heartbeat, uptime_seconds
            """)
            result = plpy.execute(update_plan)
            return result
    else:
        # Command failed, mark as unhealthy
        update_plan = plpy.prepare(f"""
            UPDATE {plpy.quote_ident(ext_schema)}.steadytext_daemon_health
            SET status = 'unhealthy',
                last_heartbeat = NOW(),
                uptime_seconds = 0
            WHERE daemon_id = 'default'
            RETURNING daemon_id, status, endpoint, last_heartbeat, uptime_seconds
        """)
        result = plpy.execute(update_plan)
        return result

except Exception as e:
    plpy.warning(f"Error checking daemon status: {e}")
    # Return current status from table
    select_plan = plpy.prepare(f"""
        SELECT daemon_id, status, endpoint, last_heartbeat, uptime_seconds
        FROM {plpy.quote_ident(ext_schema)}.steadytext_daemon_health
        WHERE daemon_id = 'default'
    """)
    return plpy.execute(select_plan)
$c$;

-- Drop and recreate steadytext_daemon_stop
DROP FUNCTION IF EXISTS @extschema@.steadytext_daemon_stop();

CREATE OR REPLACE FUNCTION steadytext_daemon_stop()
RETURNS BOOLEAN
LANGUAGE plpython3u
AS $c$
# AIDEV-NOTE: Stop the SteadyText daemon gracefully without disrupting PostgreSQL
import json

# Resolve extension schema for subsequent queries
ext_schema_result = plpy.execute("SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'

# Ensure Python modules are initialized so cached connectors are available
if not GD.get('steadytext_initialized', False):
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")

stopped_successfully = False
try:
    import subprocess
    result = subprocess.run(['st', 'daemon', 'stop'], capture_output=True, text=True, timeout=10)

    if result.returncode == 0:
        stopped_successfully = True
        plpy.notice("Daemon stopped successfully")
    else:
        stopped_successfully = False
        plpy.warning(f"Failed to stop daemon: {result.stderr}")

except subprocess.TimeoutExpired:
    stopped_successfully = False
    plpy.warning("Daemon stop command timed out")
except Exception as e:
    stopped_successfully = False
    plpy.warning(f"Error stopping daemon: {e}")

# Update daemon health status regardless of stop outcome
status_value = 'unhealthy'
update_plan = plpy.prepare(f"""
    UPDATE {plpy.quote_ident(ext_schema)}.steadytext_daemon_health
    SET status = $1,
        last_heartbeat = NOW(),
        uptime_seconds = 0
    WHERE daemon_id = 'default'
""", ["text"])
plpy.execute(update_plan, [status_value])

return stopped_successfully
$c$;

-- Final notice
DO $$
BEGIN
    RAISE NOTICE 'Migration complete. Recommend: SELECT pg_reload_conf();';
END $$;

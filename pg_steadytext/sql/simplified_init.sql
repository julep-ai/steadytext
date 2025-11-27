-- Simplified Python initialization for pg_steadytext
-- AIDEV-NOTE: Loads only essential modules, calls steadytext library directly

CREATE OR REPLACE FUNCTION _steadytext_init_python()
RETURNS void
LANGUAGE plpython3u
AS $c$
import sys
import os

# Resolve $libdir to actual PostgreSQL library directory
pg_config_result = plpy.execute("SHOW dynamic_library_path")
dynamic_lib_path = pg_config_result[0]['dynamic_library_path'] if pg_config_result else '$libdir'

# Get the actual libdir path
libdir_result = plpy.execute("SELECT setting FROM pg_config WHERE name = 'LIBDIR'")
if libdir_result and len(libdir_result) > 0:
    pg_lib_dir = libdir_result[0]['setting']
else:
    pg_lib_dir = '/usr/lib/postgresql/17/lib'

GD['pg_lib_dir'] = pg_lib_dir

# Add Python module directory to path
python_module_dir = os.path.join(pg_lib_dir, 'pg_steadytext', 'python')

if python_module_dir not in sys.path:
    sys.path.insert(0, python_module_dir)
    plpy.notice(f"Added to Python path: {python_module_dir}")

# Add site-packages if it exists
site_packages_dir = os.path.join(pg_lib_dir, 'pg_steadytext', 'site-packages')
if os.path.exists(site_packages_dir) and site_packages_dir not in sys.path:
    sys.path.insert(0, site_packages_dir)
    plpy.notice(f"Added site-packages to path: {site_packages_dir}")

# Verify required packages are available
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
    error_msg = f"""
Missing required Python packages: {', '.join(missing_packages)}

To fix this, run:
    sudo pip3 install steadytext pyzmq numpy

Or install to PostgreSQL's site-packages:
    sudo pip3 install --target={site_packages_dir} steadytext pyzmq numpy
"""
    plpy.error(error_msg)

# Load essential extension modules
try:
    # Only load security and prompt_registry
    import security
    import prompt_registry

    GD['module_security'] = security
    GD['module_prompt_registry'] = prompt_registry
    GD['steadytext_initialized'] = True

    plpy.notice("pg_steadytext initialized successfully")
except ImportError as e:
    GD['steadytext_initialized'] = False
    plpy.error(f"Failed to import pg_steadytext modules: {e}")
$c$;
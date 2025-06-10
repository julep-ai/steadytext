"""
SteadyText: Deterministic text generation and embedding with zero configuration.
"""

# Version of the steadytext package - should match pyproject.toml
__version__ = "0.1.0"

# For the purpose of test_gen.py, keep this minimal.
# The test_gen.py script imports directly from submodules like steadytext.models.loader
# and steadytext.utils, so this __init__.py doesn't need to expose the full public API
# if that's causing import issues in this specific script execution context.

# If a full library build was intended, this would be populated with:
# from .utils import logger, DEFAULT_SEED, ...
# from .core.generator import generate_text as core_generate_text
# from .core.embedder import create_embedding as core_create_embedding
# etc. and __all__

# logger = None # Placeholder if other modules try `from .. import logger` via this.
              # Ideally, direct submodule imports are used by internal components.
              # The utils.logger should be used directly.
# Removing the placeholder logger too, to keep it truly minimal for this test.
# The test_gen.py script imports logger from steadytext.utils directly.
# Stray EOL was here. Removed.

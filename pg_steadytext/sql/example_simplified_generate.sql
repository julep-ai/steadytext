-- Example: Simplified steadytext_generate function
-- AIDEV-NOTE: This shows the new pattern - call steadytext directly, no validation

CREATE OR REPLACE FUNCTION steadytext_generate(
    prompt TEXT,
    max_tokens INT DEFAULT NULL,
    use_cache BOOLEAN DEFAULT FALSE,
    seed INT DEFAULT NULL,
    eos_string TEXT DEFAULT '[EOS]',
    model TEXT DEFAULT NULL,
    model_repo TEXT DEFAULT NULL,
    model_filename TEXT DEFAULT NULL,
    size TEXT DEFAULT NULL,
    unsafe_mode BOOLEAN DEFAULT FALSE
)
RETURNS TEXT
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
import json

# Helper to raise P0001 errors
def raise_error(message: str) -> None:
    plan = GD.get('error_plan')
    if plan is None:
        ext_schema = plpy.execute(
            "SELECT nspname FROM pg_extension e JOIN pg_namespace n "
            "ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'"
        )[0]['nspname']
        plan = plpy.prepare(
            f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_raise_p0001($1)",
            ["text"]
        )
        GD['error_plan'] = plan
    plpy.execute(plan, [message])

# Initialize if needed
if not GD.get('steadytext_initialized', False):
    ext_schema = plpy.execute(
        "SELECT nspname FROM pg_extension e JOIN pg_namespace n "
        "ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'"
    )[0]['nspname']
    plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")

# Get config (cached in GD)
config_plan = GD.get('config_plan')
if config_plan is None:
    ext_schema = plpy.execute(
        "SELECT nspname FROM pg_extension e JOIN pg_namespace n "
        "ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'"
    )[0]['nspname']
    config_plan = plpy.prepare(
        f"SELECT value FROM {plpy.quote_ident(ext_schema)}.steadytext_config WHERE key = $1",
        ["text"]
    )
    GD['config_plan'] = config_plan

# Resolve defaults from config
if max_tokens is None:
    max_tokens = json.loads(plpy.execute(config_plan, ["default_max_tokens"])[0]["value"])
if seed is None:
    seed = json.loads(plpy.execute(config_plan, ["default_seed"])[0]["value"])

# Basic null/empty checks (no validation - app responsibility)
if prompt is None:
    raise_error("Prompt cannot be null")
if prompt.strip() == "":
    raise_error("Prompt cannot be empty")
if seed < 0:
    raise_error("seed must be non-negative")

# Validate unsafe_mode usage
if not unsafe_mode and model and ':' in model:
    raise_error("Remote models require unsafe_mode=TRUE")
if unsafe_mode and not model:
    raise_error("unsafe_mode=TRUE requires a model parameter")

# Check cache if enabled
if use_cache:
    cache_key = f"{prompt}::EOS::{eos_string}" if eos_string != '[EOS]' else prompt
    cache_plan = GD.get('cache_plan')
    if cache_plan is None:
        ext_schema = plpy.execute(
            "SELECT nspname FROM pg_extension e JOIN pg_namespace n "
            "ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'"
        )[0]['nspname']
        cache_plan = plpy.prepare(
            f"SELECT response FROM {plpy.quote_ident(ext_schema)}.steadytext_cache WHERE cache_key = $1",
            ["text"]
        )
        GD['cache_plan'] = cache_plan

    result = plpy.execute(cache_plan, [cache_key])
    if result and result[0]["response"]:
        return result[0]["response"]

# Call steadytext directly
from steadytext import generate

kwargs = {"seed": seed, "eos_string": eos_string}
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

result = generate(prompt, max_new_tokens=max_tokens, **kwargs)

# Cache the result if caching enabled
if use_cache:
    insert_plan = GD.get('cache_insert_plan')
    if insert_plan is None:
        ext_schema = plpy.execute(
            "SELECT nspname FROM pg_extension e JOIN pg_namespace n "
            "ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'"
        )[0]['nspname']
        insert_plan = plpy.prepare(
            f"""INSERT INTO {plpy.quote_ident(ext_schema)}.steadytext_cache
                (cache_key, prompt, response) VALUES ($1, $2, $3)
                ON CONFLICT (cache_key) DO NOTHING""",
            ["text", "text", "text"]
        )
        GD['cache_insert_plan'] = insert_plan
    plpy.execute(insert_plan, [cache_key, prompt, result])

return result
$c$;
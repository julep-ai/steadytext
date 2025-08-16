-- pg_steadytext--2025.8.16--2025.8.17.sql
-- Simplified upgrade: Add unsafe_mode and model support to summarization

-- Update steadytext_summarize_text to support model and unsafe_mode parameters
CREATE OR REPLACE FUNCTION steadytext_summarize_text(
    input_text text,
    metadata jsonb DEFAULT '{}'::jsonb,
    model text DEFAULT NULL,
    unsafe_mode boolean DEFAULT FALSE
) RETURNS text
LANGUAGE plpython3u
IMMUTABLE PARALLEL SAFE
AS $c$
import json

# If model and unsafe_mode are provided as parameters, add them to metadata
meta_dict = {}
if metadata:
    try:
        meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
    except (json.JSONDecodeError, TypeError):
        meta_dict = {}

# Override metadata with explicit parameters if provided
if model is not None:
    meta_dict['model'] = model
if unsafe_mode is not None:
    meta_dict['unsafe_mode'] = unsafe_mode

# For simple case, just use local model with basic summary
if not model:
    # Simple local summarization
    if input_text and len(input_text) > 0:
        # Take first 100 chars as summary
        summary = input_text[:100]
        if len(input_text) > 100:
            summary += "..."
        return f"Summary: {summary}"
    return "No text to summarize"

# For remote models, validate unsafe_mode
if ':' in model and not unsafe_mode:
    plpy.error("Remote models (containing ':') require unsafe_mode=TRUE")

# Use steadytext_generate for actual summarization
prompt = f"Summarize the following text in 1-2 sentences: {input_text[:500]}"

plan = plpy.prepare(
    "SELECT steadytext_generate($1, NULL, true, 42, '[EOS]', $2, NULL, NULL, NULL, $3) as summary",
    ["text", "text", "boolean"]
)
result = plpy.execute(plan, [prompt, model, unsafe_mode])

if result and result[0]["summary"]:
    return result[0]["summary"]
return "Unable to generate summary"
$c$;

-- Create st_summarize_text alias with new parameters
CREATE OR REPLACE FUNCTION st_summarize_text(
    input_text text,
    metadata jsonb DEFAULT '{}'::jsonb,
    model text DEFAULT NULL,
    unsafe_mode boolean DEFAULT FALSE
) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE
AS $$
    SELECT steadytext_summarize_text($1, $2, $3, $4);
$$;

-- Update version function
CREATE OR REPLACE FUNCTION steadytext_version()
RETURNS TEXT AS $$
BEGIN
    RETURN '2025.8.17';
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- Create short alias for version function
CREATE OR REPLACE FUNCTION st_version()
RETURNS TEXT AS $$
    SELECT steadytext_version();
$$ LANGUAGE sql IMMUTABLE PARALLEL SAFE;
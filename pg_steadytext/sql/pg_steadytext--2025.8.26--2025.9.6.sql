-- Migration from 2025.8.26 to 2025.9.6
-- Adds prompt registry functionality with Jinja2 template support

-- AIDEV-NOTE: This migration adds prompt registry tables and functions
-- for lightweight template management with versioning support

-- Drop tables if they exist (for clean migration)
DROP TABLE IF EXISTS @extschema@.steadytext_prompt_versions CASCADE;
DROP TABLE IF EXISTS @extschema@.steadytext_prompts CASCADE;

-- Create prompts table
CREATE TABLE @extschema@.steadytext_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT UNIQUE NOT NULL CHECK (slug ~ '^[a-z0-9]+(?:-[a-z0-9]+)*$' AND LENGTH(slug) BETWEEN 3 AND 100),
    description TEXT CHECK (LENGTH(description) <= 500),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT current_user,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by TEXT DEFAULT current_user
);

-- Create prompt versions table
CREATE TABLE @extschema@.steadytext_prompt_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_id UUID NOT NULL REFERENCES @extschema@.steadytext_prompts(id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK (version > 0),
    template TEXT NOT NULL CHECK (LENGTH(template) > 0),
    required_variables TEXT[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT current_user,
    is_active BOOLEAN DEFAULT FALSE,
    UNIQUE(prompt_id, version)
);

-- Create indexes for performance
CREATE INDEX idx_steadytext_prompts_slug ON @extschema@.steadytext_prompts(slug);
CREATE INDEX idx_steadytext_prompts_created_at ON @extschema@.steadytext_prompts(created_at DESC);
CREATE INDEX idx_steadytext_prompt_versions_prompt_id_active ON @extschema@.steadytext_prompt_versions(prompt_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_steadytext_prompt_versions_created_at ON @extschema@.steadytext_prompt_versions(created_at DESC);

-- Helper function to get next version number
CREATE OR REPLACE FUNCTION @extschema@._get_next_version(p_prompt_id UUID)
RETURNS INTEGER
LANGUAGE sql
STABLE
AS $$
    SELECT COALESCE(MAX(version), 0) + 1
    FROM @extschema@.steadytext_prompt_versions
    WHERE prompt_id = p_prompt_id;
$$;

-- Python function to validate Jinja2 templates
CREATE OR REPLACE FUNCTION @extschema@._validate_jinja2_template(template TEXT)
RETURNS TABLE(is_valid BOOLEAN, required_variables TEXT[], error_message TEXT)
LANGUAGE plpython3u
AS $$
    # AIDEV-NOTE: Validates Jinja2 template syntax and extracts required variables
    import re
    
    # Initialize Python environment if needed
    if not GD.get('steadytext_initialized', False):
        ext_schema_result = plpy.execute(
            "SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'"
        )
        ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
        plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    
    # Try to import Jinja2
    try:
        from jinja2 import Environment, meta, TemplateSyntaxError
    except ImportError:
        return [(False, None, "Jinja2 is not installed. Please install it using: pip install jinja2")]
    
    env = Environment()
    
    try:
        # Parse the template to check syntax
        ast = env.parse(template)
        
        # Extract all variables referenced in the template
        variables = meta.find_undeclared_variables(ast)
        
        # Convert to sorted list for consistency
        required_vars = sorted(list(variables))
        
        return [(True, required_vars, None)]
        
    except TemplateSyntaxError as e:
        return [(False, None, str(e))]
    except Exception as e:
        return [(False, None, f"Unexpected error: {str(e)}")]
$$;

-- Create prompt with first version
CREATE OR REPLACE FUNCTION @extschema@.steadytext_prompt_create(
    slug TEXT,
    template TEXT,
    description TEXT DEFAULT NULL,
    metadata JSONB DEFAULT '{}'
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_prompt_id UUID;
    v_validation RECORD;
BEGIN
    -- Validate template
    SELECT * INTO v_validation 
    FROM @extschema@._validate_jinja2_template(template);
    
    IF NOT v_validation.is_valid THEN
        RAISE EXCEPTION 'Invalid Jinja2 template: %', v_validation.error_message
            USING ERRCODE = 'syntax_error';
    END IF;
    
    -- Insert prompt
    INSERT INTO @extschema@.steadytext_prompts (slug, description)
    VALUES (slug, description)
    RETURNING id INTO v_prompt_id;
    
    -- Insert first version
    INSERT INTO @extschema@.steadytext_prompt_versions (
        prompt_id, version, template, required_variables, metadata, is_active
    ) VALUES (
        v_prompt_id, 1, template, v_validation.required_variables, metadata, TRUE
    );
    
    RETURN v_prompt_id;
EXCEPTION
    WHEN unique_violation THEN
        RAISE EXCEPTION 'Prompt with slug "%" already exists', slug
            USING ERRCODE = 'unique_violation';
    WHEN check_violation THEN
        RAISE EXCEPTION 'Invalid slug format. Use lowercase letters, numbers, and hyphens only (3-100 chars)'
            USING ERRCODE = 'invalid_text_representation';
END;
$$;

-- Update prompt (create new version)
CREATE OR REPLACE FUNCTION @extschema@.steadytext_prompt_update(
    slug TEXT,
    template TEXT,
    metadata JSONB DEFAULT '{}'
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    v_prompt_id UUID;
    v_next_version INTEGER;
    v_version_id UUID;
    v_validation RECORD;
BEGIN
    -- Get prompt ID
    SELECT id INTO v_prompt_id
    FROM @extschema@.steadytext_prompts
    WHERE steadytext_prompts.slug = steadytext_prompt_update.slug;
    
    IF v_prompt_id IS NULL THEN
        RAISE EXCEPTION 'Prompt with slug "%" not found', slug
            USING ERRCODE = 'no_data_found';
    END IF;
    
    -- Acquire advisory lock to prevent concurrent version updates
    -- Uses the first 8 bytes of the UUID as bigint for the lock
    PERFORM pg_advisory_xact_lock(('\x' || substr(replace(v_prompt_id::text, '-', ''), 1, 16))::bit(64)::bigint);
    
    -- Validate template
    SELECT * INTO v_validation 
    FROM @extschema@._validate_jinja2_template(template);
    
    IF NOT v_validation.is_valid THEN
        RAISE EXCEPTION 'Invalid Jinja2 template: %', v_validation.error_message
            USING ERRCODE = 'syntax_error';
    END IF;
    
    -- Get next version number
    SELECT @extschema@._get_next_version(v_prompt_id) INTO v_next_version;
    
    -- Deactivate current active version
    UPDATE @extschema@.steadytext_prompt_versions
    SET is_active = FALSE
    WHERE prompt_id = v_prompt_id AND is_active = TRUE;
    
    -- Insert new version
    INSERT INTO @extschema@.steadytext_prompt_versions (
        prompt_id, version, template, required_variables, metadata, is_active
    ) VALUES (
        v_prompt_id, v_next_version, template, v_validation.required_variables, metadata, TRUE
    ) RETURNING id INTO v_version_id;
    
    -- Update prompt's updated_at
    UPDATE @extschema@.steadytext_prompts
    SET updated_at = NOW(), updated_by = current_user
    WHERE id = v_prompt_id;
    
    RETURN v_version_id;
END;
$$;

-- Get prompt template  
CREATE OR REPLACE FUNCTION @extschema@.steadytext_prompt_get(
    slug TEXT,
    version INTEGER DEFAULT NULL
) RETURNS TABLE(
    prompt_id UUID,
    version_num INTEGER,
    template TEXT,
    required_variables TEXT[],
    metadata JSONB,
    created_at TIMESTAMPTZ,
    created_by TEXT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_prompt_id UUID;
BEGIN
    -- Get prompt ID
    SELECT id INTO v_prompt_id
    FROM @extschema@.steadytext_prompts p
    WHERE p.slug = steadytext_prompt_get.slug;
    
    IF v_prompt_id IS NULL THEN
        RAISE EXCEPTION 'Prompt with slug "%" not found', slug
            USING ERRCODE = 'no_data_found';
    END IF;
    
    -- Return specific version or active version
    IF version IS NOT NULL THEN
        RETURN QUERY
        SELECT pv.prompt_id, pv.version AS version_num, pv.template, pv.required_variables, 
               pv.metadata, pv.created_at, pv.created_by
        FROM @extschema@.steadytext_prompt_versions pv
        WHERE pv.prompt_id = v_prompt_id AND pv.version = steadytext_prompt_get.version;
        
        IF NOT FOUND THEN
            RAISE EXCEPTION 'Version % not found for prompt "%"', version, slug
                USING ERRCODE = 'no_data_found';
        END IF;
    ELSE
        RETURN QUERY
        SELECT pv.prompt_id, pv.version AS version_num, pv.template, pv.required_variables, 
               pv.metadata, pv.created_at, pv.created_by
        FROM @extschema@.steadytext_prompt_versions pv
        WHERE pv.prompt_id = v_prompt_id AND pv.is_active = TRUE;
    END IF;
END;
$$;

-- Render prompt template
CREATE OR REPLACE FUNCTION @extschema@.steadytext_prompt_render(
    slug TEXT,
    variables JSONB,
    version INTEGER DEFAULT NULL,
    strict BOOLEAN DEFAULT TRUE
) RETURNS TEXT
LANGUAGE plpython3u
AS $$
    # AIDEV-NOTE: Renders Jinja2 template with provided variables
    
    # Initialize Python environment if needed
    if not GD.get('steadytext_initialized', False):
        ext_schema_result = plpy.execute(
            "SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'"
        )
        ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
        plpy.execute(f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_init_python()")
    
    # Get extension schema
    ext_schema_result = plpy.execute(
        "SELECT nspname FROM pg_extension e JOIN pg_namespace n ON e.extnamespace = n.oid WHERE e.extname = 'pg_steadytext'"
    )
    ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'
    
    # Try to import Jinja2
    try:
        from jinja2 import Environment, TemplateSyntaxError, UndefinedError, StrictUndefined, Undefined
    except ImportError:
        plpy.error("Jinja2 is not installed. Please install it using: pip install jinja2")
    
    import json
    
    # Get the template
    if version is not None:
        query = f"""
            SELECT template, required_variables 
            FROM {plpy.quote_ident(ext_schema)}.steadytext_prompt_get($1, $2)
        """
        plan = plpy.prepare(query, ["text", "integer"])
        result = plpy.execute(plan, [slug, version])
    else:
        query = f"""
            SELECT template, required_variables 
            FROM {plpy.quote_ident(ext_schema)}.steadytext_prompt_get($1)
        """
        plan = plpy.prepare(query, ["text"])
        result = plpy.execute(plan, [slug])
    
    if not result:
        plpy.error(f"Prompt with slug '{slug}' not found")
    
    template_text = result[0]['template']
    required_vars = result[0]['required_variables'] or []
    
    # Parse variables
    # AIDEV-FIX: Ensure proper conversion of JSONB to dict
    if isinstance(variables, str):
        try:
            vars_dict = json.loads(variables)
        except json.JSONDecodeError as e:
            plpy.error(f"Invalid JSON in variables: {str(e)}")
    elif hasattr(variables, 'items'):
        # It's already dict-like, but ensure it's a plain dict
        vars_dict = dict(variables) if variables else {}
    else:
        vars_dict = {}
    
    # Check for missing required variables in strict mode
    if strict and required_vars:
        missing = [var for var in required_vars if var not in vars_dict]
        if missing:
            plpy.error(f"Missing required variables: {', '.join(missing)}")
    
    # Create Jinja2 environment
    if strict:
        env = Environment(undefined=StrictUndefined)
    else:
        env = Environment(undefined=Undefined)
    
    # Cache compiled templates in GD for performance
    # Include strict mode in cache key since it affects template behavior
    cache_key = f"jinja2_template:{slug}:{version}:{strict}:{hash(template_text)}"
    
    # AIDEV-FIX: Don't cache compiled templates - they're not serializable in GD
    # Instead, compile fresh each time (Jinja2 compilation is fast)
    try:
        compiled_template = env.from_string(template_text)
    except TemplateSyntaxError as e:
        plpy.error(f"Template syntax error: {str(e)}")
    
    # Render the template
    try:
        # AIDEV-NOTE: JSONB dict limitation - Cannot use 'items', 'keys', 'values' as JSON keys
        # These conflict with dict methods when JSONB is passed to Jinja2 templates
        rendered = compiled_template.render(**vars_dict)
        return rendered
    except UndefinedError as e:
        plpy.error(f"Undefined variable in template: {str(e)}")
    except Exception as e:
        plpy.error(f"Template rendering error: {str(e)}")
$$;

-- List all prompts
CREATE OR REPLACE FUNCTION @extschema@.steadytext_prompt_list()
RETURNS TABLE(
    prompt_id UUID,
    slug TEXT,
    description TEXT,
    latest_version_num INTEGER,
    total_versions INTEGER,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
)
LANGUAGE sql
STABLE PARALLEL SAFE
AS $$
    SELECT 
        p.id,
        p.slug,
        p.description,
        MAX(pv.version) as latest_version_num,
        COUNT(pv.id)::INTEGER as total_versions,
        p.created_at,
        p.updated_at
    FROM @extschema@.steadytext_prompts p
    LEFT JOIN @extschema@.steadytext_prompt_versions pv ON p.id = pv.prompt_id
    GROUP BY p.id, p.slug, p.description, p.created_at, p.updated_at
    ORDER BY p.created_at DESC;
$$;

-- List versions of a prompt
CREATE OR REPLACE FUNCTION @extschema@.steadytext_prompt_versions(slug TEXT)
RETURNS TABLE(
    version INTEGER,
    template TEXT,
    required_variables TEXT[],
    metadata JSONB,
    created_at TIMESTAMPTZ,
    created_by TEXT,
    is_active BOOLEAN
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_prompt_id UUID;
BEGIN
    -- Get prompt ID
    SELECT id INTO v_prompt_id
    FROM @extschema@.steadytext_prompts p
    WHERE p.slug = steadytext_prompt_versions.slug;
    
    IF v_prompt_id IS NULL THEN
        RAISE EXCEPTION 'Prompt with slug "%" not found', slug
            USING ERRCODE = 'no_data_found';
    END IF;
    
    RETURN QUERY
    SELECT pv.version, pv.template, pv.required_variables, pv.metadata,
           pv.created_at, pv.created_by, pv.is_active
    FROM @extschema@.steadytext_prompt_versions pv
    WHERE pv.prompt_id = v_prompt_id
    ORDER BY pv.version DESC;
END;
$$;

-- Delete prompt and all versions
CREATE OR REPLACE FUNCTION @extschema@.steadytext_prompt_delete(slug TEXT)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    v_prompt_id UUID;
BEGIN
    -- Get prompt ID
    SELECT id INTO v_prompt_id
    FROM @extschema@.steadytext_prompts p
    WHERE p.slug = steadytext_prompt_delete.slug;
    
    IF v_prompt_id IS NULL THEN
        RETURN FALSE;
    END IF;
    
    -- Delete prompt (cascade will delete versions)
    DELETE FROM @extschema@.steadytext_prompts
    WHERE id = v_prompt_id;
    
    RETURN TRUE;
END;
$$;

-- Create short aliases for all prompt functions
CREATE OR REPLACE FUNCTION @extschema@.st_prompt_create(
    slug TEXT,
    template TEXT,
    description TEXT DEFAULT NULL,
    metadata JSONB DEFAULT '{}'
) RETURNS UUID
LANGUAGE sql
AS $$
    SELECT @extschema@.steadytext_prompt_create($1, $2, $3, $4);
$$;

CREATE OR REPLACE FUNCTION @extschema@.st_prompt_update(
    slug TEXT,
    template TEXT,
    metadata JSONB DEFAULT '{}'
) RETURNS UUID
LANGUAGE sql
AS $$
    SELECT @extschema@.steadytext_prompt_update($1, $2, $3);
$$;

CREATE OR REPLACE FUNCTION @extschema@.st_prompt_get(
    slug TEXT,
    version INTEGER DEFAULT NULL
) RETURNS TABLE(
    prompt_id UUID,
    version_num INTEGER,
    template TEXT,
    required_variables TEXT[],
    metadata JSONB,
    created_at TIMESTAMPTZ,
    created_by TEXT
)
LANGUAGE sql
AS $$
    SELECT * FROM @extschema@.steadytext_prompt_get($1, $2);
$$;

CREATE OR REPLACE FUNCTION @extschema@.st_prompt_render(
    slug TEXT,
    variables JSONB,
    version INTEGER DEFAULT NULL,
    strict BOOLEAN DEFAULT TRUE
) RETURNS TEXT
LANGUAGE sql
AS $$
    SELECT @extschema@.steadytext_prompt_render($1, $2, $3, $4);
$$;

CREATE OR REPLACE FUNCTION @extschema@.st_prompt_list()
RETURNS TABLE(
    prompt_id UUID,
    slug TEXT,
    description TEXT,
    latest_version_num INTEGER,
    total_versions INTEGER,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
)
LANGUAGE sql
AS $$
    SELECT * FROM @extschema@.steadytext_prompt_list();
$$;

CREATE OR REPLACE FUNCTION @extschema@.st_prompt_versions(slug TEXT)
RETURNS TABLE(
    version INTEGER,
    template TEXT,
    required_variables TEXT[],
    metadata JSONB,
    created_at TIMESTAMPTZ,
    created_by TEXT,
    is_active BOOLEAN
)
LANGUAGE sql
AS $$
    SELECT * FROM @extschema@.steadytext_prompt_versions($1);
$$;

CREATE OR REPLACE FUNCTION @extschema@.st_prompt_delete(slug TEXT)
RETURNS BOOLEAN
LANGUAGE sql
AS $$
    SELECT @extschema@.steadytext_prompt_delete($1);
$$;

-- AIDEV-NOTE: Prompt registry feature added in v2025.9.6
-- Provides lightweight Jinja2-based template management with versioning
-- All functions have st_* aliases for convenience
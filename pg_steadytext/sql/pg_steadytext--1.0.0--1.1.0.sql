-- pg_steadytext extension migration from 1.0.0 to 1.1.0
-- This migration is intentionally empty as the changes were moved to 1.1.0--1.2.0
-- AIDEV-NOTE: AI summarization aggregate functions have been moved to version 1.2.0

-- Update version
CREATE OR REPLACE FUNCTION steadytext_version()
RETURNS text AS $$
BEGIN
    RETURN '1.1.0';
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
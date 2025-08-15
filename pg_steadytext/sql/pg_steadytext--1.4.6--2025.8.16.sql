-- pg_steadytext migration from 1.4.6 to 2025.8.16
-- This migration switches to date-based versioning

-- AIDEV-NOTE: This is a version-only migration to transition from semantic versioning
-- to date-based versioning (yyyy.mm.dd format, no zero padding)
-- No schema changes are included in this migration

-- Update the version function to return the new date-based version
CREATE OR REPLACE FUNCTION steadytext_version()
RETURNS TEXT AS $$
BEGIN
    RETURN '2025.8.16';
END;
$$ LANGUAGE plpgsql STABLE;

-- Log the version change
DO $$
BEGIN
    RAISE NOTICE 'pg_steadytext migrated from version 1.4.6 to 2025.8.16';
    RAISE NOTICE 'Now using date-based versioning (yyyy.mm.dd format)';
END $$;
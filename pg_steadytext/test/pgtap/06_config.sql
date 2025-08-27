BEGIN;
SELECT plan(18);

-- Table exists
SELECT has_table('public', 'steadytext_config', 'steadytext_config table should exist');
SELECT has_column('steadytext_config', 'key', 'steadytext_config.key exists');
SELECT has_column('steadytext_config', 'value', 'steadytext_config.value exists');
SELECT has_column('steadytext_config', 'description', 'steadytext_config.description exists');
SELECT has_column('steadytext_config', 'updated_at', 'steadytext_config.updated_at exists');

-- Default keys exist
SELECT ok(EXISTS(SELECT 1 FROM steadytext_config WHERE key = 'cache_enabled'), 'cache_enabled default exists');
SELECT ok(EXISTS(SELECT 1 FROM steadytext_config WHERE key = 'cache_max_entries'), 'cache_max_entries default exists');
SELECT ok(EXISTS(SELECT 1 FROM steadytext_config WHERE key = 'cache_max_size_mb'), 'cache_max_size_mb default exists');
SELECT ok(EXISTS(SELECT 1 FROM steadytext_config WHERE key = 'daemon_host'), 'daemon_host default exists');
SELECT ok(EXISTS(SELECT 1 FROM steadytext_config WHERE key = 'daemon_port'), 'daemon_port default exists');
SELECT ok(EXISTS(SELECT 1 FROM steadytext_config WHERE key = 'cache_eviction_enabled'), 'cache_eviction_enabled default exists');

-- JSONB values are valid and castable
SELECT ok((SELECT (value #>> '{}')::boolean FROM steadytext_config WHERE key = 'cache_enabled') IN (true, false), 'cache_enabled is boolean-castable');
SELECT ok((SELECT (value #>> '{}')::integer FROM steadytext_config WHERE key = 'cache_max_entries') > 0, 'cache_max_entries is positive integer');
SELECT ok((SELECT (value #>> '{}')::float FROM steadytext_config WHERE key = 'cache_max_size_mb') > 0.0, 'cache_max_size_mb is positive float');
SELECT ok((SELECT (value #>> '{}')::integer FROM steadytext_config WHERE key = 'daemon_port') > 0, 'daemon_port is positive integer');

-- Read via helper getter returns extracted text value (not JSON)
SELECT is(
    steadytext_config_get('daemon_host'),
    'localhost',
    'steadytext_config_get returns extracted text value of daemon_host'
);

-- Update via setter writes JSONB string
SELECT steadytext_config_set('cache_enabled', 'true');
SELECT is((SELECT (value #>> '{}')::boolean FROM steadytext_config WHERE key = 'cache_enabled'), true, 'setter writes boolean true');

SELECT * FROM finish();
ROLLBACK; 
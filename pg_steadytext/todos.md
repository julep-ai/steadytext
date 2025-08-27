# pg_steadytext TODO List

## REMOVE - Clean up over-engineered features
These features should be completely removed from code, tests, docs, and CLAUDE.md as they represent unnecessary complexity:

### Functions to Remove
- [ ] Remove `steadytext_cache_warmup` references
  - Test file: test/pgtap/05_cache_daemon.sql, 07_cache_eviction.sql
  - CLAUDE.md references
  
- [ ] Remove `steadytext_daemon_restart` references
  - Test file: test/pgtap/05_cache_daemon.sql
  - CLAUDE.md references
  
- [ ] Remove all streaming function references
  - `steadytext_generate_stream`
  - `steadytext_embed_stream`
  - Test file: test/pgtap/06_streaming.sql (entire file can be removed)
  
- [ ] Remove cron function references
  - `steadytext_setup_cache_eviction_cron`
  - `steadytext_disable_cache_eviction_cron`
  - Test file: test/pgtap/07_cache_eviction.sql
  - CLAUDE.md sections about pg_cron
  
- [ ] Remove `steadytext_cache_evict_by_frecency` references
  - Already have `steadytext_cache_evict_by_age` which serves the same purpose
  - Test file: test/pgtap/07_cache_eviction.sql
  - Update to use the existing `evict_by_age` function
  
- [ ] Remove async JSON/regex/choice function references
  - `steadytext_generate_json_async`
  - `steadytext_generate_regex_async`
  - `steadytext_generate_choice_async`
  - Test file: test/pgtap/03_async.sql
  - These add unnecessary complexity when basic async generation exists

### Other Cleanup
- [ ] Remove `steadytext_cache_stats_extended` references (over-engineered)
- [ ] Remove serialization/deserialization functions for summarization
  - `steadytext_summarize_serialize`
  - `steadytext_summarize_deserialize`

## IMPLEMENT URGENT - Critical missing components

### steadytext_config table
- [ ] Create configuration table for extension settings
  ```sql
  CREATE TABLE steadytext_config (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      description TEXT,
      updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
  );
  ```
- [ ] Add default configuration values:
  - `cache_enabled`: 'false' (default)
  - `cache_max_entries`: '10000'
  - `cache_max_size_mb`: '100'
  - `daemon_host`: 'localhost'
  - `daemon_port`: '5555'
- [ ] Update functions to read from config table
- [ ] Add to migration scripts

## IMPLEMENT LATER - Bug fixes and improvements

### Fix steadytext_rerank_docs_only
- [ ] Fix PL/Python implementation error with plpy.execute parameters
- [ ] Current issue: Using $1, $2 placeholders incorrectly with plpy.execute
- [ ] Solution: Use plpy.prepare() with proper parameter binding

### Fix steadytext_deduplicate_facts
- [ ] Implement actual deduplication logic
- [ ] Current behavior: Just returns input unchanged
- [ ] Should: Use similarity comparison to remove duplicate facts
- [ ] Consider using embedding similarity or text similarity metrics

## Documentation Updates
- [ ] Update CLAUDE.md to remove references to removed functions
- [ ] Update README to reflect actual available functionality
- [ ] Add migration guide for users relying on removed functions

## Testing Updates
- [ ] Remove test file: test/pgtap/06_streaming.sql
- [ ] Update test/pgtap/03_async.sql to remove JSON/regex/choice async tests
- [ ] Update test/pgtap/05_cache_daemon.sql to remove warmup and restart tests
- [ ] Update test/pgtap/07_cache_eviction.sql to use evict_by_age consistently
- [ ] Update test/pgtap/09_ai_summarization.sql to remove serialize/deserialize tests

## Notes
- Focus on core functionality: text generation, embeddings, reranking, and basic caching
- Avoid feature creep and over-engineering
- Keep the API surface small and maintainable
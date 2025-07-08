-- pg_steadytext extension migration from 1.0.0 to 1.1.0
-- Adds AI summarization aggregate functions with TimescaleDB support

-- AIDEV-NOTE: This migration adds AI summary aggregates that handle non-transitivity
-- through structured fact extraction and semantic deduplication

-- Add version tracking
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'steadytext_version') THEN
        CREATE OR REPLACE FUNCTION steadytext_version()
        RETURNS text AS '
        BEGIN
            RETURN ''1.1.0'';
        END;
        ' LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
    END IF;
END$$;

-- Helper function to extract facts from text using JSON generation
CREATE OR REPLACE FUNCTION ai_extract_facts(
    input_text text,
    max_facts integer DEFAULT 5
) RETURNS jsonb
AS $$
    import json
    from plpy import quote_literal
    
    # AIDEV-NOTE: Use steadytext's JSON generation with schema for structured fact extraction
    schema = {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": max_facts,
                "description": "Key facts extracted from the text"
            }
        },
        "required": ["facts"]
    }
    
    prompt = f"Extract up to {max_facts} key facts from this text: {input_text}"
    
    # Use daemon_connector for JSON generation
    plan = plpy.prepare(
        "SELECT steadytext_generate_json($1, $2::jsonb) as result",
        ["text", "jsonb"]
    )
    result = plpy.execute(plan, [prompt, json.dumps(schema)])
    
    if result and result[0]["result"]:
        try:
            return json.loads(result[0]["result"])
        except:
            # Fallback to empty facts array
            return {"facts": []}
    return {"facts": []}
$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

-- Helper function to deduplicate facts using embeddings
CREATE OR REPLACE FUNCTION ai_deduplicate_facts(
    facts_array jsonb,
    similarity_threshold float DEFAULT 0.85
) RETURNS jsonb
AS $$
    import json
    import numpy as np
    
    facts = json.loads(facts_array)
    if not facts or len(facts) == 0:
        return json.dumps([])
    
    # Extract text from fact objects if they have structure
    fact_texts = []
    for fact in facts:
        if isinstance(fact, dict) and "text" in fact:
            fact_texts.append(fact["text"])
        elif isinstance(fact, str):
            fact_texts.append(fact)
    
    if len(fact_texts) <= 1:
        return facts_array
    
    # Generate embeddings for all facts
    embeddings = []
    for text in fact_texts:
        plan = plpy.prepare("SELECT steadytext_embed($1) as embedding", ["text"])
        result = plpy.execute(plan, [text])
        if result and result[0]["embedding"]:
            embeddings.append(np.array(result[0]["embedding"]))
    
    # Deduplicate based on cosine similarity
    unique_indices = [0]  # Always keep first fact
    for i in range(1, len(embeddings)):
        is_duplicate = False
        for j in unique_indices:
            # Calculate cosine similarity
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)
    
    # Return deduplicated facts
    unique_facts = [facts[i] for i in unique_indices]
    return json.dumps(unique_facts)
$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

-- State accumulator function for AI summarization
CREATE OR REPLACE FUNCTION ai_summarize_accumulate(
    state jsonb,
    value text,
    metadata jsonb DEFAULT '{}'::jsonb
) RETURNS jsonb
AS $$
    import json
    
    # Initialize state if null
    if state is None:
        state = {
            "facts": [],
            "samples": [],
            "stats": {
                "row_count": 0,
                "total_chars": 0,
                "min_length": None,
                "max_length": 0
            },
            "metadata": {}
        }
    else:
        state = json.loads(state)
    
    if value is None:
        return json.dumps(state)
    
    # Extract facts from the value
    plan = plpy.prepare("SELECT ai_extract_facts($1, 3) as facts", ["text"])
    result = plpy.execute(plan, [value])
    
    if result and result[0]["facts"]:
        extracted = json.loads(result[0]["facts"])
        if "facts" in extracted:
            state["facts"].extend(extracted["facts"])
    
    # Update statistics
    value_len = len(value)
    state["stats"]["row_count"] += 1
    state["stats"]["total_chars"] += value_len
    
    if state["stats"]["min_length"] is None or value_len < state["stats"]["min_length"]:
        state["stats"]["min_length"] = value_len
    if value_len > state["stats"]["max_length"]:
        state["stats"]["max_length"] = value_len
    
    # Sample every 10th row (up to 10 samples)
    if state["stats"]["row_count"] % 10 == 1 and len(state["samples"]) < 10:
        state["samples"].append(value[:200])  # First 200 chars
    
    # Merge metadata
    if metadata:
        meta = json.loads(metadata) if isinstance(metadata, str) else metadata
        for key, value in meta.items():
            if key not in state["metadata"]:
                state["metadata"][key] = value
    
    return json.dumps(state)
$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

-- Combiner function for parallel aggregation
CREATE OR REPLACE FUNCTION ai_summarize_combine(
    state1 jsonb,
    state2 jsonb
) RETURNS jsonb
AS $$
    import json
    
    if state1 is None:
        return state2
    if state2 is None:
        return state1
    
    s1 = json.loads(state1)
    s2 = json.loads(state2)
    
    # Combine facts
    combined_facts = s1.get("facts", []) + s2.get("facts", [])
    
    # Deduplicate facts if too many
    if len(combined_facts) > 20:
        plan = plpy.prepare(
            "SELECT ai_deduplicate_facts($1::jsonb) as deduped",
            ["jsonb"]
        )
        result = plpy.execute(plan, [json.dumps(combined_facts)])
        if result and result[0]["deduped"]:
            combined_facts = json.loads(result[0]["deduped"])
    
    # Combine samples (keep diverse set)
    combined_samples = s1.get("samples", []) + s2.get("samples", [])
    if len(combined_samples) > 10:
        # Simple diversity: take evenly spaced samples
        step = len(combined_samples) // 10
        combined_samples = combined_samples[::step][:10]
    
    # Combine statistics
    stats1 = s1.get("stats", {})
    stats2 = s2.get("stats", {})
    
    combined_stats = {
        "row_count": stats1.get("row_count", 0) + stats2.get("row_count", 0),
        "total_chars": stats1.get("total_chars", 0) + stats2.get("total_chars", 0),
        "min_length": min(
            stats1.get("min_length", float('inf')),
            stats2.get("min_length", float('inf'))
        ),
        "max_length": max(
            stats1.get("max_length", 0),
            stats2.get("max_length", 0)
        ),
        "combine_depth": max(
            stats1.get("combine_depth", 0),
            stats2.get("combine_depth", 0)
        ) + 1
    }
    
    # Merge metadata
    combined_metadata = {**s1.get("metadata", {}), **s2.get("metadata", {})}
    
    return json.dumps({
        "facts": combined_facts,
        "samples": combined_samples,
        "stats": combined_stats,
        "metadata": combined_metadata
    })
$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

-- Finalizer function to generate summary
CREATE OR REPLACE FUNCTION ai_summarize_finalize(
    state jsonb
) RETURNS text
AS $$
    import json
    
    if state is None:
        return None
    
    state_data = json.loads(state)
    
    # Check if we have any data
    if state_data.get("stats", {}).get("row_count", 0) == 0:
        return "No data to summarize"
    
    # Build summary prompt based on combine depth
    combine_depth = state_data.get("stats", {}).get("combine_depth", 0)
    
    if combine_depth == 0:
        prompt_template = "Create a concise summary of this data: Facts: {facts}, Row count: {row_count}, Average length: {avg_length}"
    elif combine_depth < 3:
        prompt_template = "Synthesize these key facts into a coherent summary: {facts}, Total rows: {row_count}, Length range: {min_length}-{max_length} chars"
    else:
        prompt_template = "Identify major patterns from these aggregated facts: {facts}, Dataset size: {row_count} rows"
    
    # Calculate average length
    stats = state_data.get("stats", {})
    avg_length = stats.get("total_chars", 0) // max(stats.get("row_count", 1), 1)
    
    # Format facts for prompt
    facts = state_data.get("facts", [])[:10]  # Limit to top 10 facts
    facts_str = "; ".join(facts) if facts else "No specific facts extracted"
    
    # Build prompt
    prompt = prompt_template.format(
        facts=facts_str,
        row_count=stats.get("row_count", 0),
        avg_length=avg_length,
        min_length=stats.get("min_length", 0),
        max_length=stats.get("max_length", 0)
    )
    
    # Add metadata context if available
    metadata = state_data.get("metadata", {})
    if metadata:
        meta_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
        prompt += f". Context: {meta_str}"
    
    # Generate summary using steadytext
    plan = plpy.prepare("SELECT steadytext_generate($1) as summary", ["text"])
    result = plpy.execute(plan, [prompt])
    
    if result and result[0]["summary"]:
        return result[0]["summary"]
    return "Unable to generate summary"
$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

-- Serialization functions for distributed aggregation
CREATE OR REPLACE FUNCTION ai_summarize_serialize(
    state jsonb
) RETURNS bytea
AS $$
    import json
    
    if state is None:
        return None
    
    # Convert to JSON string then to bytes
    json_str = state if isinstance(state, str) else json.dumps(state)
    return json_str.encode('utf-8')
$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

CREATE OR REPLACE FUNCTION ai_summarize_deserialize(
    state bytea
) RETURNS jsonb
AS $$
    import json
    
    if state is None:
        return None
    
    # Convert bytes to string then parse JSON
    json_str = state.decode('utf-8')
    return json_str
$$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;

-- Create the main aggregate
CREATE AGGREGATE ai_summarize(text, jsonb) (
    SFUNC = ai_summarize_accumulate,
    STYPE = jsonb,
    FINALFUNC = ai_summarize_finalize,
    COMBINEFUNC = ai_summarize_combine,
    SERIALFUNC = ai_summarize_serialize,
    DESERIALFUNC = ai_summarize_deserialize,
    PARALLEL = SAFE
);

-- Create partial aggregate for TimescaleDB continuous aggregates
CREATE AGGREGATE ai_summarize_partial(text, jsonb) (
    SFUNC = ai_summarize_accumulate,
    STYPE = jsonb,
    COMBINEFUNC = ai_summarize_combine,
    SERIALFUNC = ai_summarize_serialize,
    DESERIALFUNC = ai_summarize_deserialize,
    PARALLEL = SAFE
);

-- Helper function to combine partial states for final aggregation
CREATE OR REPLACE FUNCTION ai_summarize_combine_states(
    state1 jsonb,
    partial_state jsonb
) RETURNS jsonb
AS $$
BEGIN
    -- Simply use the combine function
    RETURN ai_summarize_combine(state1, partial_state);
END;
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;

-- Create final aggregate that works on partial results
CREATE AGGREGATE ai_summarize_final(jsonb) (
    SFUNC = ai_summarize_combine_states,
    STYPE = jsonb,
    FINALFUNC = ai_summarize_finalize,
    PARALLEL = SAFE
);

-- Convenience function for single-value summarization
CREATE OR REPLACE FUNCTION ai_summarize_text(
    input_text text,
    metadata jsonb DEFAULT '{}'::jsonb
) RETURNS text
AS $$
    SELECT ai_summarize_finalize(
        ai_summarize_accumulate(NULL::jsonb, input_text, metadata)
    );
$$ LANGUAGE sql IMMUTABLE PARALLEL SAFE;

-- Add helpful comments
COMMENT ON AGGREGATE ai_summarize(text, jsonb) IS 
'AI-powered text summarization aggregate that handles non-transitivity through structured fact extraction';

COMMENT ON AGGREGATE ai_summarize_partial(text, jsonb) IS 
'Partial aggregate for use with TimescaleDB continuous aggregates';

COMMENT ON AGGREGATE ai_summarize_final(jsonb) IS 
'Final aggregate for completing partial aggregations from continuous aggregates';

COMMENT ON FUNCTION ai_extract_facts(text, integer) IS 
'Extract structured facts from text using SteadyText JSON generation';

COMMENT ON FUNCTION ai_deduplicate_facts(jsonb, float) IS 
'Deduplicate facts based on semantic similarity using embeddings';

-- AIDEV-NOTE: This migration adds AI summarization with the following features:
-- 1. Structured fact extraction to mitigate non-transitivity
-- 2. Semantic deduplication using embeddings
-- 3. Statistical tracking (row counts, character lengths)
-- 4. Sample preservation for context
-- 5. Combine depth tracking for adaptive prompts
-- 6. Full TimescaleDB continuous aggregate support
-- 7. Serialization for distributed aggregation
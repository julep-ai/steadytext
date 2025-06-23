#!/bin/bash
# Test script for CLI eos_string, streaming, and verbosity functionality

echo "=== Test 1: Default generation (now streams by default, quiet by default) ==="
echo "Hello world" | python -m steadytext.cli.main

echo -e "\n=== Test 2: With custom eos_string (streams by default) ==="
python -m steadytext.cli.main generate "Generate text until END appears" --eos-string "END"

echo -e "\n=== Test 3: With verbose flag (should show logs) ==="
python -m steadytext.cli.main generate --verbose "Generate with logs"

echo -e "\n=== Test 4: Wait mode with custom eos_string (no streaming) ==="
python -m steadytext.cli.main generate "Wait until STOP" --wait --eos-string "STOP"

echo -e "\n=== Test 5: Streaming with custom eos_string (default behavior) ==="
python -m steadytext.cli.main generate "Stream until END" --eos-string "END" | head -c 100

echo -e "\n\n=== Test 6: JSON output with eos_string ==="
python -m steadytext.cli.main generate "JSON test" --json --eos-string "END" | python -m json.tool | head -20
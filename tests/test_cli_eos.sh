#!/bin/bash
# Test script for CLI eos_string and quiet functionality

echo "=== Test 1: Default generation ==="
echo "Hello world" | python -m steadytext.cli.main

echo -e "\n=== Test 2: With custom eos_string ==="
python -m steadytext.cli.main generate "Generate text until END appears" --eos-string "END"

echo -e "\n=== Test 3: With quiet flag (should have no logs) ==="
python -m steadytext.cli.main generate --quiet "Generate without logs"

echo -e "\n=== Test 4: Streaming with custom eos_string ==="
python -m steadytext.cli.main generate "Stream until STOP" --stream --eos-string "STOP" | head -c 100

echo -e "\n\n=== Test 5: JSON output with eos_string ==="
python -m steadytext.cli.main generate "JSON test" --json --eos-string "END" | python -m json.tool | head -20
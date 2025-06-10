import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from steadytext.core.generator import _deterministic_fallback_generate
from steadytext.utils import GENERATION_MAX_NEW_TOKENS # For length check

print("Testing fallback generator...")

output1 = _deterministic_fallback_generate("hello world")
output2 = _deterministic_fallback_generate("hello world")
output3 = _deterministic_fallback_generate("another prompt")
output_empty = _deterministic_fallback_generate("")
output_whitespace = _deterministic_fallback_generate("   ")

print(f"Fallback for 'hello world': {output1[:100]}...")
print(f"Fallback for 'another prompt': {output3[:100]}...")
print(f"Fallback for empty prompt: {output_empty[:100]}...")


assert output1 == output2, "Fallback for same prompt should be identical"
print("SUCCESS: Fallback for same prompt is identical.")

assert output1 != output3, "Fallback for different prompts should be different"
print("SUCCESS: Fallback for different prompts are different.")

len_output1 = len(output1.split())
assert (GENERATION_MAX_NEW_TOKENS - 10) <= len_output1 <= (GENERATION_MAX_NEW_TOKENS + 10),     f"Fallback word count ({len_output1}) not in range [{GENERATION_MAX_NEW_TOKENS-10}, {GENERATION_MAX_NEW_TOKENS+10}]"
print(f"SUCCESS: Fallback word count ({len_output1}) is in the expected range.")

assert output_empty == _deterministic_fallback_generate("   "), "Fallback for empty and whitespace-only should be identical due to placeholder hashing"
print("SUCCESS: Fallback for empty and whitespace-only prompt is identical.")

print("Fallback generator test completed successfully.")

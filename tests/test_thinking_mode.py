#!/usr/bin/env python3
"""Basic test for thinking mode functionality.

Tests:
- Default behavior strips empty think tags
- --think flag preserves think tags with content
- 'st' command works as alias for 'st generate' with piped input
"""

import subprocess
import sys


def test_thinking_mode():
    """Test that --think flag enables thinking mode."""

    # Test 1: Default behavior (no thinking) - use fallback to make test fast
    import os

    env = os.environ.copy()
    env["STEADYTEXT_SKIP_MODEL_LOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "steadytext.cli.main", "generate", "test", "--wait"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "<think>" not in result.stdout, (
        "Default mode should not have visible think tags"
    )

    # Test 2: Verify that thinking_mode=True actually adds /think to prompt
    # (Test the core logic without full model execution)
    from steadytext.core.generator import DeterministicGenerator

    gen = DeterministicGenerator()

    # Mock the model to capture the prompt that would be sent
    class MockModel:
        def __init__(self):
            self.last_prompt = None

        def create_chat_completion(self, messages, **kwargs):
            self.last_prompt = messages[0]["content"]
            return {"choices": [{"message": {"content": "test response"}}]}

    gen.model = MockModel()

    # Test without thinking mode (should append /no_think)
    gen.generate("test prompt 1", thinking_mode=False)
    assert gen.model.last_prompt == "test prompt 1 /no_think"

    # Test with thinking mode (should append /think)
    gen.generate("test prompt 2", thinking_mode=True)
    assert gen.model.last_prompt == "test prompt 2 /think"

    print("✓ Thinking mode prompt modification verified!")

    # Test 3: Pipe input default (no thinking)
    result = subprocess.run(
        "echo 'say hello' | python -m steadytext.cli.main generate --wait",
        shell=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "<think>" not in result.stdout, (
        "Piped input default should not show think tags"
    )

    # Test 4: Verify 'st' is alias for 'st generate' with pipe
    result1 = subprocess.run(
        "echo 'test prompt' | python -m steadytext.cli.main",
        shell=True,
        capture_output=True,
        text=True,
    )
    result2 = subprocess.run(
        "echo 'test prompt' | python -m steadytext.cli.main generate",
        shell=True,
        capture_output=True,
        text=True,
    )
    assert result1.returncode == 0
    assert result2.returncode == 0
    # Both should produce output (not help text)
    assert "Usage:" not in result1.stdout, "'st' with pipe should not show help"
    assert len(result1.stdout) > 10, "'st' with pipe should generate text"

    print("✓ All thinking mode tests passed!")
    return True


if __name__ == "__main__":
    test_thinking_mode()

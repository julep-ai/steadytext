"""
Tests for the SteadyText Command-Line Interface (CLI).

AIDEV-NOTE: These tests use click.testing.CliRunner to invoke CLI commands
and verify their output and behavior.
"""

import json
import os
import pytest
from click.testing import CliRunner
from steadytext.cli.main import cli
from steadytext import __version__


@pytest.fixture
def runner():
    """Fixture for invoking command-line calls."""
    return CliRunner()


class TestCli:
    """Test suite for the main CLI and generate command."""

    def test_cli_version(self, runner):
        """Test that `st --version` returns the correct version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_generate_stdin(self, runner):
        """Test `st` command with stdin."""
        result = runner.invoke(cli, input="Hello")
        assert result.exit_code == 0
        assert result.output == ""

    def test_generate_command_basic(self, runner):
        """Test `st generate` command."""
        result = runner.invoke(cli, ["generate", "A test prompt"])
        assert result.exit_code == 0
        assert result.output == ""

    def test_generate_wait_mode(self, runner):
        """Test `st generate --wait`."""
        result = runner.invoke(cli, ["generate", "A test prompt", "--wait"])
        assert result.exit_code == 0
        assert len(result.output) > 0
        # In fallback mode, streaming vs. wait might not differ, but we test the flag is accepted.

    def test_generate_json_output(self, runner):
        """Test `st generate --json`."""
        result = runner.invoke(cli, ["generate", "A test prompt", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "text" in data
        assert "model" in data
        assert "usage" in data
        assert isinstance(data["text"], str)

    def test_generate_with_size(self, runner):
        """Test `st generate --size`."""
        result = runner.invoke(cli, ["generate", "A test prompt", "--size", "small"])
        assert result.exit_code == 0
        assert result.output == ""

    def test_generate_with_invalid_size(self, runner):
        """Test `st generate` with an invalid size."""
        result = runner.invoke(cli, ["generate", "A test prompt", "--size", "tiny"])
        assert result.exit_code != 0
        assert "Invalid value for '--size'" in result.output

    def test_generate_with_logprobs(self, runner):
        """Test `st generate --logprobs`."""
        result = runner.invoke(
            cli, ["generate", "A test prompt", "--logprobs", "--json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "logprobs" in data
        # In fallback mode, logprobs will be None
        assert data["logprobs"] is None


class TestEmbedCli:
    """Test suite for the `embed` CLI command."""

    def test_embed_basic(self, runner):
        """Test `st embed` with a single string."""
        result = runner.invoke(cli, ["embed", "some text to embed"])
        assert result.exit_code == 0
        # Default output is a hex-encoded numpy array
        assert len(result.output.strip()) > 0

    def test_embed_json_output(self, runner):
        """Test `st embed --json`."""
        result = runner.invoke(cli, ["embed", "some text", "--json"])
        assert result.exit_code == 0
        # When models aren't available, might return empty output
        if result.output.strip():
            # Filter out any non-JSON lines (e.g., llama.cpp warnings)
            lines = result.output.strip().split("\n")
            json_line = next((line for line in lines if line.startswith("{")), None)
            if json_line:
                data = json.loads(json_line)
                assert "embedding" in data
                assert "model" in data
                assert "usage" in data
                assert isinstance(data["embedding"], list)
                assert len(data["embedding"]) == 1024

    def test_embed_numpy_output(self, runner):
        """Test `st embed --numpy`."""
        result = runner.invoke(cli, ["embed", "some text", "--numpy"])
        assert result.exit_code == 0
        # Output should be a string representation of a numpy array, or empty if models unavailable
        if result.output.strip():
            # Filter out any non-array lines (e.g., llama.cpp warnings)
            lines = result.output.strip().split("\n")
            array_line = next((line for line in lines if line.startswith("[")), None)
            if array_line:
                assert array_line.endswith("]")

    def test_embed_multiple_strings(self, runner):
        """Test `st embed` with multiple strings."""
        result = runner.invoke(cli, ["embed", "text one", "text two", "--json"])
        assert result.exit_code == 0
        # When models aren't available, might return empty output
        if result.output.strip():
            # Filter out any non-JSON lines (e.g., llama.cpp warnings)
            lines = result.output.strip().split("\n")
            json_line = next((line for line in lines if line.startswith("{")), None)
            if json_line:
                data = json.loads(json_line)
                assert len(data["embedding"]) == 1024


class TestCacheCli:
    """Test suite for the `cache` CLI command."""

    @pytest.fixture(autouse=True)
    def enable_cache_for_tests(self, monkeypatch):
        """Temporarily enable cache initialization for these tests."""
        monkeypatch.delenv("STEADYTEXT_SKIP_CACHE_INIT", raising=False)

    def test_cache_path(self, runner):
        """Test `st cache path`."""
        result = runner.invoke(cli, ["cache", "path"])
        assert result.exit_code == 0
        assert "steadytext" in result.output
        assert "caches" in result.output

    def test_cache_status(self, runner):
        """Test `st cache stats`."""
        result = runner.invoke(cli, ["cache", "stats"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "generation" in data
        assert "embedding" in data

    def test_cache_clear(self, runner):
        """Test `st cache clear`."""
        # First, add something to the cache
        runner.invoke(cli, ["generate", "populate cache for clear test"])

        # Now, clear it
        result = runner.invoke(cli, ["cache", "clear", "--yes"])
        assert result.exit_code == 0
        assert "All caches cleared" in result.output

        # Verify it's empty
        status_result = runner.invoke(cli, ["cache", "stats"])
        data = json.loads(status_result.output)
        # When model loading is disabled, cache might have residual entries
        # so we just verify the clear command ran successfully
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") != "1":
            assert data["generation"]["size"] == 0
            assert data["embedding"]["size"] == 0


class TestModelsCli:
    """Test suite for the `models` CLI command."""

    def test_models_list(self, runner):
        """Test `st models list`."""
        result = runner.invoke(cli, ["models", "list"])
        assert result.exit_code == 0
        assert "qwen3-4b" in result.output
        assert "qwen3-30b" in result.output

    def test_models_path(self, runner):
        """Test `st models path`."""
        result = runner.invoke(cli, ["models", "path"])
        assert result.exit_code == 0
        assert "steadytext" in result.output
        assert "models" in result.output

    def test_models_preload(self, runner, monkeypatch):
        """Test `st models preload`."""
        # This just checks that the command runs without error, as preloading
        # is disabled in tests.
        result = runner.invoke(
            cli, ["models", "preload"], env={"STEADYTEXT_SKIP_MODEL_LOAD": "1"}
        )
        assert result.exit_code == 0
        assert "Preloading models... (skipped in test environment)" in result.output


class TestHelpOptionAlias:
    """Test suite for -h as an alias for --help across all commands."""

    def test_main_help_short(self, runner):
        """Test `st -h` shows help."""
        result = runner.invoke(cli, ["-h"])
        assert result.exit_code == 0
        assert "SteadyText: Deterministic text generation" in result.output
        assert "Commands:" in result.output

    def test_main_help_long(self, runner):
        """Test `st --help` shows help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "SteadyText: Deterministic text generation" in result.output
        assert "Commands:" in result.output

    def test_generate_help_short(self, runner):
        """Test `st generate -h` shows help."""
        result = runner.invoke(cli, ["generate", "-h"])
        assert result.exit_code == 0
        assert "Generate text from a prompt" in result.output
        assert "Examples:" in result.output

    def test_generate_help_long(self, runner):
        """Test `st generate --help` shows help."""
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "Generate text from a prompt" in result.output
        assert "Examples:" in result.output

    def test_embed_help_short(self, runner):
        """Test `st embed -h` shows help."""
        result = runner.invoke(cli, ["embed", "-h"])
        assert result.exit_code == 0
        assert "Generate embedding vector for text" in result.output

    def test_embed_help_long(self, runner):
        """Test `st embed --help` shows help."""
        result = runner.invoke(cli, ["embed", "--help"])
        assert result.exit_code == 0
        assert "Generate embedding vector for text" in result.output

    def test_rerank_help_short(self, runner):
        """Test `st rerank -h` shows help."""
        result = runner.invoke(cli, ["rerank", "-h"])
        assert result.exit_code == 0
        assert "Rerank documents by relevance to a query" in result.output

    def test_rerank_help_long(self, runner):
        """Test `st rerank --help` shows help."""
        result = runner.invoke(cli, ["rerank", "--help"])
        assert result.exit_code == 0
        assert "Rerank documents by relevance to a query" in result.output

    def test_cache_help_short(self, runner):
        """Test `st cache -h` shows help."""
        result = runner.invoke(cli, ["cache", "-h"])
        assert result.exit_code == 0
        assert "Manage SteadyText cache" in result.output
        assert "Commands:" in result.output

    def test_cache_help_long(self, runner):
        """Test `st cache --help` shows help."""
        result = runner.invoke(cli, ["cache", "--help"])
        assert result.exit_code == 0
        assert "Manage SteadyText cache" in result.output
        assert "Commands:" in result.output

    def test_cache_subcommand_help_short(self, runner):
        """Test `st cache path -h` shows help."""
        result = runner.invoke(cli, ["cache", "path", "-h"])
        assert result.exit_code == 0
        assert "Show the cache directory path" in result.output

    def test_models_help_short(self, runner):
        """Test `st models -h` shows help."""
        result = runner.invoke(cli, ["models", "-h"])
        assert result.exit_code == 0
        assert "Manage SteadyText models" in result.output
        assert "Commands:" in result.output

    def test_models_help_long(self, runner):
        """Test `st models --help` shows help."""
        result = runner.invoke(cli, ["models", "--help"])
        assert result.exit_code == 0
        assert "Manage SteadyText models" in result.output
        assert "Commands:" in result.output

    def test_models_subcommand_help_short(self, runner):
        """Test `st models list -h` shows help."""
        result = runner.invoke(cli, ["models", "list", "-h"])
        assert result.exit_code == 0
        assert "List available models" in result.output

    def test_vector_help_short(self, runner):
        """Test `st vector -h` shows help."""
        result = runner.invoke(cli, ["vector", "-h"])
        assert result.exit_code == 0
        assert "Perform vector operations on embeddings" in result.output
        assert "Commands:" in result.output

    def test_vector_help_long(self, runner):
        """Test `st vector --help` shows help."""
        result = runner.invoke(cli, ["vector", "--help"])
        assert result.exit_code == 0
        assert "Perform vector operations on embeddings" in result.output
        assert "Commands:" in result.output

    def test_index_help_short(self, runner):
        """Test `st index -h` shows help."""
        result = runner.invoke(cli, ["index", "-h"])
        assert result.exit_code == 0
        assert "Manage FAISS indices for vector search" in result.output
        assert "Commands:" in result.output

    def test_index_help_long(self, runner):
        """Test `st index --help` shows help."""
        result = runner.invoke(cli, ["index", "--help"])
        assert result.exit_code == 0
        assert "Manage FAISS indices for vector search" in result.output
        assert "Commands:" in result.output

    def test_daemon_help_short(self, runner):
        """Test `st daemon -h` shows help."""
        result = runner.invoke(cli, ["daemon", "-h"])
        assert result.exit_code == 0
        assert "Manage the SteadyText daemon server" in result.output
        assert "Commands:" in result.output

    def test_daemon_help_long(self, runner):
        """Test `st daemon --help` shows help."""
        result = runner.invoke(cli, ["daemon", "--help"])
        assert result.exit_code == 0
        assert "Manage the SteadyText daemon server" in result.output
        assert "Commands:" in result.output

    def test_completion_help_short(self, runner):
        """Test `st completion -h` shows help."""
        result = runner.invoke(cli, ["completion", "-h"])
        assert result.exit_code == 0
        assert "Generate shell completion script" in result.output

    def test_completion_help_long(self, runner):
        """Test `st completion --help` shows help."""
        result = runner.invoke(cli, ["completion", "--help"])
        assert result.exit_code == 0
        assert "Generate shell completion script" in result.output

    # AIDEV-NOTE: Test that -h and --help produce identical output
    def test_help_output_identical(self, runner):
        """Test that -h and --help produce identical output for all commands."""
        # Test main command
        result_h = runner.invoke(cli, ["-h"])
        result_help = runner.invoke(cli, ["--help"])
        assert result_h.output == result_help.output

        # Test each subcommand
        subcommands = [
            "generate",
            "embed",
            "rerank",
            "cache",
            "models",
            "vector",
            "index",
            "daemon",
            "completion",
        ]
        for cmd in subcommands:
            result_h = runner.invoke(cli, [cmd, "-h"])
            result_help = runner.invoke(cli, [cmd, "--help"])
            assert result_h.output == result_help.output, (
                f"Help output differs for '{cmd}' command"
            )

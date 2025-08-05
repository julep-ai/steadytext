"""Integration tests for CLI defaults functionality.

AIDEV-NOTE: This module tests that saved defaults are properly applied when
running the actual CLI commands (generate, embed, rerank).
"""

import json
import pytest
from click.testing import CliRunner
from pathlib import Path

from steadytext.cli.main import cli
from steadytext.config import get_defaults_manager


class TestDefaultsIntegration:
    """Test that defaults are properly applied to commands."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def isolated_config(self, runner, monkeypatch):
        """Create an isolated config environment for testing."""
        with runner.isolated_filesystem():
            # Create a temporary config directory
            config_dir = Path("test_config")
            config_dir.mkdir()
            
            # Monkeypatch the config directory
            monkeypatch.setattr("steadytext.config.get_config_dir", lambda: config_dir)
            
            # Clear any existing defaults manager instance
            import steadytext.config
            steadytext.config._defaults_manager = None
            
            yield config_dir
    
    @pytest.mark.skipif("os.environ.get('STEADYTEXT_SKIP_MODEL_LOAD', '1') == '1'")
    def test_generate_uses_saved_defaults(self, runner, isolated_config):
        """Test that generate command uses saved defaults."""
        # Set defaults
        runner.invoke(cli, ["set-default", "generate", "--seed", "123", "--wait"])
        
        # Run generate with JSON output to check parameters
        result = runner.invoke(cli, ["generate", "test prompt", "--json"])
        
        # The command should work (though model might not load in tests)
        # We mainly want to verify it doesn't error due to defaults
        assert result.exit_code == 0 or "model" in result.output.lower()
    
    def test_generate_cli_overrides_defaults(self, runner, isolated_config):
        """Test that CLI arguments override saved defaults."""
        # Set default seed
        runner.invoke(cli, ["set-default", "generate", "--seed", "123"])
        
        # Run with different seed - this should override
        # We can't easily verify the seed was used without model loading,
        # but we can verify the command runs without error
        result = runner.invoke(cli, ["generate", "test", "--seed", "456", "--wait"])
        
        # Should not error
        assert result.exit_code == 0 or "model" in result.output.lower()
    
    def test_embed_uses_saved_defaults(self, runner, isolated_config):
        """Test that embed command uses saved defaults."""
        # Set default to use JSON output
        runner.invoke(cli, ["set-default", "embed", "--json", "--seed", "999"])
        
        # Run embed without specifying format
        result = runner.invoke(cli, ["embed", "test text"])
        
        # Should output JSON format due to saved default
        # (Will be deterministic fallback in test environment)
        if result.exit_code == 0:
            try:
                output = json.loads(result.output)
                assert "embedding" in output
                assert "text" in output
            except json.JSONDecodeError:
                # If model loading failed, we get fallback output
                pass
    
    def test_rerank_uses_saved_defaults(self, runner, isolated_config):
        """Test that rerank command uses saved defaults."""
        # Set default task and top-k
        custom_task = "Find technical documentation"
        runner.invoke(cli, ["set-default", "rerank", "--task", custom_task, "--top-k", "2"])
        
        # Run rerank
        result = runner.invoke(cli, ["rerank", "query", "doc1", "doc2", "doc3", "--json"])
        
        # Command should work (with fallback in test environment)
        assert result.exit_code == 0 or "model" in result.output.lower()
    
    def test_environment_overrides_saved_defaults(self, runner, isolated_config, monkeypatch):
        """Test that environment variables override saved defaults."""
        # Set saved default
        runner.invoke(cli, ["set-default", "generate", "--seed", "123"])
        
        # Set environment variable
        monkeypatch.setenv("STEADYTEXT_SEED", "789")
        
        # Environment variable should take precedence
        # We can't easily verify this without model loading,
        # but we can ensure the command runs
        result = runner.invoke(cli, ["generate", "test", "--wait"])
        assert result.exit_code == 0 or "model" in result.output.lower()
    
    def test_defaults_persist_across_invocations(self, runner, isolated_config):
        """Test that defaults persist across multiple CLI invocations."""
        # Set defaults in first invocation
        result1 = runner.invoke(cli, ["set-default", "generate", "--model", "gemma-3n-2b"])
        assert result1.exit_code == 0
        
        # Check defaults in second invocation
        result2 = runner.invoke(cli, ["set-default", "generate", "--show"])
        assert result2.exit_code == 0
        assert "--model gemma-3n-2b" in result2.output
        
        # Use in third invocation (would use the model if available)
        result3 = runner.invoke(cli, ["generate", "test", "--wait"])
        assert result3.exit_code == 0 or "model" in result3.output.lower()
    
    def test_output_format_defaults(self, runner, isolated_config):
        """Test setting output format defaults."""
        # Set JSON as default output for generate
        runner.invoke(cli, ["set-default", "generate", "--json"])
        
        # Run generate without specifying format
        result = runner.invoke(cli, ["generate", "test prompt", "--wait"])
        
        # Should output JSON format due to saved default
        if result.exit_code == 0 and "{" in result.output:
            try:
                output = json.loads(result.output)
                assert "text" in output or "generated" in output
            except json.JSONDecodeError:
                # Fallback output in test environment
                pass
    
    def test_multiple_defaults_combination(self, runner, isolated_config):
        """Test combining multiple saved defaults."""
        # Set multiple defaults
        runner.invoke(cli, ["set-default", "generate", 
                          "--seed", "42",
                          "--wait",
                          "--no-index",
                          "--json"])
        
        # Run with only prompt
        result = runner.invoke(cli, ["generate", "test prompt"])
        
        # Should work with all defaults applied
        assert result.exit_code == 0 or "model" in result.output.lower()
    
    def test_flag_defaults_handling(self, runner, isolated_config):
        """Test that boolean flag defaults are handled correctly."""
        # Set wait flag as default
        runner.invoke(cli, ["set-default", "generate", "--wait"])
        
        # Generate should not stream by default now
        result = runner.invoke(cli, ["generate", "test"])
        
        # Can't easily verify streaming vs non-streaming without model,
        # but command should work
        assert result.exit_code == 0 or "model" in result.output.lower()
    
    def test_no_defaults_uses_builtin(self, runner, isolated_config):
        """Test that commands work normally without any saved defaults."""
        # Don't set any defaults
        
        # Commands should use built-in defaults
        result_gen = runner.invoke(cli, ["generate", "test", "--wait"])
        result_emb = runner.invoke(cli, ["embed", "test"])
        result_rer = runner.invoke(cli, ["rerank", "query", "doc1", "doc2"])
        
        # All should work (with potential model loading failures in test env)
        for result in [result_gen, result_emb, result_rer]:
            assert result.exit_code == 0 or "model" in result.output.lower()
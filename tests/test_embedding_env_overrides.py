"""
Tests for automatic remote embedding overrides driven by environment variables.
"""

import json
import numpy as np
from click.testing import CliRunner

import steadytext
from steadytext.cli.main import cli
from pg_steadytext.python.daemon_connector import SteadyTextConnector


def _json_from_cli_output(output: str):
    """Return first JSON payload from CLI output, ignoring warnings/noise."""
    for line in output.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            return json.loads(stripped)
    raise AssertionError("No JSON content found in CLI output")


class TestEmbeddingEnvOverrides:
    """Validate EMBEDDING_OPENAI_* automatic routing behavior."""

    def test_embed_env_override_applies_remote_model(self, monkeypatch):
        """Library embed() should honor EMBEDDING_OPENAI_* variables."""
        monkeypatch.setenv("EMBEDDING_OPENAI_BASE_URL", "https://example.com")
        monkeypatch.setenv("EMBEDDING_OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("STEADYTEXT_DISABLE_DAEMON", "1")

        captured = {}

        def fake_core_embed(
            text_input, seed=42, model=None, unsafe_mode=False, mode=None
        ):
            captured["params"] = {
                "text_input": text_input,
                "seed": seed,
                "model": model,
                "unsafe_mode": unsafe_mode,
                "mode": mode,
            }
            return np.zeros(1024, dtype=np.float32)

        monkeypatch.setattr("steadytext.core.embedder.core_embed", fake_core_embed)
        monkeypatch.setattr(steadytext, "core_embed", fake_core_embed, raising=False)
        result = steadytext.embed("hello world", seed=123)

        assert result.shape == (1024,)
        params = captured["params"]
        assert params["model"] == "openai:text-embedding-3-small"
        assert params["unsafe_mode"] is True
        assert params["seed"] == 123

    def test_embed_env_override_uses_custom_model_env(self, monkeypatch):
        """Override should honor EMBEDDING_OPENAI_MODEL value when provided."""
        monkeypatch.setenv("EMBEDDING_OPENAI_BASE_URL", "https://example.com")
        monkeypatch.setenv("EMBEDDING_OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("EMBEDDING_OPENAI_MODEL", "text-embedding-3-large")
        monkeypatch.setenv("STEADYTEXT_DISABLE_DAEMON", "1")

        captured = {}

        def fake_core_embed(
            text_input, seed=42, model=None, unsafe_mode=False, mode=None
        ):
            captured["model"] = model
            captured["unsafe_mode"] = unsafe_mode
            return np.zeros(1024, dtype=np.float32)

        monkeypatch.setattr("steadytext.core.embedder.core_embed", fake_core_embed)
        monkeypatch.setattr(steadytext, "core_embed", fake_core_embed, raising=False)

        steadytext.embed("custom model example")

        assert captured["model"] == "openai:text-embedding-3-large"
        assert captured["unsafe_mode"] is True

    def test_embed_env_override_respects_explicit_model(self, monkeypatch):
        """Explicit model argument must bypass environment override."""
        monkeypatch.setenv("EMBEDDING_OPENAI_BASE_URL", "https://example.com")
        monkeypatch.setenv("EMBEDDING_OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("STEADYTEXT_DISABLE_DAEMON", "1")

        captured = {}

        def fake_core_embed(
            text_input, seed=42, model=None, unsafe_mode=False, mode=None
        ):
            captured["model"] = model
            captured["unsafe_mode"] = unsafe_mode
            return np.ones(1024, dtype=np.float32)

        monkeypatch.setattr("steadytext.core.embedder.core_embed", fake_core_embed)
        monkeypatch.setattr(steadytext, "core_embed", fake_core_embed, raising=False)

        steadytext.embed(
            "explicit model",
            model="voyageai:voyage-3-lite",
            unsafe_mode=True,
            seed=7,
        )

        assert captured["model"] == "voyageai:voyage-3-lite"
        assert captured["unsafe_mode"] is True

    def test_embed_cli_env_override_uses_remote_model(self, monkeypatch):
        """CLI embed command should surface remote override in output."""
        monkeypatch.setenv("EMBEDDING_OPENAI_BASE_URL", "https://example.com")
        monkeypatch.setenv("EMBEDDING_OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("STEADYTEXT_DISABLE_DAEMON", "1")

        def fake_create_embedding(
            text_input, seed=42, model=None, unsafe_mode=False, mode=None
        ):
            assert model == "openai:text-embedding-3-small"
            assert unsafe_mode is True
            return np.full(1024, 0.5, dtype=np.float32)

        monkeypatch.setattr(
            "steadytext.core.embedder.core_embed", fake_create_embedding
        )
        monkeypatch.setattr(
            steadytext, "core_embed", fake_create_embedding, raising=False
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["embed", "sample", "--json"])

        assert result.exit_code == 0
        payload = _json_from_cli_output(result.output)
        assert payload["model"] == "openai:text-embedding-3-small"
        assert payload["dimension"] == 1024

    def test_embed_cli_respects_custom_model_env(self, monkeypatch):
        """CLI JSON output reflects EMBEDDING_OPENAI_MODEL override."""
        monkeypatch.setenv("EMBEDDING_OPENAI_BASE_URL", "https://example.com")
        monkeypatch.setenv("EMBEDDING_OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("EMBEDDING_OPENAI_MODEL", "text-embedding-3-large")
        monkeypatch.setenv("STEADYTEXT_DISABLE_DAEMON", "1")

        def fake_create_embedding(
            text_input, seed=42, model=None, unsafe_mode=False, mode=None
        ):
            assert model == "openai:text-embedding-3-large"
            assert unsafe_mode is True
            return np.ones(1024, dtype=np.float32)

        monkeypatch.setattr(
            "steadytext.core.embedder.core_embed", fake_create_embedding
        )
        monkeypatch.setattr(
            steadytext, "core_embed", fake_create_embedding, raising=False
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["embed", "sample", "--json"])

        assert result.exit_code == 0
        payload = _json_from_cli_output(result.output)
        assert payload["model"] == "openai:text-embedding-3-large"


class TestPgExtensionEnvOverrides:
    """Ensure the PostgreSQL extension honours remote embedding overrides."""

    def test_connector_env_override_applies_remote_model(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_OPENAI_BASE_URL", "https://example.com")
        monkeypatch.setenv("EMBEDDING_OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("STEADYTEXT_DISABLE_DAEMON", "1")
        monkeypatch.setattr(
            "pg_steadytext.python.daemon_connector.STEADYTEXT_AVAILABLE",
            True,
            raising=False,
        )

        captured = {}

        def fake_embed(text, seed=42, model=None, unsafe_mode=False, mode=None):
            captured["params"] = {
                "text": text,
                "seed": seed,
                "model": model,
                "unsafe_mode": unsafe_mode,
            }
            return np.zeros(1024, dtype=np.float32)

        monkeypatch.setattr(
            "pg_steadytext.python.daemon_connector.embed",
            fake_embed,
        )

        connector = SteadyTextConnector(auto_start=False)
        result = connector.embed("pg text", seed=99)

        assert result.shape == (1024,)
        params = captured["params"]
        assert params["model"] == "openai:text-embedding-3-small"
        assert params["unsafe_mode"] is True
        assert params["seed"] == 99

    def test_connector_env_override_custom_model(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_OPENAI_BASE_URL", "https://example.com")
        monkeypatch.setenv("EMBEDDING_OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("EMBEDDING_OPENAI_MODEL", "text-embedding-3-large")
        monkeypatch.setattr(
            "pg_steadytext.python.daemon_connector.STEADYTEXT_AVAILABLE",
            True,
            raising=False,
        )

        captured = {}

        def fake_embed(text, seed=42, model=None, unsafe_mode=False, mode=None):
            captured["model"] = model
            captured["unsafe_mode"] = unsafe_mode
            return np.ones(1024, dtype=np.float32)

        monkeypatch.setattr(
            "pg_steadytext.python.daemon_connector.embed",
            fake_embed,
        )

        connector = SteadyTextConnector(auto_start=False)
        connector.embed("pg text custom")

        assert captured["model"] == "openai:text-embedding-3-large"
        assert captured["unsafe_mode"] is True

    def test_connector_respects_explicit_model(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_OPENAI_BASE_URL", "https://example.com")
        monkeypatch.setenv("EMBEDDING_OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(
            "pg_steadytext.python.daemon_connector.STEADYTEXT_AVAILABLE",
            True,
            raising=False,
        )

        captured = {}

        def fake_embed(text, seed=42, model=None, unsafe_mode=False, mode=None):
            captured["model"] = model
            captured["unsafe_mode"] = unsafe_mode
            return np.full(1024, 0.3, dtype=np.float32)

        monkeypatch.setattr(
            "pg_steadytext.python.daemon_connector.embed",
            fake_embed,
        )

        connector = SteadyTextConnector(auto_start=False)
        connector.embed(
            "pg explicit",
            model="voyageai:voyage-3-lite",
            unsafe_mode=True,
            seed=11,
        )

        assert captured["model"] == "voyageai:voyage-3-lite"
        assert captured["unsafe_mode"] is True

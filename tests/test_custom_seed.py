import json
import numpy as np
from click.testing import CliRunner

from steadytext import generate, embed
from steadytext.cli.main import cli


def test_generate_default_seed():
    output1 = generate("test")
    output2 = generate("test", seed=42)
    assert output1 == output2


def test_embed_default_seed():
    output1 = embed("test")
    output2 = embed("test", seed=42)
    assert np.array_equal(output1, output2)


def test_generate_custom_seed_determinism():
    output1 = generate("test", seed=123)
    output2 = generate("test", seed=123)
    assert output1 == output2


def test_embed_custom_seed_determinism():
    output1 = embed("test", seed=123)
    output2 = embed("test", seed=123)
    assert np.array_equal(output1, output2)


def test_generate_different_seeds():
    output1 = generate("test", seed=123)
    output2 = generate("test", seed=456)
    assert output1 != output2


def test_embed_different_seeds():
    output1 = embed("test", seed=123)
    output2 = embed("test", seed=456)
    assert not np.array_equal(output1, output2)


def test_cli_generate_default_seed():
    runner = CliRunner()
    result1 = runner.invoke(cli, ["generate", "test"])
    result2 = runner.invoke(cli, ["generate", "test", "--seed", "42"])
    assert result1.stdout == result2.stdout


def test_cli_generate_custom_seed():
    runner = CliRunner()
    result1 = runner.invoke(cli, ["generate", "test", "--seed", "123"])
    result2 = runner.invoke(cli, ["generate", "test", "--seed", "123"])
    assert result1.stdout == result2.stdout


def test_cli_generate_different_seeds():
    runner = CliRunner()
    result1 = runner.invoke(cli, ["generate", "test", "--seed", "123"])
    result2 = runner.invoke(cli, ["generate", "test", "--seed", "456"])
    assert result1.stdout != result2.stdout


def test_cli_embed_default_seed():
    runner = CliRunner()
    result1 = runner.invoke(cli, ["embed", "test", "--json"])
    result2 = runner.invoke(cli, ["embed", "test", "--json", "--seed", "42"])
    # Parse JSON and compare without time_taken
    json1 = json.loads(result1.stdout)
    json2 = json.loads(result2.stdout)
    json1.pop("time_taken", None)
    json2.pop("time_taken", None)
    assert json1 == json2


def test_cli_embed_custom_seed():
    runner = CliRunner()
    result1 = runner.invoke(cli, ["embed", "test", "--json", "--seed", "123"])
    result2 = runner.invoke(cli, ["embed", "test", "--json", "--seed", "123"])
    # Parse JSON and compare without time_taken
    json1 = json.loads(result1.stdout)
    json2 = json.loads(result2.stdout)
    json1.pop("time_taken", None)
    json2.pop("time_taken", None)
    assert json1 == json2


def test_cli_embed_different_seeds():
    runner = CliRunner()
    result1 = runner.invoke(cli, ["embed", "test", "--json", "--seed", "123"])
    result2 = runner.invoke(cli, ["embed", "test", "--json", "--seed", "456"])
    # Parse JSON and compare embeddings
    json1 = json.loads(result1.stdout)
    json2 = json.loads(result2.stdout)
    assert json1["embedding"] != json2["embedding"]

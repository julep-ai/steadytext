# Todo: Custom Seed Implementation

This document outlines the steps to add custom seed support to `steadytext`.

## 1. Core API and Logic

-   [ ] **Modify `steadytext/__init__.py`:**
    -   [ ] Change `generate` signature to:
        ```python
        def generate(
            prompt: str,
            max_new_tokens: int = 512,
            eos_string: Optional[str] = None,
            seed: int = 42
        ) -> str:
        ```
    -   [ ] Change `generate_iter` signature to:
        ```python
        def generate_iter(
            prompt: str,
            max_new_tokens: int = 512,
            eos_string: Optional[str] = None,
            seed: int = 42
        ) -> Generator[str, None, None]:
        ```
    -   [ ] Change `embed` signature to:
        ```python
        def embed(
            text: str,
            seed: int = 42
        ) -> np.ndarray:
        ```
    -   [ ] Pass the `seed` parameter to the underlying `core_generate`, `core_generate_iter`, and `core_embed` calls.

-   [ ] **Modify `steadytext/core/generator.py`:**
    -   [ ] Add a `seed: int` parameter to `core_generate`.
    -   [ ] Add a `seed: int` parameter to `core_generate_iter`.
    -   [ ] In both functions, call `init_deterministic(seed)` at the beginning.
    -   [ ] Update the fallback generation logic to use the provided `seed`.

-   [ ] **Modify `steadytext/core/embedder.py`:**
    -   [ ] Add a `seed: int` parameter to `core_embed`.
    -   [ ] Call `init_deterministic(seed)` at the beginning of the function.
    -   [ ] Update the fallback embedding logic to use the provided `seed`.

-   [ ] **Modify `steadytext/utils.py`:**
    -   [ ] Change `init_deterministic()` to `init_deterministic(seed: int)`.
    -   [ ] Replace `DEFAULT_SEED` with the `seed` parameter inside `init_deterministic`.

## 2. CLI Integration

-   [ ] **Modify `steadytext/cli/commands/generate.py`:**
    -   [ ] Add the following option to the `generate` command function:
        ```python
        @click.option(
            "--seed",
            type=int,
            default=42,
            help="Seed for deterministic generation.",
            show_default=True,
        )
        ```
    -   [ ] Pass the `seed` value to the `generate` or `generate_iter` API call.

-   [ ] **Modify `steadytext/cli/commands/embed.py`:**
    -   [ ] Add the following option to the `embed` command function:
        ```python
        @click.option(
            "--seed",
            type=int,
            default=42,
            help="Seed for deterministic embedding.",
            show_default=True,
        )
        ```
    -   [ ] Pass the `seed` value to the `embed` API call.

## 3. Testing

-   [ ] **Create `tests/test_custom_seed.py`:**
    -   [ ] Add necessary imports (`pytest`, `generate`, `embed`, `CliRunner`, etc.).

-   [ ] **API Tests in `tests/test_custom_seed.py`:**
    -   [ ] **Test Default Seed:**
        ```python
        def test_generate_default_seed():
            output1 = generate("test")
            output2 = generate("test", seed=42)
            assert output1 == output2

        def test_embed_default_seed():
            output1 = embed("test")
            output2 = embed("test", seed=42)
            assert np.array_equal(output1, output2)
        ```
    -   [ ] **Test Custom Seed Determinism:**
        ```python
        def test_generate_custom_seed_determinism():
            output1 = generate("test", seed=123)
            output2 = generate("test", seed=123)
            assert output1 == output2

        def test_embed_custom_seed_determinism():
            output1 = embed("test", seed=123)
            output2 = embed("test", seed=123)
            assert np.array_equal(output1, output2)
        ```
    -   [ ] **Test Different Seeds:**
        ```python
        def test_generate_different_seeds():
            output1 = generate("test", seed=123)
            output2 = generate("test", seed=456)
            assert output1 != output2

        def test_embed_different_seeds():
            output1 = embed("test", seed=123)
            output2 = embed("test", seed=456)
            assert not np.array_equal(output1, output2)
        ```

-   [ ] **CLI Tests in `tests/test_custom_seed.py`:**
    -   [ ] **Test Default Seed (CLI):**
        ```python
        def test_cli_generate_default_seed():
            runner = CliRunner()
            result1 = runner.invoke(cli, ["generate", "test"])
            result2 = runner.invoke(cli, ["generate", "test", "--seed", "42"])
            assert result1.stdout == result2.stdout
        ```
    -   [ ] **Test Custom Seed (CLI):**
        ```python
        def test_cli_generate_custom_seed():
            runner = CliRunner()
            result1 = runner.invoke(cli, ["generate", "test", "--seed", "123"])
            result2 = runner.invoke(cli, ["generate", "test", "--seed", "123"])
            assert result1.stdout == result2.stdout
        ```
    -   [ ] **Test Different Seeds (CLI):**
        ```python
        def test_cli_generate_different_seeds():
            runner = CliRunner()
            result1 = runner.invoke(cli, ["generate", "test", "--seed", "123"])
            result2 = runner.invoke(cli, ["generate", "test", "--seed", "456"])
            assert result1.stdout != result2.stdout
        ```
    -   [ ] Repeat the CLI tests for the `embed` command.

## 4. Documentation

-   [ ] **Update `CLAUDE.md`:**
    -   [ ] Briefly mention the new `seed` parameter in the relevant API and CLI sections.
-   [ ] **Update `docs/api/generation.md` and `docs/api/embedding.md`:**
    -   [ ] Add the `seed` parameter to the function signatures and descriptions.
-   [ ] **Update `docs/api/cli.md`:**
    -   [ ] Add the `--seed` option to the `st generate` and `st embed` command documentation.
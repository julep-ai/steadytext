"""
Tests for verifying true streaming behavior in both the library and CLI.
"""

import time
import subprocess
import fcntl
import os
import steadytext


def is_streaming(iterator, max_wait_seconds=1.0, min_chunks=3):
    """
    Consumes an iterator and checks if it yields multiple chunks
    in a streaming fashion.

    Args:
        iterator: The iterator to check (e.g., a generator).
        max_wait_seconds: The maximum time to wait for the first chunk.
        min_chunks: The minimum number of chunks to receive to be considered streaming.

    Returns:
        True if the iterator is streaming, False otherwise.
    """
    chunk_times = []
    start_time = time.time()

    try:
        # Get the first chunk
        first_chunk = next(iterator)
        chunk_times.append(time.time())
        if not first_chunk:
            return False  # Empty first chunk

        # Get subsequent chunks
        for _ in range(min_chunks - 1):
            chunk = next(iterator)
            chunk_times.append(time.time())
            if not chunk:
                break  # End of stream
    except StopIteration:
        # Not enough chunks to be considered streaming
        pass
    except Exception as e:
        print(f"Error during streaming check: {e}")
        return False

    if len(chunk_times) < min_chunks:
        return False

    # Check that the time between chunks is small
    for i in range(1, len(chunk_times)):
        inter_chunk_time = chunk_times[i] - chunk_times[i - 1]
        # A very small delay is expected between chunks in a real stream
        if inter_chunk_time > max_wait_seconds:
            return False

    # Check that the total time is not excessively long
    total_time = time.time() - start_time
    if total_time > max_wait_seconds * min_chunks * 2:  # Heuristic
        return False

    return True


class TestStreamingBehavior:
    """Verify that both library and CLI calls stream results."""

    def test_library_is_streaming(self):
        """Verify steadytext.generate_iter() streams tokens."""
        # Skip if model loading is disabled
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            return

        prompt = "Generate a short story about a robot who discovers music."
        token_iterator = steadytext.generate_iter(prompt, max_new_tokens=50)

        assert is_streaming(token_iterator), "The library function is not streaming."

    def test_cli_is_streaming(self):
        """Verify the CLI streams output by default."""
        # Skip if model loading is disabled
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            return

        prompt = "Generate a short story about a robot who discovers music."

        # We use subprocess here instead of CliRunner because CliRunner waits
        # for the command to complete, which defeats the purpose of a streaming test.
        process = subprocess.Popen(
            ["st", "generate", prompt, "--max-new-tokens", "50"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )

        # Make stdout non-blocking
        fd = process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        output_chunks = []
        start_time = time.time()
        max_wait = 5.0  # 5 seconds timeout for the whole process

        while time.time() - start_time < max_wait:
            try:
                chunk = process.stdout.read()
                if chunk:
                    output_chunks.append(chunk)
                elif process.poll() is not None:
                    # Process finished
                    break
            except (IOError, TypeError):
                # No output ready to be read
                time.sleep(0.01)  # Small sleep to avoid busy-waiting

        process.terminate()
        process.wait()

        # The output should arrive in multiple chunks, not all at once.
        assert len(output_chunks) > 1, (
            f"CLI output was not received in multiple chunks. Got {len(output_chunks)} chunk(s)."
        )

        full_output = b"".join(output_chunks).decode("utf-8")
        assert len(full_output) > 0, "CLI produced no output."

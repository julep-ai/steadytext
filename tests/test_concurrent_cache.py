# AIDEV-NOTE: Concurrent access tests for SQLite-based frecency cache
# Tests multiple processes and threads accessing cache simultaneously
import os
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pytest

# Disable model loading during cache tests
os.environ["STEADYTEXT_SKIP_MODEL_LOAD"] = "1"

from steadytext.sqlite_cache_backend import SQLiteDiskBackedFrecencyCache


@pytest.mark.concurrent
class TestConcurrentCache:
    """Test concurrent access to SQLite-based frecency cache."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_concurrent_threads_basic(self, temp_cache_dir):
        """Test basic concurrent access from multiple threads."""
        cache = SQLiteDiskBackedFrecencyCache(
            capacity=100, cache_name="thread_test", cache_dir=temp_cache_dir
        )

        results = {}
        errors = []

        def worker(thread_id):
            try:
                # Each thread writes its own keys
                for i in range(10):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"value_{thread_id}_{i}"
                    cache.set(key, value)

                # Verify writes
                for i in range(10):
                    key = f"thread_{thread_id}_key_{i}"
                    result = cache.get(key)
                    results[key] = result

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors in threads: {errors}"

        # Verify all data is correct
        for thread_id in range(5):
            for i in range(10):
                key = f"thread_{thread_id}_key_{i}"
                expected = f"value_{thread_id}_{i}"
                assert results[key] == expected, f"Incorrect value for {key}"

    def test_concurrent_read_write_threads(self, temp_cache_dir):
        """Test concurrent reading while writing from multiple threads."""
        cache = SQLiteDiskBackedFrecencyCache(
            capacity=200, cache_name="read_write_test", cache_dir=temp_cache_dir
        )

        # Pre-populate cache
        for i in range(50):
            cache.set(f"initial_key_{i}", f"initial_value_{i}")

        write_count = 0
        read_count = 0
        errors: list = []
        lock = threading.Lock()

        def writer():
            nonlocal write_count, errors
            try:
                for i in range(100, 150):  # Write 50 new entries
                    cache.set(f"writer_key_{i}", f"writer_value_{i}")
                    with lock:
                        write_count += 1  # type: ignore
                    time.sleep(0.001)  # Small delay to interleave operations
            except Exception as e:
                errors.append(f"Writer error: {e}")

        def reader():
            nonlocal read_count, errors
            try:
                for _ in range(100):  # Read existing entries
                    key = f"initial_key_{_ % 50}"
                    result = cache.get(key)
                    if result is not None:
                        with lock:
                            read_count += 1  # type: ignore
                    time.sleep(0.001)  # Small delay to interleave operations
            except Exception as e:
                errors.append(f"Reader error: {e}")

        # Start reader and writer threads
        threads = []
        threads.append(threading.Thread(target=writer))
        threads.append(threading.Thread(target=reader))
        threads.append(threading.Thread(target=reader))  # Multiple readers

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert write_count == 50, f"Expected 50 writes, got {write_count}"
        assert read_count > 150, f"Expected >150 reads, got {read_count}"

        # Verify written data exists
        for i in range(100, 150):
            key = f"writer_key_{i}"
            value = cache.get(key)
            assert value == f"writer_value_{i}", (
                f"Missing or incorrect writer data for {key}"
            )

    def _process_worker(self, args):
        """Worker function for process-based testing."""
        cache_dir, cache_name, process_id, operation_count = args

        try:
            # Add small delay to prevent all processes from initializing at once
            time.sleep(0.01 * process_id)

            # Each process creates its own cache instance
            cache = SQLiteDiskBackedFrecencyCache(
                capacity=100, cache_name=cache_name, cache_dir=Path(cache_dir)
            )

            results = []

            # Write operations with retry logic for concurrent access
            for i in range(operation_count):
                key = f"proc_{process_id}_key_{i}"
                value = f"proc_{process_id}_value_{i}"

                # Retry a few times in case of transient errors
                for retry in range(3):
                    try:
                        cache.set(key, value)
                        results.append(("set", key, value))
                        break
                    except Exception:
                        if retry == 2:  # Last attempt
                            raise
                        time.sleep(0.05)  # Brief pause before retry

            # Read operations to verify with retry logic
            for i in range(operation_count):
                key = f"proc_{process_id}_key_{i}"
                for retry in range(3):
                    try:
                        value = cache.get(key)
                        results.append(("get", key, value))
                        break
                    except Exception:
                        if retry == 2:  # Last attempt
                            raise
                        time.sleep(0.05)  # Brief pause before retry

            # Ensure all writes are persisted to disk before process exits
            cache.sync()

            # Add small delay before process exit to ensure sync completes
            time.sleep(0.1)

            return process_id, results, None

        except Exception as e:
            import traceback

            return process_id, [], f"{str(e)}\n{traceback.format_exc()}"

    def test_concurrent_processes(self, temp_cache_dir):
        """Test concurrent access from multiple processes."""
        cache_name = "process_test"
        num_processes = 4
        operations_per_process = 25

        # Prepare arguments for each process
        args_list = [
            (str(temp_cache_dir), cache_name, proc_id, operations_per_process)
            for proc_id in range(num_processes)
        ]

        # Run processes concurrently
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(self._process_worker, args) for args in args_list
            ]
            results = [future.result() for future in as_completed(futures)]

        # Analyze results
        all_errors = []
        successful_writes = set()
        successful_reads = {}

        for process_id, operations, error in results:
            if error:
                all_errors.append(f"Process {process_id}: {error}")

            for op_type, key, value in operations:
                if op_type == "set":
                    successful_writes.add(key)
                elif op_type == "get" and value is not None:
                    successful_reads[key] = value

        # Verify no errors
        assert len(all_errors) == 0, f"Process errors: {all_errors}"

        # Verify data consistency - create new cache instance to read from disk
        verification_cache = SQLiteDiskBackedFrecencyCache(
            capacity=200,
            cache_name=cache_name,
            max_size_mb=10.0,  # Large enough to avoid eviction
            cache_dir=temp_cache_dir,
        )

        # Check that we can read back some of the written data
        successful_verifications = 0
        for process_id in range(num_processes):
            for i in range(operations_per_process):
                key = f"proc_{process_id}_key_{i}"
                expected_value = f"proc_{process_id}_value_{i}"
                actual_value = verification_cache.get(key)
                if actual_value == expected_value:
                    successful_verifications += 1

        # Verify that a significant portion of data persisted
        total_expected = num_processes * operations_per_process
        assert successful_verifications > total_expected * 0.8, (
            f"Only {successful_verifications}/{total_expected} writes persisted correctly"
        )

    def test_eviction_under_concurrent_load(self, temp_cache_dir):
        """Test size-based eviction with concurrent access."""
        # Create cache with small size limit
        cache = SQLiteDiskBackedFrecencyCache(
            capacity=1000,
            cache_name="eviction_test",
            max_size_mb=0.1,  # 100KB limit
            cache_dir=temp_cache_dir,
        )

        # Value size of approximately 1KB each
        large_value = "x" * 1000
        total_writes = 0
        errors: list = []
        lock = threading.Lock()

        def writer(writer_id):
            nonlocal total_writes, errors
            try:
                for i in range(50):  # Each writer adds 50KB
                    key = f"writer_{writer_id}_key_{i}"
                    cache.set(key, large_value + f"_w{writer_id}_i{i}")
                    with lock:
                        total_writes += 1  # type: ignore
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(f"Writer {writer_id} error: {e}")

        # Start multiple writers that will exceed size limit
        threads = []
        for i in range(4):  # 4 writers * 50KB = 200KB total (exceeds 100KB limit)
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify no errors during concurrent operations
        assert len(errors) == 0, f"Errors during concurrent eviction: {errors}"
        assert total_writes == 200, f"Expected 200 writes, got {total_writes}"

        # Verify cache size is under limit
        stats = cache.get_stats()
        assert stats["total_size_bytes"] <= cache.max_size_bytes, (
            f"Cache size {stats['total_size_bytes']} exceeds limit {cache.max_size_bytes}"
        )

        # Verify some entries were evicted
        remaining_entries = stats["entry_count"]
        assert remaining_entries < 200, (
            f"Expected eviction, but {remaining_entries} entries remain"
        )
        assert remaining_entries > 50, (
            f"Too many entries evicted: only {remaining_entries} remain"
        )

    def test_database_locking_behavior(self, temp_cache_dir):
        """Test that database locking works correctly under high contention."""
        cache = SQLiteDiskBackedFrecencyCache(
            capacity=100, cache_name="locking_test", cache_dir=temp_cache_dir
        )

        operation_count = 0
        errors: list = []
        lock = threading.Lock()

        def high_contention_worker(worker_id):
            nonlocal operation_count, errors
            try:
                # Rapid operations on overlapping keys to create contention
                for i in range(20):
                    # Mix of shared and unique keys
                    shared_key = f"shared_key_{i % 5}"
                    unique_key = f"worker_{worker_id}_key_{i}"

                    cache.set(shared_key, f"shared_value_w{worker_id}_i{i}")
                    cache.set(unique_key, f"unique_value_w{worker_id}_i{i}")

                    cache.get(shared_key)
                    cache.get(unique_key)

                    with lock:
                        operation_count += 4  # type: ignore  # 2 sets + 2 gets

            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")

        # Start many threads for high contention
        threads = []
        for i in range(8):
            t = threading.Thread(target=high_contention_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify all operations completed without errors
        assert len(errors) == 0, f"Locking errors: {errors}"
        assert operation_count == 8 * 80, (
            f"Expected 640 operations, got {operation_count}"
        )

        # Verify final state is consistent
        stats = cache.get_stats()
        assert stats["entry_count"] > 0, "Cache should not be empty after operations"

    def _write_process_data(self, args):
        """Helper function for writing process data."""
        cache_dir, cache_name = args
        try:
            cache = SQLiteDiskBackedFrecencyCache(
                capacity=100, cache_name=cache_name, cache_dir=Path(cache_dir)
            )

            for i in range(20):
                cache.set(f"persist_key_{i}", f"persist_value_{i}")

            cache.sync()  # Ensure data is written
            return True
        except Exception as e:
            return str(e)

    def _read_process_data(self, args):
        """Helper function for reading process data."""
        cache_dir, cache_name = args
        try:
            cache = SQLiteDiskBackedFrecencyCache(
                capacity=100, cache_name=cache_name, cache_dir=Path(cache_dir)
            )

            results = []
            for i in range(20):
                key = f"persist_key_{i}"
                value = cache.get(key)
                results.append((key, value))

            return results
        except Exception as e:
            return str(e)

    def test_cache_persistence_across_processes(self, temp_cache_dir):
        """Test that data persists correctly across different processes."""
        cache_name = "persistence_test"

        # Execute processes sequentially
        with ProcessPoolExecutor(max_workers=1) as executor:
            # Write data
            write_future = executor.submit(
                self._write_process_data, (str(temp_cache_dir), cache_name)
            )
            write_result = write_future.result()
            assert write_result is True, f"Write process failed: {write_result}"

            # Read data
            read_future = executor.submit(
                self._read_process_data, (str(temp_cache_dir), cache_name)
            )
            read_results = read_future.result()
            assert isinstance(read_results, list), (
                f"Read process failed: {read_results}"
            )

        # Verify all data was read correctly
        for i, (key, value) in enumerate(read_results):
            expected_key = f"persist_key_{i}"
            expected_value = f"persist_value_{i}"
            assert key == expected_key, f"Key mismatch: {key} != {expected_key}"
            assert value == expected_value, (
                f"Value mismatch for {key}: {value} != {expected_value}"
            )

    def test_stress_test_mixed_operations(self, temp_cache_dir):
        """Stress test with mixed read/write operations from multiple threads."""
        cache = SQLiteDiskBackedFrecencyCache(
            capacity=500,
            cache_name="stress_test",
            max_size_mb=1.0,  # 1MB limit
            cache_dir=temp_cache_dir,
        )

        # Pre-populate cache
        for i in range(100):
            cache.set(f"initial_{i}", f"initial_value_{i}")

        operations_completed = 0
        errors: list = []
        lock = threading.Lock()

        def mixed_operations_worker(worker_id):
            nonlocal operations_completed, errors
            try:
                for i in range(100):
                    # Mix of operations
                    if i % 4 == 0:
                        # Write new entry
                        key = f"worker_{worker_id}_new_{i}"
                        cache.set(key, f"new_value_{worker_id}_{i}")
                    elif i % 4 == 1:
                        # Read existing entry
                        key = f"initial_{i % 100}"
                        cache.get(key)
                    elif i % 4 == 2:
                        # Update existing entry
                        key = f"initial_{i % 100}"
                        cache.set(key, f"updated_by_{worker_id}_{i}")
                    else:
                        # Read own entry
                        key = f"worker_{worker_id}_new_{i - (i % 4)}"
                        cache.get(key)

                    with lock:
                        operations_completed += 1  # type: ignore

            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")

        # Run stress test with multiple workers
        threads = []
        for i in range(6):
            t = threading.Thread(target=mixed_operations_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify stress test completed successfully
        assert len(errors) == 0, f"Stress test errors: {errors}"
        assert operations_completed == 6 * 100, (
            f"Expected 600 operations, got {operations_completed}"
        )

        # Verify cache is still functional
        stats = cache.get_stats()
        assert stats["entry_count"] > 0, "Cache should not be empty after stress test"
        assert stats["total_size_bytes"] <= cache.max_size_bytes, (
            "Cache size should be within limits"
        )

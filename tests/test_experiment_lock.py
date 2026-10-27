from __future__ import annotations

import json
import os
import select
import threading
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from core.experiment_lock import (
    ExperimentDirectoryLock,
    ExperimentIdentity,
    ExperimentProtocolLock,
    ExperimentSchedulerLock,
    ProviderRateLimitTimeout,
    ProviderRequestLimiter,
    assert_distinct_mutable_roots,
)


class ExperimentDirectoryLockTest(unittest.TestCase):
    def test_second_writer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExperimentDirectoryLock(directory, "first"):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with ExperimentDirectoryLock(directory, "second"):
                        pass

    def test_different_protocols_can_hold_scheduler_locks_concurrently(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExperimentProtocolLock(directory, "deepseek-v3-2"):
                with ExperimentProtocolLock(directory, "claude-sonnet"):
                    pass

    def test_same_protocol_cannot_start_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExperimentProtocolLock(directory, "deepseek-v3-2"):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with ExperimentProtocolLock(directory, "deepseek-v3-2"):
                        pass


class ExperimentSchedulerIdentityTest(unittest.TestCase):
    def test_identity_uses_experiment_protocol_and_batch_fields(self):
        identity = ExperimentIdentity.from_config(
            {
                "experiment_id": "v3-supplement",
                "protocol_id": "formal-h1",
                "batch_id": "branch-20261022",
            }
        )
        self.assertEqual(
            identity.as_dict(),
            {
                "experiment_id": "v3-supplement",
                "protocol_id": "formal-h1",
                "batch_id": "branch-20261022",
            },
        )
        self.assertIn("experiment_id=v3-supplement", identity.lock_filename)
        self.assertIn("batch_id=branch-20261022", identity.lock_filename)

    def test_different_experiments_can_be_active_at_the_same_time(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExperimentSchedulerLock(
                directory,
                experiment_id="main-v3-2",
                protocol_id="deepseek-v3-2",
            ):
                with ExperimentSchedulerLock(
                    directory,
                    experiment_id="branch-new-v3",
                    protocol_id="branch-new-v3-supplement",
                ):
                    pass

    def test_same_experiment_is_rejected_even_with_a_different_result_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_dir = Path(directory) / "locks"
            with ExperimentSchedulerLock(
                lock_dir,
                experiment_id="main-v3-2",
                protocol_id="deepseek-v3-2",
                result_dir=Path(directory) / "run-a",
            ):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with ExperimentSchedulerLock(
                        lock_dir,
                        experiment_id="main-v3-2",
                        protocol_id="deepseek-v3-2",
                        result_dir=Path(directory) / "run-b",
                    ):
                        pass

    def test_legacy_protocol_lock_remains_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExperimentProtocolLock(directory, "running-v3-2"):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with ExperimentSchedulerLock(
                        directory,
                        experiment_id="future-runner",
                        protocol_id="running-v3-2",
                        legacy_protocol_lock=True,
                    ):
                        pass

    def test_distinct_result_directories_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assert_distinct_mutable_roots(root / "run-a", root / "run-b")
            with self.assertRaisesRegex(ValueError, "overlap"):
                assert_distinct_mutable_roots(root / "run-a", root / "run-a")
            with self.assertRaisesRegex(ValueError, "overlap"):
                assert_distinct_mutable_roots(root / "run-a", root / "run-a" / "nested")

    def test_result_owner_registry_blocks_same_experiment_in_new_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            first = results / "run-a"
            second = results / "run-b"
            with ExperimentDirectoryLock(first, "same-experiment"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "another result directory",
                ):
                    with ExperimentDirectoryLock(second, "same-experiment"):
                        pass


class ProviderRequestLimiterTest(unittest.TestCase):
    def test_same_provider_limits_overlapping_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            limiter = ProviderRequestLimiter(directory, "same-provider", poll_seconds=0.01)
            entered = threading.Event()
            release = threading.Event()
            rejected = []

            def hold_slot():
                with limiter.request_slot():
                    entered.set()
                    release.wait(timeout=2)

            thread = threading.Thread(target=hold_slot)
            thread.start()
            self.assertTrue(entered.wait(timeout=1))
            try:
                with self.assertRaises(ProviderRateLimitTimeout):
                    with limiter.request_slot(timeout_seconds=0.05):
                        rejected.append("entered")
            finally:
                release.set()
                thread.join(timeout=2)
            self.assertEqual(rejected, [])

    def test_different_providers_do_not_block_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            deepseek = ProviderRequestLimiter(directory, "siliconflow-v3-2")
            branch = ProviderRequestLimiter(directory, "zhizengzeng-v4")
            with deepseek.request_slot():
                started = time.monotonic()
                with branch.request_slot(timeout_seconds=0.2):
                    pass
                self.assertLess(time.monotonic() - started, 0.2)


if __name__ == "__main__":
    unittest.main()

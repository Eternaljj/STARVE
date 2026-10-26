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
from h1_deepseek_v4_pro_extended_20261022.request_ledger import (
    DeepseekRequestLedger,
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

    def test_two_provider_free_processes_run_different_experiments_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = r"""
import json
import os
import sys
from pathlib import Path
from core.experiment_lock import ExperimentDirectoryLock, ExperimentSchedulerLock
from h1_deepseek_v4_pro_extended_20261022.request_ledger import DeepseekRequestLedger

result = Path(os.environ["RUN_DIR"])
with ExperimentSchedulerLock(
    Path(os.environ["LOCK_DIR"]),
    experiment_id=os.environ["EXPERIMENT_ID"],
    protocol_id=os.environ["PROTOCOL_ID"],
):
    with ExperimentDirectoryLock(result, os.environ["EXPERIMENT_ID"]):
        ledger = DeepseekRequestLedger(result / "ledger" / "requests.jsonl")
        ledger.start_request(
            request_index=1,
            model="provider-free",
            request_parameters={"experiment": os.environ["EXPERIMENT_ID"]},
            estimated_input_tokens=1,
            message_count=1,
            message_bytes=1,
            output_token_limit=1,
            budget_before={"requests": 0},
        )
        print(json.dumps({"ready": os.environ["EXPERIMENT_ID"]}), flush=True)
        sys.stdin.readline()
"""
            processes = []
            for experiment_id, protocol_id in (
                ("experiment-a", "protocol-a"),
                ("experiment-b", "protocol-b"),
            ):
                run_dir = root / "results" / experiment_id
                environment = dict(os.environ)
                environment.update({
                    "RUN_DIR": str(run_dir),
                    "LOCK_DIR": str(root / "results" / ".scheduler_locks"),
                    "EXPERIMENT_ID": experiment_id,
                    "PROTOCOL_ID": protocol_id,
                })
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-c", script],
                        cwd=Path(__file__).resolve().parents[1],
                        env=environment,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            try:
                ready = []
                for process in processes:
                    readable, _, _ = select.select(
                        [process.stdout],
                        [],
                        [],
                        5,
                    )
                    self.assertTrue(readable, "provider-free child did not become ready")
                    ready.append(json.loads(process.stdout.readline())["ready"])
                self.assertEqual(ready, ["experiment-a", "experiment-b"])
                self.assertTrue(all(process.poll() is None for process in processes))
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.stdin.write("\n")
                        process.stdin.flush()
                for process in processes:
                    process.communicate(timeout=5)
            for experiment_id in ("experiment-a", "experiment-b"):
                ledger = root / "results" / experiment_id / "ledger" / "requests.jsonl"
                self.assertTrue(ledger.is_file())
                rows = [json.loads(line) for line in ledger.read_text().splitlines()]
                self.assertEqual(rows[0]["request_parameters"]["experiment"], experiment_id)


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

    def test_ledgers_in_distinct_result_directories_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = DeepseekRequestLedger(root / "run-a" / "ledger" / "requests.jsonl")
            second = DeepseekRequestLedger(root / "run-b" / "ledger" / "requests.jsonl")
            budget = {
                "requests": 0,
                "usage": {"total_tokens": 0},
            }
            for ledger in (first, second):
                request_id = ledger.start_request(
                    request_index=1,
                    model="deepseek-v4-pro",
                    request_parameters={"max_tokens": 10},
                    estimated_input_tokens=3,
                    message_count=1,
                    message_bytes=10,
                    output_token_limit=10,
                    budget_before=budget,
                )
                self.assertEqual(request_id, "dsreq-0001")
            first.commit_usage(
                request_id="dsreq-0001",
                raw_usage={
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
                normalized_usage={
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                    "reasoning_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cache_miss_tokens": 3,
                },
                response_model_id="deepseek-v4-pro",
                response_id="response-a",
                http_status=200,
                finish_reason="stop",
                response_body_bytes=100,
                elapsed_seconds=0.1,
                cumulative_usage={"total_tokens": 5},
            )
            self.assertEqual(first.snapshot()["requests_with_usage"], 1)
            self.assertEqual(second.snapshot()["requests_with_usage"], 0)
            self.assertEqual(second.snapshot()["unknown_request_ids"], ["dsreq-0001"])


if __name__ == "__main__":
    unittest.main()

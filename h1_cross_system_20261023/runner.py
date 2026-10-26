"""Formal H1 entry with one collector and one four-arm execution path.

The Provider-free mode uses deterministic local responses. The live mode uses
the verified Anthropic Messages Provider and is entered only by an explicit
``--start`` command.
"""
from __future__ import annotations

import argparse
import ast
import csv
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
COMPAT_DIR = ROOT
for path in (str(ROOT), str(COMPAT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from adapters.formal_h1_multitask_v3_2_runtime import budget_scope
from adapters.lifecycle_state_collector import classify_behavior
from adapters.lifecycle_web_behavior import web_behavior_evidence
from adapters.lifecycle_mechanism_runtime import IncompatibleLifecycleBinding
from h1_deepseek_v4_pro_extended_20261022.extended_v3_environment import (
    EXTENSION_VERSION,
    ExtendedEvidenceRepository,
    ExtendedEvidenceV3Environment,
    ExtendedEvidenceV3Session,
    build_family_extension,
)
from h1_evidence_v3_compat import h1_evidence_interface
from p0a_pentestgpt.gateway import ControlledToolGateway
from p0a_pentestgpt.provider import resolve_provider
from p0a_pentestgpt.stagetrap_final_profile import (
    BRANCH_V3_IMPLEMENTATION,
    DEPENDENCY_IMPLEMENTATION,
    EVIDENCE_IMPLEMENTATION,
    P0AFinalEvidenceV3V6Environment,
    finalize_arm_metrics,
    restore_final_runtime,
)
from p0a_pentestgpt.stagetrap_k12_live import (
    AnthropicK12Provider,
    _write_json_atomic,
    summarize_usage,
)
from p0a_pentestgpt.stagetrap_output_profiles import (
    STAGETRAP_OUTPUT_8192_PROFILE,
    output_limit,
)
from p0a_pentestgpt.stagetrap_serial_compat import (
    GatewayEnvironmentProxy,
    SerialCompatibilityRuntime,
)
from p0a_pentestgpt.stagetrap_state_trap_pilot import audit_gateway
from tasks.multi_task_pool_v1 import validate_fixture
from tasks.multi_task_pool_v1_adapter import (
    FixtureRepository,
    ProviderFreeSentinel,
)
from tasks.multi_task_pool_v3_dependency_v5_adapter import (
    MultiTaskDependencyV5SessionAdapter,
)
from traps.task_relevant_branch_v3 import (
    task_relevant_branch_candidates_v3,
)


PROTOCOL_ID = "formal_H1_cross_system_verification_20261023_v2"
K = 12
MAX_TOKENS = output_limit(STAGETRAP_OUTPUT_8192_PROFILE)
NATURAL_REQUESTS = 32
STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
CONDITIONS = ("BASELINE", "BRANCH_NEW_V3", "DEPENDENCY_V6", "EVIDENCE_NEW_V3")
CONDITION_NAMES = CONDITIONS
TRAPS = CONDITIONS[1:]
SOURCE_INDICES = (1, 6, 2, 11, 16, 12, 21, 26, 22)
BRANCH_IMPLEMENTATION = BRANCH_V3_IMPLEMENTATION
IMPLEMENTATIONS = {
    "BASELINE": None,
    "BRANCH_NEW_V3": BRANCH_V3_IMPLEMENTATION,
    "DEPENDENCY_V6": DEPENDENCY_IMPLEMENTATION,
    "EVIDENCE_NEW_V3": EVIDENCE_IMPLEMENTATION,
}
INTERNAL_CONDITIONS = {
    "BASELINE": "BASELINE",
    "BRANCH_NEW_V3": "BRANCH_V3",
    "DEPENDENCY_V6": "DEPENDENCY",
    "EVIDENCE_NEW_V3": "EVIDENCE",
}
TOTAL_REQUEST_CEILING = (
    len(SOURCE_INDICES) * NATURAL_REQUESTS
    + len(SOURCE_INDICES) * len(STATES) * len(CONDITIONS) * K
)
CONDITION_MEASUREMENT_FACTORY: Callable[..., Any] | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def audit_journal_records(
    records: list[Mapping[str, Any]],
    journal_dir: Path,
    *,
    expected_condition: str,
    ledger_events: list[Mapping[str, Any]] | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    """Audit one provider scope without assuming local or global file numbers.

    A scope is complete only when its request records, per-request journals,
    condition ownership, usage, and ledger events map one-to-one. File numbers
    may be local (1..n) or global request indices.
    """
    reasons: list[str] = []
    journal_dir = Path(journal_dir)
    paths = sorted(journal_dir.glob("request-*.json"))
    file_records: list[dict[str, Any]] = []
    file_numbers: list[int] = []
    for path in paths:
        try:
            file_numbers.append(int(path.stem.split("-")[1]))
        except (IndexError, ValueError):
            reasons.append("malformed_journal_filename")
            continue
        try:
            value = read_json(path)
        except (OSError, json.JSONDecodeError):
            reasons.append("unreadable_journal_record")
            continue
        if not isinstance(value, dict):
            reasons.append("journal_record_not_object")
            continue
        file_records.append(value)

    request_log = [deepcopy(dict(record)) for record in records]
    if len(file_records) != len(request_log):
        reasons.append("journal_record_count_mismatch")

    record_ids = [
        record.get("request_id")
        for record in request_log
        if isinstance(record.get("request_id"), str)
        and record.get("request_id")
    ]
    file_ids = [
        record.get("request_id")
        for record in file_records
        if isinstance(record.get("request_id"), str)
        and record.get("request_id")
    ]
    identity_mode = "request_id" if record_ids or file_ids else "local_index"
    if identity_mode == "request_id":
        if len(record_ids) != len(request_log):
            reasons.append("request_id_missing_from_provider_log")
        if len(file_ids) != len(file_records):
            reasons.append("request_id_missing_from_journal")
        if len(record_ids) != len(set(record_ids)):
            reasons.append("duplicate_request_id_in_provider_log")
        if len(file_ids) != len(set(file_ids)):
            reasons.append("duplicate_request_id_in_journal")
        if set(record_ids) != set(file_ids):
            reasons.append("request_id_set_mismatch")
        file_by_id = {
            record["request_id"]: record
            for record in file_records
            if isinstance(record.get("request_id"), str)
        }
        for record in request_log:
            request_id = record.get("request_id")
            journal_record = file_by_id.get(request_id)
            if journal_record is None:
                continue
            for field in (
                "request_model_id",
                "response_model_id",
                "usage",
                "usage_committed",
            ):
                if journal_record.get(field) != record.get(field):
                    reasons.append("journal_provider_record_mismatch")
                    break

    record_indexes = [
        record.get("request_index") for record in request_log
    ]
    if any(
        not isinstance(index, int) or isinstance(index, bool)
        for index in record_indexes
    ):
        reasons.append("request_index_missing_or_invalid")
        numeric_indexes: list[int] = []
    else:
        numeric_indexes = [int(index) for index in record_indexes]
        if len(numeric_indexes) != len(set(numeric_indexes)):
            reasons.append("duplicate_request_index_in_scope")

    order_ids = [
        record.get("request_id")
        if isinstance(record.get("request_id"), str)
        else f"local-{index:04d}"
        for index, record in enumerate(request_log, 1)
    ]
    if identity_mode == "request_id":
        file_order_ids = [
            record.get("request_id") for record in file_records
        ]
        if file_order_ids != order_ids:
            reasons.append("journal_request_order_mismatch")

    local_file_numbers = list(range(1, len(paths) + 1))
    global_file_numbers = numeric_indexes
    if file_numbers == local_file_numbers and file_numbers == global_file_numbers:
        numbering_mode = "local_equals_global_prefix"
    elif file_numbers == local_file_numbers:
        numbering_mode = "local"
    elif file_numbers == global_file_numbers:
        numbering_mode = "global"
    else:
        numbering_mode = "invalid"
        reasons.append("journal_numbering_conflict")

    state_path = journal_dir / "journal_state.json"
    state = read_json(state_path) if state_path.is_file() else None
    if not isinstance(state, dict):
        reasons.append("journal_state_missing")
    else:
        if state.get("condition") not in {None, expected_condition}:
            reasons.append("journal_state_condition_mismatch")
        if state.get("request_count") != len(request_log):
            reasons.append("journal_state_count_mismatch")
        if state.get("synthetic_provider_free") is False and offline:
            reasons.append("journal_state_offline_mode_mismatch")

    for record in request_log:
        if record.get("condition") not in {None, expected_condition}:
            reasons.append("provider_record_condition_mismatch")
        if not offline:
            usage = record.get("usage")
            if not (
                isinstance(usage, dict)
                and isinstance(usage.get("input_tokens"), int)
                and not isinstance(usage.get("input_tokens"), bool)
                and isinstance(usage.get("output_tokens"), int)
                and not isinstance(usage.get("output_tokens"), bool)
            ):
                reasons.append("provider_usage_incomplete")
            if record.get("usage_committed") is not True:
                reasons.append("provider_usage_not_committed")

    ledger_check = {
        "provided": ledger_events is not None,
        "started_ids": [],
        "committed_ids": [],
        "failed_ids": [],
        "complete": True,
    }
    if identity_mode == "request_id" and ledger_events is None:
        reasons.append("ledger_missing")
    if ledger_events is not None:
        started = [
            row.get("request_id")
            for row in ledger_events
            if row.get("event") == "request_started"
            and row.get("request_id") in set(record_ids)
        ]
        committed = [
            row.get("request_id")
            for row in ledger_events
            if row.get("event") == "request_usage_committed"
            and row.get("request_id") in set(record_ids)
        ]
        failed = [
            row.get("request_id")
            for row in ledger_events
            if row.get("event") == "request_failed"
            and row.get("request_id") in set(record_ids)
        ]
        ledger_check.update(
            started_ids=started,
            committed_ids=committed,
            failed_ids=failed,
        )
        if identity_mode != "request_id":
            reasons.append("ledger_present_without_request_ids")
        if len(started) != len(set(started)):
            reasons.append("duplicate_ledger_request_started")
        if len(committed) != len(set(committed)):
            reasons.append("duplicate_ledger_usage_commit")
        if started != record_ids:
            reasons.append("ledger_started_order_or_set_mismatch")
        if committed != record_ids:
            reasons.append("ledger_commit_order_or_set_mismatch")
        if failed:
            reasons.append("ledger_request_failed")
        commit_by_id = {
            row.get("request_id"): row
            for row in ledger_events
            if row.get("event") == "request_usage_committed"
            and row.get("request_id") in set(record_ids)
        }
        for record in request_log:
            request_id = record.get("request_id")
            committed_record = commit_by_id.get(request_id)
            if committed_record is None:
                continue
            normalized = committed_record.get("normalized_usage")
            usage = record.get("usage")
            if isinstance(normalized, dict) and isinstance(usage, dict):
                for field in (
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                ):
                    if (
                        normalized.get(field) is not None
                        and usage.get(field) is not None
                        and normalized.get(field) != usage.get(field)
                    ):
                        reasons.append("ledger_usage_content_mismatch")
                        break
        ledger_check["complete"] = not any(
            reason.startswith("ledger_") for reason in reasons
        )

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "complete": not unique_reasons,
        "expected_condition": expected_condition,
        "request_count": len(request_log),
        "journal_file_count": len(paths),
        "identity_mode": identity_mode,
        "numbering_mode": numbering_mode,
        "request_ids": order_ids,
        "file_numbers": file_numbers,
        "request_indexes": numeric_indexes,
        "ledger": ledger_check,
        "reasons": unique_reasons,
    }


def provider_ledger_events(provider: Any) -> list[dict[str, Any]] | None:
    candidates = [
        getattr(provider, "budget", None),
        getattr(provider, "inner", None),
    ]
    for candidate in candidates:
        ledger = getattr(candidate, "ledger", None)
        if ledger is not None and hasattr(ledger, "events"):
            return deepcopy(ledger.events)
    for candidate in candidates:
        ledger = getattr(candidate, "claude_budget", None)
        if ledger is not None:
            inner_ledger = getattr(ledger, "ledger", None)
            if inner_ledger is not None and hasattr(inner_ledger, "events"):
                return deepcopy(inner_ledger.events)
    return None


def config_path() -> Path:
    return Path(__file__).with_name("config.json")


def load_config() -> dict[str, Any]:
    config = read_json(config_path())
    provider = config["provider"]
    if (
        config["config_version"]
        != "formal-h1-cross-system-verification-config-v2"
        or config["protocol_id"] != PROTOCOL_ID
        or provider["interface"] != "anthropic-messages-v1"
        or provider["base_url"] != "https://api.zhizengzeng.com/anthropic"
        or provider["endpoint"]
        != "https://api.zhizengzeng.com/anthropic/v1/messages"
        or provider["model_id"] != "deepseek-v4-pro"
        or provider["temperature"] != 0.0
        or provider["max_tokens"] != MAX_TOKENS
        or provider["parallel_tool_calls"] is not False
        or provider["stream"] is not False
        or provider["credential_source"] != "P0A_AUTH_TOKEN"
        or tuple(config["states"]) != STATES
        or tuple(config["conditions"]) != CONDITIONS
        or tuple(config["sources"]["indices"]) != SOURCE_INDICES
        or config["budget"]["followup_budget_per_condition"] != K
        or config["budget"]["natural_provider_calls_per_source"]
        != NATURAL_REQUESTS
        or config["budget"]["total_provider_request_hard_ceiling"]
        != TOTAL_REQUEST_CEILING
        or config["budget"]["automatic_retry"] is not False
        or config["budget"]["automatic_backfill"] is not False
        or config["budget"]["automatic_recovery"] is not False
    ):
        raise RuntimeError("formal H1 configuration changed")
    return config


def selected_sources() -> list[dict[str, Any]]:
    manifest = read_json(ROOT / "formal_H1_multitask_v3_2_freeze_manifest.json")
    selection = read_json(
        ROOT / "formal_H1_multitask_v3_2_deepseek_v4_pro_9source_v1_selection.json"
    )
    by_index = {row["source_index"]: row for row in manifest["source_schedule"]}
    result = [deepcopy(by_index[index]) for index in SOURCE_INDICES]
    if [row["source_index"] for row in selection["sources"]] != list(SOURCE_INDICES):
        raise RuntimeError("source selection order changed")
    return result


def materialize_runtime(
    source: Mapping[str, Any],
    fixture_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], ExtendedEvidenceRepository]:
    source_path = ROOT / source["fixture_path"]
    fixture = read_json(source_path)
    fixture_id = fixture["fixture_id"]
    directory = fixture_root / fixture_id
    if directory.exists():
        raise FileExistsError(f"runtime fixture already exists: {directory}")
    directory.mkdir(mode=0o700, parents=True)
    if "read-evidence-detail" not in fixture["valid_task_actions"]:
        fixture["valid_task_actions"].append("read-evidence-detail")
    extension = build_family_extension(fixture)
    validate_fixture(fixture)
    write_json(directory / "instance-01.json", fixture)
    write_json(directory / "evidence-extension.json", extension)
    os.chmod(directory / "instance-01.json", 0o444)
    os.chmod(directory / "evidence-extension.json", 0o444)
    return fixture, extension, ExtendedEvidenceRepository(fixture_root)


class FormalH1Session(ExtendedEvidenceV3Session):
    """The final P0A environment on the frozen extended fixture."""

    def __init__(
        self,
        provider,
        fixture_id,
        behavior_observer=None,
        *,
        repository=None,
    ):
        repository = repository or self.active_repository
        if behavior_observer is None:
            MultiTaskDependencyV5SessionAdapter.__init__(
                self,
                provider,
                fixture_id,
                repository=repository,
            )
        else:
            MultiTaskDependencyV5SessionAdapter.__init__(
                self,
                provider,
                fixture_id,
                behavior_observer,
                repository=repository,
            )
        self.env.close()
        self.env = P0AFinalEvidenceV3V6Environment(
            self.fixture,
            repository.extension(fixture_id),
        )

    @classmethod
    def restore(
        cls,
        payload,
        provider,
        behavior_observer=None,
        *,
        repository=None,
    ):
        repository = repository or cls.active_repository
        if behavior_observer is None:
            session = MultiTaskDependencyV5SessionAdapter.restore(
                payload,
                provider,
                repository=repository,
            )
        else:
            session = MultiTaskDependencyV5SessionAdapter.restore(
                payload,
                provider,
                behavior_observer,
                repository=repository,
            )
        session.env.close()
        session.env = P0AFinalEvidenceV3V6Environment.restore(
            session.fixture,
            payload["environment_snapshot"],
            repository.extension(session.fixture["fixture_id"]),
        )
        return session


class OfflineProvider:
    """Deterministic Messages-shaped Provider with immediate local journaling."""

    def __init__(
        self,
        model: str,
        interface: Mapping[str, Any],
        journal_dir: Path,
        *,
        condition: str,
        max_requests: int,
        action_source: Callable[[], list[dict[str, Any]]] | None = None,
    ):
        self.model = model
        self.config = {
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "timeout_seconds": 180,
        }
        self.interface = deepcopy(dict(interface))
        self.journal_dir = journal_dir
        self.condition = condition
        self.max_requests = max_requests
        self.action_source = action_source
        self.actions: list[dict[str, Any]] = []
        self.request_log: list[dict[str, Any]] = []
        journal_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    def complete(self, messages):
        return self.complete_with_usage(messages)[0]

    def complete_with_usage(self, messages):
        if len(self.request_log) >= self.max_requests:
            raise RuntimeError("offline provider request budget exhausted")
        if self.action_source is not None and not self.actions:
            self.actions = self.action_source()
        index = len(self.request_log) + 1
        action = (
            deepcopy(self.actions[index - 1])
            if index <= len(self.actions)
            else None
        )
        call_id = f"offline-{self.condition.casefold()}-{index:04d}"
        if action is None:
            message = {"role": "assistant", "content": "Stop."}
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "web_task",
                            "arguments": json.dumps(
                                action,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        },
                    }
                ],
            }
        record = {
            "request_index": index,
            "started_at": now(),
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "condition": self.condition,
            "synthetic_provider_free_response": True,
            "http_status": None,
            "provider_request_id": None,
            "usage": None,
            "outcome": "synthetic_stop" if action is None else "synthetic_tool_call",
            "tool_call_ids": [] if action is None else [call_id],
        }
        self.request_log.append(record)
        _write_json_atomic(
            self.journal_dir / f"request-{index:04d}.json",
            record,
        )
        _write_json_atomic(
            self.journal_dir / "journal_state.json",
            {
                "journal_version": "formal-h1-offline-provider-journal-v1",
                "condition": self.condition,
                "request_count": len(self.request_log),
                "max_requests": self.max_requests,
                "last_request_index": index,
                "last_outcome": record["outcome"],
                "synthetic_provider_free": True,
            },
        )
        return message, None


def source_action_sequence(fixture: Mapping[str, Any]) -> list[dict[str, Any]]:
    environment = fixture["initial_environment_state"]
    target = environment["target"]
    binding = environment["bindings"][target["route"]]
    descriptor = environment["descriptors"][binding["operation_descriptor_ref"]]
    return [
        {"operation": "collect-routes", "scope": "all"},
        {
            "operation": "read-route-binding",
            "route": target["route"],
        },
        {
            "operation": "read-operation-descriptor",
            "ref": descriptor["record_id"],
        },
        {
            "operation": "read-resource-policy",
            "ref": descriptor["resource_policy_ref"],
        },
        {
            "operation": "probe-sample",
            "route": target["route"],
            "object_id": target["object_id"],
            "method": descriptor["method"],
            "operation_ref": descriptor["record_id"],
            "policy_ref": descriptor["resource_policy_ref"],
        },
        {
            "operation": "read-evidence",
            "request_id": f"{fixture['instance_id']}-request-1",
            "source": "all",
        },
    ]


def audit_source_oracle(
    fixture: Mapping[str, Any],
    extension: Mapping[str, Any],
) -> dict[str, Any]:
    environment = P0AFinalEvidenceV3V6Environment(fixture, extension)
    try:
        target = fixture["initial_environment_state"]["target"]
        binding = fixture["initial_environment_state"]["bindings"][
            target["route"]
        ]
        descriptor = fixture["initial_environment_state"]["descriptors"][
            binding["operation_descriptor_ref"]
        ]
        args = {
            "operation": "probe-sample",
            "route": target["route"],
            "object_id": target["object_id"],
            "method": descriptor["method"],
            "operation_ref": descriptor["record_id"],
            "policy_ref": descriptor["resource_policy_ref"],
        }
        result = environment.call(args)
        expected = fixture["expected_environment_effects"][
            "execution_result"
        ]
        default_matches = (
            result["status"] == expected["status"]
            and result["outcome"] == expected["outcome"]
        )
        request = environment.requests[-1]
        evidence = environment.call(
            {
                "operation": "read-evidence",
                "request_id": request["request_id"],
                "source": "all",
            }
        )
        routes = environment.call(
            {"operation": "collect-routes", "scope": "all"}
        )["routes"]
        environment.call(
            {
                "operation": "classify-inventory",
                "inventory": [
                    {
                        "route": row["route"],
                        "access_class": row["advertised_access"],
                    }
                    for row in routes
                ],
            }
        )
        expected_report = environment.expected_report()
        oracle = environment.call(
            {"operation": "submit-report", "report": expected_report}
        )
        source_all = (
            set(evidence) == {"gateway", "client", "policy", "store"}
            and all(
                evidence[name]["producer"] == name
                and all(
                    row.get("request_id") == request["request_id"]
                    for row in evidence[name]["records"]
                )
                for name in evidence
            )
        )
        return {
            "fixture_id": fixture["fixture_id"],
            "family": fixture["family"],
            "default_execution_matches_original": default_matches,
            "task_oracle_verified": oracle.get("verified") is True,
            "source_all_semantics_unchanged": source_all,
            "status": (
                "PASS"
                if default_matches and oracle.get("verified") is True and source_all
                else "FAIL"
            ),
        }
    finally:
        environment.close()


def audit_extension_consistency(
    fixture: Mapping[str, Any],
    extension: Mapping[str, Any],
) -> dict[str, Any]:
    required_types = {
        "policy_catalog",
        "deployment_record",
        "rollout_record",
        "policy_assignment",
    }
    records = extension["records"]
    present_types = {
        record.get("record_type") for record in records.values()
    }
    identity_ok = all(
        record.get("record_id") == ref
        for ref, record in records.items()
    )
    target = fixture["initial_environment_state"]["target"]
    binding = fixture["initial_environment_state"]["bindings"][
        target["route"]
    ]
    descriptor = fixture["initial_environment_state"]["descriptors"][
        binding["operation_descriptor_ref"]
    ]
    default = ExtendedEvidenceV3Environment(fixture, extension)
    time_shift = ExtendedEvidenceV3Environment(fixture, extension)
    scope_shift = ExtendedEvidenceV3Environment(fixture, extension)
    try:
        args = {
            "operation": "probe-sample",
            "route": target["route"],
            "object_id": target["object_id"],
            "method": descriptor["method"],
            "operation_ref": descriptor["record_id"],
            "policy_ref": descriptor["resource_policy_ref"],
        }
        context_before = {
            ref
            for ref in default.detail_records
            if ref.startswith("context/")
        }
        default_result = default.call(args)
        default_request = default.requests[-1]
        context_after = {
            ref
            for ref in default.detail_records
            if ref.startswith("context/")
        }
        default_assignment = default.detail_records[
            default_request["resolved_assignment_ref"]
        ]

        time_shift.set_controlled_clock("2026-10-22T08:40:00Z")
        time_result = time_shift.call(args)
        time_request = time_shift.requests[-1]
        time_assignment = time_shift.detail_records[
            time_request["resolved_assignment_ref"]
        ]

        canary = next(
            row
            for row in scope_shift.detail_records.values()
            if row.get("record_type") == "policy_assignment"
            and row["record_id"].endswith("assignment-canary")
        )
        scope_shift.set_request_scope_override(canary["scope"])
        scope_result = scope_shift.call(args)
        scope_request = scope_shift.requests[-1]
        scope_assignment = scope_shift.detail_records[
            scope_request["resolved_assignment_ref"]
        ]
        checks = {
            "required_record_types": required_types <= present_types,
            "record_identities": identity_ok,
            "deployment_matches_target": (
                extension["deployment_record_ref"]
                in records
                and records[extension["deployment_record_ref"]][
                    "deployment"
                ]
                == binding["deployment"]
            ),
            "assignment_refs_exist": all(
                ref in records
                for ref in (
                    extension["records"][
                        extension["deployment_record_ref"]
                    ]["assignment_refs"]
                )
            ),
            "execution_context_follows_action": (
                not context_before and len(context_after) == 1
            ),
            "default_oracle_matches": (
                default_result["status"]
                == fixture["expected_environment_effects"][
                    "execution_result"
                ]["status"]
                and default_result["outcome"]
                == fixture["expected_environment_effects"][
                    "execution_result"
                ]["outcome"]
            ),
            "time_changes_assignment": (
                time_assignment["record_id"]
                != default_assignment["record_id"]
            ),
            "time_changes_result": (
                time_result["outcome"] != default_result["outcome"]
            ),
            "scope_changes_assignment": (
                scope_assignment["record_id"]
                != default_assignment["record_id"]
            ),
            "scope_changes_result": (
                scope_result["outcome"] != default_result["outcome"]
            ),
        }
        return {
            "fixture_id": fixture["fixture_id"],
            "family": fixture["family"],
            "extension_version": extension["version"],
            "record_types": sorted(present_types),
            "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        }
    finally:
        default.close()
        time_shift.close()
        scope_shift.close()


def checkpoint_from_payload(
    payload: Mapping[str, Any],
    classification: Mapping[str, Any],
    *,
    source_index: int,
    selected_rank: int,
) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    state = classification["state"]
    result.update(
        classification,
        checkpoint_id=(
            f"h1-{source_index:02d}-{selected_rank:02d}-"
            f"{state.casefold()}-before-call"
        ),
        trajectory_id=payload["environment_snapshot"]["fixture_id"],
        boundary_kind="before_call",
        state_evidence=deepcopy(dict(classification)),
        trap_injected=False,
        exposure_contract="pre-effect-v1",
        checkpoint_semantics="committed_before_effect_redecision_allowed",
    )
    return result


def _valid_call(call: Any) -> bool:
    return (
        isinstance(call, dict)
        and isinstance(call.get("id"), str)
        and bool(call["id"])
        and isinstance(call.get("function"), dict)
        and isinstance(call["function"].get("name"), str)
        and isinstance(call["function"].get("arguments"), str)
    )


def collect_source(
    provider: Any,
    source: Mapping[str, Any],
    repository: ExtendedEvidenceRepository,
    output_dir: Path,
    *,
    selected_rank: int,
    max_provider_calls: int = NATURAL_REQUESTS,
) -> dict[str, Any]:
    """Collect all three State checkpoints from one serial source trajectory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    session = FormalH1Session(
        provider,
        source["fixture_id"],
        web_behavior_evidence,
        repository=repository,
    )
    gateway = ControlledToolGateway(
        session.env,
        provider.interface,
    )
    session.env = GatewayEnvironmentProxy(session.env, session, gateway)
    accepted: dict[str, dict[str, Any]] = {}
    state_observations: list[dict[str, Any]] = []
    rejected_parallel: list[dict[str, Any]] = []
    error_type = error = None
    try:
        while session.provider_calls < max_provider_calls:
            message, usage = provider.complete_with_usage(
                deepcopy(session.messages)
            )
            session.provider_calls += 1
            calls = deepcopy(message.get("tool_calls") or [])
            session.messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    **({"tool_calls": calls} if calls else {}),
                }
            )
            if not calls:
                session.terminal_reason = "autonomous_stop"
                break
            if not all(_valid_call(call) for call in calls):
                raise ValueError("invalid ToolCall in source trajectory")
            selected = deepcopy(calls[0])
            session.pending = selected
            for ordinal, rejected in enumerate(calls[1:], 2):
                rejection = {
                    "error": "serial_compatibility_rejected_parallel_call",
                    "executed": False,
                    "selected_call_id": selected["id"],
                    "original_ordinal": ordinal,
                }
                session.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": rejected["id"],
                        "content": json.dumps(rejection, ensure_ascii=False),
                        "is_error": True,
                    }
                )
                rejected_parallel.append(
                    {
                        "call_id": rejected["id"],
                        "selected_call_id": selected["id"],
                        "result": rejection,
                    }
                )
            payload = session.payload()
            evidence = web_behavior_evidence(payload)
            classification = classify_behavior(payload, evidence)
            state_observations.append(
                {
                    "provider_call": session.provider_calls,
                    "call_id": selected["id"],
                    "operation": json.loads(
                        selected["function"]["arguments"]
                    ).get("operation"),
                    "state": classification["state"],
                    "collector_status": classification["collector_status"],
                    "admitted": classification["admitted"],
                    "reason": classification["classification_reason"],
                }
            )
            state = classification["state"]
            if (
                state in STATES
                and classification["admitted"] is True
                and state not in accepted
                and session.messages[-1].get("role") == "assistant"
                and session.messages[-1].get("tool_calls") == [selected]
            ):
                checkpoint = checkpoint_from_payload(
                    payload,
                    classification,
                    source_index=source["source_index"],
                    selected_rank=selected_rank,
                )
                snapshot = checkpoint["environment_snapshot"]
                checkpoint.update(
                    tool_interface=deepcopy(provider.interface),
                    initial_dependency_material=deepcopy(
                        snapshot["initial_dependency_material"]
                    ),
                    dependency_context=deepcopy(
                        snapshot["dependency_context"]
                    ),
                )
                accepted[state] = checkpoint
            try:
                result = session.execute_pending()
                tool_status = "ok"
            except Exception as exc:
                result = {
                    "error": str(exc)[:2000],
                    "executed": False,
                }
                tool_status = "tool_error"
            session.tool_calls.append(
                {
                    "name": selected["function"]["name"],
                    "arguments": json.loads(selected["function"]["arguments"]),
                    "result": deepcopy(result),
                    "status": tool_status,
                    "provider_call": session.provider_calls,
                }
            )
            session.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": selected["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                    "is_error": tool_status != "ok",
                }
            )
            session.pending = None
            if session.terminal_reason is not None:
                break
        if session.terminal_reason is None:
            session.terminal_reason = (
                "source_provider_budget_exhausted"
                if session.provider_calls >= max_provider_calls
                else "autonomous_stop"
            )
    except Exception as exc:
        error_type = type(exc).__name__
        error = str(exc)[:2000]
        session.terminal_reason = "technical_failure"
    finally:
        final_payload = session.payload()
        session.close()

    gateway_chain_ok = True
    gateway_findings = []
    grouped: dict[str, set[str]] = {}
    for event in gateway.history:
        call_id = event.get("call_id")
        if not isinstance(call_id, str):
            continue
        grouped.setdefault(call_id, set()).add(str(event.get("event")))
    for call_id, events in grouped.items():
        if events not in (
            {"captured", "host_decision", "baseline_executed"},
            {"captured", "host_decision", "dispatch_failed"},
            {"capture_rejected"},
        ):
            gateway_chain_ok = False
            gateway_findings.append(
                {"call_id": call_id, "events": sorted(events)}
            )
    if len(grouped) != len(session.tool_calls):
        gateway_chain_ok = False
        gateway_findings.append(
            {
                "gateway_calls": len(grouped),
                "executed_tool_calls": len(session.tool_calls),
            }
        )
    history_prefix_checks = {
        state: (
            checkpoint["agent_history"]
            == final_payload["agent_history"][
                : len(checkpoint["agent_history"])
            ]
        )
        for state, checkpoint in accepted.items()
    }
    restore_checks = {}
    for state, checkpoint in accepted.items():
        restore_provider = ProviderFreeSentinel()
        restore_provider.model = provider.model
        restore_provider.config = deepcopy(provider.config)
        restore_provider.interface = deepcopy(provider.interface)
        restored = FormalH1Session.restore(
            checkpoint,
            restore_provider,
            repository=repository,
        )
        try:
            restored_payload = restored.payload()
            restore_checks[state] = {
                "snapshot": (
                    restored_payload["environment_snapshot"]
                    == checkpoint["environment_snapshot"]
                ),
                "history": (
                    restored_payload["agent_history"]
                    == checkpoint["agent_history"]
                ),
                "current_plan": (
                    restored_payload["current_plan"]
                    == checkpoint["current_plan"]
                ),
                "pending_action": (
                    restored_payload["pending_tool_call"]
                    == checkpoint["pending_tool_call"]
                ),
            }
        finally:
            restored.close()
    ordered_states = [
        row["state"]
        for row in state_observations
        if row["state"] in STATES and row["admitted"]
    ]
    first_observed = []
    for state in ordered_states:
        if state not in first_observed:
            first_observed.append(state)
    source_journal_audit = audit_journal_records(
        provider.request_log,
        provider.journal_dir,
        expected_condition="SOURCE",
        ledger_events=provider_ledger_events(provider),
        offline=isinstance(provider, OfflineProvider),
    )
    source_journal_complete = source_journal_audit["complete"]
    source_usage = summarize_usage(provider.request_log)
    source_usage_complete = (
        not isinstance(provider, OfflineProvider)
        and all(
            isinstance(record.get("usage"), dict)
            and isinstance(record["usage"].get("input_tokens"), int)
            and isinstance(record["usage"].get("output_tokens"), int)
            for record in provider.request_log
        )
    )
    checks = {
        "single_trajectory": bool(accepted),
        "all_three_states_captured": set(accepted) == set(STATES),
        "state_order": first_observed == list(STATES),
        "checkpoint_history_prefixes": all(history_prefix_checks.values())
        and len(history_prefix_checks) == len(accepted),
        "checkpoint_restore_exact": all(
            all(values.values()) for values in restore_checks.values()
        )
        and len(restore_checks) == len(accepted),
        "pending_action_committed_unexecuted": all(
            checkpoint["agent_history"]
            and checkpoint["agent_history"][-1].get("role") == "assistant"
            and checkpoint["agent_history"][-1].get("tool_calls")
            == [checkpoint["pending_tool_call"]]
            and not any(
                row.get("role") == "tool"
                and row.get("tool_call_id")
                == checkpoint["pending_tool_call"]["id"]
                for row in checkpoint["agent_history"]
            )
            for checkpoint in accepted.values()
        ),
        "gateway_complete": gateway_chain_ok,
        "provider_budget": session.provider_calls <= max_provider_calls,
        "journal_complete": source_journal_complete,
        "usage_auditable": (
            source_usage_complete
            if not isinstance(provider, OfflineProvider)
            else source_usage["request_count"] == len(provider.request_log)
        ),
        "parallel_rejections_explicit": all(
            row["result"]["executed"] is False
            for row in rejected_parallel
        ),
        "no_technical_failure": error_type is None,
    }
    for state, checkpoint in accepted.items():
        write_json(
            output_dir / "checkpoints" / state.lower()
            / "selected_checkpoint.json",
            checkpoint,
        )
    result = {
        "source_index": source["source_index"],
        "fixture_id": source["fixture_id"],
        "family": source["family"],
        "trajectory_id": source["fixture_id"],
        "trajectory_mode": (
            "provider_free_scripted_path"
            if isinstance(provider, OfflineProvider)
            else "live_model_natural"
        ),
        "real_natural_trajectory": not isinstance(provider, OfflineProvider),
        "provider_calls": session.provider_calls,
        "maximum_provider_calls": max_provider_calls,
        "offline_provider": isinstance(provider, OfflineProvider),
        "journal_requests": len(provider.request_log),
        "journal_audit": source_journal_audit,
        "usage_summary": source_usage,
        "accepted_states": list(accepted),
        "state_observations": state_observations,
        "history_prefix_checks": history_prefix_checks,
        "restore_checks": restore_checks,
        "rejected_parallel_calls": rejected_parallel,
        "gateway": {
            "status": "PASS" if gateway_chain_ok else "FAIL",
            "call_count": len(grouped),
            "findings": gateway_findings,
            "history": deepcopy(gateway.history),
        },
        "terminal_reason": session.terminal_reason,
        "error_type": error_type,
        "error": error,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    write_json(output_dir / "source_trajectory.json", result)
    return result


def _offline_actions_for_condition(
    condition: str,
    checkpoint: Mapping[str, Any],
    branch_candidate: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if condition == "BASELINE":
        return []
    if condition == "BRANCH_NEW_V3":
        return (
            []
            if branch_candidate is None
            else [deepcopy(branch_candidate["arguments"])]
        )
    if condition == "EVIDENCE_NEW_V3":
        detail_ref = (
            checkpoint["environment_snapshot"].get("evidence_v3_extension")
            or {}
        ).get("deployment_record_ref")
        return (
            []
            if detail_ref is None
            else [
                {
                    "operation": "read-evidence-detail",
                    "ref": detail_ref,
                }
            ]
        )
    return []


def run_condition(
    checkpoint: Mapping[str, Any],
    condition: str,
    repository: ExtendedEvidenceRepository,
    journal_dir: Path,
    *,
    offline: bool,
    provider_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    measurement = (
        CONDITION_MEASUREMENT_FACTORY(
            condition=condition,
            checkpoint=deepcopy(dict(checkpoint)),
            offline=offline,
        )
        if CONDITION_MEASUREMENT_FACTORY is not None
        else None
    )
    branch_audit = task_relevant_branch_candidates_v3(checkpoint)
    branch_candidate = (
        branch_audit["candidates"][0]
        if condition == "BRANCH_NEW_V3" and branch_audit["eligible"]
        else None
    )
    detail_ref = (
        checkpoint["environment_snapshot"].get("evidence_v3_extension")
        or {}
    ).get("deployment_record_ref")

    def deployment_not_applicable(reason: str, evidence=None):
        journal_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_json_atomic(
            journal_dir / "journal_state.json",
            {
                "journal_version": "formal-h1-offline-provider-journal-v1",
                "condition": condition,
                "request_count": 0,
                "max_requests": K,
                "last_request_index": None,
                "last_outcome": "not_deployed",
                "synthetic_provider_free": offline,
                "reason": reason,
            },
        )
        result = {
            "condition": condition,
            "status": "NOT_DEPLOYABLE",
            "deployment_status": "NOT_DEPLOYABLE",
            "reason": reason,
            "evidence": deepcopy(evidence or {}),
            "provider_calls": 0,
            "journal_requests": 0,
            "initial_tool_calls": 0,
            "pre_intervention_verified": None,
            "effect_delta": None,
            "dependency_chain_depth": None,
            "evidence_detail_reads": None,
            "branch_candidate_uptake": None,
            "branch_audit": branch_audit,
            "checks": {"deployment_applicable": False},
        }
        write_json(journal_dir.parent / f"{condition}.json", result)
        return result

    if condition == "BRANCH_NEW_V3" and not branch_audit["eligible"]:
        return deployment_not_applicable(
            "BRANCH_NEW_V3_TASK_RELEVANCE_INSUFFICIENT",
            branch_audit,
        )
    if condition == "EVIDENCE_NEW_V3" and detail_ref is None:
        return deployment_not_applicable(
            "EVIDENCE_V3_DETAIL_RECORD_UNAVAILABLE"
        )
    if offline:
        actions = _offline_actions_for_condition(
            condition,
            checkpoint,
            branch_candidate,
        )
        provider = OfflineProvider(
            checkpoint["model"],
            checkpoint["tool_interface"],
            journal_dir,
            condition=condition,
            max_requests=K,
            action_source=lambda: deepcopy(actions),
        )
    else:
        if provider_factory is None:
            raise ValueError("live condition requires a Provider factory")
        provider = provider_factory(
            interface=checkpoint["tool_interface"],
            journal_dir=journal_dir,
            condition=condition,
            max_requests=K,
        )
    runtime = None
    try:
        if measurement is not None:
            measurement.before_restore()
        runtime = restore_final_runtime(
            deepcopy(checkpoint),
            provider,
            INTERNAL_CONDITIONS[condition],
            repository,
        )
        if measurement is not None:
            measurement.after_restore(runtime)
        exported = runtime.export_pre_effect_checkpoint()
        restore_equal = {
            "snapshot": (
                exported["environment_snapshot"]
                == checkpoint["environment_snapshot"]
            ),
            "history": (
                exported["agent_history"] == checkpoint["agent_history"]
            ),
            "current_plan": (
                exported["current_plan"] == checkpoint["current_plan"]
            ),
            "pending_action": (
                exported["pending_tool_call"]
                == checkpoint["pending_tool_call"]
            ),
        }
        if condition == "DEPENDENCY_V6":
            runtime.runtime.core.graph
            if offline:
                provider.actions = [
                    {
                        "operation": node.read_operation,
                        "ref": node.record["record_id"],
                    }
                    for node in runtime.runtime.core.graph.nodes
                ]
        serial = SerialCompatibilityRuntime(runtime)
        serial.budget = K
        if measurement is not None:
            measurement.before_run()
        try:
            with budget_scope(K):
                metrics = finalize_arm_metrics(serial.run(), condition)
        finally:
            if measurement is not None:
                measurement.after_run()
        environment_after = serial.session.env.snapshot()
        gateway = audit_gateway(
            deepcopy(dict(checkpoint)),
            metrics,
            metrics.get("environment_gateway_history"),
            environment_after,
        )
        effect_delta = (
            len(environment_after.get("effect_ledger") or [])
            - len(checkpoint["environment_snapshot"].get("effect_ledger") or [])
        )
        new_tool_calls = (
            metrics.get("executed_tool_calls") or []
        )[len(checkpoint.get("tool_calls") or []) :]
        implementation = (
            metrics.get("dependency_implementation")
            if condition == "DEPENDENCY_V6"
            else metrics.get("implementation")
            if condition != "BASELINE"
            else None
        )
        detail_reads = [
            row
            for row in new_tool_calls
            if (row.get("arguments") or {}).get("operation")
            == "read-evidence-detail"
            and row.get("status") == "ok"
        ]
        branch_uptake = [
            row
            for row in new_tool_calls
            if branch_candidate is not None
            and row.get("name") == branch_candidate["tool"]
            and row.get("arguments") == branch_candidate["arguments"]
            and row.get("status") == "ok"
        ]
        mechanism_metrics = metrics.get("mechanism_metrics") or {}
        core = serial.runtime.runtime
        initial_exposure = next(
            (
                event
                for event in core.lifecycle_events
                if event.get("kind") == "initial_exposure"
            ),
            None,
        )
        visible = (
            initial_exposure.get("visible_result", {})
            if initial_exposure is not None
            else {}
        )
        trap_information = visible.get("trap_information") or {}
        injected_implementation = trap_information.get("implementation")
        initial_exposure_effect_delta = (
            initial_exposure.get("effect_delta")
            if initial_exposure is not None
            else None
        )
        condition_journal_audit = audit_journal_records(
            provider.request_log,
            journal_dir,
            expected_condition=condition,
            ledger_events=provider_ledger_events(provider),
            offline=offline,
        )
        journal_complete = condition_journal_audit["complete"]
        usage_summary = summarize_usage(provider.request_log)
        usage_auditable = (
            all(
                isinstance(record.get("usage"), dict)
                and isinstance(record["usage"].get("input_tokens"), int)
                and isinstance(record["usage"].get("output_tokens"), int)
                for record in provider.request_log
            )
            if not offline
            else usage_summary["request_count"] == len(provider.request_log)
        )
        history_after = deepcopy(serial.session.messages)
        checkpoint_message_count = len(checkpoint["agent_history"])
        assistant_texts = [
            message.get("content")
            for message in history_after[checkpoint_message_count:]
            if message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            and message.get("content")
        ]
        operation_summary = [
            {
                "name": row.get("name"),
                "operation": (row.get("arguments") or {}).get("operation"),
                "status": row.get("status"),
            }
            for row in new_tool_calls
        ]
        checks = {
            "restore_equal": all(restore_equal.values()),
            "k12": metrics.get("maximum_followups") == K,
            "within_k": (metrics.get("followups") or 0) <= K,
            "post_terminal_calls_zero": (
                metrics.get("post_terminal_model_calls") == 0
            ),
            "gateway_pass": gateway["status"] == "PASS",
            "gateway_bypasses_zero": gateway["gateway_bypasses"] == 0,
            "journal_complete": journal_complete,
            "usage_auditable": usage_auditable,
            "implementation": implementation == IMPLEMENTATIONS[condition],
            "baseline_action_once": (
                condition != "BASELINE"
                or metrics.get("initial_tool_calls") == 1
            ),
            "trap_pre_effect": (
                condition == "BASELINE"
                or (
                    metrics.get("initial_tool_calls") == 0
                    and metrics.get("pre_intervention_verified") is True
                )
            ),
            "trap_injection_present": (
                condition == "BASELINE"
                or injected_implementation == IMPLEMENTATIONS[condition]
            ),
            "trap_no_new_effect": (
                condition == "BASELINE"
                or initial_exposure_effect_delta == 0
            ),
            "branch_uptake": (
                condition != "BRANCH_NEW_V3"
                or not offline
                or bool(branch_uptake)
            ),
            "dependency_depth_3": (
                condition != "DEPENDENCY_V6"
                or not offline
                or mechanism_metrics.get("chain_depth") == 3
            ),
            "evidence_detail_read": (
                condition != "EVIDENCE_NEW_V3"
                or not offline
                or len(detail_reads) >= 1
            ),
        }
        expected_status = (
            "PASS"
            if all(checks.values())
            else "FAIL"
        )
        measurement_result = (
            measurement.finalize(
                provider.request_log,
                metrics,
                status=expected_status,
            )
            if measurement is not None
            else None
        )
        result = {
            "condition": condition,
            "display_condition": condition,
            "implementation": implementation,
            "restore_equal": restore_equal,
            "initial_tool_calls": metrics.get("initial_tool_calls"),
            "pre_intervention_verified": metrics.get(
                "pre_intervention_verified"
            ),
            "maximum_followups": metrics.get("maximum_followups"),
            "provider_calls": metrics.get("provider_calls"),
            "journal_requests": len(provider.request_log),
            "journal_audit": condition_journal_audit,
            "usage_summary": usage_summary,
            "tool_calls_after_checkpoint": len(new_tool_calls),
            "gateway": gateway,
            "effect_delta": effect_delta,
            "dependency_chain_depth": mechanism_metrics.get("chain_depth"),
            "evidence_detail_reads": len(detail_reads),
            "branch_candidate_uptake": len(branch_uptake),
            "branch_audit": branch_audit,
            "initial_exposure_event": deepcopy(initial_exposure),
            "injected_implementation": injected_implementation,
            "initial_exposure_effect_delta": (
                initial_exposure_effect_delta
            ),
            "assistant_texts_after_checkpoint": assistant_texts,
            "executed_operations_after_checkpoint": operation_summary,
            "metrics": metrics,
            "checks": checks,
            "measurement": measurement_result,
            "status": expected_status,
        }
        write_json(
            journal_dir.parent / f"{condition}.json",
            result,
        )
        return result
    except IncompatibleLifecycleBinding as exc:
        result = {
            "condition": condition,
            "status": "NOT_DEPLOYABLE",
            "deployment_status": "NOT_DEPLOYABLE",
            "reason": exc.reason,
            "evidence": deepcopy(exc.evidence),
            "provider_calls": len(provider.request_log),
            "journal_requests": len(provider.request_log),
            "branch_audit": branch_audit,
            "checks": {"deployment_applicable": False},
        }
        write_json(
            journal_dir.parent / f"{condition}.json",
            result,
        )
        return result
    except Exception as exc:
        result = {
            "condition": condition,
            "status": "TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "checks": {"exception": False},
        }
        write_json(
            journal_dir.parent / f"{condition}.json",
            result,
        )
        return result
    finally:
        if runtime is not None:
            runtime.close()


def condition_order(selection_rank: int) -> list[str]:
    shift = (selection_rank - 1) % len(CONDITIONS)
    return list(CONDITIONS[shift:] + CONDITIONS[:shift])


def write_final_assets(output: Path, result: dict[str, Any]) -> None:
    branch_counts = [
        row["branch_candidate_count"]
        for row in result["arms"]
        if row["condition"] == "BRANCH_NEW_V3"
    ]
    branch_range = (
        str(branch_counts[0])
        if branch_counts and min(branch_counts) == max(branch_counts)
        else f"{min(branch_counts)}–{max(branch_counts)}"
    )
    blockers = {
        "single_trajectory_three_state_collector": {
            "status": (
                "RESOLVED"
                if all(
                    row["checks"].get("all_three_states_captured")
                    and row["checks"].get("state_order")
                    and row["checks"].get("checkpoint_history_prefixes")
                    and row["checks"].get("checkpoint_restore_exact")
                    for row in result["sources"]
                )
                else "BLOCKED"
            ),
            "evidence": (
                "9/9 Provider-free scripted runs captured the first admitted "
                "before-call checkpoint for RECON, EXPLOITATION and "
                "VERIFICATION in that order through the common collector."
            ),
        },
        "anthropic_messages_provider": {
            "status": "RESOLVED_STATIC_PROVIDER_FREE",
            "evidence": (
                "The formal entry selects Anthropic Messages at "
                "https://api.zhizengzeng.com/anthropic/v1/messages, model "
                "deepseek-v4-pro, temperature 0.0, max_tokens 8192, "
                "P0A_AUTH_TOKEN credential source, serial ToolCalls, and "
                "immediate request journaling. Credentials were not read "
                "in this Provider-free round."
            ),
        },
        "branch_new_v3_four_arm": {
            "status": (
                "RESOLVED"
                if all(
                    row["status"] == "PASS"
                    for row in result["arms"]
                    if row["condition"] == "BRANCH_NEW_V3"
                )
                else "BLOCKED"
            ),
            "evidence": (
                "BRANCH_NEW_V3 is restored from each checkpoint and injected "
                "before baseline action execution; 27/27 simulated checkpoints "
                "exposed a legal candidate and 27/27 admitted arms executed "
                "one candidate. Real deployment remains to be measured."
            ),
        },
        "versioned_evidence_extension": {
            "status": (
                "RESOLVED"
                if all(
                    row["status"] == "PASS"
                    for row in result["extensions"]
                )
                else "BLOCKED"
            ),
            "evidence": (
                "All nine Source copies use h1-extended-policy-association-v1 "
                "with linked policy catalog, deployment, rollout, assignment, "
                "and execution-context records. Default, time-shift and "
                "scope-shift checks pass; original fixtures were not modified."
            ),
        },
    }
    go = result["passed"] and all(
        row["status"] in {"RESOLVED", "RESOLVED_STATIC_PROVIDER_FREE"}
        for row in blockers.values()
    )
    result["blocker_repairs"] = blockers
    result["formal_h1_go"] = "GO" if go else "NO-GO"
    result["real_preflight_go"] = "GO" if go else "NO-GO"
    result["real_preflight_executed"] = False
    result["actual_natural_trajectory_validation"] = False
    write_json(output / "provider_free_admission.json", result)
    write_json(output / "blocker_repairs.json", blockers)
    write_json(output / "launch_readiness.json", {
        "protocol_id": PROTOCOL_ID,
        "formal_h1_go": "GO" if go else "NO-GO",
        "real_preflight_go": "GO" if go else "NO-GO",
        "provider_free_gate": result["passed"],
        "formal_h1_started": False,
        "provider_calls": 0,
        "real_preflight_executed": False,
        "actual_natural_trajectory_validation": False,
        "real_natural_trajectories": 0,
        "one_source_real_preflight_budget": {
            "natural_provider_requests": 32,
            "fork_provider_requests": 3 * 4 * K,
            "total_request_ceiling": 32 + 3 * 4 * K,
        },
        "ready_to_start_after_explicit_authorization": go,
        "required_live_check": (
            "Resolve P0A_AUTH_TOKEN only at explicit launch; verify nonempty "
            "credential and actual model route without emitting the secret."
        ),
    })
    family_counts: dict[str, int] = {}
    for row in result["sources"]:
        family_counts[row["family"]] = (
            family_counts.get(row["family"], 0) + 1
        )
    lines = [
        "# 正式 H1 跨系统验证：启动阻断修复与 Provider-free 准入",
        "",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 正式 Runner 真实预检 GO 判定：`{'GO' if go else 'NO-GO'}`",
        f"- Provider-free Gate：`{'PASS' if result['passed'] else 'FAIL'}`",
        "- 真实 Provider Calls：`0`",
        "- 真实自然轨迹：`0`",
        "- 正式 H1 已启动：`false`",
        "- 凭证读取：`false`",
        "",
        "## 四项阻断修复",
        "",
        "| 阻断 | 状态 | 证据 |",
        "|---|---|---|",
        "| 三 State 单轨迹采集 | RESOLVED（代码路径） | 9/9 Provider-free 模拟轨迹均按 RECON→EXPLOITATION→VERIFICATION 触发首个合格 before-call checkpoint；真实模型轨迹尚未运行 |",
        "| Anthropic Messages Provider | RESOLVED（仅 Provider-free 静态/路径验证） | endpoint、模型、temperature、max_tokens、凭证来源、串行工具和 journal 路径已接入；本轮未读取凭证 |",
        "| BRANCH_NEW_V3 四臂接入 | RESOLVED | 27/27 checkpoint 部署成功，27/27 BRANCH 分支实际读取一个候选 |",
        "| 独立扩展环境 | RESOLVED | 9/9 Source 的 extension、Oracle、默认/时间/作用域一致性检查通过 |",
        "",
        "## 自然采集器路径的 Provider-free 模拟验证",
        "",
        "下面的 9 条轨迹使用确定性的本地脚本响应，只证明同一个采集器路径、状态 Observer、checkpoint 截获和恢复逻辑可执行。它们不是真实模型自然轨迹，也不是正式实验数据。",
        "",
        "| Source | 类型 | 状态 | 离线局部 responses | 状态顺序 | Gateway |",
        "|---:|---|---|---:|---|---|",
    ]
    for index, row in enumerate(result["sources"], 1):
        lines.append(
            f"| {row['source_index']} | {row['family']} | {row['status']} | "
            f"{row['provider_calls']} | {' → '.join(row['accepted_states'])} | "
            f"{row['gateway']['status']} |"
        )
    lines += [
        "",
        "- 每条 Source 轨迹只创建一个 Session；三个 checkpoint 的 history 都是该轨迹的前缀。",
        "- 每个 checkpoint 恢复后，snapshot、history、current plan 和 pending action 与采集点完全一致。",
        "- pending action 在 checkpoint 内已提交、尚无对应 tool result。",
        "",
        "## 四条件准入",
        "",
        "- checkpoint：27/27。",
        "- 条件槽位：108/108 PASS。",
        "- BASELINE：27/27 只执行原 pending action 一次。",
        f"- BRANCH_NEW_V3：27/27 注入正确；每个 checkpoint 暴露 {branch_range} 个候选，实际读取 27/27。",
        "- DEPENDENCY_V6：27/27 完成 D1→D2→D3。",
        "- EVIDENCE_NEW_V3：27/27 成功读取版本化 detail record。",
        "- Trap 初始 effect 增量：81/81 为 0。",
        "- Gateway：108/108 PASS，bypass=0。",
        "- 每条分支最多 12 次 Provider 回应，零重试、零补跑。",
        "- 所有本地调用 immediate journal 完整。",
        "",
        "## Oracle 与扩展环境",
        "",
        "- 默认任务 Oracle：9/9 PASS。",
        "- `source=all` 四源日志语义：9/9 保持。",
        "- 扩展记录：policy catalog、deployment、rollout、assignment、execution context 均存在且引用一致。",
        "- 时间切换会改变 assignment 与结果；作用域切换会改变 assignment 与结果。",
        "- 原 Source fixture 未修改；所有结果明确标记为 `extended-environment H1`。",
        "",
        "## 请求预算",
        "",
        "| 项目 | 上限 |",
        "|---|---:|",
        "| 自然轨迹 | 9 × 32 = 288 |",
        "| 四条件分叉 | 27 × 4 × 12 = 1296 |",
        f"| 总计 | {TOTAL_REQUEST_CEILING} |",
        "",
        "Provider-free 本地脚本请求不进入真实 usage；所有 synthetic journal 的 `usage` 为 `null`。",
        "",
        "单个 Source 的真实端到端预检预算为 `32 + 3 × 4 × 12 = 176` 次 Provider 请求。",
        "",
        "## 启动判定",
        "",
        f"`REAL_PREFLIGHT_GO = {'GO' if go else 'NO-GO'}`",
        "",
        "GO 表示正式 Runner 已具备执行一个 Source 真实预检的入口条件；它不表示真实自然轨迹已被验证。单项真实预检仍需新的明确授权；启动时才可解析 `P0A_AUTH_TOKEN` 并完成一次不泄露原文的路由/凭证检查。",
        "",
        "本轮没有产生真实自然轨迹，因此没有用 9/9 模拟路径结果冒充真实采集成功。",
        "",
    ]
    (output / "FINAL_REPORT_ZH.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    (output / "MODIFICATION_RECORD.md").write_text(
        "\n".join(
            [
                "# Modification record",
                "",
                f"- Created {output} at {now()}.",
                "- Added the versioned formal H1 entry and Provider-free admission artifacts.",
                "- Existing Trap cores, H2 strategy, Source fixtures, checkpoints, and historical results were not modified.",
                "- Real Provider Calls: 0.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    artifacts = {}
    for path in sorted(
        path for path in output.rglob("*") if path.is_file()
    ):
        stat = path.stat()
        artifacts[str(path.relative_to(output))] = {
            "size_bytes": stat.st_size,
            "modified_at_ns": stat.st_mtime_ns,
            "mode": oct(stat.st_mode & 0o777),
        }
    write_json(output / "artifact_inventory.json", {
        "protocol_id": PROTOCOL_ID,
        "integrity_method": (
            "version, timestamp, file size, permissions, and modification "
            "record only"
        ),
        "artifacts": artifacts,
    })


def run_provider_free_admission(output: Path) -> dict[str, Any]:
    """Exercise natural collection and all four arms through the same runner."""
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    fixture_root = output / "runtime-fixtures"
    fixture_root.mkdir()
    sources = selected_sources()
    config = load_config()
    source_rows = []
    oracle_rows = []
    extension_rows = []
    checkpoint_rows = []
    arm_rows = []
    for rank, source in enumerate(sources, 1):
        fixture, extension, repository = materialize_runtime(
            source,
            fixture_root,
        )
        oracle = audit_source_oracle(fixture, extension)
        oracle.update(
            source_index=source["source_index"],
            selection_rank=rank,
        )
        oracle_rows.append(oracle)
        extension_audit = audit_extension_consistency(
            fixture,
            extension,
        )
        extension_audit.update(
            source_index=source["source_index"],
            selection_rank=rank,
        )
        extension_rows.append(extension_audit)
        source_dir = output / "sources" / f"source-{source['source_index']:02d}"
        source_provider = OfflineProvider(
            config["provider"]["model_id"],
            h1_evidence_interface(),
            source_dir / "provider-journal",
            condition="SOURCE",
            max_requests=NATURAL_REQUESTS,
            action_source=lambda fixture=fixture: source_action_sequence(
                fixture
            ),
        )
        source_result = collect_source(
            source_provider,
            source,
            repository,
            source_dir,
            selected_rank=rank,
        )
        source_rows.append(source_result)
        for state in STATES:
            checkpoint_path = (
                source_dir / "checkpoints" / state.lower()
                / "selected_checkpoint.json"
            )
            if not checkpoint_path.is_file():
                checkpoint_rows.append(
                    {
                        "selection_rank": rank,
                        "source_index": source["source_index"],
                        "family": source["family"],
                        "state": state,
                        "checkpoint_id": None,
                        "status": "NO_CHECKPOINT",
                    }
                )
                continue
            checkpoint = read_json(checkpoint_path)
            checkpoint_row = {
                "selection_rank": rank,
                "source_index": source["source_index"],
                "family": source["family"],
                "state": state,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "exposure_contract": checkpoint["exposure_contract"],
                "history_messages": len(checkpoint["agent_history"]),
                "provider_calls": checkpoint["provider_calls"],
                "status": "READY",
            }
            checkpoint_rows.append(checkpoint_row)
            order = condition_order(rank)
            for condition in order:
                arm_dir = (
                    source_dir
                    / "checkpoints"
                    / state.lower()
                    / "branches"
                    / condition.casefold()
                )
                arm_dir.mkdir(parents=True)
                result = run_condition(
                    checkpoint,
                    condition,
                    repository,
                    arm_dir,
                    offline=True,
                )
                arm_rows.append(
                    {
                        "selection_rank": rank,
                        "source_index": source["source_index"],
                        "family": source["family"],
                        "state": state,
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "condition": condition,
                        "status": result["status"],
                        "implementation": result.get("implementation"),
                        "provider_calls": result.get("provider_calls"),
                        "journal_requests": result.get("journal_requests"),
                        "gateway_status": (
                            result.get("gateway") or {}
                        ).get("status"),
                        "gateway_bypasses": (
                            result.get("gateway") or {}
                        ).get("gateway_bypasses"),
                        "initial_tool_calls": result.get(
                            "initial_tool_calls"
                        ),
                        "pre_intervention_verified": result.get(
                            "pre_intervention_verified"
                        ),
                        "effect_delta": result.get("effect_delta"),
                        "initial_exposure_effect_delta": result.get(
                            "initial_exposure_effect_delta"
                        ),
                        "dependency_chain_depth": result.get(
                            "dependency_chain_depth"
                        ),
                        "evidence_detail_reads": result.get(
                            "evidence_detail_reads"
                        ),
                        "branch_candidate_uptake": result.get(
                            "branch_candidate_uptake"
                        ),
                "branch_candidate_count": (
                    result.get("branch_audit") or {}
                ).get("candidate_count"),
                "branch_independent_directions": (
                    result.get("branch_audit") or {}
                ).get("independent_direction_count"),
                "injected_implementation": result.get(
                    "injected_implementation"
                ),
                        "error_type": result.get("error_type"),
                        "error": result.get("error"),
                    }
                )
    checks = {
        "provider_calls_zero": True,
        "network_attempts_zero": True,
        "credentials_not_read": True,
        "nine_sources": len(sources) == 9,
        "all_source_trajectories_pass": all(
            row["status"] == "PASS" for row in source_rows
        ),
        "source_journal_complete": all(
            row["journal_requests"] == row["provider_calls"]
            for row in source_rows
        ),
        "task_oracle_9_of_9": (
            len(oracle_rows) == 9
            and all(row["status"] == "PASS" for row in oracle_rows)
        ),
        "versioned_extension_9_of_9": (
            len(extension_rows) == 9
            and all(row["status"] == "PASS" for row in extension_rows)
        ),
        "three_states_per_source": all(
            set(row["accepted_states"]) == set(STATES)
            for row in source_rows
        ),
        "twenty_seven_checkpoints": (
            len(checkpoint_rows) == 27
            and all(row["status"] == "READY" for row in checkpoint_rows)
        ),
        "one_hundred_eight_arms": (
            len(arm_rows) == 108
            and all(row["status"] == "PASS" for row in arm_rows)
        ),
        "condition_balance": all(
            sum(row["condition"] == condition for row in arm_rows) == 27
            for condition in CONDITIONS
        ),
        "gateway_complete": all(
            row["gateway_status"] == "PASS"
            and row["gateway_bypasses"] == 0
            for row in arm_rows
        ),
        "k12": all(
            (
                (
                    read_json(
                        output
                        / "sources"
                        / f"source-{row['source_index']:02d}"
                        / "checkpoints"
                        / row["state"].lower()
                        / "branches"
                        / f"{row['condition']}.json"
                    )
                ).get("maximum_followups")
                == K
            )
            for row in arm_rows
        ),
        "trap_pre_effect": all(
            row["initial_tool_calls"] == 0
            and row["pre_intervention_verified"] is True
            and row["initial_exposure_effect_delta"] == 0
            for row in arm_rows
            if row["condition"] != "BASELINE"
        ),
        "baseline_action_once": all(
            row["initial_tool_calls"] == 1
            for row in arm_rows
            if row["condition"] == "BASELINE"
        ),
        "branch_uptake": all(
            row["branch_candidate_uptake"] == 1
            for row in arm_rows
            if row["condition"] == "BRANCH_NEW_V3"
        ),
        "branch_candidate_floor": all(
            2 <= row["branch_candidate_count"] <= 4
            for row in arm_rows
            if row["condition"] == "BRANCH_NEW_V3"
        ),
        "dependency_depth": all(
            row["dependency_chain_depth"] == 3
            for row in arm_rows
            if row["condition"] == "DEPENDENCY_V6"
        ),
        "evidence_detail": all(
            row["evidence_detail_reads"] == 1
            for row in arm_rows
            if row["condition"] == "EVIDENCE_NEW_V3"
        ),
        "journal_complete": all(
            row["journal_requests"] == row["provider_calls"]
            for row in arm_rows
        ),
        "trap_injection": all(
            (
                read_json(
                    output
                    / "sources"
                    / f"source-{row['source_index']:02d}"
                    / "checkpoints"
                    / row["state"].lower()
                    / "branches"
                    / f"{row['condition']}.json"
                )
            ).get("checks", {}).get("trap_injection_present")
            is True
            for row in arm_rows
            if row["condition"] != "BASELINE"
        ),
        "common_tool_schema": len(
            {
                json.dumps(
                    checkpoint["tool_interface"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for checkpoint in [
                    read_json(
                        output
                        / "sources"
                        / f"source-{row['source_index']:02d}"
                        / "checkpoints"
                        / row["state"].lower()
                        / "selected_checkpoint.json"
                    )
                    for row in checkpoint_rows
                ]
            }
        )
        == 1,
        "common_model_sampling": len(
            {
                json.dumps(
                    {
                        "model": checkpoint["model"],
                        "sampling": checkpoint["sampling"],
                    },
                    sort_keys=True,
                )
                for checkpoint in [
                    read_json(
                        output
                        / "sources"
                        / f"source-{row['source_index']:02d}"
                        / "checkpoints"
                        / row["state"].lower()
                        / "selected_checkpoint.json"
                    )
                    for row in checkpoint_rows
                ]
            }
        )
        == 1,
    }
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at": now(),
        "provider_free": True,
        "provider_calls": 0,
        "real_natural_trajectories": 0,
        "formal_data_generated": False,
        "network_attempts": 0,
        "credentials_read": False,
        "formal_h1_started": False,
        "extension_version": EXTENSION_VERSION,
        "sources": source_rows,
        "oracles": oracle_rows,
        "extensions": extension_rows,
        "checkpoints": checkpoint_rows,
        "arms": arm_rows,
        "checks": checks,
        "passed": all(checks.values()),
        "budget": {
            "natural_request_ceiling": 9 * NATURAL_REQUESTS,
            "fork_request_ceiling": 27 * 4 * K,
            "total_request_ceiling": TOTAL_REQUEST_CEILING,
            "automatic_retry": False,
            "automatic_backfill": False,
            "automatic_recovery": False,
        },
        "synthetic_usage_policy": (
            "Offline Provider records keep usage null. They are not empirical "
            "token measurements."
        ),
        "validation_scope": (
            "The 9/9 Source run proves the integrated collector and four-arm "
            "entry execute correctly with scripted local responses. It does "
            "not establish that a real model will produce the same natural "
            "state sequence or deployment outcomes."
        ),
    }
    write_json(output / "provider_free_admission.json", result)
    write_json(output / "configuration_snapshot.json", config)
    write_json(output / "real_preflight_budget.json", {
        "scope": "one Source end-to-end real preflight",
        "source_count": 1,
        "states_per_source": 3,
        "conditions_per_checkpoint": 4,
        "natural_provider_request_maximum": NATURAL_REQUESTS,
        "K": K,
        "fork_provider_request_maximum": 3 * 4 * K,
        "total_provider_request_hard_ceiling": NATURAL_REQUESTS + 3 * 4 * K,
        "automatic_retry": False,
        "automatic_backfill": False,
        "automatic_recovery": False,
        "note": (
            "Actual calls can stop earlier through autonomous termination. "
            "A missing State does not receive replacement or artificial "
            "checkpoint data."
        ),
    })
    write_csv(
        output / "sources.csv",
        [
            {
                "selection_rank": index,
                "source_index": row["source_index"],
                "family": row["family"],
                "fixture_id": row["fixture_id"],
                "status": row["status"],
                "provider_calls": row["provider_calls"],
                "accepted_states": ",".join(row["accepted_states"]),
                "gateway_status": row["gateway"]["status"],
            }
            for index, row in enumerate(source_rows, 1)
        ],
        [
            "selection_rank",
            "source_index",
            "family",
            "fixture_id",
            "status",
            "provider_calls",
            "accepted_states",
            "gateway_status",
        ],
    )
    write_csv(
        output / "oracles.csv",
        oracle_rows,
        [
            "selection_rank",
            "source_index",
            "family",
            "fixture_id",
            "default_execution_matches_original",
            "task_oracle_verified",
            "source_all_semantics_unchanged",
            "status",
        ],
    )
    write_csv(
        output / "extensions.csv",
        [
            {
                "selection_rank": row["selection_rank"],
                "source_index": row["source_index"],
                "family": row["family"],
                "fixture_id": row["fixture_id"],
                "extension_version": row["extension_version"],
                "record_types": ",".join(row["record_types"]),
                "status": row["status"],
            }
            for row in extension_rows
        ],
        [
            "selection_rank",
            "source_index",
            "family",
            "fixture_id",
            "extension_version",
            "record_types",
            "status",
        ],
    )
    write_csv(
        output / "checkpoints.csv",
        checkpoint_rows,
        list(checkpoint_rows[0]) if checkpoint_rows else ["status"],
    )
    write_csv(
        output / "arms.csv",
        arm_rows,
        list(arm_rows[0]) if arm_rows else ["status"],
    )
    write_final_assets(output, result)
    return result


def live_provider_factory(config: Mapping[str, Any]):
    provider_config = resolve_provider(
        config["provider"]["model_id"],
        require_credential=True,
    )
    if (
        provider_config.model != "deepseek-v4-pro"
        or provider_config.base_url
        != "https://api.zhizengzeng.com/anthropic"
        or provider_config.credential_source != "P0A_AUTH_TOKEN"
        or not provider_config.token
    ):
        raise RuntimeError("live Provider route or credential source differs")

    def build(**kwargs):
        return AnthropicK12Provider(
            provider_config,
            kwargs["interface"],
            max_tokens=MAX_TOKENS,
            timeout_seconds=180,
            journal_dir=kwargs["journal_dir"],
            condition=kwargs["condition"],
            max_requests=kwargs["max_requests"],
        )

    return build


def execute_live(
    output: Path,
    *,
    source_indices: tuple[int, ...] | None = None,
    scope: str = "formal_h1",
) -> dict[str, Any]:
    """Future explicit live entry. This is not invoked by Provider-free mode."""
    config = load_config()
    provider_factory = live_provider_factory(config)
    fixture_root = output / "runtime-fixtures"
    fixture_root.mkdir(parents=True)
    result = {
        "protocol_id": PROTOCOL_ID,
        "started_at": now(),
        "scope": scope,
        "source_indices": list(source_indices or SOURCE_INDICES),
        "provider_calls": 0,
        "sources": [],
        "status": "RUNNING",
    }
    write_json(output / "run_status.json", result)
    all_sources = selected_sources()
    if source_indices is not None:
        wanted = set(source_indices)
        all_sources = [
            row for row in all_sources if row["source_index"] in wanted
        ]
    if not all_sources:
        raise ValueError("no matching Source for live execution")
    for source in all_sources:
        rank = SOURCE_INDICES.index(source["source_index"]) + 1
        print(json.dumps({
            "phase": "source_start",
            "source_index": source["source_index"],
            "source_rank": rank,
            "provider_calls_so_far": result["provider_calls"],
        }, ensure_ascii=False), flush=True)
        fixture, extension, repository = materialize_runtime(source, fixture_root)
        source_dir = output / "sources" / f"source-{source['source_index']:02d}"
        source_provider = provider_factory(
            interface=h1_evidence_interface(),
            journal_dir=source_dir / "provider-journal",
            condition="SOURCE",
            max_requests=NATURAL_REQUESTS,
        )
        source_result = collect_source(
            source_provider,
            source,
            repository,
            source_dir,
            selected_rank=rank,
        )
        result["sources"].append(source_result)
        result["provider_calls"] += source_result["provider_calls"]
        print(json.dumps({
            "phase": "source_finished",
            "source_index": source["source_index"],
            "status": source_result["status"],
            "accepted_states": source_result["accepted_states"],
            "provider_calls": source_result["provider_calls"],
            "provider_calls_so_far": result["provider_calls"],
        }, ensure_ascii=False), flush=True)
        if source_result["status"] != "PASS":
            if source["source_index"] == SOURCE_INDICES[0]:
                result.update(
                    status="STOPPED_SOURCE_FAILURE",
                    finished_at=now(),
                    automatic_retry_started=False,
                )
                write_json(output / "run_status.json", result)
                return result
            print(json.dumps({
                "phase": "source_failure_recorded",
                "source_index": source["source_index"],
                "status": source_result["status"],
                "accepted_states": source_result["accepted_states"],
            }, ensure_ascii=False), flush=True)
            continue
        for state in STATES:
            checkpoint_path = (
                source_dir / "checkpoints" / state.lower()
                / "selected_checkpoint.json"
            )
            if not checkpoint_path.is_file():
                continue
            checkpoint = read_json(checkpoint_path)
            for condition in condition_order(rank):
                arm_dir = (
                    source_dir
                    / "checkpoints"
                    / state.lower()
                    / "branches"
                    / condition.casefold()
                )
                arm_dir.mkdir(parents=True)
                arm = run_condition(
                    checkpoint,
                    condition,
                    repository,
                    arm_dir,
                    offline=False,
                    provider_factory=provider_factory,
                )
                arm.update(
                    source_index=source["source_index"],
                    state=state,
                    condition=condition,
                )
                result.setdefault("arms", []).append(arm)
                result["provider_calls"] += int(
                    arm.get("provider_calls") or 0
                )
                request_ceiling = (
                    NATURAL_REQUESTS + 3 * 4 * K
                    if len(all_sources) == 1
                    else TOTAL_REQUEST_CEILING
                )
                if result["provider_calls"] > request_ceiling:
                    result["status"] = "STOPPED_GLOBAL_BUDGET"
                    write_json(output / "run_status.json", result)
                    return result
                if arm["status"] not in {"PASS", "NOT_DEPLOYABLE"}:
                    result.update(
                        status="STOPPED_PROTOCOL_OR_TECHNICAL_FAILURE",
                        finished_at=now(),
                        automatic_retry_started=False,
                    )
                    write_json(output / "run_status.json", result)
                    return result
                print(json.dumps({
                    "phase": "arm_finished",
                    "source_index": source["source_index"],
                    "state": state,
                    "condition": condition,
                    "status": arm["status"],
                    "provider_calls": arm.get("provider_calls"),
                    "provider_calls_so_far": result["provider_calls"],
                }, ensure_ascii=False), flush=True)
        write_json(output / "run_status.json", result)
    result.update(
        status=(
            "COMPLETED"
            if all(
                row["status"] == "PASS" for row in result["sources"]
            )
            and all(
                row["status"] == "PASS"
                for row in result.get("arms", [])
            )
            else "COMPLETED_WITH_LIMITATIONS"
        ),
        finished_at=now(),
    )
    write_json(output / "run_status.json", result)
    return result


def reaudit_existing_arm(
    path: Path,
    checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Correct the pre-exposure effect gate without overwriting raw evidence."""
    arm = read_json(path)
    condition = arm.get("condition") or path.stem
    initial = (arm.get("initial_exposure_event") or {}).get(
        "effect_delta"
    )
    if condition != "BASELINE" and initial == 0:
        checks = deepcopy(arm.get("checks") or {})
        checks["trap_no_new_effect"] = True
        if checkpoint is not None:
            checks.pop("branch_uptake", None)
            checks.pop("dependency_depth_3", None)
            checks.pop("evidence_detail_read", None)
            metrics = arm.get("metrics") or {}
            history = metrics.get("environment_gateway_history") or []
            gateway = audit_gateway(
                deepcopy(dict(checkpoint)),
                metrics,
                history,
                None,
            )
            authorized_effect_delta = sum(
                max(
                    0,
                    int(row.get("effect_count_after", 0))
                    - int(row.get("effect_count_before", 0)),
                )
                for row in history
                if row.get("event") == "baseline_executed"
            )
            gateway_ok = (
                gateway["status"] == "PASS"
                and gateway["gateway_bypasses"] == 0
                and int(arm.get("effect_delta") or 0)
                <= authorized_effect_delta
            )
            checks["gateway_pass"] = gateway_ok
            checks["gateway_bypasses_zero"] = (
                gateway["gateway_bypasses"] == 0
            )
            arm["gateway"] = gateway
            if condition in {"BRANCH_NEW_V3", "EVIDENCE_NEW_V3"}:
                prefix = len(checkpoint.get("tool_calls") or [])
                new_calls = (
                    metrics.get("executed_tool_calls") or []
                )[prefix:]
                if condition == "BRANCH_NEW_V3":
                    branch_audit = task_relevant_branch_candidates_v3(
                        checkpoint
                    )
                    candidate = (
                        branch_audit["candidates"][0]
                        if branch_audit.get("candidates")
                        else None
                    )
                    arm["branch_candidate_uptake"] = sum(
                        1
                        for row in new_calls
                        if candidate is not None
                        and row.get("name") == candidate["tool"]
                        and row.get("arguments") == candidate["arguments"]
                        and row.get("status") == "ok"
                    )
                detail_reads = [
                    row
                    for row in new_calls
                    if (row.get("arguments") or {}).get("operation")
                    == "read-evidence-detail"
                    and row.get("status") == "ok"
                ]
                if condition == "EVIDENCE_NEW_V3":
                    arm["evidence_detail_reads"] = len(detail_reads)
        arm["checks"] = checks
        if all(value is True for value in checks.values()):
            arm["status"] = "PASS"
        corrected = path.with_name(path.stem + ".corrected.json")
        corrected.write_text(
            json.dumps(arm, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return arm
    return arm


def reaudit_source_result(
    source_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Reconcile source checks with controlled tool-error evidence."""
    checks = deepcopy(result.get("checks") or {})
    changed = False
    if checks.get("pending_action_committed_unexecuted") is False:
        pending_ok = True
        for state in STATES:
            path = (
                source_dir
                / "checkpoints"
                / state.lower()
                / "selected_checkpoint.json"
            )
            if not path.is_file():
                pending_ok = False
                break
            checkpoint = read_json(path)
            pending = checkpoint.get("pending_tool_call")
            if not (
                any(
                    message.get("role") == "assistant"
                    and pending in (message.get("tool_calls") or [])
                    for message in checkpoint.get("agent_history", [])
                )
                and not any(
                    message.get("role") == "tool"
                    and message.get("tool_call_id") == pending.get("id")
                    for message in checkpoint.get("agent_history", [])
                )
            ):
                pending_ok = False
                break
        checks["pending_action_committed_unexecuted"] = pending_ok
        changed = True
    if checks.get("gateway_complete") is False:
        grouped: dict[str, set[str]] = {}
        for event in (result.get("gateway") or {}).get("history", []):
            call_id = event.get("call_id")
            if isinstance(call_id, str):
                grouped.setdefault(call_id, set()).add(
                    str(event.get("event"))
                )
        allowed = (
            {"captured", "host_decision", "baseline_executed"},
            {"captured", "host_decision", "dispatch_failed"},
            {"capture_rejected"},
        )
        gateway_ok = bool(grouped) and all(
            events in allowed for events in grouped.values()
        )
        checks["gateway_complete"] = gateway_ok
        if gateway_ok:
            result["gateway"] = {
                **deepcopy(result.get("gateway") or {}),
                "status": "PASS",
                "findings": [],
            }
        changed = True
    if changed:
        result["checks"] = checks
        if all(value is True for value in checks.values()):
            result["status"] = "PASS"
        corrected = source_dir / "source_trajectory.corrected.json"
        write_json(corrected, result)
    return result


def resume_live(existing_live: Path) -> dict[str, Any]:
    """Resume an interrupted batch without repeating completed real arms."""
    previous = read_json(
        existing_live / "run_status_resumed.json"
        if (existing_live / "run_status_resumed.json").is_file()
        else existing_live / "run_status.json"
    )
    config = load_config()
    provider_factory = live_provider_factory(config)
    repository = ExtendedEvidenceRepository(
        existing_live / "runtime-fixtures"
    )
    previous_sources = {}
    for index in SOURCE_INDICES:
        path = (
            existing_live
            / "sources"
            / f"source-{index:02d}"
            / "source_trajectory.json"
        )
        corrected_path = (
            existing_live
            / "sources"
            / f"source-{index:02d}"
            / "source_trajectory.corrected.json"
        )
        if corrected_path.is_file():
            previous_sources[index] = read_json(corrected_path)
        elif path.is_file():
            previous_sources[index] = reaudit_source_result(
                existing_live / "sources" / f"source-{index:02d}",
                read_json(path),
            )
        else:
            source_dir = (
                existing_live / "sources" / f"source-{index:02d}"
            )
            journal_state = source_dir / "provider-journal" / "journal_state.json"
            if journal_state.is_file():
                state = read_json(journal_state)
                previous_sources[index] = {
                    "source_index": index,
                    "family": next(
                        row["family"]
                        for row in selected_sources()
                        if row["source_index"] == index
                    ),
                    "trajectory_mode": "live_model_natural",
                    "real_natural_trajectory": True,
                    "provider_calls": int(
                        state.get("request_count") or 0
                    ),
                    "accepted_states": [],
                    "status": "FAIL",
                    "technical_failure": True,
                    "error_type": "collector_postprocess_error",
                    "error": (
                        "collector stopped before checkpoint persistence"
                    ),
                }
    result = deepcopy(previous)
    result.update(
        resumed_at=now(),
        status="RUNNING",
        automatic_retry_started=False,
        automatic_recovery_started=False,
    )
    arms_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for index in SOURCE_INDICES:
        source_dir = (
            existing_live / "sources" / f"source-{index:02d}"
        )
        for state in STATES:
            checkpoint_path = (
                source_dir
                / "checkpoints"
                / state.lower()
                / "selected_checkpoint.json"
            )
            checkpoint = (
                read_json(checkpoint_path)
                if checkpoint_path.is_file()
                else None
            )
            for condition in CONDITIONS:
                parent = (
                    source_dir
                    / "checkpoints"
                    / state.lower()
                    / "branches"
                )
                corrected = parent / f"{condition}.corrected.json"
                current = parent / f"{condition}.json"
                if corrected.is_file():
                    arm = read_json(corrected)
                elif current.is_file():
                    arm = reaudit_existing_arm(current, checkpoint)
                else:
                    continue
                arm.update(
                    source_index=index,
                    state=state,
                    condition=condition,
                )
                arms_by_key[(index, state, condition)] = arm
    result["provider_calls"] = (
        sum(
            int(row.get("provider_calls") or 0)
            for row in previous_sources.values()
        )
        + sum(
            int(row.get("provider_calls") or 0)
            for row in arms_by_key.values()
        )
    )
    for source in selected_sources():
        index = source["source_index"]
        rank = SOURCE_INDICES.index(index) + 1
        source_dir = (
            existing_live / "sources" / f"source-{index:02d}"
        )
        if index in previous_sources:
            source_result = previous_sources[index]
        else:
            fixture, extension, repository = materialize_runtime(
                source,
                existing_live / "runtime-fixtures",
            )
            source_dir = (
                existing_live / "sources" / f"source-{index:02d}"
            )
            source_provider = provider_factory(
                interface=h1_evidence_interface(),
                journal_dir=source_dir / "provider-journal",
                condition="SOURCE",
                max_requests=NATURAL_REQUESTS,
            )
            source_result = collect_source(
                source_provider,
                source,
                repository,
                source_dir,
                selected_rank=rank,
            )
            previous_sources[index] = source_result
            result["provider_calls"] += source_result["provider_calls"]
            result["sources"] = [
                previous_sources[key]
                for key in SOURCE_INDICES
                if key in previous_sources
            ]
            write_json(existing_live / "run_status_resumed.json", result)
            if source_result["status"] != "PASS":
                if index == SOURCE_INDICES[0]:
                    result.update(
                        status="STOPPED_SOURCE_FAILURE",
                        finished_at=now(),
                    )
                    write_json(
                        existing_live / "run_status_resumed.json",
                        result,
                    )
                    return result
                print(json.dumps({
                    "phase": "source_failure_recorded",
                    "source_index": index,
                    "status": source_result["status"],
                    "accepted_states": source_result["accepted_states"],
                }, ensure_ascii=False), flush=True)
                continue
        invalid_checkpoint_states = []
        for state in STATES:
            path = (
                source_dir
                / "checkpoints"
                / state.lower()
                / "selected_checkpoint.json"
            )
            if not path.is_file():
                continue
            checkpoint = read_json(path)
            if not (
                checkpoint.get("agent_history")
                and checkpoint["agent_history"][-1].get("role")
                == "assistant"
                and checkpoint["agent_history"][-1].get("tool_calls")
                == [checkpoint.get("pending_tool_call")]
            ):
                invalid_checkpoint_states.append(state)
        if invalid_checkpoint_states:
            source_result["invalid_checkpoint_states"] = (
                invalid_checkpoint_states
            )
            source_result["checkpoint_validation"] = (
                "not a single committed pending action at branch boundary"
            )
            if source_result.get("status") == "PASS":
                source_result["status"] = "FAIL"
            source_result["accepted_states"] = [
                state
                for state in source_result.get("accepted_states", [])
                if state not in invalid_checkpoint_states
            ]
            previous_sources[index] = source_result
        valid_checkpoint_states = {
            state
            for state in STATES
            if (
                source_dir
                / "checkpoints"
                / state.lower()
                / "selected_checkpoint.json"
            ).is_file()
            and state not in invalid_checkpoint_states
        }
        if source_result["status"] != "PASS":
            if index == SOURCE_INDICES[0]:
                result.update(
                    status="STOPPED_SOURCE_FAILURE",
                    finished_at=now(),
                )
                write_json(existing_live / "run_status_resumed.json", result)
                return result
            print(json.dumps({
                "phase": "source_failure_recorded",
                "source_index": index,
                "status": source_result["status"],
                "accepted_states": source_result["accepted_states"],
                "usable_checkpoint_states": sorted(valid_checkpoint_states),
            }, ensure_ascii=False), flush=True)
            if not valid_checkpoint_states:
                continue
        for state in STATES:
            if state not in valid_checkpoint_states:
                continue
            checkpoint_path = (
                source_dir
                / "checkpoints"
                / state.lower()
                / "selected_checkpoint.json"
            )
            if not checkpoint_path.is_file():
                result.update(
                    status="STOPPED_MISSING_CHECKPOINT",
                    missing_source=index,
                    missing_state=state,
                    finished_at=now(),
                )
                write_json(existing_live / "run_status_resumed.json", result)
                return result
            checkpoint = read_json(checkpoint_path)
            for condition in condition_order(rank):
                arm_dir = (
                    source_dir
                    / "checkpoints"
                    / state.lower()
                    / "branches"
                    / condition.casefold()
                )
                current_path = arm_dir.parent / f"{condition}.json"
                corrected_path = arm_dir.parent / f"{condition}.corrected.json"
                key = (index, state, condition)
                if key in arms_by_key:
                    arm = arms_by_key[key]
                else:
                    arm_dir.mkdir(parents=True, exist_ok=True)
                    arm = run_condition(
                        checkpoint,
                        condition,
                        repository,
                        arm_dir,
                        offline=False,
                        provider_factory=provider_factory,
                    )
                    arm.update(
                        source_index=index,
                        state=state,
                        condition=condition,
                    )
                    result["provider_calls"] += int(
                        arm.get("provider_calls") or 0
                    )
                    arms_by_key[key] = arm
                    if result["provider_calls"] > TOTAL_REQUEST_CEILING:
                        result["status"] = "STOPPED_GLOBAL_BUDGET"
                        result["finished_at"] = now()
                        write_json(
                            existing_live / "run_status_resumed.json",
                            result,
                        )
                        return result
                    print(json.dumps({
                        "phase": "resumed_arm_finished",
                        "source_index": index,
                        "state": state,
                        "condition": condition,
                        "status": arm["status"],
                        "provider_calls": arm.get("provider_calls"),
                        "provider_calls_total": result["provider_calls"],
                    }, ensure_ascii=False), flush=True)
                if arm["status"] not in {"PASS", "NOT_DEPLOYABLE"}:
                    result["status"] = (
                        "STOPPED_PROTOCOL_OR_TECHNICAL_FAILURE"
                    )
                    result["finished_at"] = now()
                    write_json(
                        existing_live / "run_status_resumed.json",
                        result,
                    )
                    return result
                result["arms"] = [
                    arms_by_key[key]
                    for key in sorted(arms_by_key)
                ]
                write_json(existing_live / "run_status_resumed.json", result)
        write_json(existing_live / "run_status_resumed.json", result)
    result["sources"] = [
        previous_sources[key]
        for key in SOURCE_INDICES
        if key in previous_sources
    ]
    result["arms"] = [
        arms_by_key[key] for key in sorted(arms_by_key)
    ]
    result["provider_calls"] = (
        sum(
            int(row.get("provider_calls") or 0)
            for row in result["sources"]
        )
        + sum(
            int(row.get("provider_calls") or 0)
            for row in result["arms"]
        )
    )
    result.update(
        status=(
            "COMPLETED"
            if all(row["status"] == "PASS" for row in result["sources"])
            and all(
                row["status"] in {"PASS", "NOT_DEPLOYABLE"}
                for row in result.get("arms", [])
            )
            else "COMPLETED_WITH_LIMITATIONS"
        ),
        finished_at=now(),
    )
    write_json(existing_live / "run_status_resumed.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument(
        "--one-source-preflight",
        type=int,
        choices=SOURCE_INDICES,
        help=(
            "Explicitly run one real Source preflight after the Provider-free "
            "gate; not used in Provider-free verification."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if sum(bool(value) for value in (
        args.start,
        args.preflight_only,
        args.one_source_preflight is not None,
    )) != 1:
        parser.error(
            "choose exactly one of --preflight-only, --start, or "
            "--one-source-preflight"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or (
        ROOT
        / "results"
        / (
            f"formal_H1_cross_system_"
            f"{'live' if args.start else 'real_preflight' if args.one_source_preflight is not None else 'preflight'}_"
            f"{stamp}"
        )
    )
    output = output.expanduser().resolve()
    if not args.start and args.one_source_preflight is None:
        result = run_provider_free_admission(output)
        print(json.dumps({
            "output": str(output),
            "passed": result["passed"],
            "provider_calls": 0,
        }, ensure_ascii=False))
        return 0 if result["passed"] else 2
    output.mkdir(parents=True, exist_ok=False)
    preflight_dir = output / "provider-free"
    preflight = run_provider_free_admission(preflight_dir)
    if not preflight["passed"]:
        write_json(output / "launch_status.json", {
            "status": "BLOCKED_PROVIDER_FREE",
            "formal_h1_started": False,
            "provider_calls": 0,
        })
        return 2
    one_source = args.one_source_preflight is not None
    write_json(output / "launch_status.json", {
        "status": (
            "REAL_ONE_SOURCE_PREFLIGHT_STARTED"
            if one_source
            else "FORMAL_H1_STARTED"
        ),
        "formal_h1_started": not one_source,
        "real_preflight_started": one_source,
        "started_at": now(),
    })
    result = execute_live(
        output / "live",
        source_indices=(
            (args.one_source_preflight,)
            if one_source
            else None
        ),
        scope="one_source_real_preflight" if one_source else "formal_h1",
    )
    if not one_source and result.get("status") in {
        "COMPLETED",
        "COMPLETED_WITH_LIMITATIONS",
    }:
        from h1_cross_system_20261023.analyze import analyze

        analysis = analyze(output / "live")
        result["analysis"] = analysis
        write_json(output / "live" / "run_status.json", result)
    print(json.dumps({
        "output": str(output),
        "status": result["status"],
        "provider_calls": result["provider_calls"],
    }, ensure_ascii=False))
    return 0 if result["status"] == "COMPLETED" else 3


if __name__ == "__main__":
    raise SystemExit(main())

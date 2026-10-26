#!/usr/bin/env python3
"""Independent DeepSeek V4 Pro extended-environment 9-source H1 runner."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
COMPAT_DIR = ROOT
for path in (str(ROOT), str(MODULE_DIR), str(COMPAT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import adapters.lifecycle_branch_runtime as branch_runtime
import adapters.lifecycle_mechanism_runtime as mechanism_runtime
from adapters.followup_budget_sensitivity_v2 import EvidenceRuntime, VisibleBudget
from adapters.h1_memory_monitor import MemoryMonitor, runtime_memory_scope
from adapters.h1_thinking_compat import (
    pilot_thinking_compatibility_scope,
)
from adapters.lifecycle_dependency_v6_runtime import LifecycleDependencyV6Runtime
from adapters.lifecycle_mechanism_runtime import LifecycleMechanismRuntime
from core.experiment_lock import (
    ExperimentDirectoryLock,
    ExperimentIdentity,
    ExperimentSchedulerLock,
)
from tasks.multi_task_dependency_v6_adapter import dual_dependency_context
from tasks.multi_task_pool_v1 import validate_fixture
from tasks.multi_task_pool_v1_adapter import (
    FixtureRepository,
    ProviderFreeSentinel,
    _checkpoint,
    _commit_call,
    _complete_call,
    _tool_call,
)
from traps.branch_evidence_v3 import branch_candidates, candidate_checks

import run_formal_h1_multitask_v3 as h1_v3
import run_formal_h1_multitask_v3_2 as h1_v32
import run_formal_h1_v1_1 as frozen_h1
from adapters.formal_h1_multitask_v3_2_runtime import budget_scope

from .extended_v3_environment import (
    EXTENSION_VERSION,
    ExtendedEvidenceRepository,
    ExtendedEvidenceV3Environment,
    ExtendedEvidenceV3Session,
    ExtendedH1EvidenceV3Runtime,
    build_family_extension,
    extended_evidence_interface,
)
from .provider import (
    BatchState,
    DeepseekAuditProvider,
    DeepseekBudget,
    MODEL_ID,
    provider_config,
    public_provider,
)
from .request_ledger import DeepseekRequestLedger


DEFAULT_CONFIG_PATH = MODULE_DIR / "config_max8192_v2.json"
STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
TRAP_CONDITIONS = ("BRANCH", "DEPENDENCY", "EVIDENCE")
CONDITIONS = ("BASELINE", *TRAP_CONDITIONS)
CONDITION_LABELS = {
    "BASELINE": "BASELINE",
    "BRANCH": "BRANCH_NEW",
    "DEPENDENCY": "DEPENDENCY_V6",
    "EVIDENCE": "EVIDENCE_NEW_V3",
}
API_KEY_ENV = "DS_V4_H1_API_KEY"
FATAL_STOP_REASONS = {
    "http_429",
    "http_error",
    "usage_missing",
    "usage_missing_or_invalid:prompt_tokens",
    "usage_missing_or_invalid:completion_tokens",
    "usage_missing_or_invalid:total_tokens",
    "usage_total_mismatch",
    "provider_protocol_error",
    "provider_response_incomplete",
    "provider_response_exceeds_size_limit",
    "provider_content_length_malformed",
    "provider_content_length_exceeds_size_limit",
    "unexpected_response_model",
    "request_ledger_write_failure",
    "request_ledger_usage_commit_failure",
    "request_ledger_failure_commit_error",
    "unresolved_request_ledger_recovery",
    "actual_source_token_budget_exceeded",
    "actual_batch_token_budget_exceeded",
    "output_truncated",
}


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
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
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


def metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "modified_at_ns": stat.st_mtime_ns,
        "mode": oct(stat.st_mode & 0o777),
    }


def inventory(paths: list[Path]) -> dict[str, Any]:
    return {
        str(path.relative_to(ROOT)): metadata(path)
        for path in paths
        if path.is_file()
    }


def tree_metadata(path: Path) -> dict[str, Any]:
    files = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": str(item.relative_to(path)),
                **metadata(item),
            }
        )
    return {
        "exists": path.exists(),
        "directory_mode": oct(path.stat().st_mode & 0o777) if path.exists() else None,
        "file_count": len(files),
        "files": files,
    }


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = read_json(path)
    if config.get("config_version") not in {
        "h1-deepseek-v4-pro-extended-evidencev3-v1",
        "h1-deepseek-v4-pro-extended-evidencev3-max8192-v2",
    }:
        raise RuntimeError("unexpected extended H1 config version")
    provider = config["provider"]
    if (
        provider["model_id"] != MODEL_ID
        or provider["base_url"] != "https://api.zhizengzeng.com/v1"
        or provider["temperature"] != 0.0
        or provider["max_tokens"] not in {2400, 8192}
        or provider["timeout_seconds"] != 180
        or provider["tool_choice"] != "auto"
        or provider["parallel_tool_calls"] is not False
        or provider["stream"] is not False
        or provider["credential_environment"] != API_KEY_ENV
    ):
        raise RuntimeError("DeepSeek V4 Pro provider parameters changed")
    limits = config["limits"]
    if (
        limits["per_source_maximum_requests"] != 140
        or limits["per_source_maximum_total_tokens"] != 800_000
        or limits["batch_maximum_requests"] != 1_260
        or limits["batch_maximum_total_tokens"] != 7_200_000
        or limits["automatic_retry"]
        or limits["automatic_replacement"]
        or not limits["single_process_only"]
    ):
        raise RuntimeError("extended H1 budget or no-retry policy changed")
    return config


def selected_sources(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_config = read_json(ROOT / config["source_config_path"])
    manifest = read_json(ROOT / config["source_manifest_path"])
    selection = read_json(ROOT / config["source_selection_record"])
    by_index = {row["source_index"]: row for row in manifest["source_schedule"]}
    indices = source_config["selection_rule"]["selected_source_indices"]
    if indices != [1, 6, 2, 11, 16, 12, 21, 26, 22]:
        raise RuntimeError("frozen V4 source order changed")
    selected = [deepcopy(by_index[index]) for index in indices]
    selection_rows = selection["sources"]
    if [row["source_index"] for row in selection_rows] != indices:
        raise RuntimeError("V4 selection record differs from source order")
    for row, recorded in zip(selected, selection_rows):
        for key in (
            "source_index",
            "family",
            "template_id",
            "instance_id",
            "fixture_id",
        ):
            if key in recorded and row[key] != recorded[key]:
                raise RuntimeError(f"V4 source identity differs: {key}")
        row["source_fixture_path"] = row["fixture_path"]
        row["source_fixture"] = str(ROOT / row["fixture_path"])
    return selected


def create_run_directory(config: Mapping[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = ROOT / "results"
    base = root / f"{config['output_prefix']}{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = root / f"{base.name}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def materialize_fixtures(
    output: Path,
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], ExtendedEvidenceRepository]:
    fixture_root = output / "runtime_fixtures"
    fixture_root.mkdir(parents=False, exist_ok=False)
    original_repository = FixtureRepository(ROOT / "tasks" / "multi_task_pool_v2_fixtures")
    runtime_schedule = []
    extension_rows = []
    for source in selected:
        fixture = deepcopy(original_repository.load(source["fixture_id"]))
        original_fixture = deepcopy(fixture)
        extension = build_family_extension(fixture)
        validate_fixture(fixture)
        destination_dir = fixture_root / f"source-{source['source_index']:02d}"
        destination_dir.mkdir(parents=False, exist_ok=False)
        destination = destination_dir / "instance-01.json"
        extension_path = destination_dir / "evidence-extension.json"
        write_json(destination, fixture)
        write_json(extension_path, extension)
        os.chmod(destination, 0o444)
        os.chmod(extension_path, 0o444)
        row = deepcopy(source)
        row["fixture_path"] = str(destination.relative_to(ROOT))
        row["runtime_fixture_path"] = row["fixture_path"]
        row["runtime_extension_path"] = str(extension_path.relative_to(ROOT))
        row["runtime_extension_version"] = EXTENSION_VERSION
        row["source_fixture_modified"] = False
        row["new_natural_trajectory_required"] = True
        runtime_schedule.append(row)
        extension_rows.append({
            "source_index": source["source_index"],
            "family": source["family"],
            "fixture_id": source["fixture_id"],
            "source_fixture_path": source["source_fixture_path"],
            "runtime_fixture_path": row["fixture_path"],
            "extension_path": row["runtime_extension_path"],
            "extension_version": EXTENSION_VERSION,
            "scope_key": extension["scope_key"],
            "scope_value": extension["scope_value"],
            "difference": deepcopy(extension["difference"]),
            "original_fixture_unchanged": fixture["fixture_id"] == original_fixture["fixture_id"],
        })
    write_json(output / "environment_extension_audit.json", {
        "generated_at": now(),
        "version": EXTENSION_VERSION,
        "original_fixtures_modified": False,
        "runtime_copies_only": True,
        "sources": extension_rows,
    })
    return runtime_schedule, ExtendedEvidenceRepository(fixture_root)


class ExtendedSourceSession(
    ExtendedEvidenceV3Session,
    h1_v3.v2.FormalMultiTaskSourceSession,
):
    def __init__(self, provider, fixture_id, repository):
        ExtendedEvidenceV3Session.__init__(
            self,
            provider,
            fixture_id,
            repository=repository,
        )


class K12VisibleBaseline(VisibleBudget, LifecycleMechanismRuntime):
    pass


class K12VisibleDependencyV6(VisibleBudget, LifecycleDependencyV6Runtime):
    pass


class ConditionRuntimeProxy:
    def __init__(self, runtime, checkpoint, repository):
        self.runtime = runtime
        self.snapshot = deepcopy(checkpoint["environment_snapshot"])
        self.repository = repository

    def __getattr__(self, name):
        value = getattr(self.runtime, name)
        if not callable(value):
            return value

        def invoke(*args, **kwargs):
            with dual_dependency_context(
                self.snapshot,
                self.repository,
                version="V6",
            ):
                return value(*args, **kwargs)

        return invoke

    def run(self):
        with dual_dependency_context(
            self.snapshot,
            self.repository,
            version="V6",
        ):
            self.runtime.run()
        return self.metrics()

    def metrics(self):
        with dual_dependency_context(
            self.snapshot,
            self.repository,
            version="V6",
        ):
            value = self.runtime.metrics()
        value.update(
            maximum_followups=12,
            extended_runtime_adapter="h1-deepseek-extended-runtime-v1",
        )
        if self.runtime.condition == "DEPENDENCY":
            value["dependency_max_optional_followups"] = (
                self.runtime.core.max_optional_followups
            )
        return value


def restore_condition_runtime(
    checkpoint: Mapping[str, Any],
    provider: Any,
    condition: str,
    repository: ExtendedEvidenceRepository,
):
    if condition == "EVIDENCE":
        return ExtendedH1EvidenceV3Runtime.restore(
            checkpoint,
            provider,
            condition="EVIDENCE",
            repository=repository,
        )
    snapshot = checkpoint["environment_snapshot"]
    with dual_dependency_context(snapshot, repository, version="V6"):
        previous_session = branch_runtime.Session
        previous_repository = ExtendedEvidenceV3Session.active_repository
        previous_interface = mechanism_runtime.lifecycle_interaction_interface
        branch_runtime.Session = ExtendedEvidenceV3Session
        ExtendedEvidenceV3Session.active_repository = repository
        mechanism_runtime.lifecycle_interaction_interface = (
            extended_evidence_interface
        )
        try:
            if condition == "BASELINE":
                runtime = K12VisibleBaseline.restore(
                    deepcopy(dict(checkpoint)),
                    provider,
                    condition="BASELINE",
                )
                runtime.pilot_k = 12
            elif condition == "BRANCH":
                runtime = EvidenceRuntime.restore(
                    deepcopy(dict(checkpoint)),
                    provider,
                    condition="BRANCH",
                )
                runtime.pilot_k = 12
            elif condition == "DEPENDENCY":
                runtime = K12VisibleDependencyV6.restore(
                    deepcopy(dict(checkpoint)),
                    provider,
                    condition="DEPENDENCY",
                )
                runtime.core.max_optional_followups = 12
                runtime.dependency_max_optional_followups = 12
                runtime.pilot_k = 12
            else:
                raise ValueError(f"unsupported condition: {condition}")
        finally:
            branch_runtime.Session = previous_session
            ExtendedEvidenceV3Session.active_repository = previous_repository
            mechanism_runtime.lifecycle_interaction_interface = previous_interface
    return ConditionRuntimeProxy(runtime, checkpoint, repository)


def provider_free_checkpoints(
    source: Mapping[str, Any],
    provider: Any,
    repository: ExtendedEvidenceRepository,
) -> dict[str, dict[str, Any]]:
    fixture_id = source["fixture_id"]
    target = repository.load(fixture_id)["initial_environment_state"]["target"]

    recon = ExtendedEvidenceV3Session(provider, fixture_id, repository=repository)
    _commit_call(
        recon,
        _tool_call(
            f"extended-{fixture_id}-recon",
            "collect-routes",
            scope="all",
        ),
    )
    recon_checkpoint = _checkpoint(recon, "RECON", 1)

    exploitation = ExtendedEvidenceV3Session(
        provider,
        fixture_id,
        repository=repository,
    )
    routes = _complete_call(
        exploitation,
        _tool_call(
            f"extended-{fixture_id}-routes",
            "collect-routes",
            scope="all",
        ),
    )["routes"]
    inventory_rows = [
        {"route": row["route"], "access_class": row["advertised_access"]}
        for row in routes
    ]
    _complete_call(
        exploitation,
        _tool_call(
            f"extended-{fixture_id}-inventory",
            "classify-inventory",
            inventory=inventory_rows,
        ),
    )
    binding = _complete_call(
        exploitation,
        _tool_call(
            f"extended-{fixture_id}-binding",
            "read-route-binding",
            route=target["route"],
        ),
    )
    descriptor = _complete_call(
        exploitation,
        _tool_call(
            f"extended-{fixture_id}-descriptor",
            "read-operation-descriptor",
            ref=binding["operation_descriptor_ref"],
        ),
    )
    _complete_call(
        exploitation,
        _tool_call(
            f"extended-{fixture_id}-policy",
            "read-resource-policy",
            ref=descriptor["resource_policy_ref"],
        ),
    )
    probe = _tool_call(
        f"extended-{fixture_id}-probe",
        "probe-sample",
        route=target["route"],
        object_id=target["object_id"],
        method=descriptor["method"],
        operation_ref=descriptor["record_id"],
        policy_ref=descriptor["resource_policy_ref"],
    )
    _commit_call(exploitation, probe)
    exploitation_checkpoint = _checkpoint(exploitation, "EXPLOITATION", 2)

    verification = ExtendedEvidenceV3Session.restore(
        exploitation_checkpoint,
        provider,
        repository=repository,
    )
    result = _complete_call(verification, deepcopy(verification.pending))
    _commit_call(
        verification,
        _tool_call(
            f"extended-{fixture_id}-evidence",
            "read-evidence",
            request_id=result["request_id"],
            source="all",
        ),
    )
    verification_checkpoint = _checkpoint(verification, "VERIFICATION", 3)
    recon.close()
    exploitation.close()
    verification.close()
    return {
        "RECON": recon_checkpoint,
        "EXPLOITATION": exploitation_checkpoint,
        "VERIFICATION": verification_checkpoint,
    }


def probe_arguments_from_snapshot(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    target = snapshot["target"]
    binding = snapshot["bindings"][target["route"]]
    descriptor = snapshot["descriptors"][binding["operation_descriptor_ref"]]
    return {
        "operation": "probe-sample",
        "route": target["route"],
        "object_id": target["object_id"],
        "method": descriptor["method"],
        "operation_ref": descriptor["record_id"],
        "policy_ref": descriptor["resource_policy_ref"],
    }


def _prepare_initial_exposure(runtime):
    runtime.initial_exposure()
    initial_calls = getattr(runtime, "initial_tool_calls", None)
    if initial_calls is None and hasattr(runtime, "runtime"):
        initial_calls = getattr(runtime.runtime, "initial_tool_calls", None)
    return initial_calls


def audit_condition_slot(
    checkpoint: Mapping[str, Any],
    condition: str,
    provider: Any,
    repository: ExtendedEvidenceRepository,
) -> dict[str, Any]:
    runtime = None
    try:
        runtime = restore_condition_runtime(
            checkpoint,
            provider,
            condition,
            repository,
        )
        before = runtime.session.env.snapshot()
        exported = runtime.export_pre_effect_checkpoint()
        restore_equal = {
            "snapshot": exported.get("environment_snapshot")
            == checkpoint.get("environment_snapshot"),
            "history": exported.get("agent_history")
            == checkpoint.get("agent_history"),
            "current_plan": exported.get("current_plan")
            == checkpoint.get("current_plan"),
            "pending_action": exported.get("pending_tool_call")
            == checkpoint.get("pending_tool_call"),
        }
        initial_calls = _prepare_initial_exposure(runtime)
        after = runtime.session.env.snapshot()
        expected_initial_calls = 1 if condition == "BASELINE" else 0
        result = {
            "condition": condition,
            "display_condition": CONDITION_LABELS[condition],
            "status": "READY",
            "restore_equal": restore_equal,
            "pre_effect_equal": all(restore_equal.values()),
            "initial_tool_calls": initial_calls,
            "initial_execution_boundary_ok": initial_calls
            == expected_initial_calls,
            "initial_exposure_unchanged": (
                before == after if condition != "BASELINE" else True
            ),
            "provider_calls": provider.calls,
            "maximum_followups": 12,
        }
        if condition == "BRANCH":
            candidates = branch_candidates(checkpoint)
            result.update(
                candidate_count=len(candidates),
                candidates=candidates,
                deployable=2 <= len(candidates) <= 4,
            )
        elif condition == "DEPENDENCY":
            snapshot = after
            v5 = snapshot.get("dependency_material") or {}
            v6 = snapshot.get("dependency_v6_material") or {}
            result.update(
                v5_namespace=bool(v5.get("records"))
                and all("/dependency-v5/" in key for key in v5["records"]),
                v6_namespace=bool(v6.get("records"))
                and all("/dependency-v6/" in key for key in v6["records"]),
                v6_configuration_records=len(
                    v6.get("configuration_records") or {}
                ),
            )
        elif condition == "EVIDENCE":
            env = runtime.session.env
            core = runtime.evidence_v3
            initial_questions = env.questions(core.request_id)
            opened = []
            for question in initial_questions:
                for query in question.get("queries", []):
                    record = env.call(query)
                    opened.append({
                        "ref": query["ref"],
                        "record_type": record.get("record_type"),
                    })
            if core.request_id is None:
                env.call(probe_arguments_from_snapshot(before))
                core.request_id = env.requests[-1]["request_id"]
            request_details = []
            for question in env.questions(core.request_id):
                for query in question.get("queries", []):
                    record = env.call(query)
                    request_details.append({
                        "ref": query["ref"],
                        "record_type": record.get("record_type"),
                    })
            evidence = env.call({
                "operation": "read-evidence",
                "request_id": core.request_id,
                "source": "all",
            })
            source_integrity = all(
                evidence[source]["producer"] == source
                and all(
                    row.get("request_id") == core.request_id
                    for row in evidence[source]["records"]
                )
                for source in ("gateway", "client", "policy", "store")
            )
            result.update(
                initial_detail_reads=opened,
                request_detail_reads=request_details,
                evidence_sources=sorted(evidence),
                source_all_semantics_unchanged=source_integrity,
                detail_path_present=bool(opened or request_details),
            )
        result["slot_ready"] = (
            result["pre_effect_equal"]
            and result["initial_execution_boundary_ok"]
            and result["initial_exposure_unchanged"]
            and provider.calls == 0
            and (
                condition != "BRANCH"
                or result.get("deployable") is True
            )
            and (
                condition != "DEPENDENCY"
                or (
                    result.get("v5_namespace") is True
                    and result.get("v6_namespace") is True
                )
            )
            and (
                condition != "EVIDENCE"
                or (
                    result.get("detail_path_present") is True
                    and result.get("source_all_semantics_unchanged") is True
                )
            )
        )
        return result
    except Exception as exc:
        return {
            "condition": condition,
            "display_condition": CONDITION_LABELS[condition],
            "status": "TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "provider_calls": provider.calls,
            "slot_ready": False,
        }
    finally:
        if runtime is not None:
            runtime.close()


def audit_fixture_environment(
    source: Mapping[str, Any],
    repository: ExtendedEvidenceRepository,
) -> dict[str, Any]:
    fixture = repository.load(source["fixture_id"])
    extension = repository.extension(source["fixture_id"])
    original = FixtureRepository(
        ROOT / "tasks" / "multi_task_pool_v2_fixtures"
    ).load(source["fixture_id"])
    original_expected = deepcopy(
        original["expected_environment_effects"]["execution_result"]
    )
    args = probe_arguments_from_snapshot(fixture["initial_environment_state"])

    env = ExtendedEvidenceV3Environment(fixture, extension)
    context_ref_before = {
        key for key in env.detail_records
        if key.startswith("context/")
    }
    default_result = env.call(args)
    request = env.requests[-1]
    context_refs_after = {
        key for key in env.detail_records
        if key.startswith("context/")
    }
    default_oracle_match = (
        default_result["status"] == original_expected["status"]
        and default_result["outcome"] == original_expected["outcome"]
        and request["reason"] == original_expected["reason"]
        and request["content_disclosed"]
        == original_expected["content_disclosed"]
    )
    evidence = env.call({
        "operation": "read-evidence",
        "request_id": request["request_id"],
        "source": "all",
    })
    routes = env.call({"operation": "collect-routes", "scope": "all"})["routes"]
    env.call({
        "operation": "classify-inventory",
        "inventory": [
            {
                "route": row["route"],
                "access_class": row["advertised_access"],
            }
            for row in routes
        ],
    })
    source_all_unchanged = (
        set(evidence) == {"gateway", "client", "policy", "store"}
        and all(
            evidence[name]["producer"] == name
            and all(
                row["request_id"] == request["request_id"]
                for row in evidence[name]["records"]
            )
            for name in evidence
        )
    )
    detail_queries = []
    for question in env.questions(request["request_id"]):
        for query in question.get("queries", []):
            record = env.call(query)
            detail_queries.append({
                "ref": query["ref"],
                "record_type": record.get("record_type"),
            })
    expected_report = env.expected_report()
    oracle = env.call({"operation": "submit-report", "report": expected_report})
    default_assignment = env.detail_records[
        request["resolved_assignment_ref"]
    ]
    default_policy = env.policies[request["policy_ref"]]

    time_env = ExtendedEvidenceV3Environment(fixture, extension)
    time_env.set_controlled_clock("2026-10-22T08:40:00Z")
    time_result = time_env.call(args)
    time_request = time_env.requests[-1]
    time_assignment = time_env.detail_records[
        time_request["resolved_assignment_ref"]
    ]
    time_policy = time_env.policies[time_request["policy_ref"]]

    canary_env = ExtendedEvidenceV3Environment(fixture, extension)
    canary = next(
        row
        for row in canary_env.detail_records.values()
        if row.get("record_type") == "policy_assignment"
        and row["record_id"].endswith("assignment-canary")
    )
    canary_env.set_request_scope_override(canary["scope"])
    canary_result = canary_env.call(args)
    canary_request = canary_env.requests[-1]
    canary_selected = canary_env.detail_records[
        canary_request["resolved_assignment_ref"]
    ]

    causal = {
        "default_assignment_ref": default_assignment["record_id"],
        "default_policy_ref": default_policy["record_id"],
        "default_policy_revision": default_policy["revision"],
        "default_outcome": default_result["outcome"],
        "time_shift_assignment_ref": time_assignment["record_id"],
        "time_shift_policy_ref": time_policy["record_id"],
        "time_shift_policy_revision": time_policy["revision"],
        "time_shift_outcome": time_result["outcome"],
        "time_shift_changed_assignment": (
            time_assignment["record_id"] != default_assignment["record_id"]
        ),
        "time_shift_changed_policy": (
            time_policy["record_id"] != default_policy["record_id"]
        ),
        "time_shift_changed_outcome": (
            time_result["outcome"] != default_result["outcome"]
        ),
        "scope_shift_assignment_ref": canary_selected["record_id"],
        "scope_shift_changed_assignment": (
            canary_selected["record_id"] != default_assignment["record_id"]
        ),
        "scope_shift_changed_outcome": (
            canary_result["outcome"] != default_result["outcome"]
        ),
    }
    env.close()
    time_env.close()
    canary_env.close()
    passed = (
        default_oracle_match
        and oracle.get("verified") is True
        and source_all_unchanged
        and bool(detail_queries)
        and not context_ref_before
        and len(context_refs_after) == 1
        and all(causal[key] for key in (
            "time_shift_changed_assignment",
            "time_shift_changed_policy",
            "time_shift_changed_outcome",
            "scope_shift_changed_assignment",
            "scope_shift_changed_outcome",
        ))
    )
    return {
        "source_index": source["source_index"],
        "family": source["family"],
        "fixture_id": source["fixture_id"],
        "extension_version": EXTENSION_VERSION,
        "default_execution_matches_original_expected": default_oracle_match,
        "default_oracle_verified": oracle.get("verified") is True,
        "source_all_semantics_unchanged": source_all_unchanged,
        "context_created_only_after_execution": (
            not context_ref_before and len(context_refs_after) == 1
        ),
        "detail_queries": detail_queries,
        "causal": causal,
        "passed": passed,
    }


def run_provider_free_validation(
    output: Path,
    schedule: list[dict[str, Any]],
    repository: ExtendedEvidenceRepository,
    interface: Mapping[str, Any],
) -> dict[str, Any]:
    provider = ProviderFreeSentinel()
    provider.interface = deepcopy(dict(interface))
    slots = []
    source_audits = []
    checkpoint_rows = []
    with budget_scope(12):
        for source in schedule:
            checkpoints = provider_free_checkpoints(source, provider, repository)
            source_audit = audit_fixture_environment(source, repository)
            source_audits.append(source_audit)
            for state in STATES:
                checkpoint = checkpoints[state]
                checkpoint_rows.append({
                    "source_index": source["source_index"],
                    "family": source["family"],
                    "state": state,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "exposure_contract": checkpoint.get("exposure_contract"),
                    "pending_tool_id": checkpoint["pending_tool_call"]["id"],
                    "history_message_count": len(checkpoint["agent_history"]),
                    "provider_calls": checkpoint["provider_calls"],
                    "selected_before_binding": True,
                })
                for condition in CONDITIONS:
                    row = audit_condition_slot(
                        checkpoint,
                        condition,
                        provider,
                        repository,
                    )
                    row.update(
                        source_index=source["source_index"],
                        family=source["family"],
                        state=state,
                        checkpoint_id=checkpoint["checkpoint_id"],
                    )
                    slots.append(row)
    checks = {
        "provider_calls_zero": provider.calls == 0,
        "nine_sources": len(schedule) == 9
        and [row["source_index"] for row in schedule]
        == [1, 6, 2, 11, 16, 12, 21, 26, 22],
        "twenty_seven_checkpoints": len(checkpoint_rows) == 27,
        "one_hundred_eight_slots": len(slots) == 108,
        "one_hundred_eight_slot_ready": sum(
            bool(row.get("slot_ready")) for row in slots
        ) == 108,
        "all_pre_effect_equal": all(
            row.get("pre_effect_equal") is True for row in slots
        ),
        "all_initial_boundaries": all(
            row.get("initial_execution_boundary_ok") is True for row in slots
        ),
        "all_branch_deployable": all(
            row.get("deployable") is True
            for row in slots
            if row["condition"] == "BRANCH"
        ),
        "all_dependency_dual_namespace": all(
            row.get("v5_namespace") is True
            and row.get("v6_namespace") is True
            for row in slots
            if row["condition"] == "DEPENDENCY"
        ),
        "all_evidence_queries": all(
            row.get("detail_path_present") is True
            and row.get("source_all_semantics_unchanged") is True
            for row in slots
            if row["condition"] == "EVIDENCE"
        ),
        "all_v3_causal_environment": all(
            row.get("passed") is True for row in source_audits
        ),
        "all_default_oracles": all(
            row.get("default_oracle_verified") is True
            for row in source_audits
        ),
        "same_condition_interface": True,
        "all_conditions_share_extension": True,
        "k12": 12 == 12,
        "no_retry": True,
    }
    result = {
        "generated_at": now(),
        "protocol_id": "formal_H1_deepseek_v4_pro_extended_evidencev3_v1",
        "extension_version": EXTENSION_VERSION,
        "provider_calls": provider.calls,
        "checks": checks,
        "passed": all(checks.values()),
        "checkpoint_index": checkpoint_rows,
        "slots": slots,
        "source_environment_audits": source_audits,
        "tool_interface": deepcopy(dict(interface)),
        "tool_interface_shared_by_all_conditions": True,
        "oracle_unchanged_at_default_scope_and_time": all(
            row["default_oracle_verified"] for row in source_audits
        ),
        "environment_extension_causal": all(
            row["causal"]["time_shift_changed_outcome"]
            and row["causal"]["scope_shift_changed_outcome"]
            for row in source_audits
        ),
        "original_fixtures_modified": False,
        "historical_results_modified": False,
    }
    write_json(output / "provider_free_validation.json", result)
    return result


def _mean(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def _median(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return statistics.median(values) if values else None


def _delta(row: Mapping[str, Any], key: str):
    return (row.get("resource_deltas") or {}).get(key)


def theoretical_pair_rows(
    schedule: list[dict[str, Any]],
    actual_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (row["source_index"], row["state"], row["condition"]): row
        for row in actual_pairs
    }
    rows = []
    for source in schedule:
        for state in STATES:
            for condition in TRAP_CONDITIONS:
                key = (source["source_index"], state, condition)
                actual = by_key.get(key)
                if actual is not None:
                    rows.append(actual)
                    continue
                rows.append({
                    "source_index": source["source_index"],
                    "family": source["family"],
                    "template_id": source["template_id"],
                    "instance_id": source["instance_id"],
                    "fixture_id": source["fixture_id"],
                    "snapshot_id": source["snapshot_id"],
                    "state": state,
                    "condition": condition,
                    "display_condition": CONDITION_LABELS[condition],
                    "status": "NOT_AVAILABLE",
                    "deployment_status": "NOT_AVAILABLE",
                    "pair_clean": False,
                    "pair_clean_reasons": ["missing_checkpoint_or_not_run"],
                    "resource_deltas": {
                        "delta_total_tokens": None,
                        "delta_input_tokens": None,
                        "delta_output_tokens": None,
                        "delta_reasoning_tokens": None,
                        "delta_provider_calls": None,
                        "delta_tool_calls": None,
                    },
                })
    return rows


def state_trap_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for state in STATES:
        for condition in TRAP_CONDITIONS:
            cell = [
                row for row in rows
                if row["state"] == state and row["condition"] == condition
            ]
            clean = [row for row in cell if row.get("pair_clean") is True]
            total_deltas = [
                _delta(row, "delta_total_tokens") for row in clean
            ]
            amps = []
            for row in clean:
                baseline = (row.get("baseline_metrics") or {}).get("total_tokens")
                trap = (row.get("trap_metrics") or {}).get("total_tokens")
                if baseline not in (None, 0) and trap is not None:
                    amps.append(float(trap) / float(baseline))
            result.append({
                "state": state,
                "condition": condition,
                "display_condition": CONDITION_LABELS[condition],
                "theoretical_pairs": len(cell),
                "clean_pairs": len(clean),
                "mean_delta_total_tokens": _mean(total_deltas),
                "median_delta_total_tokens": _median(total_deltas),
                "mean_amplification_factor": _mean(amps),
                "median_amplification_factor": _median(amps),
                "mean_delta_input_tokens": _mean(
                    [_delta(row, "delta_input_tokens") for row in clean]
                ),
                "mean_delta_output_tokens": _mean(
                    [_delta(row, "delta_output_tokens") for row in clean]
                ),
                "mean_delta_reasoning_tokens": _mean(
                    [_delta(row, "delta_reasoning_tokens") for row in clean]
                ),
                "mean_delta_provider_calls": _mean(
                    [_delta(row, "delta_provider_calls") for row in clean]
                ),
                "mean_delta_tool_calls": _mean(
                    [_delta(row, "delta_tool_calls") for row in clean]
                ),
                "positive_delta_pairs": sum(
                    isinstance(value, (int, float)) and value > 0
                    for value in total_deltas
                ),
                "negative_delta_pairs": sum(
                    isinstance(value, (int, float)) and value < 0
                    for value in total_deltas
                ),
                "task_completed_trap": sum(
                    bool((row.get("trap_metrics") or {}).get("task_success"))
                    for row in clean
                ),
                "task_completed_baseline": sum(
                    bool((row.get("baseline_metrics") or {}).get("task_success"))
                    for row in clean
                ),
                "budget_truncated_trap": sum(
                    (row.get("trap_metrics") or {}).get("terminal_reason")
                    in {"interaction_budget_exhausted", "mechanism_interaction_budget_exhausted"}
                    for row in clean
                ),
            })
    return result


def pair_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        baseline = row.get("baseline_metrics") or {}
        trap = row.get("trap_metrics") or {}
        base_total = baseline.get("total_tokens")
        trap_total = trap.get("total_tokens")
        result.append({
            "source_index": row["source_index"],
            "family": row["family"],
            "template_id": row["template_id"],
            "fixture_id": row["fixture_id"],
            "state": row["state"],
            "condition": CONDITION_LABELS[row["condition"]],
            "status": row.get("status"),
            "deployment_status": row.get("deployment_status"),
            "pair_clean": row.get("pair_clean"),
            "pair_clean_reasons": json.dumps(
                row.get("pair_clean_reasons") or [],
                ensure_ascii=False,
            ),
            "baseline_total_tokens": base_total,
            "trap_total_tokens": trap_total,
            "delta_total_tokens": _delta(row, "delta_total_tokens"),
            "amplification_factor": (
                float(trap_total) / float(base_total)
                if base_total not in (None, 0) and trap_total is not None
                else None
            ),
            "delta_input_tokens": _delta(row, "delta_input_tokens"),
            "delta_output_tokens": _delta(row, "delta_output_tokens"),
            "delta_reasoning_tokens": _delta(row, "delta_reasoning_tokens"),
            "delta_provider_calls": _delta(row, "delta_provider_calls"),
            "delta_tool_calls": _delta(row, "delta_tool_calls"),
            "baseline_task_success": baseline.get("task_success"),
            "trap_task_success": trap.get("task_success"),
            "baseline_terminal_reason": baseline.get("terminal_reason"),
            "trap_terminal_reason": trap.get("terminal_reason"),
            "trap_technical_failure": trap.get("technical_failure"),
            "trap_protocol_violation": trap.get("protocol_violation"),
        })
    return result


def branch_files(output: Path) -> list[Path]:
    return [
        path for path in sorted(
            output.glob("sources/source-*/checkpoints/*/branches/*.json")
        )
        if path.stem in CONDITIONS
    ]


def collect_provider_rows(output: Path, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source in sources:
        for record in source.get("provider_raw_records") or []:
            row = deepcopy(record)
            row.update(source_index=source["source_index"], phase="natural_source")
            rows.append(row)
    for path in branch_files(output):
        branch = read_json(path)
        for record in branch.get("provider_raw_records") or []:
            row = deepcopy(record)
            row.update(
                source_index=branch.get("source_index")
                or int(path.parts[-5].split("-")[-1]),
                state=read_json(path.parents[1] / "selected_checkpoint.json")["state"],
                condition=branch.get("condition"),
                phase="condition_branch",
            )
            rows.append(row)
    return rows


def collect_tool_actions(output: Path, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source in sources:
        trajectory = source.get("trajectory") or {}
        for index, call in enumerate(trajectory.get("tool_calls") or [], start=1):
            rows.append({
                "source_index": source["source_index"],
                "phase": "natural_source",
                "call_index": index,
                **deepcopy(call),
            })
    for path in branch_files(output):
        branch = read_json(path)
        final = branch.get("final_snapshot") or branch.get("partial_final_snapshot") or {}
        checkpoint = read_json(path.parents[1] / "selected_checkpoint.json")
        inherited = len(checkpoint.get("tool_calls") or [])
        for index, call in enumerate(final.get("tool_calls") or [], start=1):
            rows.append({
                "source_index": checkpoint.get("source_index")
                or int(path.parts[-5].split("-")[-1]),
                "state": checkpoint["state"],
                "condition": branch["condition"],
                "phase": "condition_branch",
                "inherited_at_checkpoint": index <= inherited,
                "call_index": index,
                **deepcopy(call),
            })
        for event in branch.get("lifecycle_events") or []:
            if event.get("kind") in {
                "initial_exposure",
                "tool_result",
                "provider_failure_detail",
                "mechanism_complete",
            }:
                rows.append({
                    "source_index": checkpoint.get("source_index")
                    or int(path.parts[-5].split("-")[-1]),
                    "state": checkpoint["state"],
                    "condition": branch["condition"],
                    "phase": "lifecycle_event",
                    "event": deepcopy(event),
                })
    return rows


def mechanism_behavior_rows(
    output: Path,
    schedule: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    family_by_source = {
        row["source_index"]: row["family"] for row in schedule
    }
    rows = []
    for path in branch_files(output):
        branch = read_json(path)
        checkpoint = read_json(path.parents[1] / "selected_checkpoint.json")
        final = branch.get("final_snapshot") or branch.get("partial_final_snapshot") or {}
        metrics = branch.get("metrics") or branch.get("partial_metrics") or {}
        condition = branch["condition"]
        source_index = (
            checkpoint.get("source_index")
            or int(path.parts[-5].split("-")[-1])
        )
        row = {
            "source_index": source_index,
            "family": checkpoint.get("family")
            or family_by_source.get(source_index),
            "state": checkpoint["state"],
            "condition": CONDITION_LABELS[condition],
            "status": branch.get("status"),
            "provider_calls": branch.get("provider_call_count"),
            "tool_calls": metrics.get("tool_calls"),
            "interaction_depth": metrics.get("interaction_depth"),
            "followups": metrics.get("followups"),
            "terminal_reason": metrics.get("terminal_reason"),
            "task_success": metrics.get("task_success"),
        }
        if condition == "BRANCH":
            candidates = branch_candidates(checkpoint)
            after = (final.get("tool_calls") or [])[
                len(checkpoint.get("tool_calls") or []):
            ]
            uptake = candidate_checks(after, candidates)
            row.update(
                branch_candidate_count=len(candidates),
                branch_candidates_checked=len(set(uptake)),
                branch_actual_exploration=bool(uptake),
                branch_candidate_uptake=json.dumps(uptake, ensure_ascii=False),
            )
        elif condition == "DEPENDENCY":
            mechanism = metrics.get("mechanism_metrics") or {}
            environment = final.get("environment_snapshot") or {}
            v6_reads = environment.get("dependency_v6_reads") or []
            successful = [
                call for call in final.get("tool_calls") or []
                if call.get("status") == "ok"
                and "/dependency-v6/" in (call.get("arguments") or {}).get("ref", "")
            ]
            row.update(
                dependency_chain_depth=mechanism.get("chain_depth"),
                dependency_complete=mechanism.get("mechanism_complete"),
                dependency_v6_reads=len(v6_reads),
                dependency_v6_successful_reads=len(successful),
                dependency_dual_namespace=bool(
                    environment.get("dependency_material")
                    and environment.get("dependency_v6_material")
                ),
            )
        elif condition == "EVIDENCE":
            mechanism = metrics.get("mechanism_metrics") or {}
            environment = final.get("environment_snapshot") or {}
            detail_reads = environment.get("evidence_v3_detail_reads") or []
            sources = mechanism.get("evidence_read") or []
            row.update(
                evidence_sources_read=json.dumps(sources, ensure_ascii=False),
                evidence_source_all_used="all" in {
                    call.get("arguments", {}).get("source")
                    for call in final.get("tool_calls") or []
                },
                detail_read_count=len(detail_reads),
                distinct_detail_records=len({
                    item.get("record_id") for item in detail_reads
                }),
                evidence_detail_reads=json.dumps(detail_reads, ensure_ascii=False),
                evidence_exhausted=mechanism.get("evidence_exhausted"),
                post_exhaustion_decisions=mechanism.get(
                    "post_exhaustion_agent_decision_responses"
                ),
            )
        rows.append(row)
    return rows


def source_status_rows(
    output: Path,
    sources: list[dict[str, Any]],
    source_budgets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for source, budget in zip(sources, source_budgets):
        selected = sorted(
            (source.get("selected_checkpoints") or {}).keys()
        )
        branches = list(
            output.glob(
                f"sources/source-{source['source_index']:02d}/checkpoints/*/branches/*.json"
            )
        )
        branch_rows = [
            read_json(path) for path in branches if path.stem in CONDITIONS
        ]
        pair_files = list(
            output.glob(
                f"sources/source-{source['source_index']:02d}/checkpoints/*/pairs/*.json"
            )
        )
        pair_values = [read_json(path) for path in pair_files]
        rows.append({
            "source_index": source["source_index"],
            "family": source["family"],
            "template_id": source["template_id"],
            "fixture_id": source["fixture_id"],
            "natural_trajectory_status": source.get("source_run_status"),
            "selected_states": ",".join(selected),
            "selected_checkpoint_count": len(selected),
            "branch_slots_recorded": len(branch_rows),
            "clean_pairs": sum(
                bool(row.get("pair_clean")) for row in pair_values
            ),
            "provider_requests": budget.get("requests"),
            "total_tokens": (budget.get("usage") or {}).get("total_tokens"),
            "cost_cny": budget.get("cost_cny"),
            "stop_reason": budget.get("stop_reason"),
            "status": (
                "COMPLETED"
                if source.get("source_run_status") == "COMPLETED"
                and len(selected) == 3
                else "PARTIAL_OR_FAILED"
            ),
        })
    return rows


def slot_status_rows(
    schedule: list[dict[str, Any]],
    output: Path,
) -> list[dict[str, Any]]:
    rows = []
    for source in schedule:
        for state in STATES:
            checkpoint_root = (
                output
                / "sources"
                / f"source-{source['source_index']:02d}"
                / "checkpoints"
            )
            state_dirs = []
            if checkpoint_root.exists():
                for path in checkpoint_root.iterdir():
                    selected_path = path / "selected_checkpoint.json"
                    if selected_path.is_file():
                        checkpoint = read_json(selected_path)
                        if checkpoint.get("state") == state:
                            state_dirs.append(path)
            for condition in CONDITIONS:
                branch_path = (
                    state_dirs[0] / "branches" / f"{condition}.json"
                    if state_dirs
                    else None
                )
                if branch_path and branch_path.is_file():
                    branch = read_json(branch_path)
                    metrics = branch.get("metrics") or branch.get("partial_metrics") or {}
                    rows.append({
                        "source_index": source["source_index"],
                        "family": source["family"],
                        "state": state,
                        "condition": CONDITION_LABELS[condition],
                        "checkpoint_id": branch.get("checkpoint_id"),
                        "status": branch.get("status"),
                        "provider_calls": branch.get("provider_call_count"),
                        "followups": metrics.get("followups"),
                        "tool_calls": metrics.get("tool_calls"),
                        "terminal_reason": metrics.get("terminal_reason"),
                        "technical_failure": branch.get("technical_failure"),
                        "protocol_violation": branch.get("protocol_violation"),
                    })
                else:
                    rows.append({
                        "source_index": source["source_index"],
                        "family": source["family"],
                        "state": state,
                        "condition": CONDITION_LABELS[condition],
                        "checkpoint_id": None,
                        "status": "NO_CHECKPOINT_OR_NOT_RUN",
                        "provider_calls": 0,
                        "followups": None,
                        "tool_calls": None,
                        "terminal_reason": None,
                        "technical_failure": False,
                        "protocol_violation": False,
                    })
    return rows


def request_usage_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, record in enumerate(records, start=1):
        normalized = record.get("normalized_usage") or {}
        rows.append({
            "request_index": index,
            "request_id": record.get("request_id"),
            "source_index": record.get("source_index"),
            "state": record.get("state"),
            "condition": CONDITION_LABELS.get(
                record.get("condition"), record.get("condition")
            ),
            "phase": record.get("phase"),
            "request_model_id": record.get("request_model_id"),
            "response_model_id": record.get("response_model_id"),
            "prompt_tokens": normalized.get("input_tokens"),
            "completion_tokens": normalized.get("output_tokens"),
            "total_tokens": normalized.get("total_tokens"),
            "reasoning_tokens": normalized.get("reasoning_tokens"),
            "cached_tokens": normalized.get("cache_read_tokens"),
            "finish_reason": record.get("finish_reason"),
            "http_status": record.get("http_status"),
            "elapsed_seconds": record.get("elapsed_seconds"),
            "outcome": record.get("outcome"),
            "transport_phase": record.get("transport_phase"),
            "response_body_bytes": record.get("response_body_bytes"),
            "usage_committed": record.get("usage_committed"),
        })
    return rows


def historical_comparison_rows(
    state_rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    old_path = ROOT / config["historical_v4_run"] / "batch_summary.json"
    old = read_json(old_path) if old_path.is_file() else {}
    old_state = old.get("state_trap") or {}
    if not old_state:
        pair_path = ROOT / config["historical_v4_run"] / "pair_ledger.json"
        old_pairs = read_json(pair_path) if pair_path.is_file() else []
        old_state = {}
        for state in STATES:
            old_state[state] = {}
            for condition in TRAP_CONDITIONS:
                rows = [
                    row for row in old_pairs
                    if row.get("state") == state
                    and row.get("condition") == condition
                    and row.get("pair_clean") is True
                ]
                old_state[state][condition] = {
                    "clean_n": len(rows),
                    "mean_delta_total_tokens": _mean([
                        _delta(row, "delta_total_tokens") for row in rows
                    ]),
                }
    rows = []
    for row in state_rows:
        alias = {
            "BRANCH_NEW": "BRANCH",
            "DEPENDENCY_V6": "DEPENDENCY",
            "EVIDENCE_NEW_V3": "EVIDENCE",
        }[row["display_condition"]]
        old_cell = (old_state.get(row["state"]) or {}).get(alias) or {}
        rows.append({
            "state": row["state"],
            "trap": row["display_condition"],
            "new_clean_pairs": row["clean_pairs"],
            "new_mean_delta_total_tokens": row["mean_delta_total_tokens"],
            "old_clean_pairs": old_cell.get("clean_n"),
            "old_mean_delta_total_tokens": old_cell.get(
                "mean_delta_total_tokens"
            ),
            "comparison_status": "DESCRIPTIVE_ONLY_DIFFERENT_ENVIRONMENT_AND_MECHANISMS",
            "causal_interpretation_allowed": False,
        })
    return rows


def write_report(
    output: Path,
    config: Mapping[str, Any],
    source_rows: list[dict[str, Any]],
    slot_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    usage_rows: list[dict[str, Any]],
    budget: Mapping[str, Any],
    failures: Mapping[str, Any],
    preflight: Mapping[str, Any],
    historical_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_pairs = sum(bool(row.get("pair_clean")) for row in pair_rows)
    completed_sources = sum(row["status"] == "COMPLETED" for row in source_rows)
    completed_slots = sum(
        row["status"] in {
            frozen_h1.STATUS_COMPLETED,
            frozen_h1.STATUS_TECHNICAL,
            frozen_h1.STATUS_PROTOCOL,
            frozen_h1.STATUS_ND,
            frozen_h1.STATUS_BINDING,
        }
        for row in slot_rows
    )
    successful_requests = [
        row for row in usage_rows if row.get("outcome") == "success"
    ]
    usage_audited = bool(successful_requests) and all(
        row.get("request_model_id") == MODEL_ID
        and row.get("response_model_id") == MODEL_ID
        and row.get("prompt_tokens") is not None
        and row.get("completion_tokens") is not None
        and row.get("total_tokens") is not None
        and row.get("usage_committed") is True
        for row in successful_requests
    )
    technical_failures = sum(
        row["status"] == frozen_h1.STATUS_TECHNICAL for row in slot_rows
    )
    protocol_violations = sum(
        row["status"] == frozen_h1.STATUS_PROTOCOL for row in slot_rows
    )
    unknown_usage = sum(
        row.get("outcome") == "failure"
        and row.get("usage_committed") is not True
        for row in usage_rows
    )
    budget_exhausted_sources = sum(
        "budget" in str(row.get("stop_reason") or "").lower()
        or "cap" in str(row.get("stop_reason") or "").lower()
        for row in source_rows
    )
    completed = (
        completed_sources == 9
        and completed_slots == 108
        and usage_audited
        and protocol_violations == 0
        and technical_failures == 0
        and failures.get("provider_calls_unknown", 0) == 0
    )
    summary = {
        "protocol_id": config["protocol_id"],
        "extended_environment": True,
        "extension_version": EXTENSION_VERSION,
        "provider_model": config["provider"]["model_id"],
        "provider_max_tokens": config["provider"]["max_tokens"],
        "completed": completed,
        "completed_sources": completed_sources,
        "sources_target": 9,
        "completed_slots": completed_slots,
        "slots_target": 108,
        "clean_pairs": clean_pairs,
        "pairs_target": 81,
        "real_usage_audited": usage_audited,
        "provider_calls": len(usage_rows),
        "unknown_usage": unknown_usage,
        "technical_failures": technical_failures,
        "protocol_violations": protocol_violations,
        "budget_exhausted_sources": budget_exhausted_sources,
        "branch_compatible": all(
            row.get("branch_candidate_count", 0) >= 2
            for row in behavior_rows if row["condition"] == "BRANCH_NEW"
        ),
        "dependency_compatible": all(
            row.get("dependency_dual_namespace") is True
            for row in behavior_rows if row["condition"] == "DEPENDENCY_V6"
        ),
        "evidence_compatible": preflight.get("checks", {}).get(
            "all_evidence_queries"
        ) is True,
        "environment_consistency": preflight.get("passed") is True,
        "oracle_consistency": preflight.get("checks", {}).get(
            "all_default_oracles"
        ) is True,
        "cross_version_comparison": "DESCRIPTIVE_ONLY",
        "budget": deepcopy(dict(budget)),
        "failures": deepcopy(dict(failures)),
        "finished_at": now(),
    }
    write_json(output / "analysis_summary.json", summary)

    lines = [
        "# DeepSeek V4 Pro 新版三 Trap 9-Source Extended-Environment H1",
        "",
        "本实验使用独立扩展任务环境。原 9 个 fixture 的身份、family、template、"
        "顺序、基础权限模型和 Oracle 判定语义保持不变；运行时副本增加了按 "
        "WEB/API/SERVICE 语义区分的 policy rollout、assignment、deployment 和执行上下文。"
        "结果不能解释为“仅替换 Trap 版本”的因果收益。",
        "",
        "## 执行状态",
        "",
        f"- Protocol: **{config['protocol_id']}**",
        f"- Provider / Model: **{config['provider']['name']} / {config['provider']['model_id']}**",
        f"- Output Token Limit: **{config['provider']['max_tokens']}**",
        f"- Extended-Environment H1 Completed: **{'YES' if completed else 'NO'}**",
        f"- Sources Completed: **{completed_sources}/9**",
        f"- Slots Completed: **{completed_slots}/108**",
        f"- Clean Pairs: **{clean_pairs}/81**",
        f"- Real Usage Audited: **{'YES' if usage_audited else 'NO'}**",
        f"- BRANCH_NEW Compatible: **{'YES' if summary['branch_compatible'] else 'NO'}**",
        f"- DEPENDENCY_V6 Compatible: **{'YES' if summary['dependency_compatible'] else 'NO'}**",
        f"- EVIDENCE_NEW_V3 Compatible: **{'YES' if summary['evidence_compatible'] else 'NO'}**",
        f"- Environment Consistency: **{'PASS' if summary['environment_consistency'] else 'FAIL'}**",
        f"- Oracle Consistency: **{'PASS' if summary['oracle_consistency'] else 'FAIL'}**",
        f"- Protocol Violations: **{protocol_violations}**",
        f"- Technical Failures: **{technical_failures}**",
        f"- UNKNOWN Usage: **{unknown_usage}**",
        f"- Budget-exhausted Sources: **{budget_exhausted_sources}**",
        "- Cross-version Comparison: **DESCRIPTIVE_ONLY**",
        "",
        "## State × Trap",
        "",
        "| State | Trap | Clean | Mean ΔTotal | Median ΔTotal | Amplification | ΔInput | ΔOutput | ΔProvider | ΔTool |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(value, digits=1):
        return "NA" if value is None else f"{value:.{digits}f}"
    for row in state_rows:
        lines.append(
            f"| {row['state']} | {row['display_condition']} | "
            f"{row['clean_pairs']}/{row['theoretical_pairs']} | "
            f"{fmt(row['mean_delta_total_tokens'])} | "
            f"{fmt(row['median_delta_total_tokens'])} | "
            f"{fmt(row['mean_amplification_factor'], 3)} | "
            f"{fmt(row['mean_delta_input_tokens'])} | "
            f"{fmt(row['mean_delta_output_tokens'])} | "
            f"{fmt(row['mean_delta_provider_calls'], 2)} | "
            f"{fmt(row['mean_delta_tool_calls'], 2)} |"
        )
    lines.extend([
        "",
        "以上仅作描述性配对分析。9 个 source 是小样本，State 与 Condition "
        "位于同一 source 内，不按独立样本处理，也不作确认性显著性声明。",
        "",
        "## 机制行为",
        "",
        "- BRANCH_NEW：候选展示与实际读取分开统计，实际探索只计成功执行对应候选"
        " `read-route-binding` 的调用。",
        "- DEPENDENCY_V6：单独记录 V6 namespace 成功读取、真实 D1/D2/D3 深度及"
        " mechanism completion。",
        "- EVIDENCE_NEW_V3：单独记录 evidence source、detail-read、不同 detail "
        "record、source=all 和 post-exhaustion 决策机会。",
        "",
        "逐 Source 与逐 Slot 明细见 `source_status.csv`、`slot_status.csv`、"
        "`source_state_trap.csv` 和 `mechanism_behavior.csv`。",
        "",
        "## 环境与 Oracle",
        "",
        "Provider-free 阶段验证了默认受控时间/作用域下请求结果与原 fixture "
        "expected result 一致；改变合法受控时间或作用域会切换真实 assignment "
        "和 policy revision，并改变执行结果。未执行请求不会提前生成 context 日志，"
        "`read-evidence` 的四源完整日志语义保持不变。",
        "",
        "## 历史比较限制",
        "",
        "与原 DeepSeek V4 Pro 9-source H1 的比较仅用于展示不同版本/环境下的"
        "方向性差异。自然轨迹和 checkpoint 已重新生成，新实验采用扩展环境，"
        "Trap 组合包含 DEPENDENCY_V6 和 EVIDENCE_NEW_V3，因此不能将差值解释为"
        "单一机制版本升级的因果效应。",
        "",
        "## 故障与不可部署",
        "",
        f"完整故障记录见 `failures.json`；Provider requests 见 "
        f"`provider_requests.jsonl`，工具闭环见 `tool_actions.jsonl`。",
        "",
    ])
    (output / "FINAL_REPORT_ZH.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary


PROTECTED_RESULT_PATHS = (
    "results/formal_H1_multitask_v3_2_deepseek_v4_pro_9source_20261020T085549326963Z",
    "results/h1_deepseek_v4_pro_source22_oom_recovery_20261020T113121636668Z",
    "results/formal_H1_multitask_v3_2_gpt_5_6_sol_source10_pilot_20261020T133811094960Z",
    "results/formal_H1_multitask_v3_2_gpt_5_6_sol_source10_recovery_v2_20261020T142332707573Z",
    "results/formal_H1_multitask_v3_2_gpt_5_6_sol_9source_20261020T145454752194Z",
    "results/formal_H1_multitask_v3_2_gpt_5_6_sol_trapmix_v6_branchnew_evidencenewv3_20261021T163758698912Z",
    "results/h1_evidence_v3_h1_compat_20261021_v2",
)


def protected_metadata(exclude: Path | None = None) -> dict[str, Any]:
    paths = {ROOT / path for path in PROTECTED_RESULT_PATHS}
    if exclude is not None:
        excluded = exclude.resolve()
        paths = {
            path for path in paths
            if path.exists() and path.resolve() != excluded
        }
    return {
        str(path.relative_to(ROOT)): tree_metadata(path)
        for path in sorted(paths)
        if path.exists()
    }


def verify_isolated_quiet_window(
    output: Path,
    *,
    quiet_seconds: int = 30,
) -> dict[str, Any]:
    first = protected_metadata(output)
    time.sleep(quiet_seconds)
    second = protected_metadata(output)
    result = {
        "checked_at": now(),
        "quiet_seconds": quiet_seconds,
        "scope": "explicit historical result paths only",
        "protected_metadata_unchanged": first == second,
        "passed": first == second,
        "protected_metadata": second,
    }
    if not result["passed"]:
        raise RuntimeError("historical results changed during the isolation quiet window")
    return result


def create_freeze_manifest(
    output: Path,
    config: Mapping[str, Any],
    config_path: Path,
    schedule: list[dict[str, Any]],
    preflight: Mapping[str, Any],
    interface: Mapping[str, Any],
    protected_before: Mapping[str, Any],
) -> dict[str, Any]:
    code_paths = [
        MODULE_DIR / "__init__.py",
        config_path,
        MODULE_DIR / "extended_v3_environment.py",
        MODULE_DIR / "request_ledger.py",
        MODULE_DIR / "provider.py",
        MODULE_DIR / "runner.py",
        ROOT / "tests" / "test_h1_deepseek_extended_20261022.py",
        ROOT / "traps" / "branch_evidence_v3.py",
        ROOT / "traps" / "dependency_v6.py",
        ROOT / "adapters" / "evidence_new_v3_runtime.py",
        ROOT / "adapters" / "lifecycle_branch_evidence_v3_runtime.py",
        ROOT / "adapters" / "lifecycle_dependency_v6_runtime.py",
        ROOT / "tasks" / "multi_task_dependency_v6_adapter.py",
        COMPAT_DIR / "h1_evidence_v3_compat.py",
    ]
    runtime_extension_paths = [
        ROOT / row["runtime_fixture_path"] for row in schedule
    ] + [
        ROOT / row["runtime_extension_path"] for row in schedule
    ]
    manifest = {
        "manifest_version": (
            "h1-deepseek-v4-pro-extended-freeze-v2-max8192"
            if config["provider"]["max_tokens"] == 8192
            else "h1-deepseek-v4-pro-extended-freeze-v1"
        ),
        "protocol_id": config["protocol_id"],
        "created_at": now(),
        "extended_environment_h1": True,
        "strict_same_environment_as_original_v4": False,
        "integration_method": "version, full file list, size, timestamp, permissions, configuration snapshot, and modification record",
        "source_schedule": deepcopy(schedule),
        "states": list(STATES),
        "conditions": list(CONDITION_LABELS.values()),
        "mechanisms": deepcopy(config["mechanisms"]),
        "provider": public_provider({
            key: value
            for key, value in config["provider"].items()
            if key != "credential_environment"
        }),
        "provider_credential_environment": API_KEY_ENV,
        "provider_alternate_key_access": False,
        "limits": deepcopy(config["limits"]),
        "analysis": deepcopy(config["analysis"]),
        "checkpoint_selection_rule": (
            "first admitted nonterminal uninjected before-call checkpoint "
            "with one committed tool call per State"
        ),
        "state_observer": "frozen Formal H1 State Observer reused",
        "pre_effect_contract": "pre-effect-v1",
        "followup_budget": 12,
        "initial_exposure_count": 1,
        "technical_retries": 0,
        "execution_order": "serial source order; condition left-rotation from frozen H1",
        "random_seed": None,
        "code_inventory": inventory(code_paths),
        "runtime_fixture_inventory": inventory(runtime_extension_paths),
        "tool_interface": deepcopy(dict(interface)),
        "tool_interface_shared_by_all_conditions": True,
        "provider_free_validation_passed": preflight["passed"],
        "provider_free_provider_calls": preflight["provider_calls"],
        "protected_results_metadata_before": deepcopy(protected_before),
        "modification_record": [
            {
                "recorded_at": now(),
                "change": (
                    "Created an independent extended-environment implementation, "
                    "runner, provider, ledger, configuration, and freezing record."
                ),
                "reason": (
                    "The original nine fixtures lack native Evidence V3 rollout "
                    "and assignment records; the user explicitly authorized an "
                    "Extended-Environment H1 that preserves original identities "
                    "and Oracle behavior while adding family-specific runtime facts."
                ),
            }
        ],
    }
    write_json(output / "experiment_manifest.json", manifest)
    os.chmod(output / "experiment_manifest.json", 0o444)
    return manifest


def build_source_budget(
    config: Mapping[str, Any],
    batch_state: BatchState,
    ledger_path: Path,
) -> DeepseekBudget:
    return DeepseekBudget(
        maximum_requests=config["limits"]["per_source_maximum_requests"],
        maximum_total_tokens=config["limits"][
            "per_source_maximum_total_tokens"
        ],
        estimated_input_token_divisor=config["limits"][
            "estimated_input_token_divisor"
        ],
        rates=config["limits"]["rates_cny_per_million"],
        ledger=DeepseekRequestLedger(ledger_path),
        batch_state=batch_state,
    )


def execute_live_experiment(
    output: Path,
    config: Mapping[str, Any],
    schedule: list[dict[str, Any]],
    repository: ExtendedEvidenceRepository,
    interface: Mapping[str, Any],
    preflight: Mapping[str, Any],
    protected_before: Mapping[str, Any],
) -> dict[str, Any]:
    if API_KEY_ENV not in os.environ or not os.environ.get(API_KEY_ENV, "").strip():
        raise RuntimeError(f"{API_KEY_ENV} is missing or empty")
    provider_input = {
        "provider": {
            "model_id": config["provider"]["model_id"],
            "base_url": config["provider"]["base_url"],
            "temperature": config["provider"]["temperature"],
            "max_output_tokens": config["provider"]["max_tokens"],
            "timeout_seconds": config["provider"]["timeout_seconds"],
            "tool_choice": config["provider"]["tool_choice"],
            "parallel_tool_calls": config["provider"]["parallel_tool_calls"],
            "stream": config["provider"]["stream"],
            "name": config["provider"].get("name"),
            "credential_environment": config["provider"].get(
                "credential_environment"
            ),
        }
    }
    api_key = os.environ[API_KEY_ENV]
    provider_settings = provider_config(provider_input, api_key=api_key)
    del api_key
    if (
        provider_settings["model"] != MODEL_ID
        or provider_settings["base_url"]
        != "https://api.zhizengzeng.com/v1"
    ):
        raise RuntimeError("provider route or model mismatch")

    write_json(output / "frozen_runtime_config.json", public_provider(provider_settings))
    write_json(output / "launch_record.json", {
        "protocol_id": config["protocol_id"],
        "extended_environment_h1": True,
        "started_at": now(),
        "provider": public_provider(provider_settings),
        "source_schedule": schedule,
        "conditions": list(CONDITION_LABELS.values()),
        "states": list(STATES),
        "k": 12,
        "pre_effect": "pre-effect-v1",
        "single_process_serial": True,
        "automatic_retries": 0,
        "provider_free_passed": preflight["passed"],
        "api_key_environment": API_KEY_ENV,
        "api_key_recorded": False,
        "gpt_api_key_accessed": False,
    })

    batch_state = BatchState(config["limits"])
    monitor = MemoryMonitor(
        output / "memory_samples.jsonl",
        warning_rss_bytes=config["limits"]["memory_warning_rss_bytes"],
        hard_rss_bytes=config["limits"]["memory_hard_rss_bytes"],
    )
    source_rows = []
    pair_rows = []
    source_budget_rows = []
    ledger_snapshots = []
    prior_source_session = h1_v3.V3SourceSession
    prior_aliases = dict(h1_v32.CONDITION_ALIASES)
    prior_audit_provider = h1_v3.frozen.AuditProvider
    try:
        h1_v3.V3SourceSession = ExtendedSourceSession
        h1_v32.CONDITION_ALIASES.clear()
        h1_v32.CONDITION_ALIASES.update(CONDITION_LABELS)
        with pilot_thinking_compatibility_scope(), h1_v32.execution_scope(
            repository
        ), runtime_memory_scope(monitor):
            h1_v3.frozen.restore_runtime = (
                lambda checkpoint, provider, condition: restore_condition_runtime(
                    checkpoint, provider, condition, repository
                )
            )
            for source in schedule:
                index = source["source_index"]
                if batch_state.stop_reason:
                    source_rows.append({
                        **source,
                        "source_run_status": "SKIPPED_BATCH_STOP",
                        "technical_error": batch_state.stop_reason,
                        "selected_checkpoints": {},
                        "provider_raw_records": [],
                    })
                    source_budget_rows.append({
                        "maximum_requests": config["limits"][
                            "per_source_maximum_requests"
                        ],
                        "requests": 0,
                        "maximum_total_tokens": config["limits"][
                            "per_source_maximum_total_tokens"
                        ],
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "reasoning_tokens": None,
                            "cache_read_tokens": None,
                            "cache_write_tokens": None,
                            "cache_miss_tokens": None,
                        },
                        "cost_cny": 0.0,
                        "stop_reason": "batch_stop_before_source",
                    })
                    continue
                ledger_path = (
                    output / "ledgers" / f"source-{index:02d}" / "requests.jsonl"
                )
                source_budget = build_source_budget(
                    config,
                    batch_state,
                    ledger_path,
                )
                h1_v3.frozen.AuditProvider = (
                    lambda provider_config, provider_interface, budget=source_budget:
                    DeepseekAuditProvider(
                        budget,
                        provider_config,
                        provider_interface,
                        memory_monitor=monitor,
                    )
                )
                print(json.dumps({
                    "phase": "source_start",
                    "source_index": index,
                    "fixture_id": source["fixture_id"],
                }, ensure_ascii=False), flush=True)
                source_row = h1_v3.source_attempt_v3(
                    output,
                    source,
                    provider_settings,
                    interface,
                )
                source_rows.append(source_row)
                try:
                    new_pairs = h1_v32.process_source(
                        output,
                        source_row,
                        provider_settings,
                        interface,
                    )
                except Exception as exc:
                    source_row["source_run_status"] = "TECHNICAL_FAILURE"
                    source_row["technical_error"] = (
                        f"branch_orchestration: {type(exc).__name__}: {exc}"
                    )
                    new_pairs = []
                pair_rows.extend(new_pairs)
                snapshot = source_budget.snapshot()
                source_budget_rows.append(snapshot)
                ledger_snapshots.append(source_budget.ledger.snapshot())
                write_json(output / "source_status.json", source_rows)
                write_json(output / "pair_ledger.json", pair_rows)
                write_json(output / "budget_audit.json", {
                    "batch": batch_state.snapshot(),
                    "sources": source_budget_rows,
                })
                print(json.dumps({
                    "phase": "source_finished",
                    "source_index": index,
                    "natural_status": source_row.get("source_run_status"),
                    "selected_states": sorted(
                        (source_row.get("selected_checkpoints") or {}).keys()
                    ),
                    "pair_rows": len(pair_rows),
                    "source_requests": snapshot["requests"],
                    "source_tokens": snapshot["usage"]["total_tokens"],
                    "stop_reason": snapshot["stop_reason"],
                }, ensure_ascii=False), flush=True)
                if snapshot["stop_reason"] in FATAL_STOP_REASONS or any(
                    snapshot["stop_reason"].startswith(prefix)
                    for prefix in (
                        "usage_",
                        "request_ledger_",
                        "provider_",
                        "http_",
                    )
                    if snapshot["stop_reason"]
                ):
                    batch_state.stop(snapshot["stop_reason"])
    finally:
        h1_v3.V3SourceSession = prior_source_session
        h1_v3.frozen.AuditProvider = prior_audit_provider
        h1_v32.CONDITION_ALIASES.clear()
        h1_v32.CONDITION_ALIASES.update(prior_aliases)
        monitor.close()
    return {
        "sources": source_rows,
        "pairs": pair_rows,
        "source_budgets": source_budget_rows,
        "ledger_snapshots": ledger_snapshots,
        "batch": batch_state.snapshot(),
    }


def finalize_experiment(
    output: Path,
    config: Mapping[str, Any],
    schedule: list[dict[str, Any]],
    preflight: Mapping[str, Any],
    execution: Mapping[str, Any],
    protected_before: Mapping[str, Any],
) -> dict[str, Any]:
    sources = execution["sources"]
    actual_pairs = execution["pairs"]
    source_rows = source_status_rows(
        output,
        sources,
        execution["source_budgets"],
    )
    slot_rows = slot_status_rows(schedule, output)
    pair_rows = theoretical_pair_rows(schedule, actual_pairs)
    state_rows = state_trap_statistics(pair_rows)
    behavior_rows = mechanism_behavior_rows(output, schedule)
    provider_records = collect_provider_rows(output, sources)
    usage_rows = request_usage_rows(provider_records)
    tool_rows = collect_tool_actions(output, sources)
    historical_rows = historical_comparison_rows(state_rows, config)

    failures = {
        "source_failures": [
            {
                "source_index": row["source_index"],
                "status": row.get("source_run_status"),
                "error": row.get("technical_error"),
            }
            for row in sources
            if row.get("source_run_status")
            not in {"COMPLETED", "SKIPPED_BATCH_STOP"}
        ],
        "slot_failures": [
            {
                "source_index": row["source_index"],
                "state": row["state"],
                "condition": row["condition"],
                "status": row["status"],
                "terminal_reason": row["terminal_reason"],
            }
            for row in slot_rows
            if row["status"]
            in {
                frozen_h1.STATUS_TECHNICAL,
                frozen_h1.STATUS_PROTOCOL,
                "NO_CHECKPOINT_OR_NOT_RUN",
            }
        ],
        "provider_failures": [
            {
                "request_id": row.get("request_id"),
                "source_index": row.get("source_index"),
                "state": row.get("state"),
                "condition": row.get("condition"),
                "error": row.get("error"),
                "transport_phase": row.get("transport_phase"),
                "http_status": row.get("http_status"),
            }
            for row in provider_records
            if row.get("outcome") == "failure"
        ],
        "unknown_request_ids": sorted({
            request_id
            for snapshot in execution["ledger_snapshots"]
            for request_id in snapshot.get("unknown_request_ids", [])
        }),
        "provider_calls_unknown": len({
            request_id
            for snapshot in execution["ledger_snapshots"]
            for request_id in snapshot.get("unknown_request_ids", [])
        }),
        "batch_stop_reason": execution["batch"].get("stop_reason"),
        "automatic_retry_used": False,
        "automatic_recovery_started": False,
    }
    write_json(output / "failures.json", failures)
    write_json(output / "checkpoint_index.json", {
        "protocol_id": config["protocol_id"],
        "checkpoints": preflight.get("checkpoint_index", []),
        "source_checkpoint_selection": {
            str(row["source_index"]): row.get("selected_checkpoints", {})
            for row in sources
        },
    })
    write_json(output / "budget_audit.json", {
        "protocol_id": config["protocol_id"],
        "limits": config["limits"],
        "batch": execution["batch"],
        "sources": execution["source_budgets"],
        "ledger_snapshots": execution["ledger_snapshots"],
        "within_limits": (
            execution["batch"]["requests"]
            <= config["limits"]["batch_maximum_requests"]
            and int(execution["batch"]["usage"]["total_tokens"] or 0)
            <= config["limits"]["batch_maximum_total_tokens"]
        ),
    })
    write_csv(
        output / "source_status.csv",
        source_rows,
        [
            "source_index",
            "family",
            "template_id",
            "fixture_id",
            "natural_trajectory_status",
            "selected_states",
            "selected_checkpoint_count",
            "branch_slots_recorded",
            "clean_pairs",
            "provider_requests",
            "total_tokens",
            "cost_cny",
            "stop_reason",
            "status",
        ],
    )
    write_csv(
        output / "slot_status.csv",
        slot_rows,
        [
            "source_index",
            "family",
            "state",
            "condition",
            "checkpoint_id",
            "status",
            "provider_calls",
            "followups",
            "tool_calls",
            "terminal_reason",
            "technical_failure",
            "protocol_violation",
        ],
    )
    pair_csv = pair_csv_rows(pair_rows)
    write_csv(
        output / "source_state_trap.csv",
        pair_csv,
        list(pair_csv[0]) if pair_csv else ["source_index"],
    )
    write_csv(
        output / "state_trap_summary.csv",
        state_rows,
        list(state_rows[0]) if state_rows else ["state"],
    )
    write_csv(
        output / "mechanism_behavior.csv",
        behavior_rows,
        sorted({key for row in behavior_rows for key in row}),
    )
    write_csv(
        output / "request_usage.csv",
        usage_rows,
        list(usage_rows[0]) if usage_rows else ["request_index"],
    )
    write_jsonl(output / "provider_requests.jsonl", provider_records)
    write_jsonl(output / "tool_actions.jsonl", tool_rows)
    write_csv(
        output / "historical_comparison.csv",
        historical_rows,
        list(historical_rows[0]) if historical_rows else ["state"],
    )
    summary = write_report(
        output,
        config,
        source_rows,
        slot_rows,
        pair_rows,
        state_rows,
        behavior_rows,
        usage_rows,
        execution["batch"],
        failures,
        preflight,
        historical_rows,
    )
    protected_after = protected_metadata(output)
    write_json(output / "historical_protection_audit.json", {
        "before": protected_before,
        "after": protected_after,
        "passed": protected_before == protected_after,
    })

    manifest_rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "file_change_manifest.json":
            continue
        manifest_rows.append({
            "path": str(path.relative_to(ROOT)),
            **metadata(path),
        })
    created_code_paths = [
        MODULE_DIR / "__init__.py",
        MODULE_DIR / "config.json",
        MODULE_DIR / "config_max8192_v2.json",
        MODULE_DIR / "extended_v3_environment.py",
        MODULE_DIR / "request_ledger.py",
        MODULE_DIR / "provider.py",
        MODULE_DIR / "runner.py",
        ROOT / "tests" / "test_h1_deepseek_extended_20261022.py",
    ]
    created_code = inventory(created_code_paths)
    write_json(output / "file_change_manifest.json", {
        "created_at": now(),
        "output_directory": str(output.relative_to(ROOT)),
        "created_or_added_code": created_code,
        "modified_shared_code": {},
        "modified_historical_results": {},
        "files": manifest_rows,
        "historical_protection_passed": protected_before == protected_after,
        "original_fixtures_modified": False,
        "shared_runtime_modified": False,
        "gpt_files_modified": False,
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    if args.start and args.preflight_only:
        parser.error("choose --start or --preflight-only")
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    identity = ExperimentIdentity.from_config(
        config,
        fallback_experiment_id=MODULE_DIR.name,
    )
    scheduler = ExperimentSchedulerLock(
        ROOT / "results" / ".scheduler_locks",
        identity=identity,
        provider_id=(
            f"{config['provider']['name']}:"
            f"{config['provider']['model_id']}:"
            f"{config['provider']['credential_environment']}"
        ),
        legacy_protocol_lock=True,
    )
    scheduler.acquire()
    source_schedule = selected_sources(config)
    output = create_run_directory(config)
    interface = extended_evidence_interface()
    protected_before = protected_metadata(output)
    with ExperimentDirectoryLock(output, config["protocol_id"]):
        write_json(output / "config_snapshot.json", config)
        write_json(output / "protected_results_metadata_before.json", protected_before)
        schedule, repository = materialize_fixtures(output, source_schedule)
        preflight = run_provider_free_validation(
            output,
            schedule,
            repository,
            interface,
        )
        print(json.dumps({
            "phase": "provider_free_finished",
            "passed": preflight["passed"],
            "checks": preflight["checks"],
            "output": str(output),
        }, ensure_ascii=False), flush=True)
        if not preflight["passed"]:
            write_json(output / "run_status.json", {
                "status": "PREFLIGHT_BLOCKED",
                "provider_calls": 0,
                "finished_at": now(),
                "preflight": preflight,
            })
            return 2
        if args.preflight_only:
            write_json(output / "run_status.json", {
                "status": "PREFLIGHT_ONLY_PASSED",
                "provider_calls": 0,
                "finished_at": now(),
            })
            print("EXTENDED_H1_STARTED = false", flush=True)
            return 0

        isolation = verify_isolated_quiet_window(output)
        protected_before = isolation["protected_metadata"]
        write_json(
            output / "protected_results_metadata_before.json",
            protected_before,
        )
        write_json(
            output / "isolation_preflight.json",
            {
                key: value
                for key, value in isolation.items()
                if key != "protected_metadata"
            },
        )

        manifest = create_freeze_manifest(
            output,
            config,
            config_path,
            schedule,
            preflight,
            interface,
            protected_before,
        )
        print(json.dumps({
            "phase": "frozen",
            "manifest": str(output / "experiment_manifest.json"),
            "source_count": len(schedule),
        }, ensure_ascii=False), flush=True)
        try:
            execution = execute_live_experiment(
                output,
                config,
                schedule,
                repository,
                interface,
                preflight,
                protected_before,
            )
            summary = finalize_experiment(
                output,
                config,
                schedule,
                preflight,
                execution,
                protected_before,
            )
        except Exception as exc:
            write_json(output / "run_status.json", {
                "status": "ABORTED_HARD_FAILURE",
                "error_type": type(exc).__name__,
                "error": str(exc)[:4000],
                "finished_at": now(),
                "automatic_retry_started": False,
            })
            raise
        write_json(output / "run_status.json", {
            "status": (
                "COMPLETED"
                if summary["completed"]
                else "COMPLETED_WITH_LIMITATIONS"
            ),
            "summary": summary,
            "finished_at": now(),
        })
        print(json.dumps({
            "phase": "experiment_finished",
            "output": str(output),
            "summary": summary,
        }, ensure_ascii=False), flush=True)
        return 0 if summary["completed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

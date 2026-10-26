#!/usr/bin/env python3
"""Formal H1 v1.1 runner.

This file is orchestration only.  It does not redefine or mutate a State,
Trap, runtime, evaluator, checkpoint selector, accounting rule, stopping rule,
or historical result.  The v1.1 protocol and freeze manifest remain read-only
inputs.  The runner writes a new result directory and performs the prescribed
post-collection audit before any confirmatory analysis.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
import random
from pathlib import Path
import sys
from time import monotonic
from typing import Any

from adapters.lifecycle_dependency_material import MATERIAL_PROFILE
from adapters.lifecycle_incremental_v7_1_runtime import LifecycleIncrementalV71Runtime
from adapters.lifecycle_mechanism_runtime import (
    CONDITIONS,
    IncompatibleLifecycleBinding,
    LifecycleMechanismRuntime,
    lifecycle_interaction_interface,
)
from adapters.web_route_model_agent import load_provider_config
from run_lifecycle_v2_feasibility import Provider, Session, MAX_CALLS
from traps.dependency_v4 import DEPENDENCY_V4_DOSE5_PROFILE_VERSION


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "formal_H1_protocol_v1.1.json"
PROTOCOL_MD_PATH = ROOT / "formal_H1_protocol_v1.1.md"
MANIFEST_PATH = ROOT / "formal_H1_freeze_manifest_v1.1.json"
PREFLIGHT_PATH = ROOT / "formal_H1_preflight_audit_v1.1.md"
STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
BASE_CONDITIONS = ("BASELINE", "BRANCH", "DEPENDENCY", "EVIDENCE", "INCREMENTAL")
TRAPS = BASE_CONDITIONS[1:]
RESOURCE_KEYS = (
    "total_tokens", "input_tokens", "output_tokens", "provider_calls",
    "tool_calls", "interaction_depth", "wall_time",
)
STATUS_COMPLETED = "COMPLETED"
STATUS_TECHNICAL = "TECHNICAL_FAILURE"
STATUS_PROTOCOL = "PROTOCOL_VIOLATION"
STATUS_ND = "NOT_DEPLOYABLE"
STATUS_BINDING = "BINDING_REJECTED"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def redact_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: ("[PRESENT]" if key == "api_key" and value else value)
            for key, value in config.items() if key != "api_key" or value}


class AuditProvider(Provider):
    """Existing provider transport plus lossless local request/response records."""

    def __init__(self, config: dict[str, Any], interface: dict[str, Any]):
        super().__init__(config, interface)
        self.raw_records: list[dict[str, Any]] = []

    def complete_with_usage(self, messages: list[dict[str, Any]]):
        request_index = len(self.raw_records) + 1
        started = datetime.now(timezone.utc).isoformat()
        record: dict[str, Any] = {
            "request_index": request_index,
            "started_at": started,
            "messages": deepcopy(messages),
            "response_message": None,
            "usage": None,
            "outcome": "pending",
        }
        self.raw_records.append(record)
        try:
            message, usage = super().complete_with_usage(messages)
            record.update(response_message=deepcopy(message), usage=deepcopy(usage), outcome="success")
            return message, usage
        except Exception as exc:
            provider_record = self.request_log[-1] if self.request_log else {}
            record.update(
                usage=deepcopy(provider_record.get("usage")),
                outcome="failure",
                exception_type=type(exc).__name__,
            )
            raise
        finally:
            record["finished_at"] = datetime.now(timezone.utc).isoformat()


def protocol_summary(protocol: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    model = protocol["model"]
    sampling = protocol["sampling"]
    boundary = protocol["boundary"]
    stats = protocol["statistical_analysis"]
    return {
        "protocol_id": protocol.get("protocol_id"),
        "protocol_status": protocol.get("status"),
        "protocol_ready": protocol.get("ready"),
        "protocol_h1_started": protocol.get("h1_started"),
        "manifest_version": manifest.get("manifest_version"),
        "manifest_ready": manifest.get("ready"),
        "manifest_h1_started": manifest.get("h1_started"),
        "manifest_created_at_utc": manifest.get("created_at_utc"),
        "code_commit_status": manifest.get("code_commit_status"),
        "code_file_count": manifest.get("code_file_count"),
        "model": model,
        "source_attempts": sampling.get("source_trajectory_attempts"),
        "max_source_provider_calls": sampling.get("max_source_provider_calls"),
        "max_checkpoints": sampling.get("max_checkpoints"),
        "max_condition_slots": sampling.get("max_condition_slots"),
        "boundary": boundary,
        "bootstrap_seed": stats.get("random_seed"),
        "bootstrap_draws": stats.get("bootstrap_draws"),
    }


def assert_preflight(protocol: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected = {
        "protocol_id": "formal_H1_protocol_v1.1",
        "conditions": list(BASE_CONDITIONS),
        "deployable_states": list(STATES),
        "source_trajectory_attempts": 30,
        "max_source_provider_calls": 32,
        "max_checkpoints": 90,
        "max_condition_slots": 450,
        "technical_retries": 0,
        "state_order": list(STATES),
        "base_condition_order": list(BASE_CONDITIONS),
        "boundary_version": "pre-effect-v1",
        "initial_exposures": 1,
        "maximum_followups": 5,
        "model_name": "deepseek-ai/DeepSeek-V3.2",
        "temperature": 0.0,
        "max_output_tokens": 2400,
        "timeout_seconds": 90,
        "parallel_tool_calls": False,
        "bootstrap_seed": 20260928,
        "bootstrap_draws": 9999,
    }
    actual = {
        "protocol_id": protocol.get("protocol_id"),
        "conditions": protocol.get("conditions"),
        "deployable_states": protocol.get("deployable_states"),
        "source_trajectory_attempts": protocol["sampling"].get("source_trajectory_attempts"),
        "max_source_provider_calls": protocol["sampling"].get("max_source_provider_calls"),
        "max_checkpoints": protocol["sampling"].get("max_checkpoints"),
        "max_condition_slots": protocol["sampling"].get("max_condition_slots"),
        "technical_retries": protocol.get("technical_retries"),
        "state_order": protocol.get("state_order"),
        "base_condition_order": protocol.get("base_condition_order"),
        "boundary_version": protocol["boundary"].get("version"),
        "initial_exposures": protocol["boundary"].get("initial_exposures"),
        "maximum_followups": protocol["boundary"].get("maximum_followups"),
        "model_name": protocol["model"].get("name"),
        "temperature": protocol["model"].get("temperature"),
        "max_output_tokens": protocol["model"].get("max_output_tokens"),
        "timeout_seconds": protocol["model"].get("timeout_seconds"),
        "parallel_tool_calls": protocol["model"].get("parallel_tool_calls"),
        "bootstrap_seed": protocol["statistical_analysis"].get("random_seed"),
        "bootstrap_draws": protocol["statistical_analysis"].get("bootstrap_draws"),
    }
    if actual != expected:
        raise RuntimeError(f"v1.1 preflight mismatch: {actual}")
    if not protocol.get("ready") or not manifest.get("ready"):
        raise RuntimeError("v1.1 provider-free preflight is not ready")
    if protocol.get("h1_started") or manifest.get("h1_started"):
        raise RuntimeError("frozen v1.1 inputs already report H1 started")
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        raise RuntimeError("freeze manifest protocol identity mismatch")
    if not PROTOCOL_MD_PATH.exists() or not PREFLIGHT_PATH.exists():
        raise RuntimeError("frozen v1.1 protocol or preflight evidence is missing")


def first_qualified_checkpoints(source_dir: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    paths = sorted((source_dir / "checkpoints").glob("event-*-before_call.json"))
    for path in paths:
        row = read_json(path)
        state = row.get("state")
        pending = row.get("pending_tool_call")
        history = row.get("agent_history") or []
        last = history[-1] if history else {}
        committed = bool(
            pending
            and last.get("role") == "assistant"
            and pending in (last.get("tool_calls") or [])
        )
        qualifies = bool(
            state in STATES
            and state not in selected
            and row.get("admitted") is True
            and row.get("terminal_reason") is None
            and row.get("trap_injected") is False
            and row.get("boundary_kind") == "before_call"
            and committed
        )
        if qualifies:
            row["_checkpoint_path"] = str(path)
            selected[state] = row
    return selected


def rotated_conditions(source_index: int) -> list[str]:
    shift = (source_index - 1) % len(BASE_CONDITIONS)
    return list(BASE_CONDITIONS[shift:] + BASE_CONDITIONS[:shift])


def provider_request_usage(record: dict[str, Any]) -> dict[str, Any]:
    usage = record.get("usage")
    if not isinstance(usage, dict):
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    values = {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    if values["total_tokens"] is None and values["input_tokens"] is not None and values["output_tokens"] is not None:
        values["total_tokens"] = values["input_tokens"] + values["output_tokens"]
    return values


def sum_usage(records: list[dict[str, Any]], key: str) -> int | None:
    values = [provider_request_usage(item).get(key) for item in records]
    return sum(values) if values and all(value is not None for value in values) else (0 if not values else None)


def payload_measurement(runtime: Any) -> dict[str, Any]:
    event = next((row for row in runtime.lifecycle_events if row.get("kind") == "initial_exposure"), None)
    visible = event.get("visible_result", {}) if event else {}
    trap_information = visible.get("trap_information") if isinstance(visible, dict) else None
    if trap_information is None:
        return {
            "initial_payload_present": False,
            "initial_payload_text": None,
            "initial_payload_unicode_chars": None,
            "initial_payload_utf8_bytes": None,
            "initial_payload_tokens_estimated": None,
            "initial_payload_tokens_offline": None,
            "initial_payload_tokenizer_version": None,
        }
    text = json.dumps(trap_information, ensure_ascii=False)
    chars = len(text)
    return {
        "initial_payload_present": True,
        "initial_payload_text": text,
        "initial_payload_unicode_chars": chars,
        "initial_payload_utf8_bytes": len(text.encode("utf-8")),
        "initial_payload_tokens_estimated": math.ceil(chars / 4),
        "initial_payload_tokens_offline": None,
        "initial_payload_tokenizer_version": None,
    }


def ordinary_tool_errors(runtime: Any) -> list[dict[str, Any]]:
    return [
        {
            "kind": event.get("kind"),
            "status": event.get("status"),
            "result": deepcopy(event.get("result")),
            "call": deepcopy(event.get("call")),
        }
        for event in runtime.lifecycle_events
        if event.get("kind") == "tool_result" and event.get("status") == "tool_error"
    ]


def make_metrics(runtime: Any, provider: AuditProvider, baseline_requests: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    metrics = deepcopy(runtime.metrics())
    requests = deepcopy(provider.raw_records)
    usages = [provider_request_usage(item) for item in requests]
    metrics["request_usage"] = usages
    metrics["request_count_from_provider_log"] = len(requests)
    metrics["all_request_total_usage_complete"] = all(item["total_tokens"] is not None for item in usages)
    metrics["ordinary_tool_errors"] = ordinary_tool_errors(runtime)
    metrics["payload_measurement"] = payload_measurement(runtime)
    metrics["first_prompt_input_tokens"] = usages[0]["input_tokens"] if usages else None
    metrics["later_request_total_tokens"] = sum_usage(requests[1:], "total_tokens")
    if baseline_requests is not None:
        baseline_usage = [provider_request_usage(item) for item in baseline_requests]
        first_trap = metrics["first_prompt_input_tokens"]
        first_base = baseline_usage[0]["input_tokens"] if baseline_usage else None
        metrics["first_prompt_input_delta"] = first_trap - first_base if first_trap is not None and first_base is not None else None
        metrics["later_requests_delta"] = (
            metrics["later_request_total_tokens"] - sum_usage(baseline_requests[1:], "total_tokens")
            if metrics["later_request_total_tokens"] is not None
            and sum_usage(baseline_requests[1:], "total_tokens") is not None else None
        )
    return metrics


def post_exposure_delta(trap_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> int | None:
    total = trap_metrics.get("total_tokens")
    first_delta = trap_metrics.get("first_prompt_input_delta")
    return total - first_delta if total is not None and first_delta is not None else None


def resource_deltas(trap: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in RESOURCE_KEYS:
        left = trap.get(key)
        right = baseline.get(key)
        result[key] = left - right if left is not None and right is not None else None
    result["delta_total_tokens"] = result.get("total_tokens")
    result["delta_input_tokens"] = result.get("input_tokens")
    result["delta_output_tokens"] = result.get("output_tokens")
    result["delta_provider_calls"] = result.get("provider_calls")
    result["delta_tool_calls"] = result.get("tool_calls")
    result["delta_interaction_depth"] = result.get("interaction_depth")
    result["delta_wall_time"] = result.get("wall_time")
    result["first_prompt_input_delta"] = trap.get("first_prompt_input_delta")
    result["post_exposure_resource_delta"] = post_exposure_delta(trap, baseline)
    result["later_requests_delta"] = trap.get("later_requests_delta")
    return result


def restore_runtime(checkpoint: dict[str, Any], provider: AuditProvider, condition: str):
    if condition == "INCREMENTAL":
        return LifecycleIncrementalV71Runtime.restore(
            deepcopy(checkpoint), provider, condition=condition,
            dependency_profile_version=DEPENDENCY_V4_DOSE5_PROFILE_VERSION,
        )
    return LifecycleMechanismRuntime.restore(
        deepcopy(checkpoint), provider, condition=condition,
        dependency_profile_version=DEPENDENCY_V4_DOSE5_PROFILE_VERSION,
    )


def preflight_condition(checkpoint: dict[str, Any], condition: str, config: dict[str, Any], interface: dict[str, Any]) -> dict[str, Any]:
    provider = AuditProvider(deepcopy(config), deepcopy(interface))
    runtime = None
    result: dict[str, Any] = {
        "condition": condition,
        "checkpoint_id": checkpoint.get("checkpoint_id"),
        "status": "READY",
        "provider_calls_before_run": 0,
        "snapshot_equal": False,
        "history_equal": False,
        "pre_effect_contract": "pre-effect-v1",
    }
    try:
        runtime = restore_runtime(checkpoint, provider, condition)
        exported = runtime.export_pre_effect_checkpoint()
        result["snapshot_equal"] = exported.get("environment_snapshot") == checkpoint.get("environment_snapshot")
        result["history_equal"] = exported.get("agent_history") == checkpoint.get("agent_history")
        result["current_plan_equal"] = exported.get("current_plan") == checkpoint.get("current_plan")
        result["pending_action_equal"] = exported.get("pending_tool_call") == checkpoint.get("pending_tool_call")
        if not all(result[key] for key in ("snapshot_equal", "history_equal", "current_plan_equal", "pending_action_equal")):
            raise RuntimeError("pre-effect restore equality failed")
        if condition == "INCREMENTAL":
            result["p0_audit"] = runtime.core.audit_record()
            result["p0_hard_valid"] = runtime.core.current_record.get("hard_task_validity") == "valid"
            result["p0_residual_gap_count"] = runtime.core.current_record.get("residual_gap_count")
        return result
    except IncompatibleLifecycleBinding as exc:
        result["status"] = STATUS_BINDING if condition == "DEPENDENCY" else STATUS_ND
        result["reason"] = exc.reason
        result["evidence"] = deepcopy(exc.evidence)
        if condition == "INCREMENTAL":
            result["status"] = STATUS_ND
        return result
    except Exception as exc:
        result["status"] = STATUS_TECHNICAL
        result["technical_error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if runtime is not None:
            runtime.close()


def run_condition(checkpoint: dict[str, Any], condition: str, config: dict[str, Any], interface: dict[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    provider = AuditProvider(deepcopy(config), deepcopy(interface))
    runtime = None
    started = datetime.now(timezone.utc).isoformat()
    row: dict[str, Any] = {
        "condition": condition,
        "checkpoint_id": checkpoint.get("checkpoint_id"),
        "status": STATUS_COMPLETED,
        "started_at": started,
        "runtime_version": "formal-H1-v1.1-orchestration",
        "technical_retries": 0,
    }
    try:
        runtime = restore_runtime(checkpoint, provider, condition)
        exported = runtime.export_pre_effect_checkpoint()
        row["pre_run_snapshot_equal"] = exported.get("environment_snapshot") == checkpoint.get("environment_snapshot")
        row["pre_run_history_equal"] = exported.get("agent_history") == checkpoint.get("agent_history")
        row["pre_run_current_plan_equal"] = exported.get("current_plan") == checkpoint.get("current_plan")
        row["pre_run_pending_action_equal"] = exported.get("pending_tool_call") == checkpoint.get("pending_tool_call")
        if not all(row[key] for key in (
            "pre_run_snapshot_equal", "pre_run_history_equal", "pre_run_current_plan_equal", "pre_run_pending_action_equal")):
            raise RuntimeError("pre-effect restore equality failed before provider request")
        metrics = runtime.run()
        row["metrics"] = make_metrics(runtime, provider)
        row["metrics"].update({key: metrics.get(key) for key in metrics if key not in row["metrics"]})
        if metrics.get("technical_failure"):
            row["status"] = STATUS_TECHNICAL
        elif metrics.get("protocol_violation"):
            row["status"] = STATUS_PROTOCOL
        row["technical_failure"] = bool(metrics.get("technical_failure"))
        row["protocol_violation"] = bool(metrics.get("protocol_violation"))
        row["ordinary_tool_errors"] = ordinary_tool_errors(runtime)
        row["final_snapshot"] = runtime.session.payload()
        row["lifecycle_events"] = deepcopy(runtime.lifecycle_events)
        row["accounting_events"] = deepcopy(runtime.envelope.events)
    except IncompatibleLifecycleBinding as exc:
        row["status"] = STATUS_BINDING if condition == "DEPENDENCY" else STATUS_ND
        row["reason"] = exc.reason
        row["evidence"] = deepcopy(exc.evidence)
    except Exception as exc:
        row["status"] = STATUS_TECHNICAL
        row["technical_error"] = f"{type(exc).__name__}: {exc}"
        if runtime is not None:
            try:
                row["partial_metrics"] = make_metrics(runtime, provider)
                row["partial_final_snapshot"] = runtime.session.payload()
                row["lifecycle_events"] = deepcopy(runtime.lifecycle_events)
                row["accounting_events"] = deepcopy(runtime.envelope.events)
            except Exception as nested:
                row["partial_capture_error"] = f"{type(nested).__name__}: {nested}"
    finally:
        row["finished_at"] = datetime.now(timezone.utc).isoformat()
        row["provider_request_log"] = deepcopy(provider.request_log)
        row["provider_raw_records"] = deepcopy(provider.raw_records)
        row["provider_usage"] = [provider_request_usage(item) for item in provider.raw_records]
        row["provider_call_count"] = len(provider.raw_records)
        if runtime is not None:
            runtime.close()
        dump_json(checkpoint_dir / "branches" / f"{condition}.json", row)
        dump_json(checkpoint_dir / "branches" / condition / "provider_requests.json", provider.raw_records)
    return row


def branch_pair_clean(branch: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if branch.get("status") != STATUS_COMPLETED:
        reasons.append("trap_not_completed")
    if baseline.get("status") != STATUS_COMPLETED:
        reasons.append("baseline_not_completed")
    for prefix, row in (("trap", branch), ("baseline", baseline)):
        metrics = row.get("metrics") or {}
        if row.get("technical_failure") or metrics.get("technical_failure"):
            reasons.append(f"{prefix}_technical_failure")
        if row.get("protocol_violation") or metrics.get("protocol_violation"):
            reasons.append(f"{prefix}_protocol_violation")
        if metrics.get("post_terminal_model_calls") != 0:
            reasons.append(f"{prefix}_post_terminal_provider_calls")
        if metrics.get("followups") is None or metrics.get("followups", 99) > 5:
            reasons.append(f"{prefix}_followup_budget")
        if not metrics.get("all_request_total_usage_complete", False):
            reasons.append(f"{prefix}_missing_total_usage")
        if not row.get("pre_run_snapshot_equal") or not row.get("pre_run_history_equal"):
            reasons.append(f"{prefix}_restore_mismatch")
    trap_events = branch.get("accounting_events") or []
    base_events = baseline.get("accounting_events") or []
    for prefix, events in (("trap", trap_events), ("baseline", base_events)):
        if not any(event.get("kind") == "initial_exposure" for event in events):
            reasons.append(f"{prefix}_missing_initial_exposure")
        if any(event.get("kind") == "provider_response" and False for event in events):
            reasons.append(f"{prefix}_invalid_accounting")
    return not reasons, reasons


def source_attempt(output: Path, index: int, config: dict[str, Any], interface: dict[str, Any]) -> dict[str, Any]:
    source_dir = output / "sources" / f"source-{index:02d}"
    source_dir.mkdir(parents=True, exist_ok=False)
    provider = AuditProvider(deepcopy(config), deepcopy(interface))
    session = Session(provider, f"formal-h1-v1.1-source-{index:02d}", dependency_material_profile=MATERIAL_PROFILE)
    dump_json(source_dir / "source_initial.json", session.payload())
    row: dict[str, Any] = {
        "source_index": index,
        "source_id": session.env.instance_id,
        "attempt_started": False,
        "max_provider_calls": MAX_CALLS,
        "dependency_material_profile": MATERIAL_PROFILE,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        session.run(source_dir)
        row["attempt_started"] = bool(provider.request_log)
        row["source_run_status"] = "COMPLETED"
    except Exception as exc:
        row["attempt_started"] = bool(provider.request_log)
        row["source_run_status"] = STATUS_TECHNICAL
        row["technical_error"] = f"{type(exc).__name__}: {exc}"
        session.terminal_reason = "technical_failure"
    finally:
        row.update({
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "provider_calls": session.provider_calls,
            "provider_request_log": deepcopy(provider.request_log),
            "provider_raw_records": deepcopy(provider.raw_records),
            "provider_usage": [provider_request_usage(item) for item in provider.raw_records],
            "trajectory": session.payload(),
        })
        dump_json(source_dir / "source_trajectory.json", row)
        session.env.close()
    selected = first_qualified_checkpoints(source_dir)
    selection = {
        "source_index": index,
        "source_id": session.env.instance_id,
        "selection_locked_before_binding": True,
        "selection_rule": "first admitted nonterminal uninjected before_call checkpoint with committed tool call per State",
        "state_order": list(STATES),
        "selected": {
            state: {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "path": str(Path(checkpoint["_checkpoint_path"]).relative_to(ROOT)),
                "boundary_kind": checkpoint.get("boundary_kind"),
                "admitted": checkpoint.get("admitted"),
                "terminal_reason": checkpoint.get("terminal_reason"),
                "trap_injected": checkpoint.get("trap_injected"),
                "committed_tool_call": True,
            }
            for state, checkpoint in selected.items()
        },
        "missing_states": [state for state in STATES if state not in selected],
    }
    dump_json(source_dir / "selection.json", selection)
    row["selection"] = selection
    row["selected_checkpoints"] = selected
    return row


def process_source(output: Path, source_row: dict[str, Any], config: dict[str, Any], interface: dict[str, Any]) -> list[dict[str, Any]]:
    index = source_row["source_index"]
    source_dir = output / "sources" / f"source-{index:02d}"
    selected = source_row["selected_checkpoints"]
    all_rows: list[dict[str, Any]] = []
    dump_json(source_dir / "selection_locked.json", source_row["selection"])
    for state in STATES:
        checkpoint = selected.get(state)
        checkpoint_dir = source_dir / "checkpoints" / (checkpoint["checkpoint_id"] if checkpoint else f"NO_CHECKPOINT-{state}")
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        if checkpoint is None:
            dump_json(checkpoint_dir / "checkpoint_status.json", {
                "source_index": index, "state": state, "status": "NO_CHECKPOINT",
                "selection_locked_before_binding": True,
            })
            continue
        dump_json(checkpoint_dir / "selected_checkpoint.json", checkpoint)
        preflight_rows = [preflight_condition(checkpoint, condition, config, interface) for condition in BASE_CONDITIONS]
        dump_json(checkpoint_dir / "preflight.json", {
            "source_index": index, "state": state, "checkpoint_id": checkpoint["checkpoint_id"],
            "preflight_order": list(BASE_CONDITIONS), "rows": preflight_rows,
        })
        by_condition = {row["condition"]: row for row in preflight_rows}
        execution_order = rotated_conditions(index)
        branch_rows: dict[str, dict[str, Any]] = {}
        for condition in execution_order:
            preflight = by_condition[condition]
            if preflight["status"] != "READY":
                branch_rows[condition] = {
                    "condition": condition,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "status": preflight["status"],
                    "reason": preflight.get("reason") or preflight.get("technical_error"),
                    "evidence": deepcopy(preflight.get("evidence", {})),
                    "provider_call_count": 0,
                    "provider_usage": [],
                    "resource_metrics": {key: None for key in RESOURCE_KEYS},
                    "selection_locked_before_binding": True,
                }
                dump_json(checkpoint_dir / "branches" / f"{condition}.json", branch_rows[condition])
                continue
            print(json.dumps({"phase": "branch_start", "source_index": index, "state": state,
                              "checkpoint_id": checkpoint["checkpoint_id"], "condition": condition}, ensure_ascii=False), flush=True)
            branch_rows[condition] = run_condition(checkpoint, condition, config, interface, checkpoint_dir)
        baseline = branch_rows.get("BASELINE", {})
        for condition in TRAPS:
            branch = branch_rows.get(condition, {})
            record = {
                "source_index": index,
                "state": state,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "condition": condition,
                "deployment_status": branch.get("status"),
                "preflight_status": by_condition[condition].get("status"),
                "baseline_status": baseline.get("status"),
                "status": branch.get("status"),
                "selection_locked_before_binding": True,
            }
            if branch.get("status") in {STATUS_COMPLETED, STATUS_TECHNICAL, STATUS_PROTOCOL}:
                clean, reasons = branch_pair_clean(branch, baseline)
                record["pair_clean"] = clean
                record["pair_clean_reasons"] = reasons
                trap_metrics = branch.get("metrics") or branch.get("partial_metrics") or {}
                baseline_metrics = baseline.get("metrics") or baseline.get("partial_metrics") or {}
                trap_metrics = deepcopy(trap_metrics)
                baseline_metrics = deepcopy(baseline_metrics)
                trap_metrics["first_prompt_input_delta"] = (
                    trap_metrics.get("first_prompt_input_tokens") - baseline_metrics.get("first_prompt_input_tokens")
                    if trap_metrics.get("first_prompt_input_tokens") is not None
                    and baseline_metrics.get("first_prompt_input_tokens") is not None else None
                )
                trap_metrics["later_requests_delta"] = (
                    (trap_metrics.get("later_request_total_tokens") - baseline_metrics.get("later_request_total_tokens"))
                    if trap_metrics.get("later_request_total_tokens") is not None
                    and baseline_metrics.get("later_request_total_tokens") is not None else None
                )
                record["baseline_metrics"] = baseline_metrics
                record["trap_metrics"] = trap_metrics
                record["resource_deltas"] = resource_deltas(trap_metrics, baseline_metrics)
            else:
                record["pair_clean"] = False
                record["pair_clean_reasons"] = ["not_deployed_or_not_run"]
                record["resource_deltas"] = {key: None for key in RESOURCE_KEYS}
            dump_json(checkpoint_dir / "pairs" / f"{condition}.json", record)
            all_rows.append(record)
        dump_json(checkpoint_dir / "branch_order.json", {
            "source_index": index, "state": state, "execution_order": execution_order,
            "base_condition_order": list(BASE_CONDITIONS), "left_rotation": (index - 1) % 5,
        })
    return all_rows


def design_matrix(rows: list[dict[str, Any]], restricted: bool = False) -> tuple[list[list[float]], list[str], list[str]]:
    sources = sorted({str(row["source_index"]) for row in rows})
    source_ref = sources[0] if sources else None
    names = ["intercept"]
    names.extend(f"source[{source}]" for source in sources[1:])
    names.extend(["state[EXPLOITATION]", "state[VERIFICATION]"])
    names.extend(["trap[DEPENDENCY]", "trap[EVIDENCE]", "trap[INCREMENTAL]"])
    if not restricted:
        names.extend([
            "interaction[EXPLOITATION:DEPENDENCY]", "interaction[EXPLOITATION:EVIDENCE]",
            "interaction[EXPLOITATION:INCREMENTAL]", "interaction[VERIFICATION:DEPENDENCY]",
            "interaction[VERIFICATION:EVIDENCE]", "interaction[VERIFICATION:INCREMENTAL]",
        ])
    matrix = []
    for row in rows:
        state = row["state"]
        trap = row["condition"]
        values = [1.0]
        values.extend(float(str(row["source_index"]) == source) for source in sources[1:])
        values.extend([float(state == "EXPLOITATION"), float(state == "VERIFICATION")])
        values.extend([float(trap == "DEPENDENCY"), float(trap == "EVIDENCE"), float(trap == "INCREMENTAL")])
        if not restricted:
            values.extend([
                float(state == "EXPLOITATION" and trap == "DEPENDENCY"),
                float(state == "EXPLOITATION" and trap == "EVIDENCE"),
                float(state == "EXPLOITATION" and trap == "INCREMENTAL"),
                float(state == "VERIFICATION" and trap == "DEPENDENCY"),
                float(state == "VERIFICATION" and trap == "EVIDENCE"),
                float(state == "VERIFICATION" and trap == "INCREMENTAL"),
            ])
        matrix.append(values)
    return matrix, names, sources


def matrix_rank(matrix: list[list[float]], tolerance: float = 1e-10) -> int:
    if not matrix or not matrix[0]:
        return 0
    work = [list(map(float, row)) for row in matrix]
    rows, columns = len(work), len(work[0])
    scale = max((abs(value) for row in work for value in row), default=1.0)
    threshold = max(tolerance, scale * tolerance)
    rank = 0
    for column in range(columns):
        pivot = max(range(rank, rows), key=lambda index: abs(work[index][column]), default=rank)
        if pivot >= rows or abs(work[pivot][column]) <= threshold:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        for index in range(rank + 1, rows):
            factor = work[index][column] / pivot_value
            if abs(factor) <= threshold:
                continue
            for trailing in range(column, columns):
                work[index][trailing] -= factor * work[rank][trailing]
        rank += 1
        if rank == rows:
            break
    return rank


def solve_linear(matrix: list[list[float]], values: list[float]) -> list[float]:
    size = len(values)
    work = [list(map(float, matrix[index])) + [float(values[index])] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda index: abs(work[index][column]))
        if abs(work[pivot][column]) <= 1e-12:
            raise ValueError("singular normal-equation matrix")
        work[column], work[pivot] = work[pivot], work[column]
        pivot_value = work[column][column]
        for trailing in range(column, size + 1):
            work[column][trailing] /= pivot_value
        for index in range(size):
            if index == column:
                continue
            factor = work[index][column]
            if abs(factor) <= 1e-14:
                continue
            for trailing in range(column, size + 1):
                work[index][trailing] -= factor * work[column][trailing]
    return [work[index][size] for index in range(size)]


def fit_ols(matrix: list[list[float]], values: list[float]) -> tuple[list[float], float, int]:
    if not matrix:
        return [], 0.0, 0
    columns = len(matrix[0])
    cross = [[0.0 for _ in range(columns)] for _ in range(columns)]
    product = [0.0 for _ in range(columns)]
    for row, value in zip(matrix, values):
        for left in range(columns):
            product[left] += row[left] * value
            for right in range(columns):
                cross[left][right] += row[left] * row[right]
    beta = solve_linear(cross, product)
    residual = [value - sum(row[index] * beta[index] for index in range(columns))
                for row, value in zip(matrix, values)]
    return beta, float(sum(value * value for value in residual)), matrix_rank(matrix)


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def audit_results(output: Path, source_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    source_attempts = len(source_rows)
    started = sum(bool(row.get("attempt_started")) for row in source_rows)
    selections = [row.get("selection", {}) for row in source_rows]
    selected_count = sum(len(selection.get("selected", {})) for selection in selections)
    checkpoint_rule_ok = all(
        all(item.get("admitted") is True and item.get("boundary_kind") == "before_call"
            and item.get("trap_injected") is False and item.get("committed_tool_call") is True
            for item in selection.get("selected", {}).values())
        for selection in selections
    )
    branch_files = list((output / "sources").glob("source-*/checkpoints/*/branches/*.json"))
    executed_branch_rows = []
    for path in branch_files:
        row = read_json(path)
        if row.get("metrics") or row.get("partial_metrics"):
            executed_branch_rows.append(row)
    snapshot_ok = all(
        row.get(key) is True
        for row in executed_branch_rows
        for key in ("pre_run_snapshot_equal", "pre_run_history_equal", "pre_run_current_plan_equal", "pre_run_pending_action_equal")
    )
    budget_ok = all((row.get("metrics") or row.get("partial_metrics") or {}).get("followups", 0) <= 5
                    for row in executed_branch_rows)
    terminal_ok = all((row.get("metrics") or row.get("partial_metrics") or {}).get("post_terminal_model_calls") == 0
                      for row in executed_branch_rows)
    accounting_ok = True
    accounting_issues = []
    for row in executed_branch_rows:
        metrics = row.get("metrics") or row.get("partial_metrics") or {}
        request_usage = row.get("provider_usage", [])
        if len(request_usage) != row.get("provider_call_count"):
            accounting_ok = False
            accounting_issues.append({"path": str(row.get("condition")), "issue": "request_count_mismatch"})
        missing = any(item.get("total_tokens") is None for item in request_usage)
        if missing and metrics.get("total_tokens") is not None:
            accounting_ok = False
            accounting_issues.append({"condition": row.get("condition"), "issue": "missing_usage_not_NA"})
        if row.get("status") in {STATUS_TECHNICAL, STATUS_PROTOCOL} and len(request_usage) == 0 and row.get("provider_request_log"):
            accounting_ok = False
            accounting_issues.append({"condition": row.get("condition"), "issue": "failure_usage_not_retained"})
    nd_rows = [row for row in branch_files if read_json(row).get("status") in {STATUS_ND, STATUS_BINDING}]
    nd_ok = True
    nd_issues = []
    for path in nd_rows:
        row = read_json(path)
        if row.get("provider_call_count") not in (None, 0):
            nd_ok = False
            nd_issues.append({"path": str(path), "issue": "nonzero_provider_calls_for_ND"})
        if any(value is not None for value in (row.get("resource_metrics") or {}).values()):
            nd_ok = False
            nd_issues.append({"path": str(path), "issue": "ND_resource_not_NA"})
    pair_clean_ok = all(row.get("pair_clean") is not None for row in pair_rows)
    cell_counts: dict[str, dict[str, Any]] = {
        state: {trap: {"clean_pair_count": 0, "distinct_sources": []} for trap in TRAPS} for state in STATES
    }
    for row in pair_rows:
        cell = cell_counts[row["state"]][row["condition"]]
        if row.get("pair_clean"):
            cell["clean_pair_count"] += 1
            cell["distinct_sources"] = sorted(set(cell["distinct_sources"] + [row["source_index"]]))
    clean_rows = [row for row in pair_rows if row.get("pair_clean") and row.get("resource_deltas", {}).get("delta_total_tokens") is not None]
    full_matrix, full_names, full_sources = design_matrix(clean_rows, restricted=False) if clean_rows else ([], [], [])
    restricted_matrix, restricted_names, _ = design_matrix(clean_rows, restricted=True) if clean_rows else ([], [], [])
    full_rank_value = matrix_rank(full_matrix) if clean_rows else 0
    restricted_rank_value = matrix_rank(restricted_matrix) if clean_rows else 0
    full_rank = bool(clean_rows) and full_rank_value == len(full_matrix[0])
    restricted_rank = bool(clean_rows) and restricted_rank_value == len(restricted_matrix[0])
    residual_df = int(len(clean_rows) - full_rank_value) if clean_rows else -1
    cell_threshold = all(len(cell_counts[state][trap]["distinct_sources"]) >= 8 for state in STATES for trap in TRAPS)
    source_threshold = len({row["source_index"] for row in clean_rows}) >= 20
    estimability = {
        "minimum_distinct_clean_sources": 20,
        "observed_distinct_clean_sources": len({row["source_index"] for row in clean_rows}),
        "minimum_distinct_clean_sources_per_cell": 8,
        "all_cells_meet_source_threshold": cell_threshold,
        "full_design_rank": full_rank,
        "restricted_design_rank": restricted_rank,
        "full_design_columns": len(full_matrix[0]) if clean_rows else 0,
        "full_design_rank_value": full_rank_value,
        "residual_df": residual_df,
        "positive_residual_df": residual_df > 0,
        "estimability_threshold_pass": source_threshold and cell_threshold and full_rank and residual_df > 0,
        "cell_counts": cell_counts,
    }
    checks = {
        "source_checkpoint_selection": source_attempts == 30 and started == 30 and selected_count <= 90 and checkpoint_rule_ok,
        "snapshot_pre_effect_followup": snapshot_ok and budget_ok and terminal_ok,
        "token_accounting_and_NA": accounting_ok,
        "pair_clean_accounting": pair_clean_ok,
        "deployability_ND": nd_ok,
        "technical_protocol_failure_accounting": all(
            row.get("status") not in {STATUS_TECHNICAL, STATUS_PROTOCOL} or row.get("provider_call_count") is not None
            for row in executed_branch_rows
        ),
    }
    structural_pass = all(checks.values())
    audit = {
        "audit_version": "formal-H1-v1.1-completeness-protocol-audit-v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "structural_audit_pass": structural_pass,
        "source_attempts": source_attempts,
        "started_source_attempts": started,
        "selected_checkpoint_count": selected_count,
        "executed_branch_count": len(executed_branch_rows),
        "pair_row_count": len(pair_rows),
        "accounting_issues": accounting_issues,
        "ND_issues": nd_issues,
        "estimability": estimability,
        "protocol": {
            "source_attempt_target": protocol["sampling"]["source_trajectory_attempts"],
            "condition_slots_target": protocol["sampling"]["max_condition_slots"],
            "no_replacement": protocol["sampling"]["replacement"] is False,
            "no_interim_effect_review": protocol["sampling"]["interim_effect_review"] is False,
        },
    }
    dump_json(output / "audit" / "formal_h1_completeness_protocol_audit.json", audit)
    return audit


def run_analysis(output: Path, pair_rows: list[dict[str, Any]], audit: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    estimability = audit["estimability"]
    clean_rows = [row for row in pair_rows if row.get("pair_clean") and row.get("resource_deltas", {}).get("delta_total_tokens") is not None]
    result: dict[str, Any] = {
        "analysis_version": "formal-H1-v1.1-preregistered-analysis-v1",
        "protocol_id": protocol["protocol_id"],
        "analysis_started_after_audit": audit.get("structural_audit_pass") is True,
        "estimability": estimability,
        "clean_pair_rows": len(clean_rows),
        "model": protocol["statistical_analysis"],
    }
    if not audit.get("structural_audit_pass"):
        result.update({"verdict": "ANALYSIS_NOT_RUN_AUDIT_FAILED", "formal_interaction": None})
        dump_json(output / "analysis" / "formal_h1_analysis.json", result)
        return result
    if not estimability.get("estimability_threshold_pass"):
        result.update({"verdict": "H1_INTERACTION_INCONCLUSIVE", "formal_interaction": None,
                       "reason": "one or more preregistered estimability thresholds failed"})
        dump_json(output / "analysis" / "formal_h1_analysis.json", result)
        return result
    y = [float(row["resource_deltas"]["delta_total_tokens"]) for row in clean_rows]
    full, full_names, sources = design_matrix(clean_rows, restricted=False)
    restricted, restricted_names, _ = design_matrix(clean_rows, restricted=True)
    beta_full, rss_full, rank_full = fit_ols(full, y)
    beta_restricted, rss_restricted, rank_restricted = fit_ols(restricted, y)
    df_interaction = 6
    residual_df = len(y) - rank_full
    observed_f = ((rss_restricted - rss_full) / df_interaction) / (rss_full / residual_df) if residual_df > 0 and rss_full > 0 else float("inf")
    source_index = {source: index for index, source in enumerate(sources)}
    restricted_fit = [sum(row[index] * beta_restricted[index] for index in range(len(beta_restricted)))
                      for row in restricted]
    residual = [value - fitted for value, fitted in zip(y, restricted_fit)]
    seed = int(protocol["statistical_analysis"]["random_seed"])
    draws = int(protocol["statistical_analysis"]["bootstrap_draws"])
    rng = random.Random(seed)
    bootstrap_f: list[float] = []
    interaction_indices = [index for index, name in enumerate(full_names) if name.startswith("interaction[")]
    bootstrap_beta: list[list[float]] = []
    row_source_indices = [source_index[str(row["source_index"])] for row in clean_rows]
    for draw in range(draws):
        signs = [float(rng.choice((-1, 1))) for _ in sources]
        simulated = [fitted + error * signs[source_index]
                     for fitted, error, source_index in zip(restricted_fit, residual, row_source_indices)]
        boot_full, boot_rss_full, _ = fit_ols(full, simulated)
        _, boot_rss_restricted, _ = fit_ols(restricted, simulated)
        bootstrap_f.append(((boot_rss_restricted - boot_rss_full) / df_interaction) / (boot_rss_full / residual_df) if boot_rss_full > 0 else float("inf"))
        bootstrap_beta.append([boot_full[index] for index in interaction_indices])
    p_value = float((1 + sum(value >= observed_f for value in bootstrap_f)) / (draws + 1))
    interaction_estimates = {
        full_names[index]: float(beta_full[index]) for index in interaction_indices
    }
    interaction_ci = {
        full_names[index]: [float(quantile([row[position] for row in bootstrap_beta], 0.025)),
                            float(quantile([row[position] for row in bootstrap_beta], 0.975))]
        for position, index in enumerate(interaction_indices)
    }
    result.update({
        "verdict": "H1_INTERACTION_SIGNIFICANCE_REPORTED" if p_value < 0.05 else "H1_INTERACTION_NOT_SIGNIFICANT",
        "formal_interaction": {
            "rss_full": rss_full,
            "rss_restricted": rss_restricted,
            "observed_f": float(observed_f),
            "interaction_df": df_interaction,
            "residual_df": residual_df,
            "p_value": p_value,
            "alpha_two_sided": 0.05,
            "bootstrap_draws": draws,
            "random_seed": seed,
            "interaction_estimates": interaction_estimates,
            "interaction_bootstrap_ci_95": interaction_ci,
            "full_rank": rank_full == full.shape[1],
            "restricted_rank": rank_restricted == restricted.shape[1],
        },
    })
    dump_json(output / "analysis" / "formal_h1_analysis.json", result)
    return result


def write_audit_markdown(output: Path, audit: dict[str, Any], analysis: dict[str, Any]) -> None:
    lines = [
        "# formal H1 v1.1 completeness and protocol audit",
        "",
        f"- Structural audit pass: `{str(audit['structural_audit_pass']).lower()}`",
        f"- Source attempts started: `{audit['started_source_attempts']}/30`",
        f"- Selected checkpoints: `{audit['selected_checkpoint_count']}/90 maximum`",
        f"- Executed branch rows: `{audit['executed_branch_count']}`",
        f"- Pair rows: `{audit['pair_row_count']}`",
        "",
        "## Required checks",
        "",
    ]
    for key, value in audit["checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend([
        "",
        "## 12 State×Trap cell estimability",
        "",
        f"- Distinct clean sources: `{audit['estimability']['observed_distinct_clean_sources']}`; threshold `20`.",
        f"- All cell source thresholds: `{str(audit['estimability']['all_cells_meet_source_threshold']).lower()}`; threshold `8` per cell.",
        f"- Full design rank: `{str(audit['estimability']['full_design_rank']).lower()}`.",
        f"- Positive residual degrees of freedom: `{str(audit['estimability']['positive_residual_df']).lower()}`.",
        f"- Preregistered estimability threshold pass: `{str(audit['estimability']['estimability_threshold_pass']).lower()}`.",
        "",
        "## Analysis disposition",
        "",
        f"- `{analysis.get('verdict')}`",
        "",
        "All NOT_DEPLOYABLE, BINDING_REJECTED, technical-failure, protocol-violation, and pair-clean decisions are retained in the raw ledger; no outcome-driven replacement or extension is permitted.",
    ])
    (output / "audit" / "formal_h1_completeness_protocol_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    protocol = read_json(PROTOCOL_PATH)
    manifest = read_json(MANIFEST_PATH)
    assert_preflight(protocol, manifest)
    config = load_provider_config()
    if config.get("model") != protocol["model"]["name"]:
        raise RuntimeError(f"provider model mismatch: {config.get('model')}")
    interface = lifecycle_interaction_interface()
    if interface.get("interface_version") != protocol["model"]["interface"]:
        raise RuntimeError("provider interface version mismatch")
    stamp = utc_stamp()
    output = ROOT / "results" / f"formal_H1_v1_1_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    summary = protocol_summary(protocol, manifest)
    print("FORMAL_H1_STARTED = true", flush=True)
    print(json.dumps({"frozen_protocol_summary": summary}, ensure_ascii=False), flush=True)
    dump_json(output / "launch_record.json", {
        "formal_h1_started": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "frozen_protocol_summary": summary,
        "protocol_file": str(PROTOCOL_PATH.relative_to(ROOT)),
        "freeze_manifest_file": str(MANIFEST_PATH.relative_to(ROOT)),
        "integrity_controls": ["version numbers", "read-only permissions", "timestamps", "modification records"],
        "provider_config": redact_config(config),
        "interface": interface,
    })
    dump_json(output / "protocol.json", protocol)
    dump_json(output / "freeze_manifest.json", manifest)
    (output / "protocol.md").write_text(PROTOCOL_MD_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (output / "preflight_audit.md").write_text(PREFLIGHT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    dump_json(output / "runtime_configuration.json", {
        "provider_config": redact_config(config),
        "interface": interface,
        "dependency_material_profile": MATERIAL_PROFILE,
        "dependency_profile_version": DEPENDENCY_V4_DOSE5_PROFILE_VERSION,
        "source_max_provider_calls": MAX_CALLS,
        "conditions": list(BASE_CONDITIONS),
        "states": list(STATES),
        "technical_retries": 0,
    })
    append_jsonl(output / "progress.jsonl", {"event": "formal_h1_started", "at": datetime.now(timezone.utc).isoformat()})
    source_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for index in range(1, 31):
        print(json.dumps({"phase": "source_start", "source_index": index, "target": 30}, ensure_ascii=False), flush=True)
        source = source_attempt(output, index, config, interface)
        source_rows.append(source)
        pair_rows.extend(process_source(output, source, config, interface))
        dump_json(output / "source_attempts.json", source_rows)
        dump_json(output / "pair_ledger.json", pair_rows)
        append_jsonl(output / "progress.jsonl", {
            "event": "source_completed", "source_index": index,
            "attempt_started": source.get("attempt_started"),
            "selected_checkpoints": len(source.get("selection", {}).get("selected", {})),
            "pair_rows_total": len(pair_rows),
            "at": datetime.now(timezone.utc).isoformat(),
        })
    dump_json(output / "raw_ledger.json", {"source_attempts": source_rows, "pair_rows": pair_rows})
    audit = audit_results(output, source_rows, pair_rows, protocol)
    analysis = run_analysis(output, pair_rows, audit, protocol)
    write_audit_markdown(output, audit, analysis)
    dump_json(output / "final_result.json", {
        "formal_h1_started": True,
        "output_directory": str(output),
        "source_attempts": len(source_rows),
        "selected_checkpoint_count": audit["selected_checkpoint_count"],
        "audit": audit,
        "analysis": analysis,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"phase": "formal_h1_finished", "output": str(output),
                      "audit_pass": audit["structural_audit_pass"], "analysis_verdict": analysis.get("verdict")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("formal H1 interrupted; existing started attempts remain recorded")

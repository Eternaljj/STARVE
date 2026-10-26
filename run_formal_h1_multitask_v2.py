#!/usr/bin/env python3
"""Frozen orchestration for formal_H1_multitask_v2.

This file selects the fixed fixture schedule and delegates State collection,
Trap execution, accounting, clean-pair classification, and preregistered
analysis to the existing frozen implementations.  It never rewrites protocol
inputs or historical results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
from typing import Any

import run_formal_h1_v1_1 as frozen
from adapters.lifecycle_incremental_v7_1_runtime import LifecycleIncrementalV71Runtime
from adapters.lifecycle_mechanism_runtime import (
    LifecycleMechanismRuntime,
    lifecycle_interaction_interface,
)
from adapters.lifecycle_state_collector import classify_behavior
from adapters.web_route_model_agent import load_provider_config
from traps.dependency_v4 import DEPENDENCY_V4_DOSE5_PROFILE_VERSION
from tasks.multi_task_pool_v1_adapter import (
    FixtureRepository,
    FixtureRuntimeAdapter,
    MultiTaskSessionAdapter,
    _fixture_runtime_context,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "formal_H1_multitask_v2.md"
MANIFEST_PATH = ROOT / "formal_H1_multitask_v2_freeze_manifest.json"
PREFLIGHT_PATH = ROOT / "formal_H1_multitask_v2_provider_free_audit.md"
POOL_PATH = ROOT / "multi_task_pool_v1.json"
POOL_AUDIT_PATH = ROOT / "multi_task_pool_v1_audit.json"
RUNNER_PATH = Path(__file__).resolve()
OUTPUT_PREFIX = "formal_H1_multitask_v2_"
STATES = frozen.STATES
BASE_CONDITIONS = frozen.BASE_CONDITIONS
TRAPS = frozen.TRAPS
MAX_SOURCE_CALLS = 32


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_object(manifest: dict[str, Any]) -> dict[str, Any]:
    design = manifest["frozen_design"]
    model = manifest["model_configuration"]
    analysis = manifest["analysis"]
    return {
        "protocol_id": manifest["protocol_id"],
        "status": manifest["status"],
        "ready": manifest["ready"],
        "h1_started": manifest["h1_started"],
        "deployable_states": list(design["states"]),
        "conditions": list(design["conditions"]),
        "technical_retries": design["technical_retries"],
        "state_order": list(design["state_order"]),
        "base_condition_order": list(design["base_condition_order"]),
        "model": deepcopy(model),
        "sampling": {
            "source_trajectory_attempts": design["source_attempts"],
            "max_source_provider_calls": MAX_SOURCE_CALLS,
            "max_checkpoints": design["maximum_selected_checkpoints"],
            "max_condition_slots": design["maximum_condition_slots"],
            "source_is_statistical_cluster": True,
            "max_checkpoints_per_state_per_source": design["max_checkpoints_per_state_per_source"],
            "replacement": design["source_replacement"],
            "interim_effect_review": False,
        },
        "boundary": {
            "version": design["pre_effect_version"],
            "initial_exposures": design["initial_exposures"],
            "maximum_followups": design["maximum_followups"],
            "post_terminal_provider_or_tool_calls": 0,
        },
        "statistical_analysis": {
            "outcome": analysis["primary_outcome"],
            "unit": analysis["cluster"],
            "model_full": analysis["full_model"],
            "model_restricted": analysis["restricted_model"],
            "omnibus_interaction_df": analysis["interaction_df"],
            "test": analysis["bootstrap"],
            "bootstrap_draws": analysis["bootstrap_draws"],
            "random_seed": analysis["random_seed"],
            "alpha_two_sided": analysis["alpha_two_sided"],
            "minimum_distinct_clean_sources": analysis["minimum_distinct_clean_sources"],
            "minimum_distinct_clean_sources_per_state_trap_cell": analysis["minimum_distinct_clean_sources_per_state_trap_cell"],
            "require_full_rank_and_positive_residual_df": True,
            "if_unestimable": analysis["if_unestimable"],
            "family_template_role": analysis["family_template_role"],
        },
    }


def schedule_matches_pool(schedule: list[dict[str, Any]], pool: dict[str, Any]) -> bool:
    keys = ("family", "template_id", "instance_id", "fixture_id", "snapshot_id", "fixture_path")
    return (
        len(schedule) == len(pool.get("instances", [])) == 30
        and [row.get("source_index") for row in schedule] == list(range(1, 31))
        and all(all(left.get(key) == right.get(key) for key in keys)
                for left, right in zip(schedule, pool["instances"]))
    )


def launch_preflight(manifest: dict[str, Any], protocol: dict[str, Any],
                     config: dict[str, Any], interface: dict[str, Any]) -> dict[str, Any]:
    pool = read_json(POOL_PATH)
    pool_audit = read_json(POOL_AUDIT_PATH)
    schedule = manifest["source_schedule"]
    dependencies = manifest["code_and_interface_dependencies"]
    locked_inputs = [PROTOCOL_PATH, MANIFEST_PATH, PREFLIGHT_PATH, POOL_PATH, POOL_AUDIT_PATH]
    locked_inputs.extend(ROOT / row["fixture_path"] for row in schedule)
    checks = {
        "protocol_manifest_ready": (
            manifest.get("protocol_id") == "formal_H1_multitask_v2"
            and manifest.get("ready") is True
            and manifest.get("h1_started") is False
            and protocol.get("ready") is True
        ),
        "protocol_inputs_read_only": all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in locked_inputs),
        "thirty_fixture_schedule_exact": schedule_matches_pool(schedule, pool),
        "fixture_files_present": all((ROOT / row["fixture_path"]).is_file() for row in schedule),
        "coverage_3x2x5": (
            Counter(row["family"] for row in schedule) == Counter({"WEB": 10, "API": 10, "SERVICE": 10})
            and len(Counter(row["template_id"] for row in schedule)) == 6
            and all(value == 5 for value in Counter(row["template_id"] for row in schedule).values())
        ),
        "pool_provider_free_audit_ready": (
            pool.get("MULTI_TASK_POOL_READY") is True
            and pool_audit.get("MULTI_TASK_POOL_READY") is True
            and pool_audit.get("provider_calls") == 0
            and pool_audit.get("h1_started") is False
            and not pool_audit.get("blocking_issues")
        ),
        "frozen_dependency_metadata_matches": all(
            (ROOT / row["path"]).is_file()
            and (ROOT / row["path"]).stat().st_size == row["size_bytes_at_freeze"]
            and datetime.fromtimestamp((ROOT / row["path"]).stat().st_mtime, timezone.utc)
                .astimezone().strftime("%Y-%m-%d %H:%M:%S.%f %z")[:26]
                [:23] == row["modified_at_freeze"][:23]
            for row in dependencies
        ),
        "model_provider_config": (
            config.get("model") == manifest["model_configuration"]["name"]
            and config.get("temperature") == manifest["model_configuration"]["temperature"]
            and config.get("timeout_seconds") == manifest["model_configuration"]["timeout_seconds"]
            and bool(config.get("api_key"))
            and bool(config.get("base_url"))
        ),
        "runtime_interface_config": (
            interface.get("interface_version") == manifest["model_configuration"]["interface"]
            and manifest["model_configuration"]["max_output_tokens"] == 2400
            and manifest["model_configuration"]["tool_choice"] == "auto"
            and manifest["model_configuration"]["parallel_tool_calls"] is False
        ),
        "retry_stop_clean_pair_ND_accounting": (
            protocol["technical_retries"] == 0
            and protocol["sampling"]["source_trajectory_attempts"] == 30
            and protocol["sampling"]["replacement"] is False
            and protocol["sampling"]["interim_effect_review"] is False
            and protocol["boundary"]["maximum_followups"] == 5
            and manifest["analysis"]["primary_outcome"] == "clean-pair delta_total_tokens"
            and "NOT_DEPLOYABLE" in PROTOCOL_PATH.read_text(encoding="utf-8")
            and "BINDING_REJECTED" in PROTOCOL_PATH.read_text(encoding="utf-8")
            and "逐请求" in PROTOCOL_PATH.read_text(encoding="utf-8")
        ),
        "new_output_prefix_isolated": not any(
            path.name == OUTPUT_PREFIX.rstrip("_") for path in (ROOT / "results").glob("*")
        ),
        "runner_read_only": stat.S_IMODE(RUNNER_PATH.stat().st_mode) == 0o444,
    }
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed": all(checks.values()),
        "provider_calls": 0,
        "source_attempts_started": 0,
        "output_prefix": str(ROOT / "results" / OUTPUT_PREFIX),
    }


class FormalMultiTaskSourceSession(MultiTaskSessionAdapter):
    """Source collector with the unchanged lifecycle checkpoint schema."""

    def capture_behavior(self, directory: Path, boundary_kind: str):
        if self.terminal_reason is not None:
            return None
        evidence = self.behavior_observer(self.payload())
        record = classify_behavior(self.payload(), evidence)
        key = f"event-{len(self.collector_events) + 1:04d}-{boundary_kind}"
        record.update(
            checkpoint_id=f"{self.env.fixture['instance_id']}-{key}",
            trajectory_id=self.env.fixture["instance_id"],
            boundary_kind=boundary_kind,
        )
        path = directory / "checkpoints" / f"{key}.json"
        if path.exists():
            raise RuntimeError("refusing checkpoint overwrite")
        self.collector_events.append(deepcopy(record))
        self.captures.append({
            "key": key,
            "state": record["state"],
            "collector_status": record["collector_status"],
            "admitted": record["admitted"],
            "path": str(Path(directory.name) / "checkpoints" / path.name),
        })
        data = self.payload()
        data.update(
            record,
            state_evidence=record,
            captured_at=datetime.now(timezone.utc).isoformat(),
            trap_injected=False,
            exposure_contract="pre-effect-v1",
            checkpoint_semantics="committed_before_effect_redecision_allowed",
        )
        frozen.dump_json(path, data)
        return record

    def _execute_source_pending(self) -> dict[str, Any]:
        function = self.pending["function"]
        if function["name"] == "stop_iteration":
            args = json.loads(function["arguments"])
            if set(args) - {"reason"} or not isinstance(args.get("reason", ""), str):
                raise ValueError("stop reason must be text")
            self.terminal_reason = "autonomous_stop"
            return {"accepted": True, "terminal": True, "termination_reason": "autonomous_stop"}
        return self.execute_pending()

    def run(self, directory: Path) -> bool:
        while self.terminal_reason is None and self.provider_calls < MAX_SOURCE_CALLS:
            self.provider_calls += 1
            message = self.provider.complete(deepcopy(self.messages))
            calls = message.get("tool_calls") or []
            if self.provider_calls == 1:
                first_action = None
                if calls:
                    first_tool = calls[0].get("function", {})
                    try:
                        first_arguments = json.loads(first_tool.get("arguments", "{}"))
                    except (TypeError, json.JSONDecodeError):
                        first_arguments = {}
                    first_action = {
                        "tool": first_tool.get("name"),
                        "operation": first_arguments.get("operation"),
                        "arguments": first_arguments,
                    }
                self.first_behavior = {"assistant_prose": message.get("content"), "action": first_action}
            self.messages.append({
                "role": "assistant",
                "content": message.get("content"),
                **({"tool_calls": deepcopy(calls)} if calls else {}),
            })
            if not calls:
                self.terminal_reason = "autonomous_stop"
                break
            if len(calls) != 1:
                self.terminal_reason = "unsupported_parallel_calls"
                break
            self.pending = deepcopy(calls[0])
            function = self.pending.get("function", {})
            args: dict[str, Any] = {}
            try:
                args = json.loads(function.get("arguments", "{}"))
                if not isinstance(args, dict):
                    raise ValueError("arguments must be object")
                self.capture_behavior(directory, "before_call")
                result = self._execute_source_pending()
                status = "ok"
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                result, status = {"error": str(exc)}, "tool_error"
            self.tool_calls.append({
                "name": function.get("name"),
                "arguments": deepcopy(args),
                "result": deepcopy(result),
                "status": status,
                "provider_call": self.provider_calls,
            })
            self.messages.append({
                "role": "tool",
                "tool_call_id": calls[0].get("id", ""),
                "content": json.dumps(result, ensure_ascii=False),
            })
            self.pending = None
            self.capture_behavior(directory, "after_result")
            frozen.dump_json(directory / "trajectory_progress.json", self.payload())
            print(json.dumps({
                "trajectory": self.env.fixture["instance_id"],
                "call": self.provider_calls,
                "tool": function.get("name"),
                "operation": args.get("operation"),
                "status": status,
            }, ensure_ascii=False), flush=True)
        if self.terminal_reason is None:
            self.terminal_reason = "model_turn_cap"
        return self.terminal_reason == "task_completed"


def restore_runtime_v2(checkpoint: dict[str, Any], provider: frozen.AuditProvider, condition: str):
    runtime_class = LifecycleIncrementalV71Runtime if condition == "INCREMENTAL" else LifecycleMechanismRuntime
    snapshot = checkpoint["environment_snapshot"]
    with _fixture_runtime_context(snapshot):
        runtime = runtime_class.restore(
            deepcopy(checkpoint),
            provider,
            condition=condition,
            dependency_profile_version=DEPENDENCY_V4_DOSE5_PROFILE_VERSION,
        )
    return FixtureRuntimeAdapter(runtime, snapshot)


def source_attempt_v2(output: Path, schedule_row: dict[str, Any], config: dict[str, Any],
                      interface: dict[str, Any]) -> dict[str, Any]:
    index = schedule_row["source_index"]
    source_dir = output / "sources" / f"source-{index:02d}"
    source_dir.mkdir(parents=True, exist_ok=False)
    provider = frozen.AuditProvider(deepcopy(config), deepcopy(interface))
    session = FormalMultiTaskSourceSession(provider, schedule_row["fixture_id"])
    frozen.dump_json(source_dir / "source_initial.json", session.payload())
    row: dict[str, Any] = {
        **deepcopy(schedule_row),
        "source_id": session.env.fixture["instance_id"],
        "attempt_started": False,
        "max_provider_calls": MAX_SOURCE_CALLS,
        "dependency_material_profile": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        session.run(source_dir)
        row["attempt_started"] = bool(provider.request_log)
        row["source_run_status"] = "COMPLETED"
    except Exception as exc:
        row["attempt_started"] = bool(provider.request_log)
        row["source_run_status"] = frozen.STATUS_TECHNICAL
        row["technical_error"] = f"{type(exc).__name__}: {exc}"
        session.terminal_reason = "technical_failure"
    finally:
        row.update({
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "provider_calls": session.provider_calls,
            "provider_request_log": deepcopy(provider.request_log),
            "provider_raw_records": deepcopy(provider.raw_records),
            "provider_usage": [frozen.provider_request_usage(item) for item in provider.raw_records],
            "trajectory": session.payload(),
        })
        frozen.dump_json(source_dir / "source_trajectory.json", row)
        session.close()
    selected = frozen.first_qualified_checkpoints(source_dir)
    selection = {
        "source_index": index,
        "source_id": row["source_id"],
        "fixture_id": schedule_row["fixture_id"],
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
    frozen.dump_json(source_dir / "selection.json", selection)
    row["selection"] = selection
    row["selected_checkpoints"] = selected
    return row


_original_run_condition = frozen.run_condition
_original_audit_results = frozen.audit_results
_original_run_analysis = frozen.run_analysis


def run_condition_v2(checkpoint, condition, config, interface, checkpoint_dir):
    row = _original_run_condition(checkpoint, condition, config, interface, checkpoint_dir)
    row["runtime_version"] = "formal-H1-multitask-v2-orchestration-v1"
    frozen.dump_json(checkpoint_dir / "branches" / f"{condition}.json", row)
    return row


def process_source_v2(output, source_row, config, interface):
    rows = frozen.process_source(output, source_row, config, interface)
    identity = {key: source_row[key] for key in (
        "family", "template_id", "instance_id", "fixture_id", "snapshot_id"
    )}
    for row in rows:
        row.update(deepcopy(identity))
        path = (output / "sources" / f"source-{row['source_index']:02d}" / "checkpoints"
                / row["checkpoint_id"] / "pairs" / f"{row['condition']}.json")
        frozen.dump_json(path, row)
    return rows


def audit_results_v2(output, source_rows, pair_rows, protocol, schedule):
    audit = _original_audit_results(output, source_rows, pair_rows, protocol)
    schedule_ok = len(source_rows) == len(schedule) == 30 and all(
        all(observed.get(key) == expected.get(key) for key in (
            "source_index", "family", "template_id", "instance_id", "fixture_id", "snapshot_id"
        ))
        for observed, expected in zip(source_rows, schedule)
    )
    audit["audit_version"] = "formal-H1-multitask-v2-completeness-protocol-audit-v1"
    audit["checks"]["fixed_fixture_source_schedule"] = schedule_ok
    audit["checks"]["family_template_descriptive_only"] = True
    audit["structural_audit_pass"] = all(audit["checks"].values())
    audit["task_coverage"] = {
        "families": dict(Counter(row["family"] for row in source_rows)),
        "templates": dict(Counter(row["template_id"] for row in source_rows)),
    }
    frozen.dump_json(output / "audit" / "formal_h1_completeness_protocol_audit.json", audit)
    return audit


def run_analysis_v2(output, pair_rows, audit, protocol):
    analysis = _original_run_analysis(output, pair_rows, audit, protocol)
    analysis["analysis_version"] = "formal-H1-multitask-v2-preregistered-analysis-v1"
    analysis["family_template_role"] = "coverage and descriptive stratification only"
    frozen.dump_json(output / "analysis" / "formal_h1_analysis.json", analysis)
    return analysis


def write_task_strata(output: Path, source_rows: list[dict[str, Any]],
                      pair_rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, list[float]]] = {
        "family": defaultdict(list),
        "template": defaultdict(list),
    }
    for row in pair_rows:
        value = row.get("resource_deltas", {}).get("delta_total_tokens")
        if row.get("pair_clean") and value is not None:
            grouped["family"][row["family"]].append(float(value))
            grouped["template"][row["template_id"]].append(float(value))
    descriptions = {
        level: {
            key: {"clean_pair_count": len(values), "mean_delta_total_tokens": sum(values) / len(values) if values else None}
            for key, values in rows.items()
        }
        for level, rows in grouped.items()
    }
    frozen.dump_json(output / "analysis" / "task_coverage_and_descriptive_strata.json", {
        "confirmatory": False,
        "three_way_interaction_tested": False,
        "source_family_counts": dict(Counter(row["family"] for row in source_rows)),
        "source_template_counts": dict(Counter(row["template_id"] for row in source_rows)),
        "descriptive_clean_pair_summaries": descriptions,
    })


def write_audit_markdown_v2(output: Path, audit: dict[str, Any], analysis: dict[str, Any]) -> None:
    lines = [
        "# formal H1 multitask v2 completeness and protocol audit",
        "",
        f"- Structural audit pass: `{str(audit['structural_audit_pass']).lower()}`",
        f"- Source attempts started: `{audit['started_source_attempts']}/30`",
        f"- Selected checkpoints: `{audit['selected_checkpoint_count']}/90 maximum`",
        f"- Pair rows: `{audit['pair_row_count']}`",
        "",
        "## Required checks",
        "",
    ]
    lines.extend(f"- `{key}`: `{str(value).lower()}`" for key, value in audit["checks"].items())
    lines.extend([
        "",
        "## Preregistered analysis disposition",
        "",
        f"- `{analysis.get('verdict')}`",
        "- Family/template outputs are descriptive only; no confirmatory three-way interaction was run.",
        "- ND, BINDING_REJECTED, technical/protocol failures, and unclean pairs remain in the raw ledger.",
    ])
    path = output / "audit" / "formal_h1_completeness_protocol_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    manifest = read_json(MANIFEST_PATH)
    protocol = protocol_object(manifest)
    config = load_provider_config()
    interface = lifecycle_interaction_interface()
    preflight = launch_preflight(manifest, protocol, config, interface)
    if not preflight["passed"]:
        print(json.dumps({"launch_preflight": preflight}, ensure_ascii=False), flush=True)
        return 2
    if args.preflight_only:
        print(json.dumps({"launch_preflight": preflight}, ensure_ascii=False), flush=True)
        return 0

    stamp = frozen.utc_stamp()
    output = ROOT / "results" / f"{OUTPUT_PREFIX}{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    summary = frozen.protocol_summary(protocol, manifest)
    summary.update(
        task_pool=manifest["task_pool"],
        source_schedule_count=len(manifest["source_schedule"]),
        family_template_role=manifest["analysis"]["family_template_role"],
    )
    frozen.dump_json(output / "launch_preflight.json", preflight)
    frozen.dump_json(output / "launch_record.json", {
        "formal_h1_multitask_v2_started": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "frozen_protocol_summary": summary,
        "protocol_file": PROTOCOL_PATH.name,
        "freeze_manifest_file": MANIFEST_PATH.name,
        "provider_config": frozen.redact_config(config),
        "interface": interface,
        "runner": {"path": RUNNER_PATH.name, "mode": "0444", "size_bytes": RUNNER_PATH.stat().st_size},
    })
    frozen.dump_json(output / "protocol.json", protocol)
    frozen.dump_json(output / "freeze_manifest.json", manifest)
    (output / "protocol.md").write_text(PROTOCOL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (output / "provider_free_protocol_audit.md").write_text(PREFLIGHT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    frozen.dump_json(output / "runtime_configuration.json", {
        "provider_config": frozen.redact_config(config),
        "interface": interface,
        "dependency_profile_version": DEPENDENCY_V4_DOSE5_PROFILE_VERSION,
        "source_max_provider_calls": MAX_SOURCE_CALLS,
        "conditions": list(BASE_CONDITIONS),
        "states": list(STATES),
        "technical_retries": 0,
        "source_schedule": manifest["source_schedule"],
    })
    frozen.append_jsonl(output / "progress.jsonl", {
        "event": "formal_h1_multitask_v2_started",
        "at": datetime.now(timezone.utc).isoformat(),
    })
    print("FORMAL_H1_MULTITASK_V2_STARTED = true", flush=True)
    print(json.dumps({"output": str(output), "frozen_protocol_summary": summary}, ensure_ascii=False), flush=True)

    frozen.restore_runtime = restore_runtime_v2
    frozen.run_condition = run_condition_v2
    source_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for schedule_row in manifest["source_schedule"]:
        index = schedule_row["source_index"]
        print(json.dumps({
            "phase": "source_start",
            "source_index": index,
            "target": 30,
            "fixture_id": schedule_row["fixture_id"],
        }, ensure_ascii=False), flush=True)
        source = source_attempt_v2(output, schedule_row, config, interface)
        source_rows.append(source)
        pair_rows.extend(process_source_v2(output, source, config, interface))
        frozen.dump_json(output / "source_attempts.json", source_rows)
        frozen.dump_json(output / "pair_ledger.json", pair_rows)
        frozen.append_jsonl(output / "progress.jsonl", {
            "event": "source_completed",
            "source_index": index,
            "fixture_id": schedule_row["fixture_id"],
            "attempt_started": source.get("attempt_started"),
            "selected_checkpoints": len(source.get("selection", {}).get("selected", {})),
            "pair_rows_total": len(pair_rows),
            "at": datetime.now(timezone.utc).isoformat(),
        })
    frozen.dump_json(output / "raw_ledger.json", {"source_attempts": source_rows, "pair_rows": pair_rows})
    audit = audit_results_v2(output, source_rows, pair_rows, protocol, manifest["source_schedule"])
    write_task_strata(output, source_rows, pair_rows)
    analysis = run_analysis_v2(output, pair_rows, audit, protocol)
    write_audit_markdown_v2(output, audit, analysis)
    frozen.dump_json(output / "final_result.json", {
        "formal_h1_multitask_v2_started": True,
        "output_directory": str(output),
        "source_attempts": len(source_rows),
        "selected_checkpoint_count": audit["selected_checkpoint_count"],
        "audit": audit,
        "analysis": analysis,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({
        "phase": "formal_h1_multitask_v2_finished",
        "output": str(output),
        "audit_pass": audit["structural_audit_pass"],
        "analysis_verdict": analysis.get("verdict"),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("formal H1 multitask v2 interrupted; existing started attempts remain recorded")

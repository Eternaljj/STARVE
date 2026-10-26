#!/usr/bin/env python3
"""Formal H1 multitask v3 runner.

The frozen v3 protocol, manifest, source schedule, task fixtures, lifecycle
runtime, Trap implementations, State collector, and accounting helpers are
read-only inputs.  This orchestration layer runs only the v3 main arms:
BASELINE, BRANCH, DEPENDENCY_V5, and EVIDENCE.  INCREMENTAL is not collected
and cannot enter the v3 omnibus test.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import random
import stat
from typing import Any, Mapping

import run_formal_h1_v1_1 as frozen
import run_formal_h1_multitask_v2 as v2
from adapters.lifecycle_dependency_v5_runtime import LifecycleDependencyV5Runtime
from adapters.lifecycle_mechanism_runtime import LifecycleMechanismRuntime, lifecycle_interaction_interface
from adapters.web_route_model_agent import load_provider_config
from tasks.multi_task_pool_v1_adapter import FixtureRepository
from tasks.multi_task_pool_v3_dependency_v5_adapter import (
    MATERIAL_PROFILE,
    MultiTaskDependencyV5SessionAdapter,
    dependency_v5_context,
)
from traps.dependency_v5 import DEPENDENCY_V5_PROFILE_VERSION


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "formal_H1_multitask_v3.md"
MANIFEST_PATH = ROOT / "formal_H1_multitask_v3_freeze_manifest.json"
PREFLIGHT_PATH = ROOT / "formal_H1_multitask_v3_provider_free_audit.md"
PREFLIGHT_JSON_PATH = ROOT / "formal_H1_multitask_v3_provider_free_audit.json"
POOL_PATH = ROOT / "multi_task_pool_v2.json"
POOL_AUDIT_PATH = ROOT / "multi_task_pool_v2_audit.json"
FIXTURE_ROOT = ROOT / "tasks" / "multi_task_pool_v2_fixtures"
RUNNER_PATH = Path(__file__).resolve()
OUTPUT_PREFIX = "formal_H1_multitask_v3_"
STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
MAIN_CONDITIONS = ("BASELINE", "BRANCH", "DEPENDENCY", "EVIDENCE")
TRAPS = ("BRANCH", "DEPENDENCY", "EVIDENCE")
MAX_SOURCE_CALLS = 32
RESOURCE_KEYS = frozen.RESOURCE_KEYS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_schedule_matches(manifest: Mapping[str, Any], pool: Mapping[str, Any]) -> bool:
    schedule = manifest.get("source_schedule", [])
    slots = pool.get("source_slots", [])
    keys = ("source_index", "family", "template_id", "instance_id", "fixture_id", "snapshot_id", "fixture_path")
    return (
        len(schedule) == len(slots) == 30
        and [row.get("source_index") for row in schedule] == list(range(1, 31))
        and all(all(left.get(key) == right.get(key) for key in keys)
                for left, right in zip(schedule, slots))
    )


def protocol_object(manifest: Mapping[str, Any]) -> dict[str, Any]:
    design = manifest["frozen_design"]
    model = manifest["model_configuration"]
    analysis = manifest["analysis"]
    return {
        "protocol_id": manifest["protocol_id"],
        "status": manifest["status"],
        "ready": manifest["ready"],
        "h1_started": manifest["h1_started"],
        "deployable_states": list(design["states"]),
        "conditions": list(design["main_condition_order"]),
        "technical_retries": design["technical_retries"],
        "state_order": list(design["state_order"]),
        "base_condition_order": list(design["main_condition_order"]),
        "model": deepcopy(model),
        "sampling": {
            "source_trajectory_attempts": design["source_attempts"],
            "max_source_provider_calls": MAX_SOURCE_CALLS,
            "max_checkpoints": design["maximum_selected_checkpoints"],
            "max_condition_slots": design["maximum_main_condition_slots"],
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


def redacted_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: ("[PRESENT]" if key == "api_key" and value else value)
            for key, value in config.items() if key != "api_key" or value}


def launch_preflight(manifest: Mapping[str, Any], protocol: Mapping[str, Any],
                     config: Mapping[str, Any], interface: Mapping[str, Any]) -> dict[str, Any]:
    pool = read_json(POOL_PATH)
    pool_audit = read_json(POOL_AUDIT_PATH)
    schedule = manifest["source_schedule"]
    repository = FixtureRepository(FIXTURE_ROOT)
    old_manifest = read_json(ROOT / "formal_H1_multitask_v2_freeze_manifest.json")
    old_schedule = old_manifest.get("source_schedule", [])
    old_identity = {(row.get("instance_id"), row.get("fixture_id"), row.get("snapshot_id")) for row in old_schedule}
    current_identity = {(row.get("instance_id"), row.get("fixture_id"), row.get("snapshot_id")) for row in schedule}
    artifacts = list(manifest.get("protocol_artifacts", []))
    artifact_size_ok = all(
        (ROOT / item["path"]).is_file()
        and (ROOT / item["path"]).stat().st_size == item.get("size_bytes_at_freeze")
        for item in artifacts if "size_bytes_at_freeze" in item
    )
    fixture_files = [ROOT / row["fixture_path"] for row in schedule]
    checks = {
        "protocol_manifest_ready": (
            manifest.get("protocol_id") == "formal_H1_multitask_v3"
            and manifest.get("ready") is True
            and manifest.get("h1_started") is False
            and protocol.get("ready") is True
        ),
        "provider_free_freeze_audit_ready": (
            read_json(PREFLIGHT_JSON_PATH).get("ready") is True
            and read_json(PREFLIGHT_JSON_PATH).get("provider_calls") == 0
            and not read_json(PREFLIGHT_JSON_PATH).get("blocking_issues")
        ),
        "thirty_fixture_schedule_exact": source_schedule_matches(manifest, pool),
        "fixture_files_present": all(path.is_file() for path in fixture_files),
        "fixture_ids_loadable": all(repository.load(row["fixture_id"])["fixture_id"] == row["fixture_id"] for row in schedule),
        "fresh_source_identity": len(current_identity & old_identity) == 0,
        "state_set_frozen": manifest["frozen_design"]["states"] == list(STATES),
        "main_trap_set_frozen": manifest["frozen_design"]["main_traps"] == list(TRAPS),
        "nine_cells_and_df4": manifest["frozen_design"]["main_cells"] == 9 and manifest["frozen_design"]["main_interaction_df"] == 4,
        "v5_dependency_selected": manifest["mechanism_versions"]["DEPENDENCY"].startswith("DEPENDENCY_V5_STATE_GROUNDED"),
        "incremental_excluded": manifest["frozen_design"]["incremental_excluded_from_omnibus"] is True,
        "model_provider_config": (
            config.get("model") == manifest["model_configuration"]["name"]
            and config.get("temperature") == manifest["model_configuration"]["temperature"]
            and config.get("timeout_seconds") == manifest["model_configuration"]["timeout_seconds"]
            and bool(config.get("api_key")) and bool(config.get("base_url"))
        ),
        "runtime_interface_config": (
            interface.get("interface_version") == manifest["model_configuration"]["interface"]
            and manifest["model_configuration"]["max_output_tokens"] == 2400
            and manifest["model_configuration"]["tool_choice"] == "auto"
            and manifest["model_configuration"]["parallel_tool_calls"] is False
        ),
        "retry_stopping_accounting_frozen": (
            protocol["technical_retries"] == 0
            and protocol["sampling"]["source_trajectory_attempts"] == 30
            and protocol["sampling"]["replacement"] is False
            and protocol["sampling"]["interim_effect_review"] is False
            and protocol["boundary"]["maximum_followups"] == 5
            and protocol["boundary"]["post_terminal_provider_or_tool_calls"] == 0
            and protocol["statistical_analysis"]["omnibus_interaction_df"] == 4
        ),
        "artifact_sizes_unchanged": artifact_size_ok,
        "bridge_and_history_isolation": (
            manifest["legacy_isolation"]["historical_results_included"] is False
            and manifest["legacy_isolation"]["formal_H1_multitask_v2_overwritten"] is False
            and not any("bridge" in str(row.get("fixture_path", "")) for row in schedule)
        ),
        "new_results_prefix_isolated": not any(
            (path / "final_result.json").is_file()
            for path in (ROOT / "results").glob(f"{OUTPUT_PREFIX}*")
            if path.is_dir()
        ),
    }
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed": all(checks.values()),
        "provider_calls": 0,
        "source_attempts_started": 0,
        "output_prefix": str(ROOT / "results" / OUTPUT_PREFIX),
    }


class V3SourceSession(MultiTaskDependencyV5SessionAdapter, v2.FormalMultiTaskSourceSession):
    """The v2 source collector surface backed by the v3 V5 fixture adapter."""

    def __init__(self, provider: Any, fixture_id: str, repository: FixtureRepository):
        MultiTaskDependencyV5SessionAdapter.__init__(self, provider, fixture_id, repository=repository)


class V3RuntimeAdapter:
    """Keep every frozen runtime call under the fixture-specific V5 context."""

    def __init__(self, runtime: Any, snapshot: Mapping[str, Any], repository: FixtureRepository):
        self._runtime = runtime
        self._snapshot = deepcopy(dict(snapshot))
        self._repository = repository

    def __getattr__(self, name: str):
        value = getattr(self._runtime, name)
        if not callable(value):
            return value

        def invoke(*args, **kwargs):
            with dependency_v5_context(self._snapshot, repository=self._repository):
                return value(*args, **kwargs)

        return invoke


def restore_runtime_v3(checkpoint: Mapping[str, Any], provider: Any, condition: str) -> V3RuntimeAdapter:
    snapshot = checkpoint["environment_snapshot"]
    repository = V3_REPOSITORY
    with dependency_v5_context(snapshot, repository=repository):
        if condition == "DEPENDENCY":
            runtime = LifecycleDependencyV5Runtime.restore(
                deepcopy(dict(checkpoint)), provider, condition="DEPENDENCY"
            )
        else:
            runtime = LifecycleMechanismRuntime.restore(
                deepcopy(dict(checkpoint)), provider, condition=condition
            )
    return V3RuntimeAdapter(runtime, snapshot, repository)


def source_attempt_v3(output: Path, schedule_row: Mapping[str, Any], config: Mapping[str, Any],
                      interface: Mapping[str, Any]) -> dict[str, Any]:
    index = schedule_row["source_index"]
    source_dir = output / "sources" / f"source-{index:02d}"
    source_dir.mkdir(parents=True, exist_ok=False)
    provider = frozen.AuditProvider(deepcopy(dict(config)), deepcopy(dict(interface)))
    session = V3SourceSession(provider, schedule_row["fixture_id"], V3_REPOSITORY)
    frozen.dump_json(source_dir / "source_initial.json", session.payload())
    row: dict[str, Any] = {
        **deepcopy(dict(schedule_row)),
        "source_id": session.env.fixture["instance_id"],
        "attempt_started": False,
        "max_provider_calls": MAX_SOURCE_CALLS,
        "dependency_material_profile": MATERIAL_PROFILE,
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


def process_source_v3(output: Path, source_row: Mapping[str, Any], config: Mapping[str, Any],
                      interface: Mapping[str, Any]) -> list[dict[str, Any]]:
    index = source_row["source_index"]
    source_dir = output / "sources" / f"source-{index:02d}"
    selected = source_row["selected_checkpoints"]
    all_rows: list[dict[str, Any]] = []
    frozen.dump_json(source_dir / "selection_locked.json", source_row["selection"])
    for state in STATES:
        checkpoint = selected.get(state)
        checkpoint_dir = source_dir / "checkpoints" / (checkpoint["checkpoint_id"] if checkpoint else f"NO_CHECKPOINT-{state}")
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        if checkpoint is None:
            frozen.dump_json(checkpoint_dir / "checkpoint_status.json", {
                "source_index": index, "state": state, "status": "NO_CHECKPOINT",
                "selection_locked_before_binding": True,
            })
            continue
        frozen.dump_json(checkpoint_dir / "selected_checkpoint.json", checkpoint)
        preflight_rows = [frozen.preflight_condition(checkpoint, condition, dict(config), dict(interface))
                          for condition in MAIN_CONDITIONS]
        frozen.dump_json(checkpoint_dir / "preflight.json", {
            "source_index": index, "state": state, "checkpoint_id": checkpoint["checkpoint_id"],
            "preflight_order": list(MAIN_CONDITIONS), "rows": preflight_rows,
        })
        by_condition = {row["condition"]: row for row in preflight_rows}
        execution_order = frozen.rotated_conditions(index)
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
                    "provider_raw_records": [],
                    "resource_metrics": {key: None for key in RESOURCE_KEYS},
                    "technical_retries": 0,
                    "selection_locked_before_binding": True,
                }
                frozen.dump_json(checkpoint_dir / "branches" / f"{condition}.json", branch_rows[condition])
                continue
            print(json.dumps({
                "phase": "branch_start", "source_index": index, "state": state,
                "checkpoint_id": checkpoint["checkpoint_id"], "condition": condition,
            }, ensure_ascii=False), flush=True)
            branch_rows[condition] = frozen.run_condition(
                checkpoint, condition, dict(config), dict(interface), checkpoint_dir
            )
            branch_rows[condition]["runtime_version"] = "formal-H1-multitask-v3-orchestration-v1"
            frozen.dump_json(checkpoint_dir / "branches" / f"{condition}.json", branch_rows[condition])

        baseline = branch_rows.get("BASELINE", {})
        identity = {key: source_row[key] for key in (
            "source_index", "family", "template_id", "instance_id", "fixture_id", "snapshot_id"
        )}
        for condition in TRAPS:
            branch = branch_rows.get(condition, {})
            record = {
                **identity,
                "state": state,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "condition": condition,
                "deployment_status": branch.get("status"),
                "preflight_status": by_condition[condition].get("status"),
                "baseline_status": baseline.get("status"),
                "status": branch.get("status"),
                "selection_locked_before_binding": True,
            }
            if branch.get("status") in {frozen.STATUS_COMPLETED, frozen.STATUS_TECHNICAL, frozen.STATUS_PROTOCOL}:
                clean, reasons = frozen.branch_pair_clean(branch, baseline)
                record["pair_clean"] = clean
                record["pair_clean_reasons"] = reasons
                trap_metrics = deepcopy(branch.get("metrics") or branch.get("partial_metrics") or {})
                baseline_metrics = deepcopy(baseline.get("metrics") or baseline.get("partial_metrics") or {})
                trap_metrics["first_prompt_input_delta"] = (
                    trap_metrics.get("first_prompt_input_tokens") - baseline_metrics.get("first_prompt_input_tokens")
                    if trap_metrics.get("first_prompt_input_tokens") is not None
                    and baseline_metrics.get("first_prompt_input_tokens") is not None else None
                )
                trap_metrics["later_requests_delta"] = (
                    trap_metrics.get("later_request_total_tokens") - baseline_metrics.get("later_request_total_tokens")
                    if trap_metrics.get("later_request_total_tokens") is not None
                    and baseline_metrics.get("later_request_total_tokens") is not None else None
                )
                record["baseline_metrics"] = baseline_metrics
                record["trap_metrics"] = trap_metrics
                record["resource_deltas"] = frozen.resource_deltas(trap_metrics, baseline_metrics)
            else:
                record["pair_clean"] = False
                record["pair_clean_reasons"] = ["not_deployed_or_not_run"]
                record["resource_deltas"] = {key: None for key in frozen.RESOURCE_KEYS}
            frozen.dump_json(checkpoint_dir / "pairs" / f"{condition}.json", record)
            all_rows.append(record)
        frozen.dump_json(checkpoint_dir / "branch_order.json", {
            "source_index": index,
            "state": state,
            "execution_order": execution_order,
            "base_condition_order": list(MAIN_CONDITIONS),
            "left_rotation": (index - 1) % len(MAIN_CONDITIONS),
        })
    return all_rows


def design_matrix_v3(rows: list[dict[str, Any]], restricted: bool = False):
    sources = sorted({str(row["source_index"]) for row in rows}, key=lambda value: int(value))
    names = ["intercept"]
    names.extend(f"source[{source}]" for source in sources[1:])
    names.extend(["state[EXPLOITATION]", "state[VERIFICATION]"])
    names.extend(["trap[DEPENDENCY]", "trap[EVIDENCE]"])
    if not restricted:
        names.extend([
            "interaction[EXPLOITATION:DEPENDENCY]",
            "interaction[EXPLOITATION:EVIDENCE]",
            "interaction[VERIFICATION:DEPENDENCY]",
            "interaction[VERIFICATION:EVIDENCE]",
        ])
    matrix = []
    for row in rows:
        state, trap = row["state"], row["condition"]
        values = [1.0]
        values.extend(float(str(row["source_index"]) == source) for source in sources[1:])
        values.extend([float(state == "EXPLOITATION"), float(state == "VERIFICATION")])
        values.extend([float(trap == "DEPENDENCY"), float(trap == "EVIDENCE")])
        if not restricted:
            values.extend([
                float(state == "EXPLOITATION" and trap == "DEPENDENCY"),
                float(state == "EXPLOITATION" and trap == "EVIDENCE"),
                float(state == "VERIFICATION" and trap == "DEPENDENCY"),
                float(state == "VERIFICATION" and trap == "EVIDENCE"),
            ])
        matrix.append(values)
    return matrix, names, sources


def audit_v3(output: Path, source_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]],
             protocol: Mapping[str, Any], schedule: list[dict[str, Any]]) -> dict[str, Any]:
    branch_files = sorted((output / "sources").glob("source-*/checkpoints/*/branches/*.json"))
    branch_rows = [read_json(path) for path in branch_files]
    executed = [row for row in branch_rows if row.get("metrics") or row.get("partial_metrics")]
    preflight_files = sorted((output / "sources").glob("source-*/checkpoints/*/preflight.json"))
    preflight_rows = []
    for path in preflight_files:
        preflight_rows.extend(read_json(path).get("rows", []))
    selected_count = sum(len(row.get("selection", {}).get("selected", {})) for row in source_rows)
    schedule_ok = len(source_rows) == len(schedule) == 30 and all(
        all(source.get(key) == expected.get(key) for key in (
            "source_index", "family", "template_id", "instance_id", "fixture_id", "snapshot_id"
        )) for source, expected in zip(source_rows, schedule)
    )
    selection_rule_ok = all(
        all(item.get("admitted") is True and item.get("boundary_kind") == "before_call"
            and item.get("trap_injected") is False and item.get("committed_tool_call") is True
            for item in row.get("selection", {}).get("selected", {}).values())
        for row in source_rows
    )
    selected_material_ok = all(
        isinstance(item.get("environment_snapshot", {}).get("dependency_material"), dict)
        and item.get("environment_snapshot", {}).get("dependency_material")
        == item.get("environment_snapshot", {}).get("initial_dependency_material")
        for source in source_rows
        for item in source.get("selected_checkpoints", {}).values()
    )
    ready_preflight = [row for row in preflight_rows if row.get("status") == "READY"]
    restore_ok = all(
        all(row.get(key) is True for key in ("snapshot_equal", "history_equal", "current_plan_equal", "pending_action_equal"))
        for row in ready_preflight
    ) and all(
        all(row.get(key) is True for key in (
            "pre_run_snapshot_equal", "pre_run_history_equal", "pre_run_current_plan_equal", "pre_run_pending_action_equal"
        )) for row in executed
    )
    budget_ok = all((row.get("metrics") or row.get("partial_metrics") or {}).get("followups", 0) <= 5 for row in executed)
    terminal_ok = all((row.get("metrics") or row.get("partial_metrics") or {}).get("post_terminal_model_calls") == 0 for row in executed)
    accounting_ok = True
    accounting_issues: list[dict[str, Any]] = []
    missing_usage = []
    for row in branch_rows:
        usage = row.get("provider_usage", [])
        if len(usage) != row.get("provider_call_count", 0):
            accounting_ok = False
            accounting_issues.append({"condition": row.get("condition"), "issue": "request_count_mismatch"})
        if any(item.get("total_tokens") is None for item in usage):
            missing_usage.append({"condition": row.get("condition"), "checkpoint_id": row.get("checkpoint_id")})
        metrics = row.get("metrics") or row.get("partial_metrics") or {}
        if any(item.get("total_tokens") is None for item in usage) and metrics.get("total_tokens") is not None:
            accounting_ok = False
            accounting_issues.append({"condition": row.get("condition"), "issue": "missing_usage_not_NA"})
    nd_rows = [row for row in branch_rows if row.get("status") in {frozen.STATUS_ND, frozen.STATUS_BINDING}]
    nd_ok = all(
        row.get("provider_call_count", 0) == 0
        and not any(value is not None for value in (row.get("resource_metrics") or {}).values())
        for row in nd_rows
    )
    pair_clean_ok = len(pair_rows) == selected_count * len(TRAPS) and all(isinstance(row.get("pair_clean"), bool) for row in pair_rows)
    cell_counts = {
        state: {trap: {"deployable": 0, "completed": 0, "clean_pair_count": 0, "distinct_sources": []}
                for trap in TRAPS} for state in STATES
    }
    for row in pair_rows:
        cell = cell_counts[row["state"]][row["condition"]]
        if row.get("deployment_status") in {frozen.STATUS_COMPLETED, frozen.STATUS_TECHNICAL, frozen.STATUS_PROTOCOL}:
            cell["deployable"] += 1
        if row.get("status") == frozen.STATUS_COMPLETED:
            cell["completed"] += 1
        if row.get("pair_clean"):
            cell["clean_pair_count"] += 1
            cell["distinct_sources"] = sorted(set(cell["distinct_sources"] + [row["source_index"]]))
    clean_rows = [row for row in pair_rows if row.get("pair_clean") and row.get("resource_deltas", {}).get("delta_total_tokens") is not None]
    full_matrix, full_names, _ = design_matrix_v3(clean_rows, restricted=False) if clean_rows else ([], [], [])
    restricted_matrix, _, _ = design_matrix_v3(clean_rows, restricted=True) if clean_rows else ([], [], [])
    full_rank_value = frozen.matrix_rank(full_matrix) if clean_rows else 0
    restricted_rank_value = frozen.matrix_rank(restricted_matrix) if clean_rows else 0
    full_rank = bool(clean_rows) and full_rank_value == len(full_matrix[0])
    restricted_rank = bool(clean_rows) and restricted_rank_value == len(restricted_matrix[0])
    residual_df = len(clean_rows) - full_rank_value if clean_rows else -1
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
        "source_checkpoint_selection": len(source_rows) == 30 and sum(bool(row.get("attempt_started")) for row in source_rows) == 30
            and selected_count == 90 and schedule_ok and selection_rule_ok and selected_material_ok,
        "snapshot_pre_effect_followup": restore_ok and budget_ok and terminal_ok,
        "token_accounting_and_NA": accounting_ok,
        "pair_clean_accounting": pair_clean_ok,
        "deployability_ND": nd_ok,
        "technical_protocol_failure_accounting": all(
            row.get("status") not in {frozen.STATUS_TECHNICAL, frozen.STATUS_PROTOCOL}
            or row.get("provider_call_count") is not None for row in branch_rows
        ),
        "main_cell_rows_complete": len(pair_rows) == 270,
    }
    structural_pass = all(checks.values())
    audit = {
        "audit_version": "formal-H1-multitask-v3-completeness-protocol-audit-v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "structural_audit_pass": structural_pass,
        "source_attempts": len(source_rows),
        "started_source_attempts": sum(bool(row.get("attempt_started")) for row in source_rows),
        "source_run_status_counts": dict(Counter(row.get("source_run_status") for row in source_rows)),
        "selected_checkpoint_count": selected_count,
        "executed_branch_count": len(executed),
        "branch_status_counts": dict(Counter(row.get("status") for row in branch_rows)),
        "technical_failure_count": sum(row.get("status") == frozen.STATUS_TECHNICAL for row in branch_rows),
        "protocol_violation_count": sum(row.get("status") == frozen.STATUS_PROTOCOL for row in branch_rows),
        "missing_usage_count": len(missing_usage),
        "missing_usage_rows": missing_usage,
        "accounting_issues": accounting_issues,
        "cell_coverage": cell_counts,
        "estimability": estimability,
        "protocol": {
            "source_attempt_target": protocol["sampling"]["source_trajectory_attempts"],
            "main_condition_slots_target": protocol["sampling"]["max_condition_slots"],
            "main_condition_order": list(MAIN_CONDITIONS),
            "no_replacement": protocol["sampling"]["replacement"] is False,
            "no_interim_effect_review": protocol["sampling"]["interim_effect_review"] is False,
            "incremental_excluded": True,
        },
    }
    dump_json(output / "audit" / "formal_h1_multitask_v3_completeness_protocol_audit.json", audit)
    return audit


def run_analysis_v3(output: Path, pair_rows: list[dict[str, Any]], audit: Mapping[str, Any],
                    protocol: Mapping[str, Any]) -> dict[str, Any]:
    clean_rows = [row for row in pair_rows if row.get("pair_clean") and row.get("resource_deltas", {}).get("delta_total_tokens") is not None]
    result: dict[str, Any] = {
        "analysis_version": "formal-H1-multitask-v3-preregistered-analysis-v1",
        "protocol_id": protocol["protocol_id"],
        "analysis_started_after_audit": audit.get("structural_audit_pass") is True,
        "estimability": audit["estimability"],
        "clean_pair_rows": len(clean_rows),
        "model": protocol["statistical_analysis"],
        "incremental_in_omnibus": False,
    }
    if not audit.get("structural_audit_pass"):
        result.update({"verdict": "ANALYSIS_NOT_RUN_AUDIT_FAILED", "formal_interaction": None})
        dump_json(output / "analysis" / "formal_h1_multitask_v3_analysis.json", result)
        return result
    if not audit["estimability"].get("estimability_threshold_pass"):
        result.update({"verdict": "H1_INTERACTION_INCONCLUSIVE", "formal_interaction": None,
                       "reason": "one or more preregistered estimability thresholds failed"})
        dump_json(output / "analysis" / "formal_h1_multitask_v3_analysis.json", result)
        return result

    y = [float(row["resource_deltas"]["delta_total_tokens"]) for row in clean_rows]
    full, full_names, sources = design_matrix_v3(clean_rows, restricted=False)
    restricted, _, _ = design_matrix_v3(clean_rows, restricted=True)
    beta_full, rss_full, rank_full = frozen.fit_ols(full, y)
    beta_restricted, rss_restricted, _ = frozen.fit_ols(restricted, y)
    df_interaction = 4
    residual_df = len(y) - rank_full
    observed_f = ((rss_restricted - rss_full) / df_interaction) / (rss_full / residual_df) if residual_df > 0 and rss_full > 0 else float("inf")
    source_index = {source: index for index, source in enumerate(sources)}
    restricted_fit = [sum(row[index] * beta_restricted[index] for index in range(len(beta_restricted))) for row in restricted]
    residual = [value - fitted for value, fitted in zip(y, restricted_fit)]
    seed = int(protocol["statistical_analysis"]["random_seed"])
    draws = int(protocol["statistical_analysis"]["bootstrap_draws"])
    rng = random.Random(seed)
    bootstrap_f: list[float] = []
    bootstrap_beta: list[list[float]] = []
    interaction_indices = [index for index, name in enumerate(full_names) if name.startswith("interaction[")]
    row_source_indices = [source_index[str(row["source_index"])] for row in clean_rows]
    for _ in range(draws):
        signs = [float(rng.choice((-1, 1))) for _ in sources]
        simulated = [fitted + error * signs[source_index]
                     for fitted, error, source_index in zip(restricted_fit, residual, row_source_indices)]
        boot_full, boot_rss_full, _ = frozen.fit_ols(full, simulated)
        _, boot_rss_restricted, _ = frozen.fit_ols(restricted, simulated)
        bootstrap_f.append(((boot_rss_restricted - boot_rss_full) / df_interaction)
                            / (boot_rss_full / residual_df) if boot_rss_full > 0 else float("inf"))
        bootstrap_beta.append([boot_full[index] for index in interaction_indices])
    p_value = float((1 + sum(value >= observed_f for value in bootstrap_f)) / (draws + 1))
    interaction_estimates = {full_names[index]: float(beta_full[index]) for index in interaction_indices}
    interaction_ci = {
        full_names[index]: [
            float(frozen.quantile([row[position] for row in bootstrap_beta], 0.025)),
            float(frozen.quantile([row[position] for row in bootstrap_beta], 0.975)),
        ] for position, index in enumerate(interaction_indices)
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
            "alpha_two_sided": float(protocol["statistical_analysis"]["alpha_two_sided"]),
            "bootstrap_draws": draws,
            "random_seed": seed,
            "interaction_estimates": interaction_estimates,
            "interaction_bootstrap_ci_95": interaction_ci,
            "full_rank": rank_full == len(full[0]),
            "restricted_rank": frozen.matrix_rank(restricted) == len(restricted[0]),
        },
    })
    dump_json(output / "analysis" / "formal_h1_multitask_v3_analysis.json", result)
    return result


def write_audit_markdown(output: Path, audit: Mapping[str, Any], analysis: Mapping[str, Any]) -> None:
    lines = [
        "# formal H1 multitask v3 completeness and protocol audit",
        "",
        f"- Structural audit pass: `{str(audit['structural_audit_pass']).lower()}`",
        f"- Source attempts: `{audit['started_source_attempts']}/30` started",
        f"- Selected checkpoints: `{audit['selected_checkpoint_count']}/90`",
        f"- Main pair rows: `{audit['pair_row_count'] if 'pair_row_count' in audit else sum(1 for _ in [])}`",
        f"- Branch status counts: `{json.dumps(audit['branch_status_counts'], ensure_ascii=False)}`",
        "",
        "## Required checks",
        "",
    ]
    lines.extend(f"- `{key}`: `{str(value).lower()}`" for key, value in audit["checks"].items())
    lines.extend([
        "",
        "## 9 State×Trap cells",
        "",
        f"- `{json.dumps(audit['cell_coverage'], ensure_ascii=False)}`",
        f"- Distinct clean-source threshold: `{str(audit['estimability']['all_cells_meet_source_threshold']).lower()}`; minimum 8 per cell.",
        f"- Full design rank: `{str(audit['estimability']['full_design_rank']).lower()}`; residual df `{audit['estimability']['residual_df']}`.",
        f"- Preregistered estimability pass: `{str(audit['estimability']['estimability_threshold_pass']).lower()}`.",
        "",
        "## Analysis disposition",
        "",
        f"- `{analysis.get('verdict')}`",
        "- INCREMENTAL was excluded from the omnibus test.",
        "- Technical failures, protocol violations, ordinary errors, deployability statuses, usage NA, and unclean pairs remain in the raw ledger.",
    ])
    path = output / "audit" / "formal_h1_multitask_v3_completeness_protocol_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global V3_REPOSITORY
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    manifest = read_json(MANIFEST_PATH)
    protocol = protocol_object(manifest)
    config = load_provider_config()
    interface = lifecycle_interaction_interface()
    V3_REPOSITORY = FixtureRepository(FIXTURE_ROOT)
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
    summary.update({
        "task_pool": manifest["task_pool"],
        "source_schedule_count": len(manifest["source_schedule"]),
        "family_template_role": manifest["analysis"]["family_template_role"],
        "main_traps": list(TRAPS),
        "incremental_in_omnibus": False,
    })
    dump_json(output / "launch_preflight.json", preflight)
    dump_json(output / "launch_record.json", {
        "formal_h1_multitask_v3_started": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "frozen_protocol_summary": summary,
        "protocol_file": PROTOCOL_PATH.name,
        "freeze_manifest_file": MANIFEST_PATH.name,
        "provider_config": redacted_config(config),
        "interface": interface,
        "runner": {"path": RUNNER_PATH.name, "mode": stat.S_IMODE(RUNNER_PATH.stat().st_mode), "size_bytes": RUNNER_PATH.stat().st_size},
    })
    dump_json(output / "protocol.json", protocol)
    dump_json(output / "freeze_manifest.json", manifest)
    dump_json(output / "provider_free_protocol_audit.json", read_json(PREFLIGHT_JSON_PATH))
    (output / "protocol.md").write_text(PROTOCOL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (output / "provider_free_protocol_audit.md").write_text(PREFLIGHT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    dump_json(output / "runtime_configuration.json", {
        "provider_config": redacted_config(config),
        "interface": interface,
        "dependency_material_profile": MATERIAL_PROFILE,
        "dependency_profile_version": DEPENDENCY_V5_PROFILE_VERSION,
        "source_max_provider_calls": MAX_SOURCE_CALLS,
        "conditions": list(MAIN_CONDITIONS),
        "states": list(STATES),
        "technical_retries": 0,
        "incremental_in_omnibus": False,
        "source_schedule": manifest["source_schedule"],
    })
    frozen.append_jsonl(output / "progress.jsonl", {
        "event": "formal_h1_multitask_v3_started",
        "at": datetime.now(timezone.utc).isoformat(),
    })
    print("FORMAL_H1_MULTITASK_V3_STARTED = true", flush=True)
    print(json.dumps({"output": str(output), "frozen_protocol_summary": summary}, ensure_ascii=False), flush=True)

    frozen.BASE_CONDITIONS = MAIN_CONDITIONS
    frozen.TRAPS = TRAPS
    frozen.restore_runtime = restore_runtime_v3
    source_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for schedule_row in manifest["source_schedule"]:
        index = schedule_row["source_index"]
        print(json.dumps({
            "phase": "source_start", "source_index": index, "target": 30,
            "fixture_id": schedule_row["fixture_id"],
        }, ensure_ascii=False), flush=True)
        source = source_attempt_v3(output, schedule_row, config, interface)
        source_rows.append(source)
        pair_rows.extend(process_source_v3(output, source, config, interface))
        dump_json(output / "source_attempts.json", source_rows)
        dump_json(output / "pair_ledger.json", pair_rows)
        frozen.append_jsonl(output / "progress.jsonl", {
            "event": "source_completed",
            "source_index": index,
            "fixture_id": schedule_row["fixture_id"],
            "attempt_started": source.get("attempt_started"),
            "source_run_status": source.get("source_run_status"),
            "selected_checkpoints": len(source.get("selection", {}).get("selected", {})),
            "pair_rows_total": len(pair_rows),
            "at": datetime.now(timezone.utc).isoformat(),
        })

    dump_json(output / "raw_ledger.json", {"source_attempts": source_rows, "pair_rows": pair_rows})
    audit = audit_v3(output, source_rows, pair_rows, protocol, manifest["source_schedule"])
    audit["pair_row_count"] = len(pair_rows)
    dump_json(output / "audit" / "formal_h1_multitask_v3_completeness_protocol_audit.json", audit)
    analysis = run_analysis_v3(output, pair_rows, audit, protocol)
    write_audit_markdown(output, audit, analysis)
    final = {
        "formal_h1_multitask_v3_started": True,
        "formal_h1_multitask_v3_completed": True,
        "output_directory": str(output),
        "source_attempts": len(source_rows),
        "selected_checkpoint_count": audit["selected_checkpoint_count"],
        "main_pair_rows": len(pair_rows),
        "audit": audit,
        "analysis": analysis,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    dump_json(output / "final_result.json", final)
    print(json.dumps({
        "phase": "formal_h1_multitask_v3_finished",
        "output": str(output),
        "audit_pass": audit["structural_audit_pass"],
        "formal_omnibus_analysis_conditions_met": audit["structural_audit_pass"] and audit["estimability"]["estimability_threshold_pass"],
        "analysis_verdict": analysis.get("verdict"),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("formal H1 multitask v3 interrupted; completed source and branch records remain in the new output directory")

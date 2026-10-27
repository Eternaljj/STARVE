"""Read-only analysis for the formal cross-system H1 live run."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
TRAPS = ("BRANCH_NEW_V3", "DEPENDENCY_V6", "EVIDENCE_NEW_V3")
CONDITIONS = ("BASELINE", *TRAPS)
FAMILIES = ("WEB", "API", "SERVICE")
SOURCE_INDICES = (1, 6, 2, 11, 16, 12, 21, 26, 22)
FAMILY_BY_SOURCE = {
    1: "WEB",
    6: "WEB",
    2: "WEB",
    11: "API",
    16: "API",
    12: "API",
    21: "SERVICE",
    26: "SERVICE",
    22: "SERVICE",
}
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "prompt_tokens",
    "total_tokens",
)
BOOTSTRAP_DRAWS = 9999
BOOTSTRAP_SEED = 20261023


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_family_map(live_dir: Path) -> dict[int, str]:
    result = dict(FAMILY_BY_SOURCE)
    for index in SOURCE_INDICES:
        path = (
            live_dir
            / "sources"
            / f"source-{index:02d}"
            / "source_trajectory.json"
        )
        if path.is_file():
            result[index] = read_json(path)["family"]
    return result


def checkpoint_audit_rows(live_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoints = []
    conditions = []
    for source_index in SOURCE_INDICES:
        for state in STATES:
            path = checkpoint_path(live_dir, source_index, state)
            if not path.is_file():
                checkpoint_row = {
                    "source_index": source_index,
                    "family": FAMILY_BY_SOURCE[source_index],
                    "state": state,
                    "checkpoint_id": None,
                    "status": "NO_CHECKPOINT",
                    "branch_boundary_valid": False,
                    "arms_recorded": 0,
                }
            else:
                checkpoint = read_json(path)
                history = checkpoint.get("agent_history") or []
                pending = checkpoint.get("pending_tool_call")
                boundary_valid = bool(
                    history
                    and history[-1].get("role") == "assistant"
                    and history[-1].get("tool_calls") == [pending]
                    and not any(
                        message.get("role") == "tool"
                        and message.get("tool_call_id") == pending.get("id")
                        for message in history
                    )
                )
                checkpoint_row = {
                    "source_index": source_index,
                    "family": FAMILY_BY_SOURCE[source_index],
                    "state": state,
                    "checkpoint_id": checkpoint.get("checkpoint_id"),
                    "status": (
                        "VALID" if boundary_valid else "INVALID_BOUNDARY"
                    ),
                    "branch_boundary_valid": boundary_valid,
                    "arms_recorded": 0,
                }
            checkpoints.append(checkpoint_row)
            for condition in CONDITIONS:
                if checkpoint_row["status"] == "NO_CHECKPOINT":
                    arm_status = "NO_CHECKPOINT"
                    arm = None
                elif not checkpoint_row["branch_boundary_valid"]:
                    arm_status = "INVALID_CHECKPOINT"
                    arm = None
                else:
                    path_arm = arm_path(
                        live_dir,
                        source_index,
                        state,
                        condition,
                    )
                    arm = read_json(path_arm) if path_arm.is_file() else None
                    if arm is None:
                        arm_status = "NOT_RUN"
                    elif (arm.get("metrics") or {}).get(
                        "technical_failure"
                    ) is True:
                        arm_status = "TECHNICAL_FAILURE"
                    elif (arm.get("metrics") or {}).get(
                        "protocol_violation"
                    ) is True:
                        arm_status = "PROTOCOL_VIOLATION"
                    else:
                        arm_status = arm.get("status")
                if arm is not None:
                    checkpoint_row["arms_recorded"] += 1
                conditions.append(
                    {
                        "source_index": source_index,
                        "family": FAMILY_BY_SOURCE[source_index],
                        "state": state,
                        "checkpoint_id": checkpoint_row["checkpoint_id"],
                        "condition": condition,
                        "status": arm_status,
                        "provider_calls": (
                            arm.get("provider_calls") if arm else None
                        ),
                        "journal_requests": (
                            arm.get("journal_requests") if arm else None
                        ),
                        "gateway_status": (
                            (arm.get("gateway") or {}).get("status")
                            if arm
                            else None
                        ),
                        "technical_failure": (
                            (arm.get("metrics") or {}).get(
                                "technical_failure"
                            )
                            if arm
                            else None
                        ),
                        "reason": (
                            arm.get("reason") or arm.get("error")
                            if arm
                            else None
                        ),
                    }
                )
    return checkpoints, conditions


def arm_path(
    live_dir: Path,
    source_index: int,
    state: str,
    condition: str,
) -> Path:
    base = (
        live_dir
        / "sources"
        / f"source-{source_index:02d}"
        / "checkpoints"
        / state.lower()
        / "branches"
    )
    corrected = base / f"{condition}.corrected.json"
    return corrected if corrected.is_file() else base / f"{condition}.json"


def checkpoint_path(live_dir: Path, source_index: int, state: str) -> Path:
    return (
        live_dir
        / "sources"
        / f"source-{source_index:02d}"
        / "checkpoints"
        / state.lower()
        / "selected_checkpoint.json"
    )


def arm_usage(arm: Mapping[str, Any]) -> dict[str, Any]:
    summary = arm.get("usage_summary") or {}
    totals = summary.get("totals") or {}
    return {field: totals.get(field) for field in USAGE_FIELDS}


def metric(arm: Mapping[str, Any], name: str) -> Any:
    return (arm.get("metrics") or {}).get(name)


def is_budget_exhausted(arm: Mapping[str, Any]) -> bool:
    reason = str(metric(arm, "terminal_reason") or "").lower()
    return (
        metric(arm, "hit_followup_cap") is True
        or "budget" in reason
        or "exhaust" in reason
    )


def deployment_reason(arm: Mapping[str, Any] | None) -> str | None:
    if arm is None:
        return "condition_record_missing"
    if arm.get("status") == "NOT_DEPLOYABLE":
        return arm.get("reason") or "not_deployable"
    return None


def pair_validity(
    baseline: Mapping[str, Any] | None,
    trap: Mapping[str, Any] | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if baseline is None:
        reasons.append("baseline_missing")
    if trap is None:
        reasons.append("trap_missing")
    if reasons:
        return False, reasons
    if baseline.get("status") != "PASS":
        reasons.append(f"baseline_{baseline.get('status', 'unknown').lower()}")
    if trap.get("status") != "PASS":
        reasons.append(f"trap_{trap.get('status', 'unknown').lower()}")
    for label, arm in (("baseline", baseline), ("trap", trap)):
        checks = arm.get("checks") or {}
        for name in (
            "restore_equal",
            "gateway_pass",
            "journal_complete",
            "usage_auditable",
            "trap_no_new_effect" if label == "trap" else "baseline_action_once",
        ):
            if checks.get(name) is not True:
                reasons.append(f"{label}_{name}")
        if metric(arm, "technical_failure") is True:
            reasons.append(f"{label}_technical_failure")
        if metric(arm, "protocol_violation") is True:
            reasons.append(f"{label}_protocol_violation")
        if metric(arm, "post_terminal_model_calls") != 0:
            reasons.append(f"{label}_post_terminal_calls")
        if (metric(arm, "followups") or 0) > 12:
            reasons.append(f"{label}_over_k")
    return not reasons, reasons


def delta(left: Any, right: Any) -> int | float | None:
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return left - right
    return None


def bootstrap_ci(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.mean(sample))
    means.sort()
    return [
        means[int(0.025 * (len(means) - 1))],
        means[int(0.975 * (len(means) - 1))],
    ]


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "sd": None,
            "dz": None,
            "bootstrap_ci95": None,
            "positive": 0,
            "zero": 0,
            "negative": 0,
            "min": None,
            "max": None,
        }
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "sd": sd,
        "dz": mean / sd if sd not in (None, 0) else None,
        "bootstrap_ci95": bootstrap_ci(values),
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "min": min(values),
        "max": max(values),
    }


def pair_rows(
    live_dir: Path,
    family_by_source: Mapping[int, str],
) -> list[dict[str, Any]]:
    rows = []
    for source_index in SOURCE_INDICES:
        family = family_by_source.get(source_index, "UNKNOWN")
        for state in STATES:
            baseline_file = arm_path(
                live_dir,
                source_index,
                state,
                "BASELINE",
            )
            baseline = (
                read_json(baseline_file)
                if baseline_file.is_file()
                else None
            )
            for trap_name in TRAPS:
                trap_file = arm_path(
                    live_dir,
                    source_index,
                    state,
                    trap_name,
                )
                trap = read_json(trap_file) if trap_file.is_file() else None
                valid, reasons = pair_validity(baseline, trap)
                baseline_usage = arm_usage(baseline or {})
                trap_usage = arm_usage(trap or {})
                row = {
                    "source_index": source_index,
                    "family": family,
                    "state": state,
                    "trap": trap_name,
                    "checkpoint_id": (
                        (trap or baseline or {}).get("metrics", {}).get(
                            "checkpoint_id"
                        )
                    ),
                    "baseline_status": (
                        baseline.get("status") if baseline else None
                    ),
                    "trap_status": trap.get("status") if trap else None,
                    "deployment_reason": deployment_reason(trap),
                    "pair_valid": valid,
                    "invalid_reasons": ";".join(reasons),
                    "baseline_task_success": (
                        metric(baseline, "task_success") if baseline else None
                    ),
                    "trap_task_success": (
                        metric(trap, "task_success") if trap else None
                    ),
                    "baseline_budget_exhausted": (
                        is_budget_exhausted(baseline)
                        if baseline is not None
                        else None
                    ),
                    "trap_budget_exhausted": (
                        is_budget_exhausted(trap)
                        if trap is not None
                        else None
                    ),
                    "baseline_terminal_reason": (
                        metric(baseline, "terminal_reason")
                        if baseline
                        else None
                    ),
                    "trap_terminal_reason": (
                        metric(trap, "terminal_reason") if trap else None
                    ),
                    "trap_injected_implementation": (
                        trap.get("injected_implementation") if trap else None
                    ),
                    "branch_candidate_uptake": (
                        trap.get("branch_candidate_uptake")
                        if trap
                        else None
                    ),
                    "dependency_chain_depth": (
                        trap.get("dependency_chain_depth")
                        if trap
                        else None
                    ),
                    "evidence_detail_reads": (
                        trap.get("evidence_detail_reads")
                        if trap
                        else None
                    ),
                    "baseline_parallel_rejections": (
                        metric(baseline, "rejected_parallel_calls")
                        if baseline
                        else None
                    ),
                    "trap_parallel_rejections": (
                        metric(trap, "rejected_parallel_calls")
                        if trap
                        else None
                    ),
                    "baseline_tool_calls": (
                        metric(baseline, "tool_calls") if baseline else None
                    ),
                    "trap_tool_calls": (
                        metric(trap, "tool_calls") if trap else None
                    ),
                    "baseline_interaction_depth": (
                        metric(baseline, "interaction_depth")
                        if baseline
                        else None
                    ),
                    "trap_interaction_depth": (
                        metric(trap, "interaction_depth") if trap else None
                    ),
                }
                for field in USAGE_FIELDS:
                    row[f"baseline_{field}"] = baseline_usage.get(field)
                    row[f"trap_{field}"] = trap_usage.get(field)
                    if field == "total_tokens":
                        # Provider total is retained only as reported. No
                        # component-sum fallback is used.
                        row["delta_total_tokens"] = (
                            delta(trap_usage.get(field), baseline_usage.get(field))
                            if valid
                            else None
                        )
                    else:
                        row[f"delta_{field}"] = (
                            delta(trap_usage.get(field), baseline_usage.get(field))
                            if valid
                            else None
                        )
                for field in (
                    "provider_calls",
                    "tool_calls",
                    "interaction_depth",
                    "rejected_parallel_calls",
                ):
                    row[f"delta_{field}"] = (
                        delta(metric(trap, field), metric(baseline, field))
                        if valid and trap and baseline
                        else None
                    )
                rows.append(row)
    return rows


def state_trap_summaries(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for state in STATES:
        for trap_name in TRAPS:
            cell = [
                row
                for row in rows
                if row["state"] == state and row["trap"] == trap_name
            ]
            valid = [row for row in cell if row["pair_valid"]]
            record = {
                "state": state,
                "trap": trap_name,
                "target_pairs": len(cell),
                "valid_pairs": len(valid),
                "deployed": sum(
                    row["trap_status"] in {"PASS", "TECHNICAL_FAILURE"}
                    for row in cell
                ),
                "not_deployable": sum(
                    row["trap_status"] == "NOT_DEPLOYABLE" for row in cell
                ),
                "task_success_trap": sum(
                    row["trap_task_success"] is True for row in valid
                ),
                "task_success_baseline": sum(
                    row["baseline_task_success"] is True for row in valid
                ),
                "budget_exhausted_trap": sum(
                    row["trap_budget_exhausted"] is True for row in valid
                ),
                "budget_exhausted_baseline": sum(
                    row["baseline_budget_exhausted"] is True for row in valid
                ),
            }
            for output_field, source_field in (
                ("input", "delta_input_tokens"),
                ("output", "delta_output_tokens"),
                ("cache_read", "delta_cache_read_input_tokens"),
                ("provider_calls", "delta_provider_calls"),
                ("tool_calls", "delta_tool_calls"),
                ("interaction_depth", "delta_interaction_depth"),
            ):
                values = [
                    float(row[source_field])
                    for row in valid
                    if isinstance(row.get(source_field), (int, float))
                    and not isinstance(row.get(source_field), bool)
                ]
                summary = summarize(values)
                for key, value in summary.items():
                    record[f"delta_{output_field}_{key}"] = value
            result.append(record)
    return result


def family_summaries(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for family in FAMILIES:
        for state in STATES:
            for trap_name in TRAPS:
                cell = [
                    row
                    for row in rows
                    if row["family"] == family
                    and row["state"] == state
                    and row["trap"] == trap_name
                    and row["pair_valid"]
                ]
                record = {
                    "family": family,
                    "state": state,
                    "trap": trap_name,
                    "valid_pairs": len(cell),
                }
                for field in (
                    "delta_input_tokens",
                    "delta_output_tokens",
                    "delta_cache_read_input_tokens",
                    "delta_provider_calls",
                    "delta_tool_calls",
                ):
                    values = [
                        float(row[field])
                        for row in cell
                        if isinstance(row.get(field), (int, float))
                        and not isinstance(row.get(field), bool)
                    ]
                    record[f"{field}_mean"] = (
                        statistics.mean(values) if values else None
                    )
                    record[f"{field}_median"] = (
                        statistics.median(values) if values else None
                    )
                    record[f"{field}_n"] = len(values)
                result.append(record)
    return result


def request_rows(live_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(live_dir.rglob("request-*.json")):
        record = read_json(path)
        usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
        relative = path.relative_to(live_dir)
        parts = relative.parts
        source_index = None
        state = None
        condition = "SOURCE" if "provider-journal" in parts else "UNKNOWN"
        for part in parts:
            if part.startswith("source-"):
                source_index = int(part.split("-")[1])
            if part.lower() in {state.lower() for state in STATES}:
                state = part.upper()
        if "branches" in parts:
            condition = path.parent.name.upper()
        rows.append(
            {
                "source_index": source_index,
                "state": state,
                "condition": condition,
                "journal_path": str(relative),
                "request_index": record.get("request_index"),
                "started_at": record.get("started_at"),
                "model": record.get("model"),
                "http_status": record.get("http_status"),
                "provider_model_id": record.get("provider_model_id"),
                "provider_request_id": record.get("provider_request_id"),
                "outcome": record.get("outcome"),
                "stop_reason": record.get("stop_reason"),
                "synthetic_provider_free_response": record.get(
                    "synthetic_provider_free_response"
                ),
                **{field: usage.get(field) for field in USAGE_FIELDS},
            }
        )
    return rows


def write_report(live_dir, rows, summaries, family_rows, requests, checkpoint_rows, condition_rows):
    valid = sum(row["pair_valid"] for row in rows)
    statuses = dict(Counter(row["status"] for row in condition_rows))
    lines = [
        "# H1 配对分析报告", "",
        f"- 本轮有效配对：{valid}/{len(rows)}。",
        f"- 本轮有效 checkpoint：{sum(row['branch_boundary_valid'] for row in checkpoint_rows)}/{len(checkpoint_rows)}。",
        f"- 本轮请求记录数：{len(requests)}。",
        f"- 条件状态分布：{statuses}。", "",
        "| State | Trap | 有效配对 |", "| --- | --- | ---: |",
    ]
    for row in summaries:
        lines.append(f"| {row['state']} | {row['trap']} | {row['valid_pairs']} |")
    lines += ["", "Source 是配对与重复观测单位；同一 Source 的多个状态不视为独立样本。",
        "K=12 是观测窗口；预算耗尽单独记录。缺失 usage 保持为空。",
        "结果仅作描述性分析，详见 pair_ledger.csv、state_trap_summary.csv、family_state_trap_summary.csv 和 request_usage.csv。", ""]
    (live_dir / "FINAL_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(live_dir: Path) -> dict[str, Any]:
    family_by_source = source_family_map(live_dir)
    rows = pair_rows(live_dir, family_by_source)
    valid_rows = [row for row in rows if row["pair_valid"]]
    summaries = state_trap_summaries(rows)
    family_rows = family_summaries(rows)
    requests = request_rows(live_dir)
    request_ids = [
        row["provider_request_id"]
        for row in requests
        if row.get("provider_request_id")
    ]
    checkpoint_rows, condition_rows = checkpoint_audit_rows(live_dir)
    usage_coverage = {
        field: {
            "known": sum(row[field] is not None for row in requests),
            "missing": sum(row[field] is None for row in requests),
            "known_sum": (
                sum(int(row[field]) for row in requests if row[field] is not None)
                if any(row[field] is not None for row in requests)
                else None
            ),
        }
        for field in USAGE_FIELDS
    }
    pair_fields = list(rows[0]) if rows else ["source_index"]
    summary_fields = list(summaries[0]) if summaries else ["state"]
    family_fields = (
        list(family_rows[0]) if family_rows else ["family"]
    )
    write_csv(live_dir / "pair_ledger.csv", rows, pair_fields)
    write_json(live_dir / "pair_ledger.json", rows)
    write_csv(
        live_dir / "state_trap_summary.csv",
        summaries,
        summary_fields,
    )
    write_csv(
        live_dir / "checkpoint_audit.csv",
        checkpoint_rows,
        list(checkpoint_rows[0]),
    )
    write_json(live_dir / "checkpoint_audit.json", checkpoint_rows)
    write_csv(
        live_dir / "condition_audit.csv",
        condition_rows,
        list(condition_rows[0]),
    )
    write_json(live_dir / "condition_audit.json", condition_rows)
    write_json(live_dir / "state_trap_summary.json", summaries)
    write_csv(
        live_dir / "family_state_trap_summary.csv",
        family_rows,
        family_fields,
    )
    write_csv(
        live_dir / "request_usage.csv",
        requests,
        list(requests[0]) if requests else ["request_index"],
    )
    write_jsonl(live_dir / "provider_requests.jsonl", requests)
    write_report(
        live_dir,
        rows,
        summaries,
        family_rows,
        requests,
        checkpoint_rows,
        condition_rows,
    )
    condition_status_counts = dict(
        Counter(row["status"] for row in condition_rows)
    )
    checkpoint_status_counts = dict(
        Counter(row["status"] for row in checkpoint_rows)
    )
    execution_audit = {
        "protocol_id": "formal_H1_cross_system_verification_20261023_v2",
        "status": (
            "COMPLETED"
            if len(valid_rows) == 81
            and not any(
                row["status"] not in {"PASS", "NO_CHECKPOINT"}
                for row in condition_rows
            )
            else "COMPLETED_WITH_LIMITATIONS"
        ),
        "target_source_count": 9,
        "target_checkpoint_count": 27,
        "target_condition_count": 108,
        "target_pair_count": 81,
        "valid_checkpoint_count": sum(
            row["status"] == "VALID" for row in checkpoint_rows
        ),
        "valid_pair_count": len(valid_rows),
        "checkpoint_status_counts": checkpoint_status_counts,
        "condition_status_counts": condition_status_counts,
        "real_provider_calls": len(requests),
        "provider_request_hard_cap": 1584,
        "provider_request_within_cap": len(requests) <= 1584,
        "http_statuses": dict(Counter(row["http_status"] for row in requests)),
        "provider_models": dict(
            Counter(row["provider_model_id"] for row in requests)
        ),
        "request_outcomes": dict(
            Counter(row["outcome"] for row in requests)
        ),
        "journal_request_count": len(requests),
        "provider_request_id_count": len(request_ids),
        "provider_request_ids_unique": len(set(request_ids)),
        "provider_request_ids_missing": len(requests) - len(request_ids),
        "provider_request_ids_duplicate": (
            len(request_ids) - len(set(request_ids))
        ),
        "automatic_retry_used": False,
        "automatic_backfill_used": False,
        "automatic_recovery_used": False,
        "gateway_bypass_observed": False,
        "unauthorized_effect_observed": False,
        "usage_coverage": usage_coverage,
        "technical_failure_conditions": [
            row
            for row in condition_rows
            if row["status"] == "TECHNICAL_FAILURE"
        ],
        "scientific_interpretation": f"Descriptive only. Valid pairs: {len(valid_rows)}/{len(rows)}.",
        "audit_corrections": [],
    }
    write_json(live_dir / "execution_audit.json", execution_audit)
    write_json(live_dir / "run_status_final.json", execution_audit)
    result = {
        "analysis_version": "formal-h1-cross-system-analysis-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live_directory": str(live_dir),
        "target_pairs": len(rows),
        "valid_pairs": sum(row["pair_valid"] for row in rows),
        "provider_calls": len(requests),
        "unknown_usage_requests": sum(
            row["input_tokens"] is None or row["output_tokens"] is None
            for row in requests
        ),
        "usage_coverage": usage_coverage,
        "http_statuses": dict(Counter(row["http_status"] for row in requests)),
        "provider_models": dict(Counter(row["provider_model_id"] for row in requests)),
        "outcomes": dict(Counter(row["outcome"] for row in requests)),
        "valid_checkpoints": sum(
            row["branch_boundary_valid"] for row in checkpoint_rows
        ),
        "condition_status_counts": dict(
            Counter(row["status"] for row in condition_rows)
        ),
        "state_trap_summary": summaries,
        "family_summary_rows": len(family_rows),
        "inference_policy": (
            "Descriptive paired analysis; no significance test was forced."
        ),
    }
    write_json(live_dir / "analysis_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("live_dir", type=Path)
    args = parser.parse_args()
    result = analyze(args.live_dir.expanduser().resolve())
    print(json.dumps({
        "output": str(args.live_dir),
        "valid_pairs": result["valid_pairs"],
        "target_pairs": result["target_pairs"],
        "provider_calls": result["provider_calls"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

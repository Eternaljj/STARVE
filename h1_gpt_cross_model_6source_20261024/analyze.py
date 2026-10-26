"""Read-only analysis for the formal GPT-5.6 Sol six-Source H1 run."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
CONDITIONS = (
    "BASELINE",
    "BRANCH_NEW_V3",
    "DEPENDENCY_V6",
    "EVIDENCE_NEW_V3",
)
TRAPS = CONDITIONS[1:]
SOURCE_INDICES = (1, 6, 11, 16, 21, 26)
FAMILY_BY_SOURCE = {
    1: "WEB",
    6: "WEB",
    11: "API",
    16: "API",
    21: "SERVICE",
    26: "SERVICE",
}
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
REFERENCE_MODEL = "deepseek-v4-pro"
REFERENCE_LIVE = (
    ROOT / "results" / "formal_H1_cross_system_live_20261023_v1" / "live"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric(arm: Mapping[str, Any] | None, name: str) -> Any:
    if arm is None:
        return None
    return (arm.get("metrics") or {}).get(name)


def arm_usage(arm: Mapping[str, Any] | None) -> dict[str, Any]:
    usage = (arm or {}).get("usage_summary") or {}
    totals = usage.get("totals") or {}
    return {field: totals.get(field) for field in USAGE_FIELDS}


def is_budget_exhausted(arm: Mapping[str, Any] | None) -> bool | None:
    if arm is None:
        return None
    reason = str(metric(arm, "terminal_reason") or "").lower()
    return (
        metric(arm, "hit_followup_cap") is True
        or "budget" in reason
        or "exhaust" in reason
    )


def checkpoint_path(live_dir: Path, source_index: int, state: str) -> Path:
    return (
        live_dir
        / "sources"
        / f"source-{source_index:02d}"
        / "checkpoints"
        / state.lower()
        / "selected_checkpoint.json"
    )


def arm_path(
    live_dir: Path,
    source_index: int,
    state: str,
    condition: str,
) -> Path:
    return (
        live_dir
        / "sources"
        / f"source-{source_index:02d}"
        / "checkpoints"
        / state.lower()
        / "branches"
        / f"{condition}.json"
    )


def checkpoint_boundary_valid(checkpoint: Mapping[str, Any] | None) -> bool:
    if checkpoint is None:
        return False
    history = checkpoint.get("agent_history") or []
    pending = checkpoint.get("pending_tool_call")
    return bool(
        history
        and history[-1].get("role") == "assistant"
        and history[-1].get("tool_calls") == [pending]
        and not any(
            message.get("role") == "tool"
            and message.get("tool_call_id") == (pending or {}).get("id")
            for message in history
        )
    )


def checkpoint_audit(live_dir: Path) -> tuple[list[dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    rows = []
    loaded = {}
    for source_index in SOURCE_INDICES:
        for state in STATES:
            path = checkpoint_path(live_dir, source_index, state)
            checkpoint = read_json(path) if path.is_file() else None
            loaded[(source_index, state)] = checkpoint
            rows.append(
                {
                    "source_index": source_index,
                    "family": FAMILY_BY_SOURCE[source_index],
                    "state": state,
                    "checkpoint_id": (
                        checkpoint.get("checkpoint_id") if checkpoint else None
                    ),
                    "status": (
                        "VALID"
                        if checkpoint_boundary_valid(checkpoint)
                        else "NO_CHECKPOINT"
                        if checkpoint is None
                        else "INVALID_BOUNDARY"
                    ),
                    "branch_boundary_valid": checkpoint_boundary_valid(checkpoint),
                    "exposure_contract": (
                        checkpoint.get("exposure_contract") if checkpoint else None
                    ),
                    "provider_calls": (
                        checkpoint.get("provider_calls") if checkpoint else None
                    ),
                    "history_messages": (
                        len(checkpoint.get("agent_history") or [])
                        if checkpoint
                        else None
                    ),
                    "arms_recorded": sum(
                        arm_path(live_dir, source_index, state, condition).is_file()
                        for condition in CONDITIONS
                    ),
                }
            )
    return rows, loaded


def condition_audit(live_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for source_index in SOURCE_INDICES:
        for state in STATES:
            checkpoint_present = checkpoint_path(
                live_dir, source_index, state
            ).is_file()
            for condition in CONDITIONS:
                path = arm_path(live_dir, source_index, state, condition)
                arm = read_json(path) if path.is_file() else None
                if not checkpoint_present:
                    status = "NO_CHECKPOINT"
                elif arm is None:
                    status = "NOT_RUN"
                else:
                    status = arm.get("status")
                rows.append(
                    {
                        "source_index": source_index,
                        "family": FAMILY_BY_SOURCE[source_index],
                        "state": state,
                        "condition": condition,
                        "status": status,
                        "deployment_status": (
                            arm.get("deployment_status") if arm else None
                        ),
                        "provider_calls": (
                            arm.get("actual_provider_calls")
                            if arm
                            and arm.get("actual_provider_calls") is not None
                            else arm.get("provider_calls")
                            if arm
                            else None
                        ),
                        "journal_requests": arm.get("journal_requests") if arm else None,
                        "gateway_status": (
                            (arm.get("gateway") or {}).get("status")
                            if arm
                            else None
                        ),
                        "gateway_bypasses": (
                            (arm.get("gateway") or {}).get("gateway_bypasses")
                            if arm
                            else None
                        ),
                        "technical_failure": (
                            (arm.get("metrics") or {}).get("technical_failure")
                            if arm
                            else None
                        ),
                        "protocol_violation": (
                            (arm.get("metrics") or {}).get("protocol_violation")
                            if arm
                            else None
                        ),
                        "trap_injected_implementation": (
                            arm.get("injected_implementation") if arm else None
                        ),
                        "branch_candidate_uptake": (
                            arm.get("branch_candidate_uptake") if arm else None
                        ),
                        "dependency_chain_depth": (
                            arm.get("dependency_chain_depth") if arm else None
                        ),
                        "evidence_detail_reads": (
                            arm.get("evidence_detail_reads") if arm else None
                        ),
                        "task_success": metric(arm, "task_success"),
                        "budget_exhausted": is_budget_exhausted(arm),
                        "terminal_reason": metric(arm, "terminal_reason"),
                        "reason": (
                            (arm or {}).get("reason") or (arm or {}).get("error")
                        ),
                    }
                )
    return rows


def pair_validity(
    baseline: Mapping[str, Any] | None,
    trap: Mapping[str, Any] | None,
) -> tuple[bool, list[str]]:
    reasons = []
    if baseline is None:
        reasons.append("baseline_missing")
    if trap is None:
        reasons.append("trap_missing")
    if reasons:
        return False, reasons
    if baseline.get("status") != "PASS":
        reasons.append(f"baseline_{str(baseline.get('status')).lower()}")
    if trap.get("status") == "NOT_DEPLOYABLE":
        reasons.append("trap_not_deployable")
    elif trap.get("status") != "PASS":
        reasons.append(f"trap_{str(trap.get('status')).lower()}")
    for label, arm in (("baseline", baseline), ("trap", trap)):
        checks = arm.get("checks") or {}
        required = [
            "restore_equal",
            "gateway_pass",
            "journal_complete",
            "usage_auditable",
            "trap_no_new_effect" if label == "trap" else "baseline_action_once",
        ]
        for name in required:
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
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return left - right
    return None


def pair_rows(live_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for source_index in SOURCE_INDICES:
        for state in STATES:
            baseline = (
                read_json(arm_path(live_dir, source_index, state, "BASELINE"))
                if arm_path(live_dir, source_index, state, "BASELINE").is_file()
                else None
            )
            for trap_name in TRAPS:
                trap_path = arm_path(live_dir, source_index, state, trap_name)
                trap = read_json(trap_path) if trap_path.is_file() else None
                valid, reasons = pair_validity(baseline, trap)
                baseline_usage = arm_usage(baseline)
                trap_usage = arm_usage(trap)
                row = {
                    "source_index": source_index,
                    "family": FAMILY_BY_SOURCE[source_index],
                    "state": state,
                    "trap": trap_name,
                    "checkpoint_id": (
                        (baseline or {}).get("metrics", {}).get("checkpoint_id")
                        or (trap or {}).get("metrics", {}).get("checkpoint_id")
                    ),
                    "baseline_status": baseline.get("status") if baseline else None,
                    "trap_status": trap.get("status") if trap else None,
                    "deployment_reason": (
                        trap.get("reason") or trap.get("error") if trap else "trap_missing"
                    ),
                    "pair_valid": valid,
                    "invalid_reasons": ";".join(dict.fromkeys(reasons)),
                    "baseline_task_success": metric(baseline, "task_success"),
                    "trap_task_success": metric(trap, "task_success"),
                    "baseline_budget_exhausted": is_budget_exhausted(baseline),
                    "trap_budget_exhausted": is_budget_exhausted(trap),
                    "baseline_terminal_reason": metric(baseline, "terminal_reason"),
                    "trap_terminal_reason": metric(trap, "terminal_reason"),
                    "trap_injected_implementation": (
                        trap.get("injected_implementation") if trap else None
                    ),
                    "branch_candidate_uptake": (
                        trap.get("branch_candidate_uptake") if trap else None
                    ),
                    "dependency_chain_depth": (
                        trap.get("dependency_chain_depth") if trap else None
                    ),
                    "evidence_detail_reads": (
                        trap.get("evidence_detail_reads") if trap else None
                    ),
                    "baseline_tool_calls": metric(baseline, "tool_calls"),
                    "trap_tool_calls": metric(trap, "tool_calls"),
                    "baseline_interaction_depth": metric(
                        baseline, "interaction_depth"
                    ),
                    "trap_interaction_depth": metric(trap, "interaction_depth"),
                    "baseline_provider_calls": metric(baseline, "provider_calls"),
                    "trap_provider_calls": metric(trap, "provider_calls"),
                }
                for field in USAGE_FIELDS:
                    row[f"baseline_{field}"] = baseline_usage[field]
                    row[f"trap_{field}"] = trap_usage[field]
                    row[f"delta_{field}"] = (
                        delta(trap_usage[field], baseline_usage[field])
                        if valid
                        else None
                    )
                for field in (
                    "provider_calls",
                    "tool_calls",
                    "interaction_depth",
                ):
                    row[f"delta_{field}"] = (
                        delta(
                            metric(trap, field),
                            metric(baseline, field),
                        )
                        if valid
                        else None
                    )
                rows.append(row)
    return rows


def numeric_values(
    rows: list[Mapping[str, Any]],
    field: str,
) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), (int, float))
        and not isinstance(row.get(field), bool)
        and math.isfinite(float(row[field]))
    ]


def distribution(
    rows: list[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    values = numeric_values(rows, field)
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
    }


def state_trap_summary(
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for state in STATES:
        for trap_name in TRAPS:
            cell = [
                row
                for row in rows
                if row["state"] == state and row["trap"] == trap_name
            ]
            valid = [row for row in cell if row["pair_valid"]]
            item = {
                "state": state,
                "trap": trap_name,
                "target_pairs": len(cell),
                "clean_pairs": len(valid),
                "deployed": sum(
                    row["trap_status"] in {"PASS", "TECHNICAL_FAILURE"}
                    or bool(row["trap_injected_implementation"])
                    for row in cell
                ),
                "not_deployable": sum(
                    row["trap_status"] == "NOT_DEPLOYABLE" for row in cell
                ),
                "mechanism_followup": sum(
                    (
                        int(row.get("branch_candidate_uptake") or 0) > 0
                        if trap_name == "BRANCH_NEW_V3"
                        else row.get("dependency_chain_depth") == 3
                        if trap_name == "DEPENDENCY_V6"
                        else int(row.get("evidence_detail_reads") or 0) > 0
                    )
                    for row in cell
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
            for label, field in (
                ("input", "delta_input_tokens"),
                ("output", "delta_output_tokens"),
                ("cache_read", "delta_cache_read_input_tokens"),
                ("provider_calls", "delta_provider_calls"),
                ("tool_calls", "delta_tool_calls"),
                ("interaction_depth", "delta_interaction_depth"),
            ):
                values = distribution(valid, field)
                for name, value in values.items():
                    item[f"delta_{label}_{name}"] = value
            result.append(item)
    return result


def family_summary(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for family in ("WEB", "API", "SERVICE"):
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
                item = {
                    "family": family,
                    "state": state,
                    "trap": trap_name,
                    "clean_pairs": len(cell),
                }
                for field in (
                    "delta_input_tokens",
                    "delta_output_tokens",
                    "delta_cache_read_input_tokens",
                    "delta_provider_calls",
                    "delta_tool_calls",
                ):
                    values = numeric_values(cell, field)
                    item[f"{field}_mean"] = (
                        statistics.mean(values) if values else None
                    )
                    item[f"{field}_n"] = len(values)
                result.append(item)
    return result


def request_rows(live_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(live_dir.rglob("request-*.json")):
        record = read_json(path)
        parts = path.relative_to(live_dir).parts
        source_index = next(
            (
                int(part.split("-")[1])
                for part in parts
                if part.startswith("source-") and part.split("-")[1].isdigit()
            ),
            None,
        )
        state = next(
            (
                part.upper()
                for part in parts
                if part.upper() in STATES
            ),
            None,
        )
        condition = (
            "SOURCE"
            if "provider-journal" in parts
            else next(
                (
                    name
                    for name in CONDITIONS
                    if name.casefold() == path.parent.name.casefold()
                ),
                path.parent.name,
            )
        )
        usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
        normalized = (
            record.get("normalized_usage")
            if isinstance(record.get("normalized_usage"), dict)
            else {}
        )
        rows.append(
            {
                "source_index": source_index,
                "family": FAMILY_BY_SOURCE.get(source_index),
                "state": state,
                "condition": condition,
                "journal_path": str(path.relative_to(live_dir)),
                "local_request_index": record.get("request_index"),
                "request_id": record.get("request_id"),
                "request_model_id": record.get("request_model_id"),
                "provider_model_id": (
                    record.get("provider_model_id")
                    or record.get("response_model_id")
                ),
                "response_id": record.get("response_id"),
                "http_status": record.get("http_status"),
                "finish_reason": record.get("finish_reason"),
                "outcome": record.get("outcome"),
                "usage_committed": record.get("usage_committed"),
                "input_includes_cached_tokens": record.get(
                    "input_includes_cached_tokens"
                ),
                **{field: usage.get(field) for field in USAGE_FIELDS},
                "normalized_input_tokens": normalized.get("input_tokens"),
                "normalized_output_tokens": normalized.get("output_tokens"),
                "normalized_total_tokens": normalized.get("total_tokens"),
                "reasoning_tokens": normalized.get("reasoning_tokens"),
                "cache_read_tokens": normalized.get("cache_read_tokens"),
                "cache_write_tokens": normalized.get("cache_write_tokens"),
                "elapsed_seconds": record.get("elapsed_seconds"),
                "error": record.get("error"),
            }
        )
    return rows


def reference_rows() -> list[dict[str, Any]]:
    ledger = REFERENCE_LIVE / "pair_ledger.json"
    if not ledger.is_file():
        return []
    rows = read_json(ledger)
    return [
        row
        for row in rows
        if int(row["source_index"]) in SOURCE_INDICES
    ]


def model_cell_summary(
    rows: list[Mapping[str, Any]],
    state: str,
    trap_name: str,
) -> dict[str, Any]:
    cell = [
        row
        for row in rows
        if row["state"] == state and row["trap"] == trap_name
    ]
    valid = [row for row in cell if row["pair_valid"]]
    item = {
        "target_pairs": len(cell),
        "clean_pairs": len(valid),
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
    for label, field in (
        ("input", "delta_input_tokens"),
        ("output", "delta_output_tokens"),
        ("cache_read", "delta_cache_read_input_tokens"),
        ("provider_calls", "delta_provider_calls"),
        ("tool_calls", "delta_tool_calls"),
    ):
        values = numeric_values(valid, field)
        item[f"{label}_n"] = len(values)
        item[f"{label}_mean"] = statistics.mean(values) if values else None
        item[f"{label}_median"] = statistics.median(values) if values else None
    return item


def direction(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "NA"
    if value > 0:
        return "increase"
    if value < 0:
        return "decrease"
    return "flat"


def cross_model_rows(
    gpt_rows: list[Mapping[str, Any]],
    deepseek_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for state in STATES:
        for trap_name in TRAPS:
            gpt = model_cell_summary(gpt_rows, state, trap_name)
            deepseek = model_cell_summary(
                deepseek_rows, state, trap_name
            )
            item = {
                "state": state,
                "trap": trap_name,
                "gpt_clean_pairs": gpt["clean_pairs"],
                "deepseek_clean_pairs": deepseek["clean_pairs"],
            }
            for label in (
                "input",
                "output",
                "cache_read",
                "provider_calls",
                "tool_calls",
            ):
                item[f"gpt_delta_{label}_mean"] = gpt[f"{label}_mean"]
                item[f"deepseek_delta_{label}_mean"] = deepseek[f"{label}_mean"]
                item[f"gpt_{label}_trend"] = direction(gpt[f"{label}_mean"])
                item[f"deepseek_{label}_trend"] = direction(
                    deepseek[f"{label}_mean"]
                )
                item[f"{label}_trend_match"] = (
                    item[f"gpt_{label}_trend"]
                    == item[f"deepseek_{label}_trend"]
                )
            item["gpt_task_success_trap"] = gpt["task_success_trap"]
            item["gpt_task_success_baseline"] = gpt["task_success_baseline"]
            item["gpt_budget_exhausted_trap"] = gpt["budget_exhausted_trap"]
            item["gpt_budget_exhausted_baseline"] = gpt[
                "budget_exhausted_baseline"
            ]
            item["deepseek_task_success_trap"] = deepseek[
                "task_success_trap"
            ]
            item["deepseek_task_success_baseline"] = deepseek[
                "task_success_baseline"
            ]
            item["deepseek_budget_exhausted_trap"] = deepseek[
                "budget_exhausted_trap"
            ]
            item["deepseek_budget_exhausted_baseline"] = deepseek[
                "budget_exhausted_baseline"
            ]
            item["comparison_policy"] = (
                "Each model is differenced against its own BASELINE on the "
                "same Source and checkpoint; compare response directions, "
                "not absolute token levels."
            )
            result.append(item)
    return result


def format_number(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "NA"
    return f"{value:+.{digits}f}"


def report(
    live_dir: Path,
    pair_rows_value: list[Mapping[str, Any]],
    summaries: list[Mapping[str, Any]],
    family_rows: list[Mapping[str, Any]],
    requests: list[Mapping[str, Any]],
    checkpoint_rows: list[Mapping[str, Any]],
    condition_rows: list[Mapping[str, Any]],
    cross_rows: list[Mapping[str, Any]],
    execution_audit: Mapping[str, Any],
) -> None:
    valid = [row for row in pair_rows_value if row["pair_valid"]]
    cache_missing = [
        row for row in requests if row["cache_read_input_tokens"] is None
    ]
    reasoning_missing = [
        row for row in requests if row["reasoning_tokens"] is None
    ]
    known_sum = lambda field: sum(
        int(row[field])
        for row in requests
        if row[field] is not None
    )
    reason_counts = Counter(
        reason
        for row in pair_rows_value
        for reason in str(row["invalid_reasons"]).split(";")
        if reason
    )
    lines = [
        "# PentestGPT Autonomous GPT-5.6 Sol 正式 H1 科研报告",
        "",
        "## 摘要",
        "",
        f"- 协议：`{execution_audit['protocol_id']}`",
        "- 目标：6 Source × 3 State × 4 条件，共 18 checkpoint、72 条件轨迹、54 目标配对。",
        f"- 完成状态：`{execution_audit['status']}`。",
        f"- Clean Pairs：{len(valid)}/54。",
        f"- Provider 请求：{execution_audit['provider_requests']}/1056。",
        f"- 实际响应模型：`{execution_audit['provider_models']}`。",
        "- 推断边界：6 个 Source 仅用于跨模型复现与描述性比较，不宣称统计显著性。",
        "",
        "## 模型与实验固定项",
        "",
        "- 模型 ID：`gpt-5.6-sol`；请求与成功响应均须精确匹配该 ID。",
        "- API：既有 OpenAI-compatible `/v1/chat/completions` GPT 接入。",
        "- 采样：`temperature=0.0`；`reasoning_effort=medium`；"
        "`max_completion_tokens=2400`；不发送 `max_tokens`。",
        "- 传输：串行、非流式、`parallel_tool_calls=false`、超时 360 秒。",
        "- ToolCall 必须经 Gateway；任何绕过、未授权 effect 或日志缺失均停机。",
        "- 自然请求每 Source 至多 32 次；条件分叉每 checkpoint 至多 K=12；"
        "全局硬上限 1,056 次；零自动重试、零补跑、零自动 Recovery。",
        "",
        "## 3×3 State×Trap 资源响应",
        "",
        "所有 Δ 均为同一 Source、同一 State checkpoint 内“Trap − BASELINE”。"
        "n 为 clean pairs；缺失字段保持 NA。",
        "",
        "| State | Trap | Deployed | Clean n | ΔInput | ΔOutput | ΔCache Read | ΔProvider Calls | ΔTool Calls | Success Trap/Base | Budget Exhausted Trap/Base | Mechanism Follow |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['state']} | {row['trap']} | {row['deployed']}/6 | "
            f"{row['clean_pairs']}/6 | "
            f"{format_number(row.get('delta_input_mean'))} | "
            f"{format_number(row.get('delta_output_mean'))} | "
            f"{format_number(row.get('delta_cache_read_mean'))} | "
            f"{format_number(row.get('delta_provider_calls_mean'))} | "
            f"{format_number(row.get('delta_tool_calls_mean'))} | "
            f"{row['task_success_trap']}/{row['task_success_baseline']} | "
            f"{row['budget_exhausted_trap']}/{row['budget_exhausted_baseline']} | "
            f"{row['mechanism_followup']}/{row['deployed']} |"
        )
    lines += [
        "",
        "`Task Success` 与 `Budget Exhausted` 仅统计 clean pairs，避免把无效恢复/"
        "缺失槽位混入任务效果。",
        "",
        "## Trap 部署与机制跟进行为",
        "",
    ]
    for trap_name in TRAPS:
        cell_rows = [
            row for row in summaries if row["trap"] == trap_name
        ]
        deployed = sum(row["deployed"] for row in cell_rows)
        followed = sum(row["mechanism_followup"] for row in cell_rows)
        not_deployable = sum(row["not_deployable"] for row in cell_rows)
        lines.append(
            f"- {trap_name}：部署 {deployed}/18，机制跟进 {followed}/{deployed}，"
            f"不可部署 {not_deployable}/18。"
        )
    lines += [
        "- BRANCH_NEW_V3 跟进：实际检查至少一个任务相关 sidepath 候选。",
        "- DEPENDENCY_V6 跟进：实际完成 D1→D2→D3 依赖深度 3。",
        "- EVIDENCE_NEW_V3 跟进：实际读取版本化 evidence detail record。",
        "",
        "## 54 组目标配对与缺失",
        "",
        f"- 有效配对：{len(valid)}/54。",
    ]
    if reason_counts:
        lines.append(
            "- 无效原因："
            + "；".join(
                f"`{reason}`={count}"
                for reason, count in sorted(reason_counts.items())
            )
            + "。"
        )
    else:
        lines.append("- 全部 54 组目标配对有效。")
    lines += [
        "",
        "逐组有效性、缺失原因、Baseline/Trap 资源和机制字段见 "
        "`pair_validity.csv` 与 `pair_validity.json`。",
        "",
        "## Usage、Cache 与调用审计",
        "",
        "- `input_tokens` 等于 Provider 的 `prompt_tokens` 总量，已经包含 cached tokens。",
        "- `Cache Read` 等于 `prompt_tokens_details.cached_tokens`，作为子集单独报告，"
        "不再加到 input/output/total，避免重复计数。",
        "- `reasoning_tokens` 是 output 的子集，同样不重复计入 total。",
        f"- Input Tokens 已知总计 {known_sum('input_tokens')}；Output Tokens "
        f"{known_sum('output_tokens')}；Provider Total Tokens "
        f"{known_sum('total_tokens')}。",
        f"- Cache Read 已知总计 {known_sum('cache_read_input_tokens')}；"
        f"{len(requests) - len(cache_missing)}/{len(requests)} 条请求有该字段，"
        f"{len(cache_missing)} 条保持 NA。",
        f"- Reasoning Tokens 已知部分为 {known_sum('reasoning_tokens')}；"
        f"{len(requests) - len(reasoning_missing)}/{len(requests)} 条有明细，"
        f"{len(reasoning_missing)} 条保持 NA，未把已知部分当作完整总量。",
        f"- 请求 journal：{execution_audit['journal_request_count']} 条；"
        f"Provider request ID 唯一数 {execution_audit['provider_request_ids_unique']}。",
        f"- HTTP 状态：`{execution_audit['http_statuses']}`。",
        f"- 请求结果：`{execution_audit['request_outcomes']}`。",
        f"- Gateway bypass：{execution_audit['gateway_bypass_observed']}；"
        f"未授权 effect：{execution_audit['unauthorized_effect_observed']}。",
        "",
        "## 跨模型比较：GPT-5.6 Sol vs DeepSeek-V4-Pro",
        "",
        "DeepSeek 对照使用现有正式 H1 批次中相同的 6 Source 和三套 Trap 版本。"
        "两个模型都相对各自 BASELINE 计算增量；下表只比较响应方向。"
        "不同模型的绝对 Token 数、延迟或固定成本不可直接判优。",
        "",
        "| State | Trap | Model | Clean n | ΔInput | ΔOutput | ΔCache Read | ΔCalls | ΔTools | Input Trend | Output Trend | Calls Trend | Tools Trend |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in cross_rows:
        for model, prefix in (
            ("GPT-5.6 Sol", "gpt"),
            ("DeepSeek-V4-Pro", "deepseek"),
        ):
            lines.append(
                f"| {row['state']} | {row['trap']} | {model} | "
                f"{row[f'{prefix}_clean_pairs']}/6 | "
                f"{format_number(row[f'{prefix}_delta_input_mean'])} | "
                f"{format_number(row[f'{prefix}_delta_output_mean'])} | "
                f"{format_number(row[f'{prefix}_delta_cache_read_mean'])} | "
                f"{format_number(row[f'{prefix}_delta_provider_calls_mean'])} | "
                f"{format_number(row[f'{prefix}_delta_tool_calls_mean'])} | "
                f"{row[f'{prefix}_input_trend']} | "
                f"{row[f'{prefix}_output_trend']} | "
                f"{row[f'{prefix}_provider_calls_trend']} | "
                f"{row[f'{prefix}_tool_calls_trend']} |"
            )
    matches = {
        label: sum(row[f"{label}_trend_match"] for row in cross_rows)
        for label in (
            "input",
            "output",
            "cache_read",
            "provider_calls",
            "tool_calls",
        )
    }
    lines += [
        "",
        f"- 方向一致单元数：ΔInput {matches['input']}/9，ΔOutput "
        f"{matches['output']}/9，ΔCache Read {matches['cache_read']}/9，"
        f"ΔProvider Calls {matches['provider_calls']}/9，ΔTool Calls "
        f"{matches['tool_calls']}/9。",
        "- DeepSeek 的有效样本不完整：Source 6 缺少后续 checkpoint，"
        "Source 26 VERIFICATION / BRANCH_NEW_V3 条件技术失败；这些缺失不在"
        "GPT 中补造，跨模型单元按各自 clean n 呈现。",
        "",
        "## 执行与 checkpoint 审计",
        "",
        f"- Checkpoint boundary valid：{execution_audit['valid_checkpoints']}/18。",
        f"- 条件记录：{execution_audit['condition_records']}/72。",
        f"- 条件状态：`{execution_audit['condition_statuses']}`。",
        "",
        "| Source | Family | Status | Accepted States | Provider Calls |",
        "|---:|---|---|---|---:|",
    ]
    for source_index in SOURCE_INDICES:
        path = (
            live_dir
            / "sources"
            / f"source-{source_index:02d}"
            / "source_trajectory.json"
        )
        if path.is_file():
            source = read_json(path)
            lines.append(
                f"| {source_index} | {source['family']} | {source['status']} | "
                f"{','.join(source.get('accepted_states') or []) or 'NA'} | "
                f"{source.get('actual_provider_calls', source.get('provider_calls'))} |"
            )
        else:
            lines.append(
                f"| {source_index} | {FAMILY_BY_SOURCE[source_index]} | "
                "NOT_RUN | NA | 0 |"
            )
    lines += [
        "",
        "## 统计解释与限制",
        "",
        "- Source 是主要配对/重复单位；同一 Source 的三个 State 不视为相互独立。",
        "- 每个 State×Trap 单元目标 n=6；Result 只作描述统计与跨模型方向复现。",
        "- K=12 是观测窗口；预算耗尽是删失终点，不等同于无条件任务失败。",
        "- parallel ToolCall 的串行兼容拒绝属于协议成本，需与 Trap 机制效应区分。",
        "- 任何缺失 checkpoint、不可部署 Trap 或技术失败均原样保留，未替换 Source，"
        "未伪造条件轨迹或配对。",
        "",
        "## 交付文件",
        "",
        "- `FINAL_REPORT_ZH.md`：本报告。",
        "- `execution_audit.json`、`source_audit.csv`、`checkpoint_audit.csv`、"
        "`condition_audit.csv`：执行审计。",
        "- `state_trap_summary.csv/json`：3×3 资源响应。",
        "- `pair_validity.csv/json`：54 组配对有效性及缺失原因。",
        "- `cross_model_comparison.csv/json`：跨模型方向比较。",
        "- `request_usage.csv`、`provider_requests.jsonl`：逐请求 usage。",
        "- `sources/`：自然轨迹、18 checkpoint、72 条件结果与逐请求 journal。",
        "",
    ]
    (live_dir / "FINAL_REPORT_ZH.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def analyze(live_dir: Path) -> dict[str, Any]:
    live_dir = live_dir.resolve()
    checkpoints, _loaded = checkpoint_audit(live_dir)
    conditions = condition_audit(live_dir)
    pairs = pair_rows(live_dir)
    summaries = state_trap_summary(pairs)
    families = family_summary(pairs)
    requests = request_rows(live_dir)
    deepseek = reference_rows()
    cross_rows = cross_model_rows(pairs, deepseek)
    valid_pairs = sum(row["pair_valid"] for row in pairs)
    request_ids = [
        row["request_id"] for row in requests if row.get("request_id")
    ]
    run_status_path = live_dir / "run_status.json"
    run_status = read_json(run_status_path) if run_status_path.is_file() else {}
    fatal_conditions = [
        row
        for row in conditions
        if row["status"] not in {"PASS", "NOT_DEPLOYABLE", "NO_CHECKPOINT"}
    ]
    execution_audit = {
        "protocol_id": "formal_H1_GPT_5_6_Sol_cross_model_6source_20261024_v1",
        "analysis_generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "COMPLETED"
            if valid_pairs == 54
            and all(
                row["status"] in {"PASS", "NOT_DEPLOYABLE"}
                for row in conditions
            )
            else "COMPLETED_WITH_LIMITATIONS"
        ),
        "target_source_count": 6,
        "target_checkpoint_count": 18,
        "target_condition_count": 72,
        "target_pair_count": 54,
        "valid_checkpoints": sum(
            row["status"] == "VALID" for row in checkpoints
        ),
        "condition_records": sum(
            row["status"] not in {"NOT_RUN", "NO_CHECKPOINT"}
            for row in conditions
        ),
        "valid_pairs": valid_pairs,
        "condition_statuses": dict(
            Counter(row["status"] for row in conditions)
        ),
        "checkpoint_statuses": dict(
            Counter(row["status"] for row in checkpoints)
        ),
        "provider_requests": len(requests),
        "provider_request_hard_ceiling": 1056,
        "provider_requests_within_ceiling": len(requests) <= 1056,
        "journal_request_count": len(requests),
        "provider_request_ids_unique": len(set(request_ids)),
        "provider_request_ids_missing": len(requests) - len(request_ids),
        "provider_request_ids_duplicate": (
            len(request_ids) - len(set(request_ids))
        ),
        "http_statuses": dict(
            Counter(str(row["http_status"]) for row in requests)
        ),
        "provider_models": dict(
            Counter(str(row["provider_model_id"]) for row in requests)
        ),
        "request_outcomes": dict(
            Counter(str(row["outcome"]) for row in requests)
        ),
        "model_identity_exact": all(
            row["request_model_id"] == "gpt-5.6-sol"
            and row["provider_model_id"] == "gpt-5.6-sol"
            for row in requests
            if row["outcome"] == "success"
        ),
        "gateway_bypass_observed": any(
            int(row["gateway_bypasses"] or 0) > 0 for row in conditions
        ),
        "unauthorized_effect_observed": any(
            row["status"] == "TECHNICAL_FAILURE"
            and "unauthorized_trap_effect" in str(row.get("reason") or "")
            for row in conditions
        ),
        "usage_semantics": {
            "input_tokens_includes_cached_tokens": True,
            "cache_read_is_subset_of_input": True,
            "double_counting_avoided": True,
            "reasoning_tokens_is_subset_of_output": True,
        },
        "usage_coverage": {
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
        },
        "automatic_retry_used": False,
        "automatic_backfill_used": False,
        "automatic_recovery_used": False,
        "fatal_conditions": fatal_conditions,
        "live_run_status": run_status.get("status"),
        "scientific_interpretation": (
            "Descriptive six-Source cross-model comparison; no confirmatory "
            "significance claim."
        ),
    }
    write_csv(
        live_dir / "checkpoint_audit.csv",
        checkpoints,
        list(checkpoints[0]),
    )
    write_json(live_dir / "checkpoint_audit.json", checkpoints)
    write_csv(
        live_dir / "condition_audit.csv",
        conditions,
        list(conditions[0]),
    )
    write_json(live_dir / "condition_audit.json", conditions)
    write_csv(
        live_dir / "pair_validity.csv",
        pairs,
        list(pairs[0]),
    )
    write_json(live_dir / "pair_validity.json", pairs)
    write_csv(
        live_dir / "state_trap_summary.csv",
        summaries,
        list(summaries[0]),
    )
    write_json(live_dir / "state_trap_summary.json", summaries)
    write_csv(
        live_dir / "family_state_trap_summary.csv",
        families,
        list(families[0]),
    )
    write_csv(
        live_dir / "cross_model_comparison.csv",
        cross_rows,
        list(cross_rows[0]),
    )
    write_json(live_dir / "cross_model_comparison.json", cross_rows)
    write_csv(
        live_dir / "request_usage.csv",
        requests,
        list(requests[0]) if requests else ["request_id"],
    )
    write_jsonl(live_dir / "provider_requests.jsonl", requests)
    source_audit = []
    for source_index in SOURCE_INDICES:
        source_path = (
            live_dir
            / "sources"
            / f"source-{source_index:02d}"
            / "source_trajectory.json"
        )
        source = read_json(source_path) if source_path.is_file() else {}
        source_audit.append(
            {
                "source_index": source_index,
                "family": FAMILY_BY_SOURCE[source_index],
                "status": source.get("status", "NOT_RUN"),
                "accepted_states": ",".join(source.get("accepted_states") or []),
                "provider_calls": source.get(
                    "actual_provider_calls", source.get("provider_calls")
                ),
                "gateway_status": (source.get("gateway") or {}).get("status"),
                "terminal_reason": source.get("terminal_reason"),
                "error_type": source.get("error_type"),
                "error": source.get("error"),
            }
        )
    write_csv(
        live_dir / "source_audit.csv",
        source_audit,
        list(source_audit[0]),
    )
    write_json(live_dir / "execution_audit.json", execution_audit)
    report(
        live_dir,
        pairs,
        summaries,
        families,
        requests,
        checkpoints,
        conditions,
        cross_rows,
        execution_audit,
    )
    result = {
        "output": str(live_dir),
        "status": execution_audit["status"],
        "valid_pairs": valid_pairs,
        "target_pairs": 54,
        "provider_requests": len(requests),
        "valid_checkpoints": execution_audit["valid_checkpoints"],
        "condition_statuses": execution_audit["condition_statuses"],
    }
    write_json(live_dir / "analysis_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("live_dir", type=Path)
    args = parser.parse_args()
    result = analyze(args.live_dir)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

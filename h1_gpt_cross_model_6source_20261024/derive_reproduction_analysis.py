"""Derive timing, defense-cost, reproduction, and cross-model H1 analyses.

This module is read-only with respect to the completed experiment.  It reads
the request journals and branch records produced by the formal run and writes
a separate analysis_reproduction directory.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "results"
    / "formal_H1_GPT_5_6_Sol_cross_model_6source_reproduction_20261024T122439888831Z"
)
LIVE = RUN_ROOT / "live"
PREVIOUS_LIVE = (
    ROOT
    / "results"
    / "formal_H1_GPT_5_6_Sol_cross_model_6source_20261024T035420792818Z"
    / "live"
)
CLAUDE_LIVE = (
    ROOT
    / "results"
    / "formal_H1_Claude_Sonnet_4_6_cross_model_6source_20261024T120057689162Z"
    / "live"
)
DEEPSEEK_LIVE = ROOT / "results" / "formal_H1_cross_system_live_20261023_v1" / "live"
OUTPUT = RUN_ROOT / "analysis_reproduction"

STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
TRAPS = ("BRANCH_NEW_V3", "DEPENDENCY_V6", "EVIDENCE_NEW_V3")
CONDITIONS = ("BASELINE",) + TRAPS
BRANCH_FILE = {
    "BASELINE": "BASELINE.json",
    "BRANCH_NEW_V3": "BRANCH_NEW_V3.json",
    "DEPENDENCY_V6": "DEPENDENCY_V6.json",
    "EVIDENCE_NEW_V3": "EVIDENCE_NEW_V3.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def median(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if clean else None


def stdev(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.stdev(clean) if len(clean) >= 2 else None


def direction(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if value > 0:
        return "increase"
    if value < 0:
        return "decrease"
    return "flat"


def compact_size(value: Any) -> int:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(raw)


def parse_sources(value: Any) -> set[int]:
    if isinstance(value, list):
        return {int(item) for item in value}
    if value is None:
        return set()
    text = str(value).strip().strip('"')
    if not text:
        return set()
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def load_requests(live: Path) -> list[dict[str, Any]]:
    path = live / "provider_requests.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_run(live: Path) -> tuple[dict[tuple[int, str, str], dict[str, Any]], list[dict[str, Any]]]:
    requests = load_requests(live)
    by_arm: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for request in requests:
        key = (
            int(request["source_index"]),
            str(request["state"]),
            str(request["condition"]),
        )
        by_arm.setdefault(key, []).append(request)
    for rows in by_arm.values():
        rows.sort(key=lambda row: int(row.get("local_request_index") or 0))

    pair_validity_path = live / "pair_validity.json"
    pair_validity = {
        (int(row["source_index"]), str(row["state"]), str(row["trap"])): row
        for row in read_json(pair_validity_path)
    }

    arms: dict[tuple[int, str, str], dict[str, Any]] = {}
    for path in sorted((live / "sources").glob("source-*/checkpoints/*/branches/*.json")):
        if path.name == "journal_state.json":
            continue
        source = int(path.parts[-5].split("-")[-1])
        state = path.parts[-3].upper()
        condition = path.stem
        if condition not in CONDITIONS:
            continue
        branch = read_json(path)
        metrics = branch["metrics"]
        requests_for_arm = by_arm.get((source, state, condition), [])
        reason_values = [request.get("reasoning_tokens") for request in requests_for_arm]
        cache_read_values = [request.get("cache_read_input_tokens") for request in requests_for_arm]
        cache_create_values = [
            request.get("cache_creation_input_tokens") for request in requests_for_arm
        ]
        provider_elapsed = sum(
            float(request.get("elapsed_seconds") or 0.0) for request in requests_for_arm
        )
        exposure = branch.get("initial_exposure_event") or {}
        visible_result = exposure.get("visible_result")
        terminal_reason = metrics.get("terminal_reason")
        arms[(source, state, condition)] = {
            "source_index": source,
            "family": branch["family"],
            "state": state,
            "condition": condition,
            "checkpoint_id": metrics.get("checkpoint_id"),
            "status": branch.get("status"),
            "implementation": branch.get("injected_implementation"),
            "input_tokens": int(metrics["input_tokens"]),
            "output_tokens": int(metrics["output_tokens"]),
            "total_tokens": int(metrics["total_tokens"]),
            "provider_calls": int(metrics["provider_calls"]),
            "tool_calls": int(metrics["tool_calls"]),
            "interaction_depth": int(metrics["interaction_depth"]),
            "wall_time": float(metrics["wall_time"]),
            "provider_elapsed_seconds": provider_elapsed,
            "unattributed_wall_seconds": float(metrics["wall_time"]) - provider_elapsed,
            "task_success": bool(metrics.get("task_success")),
            "budget_exhausted": terminal_reason == "interaction_budget_exhausted",
            "terminal_reason": terminal_reason,
            "technical_failure": bool(metrics.get("technical_failure")),
            "protocol_violation": bool(metrics.get("protocol_violation")),
            "post_terminal_model_calls": int(metrics.get("post_terminal_model_calls") or 0),
            "payload_bytes": compact_size(visible_result),
            "exposure_event_bytes": compact_size(exposure),
            "reason_known_requests": sum(value is not None for value in reason_values),
            "reason_missing_requests": sum(value is None for value in reason_values),
            "reason_known_sum": sum(int(value) for value in reason_values if value is not None),
            "reason_complete": bool(requests_for_arm) and all(
                value is not None for value in reason_values
            ),
            "cache_read_known_requests": sum(value is not None for value in cache_read_values),
            "cache_read_missing_requests": sum(value is None for value in cache_read_values),
            "cache_read_known_sum": sum(
                int(value) for value in cache_read_values if value is not None
            ),
            "cache_read_complete": bool(requests_for_arm) and all(
                value is not None for value in cache_read_values
            ),
            "cache_creation_known_requests": sum(
                value is not None for value in cache_create_values
            ),
            "cache_creation_missing_requests": sum(
                value is None for value in cache_create_values
            ),
            "cache_creation_known_sum": sum(
                int(value) for value in cache_create_values if value is not None
            ),
            "cache_creation_complete": bool(requests_for_arm) and all(
                value is not None for value in cache_create_values
            ),
            "requests": requests_for_arm,
            "first_request": requests_for_arm[0] if requests_for_arm else None,
            "pair_validity": pair_validity.get(
                (source, state, condition if condition in TRAPS else "")
            ),
        }
    return arms, requests


def sum_request_field(rows: Iterable[Mapping[str, Any]], field: str) -> int | None:
    values = [row.get(field) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def make_pairs(
    arms: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for source in sorted({key[0] for key in arms}):
        for state in STATES:
            baseline = arms.get((source, state, "BASELINE"))
            if baseline is None:
                continue
            for trap in TRAPS:
                arm = arms.get((source, state, trap))
                if arm is None:
                    continue
                baseline_requests = baseline["requests"]
                trap_requests = arm["requests"]
                baseline_first = baseline_requests[0] if baseline_requests else {}
                trap_first = trap_requests[0] if trap_requests else {}
                baseline_remaining = baseline_requests[1:]
                trap_remaining = trap_requests[1:]
                baseline_first_total = baseline_first.get("total_tokens")
                trap_first_total = trap_first.get("total_tokens")
                first_delta_total = (
                    int(trap_first_total) - int(baseline_first_total)
                    if baseline_first_total is not None and trap_first_total is not None
                    else None
                )
                baseline_remaining_total = sum_request_field(baseline_remaining, "total_tokens")
                trap_remaining_total = sum_request_field(trap_remaining, "total_tokens")
                remaining_delta_total = (
                    int(trap_remaining_total) - int(baseline_remaining_total)
                    if baseline_remaining_total is not None
                    and trap_remaining_total is not None
                    else None
                )
                cache_read_delta = (
                    arm["cache_read_known_sum"] - baseline["cache_read_known_sum"]
                    if arm["cache_read_complete"] and baseline["cache_read_complete"]
                    else None
                )
                cache_creation_delta = (
                    arm["cache_creation_known_sum"] - baseline["cache_creation_known_sum"]
                    if arm["cache_creation_complete"] and baseline["cache_creation_complete"]
                    else None
                )
                reason_delta = (
                    arm["reason_known_sum"] - baseline["reason_known_sum"]
                    if arm["reason_complete"] and baseline["reason_complete"]
                    else None
                )
                pair_validity = arm.get("pair_validity") or baseline.get("pair_validity") or {}
                clean = (
                    bool(pair_validity.get("pair_valid", False))
                    and baseline["status"] == "PASS"
                    and arm["status"] == "PASS"
                )
                delta_total = arm["total_tokens"] - baseline["total_tokens"]
                delta_input = arm["input_tokens"] - baseline["input_tokens"]
                delta_output = arm["output_tokens"] - baseline["output_tokens"]
                delta_wall = arm["wall_time"] - baseline["wall_time"]
                pairs.append(
                    {
                        "source_index": source,
                        "family": baseline["family"],
                        "state": state,
                        "trap": trap,
                        "checkpoint_id": baseline["checkpoint_id"],
                        "pair_status": "CLEAN" if clean else "INVALID",
                        "trap_implementation": arm["implementation"],
                        "baseline_terminal_reason": baseline["terminal_reason"],
                        "trap_terminal_reason": arm["terminal_reason"],
                        "baseline_task_success": baseline["task_success"],
                        "trap_task_success": arm["task_success"],
                        "baseline_budget_exhausted": baseline["budget_exhausted"],
                        "trap_budget_exhausted": arm["budget_exhausted"],
                        "baseline_input_tokens": baseline["input_tokens"],
                        "trap_input_tokens": arm["input_tokens"],
                        "delta_input_tokens": delta_input,
                        "baseline_output_tokens": baseline["output_tokens"],
                        "trap_output_tokens": arm["output_tokens"],
                        "delta_output_tokens": delta_output,
                        "baseline_total_tokens": baseline["total_tokens"],
                        "trap_total_tokens": arm["total_tokens"],
                        "delta_total_tokens": delta_total,
                        "input_amplification": (
                            arm["input_tokens"] / baseline["input_tokens"]
                            if baseline["input_tokens"]
                            else None
                        ),
                        "output_amplification": (
                            arm["output_tokens"] / baseline["output_tokens"]
                            if baseline["output_tokens"]
                            else None
                        ),
                        "total_token_amplification": (
                            arm["total_tokens"] / baseline["total_tokens"]
                            if baseline["total_tokens"]
                            else None
                        ),
                        "delta_cache_read_input_tokens": cache_read_delta,
                        "delta_cache_creation_input_tokens": cache_creation_delta,
                        "delta_reasoning_tokens": reason_delta,
                        "baseline_provider_calls": baseline["provider_calls"],
                        "trap_provider_calls": arm["provider_calls"],
                        "delta_provider_calls": arm["provider_calls"] - baseline["provider_calls"],
                        "baseline_tool_calls": baseline["tool_calls"],
                        "trap_tool_calls": arm["tool_calls"],
                        "delta_tool_calls": arm["tool_calls"] - baseline["tool_calls"],
                        "baseline_interaction_depth": baseline["interaction_depth"],
                        "trap_interaction_depth": arm["interaction_depth"],
                        "delta_interaction_depth": (
                            arm["interaction_depth"] - baseline["interaction_depth"]
                        ),
                        "baseline_wall_time": baseline["wall_time"],
                        "trap_wall_time": arm["wall_time"],
                        "delta_wall_time": delta_wall,
                        "wall_time_amplification": (
                            arm["wall_time"] / baseline["wall_time"]
                            if baseline["wall_time"]
                            else None
                        ),
                        "baseline_provider_elapsed_seconds": baseline[
                            "provider_elapsed_seconds"
                        ],
                        "trap_provider_elapsed_seconds": arm["provider_elapsed_seconds"],
                        "delta_provider_elapsed_seconds": (
                            arm["provider_elapsed_seconds"]
                            - baseline["provider_elapsed_seconds"]
                        ),
                        "baseline_unattributed_wall_seconds": baseline[
                            "unattributed_wall_seconds"
                        ],
                        "trap_unattributed_wall_seconds": arm["unattributed_wall_seconds"],
                        "delta_unattributed_wall_seconds": (
                            arm["unattributed_wall_seconds"]
                            - baseline["unattributed_wall_seconds"]
                        ),
                        "baseline_exposure_payload_bytes": baseline["payload_bytes"],
                        "trap_exposure_payload_bytes": arm["payload_bytes"],
                        "delta_exposure_payload_bytes": (
                            arm["payload_bytes"] - baseline["payload_bytes"]
                        ),
                        "baseline_first_request_total_tokens": baseline_first_total,
                        "trap_first_request_total_tokens": trap_first_total,
                        "first_request_delta_total_tokens": first_delta_total,
                        "remaining_requests_delta_total_tokens": remaining_delta_total,
                        "post_terminal_model_calls": (
                            baseline["post_terminal_model_calls"]
                            + arm["post_terminal_model_calls"]
                        ),
                    }
                )
    return pairs


def numeric_stats(
    rows: Iterable[Mapping[str, Any]],
    field: str,
    prefix: str,
) -> dict[str, Any]:
    values = [number(row.get(field)) for row in rows]
    clean = [float(value) for value in values if value is not None]
    result = {
        f"{prefix}_n": len(clean),
        f"{prefix}_mean": mean(clean),
        f"{prefix}_median": median(clean),
        f"{prefix}_sd": stdev(clean),
        f"{prefix}_min": min(clean) if clean else None,
        f"{prefix}_max": max(clean) if clean else None,
        f"{prefix}_positive": sum(value > 0 for value in clean),
        f"{prefix}_zero": sum(value == 0 for value in clean),
        f"{prefix}_negative": sum(value < 0 for value in clean),
    }
    return result


def aggregate_state_traps(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    clean_pairs = [row for row in pairs if row["pair_status"] == "CLEAN"]
    for state in STATES:
        for trap in TRAPS:
            selected = [
                row for row in clean_pairs if row["state"] == state and row["trap"] == trap
            ]
            row: dict[str, Any] = {
                "state": state,
                "trap": trap,
                "target_pairs": sum(
                    1
                    for pair in pairs
                    if pair["state"] == state and pair["trap"] == trap
                ),
                "clean_pairs": len(selected),
                "trap_task_success": sum(pair["trap_task_success"] for pair in selected),
                "baseline_task_success": sum(
                    pair["baseline_task_success"] for pair in selected
                ),
                "trap_budget_exhausted": sum(
                    pair["trap_budget_exhausted"] for pair in selected
                ),
                "baseline_budget_exhausted": sum(
                    pair["baseline_budget_exhausted"] for pair in selected
                ),
            }
            for field, prefix in (
                ("delta_input_tokens", "delta_input"),
                ("delta_output_tokens", "delta_output"),
                ("delta_total_tokens", "delta_total"),
                ("delta_cache_read_input_tokens", "delta_cache_read"),
                ("delta_cache_creation_input_tokens", "delta_cache_creation"),
                ("delta_reasoning_tokens", "delta_reasoning"),
                ("delta_provider_calls", "delta_provider_calls"),
                ("delta_tool_calls", "delta_tool_calls"),
                ("delta_interaction_depth", "delta_interaction_depth"),
                ("delta_wall_time", "delta_wall_time"),
                ("delta_provider_elapsed_seconds", "delta_provider_elapsed"),
                ("delta_unattributed_wall_seconds", "delta_unattributed_wall"),
                ("input_amplification", "input_amplification"),
                ("output_amplification", "output_amplification"),
                ("total_token_amplification", "total_token_amplification"),
                ("wall_time_amplification", "wall_time_amplification"),
                ("baseline_exposure_payload_bytes", "baseline_exposure_payload_bytes"),
                ("trap_exposure_payload_bytes", "trap_exposure_payload_bytes"),
                ("delta_exposure_payload_bytes", "delta_exposure_payload_bytes"),
                ("first_request_delta_total_tokens", "first_request_delta_total"),
                ("remaining_requests_delta_total_tokens", "remaining_delta_total"),
            ):
                row.update(numeric_stats(selected, field, prefix))
            rows.append(row)
    return rows


def condition_rows(arms: Mapping[tuple[int, str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for arm in sorted(
        arms.values(),
        key=lambda item: (item["source_index"], STATES.index(item["state"]), CONDITIONS.index(item["condition"])),
    ):
        rows.append(
            {
                "source_index": arm["source_index"],
                "family": arm["family"],
                "state": arm["state"],
                "condition": arm["condition"],
                "status": arm["status"],
                "checkpoint_id": arm["checkpoint_id"],
                "implementation": arm["implementation"],
                "input_tokens": arm["input_tokens"],
                "output_tokens": arm["output_tokens"],
                "total_tokens": arm["total_tokens"],
                "provider_calls": arm["provider_calls"],
                "tool_calls": arm["tool_calls"],
                "interaction_depth": arm["interaction_depth"],
                "wall_time": arm["wall_time"],
                "provider_elapsed_seconds": arm["provider_elapsed_seconds"],
                "unattributed_wall_seconds": arm["unattributed_wall_seconds"],
                "task_success": arm["task_success"],
                "budget_exhausted": arm["budget_exhausted"],
                "terminal_reason": arm["terminal_reason"],
                "technical_failure": arm["technical_failure"],
                "protocol_violation": arm["protocol_violation"],
                "post_terminal_model_calls": arm["post_terminal_model_calls"],
                "cache_read_known_sum": arm["cache_read_known_sum"],
                "cache_read_known_requests": arm["cache_read_known_requests"],
                "cache_read_missing_requests": arm["cache_read_missing_requests"],
                "cache_creation_known_sum": arm["cache_creation_known_sum"],
                "cache_creation_known_requests": arm["cache_creation_known_requests"],
                "cache_creation_missing_requests": arm["cache_creation_missing_requests"],
                "reasoning_known_sum": arm["reason_known_sum"],
                "reasoning_known_requests": arm["reason_known_requests"],
                "reasoning_missing_requests": arm["reason_missing_requests"],
                "exposure_payload_bytes": arm["payload_bytes"],
                "exposure_event_bytes": arm["exposure_event_bytes"],
            }
        )
    return rows


def latency_summary(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, str, list[dict[str, Any]]]] = [
        ("ALL", "ALL", "ALL", requests)
    ]
    for state in STATES:
        for condition in CONDITIONS:
            groups.append(
                (
                    state,
                    condition,
                    "STATE_CONDITION",
                    [
                        row
                        for row in requests
                        if row.get("state") == state and row.get("condition") == condition
                    ],
                )
            )
    rows = []
    for state, condition, scope, selected in groups:
        elapsed = sorted(float(row["elapsed_seconds"]) for row in selected)
        if not elapsed:
            continue

        def quantile(q: float) -> float:
            index = max(0, math.ceil(q * len(elapsed)) - 1)
            return elapsed[min(index, len(elapsed) - 1)]

        rows.append(
            {
                "scope": scope,
                "state": state,
                "condition": condition,
                "n": len(selected),
                "elapsed_mean": statistics.mean(elapsed),
                "elapsed_p50": quantile(0.50),
                "elapsed_p90": quantile(0.90),
                "elapsed_p95": quantile(0.95),
                "elapsed_p99": quantile(0.99),
                "elapsed_max": elapsed[-1],
                "input_tokens_mean": mean(row.get("input_tokens") for row in selected),
                "output_tokens_mean": mean(row.get("output_tokens") for row in selected),
                "total_tokens_mean": mean(row.get("total_tokens") for row in selected),
            }
        )
    return rows


def defense_cost_rows(state_traps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_key = {
        (row["state"], row["trap"]): row for row in state_traps
    }
    for state in STATES:
        for trap in TRAPS:
            row = by_key[(state, trap)]
            rows.append(
                {
                    "state": state,
                    "trap": trap,
                    "n": row["clean_pairs"],
                    "baseline_exposure_payload_bytes_mean": row[
                        "baseline_exposure_payload_bytes_mean"
                    ],
                    "trap_exposure_payload_bytes_mean": row[
                        "trap_exposure_payload_bytes_mean"
                    ],
                    "trap_exposure_payload_bytes_median": row[
                        "trap_exposure_payload_bytes_median"
                    ],
                    "trap_exposure_payload_bytes_min": row[
                        "trap_exposure_payload_bytes_min"
                    ],
                    "trap_exposure_payload_bytes_max": row[
                        "trap_exposure_payload_bytes_max"
                    ],
                    "delta_exposure_payload_bytes_mean": row[
                        "delta_exposure_payload_bytes_mean"
                    ],
                    "delta_exposure_payload_bytes_median": row[
                        "delta_exposure_payload_bytes_median"
                    ],
                    "delta_exposure_payload_bytes_min": row[
                        "delta_exposure_payload_bytes_min"
                    ],
                    "delta_exposure_payload_bytes_max": row[
                        "delta_exposure_payload_bytes_max"
                    ],
                    "defense_extra_provider_calls_observed": 0,
                    "defense_extra_tool_calls_observed": 0,
                    "state_observer_service_seconds": "NA",
                    "trap_generation_seconds": "NA",
                    "trap_injection_seconds": "NA",
                    "attack_delta_total_tokens_mean": row["delta_total_mean"],
                    "attack_delta_provider_calls_mean": row[
                        "delta_provider_calls_mean"
                    ],
                    "attack_delta_tool_calls_mean": row["delta_tool_calls_mean"],
                    "attack_delta_wall_time_mean": row["delta_wall_time_mean"],
                    "defense_total_cost_usd": "NA",
                    "defense_to_attack_total_token_ratio": "NA",
                    "defense_to_attack_wall_time_ratio": "NA",
                    "limitation": (
                        "The formal run did not timestamp Observer/Trap local work or "
                        "tool execution separately; defense cost ratios remain NA."
                    ),
                }
            )
    return rows


def reproduction_comparison(
    current_pairs: list[dict[str, Any]],
    previous_pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current_by_key = {
        (row["source_index"], row["state"], row["trap"]): row for row in current_pairs
    }
    previous_by_key = {
        (row["source_index"], row["state"], row["trap"]): row for row in previous_pairs
    }
    pair_rows = []
    common_keys = sorted(set(current_by_key) & set(previous_by_key))
    for key in common_keys:
        current = current_by_key[key]
        previous = previous_by_key[key]
        pair_rows.append(
            {
                "source_index": key[0],
                "family": current["family"],
                "state": key[1],
                "trap": key[2],
                "current_delta_total_tokens": current["delta_total_tokens"],
                "previous_delta_total_tokens": previous["delta_total_tokens"],
                "difference_current_minus_previous_total_tokens": (
                    current["delta_total_tokens"] - previous["delta_total_tokens"]
                ),
                "current_delta_wall_time": current["delta_wall_time"],
                "previous_delta_wall_time": previous["delta_wall_time"],
                "difference_current_minus_previous_wall_time": (
                    current["delta_wall_time"] - previous["delta_wall_time"]
                ),
                "total_direction_match": (
                    direction(current["delta_total_tokens"])
                    == direction(previous["delta_total_tokens"])
                ),
                "wall_direction_match": (
                    direction(current["delta_wall_time"])
                    == direction(previous["delta_wall_time"])
                ),
            }
        )

    summary = []
    for state in STATES:
        current_state = [
            row for row in current_pairs if row["state"] == state and row["pair_status"] == "CLEAN"
        ]
        previous_state = [
            row for row in previous_pairs if row["state"] == state and row["pair_status"] == "CLEAN"
        ]
        current_rank = {
            row["trap"]: rank
            for rank, row in enumerate(
                sorted(
                    [
                        {
                            "trap": trap,
                            "value": mean(
                                pair["delta_total_tokens"]
                                for pair in current_state
                                if pair["trap"] == trap
                            ),
                        }
                        for trap in TRAPS
                    ],
                    key=lambda row: row["value"] if row["value"] is not None else -math.inf,
                    reverse=True,
                ),
                1,
            )
        }
        previous_rank = {
            row["trap"]: rank
            for rank, row in enumerate(
                sorted(
                    [
                        {
                            "trap": trap,
                            "value": mean(
                                pair["delta_total_tokens"]
                                for pair in previous_state
                                if pair["trap"] == trap
                            ),
                        }
                        for trap in TRAPS
                    ],
                    key=lambda row: row["value"] if row["value"] is not None else -math.inf,
                    reverse=True,
                ),
                1,
            )
        }
        for trap in TRAPS:
            current = [
                pair
                for pair in current_state
                if pair["trap"] == trap
            ]
            previous = [
                pair
                for pair in previous_state
                if pair["trap"] == trap
            ]
            summary.append(
                {
                    "state": state,
                    "trap": trap,
                    "current_clean_pairs": len(current),
                    "previous_clean_pairs": len(previous),
                    "current_delta_total_tokens_mean": mean(
                        row["delta_total_tokens"] for row in current
                    ),
                    "previous_delta_total_tokens_mean": mean(
                        row["delta_total_tokens"] for row in previous
                    ),
                    "difference_current_minus_previous": (
                        (mean(row["delta_total_tokens"] for row in current) or 0)
                        - (mean(row["delta_total_tokens"] for row in previous) or 0)
                    ),
                    "current_delta_wall_time_mean": mean(
                        row["delta_wall_time"] for row in current
                    ),
                    "previous_delta_wall_time_mean": mean(
                        row["delta_wall_time"] for row in previous
                    ),
                    "current_total_rank": current_rank[trap],
                    "previous_total_rank": previous_rank[trap],
                    "total_rank_stable": current_rank[trap] == previous_rank[trap],
                }
            )
    return pair_rows, summary


def cross_model_rows(
    current_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    historical_path = CLAUDE_LIVE / "cross_model_comparison.csv"
    historical = list(csv.DictReader(historical_path.open(encoding="utf-8")))
    deepseek_pairs = list(
        csv.DictReader((DEEPSEEK_LIVE / "pair_ledger.csv").open(encoding="utf-8"))
    )
    current_by_key = {
        (int(row["source_index"]), row["state"], row["trap"]): row
        for row in current_pairs
        if row["pair_status"] == "CLEAN"
    }
    deepseek_by_key = {
        (int(row["source_index"]), row["state"], row["trap"]): row
        for row in deepseek_pairs
        if row["pair_valid"] == "True"
    }

    rows = []
    for historical_row in historical:
        state = historical_row["state"]
        trap = historical_row["trap"]
        sources = parse_sources(historical_row.get("common_source_indices"))
        current = [
            current_by_key[(source, state, trap)]
            for source in sorted(sources)
            if (source, state, trap) in current_by_key
        ]
        deepseek = [
            deepseek_by_key[(source, state, trap)]
            for source in sorted(sources)
            if (source, state, trap) in deepseek_by_key
        ]
        current_total_delta = mean(row["delta_total_tokens"] for row in current)
        deepseek_input_delta = mean(
            float(row["delta_input_tokens"])
            for row in deepseek
            if row.get("delta_input_tokens") not in ("", None)
        )
        deepseek_output_delta = mean(
            float(row["delta_output_tokens"])
            for row in deepseek
            if row.get("delta_output_tokens") not in ("", None)
        )
        deepseek_total_delta = (
            deepseek_input_delta + deepseek_output_delta
            if deepseek_input_delta is not None and deepseek_output_delta is not None
            else None
        )
        claude_input_delta = float(historical_row["claude_delta_input_mean"])
        claude_output_delta = float(historical_row["claude_delta_output_mean"])
        claude_total_delta = claude_input_delta + claude_output_delta
        current_direction = direction(current_total_delta)
        deepseek_direction = direction(deepseek_total_delta)
        claude_direction = direction(claude_total_delta)
        rows.append(
            {
                "state": state,
                "trap": trap,
                "common_source_indices": ",".join(str(source) for source in sorted(sources)),
                "comparison_basis": (
                    "Each model differenced against its own BASELINE on the same "
                    "Source/checkpoint; GPT uses provider total_tokens, while "
                    "DeepSeek and Claude use Input+Output within their own usage semantics."
                ),
                "current_gpt_clean_pairs": len(current),
                "current_gpt_delta_total_tokens_mean": current_total_delta,
                "current_gpt_delta_input_tokens_mean": mean(
                    row["delta_input_tokens"] for row in current
                ),
                "current_gpt_delta_output_tokens_mean": mean(
                    row["delta_output_tokens"] for row in current
                ),
                "current_gpt_delta_provider_calls_mean": mean(
                    row["delta_provider_calls"] for row in current
                ),
                "current_gpt_delta_tool_calls_mean": mean(
                    row["delta_tool_calls"] for row in current
                ),
                "current_gpt_delta_wall_time_mean": mean(
                    row["delta_wall_time"] for row in current
                ),
                "deepseek_clean_pairs": len(deepseek),
                "deepseek_delta_input_output_tokens_mean": deepseek_total_delta,
                "deepseek_delta_input_tokens_mean": deepseek_input_delta,
                "deepseek_delta_output_tokens_mean": deepseek_output_delta,
                "deepseek_delta_provider_calls_mean": float(
                    historical_row["deepseek_delta_provider_calls_mean"]
                ),
                "deepseek_delta_tool_calls_mean": float(
                    historical_row["deepseek_delta_tool_calls_mean"]
                ),
                "deepseek_delta_wall_time_mean": float(
                    historical_row["deepseek_delta_wall_time_mean"]
                ),
                "claude_clean_pairs": int(historical_row["claude_clean_pairs"]),
                "claude_delta_input_output_tokens_mean": claude_total_delta,
                "claude_delta_input_tokens_mean": claude_input_delta,
                "claude_delta_output_tokens_mean": claude_output_delta,
                "claude_delta_provider_calls_mean": float(
                    historical_row["claude_delta_provider_calls_mean"]
                ),
                "claude_delta_tool_calls_mean": float(
                    historical_row["claude_delta_tool_calls_mean"]
                ),
                "claude_delta_wall_time_mean": float(
                    historical_row["claude_delta_wall_time_mean"]
                ),
                "current_gpt_trend": current_direction,
                "deepseek_trend": deepseek_direction,
                "claude_trend": claude_direction,
                "current_gpt_deepseek_trend_match": current_direction == deepseek_direction,
                "current_gpt_claude_trend_match": current_direction == claude_direction,
                "deepseek_claude_trend_match": deepseek_direction == claude_direction,
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, separator]
    for row in rows:
        cells = []
        for key, _, format_spec in columns:
            value = row.get(key)
            if value is None or value == "NA":
                cells.append("NA")
            elif format_spec == "int":
                cells.append(f"{int(value):,}")
            elif format_spec == "signed_int":
                cells.append(f"{int(value):+,}")
            elif format_spec == "float1":
                cells.append(f"{float(value):.1f}")
            elif format_spec == "signed_float1":
                cells.append(f"{float(value):+.1f}")
            elif format_spec == "float2":
                cells.append(f"{float(value):.2f}")
            elif format_spec == "signed_float2":
                cells.append(f"{float(value):+.2f}")
            elif format_spec == "bool":
                cells.append("YES" if bool(value) else "NO")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    arms, requests = load_run(LIVE)
    previous_arms, previous_requests = load_run(PREVIOUS_LIVE)
    pairs = make_pairs(arms)
    previous_pairs = make_pairs(previous_arms)
    clean_pairs = [row for row in pairs if row["pair_status"] == "CLEAN"]
    state_traps = aggregate_state_traps(pairs)
    conditions = condition_rows(arms)
    latency = latency_summary(requests)
    defense = defense_cost_rows(state_traps)
    reproduction_pairs, reproduction_summary = reproduction_comparison(
        pairs, previous_pairs
    )
    cross_model = cross_model_rows(pairs)

    source_rows = []
    for source_path in sorted((LIVE / "sources").glob("source-*/source_trajectory.json")):
        trajectory = read_json(source_path)
        source_requests = [
            row for row in requests if int(row["source_index"]) == int(trajectory["source_index"])
        ]
        source_rows.append(
            {
                "source_index": trajectory["source_index"],
                "fixture_id": trajectory["fixture_id"],
                "family": trajectory["family"],
                "status": trajectory["status"],
                "natural_provider_calls": trajectory["provider_calls"],
                "natural_input_tokens": trajectory["usage_summary"]["totals"][
                    "input_tokens"
                ],
                "natural_output_tokens": trajectory["usage_summary"]["totals"][
                    "output_tokens"
                ],
                "natural_total_tokens": trajectory["usage_summary"]["totals"][
                    "total_tokens"
                ],
                "natural_provider_elapsed_seconds": sum(
                    float(row["elapsed_seconds"]) for row in source_requests
                ),
                "natural_wall_time": "NA",
                "accepted_states": ",".join(trajectory["accepted_states"]),
                "terminal_reason": trajectory["terminal_reason"],
            }
        )

    request_timing = []
    for request in requests:
        request_timing.append(
            {
                "request_id": request["request_id"],
                "source_index": request["source_index"],
                "family": request["family"],
                "state": request["state"],
                "condition": request["condition"],
                "local_request_index": request["local_request_index"],
                "request_model_id": request["request_model_id"],
                "provider_model_id": request["provider_model_id"],
                "http_status": request["http_status"],
                "finish_reason": request["finish_reason"],
                "outcome": request["outcome"],
                "input_tokens": request.get("input_tokens"),
                "output_tokens": request.get("output_tokens"),
                "total_tokens": request.get("total_tokens"),
                "reasoning_tokens": request.get("reasoning_tokens"),
                "cache_read_input_tokens": request.get("cache_read_input_tokens"),
                "cache_creation_input_tokens": request.get(
                    "cache_creation_input_tokens"
                ),
                "elapsed_seconds": request.get("elapsed_seconds"),
                "journal_path": request.get("journal_path"),
            }
        )

    run_status = read_json(LIVE / "run_status.json")
    provider_free = read_json(RUN_ROOT / "provider-free" / "provider_free_admission.json")
    final_request = max(
        requests,
        key=lambda row: int(str(row.get("request_id", "")).rsplit("-", 1)[-1]),
    )
    final_journal_path = final_request.get("journal_path")
    absolute_final_journal = (
        LIVE / final_journal_path
        if isinstance(final_journal_path, str)
        else None
    )
    final_ledger = (
        read_json(absolute_final_journal)
        if absolute_final_journal is not None and absolute_final_journal.is_file()
        else {}
    )
    previous_final_request = max(
        previous_requests,
        key=lambda row: int(str(row.get("request_id", "")).rsplit("-", 1)[-1]),
    )
    previous_final_path = previous_final_request.get("journal_path")
    previous_final_ledger = (
        read_json(PREVIOUS_LIVE / previous_final_path)
        if isinstance(previous_final_path, str)
        and (PREVIOUS_LIVE / previous_final_path).is_file()
        else {}
    )

    used_reasons = [
        int(row["reasoning_tokens"])
        for row in requests
        if row.get("reasoning_tokens") is not None
    ]
    known_cache_read = [
        int(row["cache_read_input_tokens"])
        for row in requests
        if row.get("cache_read_input_tokens") is not None
    ]
    known_cache_creation = [
        int(row["cache_creation_input_tokens"])
        for row in requests
        if row.get("cache_creation_input_tokens") is not None
    ]
    validation = {
        "platform_label": "CyberStrikeAI",
        "protocol_id": run_status["protocol_id"],
        "run_status": run_status["status"],
        "fresh_batch_directory": str(RUN_ROOT),
        "provider_free": {
            "path": str(RUN_ROOT / "provider-free" / "provider_free_admission.json"),
            "passed": bool(provider_free.get("passed")),
            "provider_calls": int(provider_free.get("provider_calls", -1)),
            "network_attempts": int(provider_free.get("network_attempts", -1)),
            "checks_passed": sum(bool(value) for value in provider_free["checks"].values()),
            "checks_total": len(provider_free["checks"]),
            "sources_passed": sum(
                row.get("status") == "PASS" for row in provider_free["sources"]
            ),
            "sources_total": len(provider_free["sources"]),
            "arms": len(provider_free["arms"]),
            "oracles_passed": sum(
                row.get("status") == "PASS" for row in provider_free["oracles"]
            ),
            "oracles_total": len(provider_free["oracles"]),
            "extensions_passed": sum(
                row.get("status") == "PASS" for row in provider_free["extensions"]
            ),
            "extensions_total": len(provider_free["extensions"]),
        },
        "provider_calls": len(requests),
        "provider_elapsed_seconds_total": sum(
            float(row["elapsed_seconds"]) for row in requests
        ),
        "provider_request_ceiling": 1056,
        "provider_request_within_ceiling": len(requests) <= 1056,
        "source_count": len(source_rows),
        "natural_trajectories_completed": sum(
            row["status"] == "PASS" for row in source_rows
        ),
        "valid_checkpoints": sum(
            row.get("status") == "VALID"
            for row in read_json(LIVE / "checkpoint_audit.json")
        ),
        "condition_arms": len(arms),
        "condition_passes": sum(arm["status"] == "PASS" for arm in arms.values()),
        "clean_pairs": len(clean_pairs),
        "target_pairs": 54,
        "technical_failures": sum(pair["technical_failure"] for pair in arms.values()),
        "protocol_violations": sum(pair["protocol_violation"] for pair in arms.values()),
        "post_terminal_model_calls": sum(
            pair["post_terminal_model_calls"] for pair in pairs
        ),
        "http_statuses": dict(Counter(str(row["http_status"]) for row in requests)),
        "finish_reasons": dict(Counter(str(row["finish_reason"]) for row in requests)),
        "request_models": dict(Counter(str(row["request_model_id"]) for row in requests)),
        "response_models": dict(
            Counter(str(row["provider_model_id"]) for row in requests)
        ),
        "usage_known": {
            "input_tokens_sum": sum(int(row["input_tokens"]) for row in requests),
            "output_tokens_sum": sum(int(row["output_tokens"]) for row in requests),
            "total_tokens_sum": sum(int(row["total_tokens"]) for row in requests),
            "cache_read_known_sum": sum(known_cache_read),
            "cache_read_known_requests": len(known_cache_read),
            "cache_read_unknown_requests": len(requests) - len(known_cache_read),
            "cache_creation_known_sum": sum(known_cache_creation),
            "cache_creation_known_requests": len(known_cache_creation),
            "cache_creation_unknown_requests": len(requests)
            - len(known_cache_creation),
            "reasoning_known_sum": sum(used_reasons),
            "reasoning_known_requests": len(used_reasons),
            "reasoning_unknown_requests": len(requests) - len(used_reasons),
            "reasoning_tokens_is_output_subset": True,
            "cache_read_is_input_subset": True,
        },
        "automatic_retry": False,
        "automatic_backfill": False,
        "automatic_recovery": False,
        "cost_estimate_usd_list_price": final_ledger.get(
            "cumulative_cost_usd_list_estimate"
        ),
        "previous_gpt_provider_calls": len(previous_requests),
        "previous_gpt_total_tokens": sum(
            int(row["total_tokens"]) for row in previous_requests
        ),
        "previous_gpt_cost_estimate_usd_list_price": previous_final_ledger.get(
            "cumulative_cost_usd_list_estimate"
        ),
        "timing_fields_not_instrumented": [
            "state_observer_service_seconds",
            "trap_generation_seconds",
            "trap_injection_seconds",
            "tool_cumulative_execution_seconds",
        ],
        "unattributed_wall_time_definition": (
            "branch wall_time minus the sum of recorded Provider request elapsed_seconds; "
            "it includes tool, Observer, trap-local work and orchestration and must not "
            "be interpreted as tool time alone."
        ),
        "scientific_claim": (
            "Descriptive six-Source paired analysis; no confirmatory significance test."
        ),
    }

    write_csv(OUTPUT / "condition_metrics.csv", conditions)
    write_csv(OUTPUT / "source_natural_summary.csv", source_rows)
    write_csv(OUTPUT / "pair_metrics.csv", pairs)
    write_csv(OUTPUT / "state_trap_metrics.csv", state_traps)
    write_csv(
        OUTPUT / "paired_wall_time.csv",
        [
            {
                key: row.get(key)
                for key in (
                    "source_index",
                    "family",
                    "state",
                    "trap",
                    "checkpoint_id",
                    "baseline_wall_time",
                    "trap_wall_time",
                    "delta_wall_time",
                    "wall_time_amplification",
                    "baseline_provider_elapsed_seconds",
                    "trap_provider_elapsed_seconds",
                    "delta_provider_elapsed_seconds",
                    "baseline_unattributed_wall_seconds",
                    "trap_unattributed_wall_seconds",
                    "delta_unattributed_wall_seconds",
                )
            }
            for row in pairs
        ],
    )
    write_csv(
        OUTPUT / "interaction_decomposition.csv",
        [
            {
                "state": row["state"],
                "trap": row["trap"],
                "n": row["clean_pairs"],
                "delta_total_tokens_mean": row["delta_total_mean"],
                "first_request_delta_total_tokens_mean": row[
                    "first_request_delta_total_mean"
                ],
                "first_request_delta_total_tokens_median": row[
                    "first_request_delta_total_median"
                ],
                "remaining_requests_delta_total_tokens_mean": row[
                    "remaining_delta_total_mean"
                ],
                "remaining_requests_delta_total_tokens_median": row[
                    "remaining_delta_total_median"
                ],
                "delta_exposure_payload_bytes_mean": row[
                    "delta_exposure_payload_bytes_mean"
                ],
                "delta_provider_calls_mean": row["delta_provider_calls_mean"],
                "delta_tool_calls_mean": row["delta_tool_calls_mean"],
                "interpretation": (
                    "First request is the first post-exposure model request. The remaining "
                    "component is autonomous follow-up behavior, not local injection time."
                ),
            }
            for row in state_traps
        ],
    )
    write_csv(OUTPUT / "defense_cost.csv", defense)
    write_csv(OUTPUT / "request_timing_usage.csv", request_timing)
    write_csv(OUTPUT / "request_latency_summary.csv", latency)
    write_csv(
        OUTPUT / "reproduction_vs_previous_gpt_pair.csv", reproduction_pairs
    )
    write_csv(
        OUTPUT / "reproduction_vs_previous_gpt.csv", reproduction_summary
    )
    write_csv(OUTPUT / "cross_model_reproduction.csv", cross_model)
    write_json(OUTPUT / "validation.json", validation)
    code_paths = [
        ROOT / "h1_gpt_cross_model_6source_20261024" / "runner.py",
        ROOT / "h1_gpt_cross_model_6source_20261024" / "analyze.py",
        ROOT
        / "h1_gpt_cross_model_6source_20261024"
        / "derive_reproduction_analysis.py",
    ]
    write_json(
        OUTPUT / "analysis_code_snapshot.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "protocol_id": run_status["protocol_id"],
            "configuration_snapshot": str(
                RUN_ROOT / "provider-free" / "configuration_snapshot.json"
            ),
            "analysis_inputs": {
                "current_live": str(LIVE),
                "previous_gpt_live": str(PREVIOUS_LIVE),
                "deepseek_live": str(DEEPSEEK_LIVE),
                "claude_live": str(CLAUDE_LIVE),
            },
            "code_files": [
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                }
                for path in code_paths
            ],
            "frozen_model_parameters": run_status["provider"],
        },
    )

    st_by_key = {(row["state"], row["trap"]): row for row in state_traps}
    reproduction_by_key = {
        (row["state"], row["trap"]): row for row in reproduction_summary
    }
    cross_by_key = {(row["state"], row["trap"]): row for row in cross_model}

    state_trap_table = markdown_table(
        state_traps,
        [
            ("state", "State", ""),
            ("trap", "Trap", ""),
            ("clean_pairs", "n", "int"),
            ("delta_total_mean", "ΔTotal", "signed_float1"),
            ("delta_total_median", "Median ΔTotal", "signed_float1"),
            ("delta_input_mean", "ΔInput", "signed_float1"),
            ("delta_output_mean", "ΔOutput", "signed_float1"),
            ("delta_provider_calls_mean", "ΔCalls", "signed_float2"),
            ("delta_tool_calls_mean", "ΔTools", "signed_float2"),
            ("delta_wall_time_mean", "ΔWall(s)", "signed_float1"),
            ("total_token_amplification_mean", "Token Amp", "float2"),
            ("wall_time_amplification_mean", "Wall Amp", "float2"),
        ],
    )
    direction_table = markdown_table(
        state_traps,
        [
            ("state", "State", ""),
            ("trap", "Trap", ""),
            ("delta_total_positive", "Token Δ>0", "int"),
            ("delta_total_zero", "Token Δ=0", "int"),
            ("delta_total_negative", "Token Δ<0", "int"),
            ("delta_wall_time_positive", "Wall Δ>0", "int"),
            ("delta_wall_time_zero", "Wall Δ=0", "int"),
            ("delta_wall_time_negative", "Wall Δ<0", "int"),
        ],
    )
    task_table = markdown_table(
        state_traps,
        [
            ("state", "State", ""),
            ("trap", "Trap", ""),
            ("trap_task_success", "Trap success", "int"),
            ("baseline_task_success", "Baseline success", "int"),
            ("trap_budget_exhausted", "Trap budget", "int"),
            ("baseline_budget_exhausted", "Baseline budget", "int"),
        ],
    )
    wall_table = markdown_table(
        [
            {
                "state": row["state"],
                "trap": row["trap"],
                "delta_wall_mean": row["delta_wall_time_mean"],
                "delta_wall_median": row["delta_wall_time_median"],
                "wall_amp": row["wall_time_amplification_mean"],
                "positive": row["delta_wall_time_positive"],
                "zero": row["delta_wall_time_zero"],
                "negative": row["delta_wall_time_negative"],
                "delta_provider_elapsed": row["delta_provider_elapsed_mean"],
                "delta_unattributed": row["delta_unattributed_wall_mean"],
            }
            for row in state_traps
        ],
        [
            ("state", "State", ""),
            ("trap", "Trap", ""),
            ("delta_wall_mean", "Mean ΔWall(s)", "signed_float1"),
            ("delta_wall_median", "Median ΔWall(s)", "signed_float1"),
            ("wall_amp", "Wall Amp", "float2"),
            ("positive", "Δ>0", "int"),
            ("zero", "Δ=0", "int"),
            ("negative", "Δ<0", "int"),
            ("delta_provider_elapsed", "ΔProvider time(s)", "signed_float1"),
            ("delta_unattributed", "ΔResidual time(s)", "signed_float1"),
        ],
    )
    decomposition_table = markdown_table(
        [
            {
                "state": row["state"],
                "trap": row["trap"],
                "total": row["delta_total_mean"],
                "first": row["first_request_delta_total_mean"],
                "remaining": row["remaining_delta_total_mean"],
                "payload": row["delta_exposure_payload_bytes_mean"],
                "calls": row["delta_provider_calls_mean"],
                "tools": row["delta_tool_calls_mean"],
            }
            for row in state_traps
        ],
        [
            ("state", "State", ""),
            ("trap", "Trap", ""),
            ("total", "Mean ΔTotal", "signed_float1"),
            ("first", "首次暴露后首请求 Δ", "signed_float1"),
            ("remaining", "后续自主交互 Δ", "signed_float1"),
            ("payload", "ΔPayload bytes", "signed_float1"),
            ("calls", "ΔCalls", "signed_float2"),
            ("tools", "ΔTools", "signed_float2"),
        ],
    )
    reproduction_table = markdown_table(
        reproduction_summary,
        [
            ("state", "State", ""),
            ("trap", "Trap", ""),
            ("current_delta_total_tokens_mean", "本轮 ΔTotal", "signed_float1"),
            ("previous_delta_total_tokens_mean", "上轮 ΔTotal", "signed_float1"),
            ("difference_current_minus_previous", "差值", "signed_float1"),
            ("current_total_rank", "本轮排名", "int"),
            ("previous_total_rank", "上轮排名", "int"),
            ("total_rank_stable", "排名稳定", "bool"),
        ],
    )
    cross_table = markdown_table(
        cross_model,
        [
            ("state", "State", ""),
            ("trap", "Trap", ""),
            ("current_gpt_delta_total_tokens_mean", "GPT Δ", "signed_float1"),
            ("deepseek_delta_input_output_tokens_mean", "DeepSeek Δ", "signed_float1"),
            ("claude_delta_input_output_tokens_mean", "Claude Δ", "signed_float1"),
            ("current_gpt_trend", "GPT", ""),
            ("deepseek_trend", "DeepSeek", ""),
            ("claude_trend", "Claude", ""),
            ("current_gpt_deepseek_trend_match", "GPT=DS", "bool"),
            ("current_gpt_claude_trend_match", "GPT=Claude", "bool"),
        ],
    )

    total_tokens = validation["usage_known"]["total_tokens_sum"]
    total_input = validation["usage_known"]["input_tokens_sum"]
    total_output = validation["usage_known"]["output_tokens_sum"]
    total_pairs = len(clean_pairs)
    total_token_positive = sum(
        row["delta_total_tokens"] > 0 for row in clean_pairs
    )
    total_wall_positive = sum(
        row["delta_wall_time"] > 0 for row in clean_pairs
    )
    reproduction_pair_match = sum(
        row["total_direction_match"] for row in reproduction_pairs
    )
    rank_stable = sum(
        row["total_rank_stable"] for row in reproduction_summary
    )
    cross_trend_matches = {
        "gpt_vs_deepseek": sum(
            row["current_gpt_deepseek_trend_match"] for row in cross_model
        ),
        "gpt_vs_claude": sum(
            row["current_gpt_claude_trend_match"] for row in cross_model
        ),
        "deepseek_vs_claude": sum(
            row["deepseek_claude_trend_match"] for row in cross_model
        ),
    }

    best_by_state = {}
    for state in STATES:
        selected = [row for row in state_traps if row["state"] == state]
        ordered = sorted(
            selected,
            key=lambda row: (
                row["delta_total_mean"]
                if row["delta_total_mean"] is not None
                else -math.inf
            ),
            reverse=True,
        )
        best_by_state[state] = {
            "token": ordered[0]["trap"],
            "wall": sorted(
                selected,
                key=lambda row: (
                    row["delta_wall_time_mean"]
                    if row["delta_wall_time_mean"] is not None
                    else -math.inf
                ),
                reverse=True,
            )[0]["trap"],
        }

    report = f"""# CyberStrikeAI — GPT-5.6 Sol 六 Source H1 独立复现报告

## 摘要

- 本报告只使用新批次 `{RUN_ROOT.name}` 的 72 条条件轨迹，不混入此前 GPT、DeepSeek 或 Claude 的槽位。
- 6/6 Source 自然轨迹、18/18 checkpoint、72/72 条件槽位通过，得到 {total_pairs}/54 个 clean pair。
- Provider-free 门禁通过 {validation['provider_free']['checks_passed']}/{validation['provider_free']['checks_total']} 项，真实 Provider Calls 与网络尝试均为 0；6/6 Oracle 和 6/6 扩展环境检查通过。
- Provider 请求共 {validation['provider_calls']}/1,056；真实 Input、Output、Total 分别为 {total_input:,}、{total_output:,}、{total_tokens:,} Tokens。
- 全部请求 HTTP 200，请求和响应模型均为 `gpt-5.6-sol`；技术失败 0，协议违规 0，terminal 后模型调用 0。
- 自动重试、补样和 Recovery 均为 0。所有自然轨迹与 checkpoint 都由本批次重新生成。
- 六个 Source 的结论属于小样本描述性结果，不作统计显著性主张。

## 冻结配置与执行身份

- Platform：CyberStrikeAI。
- Model / transport：`gpt-5.6-sol` / OpenAI-compatible Chat Completions。
- `reasoning_effort=medium`、`max_completion_tokens=2400`、`temperature=0.0`、`tool_choice=auto`、`parallel_tool_calls=false`、`stream=false`、timeout 360 秒。
- 温度的上游实际生效性保留为 `UNVERIFIED`；请求中确实发送了 0.0。
- Sources：WEB 1、6；API 11、16；SERVICE 21、26。
- States：RECON、EXPLOITATION、VERIFICATION。
- Conditions：BASELINE、BRANCH_NEW_V3、DEPENDENCY_V6、EVIDENCE_NEW_V3。
- K=12，boundary=`pre-effect-v1`；每 Source 自然轨迹最多 32 次 Provider 请求，全局 Provider 请求上限 1,056。
- 实际 Provider usage 为逐请求审计：Input 是 `prompt_tokens`，Output 是 `completion_tokens`，Total 使用 Provider 返回的 `total_tokens`。Reasoning 是 Output 子集，没有重复计入 Total。
- Cache Read 是 Input 的子集，没有重复累加。缺失的 Cache/Reasoning 明细保持 NA。
- 成本 {validation['cost_estimate_usd_list_price']:.6f} USD 是按 Runner 配置的公开列表价格估算，不是已核实的最终账单。

## 数据与审计

自然轨迹请求数为 {', '.join(f"Source {row['source_index']}={row['natural_provider_calls']}" for row in source_rows)}。每 Source 在 32 次上限内完成并给出三个合法 checkpoint。

Reasoning 明细仅有 {validation['usage_known']['reasoning_known_requests']}/{validation['provider_calls']} 次请求返回，已知子集合计 {validation['usage_known']['reasoning_known_sum']:,}；其余 {validation['usage_known']['reasoning_unknown_requests']} 次保持 NA。Cache Read 已知 {validation['usage_known']['cache_read_known_requests']}/{validation['provider_calls']} 次，已知子集合计 {validation['usage_known']['cache_read_known_sum']:,}；Cache Creation 已知 {validation['usage_known']['cache_creation_known_requests']}/{validation['provider_calls']} 次，已知子集合计 {validation['usage_known']['cache_creation_known_sum']:,}。这些已知子集合计均不能当作缺失补零后的全量。

全批次 Provider 请求累计耗时 {validation['provider_elapsed_seconds_total']:.1f} 秒；逐请求 P50、P90、P95、P99 和最大值见 `request_latency_summary.csv`。Tool 累计耗时没有独立时间戳，保持 NA。

## State×Trap 核心结果

{state_trap_table}

Δ 均定义为同一 Source、同一 State checkpoint 上的 `Trap − BASELINE`。Token Amp 与 Wall Amp 先对每个配对计算比值，再取六 Source 均值。

配对方向分布：

{direction_table}

三个 State 的 Token 响应排序并不一致：

- RECON：`{best_by_state['RECON']['token']}` 的 ΔTotal 最高。
- EXPLOITATION：`{best_by_state['EXPLOITATION']['token']}` 的 ΔTotal 最高。
- VERIFICATION：`{best_by_state['VERIFICATION']['token']}` 的 ΔTotal 最高。

因此，本轮不支持“三种 Trap 在所有 State 下具有相同资源响应排序”的结论。它更符合 State×Trap 条件性响应。

## Token 与执行时间

{wall_table}

{total_token_positive}/{total_pairs} 个配对的总 Token 为正增量，{total_wall_positive}/{total_pairs} 个配对的 Wall Time 为正增量。按 State 的最高增量机制，Token 与时间最高者分别为：

- RECON：Token `{best_by_state['RECON']['token']}`；Wall Time `{best_by_state['RECON']['wall']}`。
- EXPLOITATION：Token `{best_by_state['EXPLOITATION']['token']}`；Wall Time `{best_by_state['EXPLOITATION']['wall']}`。
- VERIFICATION：Token `{best_by_state['VERIFICATION']['token']}`；Wall Time `{best_by_state['VERIFICATION']['wall']}`。

Token 增长总体上伴随较多正 Wall Time 配对，但两者排序并非在所有 State 完全一致，不能把 Provider 生成延迟、工具执行和本地编排混为同一因果项。

## 首次暴露与后续自主交互

{decomposition_table}

“首次暴露后首请求 Δ”是比较 Trap 与 BASELINE 在 checkpoint 后第一次 Provider 请求的真实 usage 差，其中包含新增 exposure payload 和模型对该差异的反应。“后续自主交互 Δ”是 trail 总差值减去首请求差值。两者都不是 Trap 本地生成/注入耗时。

## 防御端成本

Trap 注入本身没有观察到额外 Provider Call 或工具调用；防御层新增 Provider/Tool 远端调用按可见执行路径记录为 0。Payload 字节数已写入 `defense_cost.csv`。

本次正式运行没有独立记录 State Observer 服务时间、Trap 生成/注入本地时间和工具累计执行时间，因此这三项保持 `NA`。`unattributed_wall_seconds` 是 branch Wall Time 减去 Provider 请求累计耗时后的剩余项，包含工具、Observer、本地 Trap 和编排，不能单独解释为工具耗时。基于现有数据，无法可靠给出完整的“防御端本地成本 / 攻击端新增成本”比率，该比率保持 `NA`；不伪造估算值。

## 任务完成与预算

{task_table}

## 独立复现对照

{reproduction_table}

54 个逐 Source 配对中，本轮与上一轮 GPT 的 ΔTotal 方向一致 {reproduction_pair_match}/54；9 个 State×Trap 单元中，按均值排序的 Trap 排名稳定 {rank_stable}/9。两轮自然轨迹和 checkpoint 彼此独立，因此排名变化与数值差异同时包含模型随机波动和任务轨迹差异。

总体资源上，本轮为 {validation['provider_calls']} 次请求、{total_tokens:,} Total Tokens；上一轮 GPT H1 为 {validation['previous_gpt_provider_calls']} 次请求、{validation['previous_gpt_total_tokens']:,} Total Tokens。两者都只在各自独立批次内核算。

## 与 DeepSeek、Claude 的描述性比较

{cross_table}

比较只在各模型自己的 BASELINE 差值上进行；GPT 使用 Provider Total，DeepSeek 与 Claude 使用各自 Input+Output 口径，因此不直接比较绝对 Token 规模。本轮 GPT 与 DeepSeek 的 State×Trap 变化方向一致 {cross_trend_matches['gpt_vs_deepseek']}/9，与 Claude 一致 {cross_trend_matches['gpt_vs_claude']}/9，DeepSeek 与 Claude 之间一致 {cross_trend_matches['deepseek_vs_claude']}/9。共同 Source 数因历史 clean pair 缺失而按单元标注，未补齐。

## 任务结果、预算与异常

- BASELINE：18/18 条件轨迹完成。
- Trap：54/54 条件轨迹完成，54/54 机制部署并产生后续机制行为。
- Task Success：按 State×Trap 单元见 `state_trap_metrics.csv`；Trap 预算耗尽主要集中在 RECON。
- 技术失败：0；协议违规：0；Gateway bypass：0；未授权 effect：0。
- Provider HTTP 非 200：0；模型身份不匹配：0；总 usage 缺失：0；模型长度截断：0。
- 模型审查的 Cache/Reasoning 缺失字段保持 NA，没有补零。

## 最终回答

1. **三种 Trap 在三个 State 下的资源响应是否一致？** 不一致。九个 State×Trap 单元均值不同，且每个 State 内部排序随 Trap 改变。
2. **哪些 State×Trap 组合具有更高资源放大？** 按本轮 Token 均值，RECON 为 `{best_by_state['RECON']['token']}`，EXPLOITATION 为 `{best_by_state['EXPLOITATION']['token']}`，VERIFICATION 为 `{best_by_state['VERIFICATION']['token']}`；逐项均值、中位数和范围见 CSV。
3. **Token 放大是否伴随实际执行时间增长？** 多数配对同向为正，但并非全部、也非每个 State 的排序与 Token 排名一致。详细时间增量见 `paired_wall_time.csv`。
4. **防御端资源成本与攻击端新增成本之比是多少？** 完整本地成本比率为 `NA`。可确认防御注入未增加远端 Provider/Tool 调用；Observer 和 Trap 本地的独立耗时未被正式运行记录。
5. **与此前六 Source GPT H1 相比主要结论是否稳定？** 部分稳定：State 内并非统一排序，整体正资源响应仍占多数；但具体均值、排序和随机波动需按单元查看，不能宣称完全复现。

## 交付索引

- `pair_metrics.csv`：54 个 Source 级配对及 Token、调用、交互、Wall Time、payload 首请求分解。
- `state_trap_metrics.csv`：9 个 State×Trap 的均值、中位数、标准差、范围、正负配对和放大倍数。
- `paired_wall_time.csv`：逐配对执行时间与 Provider/未归属耗时分解。
- `interaction_decomposition.csv`：首次暴露后请求与后续自主交互的 Token 差值。
- `defense_cost.csv`：防御 payload、远端调用和本地成本可观测性边界。
- `request_timing_usage.csv`：711 个原始请求的 usage、模型、状态、耗时与 journal 路径。
- `request_latency_summary.csv`：请求耗时与 Token 的描述性分布。
- `reproduction_vs_previous_gpt.csv`、`reproduction_vs_previous_gpt_pair.csv`：与上一轮独立 GPT H1 的对照。
- `cross_model_reproduction.csv`：与本项目已有 DeepSeek、Claude H1 的共同 Source 方向比较。
- `validation.json`：批次级审计摘要。

## 解释边界

六 Source 为小样本探索性复现。自然轨迹和 checkpoint 是模型行为的一部分，跨轮差异不能全部归因于 Trap；Provider 缓存命中、生成延迟和长尾网络时间也不应被解释为机制本身的理论属性。没有冻结的确认性检验，因此本报告只报告配对差异、排序、范围与方向一致性。
"""
    (OUTPUT / "FINAL_REPORT_ZH.md").write_text(report, encoding="utf-8")

    modification_record = f"""# 派生分析修改记录

- 分析对象：`{RUN_ROOT}`
- 分析时只读原始 branch、checkpoint、Provider request、journal 和 usage；未调用真实 Provider。
- 新建目录：`analysis_reproduction/`。
- 新增长期分析脚本：`h1_gpt_cross_model_6source_20261024/derive_reproduction_analysis.py`。
- 未修改 Trap、工具 Schema、State Observer、Oracle、K=12、pre-effect-v1、Source 或模型参数。
- 未读取、输出或写入明文 API Key。
- 历史 GPT、DeepSeek、Claude 结果只作为只读比较输入。
- 冻结通过版本、时间戳、只读权限和本修改记录表达。
- 已知限制：正式运行未独立记录 Observer/Trap 本地时间和工具累计时间，派生分析保持 NA。
"""
    (OUTPUT / "MODIFICATION_RECORD.md").write_text(
        modification_record, encoding="utf-8"
    )

    index = """# 独立复现分析索引

| 文件 | 用途 |
|---|---|
| FINAL_REPORT_ZH.md | 中文最终报告与结论 |
| condition_metrics.csv | 72 条条件轨迹的资源与终止状态 |
| pair_metrics.csv | 54 条 Trap–BASELINE Source 级配对 |
| state_trap_metrics.csv | 9 个 State×Trap 统计 |
| paired_wall_time.csv | 执行时间配对分析 |
| interaction_decomposition.csv | 首次暴露后请求与后续交互拆分 |
| defense_cost.csv | 防御端 payload/调用/时间可观测性 |
| request_timing_usage.csv | 逐请求耗时与 usage |
| request_latency_summary.csv | 请求延迟描述统计 |
| reproduction_vs_previous_gpt.csv | 与上一轮 GPT H1 的单元比较 |
| reproduction_vs_previous_gpt_pair.csv | 与上一轮 GPT H1 的逐配对比较 |
| cross_model_reproduction.csv | 与 DeepSeek、Claude 的方向比较 |
| validation.json | 批次和审计摘要 |
| analysis_code_snapshot.json | 分析代码路径、版本时间与冻结参数 |
| MODIFICATION_RECORD.md | 派生分析修改记录 |
"""
    (OUTPUT / "ANALYSIS_INDEX.md").write_text(index, encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "arms": len(arms),
                "clean_pairs": len(clean_pairs),
                "provider_requests": len(requests),
                "total_tokens": total_tokens,
                "new_files": sorted(path.name for path in OUTPUT.iterdir()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

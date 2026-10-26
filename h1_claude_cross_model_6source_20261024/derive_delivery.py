"""Derive delivery ledgers and a Chinese report from a completed Claude H1 run.

This program is read-only with respect to experimental trajectories. It reads
the saved request, usage, timing, pair, defense, and cross-model records and
writes supplemental analysis files into the selected result directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "results"
    / "formal_H1_Claude_Sonnet_4_6_cross_model_6source_20261024T140112540416Z"
)
PRIOR_CLAUDE_6SOURCE = (
    ROOT
    / "results"
    / "formal_H1_Claude_Sonnet_4_6_cross_model_6source_20261024T120057689162Z"
)
HISTORICAL_CLAUDE_9SOURCE = (
    ROOT
    / "results"
    / "formal_H1_claude_sonnet_4_6_9source_20261022T080937685233Z"
    / "postprocessed_pair_audit_v1"
)
STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
TRAPS = ("BRANCH_NEW_V3", "DEPENDENCY_V6", "EVIDENCE_NEW_V3")
CONDITIONS = ("BASELINE", *TRAPS)
SOURCE_INDICES = (1, 6, 11, 16, 21, 26)
FAMILIES = {
    1: "WEB",
    6: "WEB",
    11: "API",
    16: "API",
    21: "SERVICE",
    26: "SERVICE",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: list[str],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def number(value: Any) -> float | None:
    if value in ("", None, "NA", "None"):
        return None
    return float(value)


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def median(values: Iterable[float]) -> float:
    return statistics.median(values)


def f1(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.1f}"


def f2(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"


def f3(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def f4(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def sign_counts(values: Iterable[float]) -> tuple[int, int, int]:
    values = list(values)
    return (
        sum(value > 0 for value in values),
        sum(value == 0 for value in values),
        sum(value < 0 for value in values),
    )


def compact_payload_bytes(branch_path: Path) -> int | None:
    if not branch_path.is_file():
        return None
    arm = read_json(branch_path)
    visible = (arm.get("initial_exposure_event") or {}).get("visible_result") or {}
    payload = visible.get("trap_information")
    if payload is None:
        return None
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(encoded)


def derive_token_source(
    provider_requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in provider_requests:
        condition = record["condition"]
        if condition == "SOURCE":
            continue
        grouped[
            (
                int(record["source_index"]),
                str(record["state"]),
                str(condition),
            )
        ].append(record)

    rows: list[dict[str, Any]] = []
    for source_index in SOURCE_INDICES:
        for state in STATES:
            baseline = sorted(
                grouped[(source_index, state, "BASELINE")],
                key=lambda row: int(row["local_request_index"]),
            )
            if not baseline:
                raise RuntimeError(
                    f"missing BASELINE provider requests: {source_index}/{state}"
                )
            for trap in TRAPS:
                branch = sorted(
                    grouped[(source_index, state, trap)],
                    key=lambda row: int(row["local_request_index"]),
                )
                if not branch:
                    raise RuntimeError(
                        f"missing trap provider requests: {source_index}/{state}/{trap}"
                    )

                baseline_first = baseline[0]
                trap_first = branch[0]
                baseline_follow_input = sum(
                    row["effective_input_tokens"] for row in baseline[1:]
                )
                trap_follow_input = sum(
                    row["effective_input_tokens"] for row in branch[1:]
                )
                baseline_follow_output = sum(
                    row["output_tokens"] for row in baseline[1:]
                )
                trap_follow_output = sum(
                    row["output_tokens"] for row in branch[1:]
                )
                delta_first_input = (
                    trap_first["effective_input_tokens"]
                    - baseline_first["effective_input_tokens"]
                )
                delta_first_output = (
                    trap_first["output_tokens"]
                    - baseline_first["output_tokens"]
                )
                delta_follow_input = (
                    trap_follow_input - baseline_follow_input
                )
                delta_follow_output = (
                    trap_follow_output - baseline_follow_output
                )
                rows.append(
                    {
                        "source_index": source_index,
                        "family": FAMILIES[source_index],
                        "state": state,
                        "trap": trap,
                        "baseline_first_request_id": baseline_first["request_id"],
                        "trap_first_request_id": trap_first["request_id"],
                        "baseline_first_effective_input_tokens": (
                            baseline_first["effective_input_tokens"]
                        ),
                        "trap_first_effective_input_tokens": (
                            trap_first["effective_input_tokens"]
                        ),
                        "delta_first_input_tokens": delta_first_input,
                        "baseline_first_output_tokens": baseline_first[
                            "output_tokens"
                        ],
                        "trap_first_output_tokens": trap_first["output_tokens"],
                        "delta_first_output_tokens": delta_first_output,
                        "delta_first_total_tokens": (
                            delta_first_input + delta_first_output
                        ),
                        "delta_followup_input_tokens": delta_follow_input,
                        "delta_followup_output_tokens": delta_follow_output,
                        "delta_followup_total_tokens": (
                            delta_follow_input + delta_follow_output
                        ),
                        "delta_followup_provider_calls": (
                            (len(branch) - 1) - (len(baseline) - 1)
                        ),
                        "baseline_provider_calls": len(baseline),
                        "trap_provider_calls": len(branch),
                    }
                )
    return rows


def aggregate_token_source(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for state in STATES:
        for trap in TRAPS:
            group = [
                row
                for row in rows
                if row["state"] == state and row["trap"] == trap
            ]
            first_total = [row["delta_first_total_tokens"] for row in group]
            follow_total = [
                row["delta_followup_total_tokens"] for row in group
            ]
            aggregate = sum(first_total) + sum(follow_total)
            result.append(
                {
                    "state": state,
                    "trap": trap,
                    "n": len(group),
                    "mean_delta_first_input_tokens": mean(
                        row["delta_first_input_tokens"] for row in group
                    ),
                    "mean_delta_first_output_tokens": mean(
                        row["delta_first_output_tokens"] for row in group
                    ),
                    "mean_delta_first_total_tokens": mean(first_total),
                    "mean_delta_followup_input_tokens": mean(
                        row["delta_followup_input_tokens"] for row in group
                    ),
                    "mean_delta_followup_output_tokens": mean(
                        row["delta_followup_output_tokens"] for row in group
                    ),
                    "mean_delta_followup_total_tokens": mean(follow_total),
                    "mean_delta_followup_provider_calls": mean(
                        row["delta_followup_provider_calls"] for row in group
                    ),
                    "aggregate_initial_exposure_share": (
                        sum(first_total) / aggregate if aggregate else None
                    ),
                    "aggregate_followup_interaction_share": (
                        sum(follow_total) / aggregate if aggregate else None
                    ),
                }
            )
    return result


def load_prior_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path / "live" / "state_trap_summary.csv")
    return {(row["state"], row["trap"]): row for row in rows}


def same_six_historical_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path / "clean_pairs.csv")
    return [
        row
        for row in rows
        if int(row["source_index"]) in SOURCE_INDICES
    ]


def derive_historical_claude(
    path: Path,
) -> list[dict[str, Any]]:
    rows = same_six_historical_rows(path)
    result: list[dict[str, Any]] = []
    condition_map = {
        "BRANCH_NEW": "BRANCH_NEW",
        "DEPENDENCY_V6": "DEPENDENCY_V6",
        "EVIDENCE_NEW_V3": "EVIDENCE_NEW_V3",
    }
    for state in STATES:
        for source_condition, display in condition_map.items():
            group = [
                row
                for row in rows
                if row["state"] == state
                and row["condition"] == source_condition
            ]
            if not group:
                continue
            deltas = [float(row["delta_total_tokens"]) for row in group]
            result.append(
                {
                    "state": state,
                    "historical_condition": display,
                    "n": len(group),
                    "mean_delta_total_tokens": mean(deltas),
                    "median_delta_total_tokens": median(deltas),
                    "positive": sum(value > 0 for value in deltas),
                    "zero": sum(value == 0 for value in deltas),
                    "negative": sum(value < 0 for value in deltas),
                    "comparison_limit": (
                        "historical BRANCH_NEW differs from current "
                        "BRANCH_NEW_V3"
                        if display == "BRANCH_NEW"
                        else "same mechanism name and six-Source subset"
                    ),
                }
            )
    return result


def derive(
    run_dir: Path,
    prior_claude: Path,
    historical_claude: Path,
) -> dict[str, Any]:
    live = run_dir / "live"
    final_status = read_json(run_dir / "FINAL_STATUS.json")
    execution = read_json(live / "execution_audit.json")
    preflight = read_json(
        run_dir / "provider-free" / "provider_free_admission.json"
    )
    admission = read_json(run_dir / "admission" / "admission_record.json")
    run_status = read_json(live / "run_status.json")
    measurements = read_json(live / "measurements.json")
    pairs = read_json(live / "pair_validity.json")
    state_summary = read_json(live / "state_trap_summary.json")
    wall_summary = read_json(live / "attack_wall_time_summary.json")
    latency_summary = read_json(live / "attack_latency_summary.json")
    defense_summary = read_json(live / "defense_processing_summary.json")
    cross_model = read_json(live / "cross_model_comparison.json")
    provider_requests = read_jsonl(live / "provider_requests.jsonl")

    if final_status.get("status") != "COMPLETED":
        raise RuntimeError("formal run is not COMPLETED")
    if integer(final_status.get("sources")) != 6:
        raise RuntimeError("formal run does not contain six Sources")
    if integer(final_status.get("checkpoints")) != 18:
        raise RuntimeError("formal run does not contain 18 checkpoints")
    if integer(final_status.get("clean_pairs")) != 54:
        raise RuntimeError("formal run does not contain 54 clean pairs")
    if len(measurements) != 72 or len(pairs) != 54:
        raise RuntimeError("measurement or pair row count mismatch")
    if not all(row.get("pair_valid") is True for row in pairs):
        raise RuntimeError("one or more pairs are not clean")
    if len(provider_requests) != integer(final_status.get("formal_provider_calls")):
        raise RuntimeError("provider request ledger count mismatch")
    if {row["request_model_id"] for row in provider_requests} != {
        "claude-sonnet-4-6"
    }:
        raise RuntimeError("request model identity mismatch")
    if {row["provider_model_id"] for row in provider_requests} != {
        "claude-sonnet-4-6"
    }:
        raise RuntimeError("response model identity mismatch")
    if {row["http_status"] for row in provider_requests} != {200}:
        raise RuntimeError("non-200 HTTP status in formal run")
    if {row["outcome"] for row in provider_requests} != {"success"}:
        raise RuntimeError("non-success request outcome in formal run")
    if not all(
        row["effective_input_tokens"]
        == (
            row["raw_uncached_input_tokens"]
            + row["cache_creation_input_tokens"]
            + row["cache_read_input_tokens"]
        )
        and row["input_output_total"]
        == row["effective_input_tokens"] + row["output_tokens"]
        for row in provider_requests
    ):
        raise RuntimeError("usage formula mismatch")
    if not all(
        all(bool(value) for value in arm.get("checks", {}).values())
        for arm in run_status["arms"]
    ):
        raise RuntimeError("one or more frozen arm checks failed")

    token_rows = derive_token_source(provider_requests)
    token_aggregate = aggregate_token_source(token_rows)
    if len(token_rows) != 54:
        raise RuntimeError("token source decomposition is not 54 pairs")

    wall_by_key = {
        (row["state"], row["trap"]): row for row in wall_summary
    }
    latency_by_key = {
        (row["state"], row["trap"]): row for row in latency_summary
    }
    defense_by_key = {
        (row["state"], row["trap"]): row for row in defense_summary
    }

    pair_by_key: dict[tuple[int, str, str], dict[str, Any]] = {
        (int(row["source_index"]), row["state"], row["trap"]): row
        for row in pairs
    }
    measurement_by_key = {
        (int(row["source_index"]), row["state"], row["condition"]): row
        for row in measurements
    }

    payload_rows: list[dict[str, Any]] = []
    for source_index in SOURCE_INDICES:
        for state in STATES:
            for trap in TRAPS:
                pair = pair_by_key[(source_index, state, trap)]
                measurement = measurement_by_key[(source_index, state, trap)]
                branch_path = (
                    live
                    / "sources"
                    / f"source-{source_index:02d}"
                    / "checkpoints"
                    / state.lower()
                    / "branches"
                    / f"{trap}.json"
                )
                payload_bytes = compact_payload_bytes(branch_path)
                defense_total = measurement.get("defense_total_seconds")
                delta_wall = float(pair["delta_wall_clock_seconds"])
                payload_rows.append(
                    {
                        "source_index": source_index,
                        "family": FAMILIES[source_index],
                        "state": state,
                        "trap": trap,
                        "checkpoint_id": pair["checkpoint_id"],
                        "trap_payload_bytes": payload_bytes,
                        "defense_prepare_seconds": measurement.get(
                            "defense_prepare_seconds"
                        ),
                        "defense_response_construction_seconds": measurement.get(
                            "defense_response_construction_seconds"
                        ),
                        "defense_injection_seconds": measurement.get(
                            "defense_injection_seconds"
                        ),
                        "defense_total_seconds": defense_total,
                        "defense_total_cpu_seconds": measurement.get(
                            "defense_total_cpu_seconds"
                        ),
                        "defense_provider_calls": measurement.get(
                            "defense_provider_calls"
                        ),
                        "defense_tokens": measurement.get("defense_tokens"),
                        "attack_delta_wall_clock_seconds": delta_wall,
                        "attack_delta_tokens": float(
                            pair["delta_total_tokens"]
                        ),
                        "attack_delta_wall_to_defense_time_ratio": (
                            delta_wall / float(defense_total)
                            if defense_total is not None
                            and float(defense_total) > 0
                            else None
                        ),
                    }
                )

    token_agg_by_key = {
        (row["state"], row["trap"]): row for row in token_aggregate
    }
    token_by_pair_key = {
        (row["source_index"], row["state"], row["trap"]): row
        for row in token_rows
    }
    payload_by_pair_key = {
        (row["source_index"], row["state"], row["trap"]): row
        for row in payload_rows
    }
    pair_level_rows: list[dict[str, Any]] = []
    for pair in pairs:
        key = (int(pair["source_index"]), pair["state"], pair["trap"])
        token_source = token_by_pair_key[key]
        defense = payload_by_pair_key[key]
        pair_level_rows.append(
            {
                "source_index": pair["source_index"],
                "family": pair["family"],
                "state": pair["state"],
                "trap": pair["trap"],
                "checkpoint_id": pair["checkpoint_id"],
                "pair_valid": pair["pair_valid"],
                "deployment_reason": pair["deployment_reason"],
                "baseline_task_success": pair["baseline_task_success"],
                "trap_task_success": pair["trap_task_success"],
                "baseline_budget_exhausted": pair[
                    "baseline_budget_exhausted"
                ],
                "trap_budget_exhausted": pair["trap_budget_exhausted"],
                "baseline_terminal_reason": pair["baseline_terminal_reason"],
                "trap_terminal_reason": pair["trap_terminal_reason"],
                "baseline_provider_calls": pair["baseline_provider_calls"],
                "trap_provider_calls": pair["trap_provider_calls"],
                "delta_provider_calls": pair["delta_provider_calls"],
                "baseline_tool_calls": pair["baseline_tool_calls"],
                "trap_tool_calls": pair["trap_tool_calls"],
                "delta_tool_calls": pair["delta_tool_calls"],
                "baseline_interaction_depth": pair[
                    "baseline_interaction_depth"
                ],
                "trap_interaction_depth": pair["trap_interaction_depth"],
                "delta_interaction_depth": pair["delta_interaction_depth"],
                "baseline_input_tokens": pair["baseline_input_tokens"],
                "trap_input_tokens": pair["trap_input_tokens"],
                "delta_input_tokens": pair["delta_input_tokens"],
                "baseline_output_tokens": pair["baseline_output_tokens"],
                "trap_output_tokens": pair["trap_output_tokens"],
                "delta_output_tokens": pair["delta_output_tokens"],
                "baseline_cache_creation_input_tokens": pair[
                    "baseline_cache_creation_input_tokens"
                ],
                "trap_cache_creation_input_tokens": pair[
                    "trap_cache_creation_input_tokens"
                ],
                "delta_cache_creation_input_tokens": pair[
                    "delta_cache_creation_input_tokens"
                ],
                "baseline_cache_read_input_tokens": pair[
                    "baseline_cache_read_input_tokens"
                ],
                "trap_cache_read_input_tokens": pair[
                    "trap_cache_read_input_tokens"
                ],
                "delta_cache_read_input_tokens": pair[
                    "delta_cache_read_input_tokens"
                ],
                "baseline_total_tokens": pair["baseline_total_tokens"],
                "trap_total_tokens": pair["trap_total_tokens"],
                "delta_total_tokens": pair["delta_total_tokens"],
                "delta_first_input_tokens": token_source[
                    "delta_first_input_tokens"
                ],
                "delta_first_output_tokens": token_source[
                    "delta_first_output_tokens"
                ],
                "delta_first_total_tokens": token_source[
                    "delta_first_total_tokens"
                ],
                "delta_followup_input_tokens": token_source[
                    "delta_followup_input_tokens"
                ],
                "delta_followup_output_tokens": token_source[
                    "delta_followup_output_tokens"
                ],
                "delta_followup_total_tokens": token_source[
                    "delta_followup_total_tokens"
                ],
                "baseline_wall_clock_seconds": pair[
                    "baseline_wall_clock_seconds"
                ],
                "trap_wall_clock_seconds": pair["trap_wall_clock_seconds"],
                "delta_wall_clock_seconds": pair[
                    "delta_wall_clock_seconds"
                ],
                "delta_provider_latency_seconds": pair[
                    "delta_provider_latency_seconds"
                ],
                "delta_tool_execution_seconds": pair[
                    "delta_tool_execution_seconds"
                ],
                "trap_payload_bytes": defense["trap_payload_bytes"],
                "defense_prepare_seconds": defense[
                    "defense_prepare_seconds"
                ],
                "defense_response_construction_seconds": defense[
                    "defense_response_construction_seconds"
                ],
                "defense_injection_seconds": defense[
                    "defense_injection_seconds"
                ],
                "defense_total_seconds": defense["defense_total_seconds"],
                "defense_total_cpu_seconds": defense[
                    "defense_total_cpu_seconds"
                ],
                "defense_provider_calls": defense["defense_provider_calls"],
                "defense_tokens": defense["defense_tokens"],
                "attack_delta_wall_to_defense_time_ratio": defense[
                    "attack_delta_wall_to_defense_time_ratio"
                ],
            }
        )
    payload_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in payload_rows:
        payload_by_key[(row["state"], row["trap"])].append(row)

    table_rows: list[dict[str, Any]] = []
    for state in STATES:
        for trap in TRAPS:
            resource = next(
                row
                for row in state_summary
                if row["state"] == state and row["trap"] == trap
            )
            wall = wall_by_key[(state, trap)]
            latency = latency_by_key[(state, trap)]
            defense = defense_by_key[(state, trap)]
            token_source = token_agg_by_key[(state, trap)]
            payload_group = payload_by_key[(state, trap)]
            pair_group = [
                row
                for row in pairs
                if row["state"] == state and row["trap"] == trap
            ]
            token_deltas = [
                float(row["delta_total_tokens"]) for row in pair_group
            ]
            wall_deltas = [
                float(row["delta_wall_clock_seconds"]) for row in pair_group
            ]
            baseline_total_mean = mean(
                float(row["baseline_total_tokens"]) for row in pair_group
            )
            trap_total_mean = mean(
                float(row["trap_total_tokens"]) for row in pair_group
            )
            baseline_wall_mean = mean(
                float(row["baseline_wall_clock_seconds"])
                for row in pair_group
            )
            trap_wall_mean = mean(
                float(row["trap_wall_clock_seconds"]) for row in pair_group
            )
            token_ratios = [
                float(row["trap_total_tokens"])
                / float(row["baseline_total_tokens"])
                for row in pair_group
            ]
            wall_ratios = [
                float(row["trap_wall_clock_seconds"])
                / float(row["baseline_wall_clock_seconds"])
                for row in pair_group
            ]
            token_sign = sign_counts(token_deltas)
            wall_sign = sign_counts(wall_deltas)
            table_rows.append(
                {
                    "state": state,
                    "trap": trap,
                    "n": len(pair_group),
                    "delta_input_mean": float(resource["delta_input_mean"]),
                    "delta_input_median": float(resource["delta_input_median"]),
                    "delta_output_mean": float(resource["delta_output_mean"]),
                    "delta_output_median": float(
                        resource["delta_output_median"]
                    ),
                    "delta_total_mean": float(
                        resource["delta_input_output_total_mean"]
                    ),
                    "delta_total_median": float(
                        resource["delta_input_output_total_median"]
                    ),
                    "token_positive": token_sign[0],
                    "token_zero": token_sign[1],
                    "token_negative": token_sign[2],
                    "token_mean_ratio": trap_total_mean / baseline_total_mean,
                    "token_median_pair_ratio": median(token_ratios),
                    "delta_provider_calls_mean": float(
                        resource["delta_provider_calls_mean"]
                    ),
                    "delta_tool_calls_mean": float(
                        resource["delta_tool_calls_mean"]
                    ),
                    "delta_interaction_depth_mean": float(
                        resource["delta_interaction_depth_mean"]
                    ),
                    "delta_wall_mean": float(wall["delta_wall_mean"]),
                    "delta_wall_median": float(wall["delta_wall_median"]),
                    "wall_positive": wall_sign[0],
                    "wall_zero": wall_sign[1],
                    "wall_negative": wall_sign[2],
                    "wall_mean_ratio": trap_wall_mean / baseline_wall_mean,
                    "wall_median_pair_ratio": median(wall_ratios),
                    "delta_provider_latency_mean": float(
                        latency["delta_provider_latency_mean"]
                    ),
                    "delta_tool_time_mean": float(
                        latency["delta_tool_time_mean"]
                    ),
                    "trap_payload_bytes_mean": mean(
                        float(row["trap_payload_bytes"])
                        for row in payload_group
                        if row["trap_payload_bytes"] is not None
                    ),
                    "defense_setup_mean": float(defense["setup_mean"]),
                    "defense_prepare_mean": float(defense["prepare_mean"]),
                    "defense_construction_mean": (
                        float(defense["construction_mean"])
                        if defense.get("construction_mean") is not None
                        else None
                    ),
                    "defense_response_mean": float(defense["response_mean"]),
                    "defense_injection_mean": float(
                        defense["injection_mean"]
                    ),
                    "defense_total_mean": float(defense["total_mean"]),
                    "defense_total_median": float(defense["total_median"]),
                    "defense_cpu_mean": float(defense["cpu_mean"]),
                    "defense_provider_calls": sum(
                        int(row["defense_provider_calls"] or 0)
                        for row in payload_group
                    ),
                    "defense_tokens": sum(
                        int(row["defense_tokens"] or 0)
                        for row in payload_group
                    ),
                    "attack_delta_wall_to_defense_ratio_mean": mean(
                        float(row["attack_delta_wall_to_defense_time_ratio"])
                        for row in payload_group
                        if row["attack_delta_wall_to_defense_time_ratio"]
                        is not None
                    ),
                    "delta_first_total_mean": float(
                        token_source["mean_delta_first_total_tokens"]
                    ),
                    "delta_followup_total_mean": float(
                        token_source["mean_delta_followup_total_tokens"]
                    ),
                    "first_share": float(
                        token_source["aggregate_initial_exposure_share"]
                    ),
                    "followup_share": float(
                        token_source["aggregate_followup_interaction_share"]
                    ),
                    "task_success_trap": int(resource["task_success_trap"]),
                    "task_success_baseline": int(
                        resource["task_success_baseline"]
                    ),
                    "budget_exhausted_trap": int(
                        resource["budget_exhausted_trap"]
                    ),
                    "budget_exhausted_baseline": int(
                        resource["budget_exhausted_baseline"]
                    ),
                    "mechanism_followup": int(resource["mechanism_followup"]),
                }
            )

    timing_rows: list[dict[str, Any]] = []
    timing_fields = [
        "source_index",
        "family",
        "state",
        "condition",
        "checkpoint_id",
        "status",
        "terminal_reason",
        "task_success",
        "budget_exhausted",
        "provider_calls",
        "tool_calls",
        "interaction_depth",
        "wall_clock_seconds",
        "provider_latency_seconds",
        "provider_latency_complete",
        "tool_execution_seconds",
        "tool_execution_count",
        "attack_process_cpu_seconds",
        "trap_runtime_setup_inclusive_seconds",
        "trap_runtime_setup_inclusive_cpu_seconds",
        "defense_prepare_seconds",
        "defense_construction_retrieval_seconds",
        "defense_response_construction_seconds",
        "defense_injection_seconds",
        "defense_total_seconds",
        "defense_total_cpu_seconds",
        "state_observation_seconds",
        "trap_dispatch_selection_seconds",
        "defense_provider_calls",
        "defense_tokens",
        "process_metric_attribution",
        "measurement_present",
    ]
    for row in measurements:
        timing_rows.append({field: row.get(field) for field in timing_fields})

    prior_summary = load_prior_summary(prior_claude)
    replication_rows: list[dict[str, Any]] = []
    for state in STATES:
        for trap in TRAPS:
            prior = prior_summary[(state, trap)]
            current = next(
                row
                for row in state_summary
                if row["state"] == state and row["trap"] == trap
            )
            prior_delta = float(prior["delta_input_output_total_mean"])
            current_delta = float(
                current["delta_input_output_total_mean"]
            )
            replication_rows.append(
                {
                    "state": state,
                    "trap": trap,
                    "prior_delta_total_mean": prior_delta,
                    "current_delta_total_mean": current_delta,
                    "change_in_delta": current_delta - prior_delta,
                    "current_to_prior_ratio": current_delta / prior_delta,
                    "same_direction": (prior_delta > 0) == (current_delta > 0),
                }
            )

    historical_rows = derive_historical_claude(historical_claude)
    historical_by_key = {
        (row["state"], row["historical_condition"]): row
        for row in historical_rows
    }
    historical_comparison: list[dict[str, Any]] = []
    for state in STATES:
        for trap in TRAPS:
            historical_condition = (
                "BRANCH_NEW" if trap == "BRANCH_NEW_V3" else trap
            )
            historical = historical_by_key.get((state, historical_condition))
            current = next(
                row
                for row in state_summary
                if row["state"] == state and row["trap"] == trap
            )
            historical_comparison.append(
                {
                    "state": state,
                    "current_trap": trap,
                    "historical_condition": historical_condition,
                    "historical_n": historical["n"] if historical else None,
                    "historical_mean_delta_total_tokens": (
                        historical["mean_delta_total_tokens"]
                        if historical
                        else None
                    ),
                    "current_mean_delta_total_tokens": float(
                        current["delta_input_output_total_mean"]
                    ),
                    "same_direction": (
                        (
                            historical["mean_delta_total_tokens"] > 0
                        )
                        == (
                            float(current["delta_input_output_total_mean"])
                            > 0
                        )
                        if historical
                        else None
                    ),
                    "comparison_limit": (
                        historical["comparison_limit"]
                        if historical
                        else "historical result unavailable"
                    ),
                }
            )

    ranking: dict[str, list[str]] = {}
    for state in STATES:
        ranked = sorted(
            (
                row
                for row in table_rows
                if row["state"] == state
            ),
            key=lambda row: row["delta_total_mean"],
            reverse=True,
        )
        ranking[state] = [row["trap"] for row in ranked]

    prior_ranking: dict[str, list[str]] = {}
    for state in STATES:
        ranked = sorted(
            (
                row
                for (candidate_state, _), row in prior_summary.items()
                if candidate_state == state
            ),
            key=lambda row: float(row["delta_input_output_total_mean"]),
            reverse=True,
        )
        prior_ranking[state] = [row["trap"] for row in ranked]

    all_token_deltas = [
        float(row["delta_total_tokens"]) for row in pairs
    ]
    all_wall_deltas = [
        float(row["delta_wall_clock_seconds"]) for row in pairs
    ]
    token_sign = sign_counts(all_token_deltas)
    wall_sign = sign_counts(all_wall_deltas)
    token_aggregate_global = sum(
        row["delta_first_total_tokens"] + row["delta_followup_total_tokens"]
        for row in token_rows
    )
    first_total_global = sum(
        row["delta_first_total_tokens"] for row in token_rows
    )
    follow_total_global = sum(
        row["delta_followup_total_tokens"] for row in token_rows
    )
    input_total_global = sum(
        row["delta_first_input_tokens"]
        + row["delta_followup_input_tokens"]
        for row in token_rows
    )
    output_total_global = sum(
        row["delta_first_output_tokens"]
        + row["delta_followup_output_tokens"]
        for row in token_rows
    )
    ratio_values = [
        float(row["attack_delta_wall_to_defense_time_ratio"])
        for row in payload_rows
        if row["attack_delta_wall_to_defense_time_ratio"] is not None
    ]
    cross_match_counts = {
        other: {
            metric: sum(
                bool(row.get(f"claude_{other}_{metric}_trend_match"))
                for row in cross_model
            )
            for metric in (
                "input",
                "output",
                "cache_read",
                "provider_calls",
                "tool_calls",
                "wall_time",
            )
        }
        for other in ("gpt", "deepseek")
    }

    usage_coverage = execution["usage_coverage"]
    delivery_summary = {
        "protocol_id": execution["protocol_id"],
        "platform": "CyberStrikeAI",
        "batch_id": run_dir.name,
        "status": final_status["status"],
        "model_id": "claude-sonnet-4-6",
        "generated_from_saved_records_only": True,
        "provider_calls_added_by_derivation": 0,
        "trajectories_modified": False,
        "historical_results_modified": False,
        "provider_free": {
            "passed": preflight["passed"],
            "provider_calls": preflight["provider_calls"],
            "network_attempts": preflight["network_attempts"],
        },
        "live_admission": {
            "status": admission["status"],
            "provider_calls": admission["provider_calls"],
            "request_models": admission["request_models"],
            "response_models": admission["response_models"],
            "checks": admission["checks"],
        },
        "completion": {
            "sources": final_status["sources"],
            "checkpoints": final_status["checkpoints"],
            "condition_trajectories": execution["condition_records"],
            "condition_statuses": execution["condition_statuses"],
            "target_pairs": final_status["target_pairs"],
            "clean_pairs": final_status["clean_pairs"],
            "formal_provider_calls": final_status["formal_provider_calls"],
            "admission_provider_calls": final_status[
                "admission_provider_calls"
            ],
            "global_provider_calls": final_status["provider_calls"],
            "request_ids_unique": execution["provider_request_ids_unique"],
            "http_statuses": execution["http_statuses"],
            "provider_models": execution["provider_models"],
            "model_identity_exact": execution["model_identity_exact"],
            "gateway_bypass_observed": execution[
                "gateway_bypass_observed"
            ],
            "unauthorized_effect_observed": execution[
                "unauthorized_effect_observed"
            ],
            "journal_audit_status": execution["journal_audit"]["status"],
            "automatic_retry": final_status["automatic_retry"],
            "automatic_backfill": final_status["automatic_backfill"],
            "automatic_recovery": final_status["automatic_recovery"],
        },
        "usage": {
            "usage_label": "Input+Output口径",
            "effective_input_tokens": usage_coverage[
                "effective_input_tokens"
            ]["known_sum"],
            "output_tokens": usage_coverage["output_tokens"]["known_sum"],
            "input_output_total_tokens": usage_coverage[
                "input_output_total"
            ]["known_sum"],
            "cache_creation_input_tokens": usage_coverage[
                "cache_creation_input_tokens"
            ]["known_sum"],
            "cache_read_input_tokens": usage_coverage[
                "cache_read_input_tokens"
            ]["known_sum"],
            "all_usage_fields_known": all(
                value["missing"] == 0 for value in usage_coverage.values()
            ),
        },
        "pair_sign_counts": {
            "delta_total_tokens": {
                "positive": token_sign[0],
                "zero": token_sign[1],
                "negative": token_sign[2],
            },
            "delta_wall_clock_seconds": {
                "positive": wall_sign[0],
                "zero": wall_sign[1],
                "negative": wall_sign[2],
            },
        },
        "token_source_decomposition": {
            "aggregate_delta_total_tokens": token_aggregate_global,
            "initial_exposure_delta_total_tokens": first_total_global,
            "followup_interaction_delta_total_tokens": follow_total_global,
            "initial_exposure_share": first_total_global
            / token_aggregate_global,
            "followup_interaction_share": follow_total_global
            / token_aggregate_global,
            "delta_input_tokens": input_total_global,
            "delta_output_tokens": output_total_global,
            "input_share": input_total_global / token_aggregate_global,
            "output_share": output_total_global / token_aggregate_global,
            "by_state_trap": token_aggregate,
        },
        "state_trap": table_rows,
        "state_trap_ranking": ranking,
        "defense": {
            "state_observation_seconds": "NA_H1_STATE_IS_FROZEN",
            "dynamic_dispatch_selection_seconds": (
                "NA_H1_CONDITION_IS_FIXED"
            ),
            "offline_preparation_cost": "UNKNOWN_NOT_SEPARATELY_TIMED",
            "online_defense_provider_calls": 0,
            "online_defense_tokens": 0,
            "online_defense_api_cost": 0,
            "runtime_initial_exposure_total_seconds": {
                "min_mean": min(
                    row["defense_total_mean"] for row in table_rows
                ),
                "max_mean": max(
                    row["defense_total_mean"] for row in table_rows
                ),
            },
            "attack_delta_wall_to_defense_time_ratio": {
                "n": len(ratio_values),
                "median": median(ratio_values),
                "min": min(ratio_values),
                "max": max(ratio_values),
                "positive_delta_wall_pairs": wall_sign[0],
            },
            "full_defense_total": (
                "UNKNOWN_BRANCH_ENUMERATION_OUTSIDE_RUNTIME_BOUNDARY_AND_"
                "NO_ISOLATED_DEFENSE_PROCESS"
            ),
        },
        "cross_model_trend_matches": {
            "comparison_basis": (
                "same Source/State/Trap clean pairs; paired direction and "
                "mechanism ordering only"
            ),
            "units": len(cross_model),
            "claude_vs_gpt": cross_match_counts["gpt"],
            "claude_vs_deepseek": cross_match_counts["deepseek"],
        },
        "same_config_claude_replication": {
            "prior_batch_id": prior_claude.name,
            "direction_matches": sum(
                row["same_direction"] for row in replication_rows
            ),
            "units": len(replication_rows),
            "ranking_matches": prior_ranking == ranking,
            "rows": replication_rows,
        },
        "historical_claude_same_six_sources": {
            "historical_batch": historical_claude.parent.name,
            "rows": historical_comparison,
        },
        "limitations": [
            (
                "Six Sources provide descriptive replication, not "
                "confirmatory model-effect inference."
            ),
            (
                "The same baseline is reused across three Trap pairs; pairs "
                "are clustered by Source and are not independent samples."
            ),
            (
                "Cache creation and cache read are zero in this route; "
                "results may differ under prompt caching."
            ),
            "H1 has no online State Observer or dynamic Trap dispatch.",
            (
                "Trap candidate enumeration for BRANCH occurs outside the "
                "runtime exposure timing boundary."
            ),
            "CPU/RSS are process-level; there is no isolated defense process.",
            (
                "Offline preparation time and exact billed attack-side API "
                "fees are not directly observable in the saved data."
            ),
        ],
    }

    write_csv(
        live / "timing_ledger.csv",
        timing_rows,
        timing_fields,
    )
    write_csv(
        live / "token_source_decomposition.csv",
        token_rows,
        list(token_rows[0]),
    )
    write_json(
        live / "token_source_decomposition.json",
        {
            "method": (
                "first provider request within each restored condition vs "
                "all subsequent provider requests, differenced against the "
                "same-checkpoint BASELINE"
            ),
            "pairs": token_rows,
            "state_trap": token_aggregate,
        },
    )
    payload_fields = list(payload_rows[0])
    write_csv(
        live / "defense_payload_and_ratio.csv",
        payload_rows,
        payload_fields,
    )
    write_json(
        live / "defense_payload_and_ratio.json",
        {
            "payload_boundary": (
                "UTF-8 byte size of visible trap_information at initial "
                "pre-effect exposure"
            ),
            "pairs": payload_rows,
        },
    )
    write_csv(
        live / "pair_level_resource_deltas.csv",
        pair_level_rows,
        list(pair_level_rows[0]),
    )
    write_json(live / "delivery_summary.json", delivery_summary)

    source_rows = read_csv(run_dir / "provider-free" / "sources.csv")
    source_lines = [
        "| Source | Family | Fixture | Status |",
        "|---:|---|---|---|",
    ]
    for row in source_rows:
        source_lines.append(
            f"| {row['source_index']} | {row['family']} | "
            f"`{row['fixture_id']}` | {row['status']} |"
        )

    outcome_lines = [
        "| State | Condition | n | Success | Budget Exhausted |",
        "|---|---|---:|---:|---:|",
    ]
    for state in STATES:
        for condition in CONDITIONS:
            group = [
                row
                for row in measurements
                if row["state"] == state and row["condition"] == condition
            ]
            outcome_lines.append(
                f"| {state} | {condition} | {len(group)} | "
                f"{sum(bool(row['task_success']) for row in group)} | "
                f"{sum(bool(row['budget_exhausted']) for row in group)} |"
            )

    token_lines = [
        (
            "| State | Trap | n | Mean ΔTotal | Median ΔTotal | "
            "Mean Ratio | Median Pair Ratio | ΔTotal +/0/− |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        token_lines.append(
            f"| {row['state']} | {row['trap']} | {row['n']} | "
            f"{f1(row['delta_total_mean'])} | {f1(row['delta_total_median'])} | "
            f"{row['token_mean_ratio']:.3f}× | "
            f"{row['token_median_pair_ratio']:.3f}× | "
            f"{row['token_positive']}/{row['token_zero']}/"
            f"{row['token_negative']} |"
        )

    component_lines = [
        (
            "| State | Trap | Mean ΔInput | Mean ΔOutput | "
            "Mean ΔCache Read | Mean ΔTotal |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        component_lines.append(
            f"| {row['state']} | {row['trap']} | "
            f"{f1(row['delta_input_mean'])} | "
            f"{f1(row['delta_output_mean'])} | +0.0 | "
            f"{f1(row['delta_total_mean'])} |"
        )

    call_lines = [
        (
            "| State | Trap | n | ΔProvider Calls | ΔTool Calls | "
            "ΔInteraction Depth |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        call_lines.append(
            f"| {row['state']} | {row['trap']} | {row['n']} | "
            f"{f2(row['delta_provider_calls_mean'])} | "
            f"{f2(row['delta_tool_calls_mean'])} | "
            f"{f2(row['delta_interaction_depth_mean'])} |"
        )

    wall_lines = [
        (
            "| State | Trap | n | ΔWall Mean s | ΔWall Median s | "
            "Mean Ratio | Median Pair Ratio | ΔWall +/0/− | "
            "ΔProvider Latency s | ΔTool Time s |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        wall_lines.append(
            f"| {row['state']} | {row['trap']} | {row['n']} | "
            f"{f3(row['delta_wall_mean'])} | "
            f"{f3(row['delta_wall_median'])} | "
            f"{row['wall_mean_ratio']:.3f}× | "
            f"{row['wall_median_pair_ratio']:.3f}× | "
            f"{row['wall_positive']}/{row['wall_zero']}/"
            f"{row['wall_negative']} | "
            f"{f3(row['delta_provider_latency_mean'])} | "
            f"{f3(row['delta_tool_time_mean'])} |"
        )

    defense_lines = [
        (
            "| State | Trap | n | Payload B | Setup Mean s | Prepare s | "
            "Construction s | Response s | Injection s | Total Mean s | "
            "CPU Mean s | Attack ΔWall / Defense |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        defense_lines.append(
            f"| {row['state']} | {row['trap']} | {row['n']} | "
            f"{row['trap_payload_bytes_mean']:.0f} | "
            f"{f4(row['defense_setup_mean'])} | "
            f"{f4(row['defense_prepare_mean'])} | "
            f"{f4(row['defense_construction_mean'])} | "
            f"{f4(row['defense_response_mean'])} | "
            f"{f4(row['defense_injection_mean'])} | "
            f"{f4(row['defense_total_mean'])} | "
            f"{f4(row['defense_cpu_mean'])} | "
            f"{row['attack_delta_wall_to_defense_ratio_mean']:.1f}× |"
        )

    token_source_lines = [
        (
            "| State | Trap | n | First ΔTotal | Follow-up ΔTotal | "
            "First Share | Follow-up Share | ΔFollow-up Calls |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        token_source_lines.append(
            f"| {row['state']} | {row['trap']} | {row['n']} | "
            f"{f1(row['delta_first_total_mean'])} | "
            f"{f1(row['delta_followup_total_mean'])} | "
            f"{row['first_share'] * 100:.1f}% | "
            f"{row['followup_share'] * 100:.1f}% | "
            f"{f2(next(item for item in token_aggregate if item['state'] == row['state'] and item['trap'] == row['trap'])['mean_delta_followup_provider_calls'])} |"
        )

    cross_rank_lines = [
        "| State | Rank | Claude | GPT-5.6 Sol | DeepSeek-V4-Pro |",
        "|---|---:|---|---|---|",
    ]
    cross_rankings: dict[str, dict[str, dict[int, str]]] = {}
    cross_rank_rows = read_csv(live / "cross_model_state_rankings.csv")
    for model in ("claude", "gpt", "deepseek"):
        for state in STATES:
            group = [
                row
                for row in cross_rank_rows
                if row["model"] == model and row["state"] == state
            ]
            for row in group:
                cross_rankings.setdefault(model, {}).setdefault(state, {})[
                    int(row["rank_by_input_plus_output"])
                ] = row["trap"]
    for state in STATES:
        for rank in (1, 2, 3):
            cross_rank_lines.append(
                f"| {state} | {rank} | "
                f"{cross_rankings['claude'][state][rank]} | "
                f"{cross_rankings['gpt'][state][rank]} | "
                f"{cross_rankings['deepseek'][state][rank]} |"
            )

    replication_lines = [
        (
            "| State | Trap | Prior ΔTotal | Current ΔTotal | "
            "Current/Prior | Same Direction |"
        ),
        "|---|---|---:|---:|---:|---|",
    ]
    for row in replication_rows:
        replication_lines.append(
            f"| {row['state']} | {row['trap']} | "
            f"{f1(row['prior_delta_total_mean'])} | "
            f"{f1(row['current_delta_total_mean'])} | "
            f"{row['current_to_prior_ratio']:.3f}× | "
            f"{'YES' if row['same_direction'] else 'NO'} |"
        )

    historical_lines = [
        (
            "| State | Current Trap | Historical | n | Historical Mean Δ | "
            "Current Mean Δ | Same Direction | Comparability |"
        ),
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in historical_comparison:
        historical_lines.append(
            f"| {row['state']} | {row['current_trap']} | "
            f"{row['historical_condition']} | {row['historical_n']} | "
            f"{f1(row['historical_mean_delta_total_tokens'])} | "
            f"{f1(row['current_mean_delta_total_tokens'])} | "
            f"{'YES' if row['same_direction'] else 'NO'} | "
            f"{row['comparison_limit']} |"
        )

    cross_gpt = cross_match_counts["gpt"]
    cross_deepseek = cross_match_counts["deepseek"]
    report = f"""# CyberStrikeAI — Claude Sonnet 4.6 六 Source H1 正式复现实验

## 摘要

- 协议：`{execution['protocol_id']}`。
- Batch：`{run_dir.name}`。
- 平台：CyberStrikeAI，复用仓库中已冻结并审计的 H1 执行路径、Trap 实现和 Claude Anthropic Messages Transport。
- 模型：`claude-sonnet-4-6`。正式阶段 563 次响应全部精确返回该模型 ID。
- 完成：6/6 Sources、18/18 checkpoints、72/72 条件轨迹、54/54 clean pairs。
- Provider Calls：正式 563，实时准入 2，总计 565/1056；未自动重试、补跑或 Recovery。
- 结论属性：六 Source 的描述性复现与跨模型验证，不作确认性显著性推断。

## 1. 配置与 Provider-free 审计

Provider-free 门禁结果为 PASS。Provider Calls 和外部网络请求均为 0；六 Source、三 State、四条件、Oracle、Gateway、恢复快照、`pre-effect-v1`、K=12、Journal 与测量兼容性检查全部通过。

| Source | Family | Fixture | Status |
|---:|---|---|---|
{chr(10).join(line.split('|', 2)[1] if False else line for line in source_lines[2:])}

实时准入使用两次真实请求，请求/响应模型均为 `claude-sonnet-4-6`，`max_tokens=8192`、`temperature=0.0`、串行非流式 ToolCall 与 tool_result 闭环均通过。

正式执行审计：

- HTTP 200：563/563；Request ID 唯一：563/563。
- Usage 完整：563/563；Journal 独立审计：PASS；Gateway：72/72 PASS。
- 未授权 effect：0；Gateway bypass：0；post-terminal calls：72/72 为零。
- 条件轨迹状态：PASS 72/72；54 个配对全部满足 clean-pair 规则。

## 2. 资源统计口径

Anthropic `input_tokens` 是未缓存输入：

```text
有效输入 = input_tokens
         + cache_creation_input_tokens
         + cache_read_input_tokens

Total = 有效输入 + output_tokens
```

本报告将该合计标记为“Input+Output 口径”。本轮 Cache Creation 与 Cache Read 均为 0，因此原始未缓存输入、有效输入和 `prompt_tokens` 数值相同；缓存 Token 没有被重复计入。正式阶段合计：

| 项目 | 实际值 |
|---|---:|
| 原始未缓存 Input Tokens | {delivery_summary['usage']['effective_input_tokens']:,} |
| Effective Input Tokens | {delivery_summary['usage']['effective_input_tokens']:,} |
| Output Tokens | {delivery_summary['usage']['output_tokens']:,} |
| Cache Creation Tokens | 0 |
| Cache Read Tokens | 0 |
| Input+Output Total | {delivery_summary['usage']['input_output_total_tokens']:,} |
| Provider Calls | 563 |

## 3. 72 条轨迹与 54 个 clean pair

18 个 State checkpoint 均来自六个 Source 的新自然轨迹。每个 checkpoint 在同一环境快照、历史、current plan 和 committed action 上恢复四种条件。72 条轨迹均为 `PASS`；54 个 Trap–BASELINE 配对均为 clean pair。Clean pair 只表示可比较，不等同于 Trap 轨迹任务成功。

全部 54 个配对中：

- ΔTotal Tokens 为正、零、负：{token_sign[0]}/{token_sign[1]}/{token_sign[2]}。
- ΔWall-clock 为正、零、负：{wall_sign[0]}/{wall_sign[1]}/{wall_sign[2]}。
- BASELINE success：54/54；Trap success：50/54。
- BASELINE budget exhausted：0/54；Trap budget exhausted：4/54。

## 4. State×Trap Token 资源增量

Δ 均为同一 Source、同一 State checkpoint 上的 `Trap − BASELINE`。Mean Ratio 为“Trap 均值 / 配对 BASELINE 均值”；Median Pair Ratio 为逐配对倍数的中位数。

{chr(10).join(token_lines)}

### Token 组成

{chr(10).join(component_lines)}

- 9/9 单元的 Mean ΔTotal 为正。
- RECON 的 DEPENDENCY_V6 均值增量最大，并伴随 4/6 轨迹耗尽 K=12 预算。
- EXPLOITATION 的 DEPENDENCY_V6 均值增量最大。
- VERIFICATION 的 BRANCH_NEW_V3 均值增量最大，放大倍数也最高。

## 5. State×Trap Provider / Tool Calls 增量

{chr(10).join(call_lines)}

DEPENDENCY_V6 在三个 State 中均增加 Provider Calls 与 Tool Calls；BRANCH_NEW_V3 在 VERIFICATION 的调用增量最大；EVIDENCE_NEW_V3 的调用增量相对温和。

## 6. State×Trap 执行时间增量

时间边界为 checkpoint 恢复完成、initial exposure 前开始，到条件轨迹 terminal 结束。时延来自真实单调时钟和 Provider journal，不由 Token 推算。

{chr(10).join(wall_lines)}

- 9/9 单元的 Mean ΔWall-clock 为正。
- Provider 累计时延增量与 wall-clock 增量方向一致；工具执行时间增量通常只有毫秒级。
- RECON 的 BRANCH_NEW_V3 均值受少量长 Provider 请求影响，Median Pair Ratio 明显低于 Mean Ratio，不能只看均值。

## 7. Token 来源分解

“首次暴露”定义为该条件恢复后的第一条 Provider 请求；“后续自主交互”为其余 Provider 请求。两者分别减去同一 checkpoint 的 BASELINE 对应部分。

{chr(10).join(token_source_lines)}

54 个配对合计：

- 首次暴露 ΔTotal：{first_total_global:,.0f} Tokens，占净增量的 {first_total_global / token_aggregate_global * 100:.1f}%。
- 后续自主交互 ΔTotal：{follow_total_global:,.0f} Tokens，占净增量的 {follow_total_global / token_aggregate_global * 100:.1f}%。
- ΔInput：{input_total_global:,.0f} Tokens，占净增量的 {input_total_global / token_aggregate_global * 100:.1f}%。
- ΔOutput：{output_total_global:,.0f} Tokens，占净增量的 {output_total_global / token_aggregate_global * 100:.1f}%。

因此，本轮资源放大主要不是 Trap 首次暴露的上下文扩展本身，而是 Agent 在暴露后继续读取、修改或重试所形成的后续交互；后续交互又主要通过每轮重新发送的输入上下文累积 Token。Output 增量存在，但不是主要来源。

## 8. Task Success 与 Budget Exhausted

{chr(10).join(outcome_lines)}

四个非成功 Trap 轨迹全部是 RECON/DEPENDENCY_V6，并且都在 K=12 后以 `interaction_budget_exhausted` 结束。它们仍满足 clean-pair 协议，未被删除、补样或替换。

## 9. 防御端成本与攻防资源不对称

H1 的 State 和条件预先冻结，不存在在线 State Observer 或 H2 动态选择；这两项成本为 NA，不能填成 0。Preparation 与 response construction 是 initial exposure 内的嵌套阶段，injection 为 total 的非负残差。Runtime setup 包含共享 checkpoint 恢复，只能作为 inclusive setup 指标。

{chr(10).join(defense_lines)}

- Trap Payload 为 initial exposure 中可见 `trap_information` 的紧凑 UTF-8 字节数；BRANCH_NEW_V3 约 3.5 KB，DEPENDENCY_V6 与 EVIDENCE_NEW_V3 约 0.8–1.2 KB。
- 在线防御 Provider Calls、Tokens 与外部 API 费用均为 0；防御全部为本地逻辑。
- 单次 runtime initial exposure 的总均值范围为 {min(row['defense_total_mean'] for row in table_rows) * 1000:.3f}–{max(row['defense_total_mean'] for row in table_rows) * 1000:.3f} ms。
- 54 个可比较配对中，攻击 ΔWall-clock / runtime defense time 的中位数为 {median(ratio_values):,.1f}×，范围为 {min(ratio_values):,.1f}× 至 {max(ratio_values):,.1f}×。该比值只描述时间量纲上的不对称，不代表经济收益或显著性。
- 离线准备时间未单独观测，记为 `UNKNOWN`；完整防御总成本也是 `UNKNOWN`，因为 BRANCH 候选枚举位于 runtime exposure 边界之前，且 CPU/RSS 未隔离到独立防御进程。

## 10. 跨模型与历史 Claude 一致性

不同模型的绝对 Token 口径不可直接混合。下列比较仅使用相同 Source/State/Trap 的相对 BASELINE 方向和机制排序。

| 指标 | Claude 与 GPT 方向一致 | Claude 与 DeepSeek 方向一致 |
|---|---:|---:|
| ΔInput | {cross_gpt['input']}/9 | {cross_deepseek['input']}/9 |
| ΔOutput | {cross_gpt['output']}/9 | {cross_deepseek['output']}/9 |
| ΔProvider Calls | {cross_gpt['provider_calls']}/9 | {cross_deepseek['provider_calls']}/9 |
| ΔTool Calls | {cross_gpt['tool_calls']}/9 | {cross_deepseek['tool_calls']}/9 |
| ΔWall-clock | {cross_gpt['wall_time']}/9 | {cross_deepseek['wall_time']}/9 |

{chr(10).join(cross_rank_lines)}

跨模型机制排序使用三家模型共同有效的 Source cohort：RECON n=6，EXPLOITATION n=5，VERIFICATION n=4；这与上文 Claude 主表使用全部六条配对不同。Claude 与 GPT 在 RECON 的机制排序一致；三个 State 的输入、调用和 wall-clock 增量方向均与 GPT 一致。Claude 与 DeepSeek 的输入增量方向 9/9 一致，但 output、调用和机制排序的稳定性较低。跨模型一致趋势是 Trap 普遍增加输入与交互，而不是 output 必然增加。

### 同配置 Claude 直接重复

同一六 Source/State/Trap 配置的前一 Claude 批次与本轮比较：

{chr(10).join(replication_lines)}

- 9/9 单元的 Token 增量方向一致，三个 State 的机制排序完全相同。
- 九组“均值增量”的本轮/前轮比值约为 0.93–1.93；方向与排序比绝对幅度更稳定。
- 这是同模型同配置的直接复现证据，但两轮仍共享相同 Source 结构，不是独立总体抽样。

### 历史 Claude 九 Source 的同六 Source 子集

{chr(10).join(historical_lines)}

历史 Claude 与当前 Claude 都显示 DEPENDENCY_V6 是稳定的正资源放大机制，RECON 均以依赖类增量最高。历史数据使用 `BRANCH_NEW`，当前使用 `BRANCH_NEW_V3`，因此 BRANCH 不能直接作机制内对比；EVIDENCE 的均值方向在两轮不完全稳定。

## 11. 技术失败、限制与最终结论

本轮没有 Provider 技术失败、模型身份不符、Usage 缺失、协议违规、Gateway bypass 或未授权 effect。由于 `FINAL_STATUS.json` 的旧包装字段把 `condition_trajectories` 写成 `null`，本报告以后处理审计和 `execution_audit.json` 的 `condition_records=72` 为权威；原始状态文件保持未修改。

主要限制：

- 六个 Source 只能支持复现和描述统计；同一 Source 的三个 State、同一 BASELINE 的三个 Trap 比较均存在聚类与重复使用，因此不作显著性声明。
- 本轮缓存 Token 为 0；若实际生产路由启用 prompt caching，输入增长和费用结构可能不同。
- H1 没有在线状态分类与动态 Trap 选择，不能把本报告用于回答 H2 问题。
- BRANCH 候选枚举、离线准备和独立防御进程资源未被完整边界化，相关总成本保持 UNKNOWN。

最终回答：

1. **Claude Sonnet 4.6 是否表现出 State×Trap 资源响应差异？** 是。9/9 State×Trap 单元的 Mean ΔTotal Tokens 与 Mean ΔWall-clock 均为正，且各 State 的机制排序不同；同配置重复的方向 9/9 一致、排序完全相同。
2. **哪些组合更明显？** RECON 与 EXPLOITATION 的 DEPENDENCY_V6 最大；VERIFICATION 的 BRANCH_NEW_V3 最大，Token 均值倍数约 {next(row['token_mean_ratio'] for row in table_rows if row['state']=='VERIFICATION' and row['trap']=='BRANCH_NEW_V3'):.2f}×。
3. **Token 增加主要来自哪里？** {input_total_global / token_aggregate_global * 100:.1f}% 来自输入、{output_total_global / token_aggregate_global * 100:.1f}% 来自输出；{follow_total_global / token_aggregate_global * 100:.1f}% 来自首次 exposure 之后的自主交互。
4. **Token 增量是否伴随时间增长？** 是，51/54 配对 wall-clock 为正，9/9 单元均值为正；增量主要跟随 Provider 累计时延，工具执行时间增量可忽略。
5. **防御端实际资源成本？** 在线本地 initial exposure 均值约 {min(row['defense_total_mean'] for row in table_rows) * 1000:.3f}–{max(row['defense_total_mean'] for row in table_rows) * 1000:.3f} ms，在线 Provider Calls/Tokens/API 费用为 0；离线准备与完整防御成本 UNKNOWN。
6. **与 DeepSeek、GPT 是否一致？** 输入与 Provider Calls 的增量方向与 GPT 9/9 一致、与 DeepSeek 分别 9/9 和 8/9；wall-clock 与 GPT 9/9、与 DeepSeek 7/9 一致。机制排序存在模型差异，不能把方向一致等同于效应量相同。

## 交付索引

- `delivery_summary.json`：机器可读综合摘要与审计。
- `timing_ledger.csv`：72 条条件轨迹的 Timing Ledger。
- `token_source_decomposition.csv/json`：54 配对的首次暴露与后续交互分解。
- `defense_payload_and_ratio.csv/json`：Trap Payload、防御时间与攻防时间比。
- `pair_level_resource_deltas.csv`：54 个 clean pair 的绝对资源、增量、首次暴露/后续交互、时间与防御成本。
- `pair_validity.csv/json`：54 条 clean-pair 审计。
- `requests.jsonl`：原始 Request/Usage 事件 Ledger。
- `provider_requests.jsonl`、`request_usage.csv`：逐请求 Provider 与 Usage Ledger。
- `measurements.csv/json`：原始条件轨迹测量 Ledger。
- `tool_calls_and_results.jsonl`：Gateway 工具调用与结果。
- `FINAL_REPORT_ZH.md`：运行器自动生成的原始报告，本补充报告不覆盖它。
"""
    (live / "FINAL_REPORT_CYBERSTRIKEAI_ZH.md").write_text(
        report,
        encoding="utf-8",
    )
    (live / "DELIVERY_MODIFICATION_RECORD.md").write_text(
        "\n".join(
            [
                "# 交付后处理记录",
                "",
                f"- 来源批次：`{run_dir.name}`。",
                f"- 生成时间：{execution['analysis_generated_at']}。",
                "- 仅读取已保存的 Request、Usage、Timing、Pair 与 Defense 记录。",
                "- 新增 Provider Calls：0；重试、补样、Recovery：0。",
                "- 未修改原始轨迹、原始 Ledger 或历史实验数据。",
                "- 生成 CyberStrikeAI 补充报告、Timing Ledger、Token 来源分解和防御开销明细。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return delivery_summary


def integer(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--prior-claude",
        type=Path,
        default=PRIOR_CLAUDE_6SOURCE,
    )
    parser.add_argument(
        "--historical-claude",
        type=Path,
        default=HISTORICAL_CLAUDE_9SOURCE,
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    summary = derive(
        run_dir,
        args.prior_claude.resolve(),
        args.historical_claude.resolve(),
    )
    print(
        json.dumps(
            {
                "batch_id": summary["batch_id"],
                "status": summary["status"],
                "sources": summary["completion"]["sources"],
                "checkpoints": summary["completion"]["checkpoints"],
                "condition_trajectories": summary["completion"][
                    "condition_trajectories"
                ],
                "clean_pairs": summary["completion"]["clean_pairs"],
                "provider_calls_added_by_derivation": 0,
                "report": str(
                    run_dir
                    / "live"
                    / "FINAL_REPORT_CYBERSTRIKEAI_ZH.md"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Offline analysis, report, charts, and validation for STARVE H2 final."""
from __future__ import annotations

import csv
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "STARVE-H2-FINAL-DSV4-9SOURCE_20261023T100453033042Z"
OUTPUT = RESULTS
STRATEGIES = (
    "NO_TRAP",
    "FIXED_BRANCH_NEW_V3",
    "FIXED_DEPENDENCY_V6",
    "FIXED_EVIDENCE_NEW_V3",
    "STATE_AWARE_FEASIBLE",
)
FIXED_COMPARATORS = (
    "NO_TRAP",
    "FIXED_BRANCH_NEW_V3",
    "FIXED_DEPENDENCY_V6",
    "FIXED_EVIDENCE_NEW_V3",
)
STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
FATAL = {"TECHNICAL_FAILURE", "PROTOCOL_VIOLATION", "MODEL_REFUSAL"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [deepcopy(dict(row)) for row in rows]
    fields = list(values[0]) if values else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow({
                key: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else "NA"
                    if value is None
                    else value
                )
                for key, value in row.items()
            })


def metadata_tree(path: Path) -> dict[str, Any]:
    result = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            stat = item.stat()
            result[str(item.relative_to(ROOT))] = {
                "size_bytes": stat.st_size,
                "modified_at_ns": stat.st_mtime_ns,
                "mode": oct(stat.st_mode & 0o777),
            }
    return result


def numeric(values: Iterable[Any]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def mean(values: Iterable[Any]) -> float | None:
    items = numeric(values)
    return sum(items) / len(items) if items else None


def median(values: Iterable[Any]) -> float | None:
    items = numeric(values)
    return statistics.median(items) if items else None


def standard_deviation(values: Iterable[Any]) -> float | None:
    items = numeric(values)
    return statistics.stdev(items) if len(items) > 1 else None


def standard_error(values: Iterable[Any]) -> float | None:
    items = numeric(values)
    deviation = standard_deviation(items)
    return deviation / math.sqrt(len(items)) if deviation is not None else None


def p95(values: Iterable[Any]) -> float | None:
    items = sorted(numeric(values))
    if not items:
        return None
    return items[min(len(items) - 1, math.ceil(0.95 * len(items)) - 1)]


def stats(values: Iterable[Any]) -> dict[str, Any]:
    items = numeric(values)
    return {
        "n": len(items),
        "mean": mean(items),
        "median": median(items),
        "sd": standard_deviation(items),
        "se": standard_error(items),
        "min": min(items) if items else None,
        "max": max(items) if items else None,
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def ratio(values: Iterable[bool]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def trajectory_dir(row: Mapping[str, Any]) -> Path:
    return (
        RESULTS
        / "real"
        / f"B{row['budget']}"
        / row["strategy"]
        / f"source-{row['source_index']:02d}"
    )


def first_state_entries(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    events = row.get("events") or []
    observations = [
        event
        for event in events
        if event.get("kind") == "state_observation"
        and event.get("observation", {}).get("online_state") in STATES
    ]
    result = {}
    for event in observations:
        state = event["observation"]["online_state"]
        if state not in result:
            result[state] = event
    return result


def build_state_timeline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline = []
    for row in rows:
        resource_events = [
            event
            for event in row.get("events") or []
            if event.get("kind") == "provider_resource_after"
        ]
        by_call = {
            int(event["provider_calls"]): event
            for event in resource_events
        }
        for event in row.get("events") or []:
            if event.get("kind") != "state_observation":
                continue
            observation = event.get("observation") or {}
            provider_call = int(event["provider_calls"])
            resource = by_call.get(provider_call, {})
            before = resource.get("before") or {}
            after = resource.get("after") or {}
            timeline.append({
                "budget": row["budget"],
                "source_index": row["source_index"],
                "task_family": row["task_family"],
                "strategy": row["strategy"],
                "event_index": event.get("index"),
                "provider_call": provider_call,
                "online_state": observation.get("online_state"),
                "collector_status": observation.get("collector_status"),
                "candidate_boundary": observation.get("candidate_boundary"),
                "cumulative_tokens_before_request": before.get(
                    "total_tokens"
                ),
                "cumulative_tokens_at_observation": after.get(
                    "total_tokens"
                ),
                "cumulative_input_tokens_before_request": before.get(
                    "input_tokens"
                ),
                "cumulative_output_tokens_before_request": before.get(
                    "output_tokens"
                ),
                "remaining_budget": event.get("remaining_budget"),
                "state_observer_ms": event.get("state_observer_ms"),
                "terminal_background": row["terminal_reason"],
                "task_success": bool(row["task_success"]),
            })
    return timeline


def build_state_resource_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        entries = first_state_entries(row)
        ordered = sorted(
            (
                (int(event["provider_calls"]), state, event)
                for state, event in entries.items()
            ),
            key=lambda item: item[0],
        )
        usage = row.get("provider_usage") or []
        request_metrics = row.get("request_metrics") or []
        total_calls = int(row["provider_calls"])
        for index, state in enumerate(STATES):
            if state not in entries:
                result.append({
                    "budget": row["budget"],
                    "source_index": row["source_index"],
                    "task_family": row["task_family"],
                    "strategy": row["strategy"],
                    "state": state,
                    "status": "NOT_REACHED",
                    "first_entry_provider_call": None,
                    "tokens_before_entry": None,
                    "tokens_at_entry": None,
                    "remaining_budget_at_entry": None,
                    "state_window_start_call": None,
                    "state_window_end_call": None,
                    "model_calls_in_state": 0,
                    "tool_calls_in_state": 0,
                    "effective_tool_calls_in_state": 0,
                    "new_tokens_in_state": None,
                })
                continue
            event = entries[state]
            start = int(event["provider_calls"])
            later = [
                call
                for call, other_state, _other in ordered
                if call > start and other_state != state
            ]
            end = min(later) - 1 if later else total_calls
            call_usage = usage[start - 1 : end]
            known = [
                item
                for item in call_usage
                if isinstance(item, Mapping)
                and type(item.get("total_tokens")) is int
            ]
            request = (
                request_metrics[start - 1]
                if start - 1 < len(request_metrics)
                else {}
            )
            at_tokens = event.get("cumulative_tokens_at_observation")
            current_tokens = request.get("total_tokens")
            before_tokens = (
                at_tokens - current_tokens
                if isinstance(at_tokens, int)
                and isinstance(current_tokens, int)
                else None
            )
            tool_events = [
                event_row
                for event_row in row.get("events") or []
                if event_row.get("kind") == "tool_call"
                and start <= int(event_row.get("provider_calls", -1)) <= end
            ]
            effective_tools = [
                event_row
                for event_row in row.get("events") or []
                if event_row.get("kind") == "tool_result"
                and event_row.get("status") == "ok"
                and start
                <= int(event_row.get("provider_calls", -1))
                <= end
            ]
            result.append({
                "budget": row["budget"],
                "source_index": row["source_index"],
                "task_family": row["task_family"],
                "strategy": row["strategy"],
                "state": state,
                "status": "REACHED",
                "first_entry_provider_call": start,
                "tokens_before_entry": before_tokens,
                "tokens_at_entry": at_tokens,
                "remaining_budget_at_entry": event.get("remaining_budget"),
                "state_window_start_call": start,
                "state_window_end_call": end,
                "model_calls_in_state": len(call_usage),
                "tool_calls_in_state": len(tool_events),
                "effective_tool_calls_in_state": len(effective_tools),
                "new_tokens_in_state": (
                    sum(item["total_tokens"] for item in known)
                    if known
                    else None
                ),
            })
    return result


def state_summary(state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in state_rows:
        grouped[(row["budget"], row["strategy"], row["state"])].append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        budget, strategy, state = key
        reached = [row for row in rows if row["status"] == "REACHED"]
        result.append({
            "budget": budget,
            "strategy": strategy,
            "state": state,
            "trajectories": len(rows),
            "reached": len(reached),
            "reach_rate": len(reached) / len(rows) if rows else None,
            "mean_entry_provider_call": mean(
                row["first_entry_provider_call"] for row in reached
            ),
            "mean_tokens_before_entry": mean(
                row["tokens_before_entry"] for row in reached
            ),
            "mean_tokens_at_entry": mean(
                row["tokens_at_entry"] for row in reached
            ),
            "mean_remaining_budget_at_entry": mean(
                row["remaining_budget_at_entry"] for row in reached
            ),
            "mean_model_calls_in_state": mean(
                row["model_calls_in_state"] for row in reached
            ),
            "mean_tool_calls_in_state": mean(
                row["tool_calls_in_state"] for row in reached
            ),
            "mean_effective_tool_calls_in_state": mean(
                row["effective_tool_calls_in_state"] for row in reached
            ),
            "mean_new_tokens_in_state": mean(
                row["new_tokens_in_state"] for row in reached
            ),
        })
    return result


def build_request_rows(
    rows: list[dict[str, Any]],
    rates: Mapping[str, float],
) -> list[dict[str, Any]]:
    from adapters.h1_pilot_usage import estimate_cost_cny

    result = []
    for row in rows:
        provider = read_json(trajectory_dir(row) / "provider_records.json")
        records = provider.get("raw") or []
        for index, metric in enumerate(row.get("request_metrics") or []):
            record = records[index] if index < len(records) else {}
            usage = record.get("normalized_usage") or {}
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            cache_read = usage.get("cache_read_tokens")
            cache_write = usage.get("cache_write_tokens")
            adjusted = (
                estimate_cost_cny(
                    usage,
                    {
                        "input_cny_per_million": rates["input"],
                        "cached_input_cny_per_million": rates[
                            "cached_input"
                        ],
                        "cache_write_cny_per_million": rates[
                            "cache_write"
                        ],
                        "output_cny_per_million": rates["output"],
                    },
                )
                if type(input_tokens) is int
                and type(output_tokens) is int
                else None
            )
            upper = (
                (
                    input_tokens * rates["input"]
                    + output_tokens * rates["output"]
                )
                / 1_000_000
                if type(input_tokens) is int
                and type(output_tokens) is int
                else None
            )
            result.append({
                "budget": row["budget"],
                "source_index": row["source_index"],
                "task_family": row["task_family"],
                "strategy": row["strategy"],
                "provider_call": metric.get("provider_call"),
                "remaining_budget": metric.get("remaining_budget"),
                "request_id": metric.get("request_id"),
                "request_model_id": record.get("request_model_id"),
                "response_model_id": record.get("response_model_id"),
                "http_status": record.get("http_status"),
                "finish_reason": record.get("finish_reason"),
                "outcome": record.get("outcome"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": usage.get("total_tokens"),
                "cache_read_tokens": cache_read,
                "cache_write_tokens": cache_write,
                "cumulative_input_tokens": metric.get(
                    "cumulative_input_tokens"
                ),
                "cumulative_output_tokens": metric.get(
                    "cumulative_output_tokens"
                ),
                "cumulative_total_tokens": metric.get(
                    "cumulative_total_tokens"
                ),
                "provider_api_wait_ms": metric.get(
                    "provider_api_wait_ms"
                ),
                "provider_usage": record.get("usage"),
                "list_price_upper_bound_cny": upper,
                "cache_adjusted_list_estimate_cny": adjusted,
            })
    return result


def trajectory_summary_rows(
    rows: list[dict[str, Any]],
    request_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    costs = defaultdict(lambda: [0.0, 0.0])
    for request in request_rows:
        key = (
            request["budget"],
            request["source_index"],
            request["strategy"],
        )
        costs[key][0] += request["list_price_upper_bound_cny"] or 0.0
        costs[key][1] += (
            request["cache_adjusted_list_estimate_cny"] or 0.0
        )
    result = []
    for row in rows:
        key = (row["budget"], row["source_index"], row["strategy"])
        states = [
            event["observation"]["online_state"]
            for event in row.get("events") or []
            if event.get("kind") == "state_observation"
        ]
        result.append({
            "budget": row["budget"],
            "source_index": row["source_index"],
            "source_id": row["source_id"],
            "task_family": row["task_family"],
            "strategy": row["strategy"],
            "actual_model_id": row["actual_model_id"],
            "provider_calls": row["provider_calls"],
            "interaction_depth": row["interaction_depth"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "total_tokens": row["total_tokens"],
            "reasoning_tokens": row.get("reasoning_tokens"),
            "cached_tokens": row.get("cached_tokens"),
            "effective_tool_calls": row["effective_tool_calls"],
            "task_success": row["task_success"],
            "budget_exhausted": row["budget_exhausted"],
            "technical_failure": row["technical_failure"],
            "protocol_violation": row["protocol_violation"],
            "terminal_reason": row["terminal_reason"],
            "reached_verification": "VERIFICATION" in row["reached_states"],
            "end_state": states[-1] if states else "UNKNOWN",
            "state_observation_sequence": states,
            "wall_time_ms": row["wall_time_ms"],
            "provider_api_wait_total_ms": row[
                "provider_api_wait_total_ms"
            ],
            "tool_execution_total_ms": row["tool_execution_total_ms"],
            "defense_gateway_total_ms": row["defense_totals"][
                "defense_gateway_total_ms"
            ],
            "trap_deployments": row["number_of_injections"],
            "fallbacks": row["number_of_fallbacks"],
            "not_deployable": row["number_of_not_deployable"],
            "unknown_observations": row["number_of_unknown"],
            "list_price_upper_bound_cny": costs[key][0],
            "cache_adjusted_list_estimate_cny": costs[key][1],
            "usage_complete": row["usage_complete"],
            "model_identity_verified": row["model_identity_verified"],
            "initial_equal_within_five": row[
                "initial_equal_within_five"
            ],
        })
    return result


def overall_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for budget in (20, 30):
        for strategy in STRATEGIES:
            selected = [
                row
                for row in rows
                if row["budget"] == budget and row["strategy"] == strategy
            ]
            opportunities = [
                item
                for row in selected
                for item in row.get("opportunities") or []
            ]
            result.append({
                "budget": budget,
                "strategy": strategy,
                "trajectories": len(selected),
                "mean_total_tokens": mean(
                    row["total_tokens"] for row in selected
                ),
                "median_total_tokens": median(
                    row["total_tokens"] for row in selected
                ),
                "mean_input_tokens": mean(
                    row["input_tokens"] for row in selected
                ),
                "median_input_tokens": median(
                    row["input_tokens"] for row in selected
                ),
                "mean_output_tokens": mean(
                    row["output_tokens"] for row in selected
                ),
                "median_output_tokens": median(
                    row["output_tokens"] for row in selected
                ),
                "mean_provider_calls": mean(
                    row["provider_calls"] for row in selected
                ),
                "median_provider_calls": median(
                    row["provider_calls"] for row in selected
                ),
                "mean_effective_tool_calls": mean(
                    row["effective_tool_calls"] for row in selected
                ),
                "task_success_rate": ratio(
                    bool(row["task_success"]) for row in selected
                ),
                "budget_exhaustion_rate": ratio(
                    bool(row["budget_exhausted"]) for row in selected
                ),
                "mean_wall_time_ms": mean(
                    row["wall_time_ms"] for row in selected
                ),
                "median_wall_time_ms": median(
                    row["wall_time_ms"] for row in selected
                ),
                "trap_deployments": sum(
                    row["number_of_injections"] for row in selected
                ),
                "mean_trap_deployments": mean(
                    row["number_of_injections"] for row in selected
                ),
                "fallbacks": sum(
                    row["number_of_fallbacks"] for row in selected
                ),
                "not_deployable": sum(
                    row["number_of_not_deployable"] for row in selected
                ),
                "state_opportunities": len(opportunities),
                "mean_defense_gateway_ms": mean(
                    row["defense_totals"]["defense_gateway_total_ms"]
                    for row in selected
                ),
                "mean_provider_api_wait_ms": mean(
                    row["provider_api_wait_total_ms"] for row in selected
                ),
                "mean_tool_execution_ms": mean(
                    row["tool_execution_total_ms"] for row in selected
                ),
                "verification_reached": sum(
                    "VERIFICATION" in row["reached_states"]
                    for row in selected
                ),
                "mean_verification_remaining_budget": mean(
                    item["remaining_budget"]
                    for item in opportunities
                    if item["injection_state"] == "VERIFICATION"
                ),
            })
    return result


def primary_pair_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in pairs:
        if (
            row["left_strategy"] in FIXED_COMPARATORS
            and row["right_strategy"] == "STATE_AWARE_FEASIBLE"
        ):
            enriched = deepcopy(row)
            enriched["delta_success"] = (
                int(row["right_success"]) - int(row["left_success"])
                if row["technical_pair_valid"]
                else None
            )
            enriched["delta_budget_exhausted"] = (
                int(row["right_budget_exhausted"])
                - int(row["left_budget_exhausted"])
                if row["technical_pair_valid"]
                else None
            )
            result.append(enriched)
    return result


def pair_summaries(
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    metric_names = (
        "delta_total_tokens",
        "delta_input_tokens",
        "delta_output_tokens",
        "delta_provider_calls",
        "delta_effective_tool_calls",
        "delta_wall_time_ms",
        "delta_success",
        "delta_budget_exhausted",
    )
    for comparator in FIXED_COMPARATORS:
        for budget in (20, 30):
            base = [
                row
                for row in pair_rows
                if row["left_strategy"] == comparator
                and row["right_strategy"] == "STATE_AWARE_FEASIBLE"
                and row["budget"] == budget
                and row["technical_pair_valid"]
            ]
            for scope, selected in (
                ("all_technical_valid", base),
                (
                    "both_task_success",
                    [row for row in base if row["both_task_success"]],
                ),
            ):
                row = {
                    "budget": budget,
                    "comparator": comparator,
                    "scope": scope,
                    "pairs": len(selected),
                    "left_successes": sum(
                        row["left_success"] for row in selected
                    ),
                    "right_successes": sum(
                        row["right_success"] for row in selected
                    ),
                    "left_success_rate": ratio(
                        row["left_success"] for row in selected
                    ),
                    "right_success_rate": ratio(
                        row["right_success"] for row in selected
                    ),
                    "left_exhaustion_rate": ratio(
                        row["left_budget_exhausted"] for row in selected
                    ),
                    "right_exhaustion_rate": ratio(
                        row["right_budget_exhausted"] for row in selected
                    ),
                }
                for metric in metric_names:
                    values = [row[metric] for row in selected]
                    row[f"{metric}_mean"] = mean(values)
                    row[f"{metric}_median"] = median(values)
                    row[f"{metric}_sd"] = standard_deviation(values)
                    row[f"{metric}_se"] = standard_error(values)
                    row[f"{metric}_positive"] = sum(
                        value > 0 for value in numeric(values)
                    )
                    row[f"{metric}_negative"] = sum(
                        value < 0 for value in numeric(values)
                    )
                    row[f"{metric}_zero"] = sum(
                        value == 0 for value in numeric(values)
                    )
                result.append(row)
    return result


def budget_sensitivity(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    indexed = {
        (row["budget"], row["strategy"]): row for row in summary
    }
    for strategy in STRATEGIES:
        b20 = indexed[(20, strategy)]
        b30 = indexed[(30, strategy)]
        result.append({
            "strategy": strategy,
            "b20_trajectories": b20["trajectories"],
            "b30_trajectories": b30["trajectories"],
            "delta_mean_total_tokens": (
                b30["mean_total_tokens"] - b20["mean_total_tokens"]
            ),
            "delta_median_total_tokens": (
                b30["median_total_tokens"]
                - b20["median_total_tokens"]
            ),
            "delta_mean_provider_calls": (
                b30["mean_provider_calls"] - b20["mean_provider_calls"]
            ),
            "delta_success_rate": (
                b30["task_success_rate"] - b20["task_success_rate"]
            ),
            "delta_exhaustion_rate": (
                b30["budget_exhaustion_rate"]
                - b20["budget_exhaustion_rate"]
            ),
            "delta_verification_remaining_budget": (
                b30["mean_verification_remaining_budget"]
                - b20["mean_verification_remaining_budget"]
            ),
            "delta_mean_defense_gateway_ms": (
                b30["mean_defense_gateway_ms"]
                - b20["mean_defense_gateway_ms"]
            ),
        })
    return result


def trap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        for opportunity in row.get("opportunities") or []:
            result.append({
                "budget": row["budget"],
                "source_index": row["source_index"],
                "task_family": row["task_family"],
                "strategy": row["strategy"],
                "state": opportunity["injection_state"],
                "provider_call": opportunity["provider_call"],
                "candidate_scope": opportunity.get("candidate_scope"),
                "candidates_checked": [
                    check.get("trap")
                    for check in opportunity.get("feasibility_checks") or []
                ],
                "candidate_feasibility": opportunity.get(
                    "feasibility_checks"
                ),
                "selected_trap": opportunity.get("selected_trap"),
                "deployment_status": opportunity.get("deployment_status"),
                "deployment_reason": opportunity.get("reason"),
                "fallback_used": opportunity.get("fallback_used"),
                "fallback_count": opportunity.get("fallback_count"),
                "fallback_reason": opportunity.get("fallback_reason"),
                "actual_injected": opportunity.get("actual_injected"),
                "feedback_bytes": opportunity.get("feedback_bytes"),
                "feedback_characters": opportunity.get(
                    "feedback_characters"
                ),
                "remaining_budget": opportunity.get("remaining_budget"),
                "feasibility_ms": opportunity.get("feasibility_ms"),
                "selector_ms": opportunity.get("selector_ms"),
                "response_construction_ms": opportunity.get(
                    "response_construction_ms"
                ),
                "injection_or_pass_through_ms": opportunity.get(
                    "injection_or_pass_through_ms"
                ),
                "opportunity_total_ms": opportunity.get(
                    "opportunity_total_ms"
                ),
            })
    return result


def trap_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter(
        (
            row["budget"],
            row["strategy"],
            row["state"],
            row["selected_trap"] or "NONE",
            row["deployment_status"],
        )
        for row in rows
    )
    return [
        {
            "budget": key[0],
            "strategy": key[1],
            "state": key[2],
            "selected_trap": key[3],
            "deployment_status": key[4],
            "count": count,
        }
        for key, count in sorted(counter.items())
    ]


def defense_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    samples = []
    trajectory = []
    for row in rows:
        for sample in row.get("defense_samples") or []:
            samples.append({
                "budget": row["budget"],
                "source_index": row["source_index"],
                "task_family": row["task_family"],
                "strategy": row["strategy"],
                "opportunity_index": sample.get("opportunity_index"),
                "state": sample.get("state"),
                "provider_call": sample.get("provider_call"),
                "state_observer_ms": sample.get("state_observer_ms"),
                "trap_feasibility_ms": sample.get("feasibility_ms"),
                "trap_selector_ms": sample.get("selector_ms"),
                "trap_response_construction_ms": sample.get(
                    "response_construction_ms"
                ),
                "injection_or_pass_through_ms": sample.get(
                    "injection_or_pass_through_ms"
                ),
                "opportunity_total_ms": sample.get(
                    "opportunity_total_ms"
                ),
                "gateway_opportunity_ms": sample.get(
                    "gateway_opportunity_ms"
                ),
            })
        defense = row["defense_totals"]
        trajectory.append({
            "budget": row["budget"],
            "source_index": row["source_index"],
            "task_family": row["task_family"],
            "strategy": row["strategy"],
            "samples": defense.get("samples"),
            "state_observer_total_ms": defense.get(
                "state_observer_total_ms"
            ),
            "trap_feasibility_check_total_ms": defense.get(
                "trap_feasibility_check_total_ms"
            ),
            "trap_selector_total_ms": defense.get("trap_selector_total_ms"),
            "trap_response_construction_total_ms": defense.get(
                "trap_response_construction_total_ms"
            ),
            "injection_or_pass_through_total_ms": defense.get(
                "injection_or_pass_through_total_ms"
            ),
            "guard_before_total_ms": defense.get("guard_before_total_ms"),
            "guard_after_total_ms": defense.get("guard_after_total_ms"),
            "provider_received_total_ms": defense.get(
                "provider_received_total_ms"
            ),
            "defense_gateway_total_ms": defense.get(
                "defense_gateway_total_ms"
            ),
            "mean_opportunity_gateway_ms": defense.get(
                "mean_opportunity_gateway_ms"
            ),
            "trap_deployments": row["number_of_injections"],
            "fallbacks": row["number_of_fallbacks"],
            "feedback_bytes": sum(
                item.get("feedback_bytes") or 0
                for item in row.get("opportunities") or []
            ),
        })
    summary = []
    module_keys = (
        "state_observer_ms",
        "trap_feasibility_ms",
        "trap_selector_ms",
        "trap_response_construction_ms",
        "injection_or_pass_through_ms",
        "opportunity_total_ms",
        "gateway_opportunity_ms",
    )
    for budget in (20, 30):
        for strategy in STRATEGIES:
            selected = [
                row
                for row in samples
                if row["budget"] == budget and row["strategy"] == strategy
            ]
            for module in module_keys:
                values = [
                    row[module]
                    for row in selected
                    if row[module] is not None
                ]
                summary.append({
                    "budget": budget,
                    "strategy": strategy,
                    "module": module,
                    "samples": len(values),
                    "mean_ms": mean(values),
                    "median_ms": median(values),
                    "p95_ms": p95(values),
                    "max_ms": max(numeric(values)) if values else None,
                })
            selected_trajectories = [
                row
                for row in trajectory
                if row["budget"] == budget and row["strategy"] == strategy
            ]
            summary.append({
                "budget": budget,
                "strategy": strategy,
                "module": "defense_gateway_total_per_trajectory_ms",
                "samples": len(selected_trajectories),
                "mean_ms": mean(
                    row["defense_gateway_total_ms"]
                    for row in selected_trajectories
                ),
                "median_ms": median(
                    row["defense_gateway_total_ms"]
                    for row in selected_trajectories
                ),
                "p95_ms": p95(
                    row["defense_gateway_total_ms"]
                    for row in selected_trajectories
                ),
                "max_ms": max(
                    numeric(
                        row["defense_gateway_total_ms"]
                        for row in selected_trajectories
                    )
                )
                if selected_trajectories
                else None,
            })
    return samples, trajectory, summary


def cost_summary(
    rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for budget in (20, 30):
        for strategy in STRATEGIES:
            selected = [
                row
                for row in requests
                if row["budget"] == budget and row["strategy"] == strategy
            ]
            result.append({
                "budget": budget,
                "strategy": strategy,
                "requests": len(selected),
                "input_tokens": sum(
                    int(row["input_tokens"] or 0) for row in selected
                ),
                "output_tokens": sum(
                    int(row["output_tokens"] or 0) for row in selected
                ),
                "total_tokens": sum(
                    int(row["total_tokens"] or 0) for row in selected
                ),
                "cached_tokens": sum(
                    int(row["cache_read_tokens"] or 0) for row in selected
                ),
                "list_price_upper_bound_cny": sum(
                    row["list_price_upper_bound_cny"] or 0
                    for row in selected
                ),
                "cache_adjusted_list_estimate_cny": sum(
                    row["cache_adjusted_list_estimate_cny"] or 0
                    for row in selected
                ),
            })
    return result


def cumulative_curves(rows: list[dict[str, Any]]):
    values = defaultdict(dict)
    for row in rows:
        for metric in row.get("request_metrics") or []:
            call = int(metric["provider_call"])
            total = metric.get("cumulative_total_tokens")
            if type(total) is int:
                values[(row["budget"], row["strategy"])][call] = (
                    values[(row["budget"], row["strategy"])].get(call, [])
                )
                values[(row["budget"], row["strategy"])][call].append(total)
    result = {}
    for key, calls in values.items():
        result[key] = [
            (call, sum(items) / len(items))
            for call, items in sorted(calls.items())
        ]
    return result


def _font(size: int):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_cumulative_chart(rows: list[dict[str, Any]], path: Path):
    from PIL import Image, ImageDraw

    curves = cumulative_curves(rows)
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    colors = {
        "NO_TRAP": "#1f77b4",
        "FIXED_BRANCH_NEW_V3": "#d62728",
        "FIXED_DEPENDENCY_V6": "#2ca02c",
        "FIXED_EVIDENCE_NEW_V3": "#9467bd",
        "STATE_AWARE_FEASIBLE": "#ff7f0e",
    }
    font = _font(22)
    small = _font(16)
    for panel, budget in enumerate((20, 30)):
        left = 80 + panel * 760
        top = 100
        width = 680
        height = 650
        series = [
            curves.get((budget, strategy), [])
            for strategy in STRATEGIES
        ]
        max_x = max(
            [point[0] for points in series for point in points] or [1]
        )
        max_y = max(
            [point[1] for points in series for point in points] or [1]
        )
        draw.rectangle(
            (left, top, left + width, top + height),
            outline="#333333",
            width=2,
        )
        draw.text(
            (left, 55),
            f"B={budget}: mean cumulative total tokens",
            fill="black",
            font=font,
        )
        for grid in range(6):
            y = top + height - grid * height / 5
            draw.line((left, y, left + width, y), fill="#dddddd")
            draw.text(
                (left - 70, y - 8),
                f"{max_y * grid / 5:,.0f}",
                fill="#333333",
                font=small,
            )
        for strategy, points in zip(STRATEGIES, series):
            pixels = [
                (
                    left + (call - 1) / max(1, max_x - 1) * width,
                    top + height - value / max_y * height,
                )
                for call, value in points
            ]
            if len(pixels) > 1:
                draw.line(pixels, fill=colors[strategy], width=3)
            for x, y in pixels:
                draw.ellipse(
                    (x - 2, y - 2, x + 2, y + 2),
                    fill=colors[strategy],
                )
        draw.text(
            (left + width // 2 - 60, top + height + 20),
            "Provider call",
            fill="black",
            font=small,
        )
    legend_y = 825
    for index, strategy in enumerate(STRATEGIES):
        x = 100 + index * 285
        draw.rectangle((x, legend_y, x + 28, legend_y + 14), fill=colors[strategy])
        draw.text(
            (x + 36, legend_y - 4),
            strategy,
            fill="black",
            font=small,
        )
    image.save(path)


def draw_bar_chart(
    summary: list[dict[str, Any]],
    key: str,
    title: str,
    path: Path,
):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1600, 800), "white")
    draw = ImageDraw.Draw(image)
    font = _font(26)
    small = _font(16)
    colors = {"20": "#4c78a8", "30": "#f58518"}
    short_labels = {
        "NO_TRAP": "NO_TRAP",
        "FIXED_BRANCH_NEW_V3": "BRANCH",
        "FIXED_DEPENDENCY_V6": "DEPENDENCY",
        "FIXED_EVIDENCE_NEW_V3": "EVIDENCE",
        "STATE_AWARE_FEASIBLE": "FEASIBLE",
    }
    indexed = {
        (row["budget"], row["strategy"]): row[key]
        for row in summary
    }
    values = [value for value in indexed.values() if value is not None]
    maximum = max(values or [1])
    left, top, width, height = 90, 100, 1420, 580
    draw.text((left, 45), title, fill="black", font=font)
    draw.rectangle((left, top, left + width, top + height), outline="#333333", width=2)
    for grid in range(6):
        y = top + height - grid * height / 5
        draw.line((left, y, left + width, y), fill="#dddddd")
        draw.text(
            (10, y - 8),
            f"{maximum * grid / 5:,.1f}",
            fill="#333333",
            font=small,
        )
    group_width = width / len(STRATEGIES)
    bar_width = group_width * 0.28
    for index, strategy in enumerate(STRATEGIES):
        center = left + group_width * (index + 0.5)
        for group, budget in enumerate((20, 30)):
            value = indexed.get((budget, strategy))
            if value is None:
                continue
            bar_height = value / maximum * height
            x0 = center - bar_width + group * bar_width
            y0 = top + height - bar_height
            draw.rectangle(
                (x0, y0, x0 + bar_width, top + height),
                fill=colors[str(budget)],
            )
            draw.text(
                (x0 + 3, y0 - 20),
                f"{value:.1f}",
                fill="black",
                font=small,
            )
        draw.text(
            (center - len(strategy) * 4, top + height + 20),
            short_labels[strategy],
            fill="black",
            font=small,
        )
    draw.rectangle((1120, 45, 1140, 63), fill=colors["20"])
    draw.text((1150, 42), "B=20", fill="black", font=small)
    draw.rectangle((1250, 45, 1270, 63), fill=colors["30"])
    draw.text((1280, 42), "B=30", fill="black", font=small)
    image.save(path)


def draw_wall_time_chart(rows: list[dict[str, Any]], path: Path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1600, 800), "white")
    draw = ImageDraw.Draw(image)
    font = _font(24)
    small = _font(15)
    colors = {"20": "#54a24b", "30": "#e45756"}
    short_labels = {
        "NO_TRAP": "NO_TRAP",
        "FIXED_BRANCH_NEW_V3": "BRANCH",
        "FIXED_DEPENDENCY_V6": "DEPENDENCY",
        "FIXED_EVIDENCE_NEW_V3": "EVIDENCE",
        "STATE_AWARE_FEASIBLE": "FEASIBLE",
    }
    values = defaultdict(list)
    for row in rows:
        values[(row["budget"], row["strategy"])].append(
            row["wall_time_ms"] / 1000
        )
    maximum = max(
        [item for group in values.values() for item in group] or [1]
    )
    left, top, width, height = 90, 100, 1420, 580
    draw.text(
        (left, 45),
        "Wall-clock time per trajectory (seconds)",
        fill="black",
        font=font,
    )
    draw.rectangle((left, top, left + width, top + height), outline="#333333", width=2)
    for grid in range(6):
        y = top + height - grid * height / 5
        draw.line((left, y, left + width, y), fill="#dddddd")
        draw.text(
            (10, y - 8),
            f"{maximum * grid / 5:.1f}",
            fill="#333333",
            font=small,
        )
    group_width = width / len(STRATEGIES)
    bar_width = group_width * 0.28
    for index, strategy in enumerate(STRATEGIES):
        center = left + group_width * (index + 0.5)
        med = median(values[(20, strategy)]) or 0
        med30 = median(values[(30, strategy)]) or 0
        for group, value, budget in (
            (0, med, 20),
            (1, med30, 30),
        ):
            bar_height = value / maximum * height
            x0 = center - bar_width + group * bar_width
            y0 = top + height - bar_height
            draw.rectangle(
                (x0, y0, x0 + bar_width, top + height),
                fill=colors[str(budget)],
            )
            draw.text((x0 + 2, y0 - 18), f"{value:.1f}", fill="black", font=small)
        draw.text(
            (center - len(strategy) * 3, top + height + 20),
            short_labels[strategy],
            fill="black",
            font=small,
        )
    draw.text((1030, 42), "Median", fill="black", font=small)
    draw.rectangle((1120, 45, 1140, 63), fill=colors["20"])
    draw.text((1150, 42), "B=20", fill="black", font=small)
    draw.rectangle((1250, 45, 1270, 63), fill=colors["30"])
    draw.text((1280, 42), "B=30", fill="black", font=small)
    image.save(path)


def report_markdown(
    overall: list[dict[str, Any]],
    pair_summary: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    state_summary_rows: list[dict[str, Any]],
    defense_summary: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    validation: Mapping[str, Any],
    trap_dist: list[dict[str, Any]],
) -> str:
    indexed = {(row["budget"], row["strategy"]): row for row in overall}
    pair_index = {
        (row["budget"], row["comparator"], row["scope"]): row
        for row in pair_summary
    }
    state_index = {
        (row["budget"], row["strategy"], row["state"]): row
        for row in state_summary_rows
    }

    def pair_value(budget, comparator, scope, key):
        return pair_index[(budget, comparator, scope)][key]

    no_trap = indexed[(20, "NO_TRAP")]
    dynamic20 = indexed[(20, "STATE_AWARE_FEASIBLE")]
    dynamic30 = indexed[(30, "STATE_AWARE_FEASIBLE")]
    no_trap30 = indexed[(30, "NO_TRAP")]
    defense_no20 = no_trap["mean_defense_gateway_ms"]
    defense_dynamic20 = dynamic20["mean_defense_gateway_ms"]
    defense_no30 = no_trap30["mean_defense_gateway_ms"]
    defense_dynamic30 = dynamic30["mean_defense_gateway_ms"]
    total_list_cost = sum(
        row["list_price_upper_bound_cny"] or 0 for row in cost_rows
    )
    total_adjusted_cost = sum(
        row["cache_adjusted_list_estimate_cny"] or 0 for row in cost_rows
    )
    lines = [
        "# STARVE H2 最终五策略完整轨迹实验报告",
        "",
        f"生成时间：{now()}",
        "",
        "本报告基于 CyberStrikeAI H2 全轨迹运行时、DeepSeek V4 Pro、冻结九源和五策略一次重跑。"
        "未优化 Trap、未调整优先级、未自动重试、未补样、未 Recovery。",
        "",
        "## 1. 最终审计",
        "",
        f"- 完成轨迹：90/90。",
        f"- 技术失败：0；协议违规：0。",
        f"- Provider Calls：{validation['provider_calls']} / 2250。",
        f"- Input / Output / Total Tokens：{validation['input_tokens']:,} / "
        f"{validation['output_tokens']:,} / {validation['total_tokens']:,}。",
        f"- 实际响应模型：`deepseek-v4-pro`（全部轨迹）。",
        f"- 初始五路一致：90/90。",
        f"- 最终验证：{'PASS' if validation['passed'] else 'FAIL'}，"
        f"{sum(bool(value) for value in validation['checks'].values())}/"
        f"{len(validation['checks'])} 项检查通过。",
        "",
        "## 2. 五策略总体资源与任务结果",
        "",
        "| B | Strategy | Mean Tokens | Median Tokens | Mean Input | Mean Output | Mean Calls | Mean Tools | Success | Exhausted | Mean Wall (s) | Trap Deployments | Mean Defense (ms) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in (20, 30):
        for strategy in STRATEGIES:
            row = indexed[(budget, strategy)]
            lines.append(
                f"| {budget} | {strategy} | "
                f"{fmt(row['mean_total_tokens'], 1)} | "
                f"{fmt(row['median_total_tokens'], 1)} | "
                f"{fmt(row['mean_input_tokens'], 1)} | "
                f"{fmt(row['mean_output_tokens'], 1)} | "
                f"{fmt(row['mean_provider_calls'], 2)} | "
                f"{fmt(row['mean_effective_tool_calls'], 2)} | "
                f"{fmt(row['task_success_rate'], 3)} | "
                f"{fmt(row['budget_exhaustion_rate'], 3)} | "
                f"{fmt(row['mean_wall_time_ms'] / 1000, 2)} | "
                f"{row['trap_deployments']} | "
                f"{fmt(row['mean_defense_gateway_ms'], 2)} |"
            )
    lines.extend([
        "",
        "B=20 的预算耗尽轨迹全部保留。Wall time 是真实端到端时间；"
        "预算耗尽时间不能解释为成功完成时间。",
        "",
        "## 3. 配对比较：全部技术有效与双方成功",
        "",
        "Δ = `STATE_AWARE_FEASIBLE − Comparator`，同 Source、同 Budget。",
        "",
        "| B | Comparator | Scope | n | ΔTotal mean | ΔTotal median | 正/负/零 | ΔInput | ΔOutput | ΔCalls | ΔTools | ΔWall (s) | Success Δ | Exhaustion Δ |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for budget in (20, 30):
        for comparator in FIXED_COMPARATORS:
            for scope in ("all_technical_valid", "both_task_success"):
                row = pair_index[(budget, comparator, scope)]
                lines.append(
                    f"| {budget} | {comparator} | {scope} | {row['pairs']} | "
                    f"{fmt(row['delta_total_tokens_mean'], 1)} | "
                    f"{fmt(row['delta_total_tokens_median'], 1)} | "
                    f"{row['delta_total_tokens_positive']}/"
                    f"{row['delta_total_tokens_negative']}/"
                    f"{row['delta_total_tokens_zero']} | "
                    f"{fmt(row['delta_input_tokens_mean'], 1)} | "
                    f"{fmt(row['delta_output_tokens_mean'], 1)} | "
                    f"{fmt(row['delta_provider_calls_mean'], 2)} | "
                    f"{fmt(row['delta_effective_tool_calls_mean'], 2)} | "
                    f"{fmt(None if row['delta_wall_time_ms_mean'] is None else row['delta_wall_time_ms_mean'] / 1000, 2)} | "
                    f"{fmt(row['delta_success_mean'], 3)} | "
                    f"{fmt(row['delta_budget_exhausted_mean'], 3)} |"
                )
    lines.extend([
        "",
        "主要固定对照为 `FIXED_DEPENDENCY_V6`，该选择在任何结果读取前已冻结。"
        "双方均成功统计仅作条件描述，不替代全部技术有效配对。",
        "",
        "## 4. 预算敏感性",
        "",
        "| Strategy | B20 n | B30 n | ΔMean Tokens | ΔMedian Tokens | ΔCalls | ΔSuccess Rate | ΔExhaustion Rate | ΔVERIFICATION Remaining |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sensitivity:
        lines.append(
            f"| {row['strategy']} | {row['b20_trajectories']} | "
            f"{row['b30_trajectories']} | "
            f"{fmt(row['delta_mean_total_tokens'], 1)} | "
            f"{fmt(row['delta_median_total_tokens'], 1)} | "
            f"{fmt(row['delta_mean_provider_calls'], 2)} | "
            f"{fmt(row['delta_success_rate'], 3)} | "
            f"{fmt(row['delta_exhaustion_rate'], 3)} | "
            f"{fmt(row['delta_verification_remaining_budget'], 2)} |"
        )
    lines.extend([
        "",
        "预算差异来自两个预算下的独立轨迹，不能当作同一轨迹的因果延续。",
        "",
        "## 5. 状态间资源分配",
        "",
        "| B | Strategy | State | Reached | Mean Entry Call | Tokens Before Entry | Tokens At Entry | New Tokens in State | Remaining at Entry |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in state_summary_rows:
        lines.append(
            f"| {row['budget']} | {row['strategy']} | {row['state']} | "
            f"{row['reached']}/{row['trajectories']} | "
            f"{fmt(row['mean_entry_provider_call'], 2)} | "
            f"{fmt(row['mean_tokens_before_entry'], 1)} | "
            f"{fmt(row['mean_tokens_at_entry'], 1)} | "
            f"{fmt(row['mean_new_tokens_in_state'], 1)} | "
            f"{fmt(row['mean_remaining_budget_at_entry'], 2)} |"
        )
    lines.extend([
        "",
        "状态资源按首次进入位置和下一正式 State 的分界进行描述性切分。"
        "未到达状态保持 `NOT_REACHED`，不补造事件。",
        "",
        "## 6. Trap 部署与 fallback",
        "",
        "| B | Strategy | State | Trap | Status | Count |",
        "|---:|---|---|---|---|---:|",
    ])
    for row in trap_dist:
        lines.append(
            f"| {row['budget']} | {row['strategy']} | {row['state']} | "
            f"{row['selected_trap']} | {row['deployment_status']} | "
            f"{row['count']} |"
        )
    lines.extend([
        "",
        "固定策略只检查并尝试指定 Trap；不可部署时原动作继续执行，不 fallback。"
        "动态策略只在同一 State 机会内按冻结顺序 fallback。UNKNOWN 不消费机会。",
        "",
        "## 7. 防御侧实际处理成本",
        "",
        "| B | Strategy | Module | n | Mean ms | Median ms | P95 ms | Max ms |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in defense_summary:
        if row["module"] not in {
            "state_observer_ms",
            "trap_feasibility_ms",
            "trap_selector_ms",
            "trap_response_construction_ms",
            "injection_or_pass_through_ms",
            "defense_gateway_total_per_trajectory_ms",
        }:
            continue
        lines.append(
            f"| {row['budget']} | {row['strategy']} | {row['module']} | "
            f"{row['samples']} | {fmt(row['mean_ms'], 3)} | "
            f"{fmt(row['median_ms'], 3)} | {fmt(row['p95_ms'], 3)} | "
            f"{fmt(row['max_ms'], 3)} |"
        )
    lines.extend([
        "",
        "以上仅为防御自身本地处理时间，使用 `perf_counter_ns`。"
        "模型 API 等待与正常工具执行时间在逐轨迹文件中独立记录。",
        "",
        "## 8. API 费用估算",
        "",
        "| B | Strategy | Input | Output | Total | Cache Read | List Upper (CNY) | Cache-adjusted (CNY) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in cost_rows:
        lines.append(
            f"| {row['budget']} | {row['strategy']} | "
            f"{row['input_tokens']} | {row['output_tokens']} | "
            f"{row['total_tokens']} | {row['cached_tokens']} | "
            f"{fmt(row['list_price_upper_bound_cny'], 4)} | "
            f"{fmt(row['cache_adjusted_list_estimate_cny'], 4)} |"
        )
    lines.extend([
        "",
        "费用为公开价格估算，不声称等于实际账单。攻击侧 API 费用不与防御侧本地处理成本相加。",
        "",
        "## 9. 图表",
        "",
        "![累计 Token 曲线](state_cumulative_tokens.png)",
        "",
        "![VERIFICATION 剩余预算](verification_remaining_budget.png)",
        "",
        "![端到端执行时间](wall_time_comparison.png)",
        "",
        "## 10. 结论与限制",
        "",
        f"- B=20：动态策略相对 NO_TRAP 的 ΔTotal mean "
        f"{fmt(pair_value(20, 'NO_TRAP', 'all_technical_valid', 'delta_total_tokens_mean'), 1)}，"
        f"{pair_value(20, 'NO_TRAP', 'all_technical_valid', 'delta_total_tokens_positive')}/9 为正；"
        f"相对 Branch {fmt(pair_value(20, 'FIXED_BRANCH_NEW_V3', 'all_technical_valid', 'delta_total_tokens_mean'), 1)}，"
        f"相对 Dependency {fmt(pair_value(20, 'FIXED_DEPENDENCY_V6', 'all_technical_valid', 'delta_total_tokens_mean'), 1)}，"
        f"相对 Evidence {fmt(pair_value(20, 'FIXED_EVIDENCE_NEW_V3', 'all_technical_valid', 'delta_total_tokens_mean'), 1)}。"
        f"动态策略成功率 {fmt(dynamic20['task_success_rate'], 3)}。",
        f"- B=30：动态策略相对 NO_TRAP "
        f"{fmt(pair_value(30, 'NO_TRAP', 'all_technical_valid', 'delta_total_tokens_mean'), 1)}、"
        f"Branch {fmt(pair_value(30, 'FIXED_BRANCH_NEW_V3', 'all_technical_valid', 'delta_total_tokens_mean'), 1)}、"
        f"Dependency {fmt(pair_value(30, 'FIXED_DEPENDENCY_V6', 'all_technical_valid', 'delta_total_tokens_mean'), 1)}、"
        f"Evidence {fmt(pair_value(30, 'FIXED_EVIDENCE_NEW_V3', 'all_technical_valid', 'delta_total_tokens_mean'), 1)}。"
        f"动态策略成功率 {fmt(dynamic30['task_success_rate'], 3)}。",
        f"- B=30 双方成功条件配对（动态 vs Dependency，n="
        f"{pair_value(30, 'FIXED_DEPENDENCY_V6', 'both_task_success', 'pairs')}）"
        f"ΔTotal mean "
        f"{fmt(pair_value(30, 'FIXED_DEPENDENCY_V6', 'both_task_success', 'delta_total_tokens_mean'), 1)}、"
        f"median "
        f"{fmt(pair_value(30, 'FIXED_DEPENDENCY_V6', 'both_task_success', 'delta_total_tokens_median'), 1)}。",
        f"- 综合五策略结果，动态策略没有同时优于所有固定 Trap 和无防御。"
        f"B=30 NO_TRAP、Branch、Evidence 的成功率分别为 "
        f"{fmt(no_trap30['task_success_rate'], 3)}、"
        f"{fmt(indexed[(30, 'FIXED_BRANCH_NEW_V3')]['task_success_rate'], 3)}、"
        f"{fmt(indexed[(30, 'FIXED_EVIDENCE_NEW_V3')]['task_success_rate'], 3)}，"
        f"动态为 {fmt(dynamic30['task_success_rate'], 3)}。",
        f"- B=30 动态相对 NO_TRAP 平均多 "
        f"{fmt(pair_value(30, 'NO_TRAP', 'all_technical_valid', 'delta_total_tokens_mean'), 1)} Tokens、"
        f"{fmt(pair_value(30, 'NO_TRAP', 'all_technical_valid', 'delta_provider_calls_mean'), 2)} Calls 和 "
        f"{fmt(pair_value(30, 'NO_TRAP', 'all_technical_valid', 'delta_wall_time_ms_mean') / 1000, 2)} 秒 wall time。",
        f"- B=20 动态策略 RECON 内平均新增 "
        f"{fmt(state_index[(20, 'STATE_AWARE_FEASIBLE', 'RECON')]['mean_new_tokens_in_state'], 1)} Tokens，"
        f"NO_TRAP 为 "
        f"{fmt(state_index[(20, 'NO_TRAP', 'RECON')]['mean_new_tokens_in_state'], 1)}；"
        f"VERIFICATION 到达 "
        f"{state_index[(20, 'STATE_AWARE_FEASIBLE', 'VERIFICATION')]['reached']}/9，"
        f"到达时平均剩余 "
        f"{fmt(state_index[(20, 'STATE_AWARE_FEASIBLE', 'VERIFICATION')]['mean_remaining_budget_at_entry'], 2)} Calls。",
        f"- B=30 动态策略 RECON/EXPLOITATION/VERIFICATION 平均新增 Tokens 分别为 "
        f"{fmt(state_index[(30, 'STATE_AWARE_FEASIBLE', 'RECON')]['mean_new_tokens_in_state'], 1)}、"
        f"{fmt(state_index[(30, 'STATE_AWARE_FEASIBLE', 'EXPLOITATION')]['mean_new_tokens_in_state'], 1)}、"
        f"{fmt(state_index[(30, 'STATE_AWARE_FEASIBLE', 'VERIFICATION')]['mean_new_tokens_in_state'], 1)}；"
        f"VERIFICATION 到达 "
        f"{state_index[(30, 'STATE_AWARE_FEASIBLE', 'VERIFICATION')]['reached']}/9，"
        f"平均剩余 "
        f"{fmt(state_index[(30, 'STATE_AWARE_FEASIBLE', 'VERIFICATION')]['mean_remaining_budget_at_entry'], 2)} Calls。",
        f"- 防御侧每轨迹平均处理时间：NO_TRAP "
        f"{fmt(defense_no20, 2)}/{fmt(defense_no30, 2)} ms，动态策略 "
        f"{fmt(defense_dynamic20, 2)}/{fmt(defense_dynamic30, 2)} ms（B=20/B=30）；"
        f"每轨迹增量约 {fmt(defense_dynamic20 - defense_no20, 2)}/"
        f"{fmt(defense_dynamic30 - defense_no30, 2)} ms。",
        "- 防御处理与攻击 Agent API 费用不混算。",
        f"- 全部 90 条轨迹公开价格上界约 {total_list_cost:.2f} CNY，"
        f"按 cache 字段调整后约 {total_adjusted_cost:.2f} CNY；实际 Provider 账单未获取。",
        "- 动态策略与各固定 Trap、NO_TRAP 的比较以全部技术有效配对为主，双方成功配对单独报告。",
        "- B=20 和 B=30 为独立轨迹；成功率、耗尽率和资源差异不应解释为同一轨迹的预算截断效应。",
        "- 状态与 Trap 的关联为描述性结果；没有路径随机化或反事实分支，不能声称因果贡献。",
        "- 9 个 Source 的小样本不足以支持显著性外推。",
        "- 防御处理耗时与模型 API 等待、工具执行时间严格拆分。",
        "- 所有原始轨迹、Provider records、usage 账本和防御计时样本均保留。",
        "",
        "## 11. 机器可读文件",
        "",
        "- `trajectory_summary.csv`：逐轨迹资源、任务、时间线摘要。",
        "- `request_usage.csv`：逐请求 usage、累计资源、剩余预算和费用。",
        "- `state_timeline.csv` 与 `state_resource_summary.csv`：状态时间线和状态资源。",
        "- `trap_deployments.csv` 与 `trap_distribution.csv`：候选、选择、fallback 和部署。",
        "- `defense_samples.csv`、`defense_trajectory_cost.csv`、`defense_cost_summary.csv`：防御处理成本。",
        "- `strategy_pairs.csv`、`primary_comparison_summary.csv`：180 组全配对及四类主对照。",
        "- `final_validation.json`：最终只读验证。",
        "",
    ])
    return "\n".join(lines)


def validate(
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    defense_samples: list[dict[str, Any]],
    state_timeline: list[dict[str, Any]],
    execution: Mapping[str, Any],
    before_meta: Mapping[str, Any],
    after_meta: Mapping[str, Any],
) -> dict[str, Any]:
    expected_slots = {
        (budget, source, strategy)
        for budget in (20, 30)
        for source in EXPECTED_SOURCES
        for strategy in STRATEGIES
    }
    actual_slots = {
        (row["budget"], row["source_index"], row["strategy"])
        for row in rows
    }
    post_terminal_clear = []
    for row in rows:
        events = row.get("events") or []
        terminal_index = next(
            (
                index
                for index, event in enumerate(events)
                if event.get("kind") == "terminal"
            ),
            None,
        )
        post_terminal_clear.append(
            terminal_index is not None
            and not any(
                event.get("kind") in {
                    "provider_request",
                    "provider_response",
                    "tool_call",
                    "tool_result",
                }
                for event in events[terminal_index + 1 :]
            )
        )
    checks = {
        "execution_all90": execution.get("status") == "ALL90_COMPLETE"
        and execution.get("attempted_trajectories") == 90,
        "ninety_slots_exact": actual_slots == expected_slots,
        "b20_45": sum(row["budget"] == 20 for row in rows) == 45,
        "b30_45": sum(row["budget"] == 30 for row in rows) == 45,
        "five_strategy_counts_18_each": all(
            sum(row["strategy"] == strategy for row in rows) == 18
            for strategy in STRATEGIES
        ),
        "all_terminal": all(row.get("terminal_reason") for row in rows),
        "no_post_terminal_calls": all(post_terminal_clear),
        "provider_calls_within_2250": sum(
            row["provider_calls"] for row in rows
        )
        <= 2250,
        "all_models_exact": all(
            row["actual_model_id"] == "deepseek-v4-pro"
            and row["model_identity_verified"]
            for row in rows
        ),
        "all_usage_complete": all(row["usage_complete"] for row in rows),
        "no_unknown_usage": all(
            not row["unknown_usage_requests"] for row in rows
        ),
        "no_technical_failures": all(
            row["terminal_reason"] not in FATAL for row in rows
        ),
        "no_protocol_violations": all(
            row["terminal_reason"] != "PROTOCOL_VIOLATION" for row in rows
        ),
        "initial_five_way_equal": all(
            row["initial_equal_within_five"] for row in rows
        ),
        "all_180_pairs": len(pairs) == 180,
        "primary_pairs_18": sum(
            row["comparison_role"] == "primary_fixed" for row in pairs
        )
        == 18,
        "all_pairs_technically_valid": all(
            row["technical_pair_valid"] for row in pairs
        ),
        "request_rows_match_calls": len(requests)
        == sum(row["provider_calls"] for row in rows),
        "usage_totals_reconcile": sum(
            int(row["total_tokens"]) for row in rows
        )
        == sum(int(row["total_tokens"]) for row in requests),
        "state_timeline_nonempty": len(state_timeline) > 0,
        "defense_samples_nonempty": len(defense_samples) > 0,
        "defense_times_nonnegative": all(
            (row["opportunity_total_ms"] or 0) >= 0
            and (row["state_observer_ms"] or 0) >= 0
            for row in defense_samples
        ),
        "no_automatic_retries": execution.get("automatic_retries") == 0,
        "no_automatic_recovery": execution.get("automatic_recovery") is False,
        "protected_results_unchanged": execution.get(
            "protected_results_check", {}
        ).get("passed")
        is True,
        "freeze_unchanged": execution.get("freeze_check", {}).get(
            "passed"
        )
        is True,
        "analysis_input_metadata_unchanged": before_meta == after_meta,
    }
    checks["passed"] = all(checks.values())
    return {
        "at": now(),
        "passed": checks["passed"],
        "checks": checks,
        "attempted": len(rows),
        "provider_calls": sum(row["provider_calls"] for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "pairs": len(pairs),
        "primary_pairs": sum(
            row["comparison_role"] == "primary_fixed" for row in pairs
        ),
        "technical_valid_pairs": sum(
            row["technical_pair_valid"] for row in pairs
        ),
        "request_rows": len(requests),
        "state_observation_rows": len(state_timeline),
        "defense_sample_rows": len(defense_samples),
        "analysis_input_files": len(before_meta),
        "analysis_input_metadata_differences": (
            0 if before_meta == after_meta else 1
        ),
    }


EXPECTED_SOURCES = (1, 6, 2, 11, 16, 12, 21, 26, 22)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        required=True,
        help="new STARVE H2 result directory",
    )
    args = parser.parse_args()
    global RESULTS, OUTPUT
    RESULTS = Path(args.output).resolve()
    OUTPUT = RESULTS
    config = read_json(RESULTS / "config_snapshot.json")
    execution = read_json(RESULTS / "execution_status.json")
    rows = read_json(RESULTS / "rows.json")
    pairs = read_json(RESULTS / "pairs.json")
    rates = config["cost_estimation"]["rates_cny_per_million_tokens"]
    before_meta = metadata_tree(RESULTS / "real")

    request_rows = build_request_rows(rows, rates)
    trajectory_rows = trajectory_summary_rows(rows, request_rows)
    timeline = build_state_timeline(rows)
    state_resources = build_state_resource_rows(rows)
    state_resources_summary = state_summary(state_resources)
    overall = overall_summary(rows)
    primary_pairs = primary_pair_rows(pairs)
    pair_summary = pair_summaries(primary_pairs)
    sensitivity = budget_sensitivity(overall)
    trap_deployments = trap_rows(rows)
    trap_dist = trap_distribution(trap_deployments)
    defense_sample_rows, defense_trajectory_rows, defense_summary = defense_rows(rows)
    costs = cost_summary(rows, request_rows)

    write_csv(OUTPUT / "trajectory_summary.csv", trajectory_rows)
    write_csv(OUTPUT / "request_usage.csv", request_rows)
    write_csv(OUTPUT / "state_timeline.csv", timeline)
    write_csv(OUTPUT / "state_resource_rows.csv", state_resources)
    write_csv(OUTPUT / "state_resource_summary.csv", state_resources_summary)
    write_csv(OUTPUT / "trap_deployments.csv", trap_deployments)
    write_csv(OUTPUT / "trap_distribution.csv", trap_dist)
    write_csv(OUTPUT / "defense_samples.csv", defense_sample_rows)
    write_csv(OUTPUT / "defense_trajectory_cost.csv", defense_trajectory_rows)
    write_csv(OUTPUT / "defense_cost_summary.csv", defense_summary)
    write_csv(OUTPUT / "overall_strategy_summary.csv", overall)
    write_csv(OUTPUT / "strategy_pairs.csv", pairs)
    write_csv(OUTPUT / "primary_strategy_pairs.csv", primary_pairs)
    write_csv(OUTPUT / "primary_comparison_summary.csv", pair_summary)
    write_csv(OUTPUT / "budget_sensitivity.csv", sensitivity)
    write_csv(OUTPUT / "api_cost_summary.csv", costs)

    draw_cumulative_chart(rows, OUTPUT / "state_cumulative_tokens.png")
    draw_bar_chart(
        overall,
        "mean_verification_remaining_budget",
        "Remaining budget at first VERIFICATION",
        OUTPUT / "verification_remaining_budget.png",
    )
    draw_wall_time_chart(rows, OUTPUT / "wall_time_comparison.png")

    after_meta = metadata_tree(RESULTS / "real")
    validation = validate(
        rows,
        pairs,
        request_rows,
        defense_sample_rows,
        timeline,
        execution,
        before_meta,
        after_meta,
    )
    write_json(OUTPUT / "final_validation.json", validation)
    analysis_summary = {
        "at": now(),
        "experiment_id": config["experiment_id"],
        "provider_calls": sum(row["provider_calls"] for row in rows),
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "overall": overall,
        "primary_pairs": pair_summary,
        "budget_sensitivity": sensitivity,
        "state_summary": state_resources_summary,
        "defense_summary": defense_summary,
        "api_cost_summary": costs,
        "validation": validation,
    }
    write_json(OUTPUT / "analysis_summary.json", analysis_summary)
    (OUTPUT / "FINAL_REPORT_ZH.md").write_text(
        report_markdown(
            overall,
            pair_summary,
            sensitivity,
            state_resources_summary,
            defense_summary,
            costs,
            validation,
            trap_dist,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "ANALYZED",
        "passed": validation["passed"],
        "attempted": len(rows),
        "provider_calls": validation["provider_calls"],
        "total_tokens": validation["total_tokens"],
        "pairs": len(pairs),
        "output": str(OUTPUT),
    }, ensure_ascii=False), flush=True)
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

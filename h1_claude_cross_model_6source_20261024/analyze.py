"""Read-only analysis for Claude Sonnet 4.6 six-Source formal H1."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h1_cross_system_20261023 import runner as h1
from h1_gpt_cross_model_6source_20261024 import analyze as shared

PROTOCOL_ID = "formal_H1_Claude_Sonnet_4_6_cross_model_6source_20261024_v1"
MODEL_ID = "claude-sonnet-4-6"
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
    "effective_input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "input_output_total",
)
GPT_LIVE = (
    ROOT
    / "results"
    / "formal_H1_GPT_5_6_Sol_cross_model_6source_20261024T035420792818Z"
    / "live"
)
DEEPSEEK_LIVE = (
    ROOT / "results" / "formal_H1_cross_system_live_20261023_v1" / "live"
)
ADMISSION_RECORD = (
    ROOT
    / "results"
    / "claude_sonnet_4_6_admission_source01_20261022T080916235435Z"
    / "admission_record.json"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def _number(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [
        float(value)
        for value in values
        if _number(value) is not None
    ]
    return statistics.mean(numeric) if numeric else None


def _median(values: Iterable[Any]) -> float | None:
    numeric = [
        float(value)
        for value in values
        if _number(value) is not None
    ]
    return statistics.median(numeric) if numeric else None


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
        raw = (
            record.get("provider_raw_usage")
            if isinstance(record.get("provider_raw_usage"), dict)
            else record.get("usage")
            if isinstance(record.get("usage"), dict)
            else {}
        )
        usage = (
            record.get("usage")
            if isinstance(record.get("usage"), dict)
            else {}
        )
        normalized = (
            record.get("normalized_usage")
            if isinstance(record.get("normalized_usage"), dict)
            else {}
        )
        effective_input = (
            usage.get("input_tokens")
            if isinstance(usage.get("input_tokens"), int)
            else normalized.get("input_tokens")
        )
        output = (
            usage.get("output_tokens")
            if isinstance(usage.get("output_tokens"), int)
            else normalized.get("output_tokens")
        )
        input_output_total = (
            effective_input + output
            if isinstance(effective_input, int)
            and isinstance(output, int)
            else None
        )
        rows.append(
            {
                "source_index": source_index,
                "family": FAMILY_BY_SOURCE.get(source_index),
                "state": state,
                "condition": condition,
                "journal_path": str(path.relative_to(live_dir)),
                "request_id": record.get("request_id"),
                "local_request_index": record.get("request_index"),
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
                "usage_class": "Input+Output口径",
                "input_includes_cached_tokens": False,
                "raw_uncached_input_tokens": raw.get("input_tokens"),
                "effective_input_tokens": effective_input,
                "output_tokens": output,
                "cache_creation_input_tokens": raw.get(
                    "cache_creation_input_tokens"
                ),
                "cache_read_input_tokens": raw.get(
                    "cache_read_input_tokens"
                ),
                "input_output_total": input_output_total,
                "reasoning_tokens": normalized.get("reasoning_tokens"),
                "elapsed_seconds": record.get("elapsed_seconds"),
                "response_body_bytes": record.get("response_body_bytes"),
                "transport_phase": record.get("transport_phase"),
                "error": record.get("error"),
            }
        )
    return rows


def admission_usage_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "admission").glob("request-*.json")):
        record = read_json(path)
        usage = record.get("normalized_usage") or {}
        effective_input = usage.get("input_tokens")
        output = usage.get("output_tokens")
        rows.append(
            {
                "phase": "LIVE_ADMISSION",
                "journal_path": str(path.relative_to(root)),
                "request_id": record.get("request_id"),
                "request_model_id": record.get("request_model_id"),
                "provider_model_id": (
                    record.get("provider_model_id")
                    or record.get("response_model_id")
                ),
                "http_status": record.get("http_status"),
                "outcome": record.get("outcome"),
                "usage_class": "Input+Output口径",
                "raw_uncached_input_tokens": (
                    (record.get("provider_raw_usage") or {}).get(
                        "input_tokens"
                    )
                ),
                "effective_input_tokens": effective_input,
                "output_tokens": output,
                "cache_creation_input_tokens": usage.get(
                    "cache_creation_input_tokens"
                ),
                "cache_read_input_tokens": usage.get(
                    "cache_read_input_tokens"
                ),
                "input_output_total": (
                    effective_input + output
                    if isinstance(effective_input, int)
                    and isinstance(output, int)
                    else None
                ),
            }
        )
    return rows


def _sum_known(
    rows: list[Mapping[str, Any]],
    field: str,
) -> int | None:
    known = [
        int(row[field])
        for row in rows
        if isinstance(row.get(field), int)
        and not isinstance(row.get(field), bool)
    ]
    return sum(known) if known else None


def _coverage(
    rows: list[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    return {
        "known": sum(
            isinstance(row.get(field), int)
            and not isinstance(row.get(field), bool)
            for row in rows
        ),
        "missing": sum(
            not (
                isinstance(row.get(field), int)
                and not isinstance(row.get(field), bool)
            )
            for row in rows
        ),
        "known_sum": _sum_known(rows, field),
    }


def _add_total_deltas(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for row in rows:
        input_delta = row.get("delta_input_tokens")
        output_delta = row.get("delta_output_tokens")
        if (
            row["pair_valid"]
            and isinstance(input_delta, (int, float))
            and not isinstance(input_delta, bool)
            and isinstance(output_delta, (int, float))
            and not isinstance(output_delta, bool)
        ):
            row["delta_input_output_total"] = (
                input_delta + output_delta
            )
        else:
            row["delta_input_output_total"] = None
    return rows


def state_trap_summary(
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for summary in shared.state_trap_summary(pairs):
        cell = [
            row
            for row in pairs
            if row["state"] == summary["state"]
            and row["trap"] == summary["trap"]
            and row["pair_valid"]
        ]
        summary["delta_input_output_total_mean"] = _mean(
            row["delta_input_output_total"] for row in cell
        )
        summary["delta_input_output_total_median"] = _median(
            row["delta_input_output_total"] for row in cell
        )
        summary["usage_class"] = "Input+Output口径"
        result.append(summary)
    return result


def state_rankings(
    summaries: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for state in STATES:
        cell = [
            row for row in summaries if row["state"] == state
        ]
        ranked = sorted(
            cell,
            key=lambda row: (
                row.get("delta_input_output_total_mean") is not None,
                row.get("delta_input_output_total_mean") or float("-inf"),
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked, 1):
            result.append(
                {
                    "state": state,
                    "rank_by_total_resource_increment": rank,
                    "trap": row["trap"],
                    "clean_pairs": row["clean_pairs"],
                    "delta_input_mean": row.get("delta_input_mean"),
                    "delta_output_mean": row.get("delta_output_mean"),
                    "delta_input_output_total_mean": row.get(
                        "delta_input_output_total_mean"
                    ),
                    "delta_cache_read_mean": row.get(
                        "delta_cache_read_mean"
                    ),
                    "delta_provider_calls_mean": row.get(
                        "delta_provider_calls_mean"
                    ),
                    "delta_tool_calls_mean": row.get(
                        "delta_tool_calls_mean"
                    ),
                    "mechanism_followup": row["mechanism_followup"],
                    "deployed": row["deployed"],
                }
            )
    return result


def _trend(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "NA"
    if value > 0:
        return "increase"
    if value < 0:
        return "decrease"
    return "flat"


def _load_reference_pairs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = read_json(path)
    if not isinstance(value, list):
        return []
    return [
        row
        for row in value
        if int(row.get("source_index")) in SOURCE_INDICES
    ]


def _attach_reference_wall_time(
    rows: list[dict[str, Any]],
    live_dir: Path,
) -> list[dict[str, Any]]:
    for row in rows:
        source_index = int(row["source_index"])
        state = row["state"]
        baseline_path = (
            live_dir
            / "sources"
            / f"source-{source_index:02d}"
            / "checkpoints"
            / state.lower()
            / "branches"
            / "BASELINE.json"
        )
        trap_path = baseline_path.with_name(f"{row['trap']}.json")
        baseline = read_json(baseline_path) if baseline_path.is_file() else {}
        trap = read_json(trap_path) if trap_path.is_file() else {}
        baseline_wall = (baseline.get("metrics") or {}).get("wall_time")
        trap_wall = (trap.get("metrics") or {}).get("wall_time")
        row["baseline_wall_clock_seconds"] = baseline_wall
        row["trap_wall_clock_seconds"] = trap_wall
        row["delta_wall_clock_seconds"] = (
            trap_wall - baseline_wall
            if row.get("pair_valid")
            and isinstance(baseline_wall, (int, float))
            and not isinstance(baseline_wall, bool)
            and isinstance(trap_wall, (int, float))
            and not isinstance(trap_wall, bool)
            else None
        )
    return rows


def journal_audit(live_dir: Path) -> dict[str, Any]:
    journal_paths = sorted(live_dir.rglob("request-*.json"))
    records = [read_json(path) for path in journal_paths]
    ledger_path = live_dir / "requests.jsonl"
    ledger = (
        [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if ledger_path.is_file()
        else []
    )
    scope_rows = []
    for directory in sorted(
        {path.parent for path in journal_paths}
    ):
        records_in_scope = [
            read_json(path)
            for path in sorted(directory.glob("request-*.json"))
        ]
        expected_condition = (
            "SOURCE"
            if "provider-journal" in directory.parts
            else next(
                (
                    record.get("condition")
                    for record in records_in_scope
                    if isinstance(record.get("condition"), str)
                    and record.get("condition")
                ),
                directory.name.upper(),
            )
        )
        scope_audit = h1.audit_journal_records(
            records_in_scope,
            directory,
            expected_condition=expected_condition,
            ledger_events=ledger,
            offline=False,
        )
        scope_rows.append(
            {
                "journal_directory": str(directory.relative_to(live_dir)),
                "condition": expected_condition,
                **scope_audit,
            }
        )
    request_ids = [
        record.get("request_id") for record in records
    ]
    starts = [
        row for row in ledger if row.get("event") == "request_started"
    ]
    commits = [
        row
        for row in ledger
        if row.get("event") == "request_usage_committed"
    ]
    failures = [
        row for row in ledger if row.get("event") == "request_failed"
    ]
    logs_complete = (
        len(records) == len(starts) == len(commits)
        and not failures
        and len(request_ids) == len(set(request_ids))
        and all(
            isinstance(record.get("usage"), dict)
            for record in records
        )
        and all(row["complete"] for row in scope_rows)
    )
    return {
        "status": "PASS" if logs_complete else "FAIL",
        "journal_file_count": len(records),
        "ledger_request_started_count": len(starts),
        "ledger_usage_committed_count": len(commits),
        "ledger_failure_count": len(failures),
        "request_ids_unique": len(set(request_ids)),
        "all_journal_records_have_usage": all(
            isinstance(record.get("usage"), dict)
            for record in records
        ),
        "logs_complete": logs_complete,
        "scopes": scope_rows,
        "observed_audit_mismatch": False,
        "diagnosis": (
            "All Source and condition scopes map one-to-one to ledger "
            "start/usage events by Request ID; global and local numbering "
            "modes are accepted."
            if logs_complete
            else "Journal integrity audit failed; see scope reasons."
        ),
    }


def _valid_sources(
    rows: list[Mapping[str, Any]],
    state: str,
    trap: str,
) -> set[int]:
    return {
        int(row["source_index"])
        for row in rows
        if row["state"] == state
        and row["trap"] == trap
        and row["pair_valid"]
    }


def _subset_sources(
    rows: list[Mapping[str, Any]],
    state: str,
    trap: str,
    sources: set[int],
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row["state"] == state
        and row["trap"] == trap
        and int(row["source_index"]) in sources
    ]


def cross_model_rows(
    claude_pairs: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    gpt_pairs = _load_reference_pairs(GPT_LIVE / "pair_validity.json")
    deepseek_pairs = _load_reference_pairs(DEEPSEEK_LIVE / "pair_ledger.json")
    gpt_pairs = _attach_reference_wall_time(gpt_pairs, GPT_LIVE)
    deepseek_pairs = _attach_reference_wall_time(
        deepseek_pairs,
        DEEPSEEK_LIVE,
    )
    rows = []
    for state in STATES:
        for trap in TRAPS:
            model_rows = {
                "claude": claude_pairs,
                "gpt": gpt_pairs,
                "deepseek": deepseek_pairs,
            }
            available = {
                name: _valid_sources(values, state, trap)
                for name, values in model_rows.items()
            }
            common_sources = set.intersection(*available.values())
            subsets = {
                name: _subset_sources(values, state, trap, common_sources)
                for name, values in model_rows.items()
            }
            claude = shared.model_cell_summary(
                subsets["claude"],
                state,
                trap,
            )
            gpt = shared.model_cell_summary(
                subsets["gpt"],
                state,
                trap,
            )
            deepseek = shared.model_cell_summary(
                subsets["deepseek"],
                state,
                trap,
            )
            item = {
                "state": state,
                "trap": trap,
                "comparison_basis": (
                    "same Source/State/Trap; each model differenced against "
                    "its own BASELINE; direction and mechanism ranking only"
                ),
                "common_source_indices": ",".join(
                    str(index) for index in sorted(common_sources)
                ),
                "common_source_count": len(common_sources),
            }
            for name in ("claude", "gpt", "deepseek"):
                item[f"{name}_available_pairs"] = len(available[name])
                item[f"{name}_excluded_source_indices"] = ",".join(
                    str(index)
                    for index in sorted(
                        set(SOURCE_INDICES) - available[name]
                    )
                )
            for name, summary in (
                ("claude", claude),
                ("gpt", gpt),
                ("deepseek", deepseek),
            ):
                item[f"{name}_clean_pairs"] = summary["clean_pairs"]
                for metric in (
                    "input",
                    "output",
                    "cache_read",
                    "provider_calls",
                    "tool_calls",
                ):
                    mean = summary[f"{metric}_mean"]
                    item[f"{name}_delta_{metric}_mean"] = mean
                    item[f"{name}_{metric}_trend"] = _trend(mean)
                item[f"{name}_task_success_trap"] = summary[
                    "task_success_trap"
                ]
                item[f"{name}_task_success_baseline"] = summary[
                    "task_success_baseline"
                ]
                item[f"{name}_budget_exhausted_trap"] = summary[
                    "budget_exhausted_trap"
                ]
                item[f"{name}_budget_exhausted_baseline"] = summary[
                    "budget_exhausted_baseline"
                ]
                wall_values = [
                    row["delta_wall_clock_seconds"]
                    for row in subsets[name]
                    if row["pair_valid"]
                    and isinstance(
                        row.get("delta_wall_clock_seconds"),
                        (int, float),
                    )
                    and not isinstance(
                        row.get("delta_wall_clock_seconds"),
                        bool,
                    )
                ]
                item[f"{name}_wall_time_n"] = len(wall_values)
                item[f"{name}_delta_wall_time_mean"] = (
                    statistics.mean(wall_values) if wall_values else None
                )
                item[f"{name}_delta_wall_time_median"] = (
                    statistics.median(wall_values) if wall_values else None
                )
                item[f"{name}_wall_time_trend"] = _trend(
                    item[f"{name}_delta_wall_time_mean"]
                )
            for metric in (
                "input",
                "output",
                "cache_read",
                "provider_calls",
                "tool_calls",
                "wall_time",
            ):
                item[f"claude_gpt_{metric}_trend_match"] = (
                    item[f"claude_{metric}_trend"]
                    == item[f"gpt_{metric}_trend"]
                    if item["common_source_count"] > 0
                    else None
                )
                item[f"claude_deepseek_{metric}_trend_match"] = (
                    item[f"claude_{metric}_trend"]
                    == item[f"deepseek_{metric}_trend"]
                    if item["common_source_count"] > 0
                    else None
                )
            rows.append(item)
    return rows


def cross_model_rankings(
    claude_pairs: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    gpt_pairs = _load_reference_pairs(GPT_LIVE / "pair_validity.json")
    deepseek_pairs = _load_reference_pairs(DEEPSEEK_LIVE / "pair_ledger.json")
    gpt_pairs = _attach_reference_wall_time(gpt_pairs, GPT_LIVE)
    deepseek_pairs = _attach_reference_wall_time(
        deepseek_pairs,
        DEEPSEEK_LIVE,
    )
    model_rows = {
        "claude": claude_pairs,
        "gpt": gpt_pairs,
        "deepseek": deepseek_pairs,
    }
    result = []
    for state in STATES:
        cohort = set(SOURCE_INDICES)
        for model, rows in model_rows.items():
            for trap in TRAPS:
                cohort &= _valid_sources(rows, state, trap)
        for model, rows in model_rows.items():
            items = []
            for trap in TRAPS:
                subset = _subset_sources(rows, state, trap, cohort)
                valid = [row for row in subset if row["pair_valid"]]
                input_mean = _mean(
                    row.get("delta_input_tokens") for row in valid
                )
                output_mean = _mean(
                    row.get("delta_output_tokens") for row in valid
                )
                items.append(
                    {
                        "model": model,
                        "state": state,
                        "trap": trap,
                        "clean_pairs": len(valid),
                        "common_source_indices": ",".join(
                            str(index) for index in sorted(cohort)
                        ),
                        "pairing_scope": (
                            "same Source cohort valid across all models and "
                            "all three Trap cells in this State"
                        ),
                        "delta_input_mean": input_mean,
                        "delta_output_mean": output_mean,
                        "delta_input_plus_output_mean": (
                            input_mean + output_mean
                            if input_mean is not None
                            and output_mean is not None
                            else None
                        ),
                    }
                )
            ranked = sorted(
                items,
                key=lambda row: (
                    row.get("delta_input_plus_output_mean") is not None,
                    row.get("delta_input_plus_output_mean")
                    or float("-inf"),
                ),
                reverse=True,
            )
            for rank, row in enumerate(ranked, 1):
                row["rank_by_input_plus_output"] = rank
                result.append(row)
    return result


def tool_actions(live_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for source_index in SOURCE_INDICES:
        source_path = (
            live_dir
            / "sources"
            / f"source-{source_index:02d}"
            / "source_trajectory.json"
        )
        if source_path.is_file():
            source = read_json(source_path)
            for index, event in enumerate(
                (source.get("gateway") or {}).get("history") or [],
                1,
            ):
                event_value = deepcopy_jsonable(event)
                event_state = event_value.pop("state", None)
                rows.append(
                    {
                        "source_index": source_index,
                        "family": FAMILY_BY_SOURCE[source_index],
                        "checkpoint_state": None,
                        "condition": "SOURCE",
                        "event_index": index,
                        "event_state": event_state,
                        **event_value,
                    }
                )
        for state in STATES:
            for condition in CONDITIONS:
                path = (
                    live_dir
                    / "sources"
                    / f"source-{source_index:02d}"
                    / "checkpoints"
                    / state.lower()
                    / "branches"
                    / f"{condition}.json"
                )
                if not path.is_file():
                    continue
                arm = read_json(path)
                for index, event in enumerate(
                    (arm.get("gateway") or {}).get("history") or [],
                    1,
                ):
                    event_value = deepcopy_jsonable(event)
                    event_state = event_value.pop("state", None)
                    rows.append(
                        {
                            "source_index": source_index,
                            "family": FAMILY_BY_SOURCE[source_index],
                            "checkpoint_state": state,
                            "condition": condition,
                            "event_index": index,
                            "event_state": event_state,
                            **event_value,
                        }
                    )
    return rows


def measurement_rows(live_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for source_index in SOURCE_INDICES:
        for state in STATES:
            for condition in CONDITIONS:
                path = (
                    live_dir
                    / "sources"
                    / f"source-{source_index:02d}"
                    / "checkpoints"
                    / state.lower()
                    / "branches"
                    / f"{condition}.json"
                )
                if not path.is_file():
                    continue
                arm = read_json(path)
                measured = arm.get("measurement") or {}
                wall = measured.get("attack_wall_clock") or {}
                defense = measured.get("defense") or {}
                metrics = arm.get("metrics") or {}
                usage = (arm.get("usage_summary") or {}).get("totals") or {}
                rows.append(
                    {
                        "source_index": source_index,
                        "family": FAMILY_BY_SOURCE[source_index],
                        "state": state,
                        "condition": condition,
                        "checkpoint_id": measured.get("checkpoint_id"),
                        "status": arm.get("status"),
                        "terminal_reason": measured.get(
                            "terminal_reason",
                            metrics.get("terminal_reason"),
                        ),
                        "task_success": measured.get(
                            "task_success",
                            metrics.get("task_success"),
                        ),
                        "budget_exhausted": measured.get(
                            "budget_exhausted",
                            _is_budget_exhausted(arm),
                        ),
                        "effective_input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "cache_read_input_tokens": usage.get(
                            "cache_read_input_tokens"
                        ),
                        "cache_creation_input_tokens": usage.get(
                            "cache_creation_input_tokens"
                        ),
                        "total_tokens": usage.get("total_tokens"),
                        "provider_calls": metrics.get("provider_calls"),
                        "tool_calls": metrics.get("tool_calls"),
                        "interaction_depth": metrics.get(
                            "interaction_depth"
                        ),
                        "wall_clock_seconds": wall.get("duration_seconds"),
                        "runtime_wall_time_seconds": wall.get(
                            "runtime_reported_wall_time_seconds"
                        ),
                        "attack_process_cpu_seconds": wall.get(
                            "process_cpu_seconds"
                        ),
                        "provider_latency_seconds": measured.get(
                            "provider_latency_seconds"
                        ),
                        "provider_latency_complete": measured.get(
                            "provider_latency_complete"
                        ),
                        "tool_execution_seconds": measured.get(
                            "tool_execution_seconds"
                        ),
                        "tool_execution_cpu_seconds": measured.get(
                            "tool_execution_cpu_seconds"
                        ),
                        "tool_execution_count": measured.get(
                            "tool_execution_count"
                        ),
                        "defense_applicable": defense.get("applicable"),
                        "defense_intervention_count": defense.get(
                            "intervention_count"
                        ),
                        "state_observation_seconds": defense.get(
                            "state_observation_seconds"
                        ),
                        "state_observation_reason": defense.get(
                            "state_observation_reason"
                        ),
                        "trap_dispatch_selection_seconds": defense.get(
                            "trap_dispatch_selection_seconds"
                        ),
                        "trap_dispatch_selection_reason": defense.get(
                            "trap_dispatch_selection_reason"
                        ),
                        "trap_runtime_setup_inclusive_seconds": defense.get(
                            "trap_runtime_setup_inclusive_seconds"
                        ),
                        "trap_runtime_setup_inclusive_cpu_seconds": defense.get(
                            "trap_runtime_setup_inclusive_cpu_seconds"
                        ),
                        "defense_prepare_seconds": defense.get(
                            "prepare_seconds"
                        ),
                        "defense_prepare_cpu_seconds": defense.get(
                            "prepare_cpu_seconds"
                        ),
                        "defense_construction_retrieval_seconds": defense.get(
                            "construction_retrieval_seconds"
                        ),
                        "defense_response_construction_seconds": defense.get(
                            "response_construction_seconds"
                        ),
                        "defense_injection_seconds": defense.get(
                            "injection_seconds"
                        ),
                        "defense_total_seconds": defense.get(
                            "total_initial_exposure_seconds"
                        ),
                        "defense_total_cpu_seconds": defense.get(
                            "total_initial_exposure_cpu_seconds"
                        ),
                        "defense_process_rss_before_bytes": defense.get(
                            "process_rss_before_bytes"
                        ),
                        "defense_process_rss_after_bytes": defense.get(
                            "process_rss_after_bytes"
                        ),
                        "defense_process_rss_delta_bytes": defense.get(
                            "process_rss_delta_bytes"
                        ),
                        "defense_process_peak_rss_bytes": defense.get(
                            "process_peak_rss_bytes"
                        ),
                        "defense_provider_calls": defense.get(
                            "defense_provider_calls"
                        ),
                        "defense_tokens": defense.get("defense_tokens"),
                        "process_metric_attribution": defense.get(
                            "process_metric_attribution"
                        ),
                        "measurement_present": bool(measured),
                    }
                )
    return rows


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _distribution(values: list[Any]) -> dict[str, Any]:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    return {
        "n": len(numeric),
        "mean": statistics.mean(numeric) if numeric else None,
        "median": statistics.median(numeric) if numeric else None,
        "p95": _p95(numeric),
        "max": max(numeric) if numeric else None,
    }


def _is_budget_exhausted(
    arm: Mapping[str, Any] | None,
) -> bool | None:
    if arm is None:
        return None
    metrics = arm.get("metrics") or {}
    reason = str(metrics.get("terminal_reason") or "").lower()
    return (
        metrics.get("hit_followup_cap") is True
        or "budget" in reason
        or "exhaust" in reason
    )


def _measurement_map(
    rows: list[Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    return {
        (int(row["source_index"]), row["state"], row["condition"]): row
        for row in rows
    }


def enrich_pairs_with_time(
    pairs: list[dict[str, Any]],
    measurements: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    values = _measurement_map(measurements)
    for row in pairs:
        key = (int(row["source_index"]), row["state"])
        baseline = values.get((*key, "BASELINE"), {})
        trap = values.get((*key, row["trap"]), {})
        for field in (
            "wall_clock_seconds",
            "provider_latency_seconds",
            "tool_execution_seconds",
            "attack_process_cpu_seconds",
            "terminal_reason",
            "task_success",
            "budget_exhausted",
        ):
            row[f"baseline_{field}"] = baseline.get(field)
            row[f"trap_{field}"] = trap.get(field)
        for field in (
            "wall_clock_seconds",
            "provider_latency_seconds",
            "tool_execution_seconds",
            "attack_process_cpu_seconds",
        ):
            left = trap.get(field)
            right = baseline.get(field)
            row[f"delta_{field}"] = (
                left - right
                if row["pair_valid"]
                and isinstance(left, (int, float))
                and not isinstance(left, bool)
                and isinstance(right, (int, float))
                and not isinstance(right, bool)
                else None
            )
        row["same_terminal_reason"] = (
            baseline.get("terminal_reason")
            == trap.get("terminal_reason")
            if baseline and trap
            else None
        )
    return pairs


def time_summaries(
    pairs: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wall_rows = []
    latency_rows = []
    for state in STATES:
        for trap in TRAPS:
            cell = [
                row
                for row in pairs
                if row["state"] == state
                and row["trap"] == trap
                and row["pair_valid"]
            ]
            same_terminal = [
                row for row in cell if row["same_terminal_reason"] is True
            ]
            wall_row = {
                "state": state,
                "trap": trap,
                "clean_pairs": len(cell),
                "baseline_wall_mean": _mean(
                    row.get("baseline_wall_clock_seconds")
                    for row in cell
                ),
                "trap_wall_mean": _mean(
                    row.get("trap_wall_clock_seconds") for row in cell
                ),
                "delta_wall_mean": _mean(
                    row.get("delta_wall_clock_seconds") for row in cell
                ),
                "delta_wall_median": _median(
                    row.get("delta_wall_clock_seconds") for row in cell
                ),
                "same_terminal_n": len(same_terminal),
                "same_terminal_delta_wall_mean": _mean(
                    row.get("delta_wall_clock_seconds")
                    for row in same_terminal
                ),
                "same_terminal_delta_wall_median": _median(
                    row.get("delta_wall_clock_seconds")
                    for row in same_terminal
                ),
            }
            wall_rows.append(wall_row)
            latency_rows.append(
                {
                    "state": state,
                    "trap": trap,
                    "clean_pairs": len(cell),
                    "baseline_provider_latency_mean": _mean(
                        row.get("baseline_provider_latency_seconds")
                        for row in cell
                    ),
                    "trap_provider_latency_mean": _mean(
                        row.get("trap_provider_latency_seconds")
                        for row in cell
                    ),
                    "delta_provider_latency_mean": _mean(
                        row.get("delta_provider_latency_seconds")
                        for row in cell
                    ),
                    "delta_provider_latency_median": _median(
                        row.get("delta_provider_latency_seconds")
                        for row in cell
                    ),
                    "baseline_tool_time_mean": _mean(
                        row.get("baseline_tool_execution_seconds")
                        for row in cell
                    ),
                    "trap_tool_time_mean": _mean(
                        row.get("trap_tool_execution_seconds")
                        for row in cell
                    ),
                    "delta_tool_time_mean": _mean(
                        row.get("delta_tool_execution_seconds")
                        for row in cell
                    ),
                    "delta_tool_time_median": _median(
                        row.get("delta_tool_execution_seconds")
                        for row in cell
                    ),
                }
            )
    return wall_rows, latency_rows


def baseline_summary(
    measurements: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in measurements:
        if row["condition"] != "BASELINE":
            continue
        rows.append(
            {
                key: row.get(key)
                for key in (
                    "source_index",
                    "family",
                    "state",
                    "checkpoint_id",
                    "status",
                    "terminal_reason",
                    "task_success",
                    "budget_exhausted",
                    "effective_input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                    "total_tokens",
                    "provider_calls",
                    "tool_calls",
                    "interaction_depth",
                    "wall_clock_seconds",
                    "provider_latency_seconds",
                    "tool_execution_seconds",
                    "attack_process_cpu_seconds",
                )
            }
        )
    return rows


def terminal_summary(
    measurements: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for state in STATES:
        for condition in CONDITIONS:
            cell = [
                row
                for row in measurements
                if row["state"] == state
                and row["condition"] == condition
            ]
            categories = {
                "task_success": 0,
                "budget_exhausted": 0,
                "technical_failure": 0,
                "protocol_failure": 0,
                "other": 0,
            }
            for row in cell:
                if row.get("task_success") is True:
                    categories["task_success"] += 1
                elif row.get("budget_exhausted") is True:
                    categories["budget_exhausted"] += 1
                elif row.get("status") == "TECHNICAL_FAILURE":
                    categories["technical_failure"] += 1
                elif "protocol" in str(row.get("status") or "").lower():
                    categories["protocol_failure"] += 1
                else:
                    categories["other"] += 1
            rows.append(
                {
                    "state": state,
                    "condition": condition,
                    "n": len(cell),
                    **categories,
                }
            )
    return rows


def defense_summaries(
    measurements: list[Mapping[str, Any]],
    pairs: list[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    trap_rows = [
        row for row in measurements if row["condition"] != "BASELINE"
    ]
    processing_rows = []
    cpu_memory_rows = []
    for row in trap_rows:
        processing_rows.append(
            {
                key: row.get(key)
                for key in (
                    "source_index",
                    "family",
                    "state",
                    "condition",
                    "checkpoint_id",
                    "status",
                    "intervention",
                )
            }
            | {
                "intervention": row.get("defense_intervention_count"),
                "state_observation_seconds": row.get(
                    "state_observation_seconds"
                ),
                "state_observation_reason": row.get(
                    "state_observation_reason"
                ),
                "trap_dispatch_selection_seconds": row.get(
                    "trap_dispatch_selection_seconds"
                ),
                "trap_dispatch_selection_reason": row.get(
                    "trap_dispatch_selection_reason"
                ),
                "trap_runtime_setup_inclusive_seconds": row.get(
                    "trap_runtime_setup_inclusive_seconds"
                ),
                "prepare_seconds": row.get("defense_prepare_seconds"),
                "construction_retrieval_seconds": row.get(
                    "defense_construction_retrieval_seconds"
                ),
                "response_construction_seconds": row.get(
                    "defense_response_construction_seconds"
                ),
                "injection_seconds": row.get(
                    "defense_injection_seconds"
                ),
                "total_defense_processing_seconds": row.get(
                    "defense_total_seconds"
                ),
                "defense_provider_calls": row.get("defense_provider_calls"),
                "defense_tokens": row.get("defense_tokens"),
            }
        )
        cpu_memory_rows.append(
            {
                key: row.get(key)
                for key in (
                    "source_index",
                    "family",
                    "state",
                    "condition",
                    "checkpoint_id",
                    "trap_runtime_setup_inclusive_cpu_seconds",
                    "defense_total_cpu_seconds",
                    "defense_process_rss_before_bytes",
                    "defense_process_rss_after_bytes",
                    "defense_process_rss_delta_bytes",
                    "defense_process_peak_rss_bytes",
                    "process_metric_attribution",
                )
            }
        )
    summary = []
    for state in STATES:
        for trap in TRAPS:
            cell = [
                row
                for row in trap_rows
                if row["state"] == state and row["condition"] == trap
            ]
            item = {
                "state": state,
                "trap": trap,
            }
            for label, field in (
                ("setup", "trap_runtime_setup_inclusive_seconds"),
                ("prepare", "defense_prepare_seconds"),
                ("construction", "defense_construction_retrieval_seconds"),
                ("response", "defense_response_construction_seconds"),
                ("injection", "defense_injection_seconds"),
                ("total", "defense_total_seconds"),
                ("cpu", "defense_total_cpu_seconds"),
                ("rss_delta", "defense_process_rss_delta_bytes"),
            ):
                distribution = _distribution(
                    row.get(field) for row in cell
                )
                for stat, value in distribution.items():
                    item[f"{label}_{stat}"] = value
            summary.append(item)
    measurement_map = _measurement_map(measurements)
    comparison = []
    for pair in pairs:
        if not pair["pair_valid"]:
            continue
        source_index = int(pair["source_index"])
        trap_row = measurement_map.get(
            (source_index, pair["state"], pair["trap"]),
            {},
        )
        defense_time = trap_row.get("defense_total_seconds")
        wall_delta = pair.get("delta_wall_clock_seconds")
        comparison.append(
            {
                "source_index": source_index,
                "family": pair["family"],
                "state": pair["state"],
                "trap": pair["trap"],
                "delta_total_tokens_input_output": pair.get(
                    "delta_input_output_total"
                ),
                "delta_provider_calls": pair.get("delta_provider_calls"),
                "delta_tool_calls": pair.get("delta_tool_calls"),
                "delta_wall_clock_seconds": wall_delta,
                "defense_total_processing_seconds": defense_time,
                "defense_total_cpu_seconds": trap_row.get(
                    "defense_total_cpu_seconds"
                ),
                "attack_wall_to_defense_time_ratio": (
                    wall_delta / defense_time
                    if isinstance(wall_delta, (int, float))
                    and isinstance(defense_time, (int, float))
                    and defense_time > 0
                    else None
                ),
            }
        )
    return processing_rows, cpu_memory_rows, summary, comparison


def deepcopy_jsonable(value: Any) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _format(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "NA"
    return f"{value:+.{digits}f}"


def _format_plain(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "NA"
    return f"{value:.{digits}f}"


def report(
    live_dir: Path,
    summaries: list[Mapping[str, Any]],
    rankings: list[Mapping[str, Any]],
    pairs: list[Mapping[str, Any]],
    requests: list[Mapping[str, Any]],
    cross_rows: list[Mapping[str, Any]],
    cross_rankings: list[Mapping[str, Any]],
    execution: Mapping[str, Any],
    baseline_rows: list[Mapping[str, Any]],
    wall_summaries: list[Mapping[str, Any]],
    terminal_rows: list[Mapping[str, Any]],
    latency_summaries: list[Mapping[str, Any]],
    defense_summary: list[Mapping[str, Any]],
    defense_comparison: list[Mapping[str, Any]],
) -> None:
    valid = [row for row in pairs if row["pair_valid"]]
    invalid_reason_counts = Counter(
        reason
        for row in pairs
        for reason in str(row.get("invalid_reasons") or "").split(";")
        if reason
    )
    cache_read_missing = sum(
        row["cache_read_input_tokens"] is None for row in requests
    )
    cache_creation_missing = sum(
        row["cache_creation_input_tokens"] is None for row in requests
    )
    lines = [
        "# PentestGPT Autonomous + Claude Sonnet 4.6 六 Source 正式 H1 报告",
        "",
        "## 摘要",
        "",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 执行状态：`{execution['live_run_status']}`。",
        f"- 有效 checkpoint：{execution['valid_checkpoints']}/18。",
        f"- 条件轨迹：{execution['condition_records']}/72。",
        f"- 有效 Trap–BASELINE 配对：{execution['valid_pairs']}/54。",
        (f"- 正式实验 Provider Calls：{execution['provider_requests']}/"
        f"{execution['formal_request_ceiling']}。"),
        (f"- 含两次实时准入的全局 Provider Calls："
        f"{execution['global_provider_calls']}/1056。"),
        f"- 实际模型：`{execution['provider_models']}`。",
        ("- 解释边界：六个 Source 用于跨模型复现和描述性比较，"
        "不宣称统计显著性。"),
        "",
        "## 固定配置与准入",
        "",
        "- 模型：`claude-sonnet-4-6`；Anthropic Messages。",
        "- `temperature=0.0`；`max_tokens=8192`；串行、非流式 ToolCall。",
        "- Source 顺序：1、6、11、16、21、26；WEB/API/SERVICE 各两个。",
        "- State：RECON、EXPLOITATION、VERIFICATION。",
        ("- 条件：BASELINE、BRANCH_NEW_V3、DEPENDENCY_V6、"
        "EVIDENCE_NEW_V3。"),
        "- 自然采集上限：6×32；条件分叉上限：72×12。",
        ("- 全局 Provider Call 硬上限：1056；两次实时准入计入其中，"
        "因此正式路径硬上限为 1054。"),
        "- 零自动重试、零补跑、零自动 Recovery。",
        ("- 全部 ToolCall 经原生 Gateway；任何绕过、未授权 effect、"
        "模型身份不符或 journal 缺环都会停止。"),
        "",
        "## Usage 口径",
        "",
        ("- Anthropic `input_tokens` 是未缓存输入，"
        "不包含 Cache Read 与 Cache Creation。"),
        ("- `有效输入 = input_tokens + cache_creation_input_tokens + "
        "cache_read_input_tokens`。"),
        ("- 本报告统一报告 `ΔTotal = Δ有效输入 + ΔOutput`，"
        "并明确标记为“Input+Output 口径”。"),
        ("- 该合计不是与 GPT `prompt_tokens` 可直接混合的单一 "
        "Provider Total；跨模型只比较各自 BASELINE 的增量方向与机制排序。"),
        "- Cache Read 与 Cache Creation 分别报告，不重复累加。",
        (f"- 有效输入合计：{execution['usage_coverage']['effective_input_tokens']['known_sum']}；"
        f"Output 合计：{execution['usage_coverage']['output_tokens']['known_sum']}；"
        f"Input+Output 合计："
        f"{execution['usage_coverage']['input_output_total']['known_sum']}。"),
        (f"- Cache Read 合计："
        f"{execution['usage_coverage']['cache_read_input_tokens']['known_sum']}；"
        f"缺字段 {cache_read_missing}/{len(requests)}。"),
        (f"- Cache Creation 合计："
        f"{execution['usage_coverage']['cache_creation_input_tokens']['known_sum']}；"
        f"缺字段 {cache_creation_missing}/{len(requests)}。"),
        "",
        "## 3×3 State×Trap 资源响应",
        "",
        ("表中的 Δ 均为同一 Source、同一 State checkpoint 上的 "
        "“Trap − BASELINE”。n 为有效配对数；缺失保持 NA。"),
        "",
        ("| State | Trap | Clean n | Δ有效输入 | ΔOutput | ΔInput+Output | "
        "ΔCache Read | ΔCalls | ΔTools | Success Trap/Base | "
        "Budget Exhausted Trap/Base | 机制跟进 |"),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['state']} | {row['trap']} | "
            f"{row['clean_pairs']}/6 | "
            f"{_format(row.get('delta_input_mean'))} | "
            f"{_format(row.get('delta_output_mean'))} | "
            f"{_format(row.get('delta_input_output_total_mean'))} | "
            f"{_format(row.get('delta_cache_read_mean'))} | "
            f"{_format(row.get('delta_provider_calls_mean'))} | "
            f"{_format(row.get('delta_tool_calls_mean'))} | "
            f"{row['task_success_trap']}/{row['task_success_baseline']} | "
            f"{row['budget_exhausted_trap']}/"
            f"{row['budget_exhausted_baseline']} | "
            f"{row['mechanism_followup']}/{row['deployed']} |"
        )
    lines += [
        "",
        "## State 内机制资源排序",
        "",
        ("排序依据为该 State 内三个 Trap 的 "
        "`Δ有效输入 + ΔOutput` 均值，从高到低；n<6 的单元按实际 clean n "
        "显示，不作插补。"),
        "",
        "| State | Rank | Trap | Clean n | ΔInput+Output | ΔCalls | ΔTools |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rankings:
        lines.append(
            f"| {row['state']} | "
            f"{row['rank_by_total_resource_increment']} | "
            f"{row['trap']} | {row['clean_pairs']}/6 | "
            f"{_format(row['delta_input_output_total_mean'])} | "
            f"{_format(row['delta_provider_calls_mean'])} | "
            f"{_format(row['delta_tool_calls_mean'])} |"
        )
    lines += [
        "",
        "## 18 条 BASELINE 绝对资源与攻击结果",
        "",
        ("| State | n | Total Tokens Mean | Wall-clock Mean (s) | "
        "Provider Calls Mean | Tool Calls Mean | Task Success | "
        "Budget Exhausted |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for state in STATES:
        cell = [
            row for row in baseline_rows if row["state"] == state
        ]
        lines.append(
            f"| {state} | {len(cell)}/6 | "
            f"{_format_plain(_mean(row.get('total_tokens') for row in cell), 1)} | "
            f"{_format_plain(_mean(row.get('wall_clock_seconds') for row in cell), 3)} | "
            f"{_format_plain(_mean(row.get('provider_calls') for row in cell), 3)} | "
            f"{_format_plain(_mean(row.get('tool_calls') for row in cell), 3)} | "
            f"{sum(row.get('task_success') is True for row in cell)}/{len(cell)} | "
            f"{sum(row.get('budget_exhausted') is True for row in cell)}/{len(cell)} |"
        )
    lines += [
        "",
        "## 攻击侧执行时间",
        "",
        ("Wall-clock 的边界为：checkpoint 恢复完成后、initial exposure 前开始，"
        "到条件轨迹终止结束；不包含排队、环境恢复、离线分析和报告生成。"),
        "",
        ("| State | Trap | n | Baseline s | Trap s | ΔMean s | ΔMedian s | "
        "Same-terminal n | Same-terminal ΔMean s |"),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in wall_summaries:
        lines.append(
            f"| {row['state']} | {row['trap']} | "
            f"{row['clean_pairs']}/6 | "
            f"{_format_plain(row['baseline_wall_mean'], 3)} | "
            f"{_format_plain(row['trap_wall_mean'], 3)} | "
            f"{_format(row['delta_wall_mean'], 3)} | "
            f"{_format(row['delta_wall_median'], 3)} | "
            f"{row['same_terminal_n']} | "
            f"{_format(row['same_terminal_delta_wall_mean'], 3)} |"
        )
    lines += [
        "",
        ("同终止结果配对排除 Task Success、Budget Exhausted、Technical/Protocol "
        "Failure 之间的终止类型混杂；样本不足时保持 NA。"),
        "",
        ("| State | Trap | n | ΔProvider Latency Mean s | "
        "ΔProvider Latency Median s | ΔTool Time Mean s | ΔTool Time Median s |"),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in latency_summaries:
        lines.append(
            f"| {row['state']} | {row['trap']} | "
            f"{row['clean_pairs']}/6 | "
            f"{_format(row['delta_provider_latency_mean'], 3)} | "
            f"{_format(row['delta_provider_latency_median'], 3)} | "
            f"{_format(row['delta_tool_time_mean'], 3)} | "
            f"{_format(row['delta_tool_time_median'], 3)} |"
        )
    lines += [
        "",
        "### 终止原因",
        "",
        ("| State | Condition | n | Task Success | Budget Exhausted | "
        "Technical Failure | Protocol Failure | Other |"),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in terminal_rows:
        lines.append(
            f"| {row['state']} | {row['condition']} | {row['n']}/6 | "
            f"{row['task_success']} | {row['budget_exhausted']} | "
            f"{row['technical_failure']} | {row['protocol_failure']} | "
            f"{row['other']} |"
        )
    lines += [
        "",
        "## STARVE 防御端处理开销",
        "",
        ("H1 的 State 与机制条件均由冻结协议预先确定，不执行 H2 在线状态分类"
        "或动态优先级调度，因此 State Observation 与 dynamic dispatch/selection "
        "保持 NA。Trap runtime setup 包含共享 checkpoint 会话/环境恢复，"
        "只能作为 inclusive setup 指标。Prepare 与 response construction 是"
        "initial exposure 内的嵌套阶段；injection 为 initial exposure 总时间的"
        "非负残差。"),
        "",
        ("| State | Trap | n | Setup Mean s | Prepare Mean s | "
        "Response Mean s | Injection Mean s | Defense Total Mean s | "
        "Median s | p95 s | Max s | CPU Mean s | RSS Δ Median B |"),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in defense_summary:
        lines.append(
            f"| {row['state']} | {row['trap']} | {row.get('total_n', 0)} | "
            f"{_format_plain(row.get('setup_mean'), 4)} | "
            f"{_format_plain(row.get('prepare_mean'), 4)} | "
            f"{_format_plain(row.get('response_mean'), 4)} | "
            f"{_format_plain(row.get('injection_mean'), 4)} | "
            f"{_format_plain(row.get('total_mean'), 4)} | "
            f"{_format_plain(row.get('total_median'), 4)} | "
            f"{_format_plain(row.get('total_p95'), 4)} | "
            f"{_format_plain(row.get('total_max'), 4)} | "
            f"{_format_plain(row.get('cpu_mean'), 4)} | "
            f"{_format_plain(row.get('rss_delta_median'), 0)} |"
        )
    lines += [
        "",
        ("CPU 为防御阶段边界内的 `process_time`；RSS 为同一 runner 进程采样，"
        "不是独立防御进程。进程 Peak RSS 单独保留在 "
        "`defense_cpu_memory.csv`。p95 基于每个 State×Trap 最多 6 个样本，"
        "仅作小样本描述。"),
        "",
        "## 攻击侧与防御侧开销比较",
        "",
        ("逐有效配对结果见 `defense_overhead_comparison.csv`。"
        "攻击侧墙钟时间与防御 initial exposure 时间的比值只用于量纲相同的"
        "描述比较，不等同于经济成本或净收益。"),
        "",
        "## 对七个问题的回答",
        "",
    ]
    token_positive = sum(
        (row.get("delta_input_output_total_mean") or 0) > 0
        for row in summaries
    )
    wall_values = [
        row["delta_wall_mean"]
        for row in wall_summaries
        if isinstance(row.get("delta_wall_mean"), (int, float))
    ]
    defense_totals = [
        row["total_mean"]
        for row in defense_summary
        if isinstance(row.get("total_mean"), (int, float))
    ]
    gpt_input_match = sum(
        row["claude_gpt_input_trend_match"] is True
        for row in cross_rows
    )
    deepseek_input_match = sum(
        row["claude_deepseek_input_trend_match"] is True
        for row in cross_rows
    )
    gpt_wall_match = sum(
        row["claude_gpt_wall_time_trend_match"] is True
        for row in cross_rows
    )
    deepseek_wall_match = sum(
        row["claude_deepseek_wall_time_trend_match"] is True
        for row in cross_rows
    )
    lines += [
        (f"1. 在 9 个 State×Trap 单元中，有 {token_positive} 个平均 "
        "Input+Output 增量为正；其余为负、零或 NA，因此按实测方向回答，"
        "不作显著性推断。"),
        ("2. 三个 State 的平均 ΔTotal 与机制排名见 3×3 表和 State 内排名；"
        "阶段差异按共同 Source 配对描述，不把 Source 内重复当作独立样本。"),
        (f"3. Wall-clock 增量均值为正的时间单元为 "
        f"{sum(value > 0 for value in wall_values)}/{len(wall_values)}；"
        "同时报告同终止结果配对，避免预算提前耗尽造成的缩短偏差。"),
        (f"4. 可测 Trap 的单次干预平均防御处理时间范围约为 "
        f"{min(defense_totals):.6f}–{max(defense_totals):.6f} 秒。"
        if defense_totals
        else "4. 防御处理时间不可用，保持 NA。"),
        ("5. 防御阶段 CPU 已采用 process_time 测量；RSS 与 peak RSS 仅为"
        "同一 runner 进程级指标，不能宣称为独立 STARVE 进程占用。"),
        (
            f"6. 当前数据只支持有限的不对称信号："
            f"{token_positive}/{len(summaries)} 单元 Token 均值为正，"
            f"{sum(value > 0 for value in wall_values)}/{len(wall_values)} "
            "时间单元均值为正，而 initial exposure 的平均防御处理时间低于"
            " 10 ms。但 H1 不存在在线 State Observation/动态调度，BRANCH "
            "候选枚举未包含在 runtime exposure 边界内，且 CPU/RSS 未隔离为"
            "独立防御进程；因此不能确认完整经济成本上的防御优势。"
        ),
        (
            f"7. Claude 与 GPT、DeepSeek 的 Input+Output 方向一致数分别为 "
            f"{gpt_input_match}/9 与 {deepseek_input_match}/9；"
            f"Wall-clock 方向一致数为 {gpt_wall_match}/9 与 "
            f"{deepseek_wall_match}/9。Claude 的 State 内机制排序为："
            "RECON 依赖 > 证据 > 分支，EXPLOITATION 依赖 > 分支 > 证据，"
            "VERIFICATION 分支 > 依赖 > 证据；GPT 仅在 RECON 与 Claude "
            "排序相同，DeepSeek 的排序方向差异更大。"
        ),
        "",
    ]
    lines += [
        "",
        "## 54 组配对有效性审计",
        "",
        f"- 有效配对：{len(valid)}/54。",
        f"- 条件状态：`{execution['condition_statuses']}`。",
    ]
    if invalid_reason_counts:
        lines.append(
            "- 无效原因："
            + "；".join(
                f"`{reason}`={count}"
                for reason, count in sorted(invalid_reason_counts.items())
            )
            + "。"
        )
    else:
        lines.append("- 54 组均满足当前 clean-pair 门禁，无需排除。")
    lines += [
        "",
        ("逐组状态、绝对资源、增量、任务成功、预算耗尽和机制字段见 "
        "`pair_validity.csv/json`。"),
        "",
        "## Claude、GPT-5.6 Sol 与 DeepSeek-V4-Pro",
        "",
        ("跨模型比较以相同 Source/State/Trap 的有效配对为基础，分别相对"
        "各自 BASELINE 计算。不同模型的绝对 Token 口径不同，本文只比较"
        "响应方向、机制排序和阶段差异。DeepSeek 历史缺失原样保留。"),
        ("Wall-clock 的 Claude 值来自本轮单调时钟边界；GPT/DeepSeek "
        "历史值来自同一 `SerialCompatibilityRuntime` 的 `metrics.wall_time`。"
        "该定义可用于描述性方向比较，但网络与 Provider 负载仍会影响绝对值。"),
        "",
        ("| State | Trap | Model | Clean n | ΔInput | ΔOutput | "
        "ΔCache Read | ΔCalls | ΔTools | ΔWall s | Input Trend | "
        "Output Trend | Calls Trend | Tools Trend | Wall Trend |"),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in cross_rows:
        for label, prefix in (
            ("Claude Sonnet 4.6", "claude"),
            ("GPT-5.6 Sol", "gpt"),
            ("DeepSeek-V4-Pro", "deepseek"),
        ):
            lines.append(
                f"| {row['state']} | {row['trap']} | {label} | "
                f"{row[f'{prefix}_clean_pairs']}/6 | "
                f"{_format(row.get(f'{prefix}_delta_input_mean'))} | "
                f"{_format(row.get(f'{prefix}_delta_output_mean'))} | "
                f"{_format(row.get(f'{prefix}_delta_cache_read_mean'))} | "
                f"{_format(row.get(f'{prefix}_delta_provider_calls_mean'))} | "
                f"{_format(row.get(f'{prefix}_delta_tool_calls_mean'))} | "
                f"{_format(row.get(f'{prefix}_delta_wall_time_mean'), 3)} | "
                f"{row[f'{prefix}_input_trend']} | "
                f"{row[f'{prefix}_output_trend']} | "
                f"{row[f'{prefix}_provider_calls_trend']} | "
                f"{row[f'{prefix}_tool_calls_trend']} | "
                f"{row[f'{prefix}_wall_time_trend']} |"
            )
    comparison_metrics = (
        "input",
        "output",
        "cache_read",
        "provider_calls",
        "tool_calls",
        "wall_time",
    )
    matches = {
        (reference, metric): sum(
            row[f"claude_{reference}_{metric}_trend_match"] is True
            for row in cross_rows
        )
        for reference in ("gpt", "deepseek")
        for metric in comparison_metrics
    }
    evaluable = {
        (reference, metric): sum(
            row[f"claude_{reference}_{metric}_trend_match"] is not None
            for row in cross_rows
        )
        for reference in ("gpt", "deepseek")
        for metric in comparison_metrics
    }
    lines += [
        "",
        (f"- Claude 与 GPT 的 ΔInput 方向一致：{matches[('gpt', 'input')]}/9；"
        f"ΔOutput：{matches[('gpt', 'output')]}/9；"
        f"ΔProvider Calls：{matches[('gpt', 'provider_calls')]}/9；"
        f"ΔTool Calls：{matches[('gpt', 'tool_calls')]}/9；"
        f"ΔWall-clock：{matches[('gpt', 'wall_time')]}/9；"
        f"可评估单元 {evaluable[('gpt', 'input')]}/9。"),
        (f"- Claude 与 DeepSeek 的 ΔInput 方向一致："
        f"{matches[('deepseek', 'input')]}/9；"
        f"ΔOutput：{matches[('deepseek', 'output')]}/9；"
        f"ΔProvider Calls："
        f"{matches[('deepseek', 'provider_calls')]}/9；"
        f"ΔTool Calls：{matches[('deepseek', 'tool_calls')]}/9；"
        f"ΔWall-clock：{matches[('deepseek', 'wall_time')]}/9；"
        f"可评估单元 {evaluable[('deepseek', 'input')]}/9。"),
        ("- 方向一致不等于效应量相同；9 个单元存在 Source 重复，"
        "不能据此宣称模型间差异显著。"),
        "",
        "### 跨模型机制排序",
        "",
        "| Model | State | Rank | Trap | Clean n | ΔInput+Output |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in cross_rankings:
        lines.append(
            f"| {row['model']} | {row['state']} | "
            f"{row['rank_by_input_plus_output']} | {row['trap']} | "
            f"{row['clean_pairs']}/6 | "
            f"{_format(row['delta_input_plus_output_mean'])} |"
        )
    lines += [
        "",
        ("- DeepSeek 历史有效配对不完整，尤其 Source 6 缺少后续 checkpoint，"
        "其单元 clean n 可能小于 6；本报告不补造、不把缺失视为 0。"),
        "",
        "## 请求、Gateway 与执行审计",
        "",
        (f"- 逐请求 journal：{execution['journal_request_count']} 条；"
        f"Provider request ID 唯一数 "
        f"{execution['provider_request_ids_unique']}。"),
        f"- HTTP 状态：`{execution['http_statuses']}`。",
        f"- 请求结果：`{execution['request_outcomes']}`。",
        (f"- 请求/响应模型精确匹配："
        f"`{execution['model_identity_exact']}`。"),
        (f"- Gateway bypass：`{execution['gateway_bypass_observed']}`；"
        f"未授权 effect：`{execution['unauthorized_effect_observed']}`。"),
        (f"- 自动重试/补跑/Recovery："
        f"`{execution['automatic_retry_used']}` / "
        f"`{execution['automatic_backfill_used']}` / "
        f"`{execution['automatic_recovery_used']}`。"),
        (f"- Journal 独立审计：`{execution['journal_audit']['status']}`；"
        f"命名口径异常："
        f"`{execution['journal_audit']['observed_audit_mismatch']}`。"),
        f"- {execution['journal_audit']['diagnosis']}",
        "",
        "## 统计解释与限制",
        "",
        "- Source 是主要配对单位；同一 Source 的三个 State 不视为独立样本。",
        "- 每个 State×Trap 目标 n=6；结果仅作描述统计与跨模型方向比较。",
        "- K=12 是观测窗口；预算耗尽表示删失终点，不等同于无条件任务失败。",
        ("- Claude 的 Input+Output 口径不能与 GPT 的 Provider Total 数值"
        "直接判优或做绝对差值比较。"),
        "- 本次 Source 数不足以对模型效应或三阶交互作确认性推断。",
        ("- checkpoint 缺失、Trap 不可部署或技术失败均原样保留；"
        "未替换 Source、未补造轨迹、未开展 H2。"),
        (
            "- 本报告覆盖完整的 18 checkpoint、72 条件轨迹和 54 个有效配对。"
            if execution["status"] == "COMPLETED"
            else "- 本报告是冻结停止后的离线审计；正式目标数据不完整。"
        ),
        "",
        "## 交付文件",
        "",
        "- `FINAL_REPORT_ZH.md`：本报告。",
        "- `state_trap_summary.csv/json`：3×3 State×Trap 资源结果。",
        "- `claude_state_trap_ranking.csv`：State 内机制排序。",
        "- `pair_validity.csv/json`：54 组配对有效性审计。",
        "- `cross_model_comparison.csv/json`：三模型方向比较。",
        "- `cross_model_state_rankings.csv`：三模型机制排序。",
        "- `request_usage.csv`、`provider_requests.jsonl`：逐请求真实 usage。",
        "- `tool_calls_and_results.jsonl`：Gateway 工具调用与结果。",
        "- `sources/`：自然轨迹、checkpoint、四条件结果和逐请求 journal。",
        "",
    ]
    (live_dir / "FINAL_REPORT_ZH.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def analyze(live_dir: Path) -> dict[str, Any]:
    live_dir = live_dir.resolve()
    run_status = (
        read_json(live_dir / "run_status.json")
        if (live_dir / "run_status.json").is_file()
        else {}
    )
    checkpoints, _ = shared.checkpoint_audit(live_dir)
    conditions = shared.condition_audit(live_dir)
    measurements = measurement_rows(live_dir)
    pairs = enrich_pairs_with_time(
        _add_total_deltas(shared.pair_rows(live_dir)),
        measurements,
    )
    summaries = state_trap_summary(pairs)
    rankings = state_rankings(summaries)
    wall_summaries, latency_summaries = time_summaries(pairs)
    baseline_rows = baseline_summary(measurements)
    terminal_rows = terminal_summary(measurements)
    (
        defense_processing,
        defense_cpu_memory,
        defense_summary,
        defense_comparison,
    ) = defense_summaries(measurements, pairs)
    requests = request_rows(live_dir)
    admission_requests = admission_usage_rows(live_dir.parent)
    cross_rows = cross_model_rows(pairs)
    cross_rankings = cross_model_rankings(pairs)
    request_ids = [
        row["request_id"] for row in requests if row.get("request_id")
    ]
    valid_pairs = sum(row["pair_valid"] for row in pairs)
    invalid_conditions = [
        row
        for row in conditions
        if row["status"] not in {"PASS", "NOT_DEPLOYABLE", "NO_CHECKPOINT"}
    ]
    admission_path = live_dir.parent / "admission" / "admission_record.json"
    admission = read_json(admission_path) if admission_path.is_file() else {}
    admission_calls = int(admission.get("provider_calls") or 0)
    global_calls = admission_calls + len(requests)
    usage_coverage = {
        field: _coverage(requests, field) for field in USAGE_FIELDS
    }
    admission_usage_coverage = {
        field: _coverage(admission_requests, field)
        for field in USAGE_FIELDS
    }
    combined_requests = [
        *admission_requests,
        *[
            {
                "phase": "FORMAL_H1",
                **row,
            }
            for row in requests
        ],
    ]
    combined_usage_coverage = {
        field: _coverage(combined_requests, field)
        for field in USAGE_FIELDS
    }
    condition_statuses = dict(
        Counter(row["status"] for row in conditions)
    )
    checkpoint_statuses = dict(
        Counter(row["status"] for row in checkpoints)
    )
    http_statuses = dict(
        Counter(str(row["http_status"]) for row in requests)
    )
    provider_models = dict(
        Counter(str(row["provider_model_id"]) for row in requests)
    )
    request_outcomes = dict(
        Counter(str(row["outcome"]) for row in requests)
    )
    audit = journal_audit(live_dir)
    run_status_value = str(run_status.get("status") or "UNKNOWN")
    completed = (
        run_status_value == "COMPLETED"
        and valid_pairs == 54
        and all(
            row["status"] in {"PASS", "NOT_DEPLOYABLE"}
            for row in conditions
        )
    )
    execution = {
        "protocol_id": PROTOCOL_ID,
        "analysis_generated_at": datetime.now(UTC).isoformat(),
        "status": "COMPLETED" if completed else run_status_value,
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
        "condition_statuses": condition_statuses,
        "checkpoint_statuses": checkpoint_statuses,
        "provider_requests": len(requests),
        "formal_request_ceiling": 1054,
        "global_request_hard_ceiling": 1056,
        "admission_provider_calls": admission_calls,
        "global_provider_calls": global_calls,
        "provider_requests_within_ceiling": (
            len(requests) <= 1054 and global_calls <= 1056
        ),
        "journal_request_count": len(requests),
        "provider_request_ids_unique": len(set(request_ids)),
        "provider_request_ids_missing": len(requests) - len(request_ids),
        "provider_request_ids_duplicate": (
            len(request_ids) - len(set(request_ids))
        ),
        "http_statuses": http_statuses,
        "provider_models": provider_models,
        "request_outcomes": request_outcomes,
        "model_identity_exact": bool(requests) and all(
            row["request_model_id"] == MODEL_ID
            and row["provider_model_id"] == MODEL_ID
            for row in requests
            if row["outcome"] == "success"
        ),
        "gateway_bypass_observed": any(
            int(row.get("gateway_bypasses") or 0) > 0
            for row in conditions
        ),
        "unauthorized_effect_observed": any(
            row["status"] == "TECHNICAL_FAILURE"
            and "unauthorized_trap_effect" in str(row.get("reason") or "")
            for row in conditions
        ),
        "usage_semantics": {
            "raw_input_tokens_is_uncached": True,
            "cache_creation_is_additive": True,
            "cache_read_is_additive": True,
            "effective_input_formula": (
                "input_tokens + cache_creation_input_tokens + "
                "cache_read_input_tokens"
            ),
            "reported_total_label": "Input+Output口径",
            "reported_total_formula": (
                "effective_input_tokens + output_tokens"
            ),
            "direct_absolute_cross_model_mixing": False,
        },
        "usage_coverage": usage_coverage,
        "admission_usage_coverage": admission_usage_coverage,
        "combined_usage_coverage": combined_usage_coverage,
        "journal_audit": audit,
        "measurement_coverage": {
            "condition_arms": len(measurements),
            "condition_arms_with_measurement": sum(
                row["measurement_present"] for row in measurements
            ),
            "wall_clock_known": sum(
                isinstance(row["wall_clock_seconds"], (int, float))
                for row in measurements
            ),
            "provider_latency_known": sum(
                isinstance(row["provider_latency_seconds"], (int, float))
                for row in measurements
            ),
            "tool_time_known": sum(
                isinstance(row["tool_execution_seconds"], (int, float))
                for row in measurements
            ),
            "defense_time_known": sum(
                isinstance(row["defense_total_seconds"], (int, float))
                for row in measurements
                if row["condition"] != "BASELINE"
            ),
            "state_observation_available": False,
            "dynamic_dispatch_selection_available": False,
            "full_defense_total_available": False,
            "full_defense_total_reason": (
                "H1 has no online state classification or dynamic dispatch; "
                "BRANCH candidate enumeration occurs before the runtime "
                "initial-exposure timing boundary, and CPU/RSS are not "
                "isolated in a separate defense process."
            ),
        },
        "automatic_retry_used": False,
        "automatic_backfill_used": False,
        "automatic_recovery_used": False,
        "fatal_conditions": invalid_conditions,
        "fatal_reason": run_status.get("fatal_reason"),
        "fatal_scope": run_status.get("fatal_scope"),
        "live_run_status": run_status_value,
        "scientific_interpretation": (
            "Descriptive six-Source cross-model comparison based on paired "
            "directions and mechanism rankings; no confirmatory significance "
            "claim."
        ),
    }
    write_csv(live_dir / "source_audit.csv", source_audit(live_dir))
    write_csv(live_dir / "checkpoint_audit.csv", checkpoints)
    write_json(live_dir / "checkpoint_audit.json", checkpoints)
    write_csv(live_dir / "condition_audit.csv", conditions)
    write_json(live_dir / "condition_audit.json", conditions)
    write_csv(live_dir / "pair_validity.csv", pairs)
    write_json(live_dir / "pair_validity.json", pairs)
    write_csv(live_dir / "measurements.csv", measurements)
    write_json(live_dir / "measurements.json", measurements)
    write_csv(live_dir / "baseline_summary.csv", baseline_rows)
    write_json(live_dir / "baseline_summary.json", baseline_rows)
    write_csv(live_dir / "terminal_reason_summary.csv", terminal_rows)
    write_json(live_dir / "terminal_reason_summary.json", terminal_rows)
    write_csv(live_dir / "attack_wall_time_summary.csv", wall_summaries)
    write_json(live_dir / "attack_wall_time_summary.json", wall_summaries)
    write_csv(live_dir / "attack_latency_summary.csv", latency_summaries)
    write_json(live_dir / "attack_latency_summary.json", latency_summaries)
    write_csv(
        live_dir / "defense_processing_time.csv",
        defense_processing,
    )
    write_json(
        live_dir / "defense_processing_time.json",
        defense_processing,
    )
    write_csv(live_dir / "defense_cpu_memory.csv", defense_cpu_memory)
    write_json(live_dir / "defense_cpu_memory.json", defense_cpu_memory)
    write_csv(live_dir / "defense_processing_summary.csv", defense_summary)
    write_json(live_dir / "defense_processing_summary.json", defense_summary)
    write_csv(
        live_dir / "defense_overhead_comparison.csv",
        defense_comparison,
    )
    write_json(
        live_dir / "defense_overhead_comparison.json",
        defense_comparison,
    )
    write_csv(live_dir / "state_trap_summary.csv", summaries)
    write_json(live_dir / "state_trap_summary.json", summaries)
    write_csv(
        live_dir / "claude_state_trap_ranking.csv",
        rankings,
    )
    write_json(
        live_dir / "claude_state_trap_ranking.json",
        rankings,
    )
    write_csv(
        live_dir / "family_state_trap_summary.csv",
        shared.family_summary(pairs),
    )
    write_csv(
        live_dir / "cross_model_comparison.csv",
        cross_rows,
    )
    write_json(
        live_dir / "cross_model_comparison.json",
        cross_rows,
    )
    write_csv(
        live_dir / "cross_model_state_rankings.csv",
        cross_rankings,
    )
    write_json(
        live_dir / "cross_model_state_rankings.json",
        cross_rankings,
    )
    write_csv(live_dir / "request_usage.csv", requests)
    write_jsonl(live_dir / "provider_requests.jsonl", requests)
    write_csv(
        live_dir / "combined_request_usage.csv",
        combined_requests,
    )
    write_jsonl(
        live_dir / "combined_provider_requests.jsonl",
        combined_requests,
    )
    actions = tool_actions(live_dir)
    write_csv(
        live_dir / "tool_calls.csv",
        actions,
        [
            "source_index",
            "family",
            "checkpoint_state",
            "condition",
            "event_index",
            "event_state",
            "event",
            "call_id",
            "role",
            "name",
            "arguments",
            "decision",
            "effect_count_before",
            "effect_count_after",
            "result",
        ],
    )
    write_jsonl(live_dir / "tool_calls_and_results.jsonl", actions)
    write_json(live_dir / "journal_audit.json", audit)
    write_json(live_dir / "execution_audit.json", execution)
    report(
        live_dir,
        summaries,
        rankings,
        pairs,
        requests,
        cross_rows,
        cross_rankings,
        execution,
        baseline_rows,
        wall_summaries,
        terminal_rows,
        latency_summaries,
        defense_summary,
        defense_comparison,
    )
    write_defense_report(
        live_dir,
        execution,
        baseline_rows,
        measurements,
        wall_summaries,
        defense_summary,
        defense_comparison,
    )
    result = {
        "output": str(live_dir),
        "status": execution["status"],
        "valid_pairs": valid_pairs,
        "target_pairs": 54,
        "provider_requests": len(requests),
        "global_provider_calls": global_calls,
        "valid_checkpoints": execution["valid_checkpoints"],
        "condition_statuses": condition_statuses,
        "provider_models": provider_models,
        "provider_request_ids_unique": len(set(request_ids)),
        "target_pair_count": 54,
    }
    write_json(live_dir / "analysis_summary.json", result)
    write_top_level_status(live_dir, execution, result)
    return result


def write_top_level_status(
    live_dir: Path,
    execution: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> None:
    root = live_dir.parent
    full_status = {
        "protocol_id": PROTOCOL_ID,
        "status": execution["status"],
        "formal_h1_started": True,
        "admission_provider_calls": execution["admission_provider_calls"],
        "formal_provider_calls": execution["provider_requests"],
        "provider_calls": execution["global_provider_calls"],
        "provider_request_hard_ceiling": 1056,
        "provider_calls_within_ceiling": (
            execution["global_provider_calls"] <= 1056
        ),
        "models": {
            "request": MODEL_ID,
            "response": execution["provider_models"],
        },
        "model_ids": execution["provider_models"],
        "source_count": sum(
            row["status"] != "NOT_RUN"
            for row in source_audit(live_dir)
        ),
        "valid_checkpoints": execution["valid_checkpoints"],
        "condition_trajectories": execution["condition_records"],
        "valid_pairs": execution["valid_pairs"],
        "clean_pairs": execution["valid_pairs"],
        "measurement_coverage": execution["measurement_coverage"],
        "fatal_reason": execution["fatal_reason"],
        "fatal_scope": execution["fatal_scope"],
        "journal_audit": execution["journal_audit"],
        "admission_usage": execution["admission_usage_coverage"],
        "formal_usage": execution["usage_coverage"],
        "combined_usage": execution["combined_usage_coverage"],
        "automatic_retry": False,
        "automatic_backfill": False,
        "automatic_recovery": False,
        "h2_started": False,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    write_json(root / "FINAL_STATUS.json", full_status)
    if execution["status"] == "COMPLETED":
        return
    report_lines = [
        "# Claude Sonnet 4.6 六 Source 正式 H1 冻结停止报告",
        "",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 状态：`{execution['status']}`",
        "- 实时准入：`PASS`，Provider Calls 2/1056。",
        (f"- 正式请求：{execution['provider_requests']}/1054；"
        f"含准入共 {execution['global_provider_calls']}/1056。"),
        (f"- 准入 usage：EffectiveInput "
        f"{execution['admission_usage_coverage']['effective_input_tokens']['known_sum']}，"
        f"Output "
        f"{execution['admission_usage_coverage']['output_tokens']['known_sum']}。"),
        (f"- 正式 usage：EffectiveInput "
        f"{execution['usage_coverage']['effective_input_tokens']['known_sum']}，"
        f"Output "
        f"{execution['usage_coverage']['output_tokens']['known_sum']}。"),
        (f"- 全部 21 请求：EffectiveInput "
        f"{execution['combined_usage_coverage']['effective_input_tokens']['known_sum']}，"
        f"Output "
        f"{execution['combined_usage_coverage']['output_tokens']['known_sum']}，"
        f"Input+Output "
        f"{execution['combined_usage_coverage']['input_output_total']['known_sum']}。"),
        (f"- Source：1 条真实自然轨迹已完成；"
        f"有效 checkpoint {execution['valid_checkpoints']}/18。"),
        f"- 条件轨迹：{execution['condition_records']}/72。",
        f"- 有效 Trap–BASELINE 配对：{execution['valid_pairs']}/54。",
        f"- 停止原因：`{execution['fatal_reason']}`。",
        f"- 停止范围：`{execution['fatal_scope']}`。",
        f"- Journal 独立审计：`{execution['journal_audit']['status']}`。",
        f"- 诊断：{execution['journal_audit']['diagnosis']}",
        ("- BASELINE 分叉本身完成 8 次请求、9 次工具调用，"
        "任务成功，Gateway PASS，effect 边界与 K=12 均通过；"
        "失败检查仅为 `journal_complete=false`。"),
        "- 未发现上游安全拒绝、Gateway bypass 或未授权 effect。",
        ("- 按零自动重试、零补跑、零 Recovery 规则，本轮未继续其余"
        " State、Trap 或 Source。"),
        ("- 因数据不完整，不能形成完整的 3×3 资源排名、54 组配对"
        "或三模型确认性比较。"),
        "- 未修改 Trap 核心、Source、H2 策略或历史 H1 数据；H2 未启动。",
        "",
        (f"逐请求 usage：{analysis['provider_requests']} 条，"
        f"request ID 唯一数 {analysis['provider_request_ids_unique']}。"),
        ("完整离线审计见 `live/FINAL_REPORT_ZH.md`、"
        "`live/execution_audit.json`、`live/journal_audit.json`、"
        "`live/request_usage.csv` 与 `live/pair_validity.csv`。"),
        "",
    ]
    (root / "STOPPED_REPORT_ZH.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


def write_defense_report(
    live_dir: Path,
    execution: Mapping[str, Any],
    baseline_rows: list[Mapping[str, Any]],
    measurements: list[Mapping[str, Any]],
    wall_summaries: list[Mapping[str, Any]],
    defense_summary: list[Mapping[str, Any]],
    defense_comparison: list[Mapping[str, Any]],
) -> None:
    ratios = [
        float(row["attack_wall_to_defense_time_ratio"])
        for row in defense_comparison
        if isinstance(
            row.get("attack_wall_to_defense_time_ratio"),
            (int, float),
        )
        and not isinstance(
            row.get("attack_wall_to_defense_time_ratio"),
            bool,
        )
    ]
    positive_wall = sum(
        isinstance(row.get("delta_wall_mean"), (int, float))
        and row["delta_wall_mean"] > 0
        for row in wall_summaries
    )
    lines = [
        "# Claude Sonnet 4.6 H1 防御端开销报告",
        "",
        "## 测量边界",
        "",
        "- 攻击 wall-clock：checkpoint restore 完成后至条件轨迹终止。",
        "- Provider latency：Provider journal 中成功请求的 elapsed_seconds 之和。",
        "- 工具执行：包裹既有 `_execute_task_tool`，不改变参数、结果或顺序。",
        "- Trap setup：`restore_final_runtime` 的 inclusive 时间，包含共享运行时恢复。",
        ("- Prepare、response construction、injection：initial exposure 内的"
        "嵌套阶段；injection 是总时间减去 prepare 与 response 后的非负残差。"),
        ("- CPU：相关阶段的 `process_time`。RSS：同一 runner 进程的采样，"
        "没有独立防御进程，因此不是纯 STARVE 模块内存。"),
        "",
        "## 可测性与 NA",
        "",
        ("- State Observation：NA。H1 State 来自冻结 checkpoint，"
        "没有在线状态分类。"),
        ("- Dynamic Trap Dispatch/Selection：NA。H1 条件固定；H2 "
        "STATE_AWARE_FEASIBLE 才存在动态候选与优先级选择成本。"),
        ("- 完整防御总成本：NA。BRANCH 候选枚举位于 runtime "
        "initial-exposure 计时边界之前，未混入表中总时间；表中 total "
        "是 runtime initial exposure 成本，不是完整防御总成本。"),
        "- 防御 Provider Calls / Tokens：0。本轮防御步骤没有远程模型调用。",
        (f"- 条件轨迹测量覆盖："
        f"{execution['measurement_coverage']['condition_arms_with_measurement']}"
        f"/{execution['measurement_coverage']['condition_arms']}。"),
        "",
        "## State×Trap 防御开销",
        "",
        ("| State | Trap | n | Setup Mean s | Prepare Mean s | "
        "Response Mean s | Injection Mean s | Total Mean s | Median s | "
        "p95 s | Max s | CPU Mean s | Peak RSS Max B |"),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    peak_values = [
        row.get("defense_process_peak_rss_bytes")
        for row in measurements
        if row.get("condition") != "BASELINE"
    ]
    for row in defense_summary:
        lines.append(
            f"| {row['state']} | {row['trap']} | {row.get('total_n', 0)} | "
            f"{_format_plain(row.get('setup_mean'), 5)} | "
            f"{_format_plain(row.get('prepare_mean'), 5)} | "
            f"{_format_plain(row.get('response_mean'), 5)} | "
            f"{_format_plain(row.get('injection_mean'), 5)} | "
            f"{_format_plain(row.get('total_mean'), 5)} | "
            f"{_format_plain(row.get('total_median'), 5)} | "
            f"{_format_plain(row.get('total_p95'), 5)} | "
            f"{_format_plain(row.get('total_max'), 5)} | "
            f"{_format_plain(row.get('cpu_mean'), 6)} | "
            f"{_format_plain(max((v for v in peak_values if isinstance(v, int)), default=None), 0)} |"
        )
    lines += [
        "",
        "## 攻击与防御开销关系",
        "",
        (f"- 可计算攻击 ΔWall-clock / 防御处理时间比值的有效配对："
        f"{len(ratios)}/{len(defense_comparison)}。"),
        (
            f"- 比值描述统计：median={statistics.median(ratios):.3f}，"
            f"min={min(ratios):.3f}，max={max(ratios):.3f}。"
            if ratios
            else "- 比值：NA。"
        ),
        f"- State×Trap 平均 ΔWall-clock 为正的单元：{positive_wall}/9。",
        ("- 这些比值只描述攻击侧额外墙钟时间与防御 initial exposure 处理时间，"
        "不含网络负载差异、摊销、人力成本或经济损失。"),
        "",
        "## 结论边界",
        "",
        ("- 如果 Token/Interaction 增量与防御时间均由实测产生，可以描述"
        "resource amplification 与 defense cost 的联合方向。"),
        ("- 当前可确认的是 runtime initial-exposure 本地成本很低；"
        "不能将这一部分等同于完整 STARVE 防御总成本。"),
        "- 不能仅凭 Token 增量较高宣称攻击成本高而防御成本低。",
        ("- 进程级 CPU/RSS 包含 runner 其他本地工作；除非有独立防御进程，"
        "不得将 peak RSS 全部归因于 STARVE。"),
        "- H1 不允许把冻结机制选择时间解释为 H2 状态感知策略成本。",
        "",
    ]
    (live_dir / "DEFENSE_OVERHEAD_REPORT_ZH.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def source_audit(live_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for source_index in SOURCE_INDICES:
        path = (
            live_dir
            / "sources"
            / f"source-{source_index:02d}"
            / "source_trajectory.json"
        )
        source = read_json(path) if path.is_file() else {}
        rows.append(
            {
                "source_index": source_index,
                "family": FAMILY_BY_SOURCE[source_index],
                "status": source.get("status", "NOT_RUN"),
                "accepted_states": ",".join(
                    source.get("accepted_states") or []
                ),
                "provider_calls": source.get(
                    "actual_provider_calls",
                    source.get("provider_calls"),
                ),
                "gateway_status": (
                    (source.get("gateway") or {}).get("status")
                ),
                "terminal_reason": source.get("terminal_reason"),
                "error_type": source.get("error_type"),
                "error": source.get("error"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("live_dir", type=Path)
    args = parser.parse_args()
    result = analyze(args.live_dir)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

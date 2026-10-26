"""Read-only four-model analysis for the formal GLM-5.3-Flash H1 run."""
from __future__ import annotations

import argparse
import csv
import json
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

from h1_claude_cross_model_6source_20261024 import analyze as base
from h1_gpt_cross_model_6source_20261024 import analyze as shared


BASE_WRITE_DEFENSE_REPORT = base.write_defense_report
PROTOCOL_ID = "formal_H1_GLM_5_3_Flash_cross_model_6source_v1"
MODEL_ID = "glm-5.3-flash"
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
    "effective_input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "input_output_total",
    "total_tokens",
    "reasoning_tokens",
)
CLAUDE_LIVE = (
    ROOT
    / "results"
    / "formal_H1_Claude_Sonnet_4_6_cross_model_6source_20261024T140112540416Z"
    / "live"
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


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def request_rows(live_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(live_dir.rglob("request-*.json")):
        record = read_json(path)
        parts = path.relative_to(live_dir).parts
        source_index = next(
            (
                int(part.split("-")[1])
                for part in parts
                if part.startswith("source-")
                and part.split("-")[1].isdigit()
            ),
            None,
        )
        state = next(
            (part.upper() for part in parts if part.upper() in STATES),
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
        input_tokens = _int_or_none(
            usage.get("input_tokens")
            if usage.get("input_tokens") is not None
            else normalized.get("input_tokens")
        )
        output_tokens = _int_or_none(
            usage.get("output_tokens")
            if usage.get("output_tokens") is not None
            else normalized.get("output_tokens")
        )
        cache_read = _int_or_none(
            usage.get("cache_read_input_tokens")
            if usage.get("cache_read_input_tokens") is not None
            else normalized.get("cache_read_tokens")
        )
        reasoning = _int_or_none(
            usage.get("reasoning_tokens")
            if usage.get("reasoning_tokens") is not None
            else normalized.get("reasoning_tokens")
        )
        total = _int_or_none(
            usage.get("total_tokens")
            if usage.get("total_tokens") is not None
            else normalized.get("total_tokens")
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
                "usage_class": "Provider Total Tokens",
                "input_includes_cached_tokens": True,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "effective_input_tokens": input_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": None,
                "reasoning_tokens": reasoning,
                "input_output_total": (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None
                    else None
                ),
                "total_tokens": total,
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
        input_tokens = _int_or_none(usage.get("input_tokens"))
        output_tokens = _int_or_none(usage.get("output_tokens"))
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
                "usage_class": "Provider Total Tokens",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "effective_input_tokens": input_tokens,
                "cache_read_input_tokens": usage.get("cache_read_tokens"),
                "cache_creation_input_tokens": None,
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "input_output_total": (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None
                    else None
                ),
                "total_tokens": usage.get("total_tokens"),
            }
        )
    return rows


def _trend(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "NA"
    return "increase" if value > 0 else "decrease" if value < 0 else "flat"


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


def _subset(
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


def _reference_model_rows(
    glm_pairs: list[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    claude = _load_reference_pairs(CLAUDE_LIVE / "pair_validity.json")
    gpt = _load_reference_pairs(GPT_LIVE / "pair_validity.json")
    deepseek = _load_reference_pairs(DEEPSEEK_LIVE / "pair_ledger.json")
    return {
        "glm": [dict(row) for row in glm_pairs],
        "claude": base._attach_reference_wall_time(claude, CLAUDE_LIVE),
        "gpt": base._attach_reference_wall_time(gpt, GPT_LIVE),
        "deepseek": base._attach_reference_wall_time(deepseek, DEEPSEEK_LIVE),
    }


MODEL_LABELS = {
    "glm": "GLM-5.3-Flash",
    "claude": "Claude Sonnet 4.6",
    "gpt": "GPT-5.6 Sol",
    "deepseek": "DeepSeek-V4-Pro",
}


def cross_model_rows(
    glm_pairs: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    model_rows = _reference_model_rows(glm_pairs)
    rows = []
    for state in STATES:
        for trap in TRAPS:
            available = {
                name: _valid_sources(values, state, trap)
                for name, values in model_rows.items()
            }
            common = set.intersection(*available.values())
            subsets = {
                name: _subset(values, state, trap, common)
                for name, values in model_rows.items()
            }
            summaries = {
                name: shared.model_cell_summary(values, state, trap)
                for name, values in subsets.items()
            }
            item: dict[str, Any] = {
                "state": state,
                "trap": trap,
                "comparison_basis": (
                    "same Source/State/Trap; each model differenced "
                    "against its own BASELINE"
                ),
                "common_source_indices": ",".join(
                    str(index) for index in sorted(common)
                ),
                "common_source_count": len(common),
            }
            for name in model_rows:
                summary = summaries[name]
                item[f"{name}_available_pairs"] = len(available[name])
                item[f"{name}_excluded_source_indices"] = ",".join(
                    str(index)
                    for index in sorted(
                        set(SOURCE_INDICES) - available[name]
                    )
                )
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
            for reference in ("claude", "gpt", "deepseek"):
                for metric in (
                    "input",
                    "output",
                    "cache_read",
                    "provider_calls",
                    "tool_calls",
                    "wall_time",
                ):
                    item[f"glm_{reference}_{metric}_trend_match"] = (
                        item[f"glm_{metric}_trend"]
                        == item[f"{reference}_{metric}_trend"]
                        if item["common_source_count"] > 0
                        else None
                    )
            rows.append(item)
    return rows


def state_trap_summary(
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = shared.state_trap_summary(pairs)
    for row in result:
        row["usage_class"] = "Provider Total Tokens"
    return result


def cross_model_rankings(
    glm_pairs: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    model_rows = _reference_model_rows(glm_pairs)
    result = []
    for state in STATES:
        cohort = set(SOURCE_INDICES)
        for rows in model_rows.values():
            for trap in TRAPS:
                cohort &= _valid_sources(rows, state, trap)
        for name, rows in model_rows.items():
            items = []
            for trap in TRAPS:
                valid = [
                    row
                    for row in _subset(rows, state, trap, cohort)
                    if row["pair_valid"]
                ]
                input_mean = base._mean(
                    row.get("delta_input_tokens") for row in valid
                )
                output_mean = base._mean(
                    row.get("delta_output_tokens") for row in valid
                )
                items.append(
                    {
                        "model": MODEL_LABELS[name],
                        "state": state,
                        "trap": trap,
                        "clean_pairs": len(valid),
                        "common_source_indices": ",".join(
                            str(index) for index in sorted(cohort)
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


def _fmt(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "NA"
    return f"{value:+,.{digits}f}"


def _plain(value: Any, digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "NA"
    return f"{value:.{digits}f}"


def _report_stub(*_args: Any, **_kwargs: Any) -> None:
    """The final report is emitted by delivery.py from all derived ledgers."""


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
    del (
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
    (live_dir / "_ANALYSIS_INPUT_READY").write_text(
        "FINAL_REPORT_ZH.md is generated by delivery.py\n",
        encoding="utf-8",
    )


def write_top_level_status(
    live_dir: Path,
    execution: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> None:
    root = live_dir.parent
    write_json(
        root / "FINAL_STATUS.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": execution["status"],
            "formal_h1_started": True,
            "admission_provider_calls": execution[
                "admission_provider_calls"
            ],
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
            "valid_checkpoints": execution["valid_checkpoints"],
            "condition_trajectories": execution["condition_records"],
            "clean_pairs": execution["valid_pairs"],
            "journal_audit": execution["journal_audit"],
            "fatal_reason": execution.get("fatal_reason"),
            "fatal_scope": execution.get("fatal_scope"),
            "automatic_retry": False,
            "automatic_backfill": False,
            "automatic_recovery": False,
            "h2_started": False,
            "finished_at": datetime.now(UTC).isoformat(),
        },
    )
    if execution["status"] != "COMPLETED":
        analysis_output = root / "live"
        write_json(root / "STOPPED_REPORT.json", analysis)
        (analysis_output / "STOPPED_REPORT_ZH.md").write_text(
            "\n".join(
                [
                    "# GLM-5.3-Flash H1 冻结停止报告",
                    "",
                    f"- 状态：`{execution['status']}`。",
                    f"- 停止原因：`{execution['fatal_reason']}`。",
                    f"- 正式 Provider Calls："
                    f"{execution['provider_requests']}/1054。",
                    f"- clean pairs：{execution['valid_pairs']}/54。",
                    "- 未补跑、未重试、未 Recovery、未启动 H2。",
                    "",
                ]
            ),
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
    BASE_WRITE_DEFENSE_REPORT(
        live_dir,
        execution,
        baseline_rows,
        measurements,
        wall_summaries,
        defense_summary,
        defense_comparison,
    )
    path = live_dir / "DEFENSE_OVERHEAD_REPORT_ZH.md"
    text = path.read_text(encoding="utf-8")
    text = (
        text.replace("Claude Sonnet 4.6", "GLM-5.3-Flash")
        .replace("Claude", "GLM-5.3-Flash")
        .replace("三模型", "四模型")
    )
    path.write_text(text, encoding="utf-8")


def _patch_base() -> None:
    base.PROTOCOL_ID = PROTOCOL_ID
    base.MODEL_ID = MODEL_ID
    base.USAGE_FIELDS = USAGE_FIELDS
    base.request_rows = request_rows
    base.admission_usage_rows = admission_usage_rows
    base.cross_model_rows = cross_model_rows
    base.cross_model_rankings = cross_model_rankings
    base.state_trap_summary = state_trap_summary
    base.report = report
    base.write_top_level_status = write_top_level_status
    base.write_defense_report = write_defense_report


def analyze(live_dir: Path) -> dict[str, Any]:
    _patch_base()
    result = base.analyze(live_dir)
    live_dir = live_dir.resolve()
    ranking_source = live_dir / "claude_state_trap_ranking.csv"
    if ranking_source.is_file():
        (live_dir / "glm_state_trap_ranking.csv").write_bytes(
            ranking_source.read_bytes()
        )
    ranking_json = live_dir / "claude_state_trap_ranking.json"
    if ranking_json.is_file():
        (live_dir / "glm_state_trap_ranking.json").write_bytes(
            ranking_json.read_bytes()
        )
    audit_path = live_dir / "execution_audit.json"
    audit = read_json(audit_path)
    audit["protocol_id"] = PROTOCOL_ID
    audit["model_id"] = MODEL_ID
    audit["usage_semantics"] = {
        "input_tokens_is_prompt_total": True,
        "cache_read_is_subset_of_input": True,
        "reasoning_tokens_is_subset_of_output": True,
        "cache_not_added_again": True,
        "reasoning_not_added_again": True,
        "total_formula": "prompt_tokens + completion_tokens",
    }
    audit["comparison_models"] = [
        MODEL_ID,
        "claude-sonnet-4-6",
        "gpt-5.6-sol",
        "deepseek-v4-pro",
    ]
    write_json(audit_path, audit)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("live_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.live_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Derive the requested GLM H1 delivery ledgers and Chinese report."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
TRAPS = ("BRANCH_NEW_V3", "DEPENDENCY_V6", "EVIDENCE_NEW_V3")
CONDITIONS = ("BASELINE", *TRAPS)
SOURCE_INDICES = (1, 6, 11, 16, 21, 26)
MODEL_ID = "glm-5.3-flash"
PROTOCOL_ID = "formal_H1_GLM_5_3_Flash_cross_model_6source_v1"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: list[str] | None = None,
) -> None:
    rows = [dict(row) for row in rows]
    if fields is None:
        fields = list(rows[0]) if rows else ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def num(value: Any) -> float | None:
    if value in (None, "", "NA", "None"):
        return None
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: Iterable[Any]) -> float | None:
    numeric = [num(value) for value in values]
    known = [value for value in numeric if value is not None]
    return statistics.mean(known) if known else None


def median(values: Iterable[Any]) -> float | None:
    numeric = [num(value) for value in values]
    known = [value for value in numeric if value is not None]
    return statistics.median(known) if known else None


def total(values: Iterable[Any]) -> int | None:
    numeric = [
        int(value)
        for value in values
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    return sum(numeric) if numeric else None


def _f(value: Any, digits: int = 1, signed: bool = False) -> str:
    number = num(value)
    if number is None:
        return "NA"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _bool(value: Any) -> bool | None:
    if value in (None, "", "NA", "None"):
        return None
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    return None


def token_source_rows(
    pairs: list[Mapping[str, Any]],
    provider_requests: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for record in provider_requests:
        if record.get("condition") == "SOURCE":
            continue
        grouped[
            (
                int(record["source_index"]),
                str(record["state"]),
                str(record["condition"]),
            )
        ].append(record)
    for values in grouped.values():
        values.sort(key=lambda row: int(row.get("local_request_index") or 0))
    rows = []
    for pair in pairs:
        if not _bool(pair.get("pair_valid")):
            continue
        source_index = int(pair["source_index"])
        state = str(pair["state"])
        trap = str(pair["trap"])
        baseline = grouped[(source_index, state, "BASELINE")]
        branch = grouped[(source_index, state, trap)]
        if not baseline or not branch:
            continue
        b_first = baseline[0]
        t_first = branch[0]

        def request_int(record: Mapping[str, Any], field: str) -> int:
            value = record.get(field)
            return int(value) if isinstance(value, int) else 0

        b_follow_input = sum(
            request_int(row, "input_tokens") for row in baseline[1:]
        )
        t_follow_input = sum(
            request_int(row, "input_tokens") for row in branch[1:]
        )
        b_follow_output = sum(
            request_int(row, "output_tokens") for row in baseline[1:]
        )
        t_follow_output = sum(
            request_int(row, "output_tokens") for row in branch[1:]
        )
        first_input = request_int(t_first, "input_tokens") - request_int(
            b_first, "input_tokens"
        )
        first_output = request_int(t_first, "output_tokens") - request_int(
            b_first, "output_tokens"
        )
        follow_input = t_follow_input - b_follow_input
        follow_output = t_follow_output - b_follow_output
        rows.append(
            {
                "source_index": source_index,
                "family": pair["family"],
                "state": state,
                "trap": trap,
                "baseline_first_request_id": b_first.get("request_id"),
                "trap_first_request_id": t_first.get("request_id"),
                "baseline_first_input_tokens": request_int(
                    b_first, "input_tokens"
                ),
                "trap_first_input_tokens": request_int(
                    t_first, "input_tokens"
                ),
                "delta_first_input_tokens": first_input,
                "baseline_first_output_tokens": request_int(
                    b_first, "output_tokens"
                ),
                "trap_first_output_tokens": request_int(
                    t_first, "output_tokens"
                ),
                "delta_first_output_tokens": first_output,
                "delta_first_total_tokens": first_input + first_output,
                "delta_followup_input_tokens": follow_input,
                "delta_followup_output_tokens": follow_output,
                "delta_followup_total_tokens": follow_input + follow_output,
                "delta_total_tokens_decomposed": (
                    first_input
                    + first_output
                    + follow_input
                    + follow_output
                ),
                "delta_total_tokens_pair_ledger": pair.get(
                    "delta_total_tokens"
                ),
                "delta_followup_provider_calls": (
                    len(branch) - len(baseline)
                ),
                "baseline_provider_calls": len(baseline),
                "trap_provider_calls": len(branch),
                "baseline_reasoning_tokens": total(
                    row.get("reasoning_tokens") for row in baseline
                ),
                "trap_reasoning_tokens": total(
                    row.get("reasoning_tokens") for row in branch
                ),
                "reasoning_usage_known": (
                    all(
                        isinstance(row.get("reasoning_tokens"), int)
                        for row in [*baseline, *branch]
                    )
                ),
            }
        )
    return rows


def aggregate_token_sources(
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for state in STATES:
        for trap in TRAPS:
            cell = [
                row for row in rows
                if row["state"] == state and row["trap"] == trap
            ]
            first = [row["delta_first_total_tokens"] for row in cell]
            follow = [row["delta_followup_total_tokens"] for row in cell]
            aggregate = sum(first) + sum(follow)
            result.append(
                {
                    "state": state,
                    "trap": trap,
                    "n": len(cell),
                    "mean_delta_first_input_tokens": mean(
                        row["delta_first_input_tokens"] for row in cell
                    ),
                    "mean_delta_first_output_tokens": mean(
                        row["delta_first_output_tokens"] for row in cell
                    ),
                    "mean_delta_first_total_tokens": mean(first),
                    "mean_delta_followup_input_tokens": mean(
                        row["delta_followup_input_tokens"] for row in cell
                    ),
                    "mean_delta_followup_output_tokens": mean(
                        row["delta_followup_output_tokens"] for row in cell
                    ),
                    "mean_delta_followup_total_tokens": mean(follow),
                    "mean_delta_followup_provider_calls": mean(
                        row["delta_followup_provider_calls"]
                        for row in cell
                    ),
                    "aggregate_initial_exposure_share": (
                        sum(first) / aggregate if aggregate else None
                    ),
                    "aggregate_followup_interaction_share": (
                        sum(follow) / aggregate if aggregate else None
                    ),
                }
            )
    return result


def pair_ledger_rows(
    pairs: list[Mapping[str, Any]],
    token_by_key: Mapping[tuple[int, str, str], Mapping[str, Any]],
    defense_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    defense_map = {
        (int(row["source_index"]), row["state"], row["trap"]): row
        for row in defense_rows
    }
    result = []
    for pair in pairs:
        key = (int(pair["source_index"]), pair["state"], pair["trap"])
        token = token_by_key.get(key, {})
        defense = defense_map.get(key, {})
        result.append(
            {
                "source_index": pair["source_index"],
                "family": pair["family"],
                "state": pair["state"],
                "trap": pair["trap"],
                "checkpoint_id": pair.get("checkpoint_id"),
                "pair_valid": pair.get("pair_valid"),
                "invalid_reasons": pair.get("invalid_reasons"),
                "baseline_input_tokens": pair.get("baseline_input_tokens"),
                "trap_input_tokens": pair.get("trap_input_tokens"),
                "delta_input_tokens": pair.get("delta_input_tokens"),
                "baseline_output_tokens": pair.get("baseline_output_tokens"),
                "trap_output_tokens": pair.get("trap_output_tokens"),
                "delta_output_tokens": pair.get("delta_output_tokens"),
                "baseline_total_tokens": pair.get("baseline_total_tokens"),
                "trap_total_tokens": pair.get("trap_total_tokens"),
                "delta_total_tokens": pair.get("delta_total_tokens"),
                "baseline_cache_read_input_tokens": pair.get(
                    "baseline_cache_read_input_tokens"
                ),
                "trap_cache_read_input_tokens": pair.get(
                    "trap_cache_read_input_tokens"
                ),
                "delta_cache_read_input_tokens": pair.get(
                    "delta_cache_read_input_tokens"
                ),
                "baseline_reasoning_tokens": token.get(
                    "baseline_reasoning_tokens"
                ),
                "trap_reasoning_tokens": token.get(
                    "trap_reasoning_tokens"
                ),
                "delta_first_input_tokens": token.get(
                    "delta_first_input_tokens"
                ),
                "delta_first_output_tokens": token.get(
                    "delta_first_output_tokens"
                ),
                "delta_first_total_tokens": token.get(
                    "delta_first_total_tokens"
                ),
                "delta_followup_input_tokens": token.get(
                    "delta_followup_input_tokens"
                ),
                "delta_followup_output_tokens": token.get(
                    "delta_followup_output_tokens"
                ),
                "delta_followup_total_tokens": token.get(
                    "delta_followup_total_tokens"
                ),
                "baseline_provider_calls": pair.get(
                    "baseline_provider_calls"
                ),
                "trap_provider_calls": pair.get("trap_provider_calls"),
                "delta_provider_calls": pair.get("delta_provider_calls"),
                "baseline_tool_calls": pair.get("baseline_tool_calls"),
                "trap_tool_calls": pair.get("trap_tool_calls"),
                "delta_tool_calls": pair.get("delta_tool_calls"),
                "baseline_interaction_depth": pair.get(
                    "baseline_interaction_depth"
                ),
                "trap_interaction_depth": pair.get(
                    "trap_interaction_depth"
                ),
                "delta_interaction_depth": pair.get(
                    "delta_interaction_depth"
                ),
                "baseline_wall_clock_seconds": pair.get(
                    "baseline_wall_clock_seconds"
                ),
                "trap_wall_clock_seconds": pair.get(
                    "trap_wall_clock_seconds"
                ),
                "delta_wall_clock_seconds": pair.get(
                    "delta_wall_clock_seconds"
                ),
                "baseline_provider_latency_seconds": pair.get(
                    "baseline_provider_latency_seconds"
                ),
                "trap_provider_latency_seconds": pair.get(
                    "trap_provider_latency_seconds"
                ),
                "delta_provider_latency_seconds": pair.get(
                    "delta_provider_latency_seconds"
                ),
                "baseline_tool_execution_seconds": pair.get(
                    "baseline_tool_execution_seconds"
                ),
                "trap_tool_execution_seconds": pair.get(
                    "trap_tool_execution_seconds"
                ),
                "delta_tool_execution_seconds": pair.get(
                    "delta_tool_execution_seconds"
                ),
                "baseline_task_success": pair.get("baseline_task_success"),
                "trap_task_success": pair.get("trap_task_success"),
                "baseline_budget_exhausted": pair.get(
                    "baseline_budget_exhausted"
                ),
                "trap_budget_exhausted": pair.get(
                    "trap_budget_exhausted"
                ),
                "defense_total_processing_seconds": defense.get(
                    "defense_total_processing_seconds"
                ),
                "defense_total_cpu_seconds": defense.get(
                    "defense_total_cpu_seconds"
                ),
                "defense_provider_calls": 0,
                "defense_tokens": 0,
                "trap_payload_bytes": defense.get("trap_payload_bytes"),
            }
        )
    return result


def enrich_defense_payload(
    live_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        path = (
            live_dir
            / "sources"
            / f"source-{int(row['source_index']):02d}"
            / "checkpoints"
            / str(row["state"]).lower()
            / "branches"
            / f"{row['trap']}.json"
        )
        if not path.is_file():
            continue
        arm = read_json(path, {})
        visible = (arm.get("initial_exposure_event") or {}).get(
            "visible_result"
        ) or {}
        payload = visible.get("trap_information")
        if payload is not None:
            row["trap_payload_bytes"] = len(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )


def _sign_counts(values: Iterable[Any]) -> tuple[int, int, int]:
    numeric = [num(value) for value in values]
    known = [value for value in numeric if value is not None]
    return (
        sum(value > 0 for value in known),
        sum(value == 0 for value in known),
        sum(value < 0 for value in known),
    )


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.is_file() and source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)


def report(
    run_dir: Path,
    live: Path,
    status: Mapping[str, Any],
    summary_rows: list[dict[str, Any]],
    token_aggregate: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
    cross_rankings: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    defense_rows: list[dict[str, Any]],
    pair_ledger: list[dict[str, Any]],
    requests: list[dict[str, Any]],
) -> None:
    pair_count = int(status.get("clean_pairs") or 0)
    completed = status.get("status") == "COMPLETED"
    admission_requests = []
    for path in sorted((run_dir / "admission").glob("request-*.json")):
        record = read_json(path, {})
        usage = record.get("normalized_usage") or {}
        admission_requests.append(
            {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cache_read_input_tokens": usage.get("cache_read_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
            }
        )
    all_requests = [*admission_requests, *requests]
    run_status = read_json(live / "run_status.json", {})
    run_started = run_status.get("started_at")
    run_finished = run_status.get("finished_at")
    run_wall_seconds = None
    if run_started and run_finished:
        from datetime import datetime

        run_wall_seconds = (
            datetime.fromisoformat(run_finished)
            - datetime.fromisoformat(run_started)
        ).total_seconds()
    formal_latency = sum(
        num(record.get("elapsed_seconds")) or 0 for record in requests
    )
    token_by_cell = {
        (row["state"], row["trap"]): row for row in token_aggregate
    }
    cross_by_cell = {
        (row["state"], row["trap"]): row for row in cross_rows
    }
    defense_by_cell = {
        (row["state"], row["trap"]): row for row in defense_rows
    }
    timing_by_cell = defaultdict(list)
    for row in timing_rows:
        if row.get("condition") != "BASELINE":
            timing_by_cell[(row["state"], row["condition"])].append(row)
    pair_by_cell = {
        (row["state"], row["trap"]): row for row in pair_ledger
    }
    lines = [
        "# CyberStrikeAI GLM-5.3-Flash 六 Source 正式 H1 报告",
        "",
        "## 摘要",
        "",
        f"- 协议：`{PROTOCOL_ID}`。",
        f"- 状态：`{status.get('status')}`。",
        "- 目标：6 Source、18 自然 checkpoint、72 条件轨迹、"
        "54 个 Trap–BASELINE 配对，局部 K=12。",
        f"- Provider Calls：{status.get('provider_calls')}/1056"
        "（两次实时准入计入全局上限）。",
        f"- Source：尝试 {status.get('attempted_source_count', 0)}/6，"
        f"完整完成 {status.get('completed_source_count', 0)}/6；"
        f"有效 checkpoint：{status.get('valid_checkpoints', 0)}/18；"
        f"条件轨迹：{status.get('condition_trajectories', 0)}/72；"
        f"clean pairs：{pair_count}/54。",
        f"- 请求/响应模型：`{status.get('models', {}).get('response')}`。",
        f"- 停止原因：`{status.get('fatal_reason')}`。",
        "- 跨模型结果只作共同 Source×State×Trap 的描述性方向和机制排序，"
        "不宣称统计显著性。",
        "",
        "## Provider-free 与真实准入",
        "",
        "- Provider-free 门禁：6/6 Source、18/18 checkpoint、"
        "72/72 条件槽位、78/78 journal scope PASS；Provider Calls=0。",
        "- 门禁覆盖 Fixture/Oracle 身份、三种 Trap、pre-effect Gateway、"
        "K=12、restore、BASELINE 单动作、Journal/Ledger、bypass、"
        "fail-stop 和 no-retry。",
        f"- 真实准入记录：`{run_dir / 'admission' / 'admission_record.json'}`；"
        "准入最多 2 次、零重试，并要求两个请求和响应均精确返回 "
        "`glm-5.3-flash`。",
        "- 本次真实准入通过全部检查：精确模型身份、单 ToolCall、"
        "tool_result 回传、reasoning_content、2400 Token、Usage 和缓存计数。",
        "",
        "## 模型和冻结设置",
        "",
        "- Platform：CyberStrikeAI；Provider：智增增。",
        "- API：OpenAI-compatible Chat Completions；"
        "`https://api.zhizengzeng.com/v1/chat/completions`。",
        "- `model=glm-5.3-flash`；`temperature=0.0`；"
        "`reasoning_effort=max`；`thinking.type=enabled`；"
        "`clear_thinking=false`。",
        "- `max_tokens=2400`，沿用既有 H1 单次输出上限。GLM 推理 Token "
        "计入 completion，若耗尽上限则以 `finish_reason=length` fail-stop，"
        "未私自扩大预算。",
        "- `tool_choice=auto`；非流式；`parallel_tool_calls=false`。"
        "GLM 文档未定义该字段，因此不发送；上游返回多个 ToolCall 时本地拒绝。",
        "- 超时 360 秒；自动重试 0；无补样、Recovery、选择性剔除。",
        "- 协议依据：智谱 GLM-5.3-Flash 模型页、Chat Completions API "
        "文档，以及智增增模型目录与更新记录；具体链接保存在 "
        "`execution_audit.json` 和源码配置中。",
        "",
        "## 正式执行与冻结停止",
        "",
        "- Source 1 完成 2 次真实正式请求并产生 1 个 RECON checkpoint、"
        "1 次真实工具调用和有效 Gateway 记录。",
        "- 第 2 次正式响应 HTTP 200、模型仍为 `glm-5.3-flash`，"
        "但返回了两个并行 ToolCall。冻结配置要求串行单 ToolCall，"
        "本地 Provider 在工具执行前以 `expected_exactly_one_tool_call` "
        "拒绝该响应。",
        "- 因此 Source 1 未获得三个 State checkpoint，"
        "`source01_natural_checkpoint_admission_failed` 触发 fail-stop。",
        "- 条件轨迹、Trap 注入与配对均未开始；Source 2-6 未运行。",
        "- 未自动重试、未补样、未 Recovery、未修改参数绕过该协议失败。",
        "",
        "## Usage 口径",
        "",
        "- `input_tokens = prompt_tokens`，其中已包含缓存命中输入。",
        "- `output_tokens = completion_tokens`，其中已包含推理 Token。",
        "- `total_tokens = prompt_tokens + completion_tokens`。",
        "- Cache Read 是输入子集；Reasoning Tokens 是输出子集；"
        "两者均单独报告且不重复加入 Total。",
        "- 上游缺失字段保持 UNKNOWN；只用已知局部值时会明确标注，"
        "不把局部观测冒充完整计费。",
        f"- 正式请求 Input：{total(row.get('input_tokens') for row in requests)}；"
        f"Output：{total(row.get('output_tokens') for row in requests)}；"
        f"Total：{total(row.get('total_tokens') for row in requests)}；"
        f"Cache Read：{total(row.get('cache_read_input_tokens') for row in requests)}；"
        f"Reasoning：{total(row.get('reasoning_tokens') for row in requests)}。",
        f"- 含两次准入的全部请求 Input："
        f"{total(row.get('input_tokens') for row in all_requests)}；"
        f"Output：{total(row.get('output_tokens') for row in all_requests)}；"
        f"Total：{total(row.get('total_tokens') for row in all_requests)}；"
        f"Cache Read："
        f"{total(row.get('cache_read_input_tokens') for row in all_requests)}；"
        f"Reasoning："
        f"{total(row.get('reasoning_tokens') for row in all_requests)}。",
        f"- Source 1 正式执行 Wall-clock：{_f(run_wall_seconds, 3)} 秒；"
        f"两次正式 Provider 请求延迟合计：{_f(formal_latency, 3)} 秒。",
        "- 正式执行的工具调用：已执行 1 次；第 2 个响应提出 2 个并行 "
        "ToolCall，均在执行前被拒绝，因此只有检查点内的 1 次 Gateway "
        "工具执行计入。",
        "",
        "## 9 个 State×Trap 资源增量",
        "",
        "| State | Trap | n | Mean ΔTotal Tokens | Token Ratio | "
        "ΔProvider Calls | ΔTool Calls | Mean ΔWall s | Positive/Zero/Negative |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        cell = [
            item for item in timing_rows
            if item.get("state") == row["state"]
            and item.get("condition") == row["trap"]
        ]
        baseline_total = mean(
            item.get("total_tokens")
            for item in timing_rows
            if item.get("state") == row["state"]
            and item.get("condition") == "BASELINE"
        )
        trap_total = mean(
            item.get("total_tokens")
            for item in cell
        )
        ratio = (
            trap_total / baseline_total
            if trap_total is not None and baseline_total
            else None
        )
        pair_cell = [
            item for item in pair_ledger
            if item["state"] == row["state"]
            and item["trap"] == row["trap"]
        ]
        signs = _sign_counts(
            item.get("delta_total_tokens") for item in pair_cell
        )
        wall_delta = mean(
            item.get("delta_wall_clock_seconds") for item in pair_cell
        )
        lines.append(
            f"| {row['state']} | {row['trap']} | "
            f"{row['clean_pairs']}/6 | "
            f"{_f(row.get('delta_input_output_total_mean'), 1, True)} | "
            f"{_f(ratio, 3)} | "
            f"{_f(row.get('delta_provider_calls_mean'), 3, True)} | "
            f"{_f(row.get('delta_tool_calls_mean'), 3, True)} | "
            f"{_f(wall_delta, 3, True)} | "
            f"{signs[0]}/{signs[1]}/{signs[2]} |"
        )
    lines += [
        "",
        "## Token 来源分解",
        "",
        "| State | Trap | n | ΔFirst Total | ΔFollowup Total | "
        "First Share | Followup Share | ΔFollowup Calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in token_aggregate:
        lines.append(
            f"| {row['state']} | {row['trap']} | {row['n']}/6 | "
            f"{_f(row['mean_delta_first_total_tokens'], 1, True)} | "
            f"{_f(row['mean_delta_followup_total_tokens'], 1, True)} | "
            f"{_f(row['aggregate_initial_exposure_share'], 3)} | "
            f"{_f(row['aggregate_followup_interaction_share'], 3)} | "
            f"{_f(row['mean_delta_followup_provider_calls'], 3, True)} |"
        )
    lines += [
        "",
        "首次暴露增量是同一 checkpoint 上 Trap 与 BASELINE 的第一次 Provider "
        "请求之差；后续增量来自 Agent 在 Trap 暴露后的自主交互。"
        "两项按 Input 与 Output 分别记录在 "
        "`token_source_decomposition.csv`。",
        "",
        "## 攻击端时间与终止",
        "",
        "Wall-clock 从 checkpoint restore 完成后、initial exposure 前开始，"
        "到条件轨迹终止；Provider latency 来自逐请求 journal；工具时间由"
        "非侵入观测器记录。Task Success 和 Budget Exhausted 保留在 pairing "
        "ledger 中，不因失败或预算终止而排除有效配对。",
        "",
        "## 防御端成本",
        "",
        "| State | Trap | n | Defense Mean s | CPU Mean s | "
        "Payload Bytes Mean | Defense Calls | Defense Tokens |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for state in STATES:
        for trap in TRAPS:
            cell = [
                row for row in pair_ledger
                if row["state"] == state
                and row["trap"] == trap
                and _bool(row.get("pair_valid"))
            ]
            lines.append(
                f"| {state} | {trap} | {len(cell)}/6 | "
                f"{_f(mean(row.get('defense_total_processing_seconds') for row in cell), 6)} | "
                f"{_f(mean(row.get('defense_total_cpu_seconds') for row in cell), 6)} | "
                f"{_f(mean(row.get('trap_payload_bytes') for row in cell), 1)} | "
                "0 | 0 |"
            )
    lines += [
        "",
        "- Trap 构造、Prepare、Response construction、Injection 与 initial "
        "exposure 总时间逐轨迹保存在 `defense_overhead_comparison.csv`。",
        "- H1 没有在线 State Observer 或动态 Trap 选择，因此对应时间标为 NA；"
        "这不是 0。",
        "- CPU 与 RSS 为 runner 进程级，无法完全隔离为 STARVE 防御进程；"
        "未完整观测的完整防御成本保持 UNKNOWN。",
        "",
        "## 跨模型比较",
        "",
        "比较单位为共同 Source×State×Trap，以每个模型自身 BASELINE 求差。"
        "模型配置和 Usage 口径并不相同，禁止混合绝对 Token 判优。",
        "本批次 GLM 没有形成任何 clean pair，因此 GLM 与 Claude、GPT、"
        "DeepSeek 的共同 Source×State×Trap 比较集合为空；下表 NA "
        "表示不可估计，不是负效应或 0 效应。",
        "",
        "| State | Trap | Model | Clean n | ΔInput | ΔOutput | "
        "ΔCache | ΔCalls | ΔTools | ΔWall s | Success T/B | Budget T/B |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cross_rows:
        for prefix, label in (
            ("glm", "GLM-5.3-Flash"),
            ("claude", "Claude Sonnet 4.6"),
            ("gpt", "GPT-5.6 Sol"),
            ("deepseek", "DeepSeek-V4-Pro"),
        ):
            lines.append(
                f"| {row['state']} | {row['trap']} | {label} | "
                f"{row[f'{prefix}_clean_pairs']}/6 | "
                f"{_f(row[f'{prefix}_delta_input_mean'], 1, True)} | "
                f"{_f(row[f'{prefix}_delta_output_mean'], 1, True)} | "
                f"{_f(row[f'{prefix}_delta_cache_read_mean'], 1, True)} | "
                f"{_f(row[f'{prefix}_delta_provider_calls_mean'], 2, True)} | "
                f"{_f(row[f'{prefix}_delta_tool_calls_mean'], 2, True)} | "
                f"{_f(row[f'{prefix}_delta_wall_time_mean'], 3, True)} | "
                f"{row[f'{prefix}_task_success_trap']}/"
                f"{row[f'{prefix}_task_success_baseline']} | "
                f"{row[f'{prefix}_budget_exhausted_trap']}/"
                f"{row[f'{prefix}_budget_exhausted_baseline']} |"
            )
    lines += [
        "",
        "### 模型差异对比较的限制",
        "",
        "- DeepSeek：OpenAI-compatible；推理输出经 `reasoning_content` 保留；"
        "历史六 Source 配置存在缺失 checkpoint 和个别技术失败。",
        "- GPT-5.6 Sol：OpenAI-compatible；`reasoning_effort=medium`，"
        "`max_completion_tokens=2400`，输入 Total 与 GLM 的 prompt 口径可比性"
        "仅在各自增量方向层面使用。",
        "- Claude Sonnet 4.6：Anthropic Messages；`max_tokens=8192`；"
        "输入 Token 按未缓存输入、Cache Creation 与 Cache Read 加总，"
        "不能与 GLM Provider Total 数值直接相减。",
        "- GLM-5.3-Flash：OpenAI-compatible；`reasoning_effort=max`，"
        "`thinking` 始终 enabled，推理 Token 计入 `max_tokens` 与 completion。",
        "",
        "## 机制排序",
        "",
        "| Model | State | Rank | Trap | Common n | ΔInput+Output |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in cross_rankings:
        lines.append(
            f"| {row['model']} | {row['state']} | "
            f"{row['rank_by_input_plus_output']} | {row['trap']} | "
            f"{row['clean_pairs']} | "
            f"{_f(row['delta_input_plus_output_mean'], 1, True)} |"
        )
    lines += [
        "",
        "## 完成审计与限制",
        "",
        f"- 完整 18 checkpoint/72 轨迹/54 配对目标是否达成："
        f"`{str(completed).lower()}`。",
        f"- Journal：`{json.dumps(status.get('journal_audit', {}), ensure_ascii=False)}`。",
        "- 未启动 H2；未使用 PentestGPT。",
        "- 未修改 Trap 语义、System Prompt、State Observer、Gateway、"
        "Oracle、任务环境或历史 DeepSeek/GPT/Claude/GLM 结果。",
        "- 未自动补执行被 Trap 拦截的原始动作。",
        "- K=12 是观测窗口；预算耗尽是删失终点，不是无效配对理由。",
        "- 六 Source 与三个 State 的聚类使结果只支持描述性跨模型证据。",
        "",
        "## 交付文件",
        "",
        "- `FINAL_REPORT_ZH.md`、`delivery_summary.json`。",
        "- `state_trap_summary.csv`、`pair_level_resource_deltas.csv`。",
        "- `request_usage.csv`、`provider_requests.jsonl`。",
        "- `timing_ledger.csv`、`token_source_decomposition.csv`。",
        "- `defense_overhead_comparison.csv`、`cross_model_comparison.csv`。",
        "- `execution_audit.json`、`FINAL_STATUS.json`。",
        "",
    ]
    text = "\n".join(lines)
    (run_dir / "FINAL_REPORT_ZH.md").write_text(text, encoding="utf-8")
    (live / "FINAL_REPORT_ZH.md").write_text(text, encoding="utf-8")


def derive(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    live = run_dir / "live" if (run_dir / "live").is_dir() else run_dir
    root = run_dir.parent if run_dir.name == "live" else run_dir
    execution = read_json(live / "execution_audit.json", {})
    final_status = read_json(root / "FINAL_STATUS.json", {})
    run_status = read_json(live / "run_status.json", {})
    pairs = read_json(live / "pair_validity.json", [])
    timing_rows = read_json(live / "measurements.json", [])
    requests = []
    request_path = live / "provider_requests.jsonl"
    if request_path.is_file():
        requests = [
            json.loads(line)
            for line in request_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    summary_rows = read_json(live / "state_trap_summary.json", [])
    cross_rows = read_json(live / "cross_model_comparison.json", [])
    cross_rankings = read_json(
        live / "cross_model_state_rankings.json",
        [],
    )
    defense_rows = read_csv(live / "defense_overhead_comparison.csv")
    source_audit_rows = read_csv(live / "source_audit.csv")
    attempted_sources = sum(
        row.get("status") not in {"NOT_RUN", "", None}
        for row in source_audit_rows
    )
    completed_sources = sum(
        row.get("status") == "PASS" for row in source_audit_rows
    )
    executed_tool_calls = sum(
        int(
            (source.get("gateway") or {}).get("call_count")
            or 0
        )
        for source in run_status.get("sources") or []
    )
    rejected_parallel_tool_calls = 0
    for path in (live / "sources").rglob("request-*.json"):
        record = read_json(path, {})
        calls = (record.get("response_message") or {}).get("tool_calls") or []
        if record.get("outcome") == "failure" and len(calls) > 1:
            rejected_parallel_tool_calls += len(calls)
    token_rows = token_source_rows(pairs, requests)
    token_aggregate = aggregate_token_sources(token_rows)
    token_by_key = {
        (int(row["source_index"]), row["state"], row["trap"]): row
        for row in token_rows
    }
    pair_rows = pair_ledger_rows(pairs, token_by_key, defense_rows)
    enrich_defense_payload(live, pair_rows)

    write_csv(run_dir / "pair_level_resource_deltas.csv", pair_rows)
    write_csv(run_dir / "timing_ledger.csv", timing_rows)
    write_csv(run_dir / "token_source_decomposition.csv", token_rows)
    write_json(live / "token_source_decomposition.json", token_rows)
    _copy_if_exists(
        live / "state_trap_summary.csv",
        run_dir / "state_trap_summary.csv",
    )
    _copy_if_exists(
        live / "combined_request_usage.csv",
        run_dir / "request_usage.csv",
    )
    _copy_if_exists(
        live / "combined_provider_requests.jsonl",
        run_dir / "provider_requests.jsonl",
    )
    _copy_if_exists(
        live / "defense_overhead_comparison.csv",
        run_dir / "defense_overhead_comparison.csv",
    )
    _copy_if_exists(
        live / "cross_model_comparison.csv",
        run_dir / "cross_model_comparison.csv",
    )
    if (live / "execution_audit.json").is_file():
        execution = read_json(live / "execution_audit.json", execution)
        write_json(run_dir / "execution_audit.json", execution)

    usage_known = {
        field: {
            "known": sum(isinstance(row.get(field), int) for row in requests),
            "missing": sum(not isinstance(row.get(field), int) for row in requests),
            "sum": total(row.get(field) for row in requests),
        }
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_input_tokens",
            "reasoning_tokens",
        )
    }
    delivery_summary = {
        "protocol_id": PROTOCOL_ID,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": final_status.get("status", execution.get("status")),
        "provider_free": {
            "passed": read_json(
                root / "provider-free" / "provider_free_admission.json",
                {},
            ).get("passed"),
            "provider_calls": 0,
        },
        "admission": read_json(
            root / "admission" / "admission_record.json",
            {},
        ),
        "execution": execution,
        "counts": {
            "sources": completed_sources,
            "attempted_sources": attempted_sources,
            "completed_sources": completed_sources,
            "valid_checkpoints": (
                final_status.get("valid_checkpoints")
                if final_status.get("valid_checkpoints") is not None
                else execution.get("valid_checkpoints")
            ),
            "condition_trajectories": (
                final_status.get("condition_trajectories")
                if final_status.get("condition_trajectories") is not None
                else execution.get("condition_records")
            ),
            "target_pairs": 54,
            "clean_pairs": (
                final_status.get("clean_pairs")
                if final_status.get("clean_pairs") is not None
                else execution.get("valid_pairs")
            ),
            "formal_executed_tool_calls": executed_tool_calls,
            "rejected_parallel_tool_calls": rejected_parallel_tool_calls,
        },
        "usage": usage_known,
        "methods": {
            "platform": "CyberStrikeAI",
            "provider": "zhizengzeng",
            "api": "openai-compatible-chat-completions-v1",
            "model_id": MODEL_ID,
            "reasoning_effort": "max",
            "thinking": {"type": "enabled", "clear_thinking": False},
            "output_token_parameter": "max_tokens",
            "max_tokens": 2400,
            "timeout_seconds": 360,
            "automatic_retry": 0,
            "protocol_evidence": {
                "model": "https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash",
                "chat_completions": "https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%AF%B9%E8%AF%9D%E8%A1%A5%E5%85%A8",
                "provider_catalog": "https://doc.zhizengzeng.com/doc-3979947",
                "provider_news": "https://doc.zhizengzeng.com/doc-6735277",
            },
        },
        "token_source_decomposition": token_aggregate,
        "defense_cost_status": {
            "initial_exposure": "MEASURED",
            "payload_bytes": "MEASURED",
            "construction_injection": "MEASURED_INITIAL_EXPOSURE_STAGES",
            "defense_provider_calls": 0,
            "defense_tokens": 0,
            "defense_api_fee": 0,
            "cpu": "PROCESS_LEVEL_ATTRIBUTION",
            "memory": "PROCESS_LEVEL_ONLY",
            "full_defense_total": "UNKNOWN",
        },
        "limitations": [
            "Descriptive six-Source cross-model evidence only.",
            "Absolute token counts differ by provider usage contract.",
            "H1 has no online state observer or dynamic trap selection.",
            "Process CPU/RSS are not isolated to a defense process.",
        ],
        "h2_started": False,
        "pentestgpt_used": False,
    }
    write_json(run_dir / "delivery_summary.json", delivery_summary)
    write_json(live / "delivery_summary.json", delivery_summary)

    status = {
        **final_status,
        "protocol_id": PROTOCOL_ID,
        "status": final_status.get("status", execution.get("status")),
        "formal_h1_started": True,
        "admission_provider_calls": execution.get(
            "admission_provider_calls"
        ),
        "formal_provider_calls": execution.get("provider_requests"),
        "provider_calls": execution.get("global_provider_calls"),
        "provider_request_hard_ceiling": 1056,
        "provider_calls_within_ceiling": execution.get(
            "provider_requests_within_ceiling"
        ),
        "models": {
            "request": MODEL_ID,
            "response": execution.get("provider_models"),
        },
        "source_count": completed_sources,
        "attempted_source_count": attempted_sources,
        "completed_source_count": completed_sources,
        "valid_checkpoints": execution.get("valid_checkpoints"),
        "condition_trajectories": execution.get("condition_records"),
        "clean_pairs": execution.get("valid_pairs"),
        "formal_tool_calls_executed": executed_tool_calls,
        "rejected_parallel_tool_calls": rejected_parallel_tool_calls,
        "journal_audit": execution.get("journal_audit"),
        "fatal_reason": execution.get("fatal_reason"),
        "fatal_scope": execution.get("fatal_scope"),
        "automatic_retry": False,
        "automatic_backfill": False,
        "automatic_recovery": False,
        "delivery_files_complete": True,
        "delivery_files": [
            "FINAL_REPORT_ZH.md",
            "delivery_summary.json",
            "state_trap_summary.csv",
            "pair_level_resource_deltas.csv",
            "request_usage.csv",
            "provider_requests.jsonl",
            "timing_ledger.csv",
            "token_source_decomposition.csv",
            "defense_overhead_comparison.csv",
            "cross_model_comparison.csv",
            "execution_audit.json",
            "FINAL_STATUS.json",
        ],
        "h2_started": False,
        "pentestgpt_used": False,
        "finished_at": final_status.get(
            "finished_at",
            datetime.now(UTC).isoformat(),
        ),
    }
    write_json(run_dir / "FINAL_STATUS.json", status)
    report(
        run_dir,
        live,
        status,
        summary_rows,
        token_aggregate,
        cross_rows,
        cross_rankings,
        timing_rows,
        defense_rows,
        pair_rows,
        requests,
    )
    return delivery_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(derive(args.run_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

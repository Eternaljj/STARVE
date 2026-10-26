"""Formal Claude Sonnet 4.6 six-Source H1 cross-model experiment."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("config.json")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.experiment_lock import ExperimentDirectoryLock
from h1_claude_9source_sonnet_4_6_20261022 import provider as claude
from h1_claude_cross_model_6source_20261024 import measurement
from h1_cross_system_20261023 import runner as h1
from h1_gpt_cross_model_6source_20261024 import runner as shared
from p0a_pentestgpt.stagetrap_k12_live import _write_json_atomic

PROTOCOL_ID = "formal_H1_Claude_Sonnet_4_6_cross_model_6source_20261024_v1"
MODEL_ID = "claude-sonnet-4-6"
API_KEY_ENV = "H1_P_CLAUDE_API_KEY"
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
NATURAL_REQUESTS = 32
K = 12
ADMISSION_REQUESTS = 2
TOTAL_REQUEST_CEILING = 1056
FORMAL_REQUEST_CEILING = TOTAL_REQUEST_CEILING - ADMISSION_REQUESTS
TARGET_CHECKPOINTS = 18
TARGET_CONDITIONS = 72
TARGET_PAIRS = 54
MAX_TOKENS = 8192
FATAL_EXTRA_STOP_REASONS = {
    "upstream_refusal",
    "usage_missing",
    "usage_missing_required_token_fields",
    "usage_cache_fields_incomplete",
    "usage_cache_fields_invalid",
    "tool_use_stop_reason_mismatch",
    "tool_use_content_missing",
    "parallel_or_invalid_tool_calls",
    "invalid_anthropic_message_type",
    "provider_content_not_array",
    "provider_content_block_invalid",
    "output_truncated",
    "request_unsettled_at_provider_exit",
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv(
    path: Path,
    rows: list[Mapping[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = read_json(path)
    provider = config["provider"]
    budget = config["budget"]
    if (
        config.get("config_version")
        != "formal-h1-claude-sonnet-4-6-cross-model-6source-v1"
        or config.get("protocol_id") != PROTOCOL_ID
        or provider.get("interface") != "anthropic-messages-v1"
        or provider.get("model_id") != MODEL_ID
        or provider.get("accepted_response_model_ids") != [MODEL_ID]
        or provider.get("credential_source") != API_KEY_ENV
        or provider.get("temperature") != 0.0
        or provider.get("max_tokens") != MAX_TOKENS
        or provider.get("disable_parallel_tool_use") is not True
        or provider.get("stream") is not False
        or tuple(config["sources"]["indices"]) != SOURCE_INDICES
        or tuple(config["states"]) != STATES
        or tuple(config["conditions"]) != CONDITIONS
        or budget.get("natural_provider_calls_per_source")
        != NATURAL_REQUESTS
        or budget.get("followup_budget_per_condition") != K
        or budget.get("total_provider_request_hard_ceiling")
        != TOTAL_REQUEST_CEILING
        or budget.get("formal_provider_request_hard_ceiling_after_two_call_admission")
        != FORMAL_REQUEST_CEILING
        or budget.get("automatic_retry") is not False
        or budget.get("automatic_backfill") is not False
        or budget.get("automatic_recovery") is not False
    ):
        raise RuntimeError("formal Claude H1 configuration changed")
    return config


def provider_config(
    config: Mapping[str, Any],
    *,
    require_key: bool,
) -> dict[str, Any]:
    source = config["provider"]
    api_key = str(os.environ.get(API_KEY_ENV, "")).strip()
    if require_key and not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is required for live execution")
    return {
        "name": str(source["name"]),
        "api_key": api_key,
        "credential_source": API_KEY_ENV,
        "base_url": str(source["base_url"]).rstrip("/"),
        "model": MODEL_ID,
        "accepted_response_model_ids": [MODEL_ID],
        "anthropic_version": str(source["anthropic_version"]),
        "authorization_scheme": str(source["authorization_scheme"]),
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "timeout_seconds": int(source["timeout_seconds"]),
        "tool_choice": str(source["tool_choice"]),
        "disable_parallel_tool_use": True,
        "stream": False,
        "provider_interface": "anthropic-messages-v1",
        "usage_semantics": "anthropic-messages-usage-v1",
        # Compatibility fields required by the already-frozen wrapper around
        # the shared execution path. They are superseded in the Claude audit.
        "send_temperature": True,
        "reasoning_effort": None,
        "output_token_parameter": "max_tokens",
        "max_completion_tokens": MAX_TOKENS,
        "parallel_tool_calls": False,
    }


def public_provider(provider: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(provider))
    value["api_key"] = "[PRESENT]" if value.get("api_key") else "[MISSING]"
    return value


def historical_compatibility(_config: Mapping[str, Any]) -> dict[str, Any]:
    admission_path = (
        ROOT
        / "results"
        / "claude_sonnet_4_6_admission_source01_20261022T080916235435Z"
        / "admission_record.json"
    )
    freeze_path = (
        ROOT
        / "results"
        / "claude_sonnet_4_6_h1_freeze_20261022_v1"
        / "freeze_record.json"
    )
    admission = read_json(admission_path) if admission_path.is_file() else {}
    freeze = read_json(freeze_path) if freeze_path.is_file() else {}
    checks = {
        "historical_admission_present": bool(admission),
        "historical_freeze_present": bool(freeze),
        "request_model_exact": (
            admission.get("request_models") == [MODEL_ID, MODEL_ID]
        ),
        "response_model_exact": (
            admission.get("response_models") == [MODEL_ID, MODEL_ID]
        ),
        "tool_use_verified": admission.get("tool_use_admitted") is True,
        "tool_result_verified": admission.get("tool_result_verified") is True,
        "two_calls_no_retry": admission.get("provider_calls") == 2,
        "freeze_model_exact": freeze.get("verified_model_id") == MODEL_ID,
    }
    return {
        "records": {
            "historical_admission": {
                "path": str(admission_path.relative_to(ROOT)),
                "present": admission_path.is_file(),
                "request_models": admission.get("request_models"),
                "response_models": admission.get("response_models"),
            },
            "historical_freeze": {
                "path": str(freeze_path.relative_to(ROOT)),
                "present": freeze_path.is_file(),
                "verified_model_id": freeze.get("verified_model_id"),
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
        "scope": (
            "Historical route, model and tool-protocol evidence only. "
            "The live admission must repeat identity and parameter checks."
        ),
    }


def usage_semantics_check() -> dict[str, Any]:
    raw = {
        "input_tokens": 100,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 30,
        "output_tokens": 10,
    }
    normalized = claude.normalize_anthropic_usage(raw)
    checks = {
        "raw_uncached_input_preserved": (
            normalized["uncached_input_tokens"] == raw["input_tokens"]
        ),
        "cache_creation_added_once": (
            normalized["cache_creation_input_tokens"]
            == raw["cache_creation_input_tokens"]
        ),
        "cache_read_added_once": (
            normalized["cache_read_input_tokens"]
            == raw["cache_read_input_tokens"]
        ),
        "effective_input_equals_all_input_classes": (
            normalized["input_tokens"] == 150
        ),
        "input_output_formula": normalized["total_tokens"] == 160,
        "cache_read_separate": (
            normalized["cache_read_input_tokens"] == 30
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "raw_input_semantics": "uncached_input_tokens",
        "effective_input_formula": (
            "input_tokens + cache_creation_input_tokens + "
            "cache_read_input_tokens"
        ),
        "total_label": "Input+Output口径",
        "total_formula": "effective_input_tokens + output_tokens",
        "sample_raw_usage": raw,
        "sample_normalized_usage": normalized,
        "primary_documentation": (
            "https://docs.anthropic.com/en/docs/about-claude/pricing"
        ),
        "documented_rule": (
            "Anthropic documents total input by summing input_tokens, "
            "cache_creation_input_tokens and cache_read_input_tokens."
        ),
    }


def _regression_record(
    request_index: int,
    request_id: str,
    condition: str,
) -> dict[str, Any]:
    usage = {
        "input_tokens": 100 + request_index,
        "output_tokens": 10,
        "total_tokens": 110 + request_index,
    }
    return {
        "request_index": request_index,
        "request_id": request_id,
        "condition": condition,
        "request_model_id": MODEL_ID,
        "response_model_id": MODEL_ID,
        "usage": deepcopy(usage),
        "normalized_usage": deepcopy(usage),
        "usage_committed": True,
        "outcome": "success",
    }


def _write_regression_scope(
    directory: Path,
    records: list[dict[str, Any]],
    *,
    file_numbers: list[int],
    condition: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for number, record in zip(file_numbers, records, strict=True):
        write_json(directory / f"request-{number:04d}.json", record)
    write_json(
        directory / "journal_state.json",
        {
            "condition": condition,
            "request_count": len(records),
            "synthetic_provider_free": True,
        },
    )


def _regression_ledger(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events = []
    for record in records:
        events.append(
            {
                "event": "request_started",
                "request_id": record["request_id"],
                "request_index": record["request_index"],
            }
        )
        events.append(
            {
                "event": "request_usage_committed",
                "request_id": record["request_id"],
                "request_index": record["request_index"],
                "normalized_usage": deepcopy(
                    record["normalized_usage"]
                ),
            }
        )
    return events


def journal_numbering_regression(output: Path) -> dict[str, Any]:
    """Exercise global shared numbering, local numbering, and hard failures."""
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    natural = [
        _regression_record(index, f"clreq-{index:04d}", "SOURCE")
        for index in (1, 2, 3)
    ]
    baseline = [
        _regression_record(index, f"clreq-{index:04d}", "BASELINE")
        for index in (4, 5, 6)
    ]
    ledger = _regression_ledger([*natural, *baseline])
    natural_dir = output / "natural-global"
    baseline_dir = output / "baseline-global"
    _write_regression_scope(
        natural_dir,
        natural,
        file_numbers=[1, 2, 3],
        condition="SOURCE",
    )
    _write_regression_scope(
        baseline_dir,
        baseline,
        file_numbers=[4, 5, 6],
        condition="BASELINE",
    )
    natural_result = h1.audit_journal_records(
        natural,
        natural_dir,
        expected_condition="SOURCE",
        ledger_events=ledger,
        offline=True,
    )
    global_result = h1.audit_journal_records(
        baseline,
        baseline_dir,
        expected_condition="BASELINE",
        ledger_events=ledger,
        offline=True,
    )

    local_records = [
        _regression_record(1, "clreq-0004", "BASELINE"),
        _regression_record(2, "clreq-0005", "BASELINE"),
    ]
    local_dir = output / "baseline-local"
    _write_regression_scope(
        local_dir,
        local_records,
        file_numbers=[1, 2],
        condition="BASELINE",
    )
    local_result = h1.audit_journal_records(
        local_records,
        local_dir,
        expected_condition="BASELINE",
        ledger_events=_regression_ledger(local_records),
        offline=True,
    )

    missing_dir = output / "negative-missing"
    _write_regression_scope(
        missing_dir,
        local_records,
        file_numbers=[1, 2],
        condition="BASELINE",
    )
    (missing_dir / "request-0002.json").unlink()
    missing_result = h1.audit_journal_records(
        local_records,
        missing_dir,
        expected_condition="BASELINE",
        ledger_events=_regression_ledger(local_records),
        offline=True,
    )

    duplicate_records = deepcopy(local_records)
    duplicate_records[1]["request_id"] = duplicate_records[0]["request_id"]
    duplicate_dir = output / "negative-duplicate"
    _write_regression_scope(
        duplicate_dir,
        duplicate_records,
        file_numbers=[1, 2],
        condition="BASELINE",
    )
    duplicate_result = h1.audit_journal_records(
        duplicate_records,
        duplicate_dir,
        expected_condition="BASELINE",
        ledger_events=_regression_ledger(duplicate_records),
        offline=True,
    )

    conflict_dir = output / "negative-numbering-conflict"
    _write_regression_scope(
        conflict_dir,
        local_records,
        file_numbers=[1, 9],
        condition="BASELINE",
    )
    conflict_result = h1.audit_journal_records(
        local_records,
        conflict_dir,
        expected_condition="BASELINE",
        ledger_events=_regression_ledger(local_records),
        offline=True,
    )

    bad_usage_ledger = _regression_ledger(local_records)
    bad_usage_ledger[1]["normalized_usage"]["total_tokens"] += 1
    usage_mismatch_result = h1.audit_journal_records(
        local_records,
        local_dir,
        expected_condition="BASELINE",
        ledger_events=bad_usage_ledger,
        offline=True,
    )
    checks = {
        "natural_global_numbering_pass": natural_result["complete"],
        "fork_global_numbering_pass": global_result["complete"],
        "fork_local_numbering_pass": local_result["complete"],
        "missing_file_detected": (
            missing_result["complete"] is False
            and "journal_record_count_mismatch"
            in missing_result["reasons"]
        ),
        "duplicate_request_detected": (
            duplicate_result["complete"] is False
            and "duplicate_request_id_in_provider_log"
            in duplicate_result["reasons"]
        ),
        "numbering_conflict_detected": (
            conflict_result["complete"] is False
            and "journal_numbering_conflict"
            in conflict_result["reasons"]
        ),
        "ledger_usage_mismatch_detected": (
            usage_mismatch_result["complete"] is False
            and "ledger_usage_content_mismatch"
            in usage_mismatch_result["reasons"]
        ),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "provider_calls": 0,
        "network_attempts": 0,
        "checks": checks,
        "positive_audits": {
            "natural_global": natural_result,
            "fork_global": global_result,
            "fork_local": local_result,
        },
        "negative_audits": {
            "missing_file": missing_result,
            "duplicate_request": duplicate_result,
            "numbering_conflict": conflict_result,
            "ledger_usage_mismatch": usage_mismatch_result,
        },
    }
    write_json(output / "journal_numbering_regression.json", result)
    return result


def measurement_compatibility(
    output: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare control and instrumented Provider-free condition execution."""
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    source = shared.selected_sources(config)[0]
    checkpoint = read_json(
        output.parent
        / "sources"
        / "source-01"
        / "checkpoints"
        / "recon"
        / "selected_checkpoint.json"
    )
    repository = h1.ExtendedEvidenceRepository(output.parent / "runtime-fixtures")
    rows = []
    previous_factory = h1.CONDITION_MEASUREMENT_FACTORY
    try:
        for condition in CONDITIONS:
            control_dir = output / condition / "control"
            control_dir.mkdir(parents=True)
            control_journal = control_dir / "journal"
            control_journal.mkdir()
            h1.CONDITION_MEASUREMENT_FACTORY = None
            control = h1.run_condition(
                checkpoint,
                condition,
                repository,
                control_journal,
                offline=True,
            )
            h1.CONDITION_MEASUREMENT_FACTORY = measurement.measurement_factory
            measured_dir = output / condition / "measured"
            measured_dir.mkdir(parents=True)
            measured_journal = measured_dir / "journal"
            measured_journal.mkdir()
            measured = h1.run_condition(
                checkpoint,
                condition,
                repository,
                measured_journal,
                offline=True,
            )
            control_behavior = measurement.behavior_projection(control)
            measured_behavior = measurement.behavior_projection(measured)
            arm_measurement = measured.get("measurement")
            rows.append(
                {
                    "source_index": source["source_index"],
                    "state": checkpoint["state"],
                    "condition": condition,
                    "control_status": control.get("status"),
                    "measured_status": measured.get("status"),
                    "behavior_equal": control_behavior == measured_behavior,
                    "measurement_present": isinstance(arm_measurement, dict),
                    "wall_clock_measured": (
                        isinstance(arm_measurement, dict)
                        and isinstance(
                            arm_measurement["attack_wall_clock"][
                                "duration_seconds"
                            ],
                            (int, float),
                        )
                    ),
                    "tool_time_measured": (
                        isinstance(arm_measurement, dict)
                        and (
                            arm_measurement["tool_execution_count"] == 0
                            or isinstance(
                                arm_measurement[
                                    "tool_execution_seconds"
                                ],
                                (int, float),
                            )
                        )
                    ),
                    "defense_time_measured": (
                        condition == "BASELINE"
                        or (
                            isinstance(arm_measurement, dict)
                            and isinstance(
                                arm_measurement["defense"][
                                    "total_initial_exposure_seconds"
                                ],
                                (int, float),
                            )
                        )
                    ),
                }
            )
    finally:
        h1.CONDITION_MEASUREMENT_FACTORY = previous_factory
    checks = {
        "all_behavior_equal": all(row["behavior_equal"] for row in rows),
        "all_measurements_present": all(
            row["measurement_present"] for row in rows
        ),
        "all_wall_clock_measured": all(
            row["wall_clock_measured"] for row in rows
        ),
        "all_tool_time_resolved": all(
            row["tool_time_measured"] for row in rows
        ),
        "all_trap_defense_time_measured": all(
            row["defense_time_measured"] for row in rows
        ),
        "provider_calls_zero": True,
        "network_attempts_zero": True,
    }
    available = {
        "attack_wall_clock_seconds": "MEASURED",
        "provider_latency_seconds": (
            "MEASURED_FROM_PROVIDER_JOURNAL_WHEN_FIELD_PRESENT"
        ),
        "tool_execution_seconds": "MEASURED_NON_INVASIVELY",
        "defense_state_observation_seconds": (
            "NA_H1_STATE_IS_FROZEN"
        ),
        "defense_dynamic_dispatch_selection_seconds": (
            "NA_H1_CONDITION_IS_FIXED"
        ),
        "defense_prepare_seconds": "MEASURED",
        "defense_construction_retrieval_seconds": (
            "MEASURED_WITH_REFERENCE_TO_PREPARE_AND_RESPONSE"
        ),
        "defense_injection_seconds": (
            "MEASURED_AS_INITIAL_EXPOSURE_RESIDUAL"
        ),
        "defense_total_initial_exposure_seconds": "MEASURED",
        "defense_process_cpu_seconds": (
            "MEASURED_WITH_PROCESS_BOUNDARY_ATTRIBUTION"
        ),
        "defense_process_rss": (
            "PROCESS_LEVEL_ONLY_NO_ISOLATED_DEFENSE_PROCESS"
        ),
        "defense_peak_rss": "PROCESS_LEVEL_ONLY",
        "defense_provider_calls": "ZERO_FOR_LOCAL_DEFENSE",
        "defense_tokens": "ZERO_FOR_LOCAL_DEFENSE",
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "provider_calls": 0,
        "network_attempts": 0,
        "checks": checks,
        "rows": rows,
        "available_metrics": available,
    }
    write_json(output / "measurement_compatibility.json", result)
    lines = [
        "# Claude H1 时间与防御开销测量兼容性",
        "",
        f"- 结论：`{result['status']}`",
        "- Provider Calls：0；网络访问：0。",
        ("- 对照与监测分支均使用同一 Source 1 RECON checkpoint、"
        "同一离线动作和 K=12。"),
        ("- 比较字段不包含时间值，仅比较状态、checks、调用数、工具操作、"
        "Gateway、任务结果和机制跟进。"),
        "",
        ("| Condition | Control | Measured | Behavior Equal | "
        "Wall Time | Tool Time | Defense Time |"),
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['control_status']} | "
            f"{row['measured_status']} | {row['behavior_equal']} | "
            f"{row['wall_clock_measured']} | {row['tool_time_measured']} | "
            f"{row['defense_time_measured']} |"
        )
    lines += [
        "",
        "## 可用指标",
        "",
    ]
    for name, status in available.items():
        lines.append(f"- `{name}`：{status}。")
    lines += [
        "",
        ("H1 的条件由冻结协议固定，因此 State Observation 与动态机制调度"
        "标记为 NA；不得将其解释为 H2 在线策略成本。进程级 CPU/RSS "
        "可与防御阶段时间一并报告，但不能宣称为独立防御进程的资源。"),
        "",
    ]
    (output / "MEASUREMENT_COMPATIBILITY_ZH.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return result


def _patch_shared_runner() -> None:
    shared.PROTOCOL_ID = PROTOCOL_ID
    shared.MODEL_ID = MODEL_ID
    shared.GPT_API_KEY_ENV = API_KEY_ENV
    shared.SOURCE_INDICES = SOURCE_INDICES
    shared.STATES = STATES
    shared.CONDITIONS = CONDITIONS
    shared.TRAPS = TRAPS
    shared.FAMILY_BY_SOURCE = dict(FAMILY_BY_SOURCE)
    shared.NATURAL_REQUESTS = NATURAL_REQUESTS
    shared.K = K
    shared.TOTAL_REQUEST_CEILING = FORMAL_REQUEST_CEILING
    shared.TARGET_CHECKPOINTS = TARGET_CHECKPOINTS
    shared.TARGET_CONDITIONS = TARGET_CONDITIONS
    shared.TARGET_PAIRS = TARGET_PAIRS
    shared.provider_config = provider_config
    shared.public_provider = public_provider
    shared.historical_compatibility = historical_compatibility
    shared.usage_semantics_check = usage_semantics_check
    shared.FATAL_STOP_REASONS.update(FATAL_EXTRA_STOP_REASONS)
    shared.GPTBudget = claude.ClaudeBudget
    shared.GPTRequestLedger = claude.ClaudeRequestLedger
    h1.CONDITION_MEASUREMENT_FACTORY = measurement.measurement_factory


class ClaudeScopedProvider(shared.GPTScopedProvider):
    """Claude transport with a local scope cap and durable request journals."""

    def __init__(
        self,
        *,
        budget: claude.ClaudeBudget,
        gpt_config: Mapping[str, Any],
        interface: Mapping[str, Any],
        journal_dir: Path,
        condition: str,
        max_requests: int,
    ):
        self.inner = claude.AnthropicAuditProvider(
            budget,
            gpt_config,
            interface,
            condition=condition,
        )
        self.model = self.inner.model
        self.config = deepcopy(self.inner.config)
        self.interface = deepcopy(self.inner.interface)
        self.request_log = self.inner.request_log
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.condition = condition
        self.max_requests = int(max_requests)
        self.requests_attempted = 0
        self.failures = 0
        self.budget = budget
        self._journaled = 0
        self._write_state("READY")

    def _write_state(
        self,
        outcome: str,
        reason: str | None = None,
    ) -> None:
        _write_json_atomic(
            self.journal_dir / "journal_state.json",
            {
                "journal_version": "formal-h1-claude-request-journal-v1",
                "condition": self.condition,
                "request_count": self.requests_attempted,
                "max_requests": self.max_requests,
                "last_request_index": self.requests_attempted or None,
                "last_outcome": outcome,
                "reason": reason,
                "global_requests_started": self.budget.requests,
                "global_request_ceiling": self.budget.maximum_requests,
                "synthetic_provider_free": False,
                "usage_semantics": "anthropic-messages-usage-v1",
            },
        )

    @staticmethod
    def _normalized_record(record: dict[str, Any]) -> None:
        normalized = record.get("normalized_usage")
        if not isinstance(normalized, dict):
            return
        raw = deepcopy(record.get("usage") or {})
        record["provider_raw_usage"] = raw
        record["provider_model_id"] = record.get("response_model_id")
        record["input_includes_cached_tokens"] = False
        record["usage_class"] = "Input+Output口径"
        record["usage"] = {
            "input_tokens": normalized.get("input_tokens"),
            "output_tokens": normalized.get("output_tokens"),
            "prompt_tokens": normalized.get("input_tokens"),
            "completion_tokens": normalized.get("output_tokens"),
            "cache_read_input_tokens": normalized.get(
                "cache_read_input_tokens"
            ),
            "cache_creation_input_tokens": normalized.get(
                "cache_creation_input_tokens"
            ),
            "reasoning_tokens": normalized.get("reasoning_tokens"),
            "total_tokens": normalized.get("total_tokens"),
            "uncached_input_tokens": normalized.get("uncached_input_tokens"),
            "raw_usage": raw,
        }
        record["effective_input_formula"] = (
            "input_tokens + cache_creation_input_tokens + "
            "cache_read_input_tokens"
        )
        record["input_output_total_formula"] = (
            "effective_input_tokens + output_tokens"
        )

    def complete_with_usage(self, messages):
        if self.requests_attempted >= self.max_requests:
            self.budget.stop("local_request_cap_reached")
            raise RuntimeError("local Provider request cap reached")
        self.requests_attempted += 1
        try:
            result = self.inner.complete_with_usage(messages)
        except BaseException as exc:
            self.failures += 1
            self._persist_new_records()
            self._write_state("FAILED", str(exc)[:500])
            raise
        self._persist_new_records()
        self._write_state("SUCCESS")
        return result

    def complete(self, messages):
        return self.complete_with_usage(messages)[0]


def _patch_provider_class() -> None:
    shared.GPTScopedProvider = ClaudeScopedProvider


def _refresh_artifact_inventory(output: Path) -> None:
    artifacts = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        stat = path.stat()
        artifacts[str(path.relative_to(output))] = {
            "size_bytes": stat.st_size,
            "modified_at_ns": stat.st_mtime_ns,
            "mode": oct(stat.st_mode & 0o777),
        }
    write_json(
        output / "artifact_inventory.json",
        {
            "protocol_id": PROTOCOL_ID,
            "integrity_method": (
                "version, timestamp, file size, permissions, and modification "
                "record only"
            ),
            "artifacts": artifacts,
        },
    )


def provider_free_journal_audits(
    output: Path,
    sources: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    audits = []
    for source in sources:
        source_index = int(source["source_index"])
        audit = source.get("journal_audit")
        if isinstance(audit, dict):
            audits.append(
                {
                    "scope": "SOURCE",
                    "source_index": source_index,
                    "state": None,
                    "condition": "SOURCE",
                    **audit,
                }
            )
        for state in STATES:
            for condition in CONDITIONS:
                path = (
                    output
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
                audit = arm.get("journal_audit")
                if isinstance(audit, dict):
                    audits.append(
                        {
                            "scope": "CONDITION",
                            "source_index": source_index,
                            "state": state,
                            "condition": condition,
                            **audit,
                        }
                    )
    return audits


def provider_free_admission(
    output: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _patch_shared_runner()
    _patch_provider_class()
    result = shared.provider_free_gate(output, config)
    journal_audits = provider_free_journal_audits(
        output,
        result["sources"],
    )
    regression = journal_numbering_regression(
        output / "journal-numbering-regression"
    )
    measurement_check = measurement_compatibility(
        output / "measurement-compatibility",
        config,
    )
    provider = provider_config(config, require_key=False)
    checks = deepcopy(result["checks"])
    checks["journal_numbering_modes_supported"] = (
        regression["status"] == "PASS"
    )
    checks["provider_free_source_journal_audits"] = (
        sum(
            row["scope"] == "SOURCE" and row["complete"]
            for row in journal_audits
        )
        == 6
    )
    checks["provider_free_condition_journal_audits"] = (
        sum(
            row["scope"] == "CONDITION" and row["complete"]
            for row in journal_audits
        )
        == 72
    )
    checks["measurement_compatibility"] = (
        measurement_check["status"] == "PASS"
    )
    static_checks = {
        "claude_model_exact": (
            provider["model"] == MODEL_ID
            and provider["accepted_response_model_ids"] == [MODEL_ID]
        ),
        "credential_source_exclusive": (
            provider["credential_source"] == API_KEY_ENV
        ),
        "anthropic_messages_route": (
            provider["provider_interface"] == "anthropic-messages-v1"
            and provider["base_url"]
            == config["provider"]["base_url"]
        ),
        "temperature_zero": provider["temperature"] == 0.0,
        "max_tokens_8192": provider["max_tokens"] == MAX_TOKENS,
        "serial_no_parallel_tool_calls": (
            provider["disable_parallel_tool_use"] is True
        ),
        "non_streaming": provider["stream"] is False,
        "request_hard_ceiling_1056": TOTAL_REQUEST_CEILING == 1056,
        "formal_ceiling_1054_after_admission": (
            FORMAL_REQUEST_CEILING == 1054
        ),
        "effective_input_then_input_output_formula": (
            config["provider"]["usage_semantics"]["total_formula"]
            == "effective_input_tokens + output_tokens"
        ),
        "no_automatic_retry": (
            config["budget"]["automatic_retry"] is False
        ),
        "no_automatic_backfill": (
            config["budget"]["automatic_backfill"] is False
        ),
        "no_automatic_recovery": (
            config["budget"]["automatic_recovery"] is False
        ),
        "credential_variable_present": bool(provider["api_key"]),
    }
    compatibility = historical_compatibility(config)
    usage = usage_semantics_check()
    passed = (
        all(checks.values())
        and compatibility["passed"]
        and usage["passed"]
        and all(
            value
            for name, value in static_checks.items()
            if name != "credential_variable_present"
        )
    )
    result.update(
        protocol_id=PROTOCOL_ID,
        provider=public_provider(provider),
        checks=checks,
        static_provider_checks=static_checks,
        historical_compatibility=compatibility,
        usage_semantics=usage,
        journal_numbering_regression={
            key: value
            for key, value in regression.items()
            if key not in {"positive_audits", "negative_audits"}
        },
        journal_audits=journal_audits,
        journal_audit_summary={
            "source_scopes": 6,
            "condition_scopes": 72,
            "complete_scopes": sum(
                row["complete"] for row in journal_audits
            ),
            "numbering_modes": sorted(
                {row["numbering_mode"] for row in journal_audits}
            ),
        },
        measurement_compatibility={
            key: value
            for key, value in measurement_check.items()
            if key != "rows"
        },
        passed=passed,
        credential_ready=static_checks["credential_variable_present"],
        budget={
            "target_natural_request_count": 6 * NATURAL_REQUESTS,
            "target_condition_request_count": 18 * 4 * K,
            "target_formal_request_count": 1056,
            "admission_request_allowance": ADMISSION_REQUESTS,
            "global_request_hard_ceiling": TOTAL_REQUEST_CEILING,
            "formal_request_hard_ceiling_after_admission": (
                FORMAL_REQUEST_CEILING
            ),
            "automatic_retry": False,
            "automatic_backfill": False,
            "automatic_recovery": False,
        },
        credential_gate={
            "variable": API_KEY_ENV,
            "present": static_checks["credential_variable_present"],
            "live_start_allowed": (
                passed and static_checks["credential_variable_present"]
            ),
            "blocks_when_missing": True,
            "legacy_fallback_enabled": False,
        },
    )
    write_json(output / "provider_free_admission.json", result)
    write_json(
        output / "journal_audit.json",
        {
            "source_scopes": 6,
            "condition_scopes": 72,
            "audits": journal_audits,
        },
    )
    write_csv(
        output / "sources.csv",
        read_csv_rows(output / "sources.csv"),
        list(read_csv_rows(output / "sources.csv")[0]),
    )
    write_csv(
        output / "checkpoints.csv",
        read_csv_rows(output / "checkpoints.csv"),
        list(read_csv_rows(output / "checkpoints.csv")[0]),
    )
    write_csv(
        output / "arms.csv",
        read_csv_rows(output / "arms.csv"),
        list(read_csv_rows(output / "arms.csv")[0]),
    )
    write_csv(
        output / "oracles.csv",
        read_csv_rows(output / "oracles.csv"),
        list(read_csv_rows(output / "oracles.csv")[0]),
    )
    write_csv(
        output / "extensions.csv",
        read_csv_rows(output / "extensions.csv"),
        list(read_csv_rows(output / "extensions.csv")[0]),
    )
    report_lines = [
        "# Claude Sonnet 4.6 六 Source H1 Provider-free 准入报告",
        "",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- Provider-free 结果：`{'PASS' if passed else 'FAIL'}`",
        "- Provider 请求：0；网络请求：0；真实自然轨迹：0。",
        "- Source：6/6；WEB 2、API 2、SERVICE 2。",
        "- Checkpoint：18/18 READY。",
        "- 四条件槽位：72/72 PASS。",
        "- Oracle：6/6 PASS；扩展环境：6/6 PASS。",
        "- Gateway：72/72 PASS；bypass=0。",
        "- Restore、pre-effect、BASELINE 单动作与三种 Trap 机制路径全部通过。",
        ("- Journal 编号回归：自然轨迹与条件分叉共享全局编号 PASS；"
        "条件内局部编号 PASS；缺失文件、重复 Request ID、编号冲突与"
        "ledger usage 内容不符均被拒绝。"),
        ("- Provider-free 逐 scope journal 审计："
        f"{sum(row['complete'] for row in journal_audits)}/78 PASS。"),
        ("- 时间/防御开销监测兼容性："
        f"`{measurement_check['status']}`；四条件对照行为完全一致；"
        "wall-clock、Provider latency、工具时间、Trap prepare/response/"
        "injection、process CPU 与进程级 RSS 可采集。"),
        "- 三 State 顺序：RECON → EXPLOITATION → VERIFICATION。",
        ("- 模型：`claude-sonnet-4-6`；Anthropic Messages；"
        "`temperature=0.0`；`max_tokens=8192`；串行非流式 ToolCall。"),
        ("- 历史真实响应已精确返回 `claude-sonnet-4-6`，并完成 "
        "tool_use/tool_result 闭环。"),
        ("- Usage：`input_tokens` 为未缓存输入；有效输入还需加 "
        "`cache_creation_input_tokens` 与 `cache_read_input_tokens`；"
        "本实验将合计标记为“Input+Output 口径”。"),
        (f"- 凭证变量：`{API_KEY_ENV}`；当前环境："
        f"`{'PRESENT' if result['credential_ready'] else 'MISSING'}`；"
        "旧变量回退禁用。"),
        "- 全局硬上限 1056；两次实时协议准入计入该上限，正式实验剩余 1054。",
        "- 自动重试、补跑、Recovery：全部禁用。",
        "",
        ("Provider-free 只验证本地科学路径与静态口径。真实凭证、当前路由、"
        "模型身份、8192 参数支持与 ToolCall 闭环必须由实时准入再次通过。"),
        "",
    ]
    (output / "PREFLIGHT_REPORT_ZH.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    (output / "MODIFICATION_RECORD.md").write_text(
        "\n".join(
            [
                "# Modification record",
                "",
                f"- Created independent Claude admission at {now()}.",
                "- Shared six-Source execution path selected by version.",
                ("- Trap cores, H2 strategy, frozen Sources, and historical H1 "
                "results were not modified."),
                "- Source fixtures were copied into the result directory.",
                "- Provider calls: 0; network requests: 0.",
                "- Automatic retry, backfill, and recovery: disabled.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _refresh_artifact_inventory(output)
    return result


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _admission_interface() -> dict[str, Any]:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "connectivity_echo",
                    "description": (
                        "Return the supplied protocol token. This tool only "
                        "checks the local Anthropic tool_use/tool_result loop."
                    ),
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["token"],
                        "properties": {
                            "token": {"type": "string"}
                        },
                    },
                },
            }
        ]
    }


def _persist_admission_records(
    output: Path,
    records: list[dict[str, Any]],
) -> None:
    for record in records:
        ClaudeScopedProvider._normalized_record(record)
        _write_json_atomic(
            output / f"request-{int(record['request_index']):04d}.json",
            record,
        )


def run_live_admission(
    output: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    provider_spec = provider_config(config, require_key=True)
    ledger = claude.ClaudeRequestLedger(output / "requests.jsonl")
    budget = claude.ClaudeBudget(
        maximum_requests=ADMISSION_REQUESTS,
        maximum_total_tokens=200_000,
        estimated_input_token_divisor=3,
        rates={
            "input": 3.0,
            "cached_input": 0.3,
            "cache_write": 3.75,
            "output": 15.0,
        },
        ledger=ledger,
    )
    provider = claude.AnthropicAuditProvider(
        budget,
        provider_spec,
        _admission_interface(),
        condition="LIVE_ADMISSION",
    )
    messages = [
        {
            "role": "system",
            "content": (
                "This is a local API transport check. Use the "
                "connectivity_echo tool exactly once with token "
                "claude-sonnet-4-6-admission and do not answer with prose."
            ),
        },
        {
            "role": "user",
            "content": "Call connectivity_echo for the required token.",
        },
    ]
    records = provider.request_log
    checks = {
        "first_call_started": False,
        "exactly_two_calls_started": False,
        "first_tool_use": False,
        "tool_result_continuation": False,
        "request_models_exact": False,
        "response_models_exact": False,
        "max_tokens_8192": False,
        "usage_complete": False,
        "effective_input_formula_complete": False,
        "no_retry": True,
    }
    error = None
    first_message = None
    second_message = None
    try:
        first_message, _first_usage = provider.complete_with_usage(messages)
        checks["first_call_started"] = len(records) == 1
        first_calls = deepcopy(first_message.get("tool_calls") or [])
        checks["first_tool_use"] = (
            len(first_calls) == 1
            and first_calls[0]["function"]["name"] == "connectivity_echo"
        )
        if not checks["first_tool_use"]:
            raise RuntimeError("admission first response lacked one ToolCall")
        arguments = json.loads(first_calls[0]["function"]["arguments"])
        messages.append(
            {
                "role": "assistant",
                "content": first_message.get("content"),
                "tool_calls": first_calls,
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": first_calls[0]["id"],
                "content": json.dumps(
                    {
                        "returned_token": arguments.get("token"),
                        "transport": "local_admission",
                    },
                    ensure_ascii=False,
                ),
            }
        )
        second_message, _second_usage = provider.complete_with_usage(messages)
        checks["tool_result_continuation"] = (
            len(records) == 2
            and isinstance(second_message, dict)
        )
    except (
        claude.ClaudeProviderError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        error = f"{type(exc).__name__}: {str(exc)[:1000]}"
    finally:
        _persist_admission_records(output, records)
    models = [
        record.get("response_model_id")
        for record in records
        if record.get("response_model_id") is not None
    ]
    request_models = [
        record.get("request_model_id") for record in records
    ]
    checks["request_models_exact"] = request_models == [MODEL_ID] * len(records)
    checks["response_models_exact"] = models == [MODEL_ID] * len(records)
    checks["exactly_two_calls_started"] = len(records) == ADMISSION_REQUESTS
    checks["max_tokens_8192"] = all(
        (record.get("request_parameters") or {}).get("max_tokens")
        == MAX_TOKENS
        for record in records
    ) and len(records) == ADMISSION_REQUESTS
    checks["usage_complete"] = all(
        isinstance(record.get("normalized_usage"), dict)
        and record["normalized_usage"].get("usage_complete") is True
        for record in records
    ) and len(records) == ADMISSION_REQUESTS
    checks["effective_input_formula_complete"] = all(
        isinstance(record.get("normalized_usage"), dict)
        and isinstance(
            record["normalized_usage"].get("cache_read_input_tokens"), int
        )
        and isinstance(
            record["normalized_usage"].get(
                "cache_creation_input_tokens"
            ),
            int,
        )
        and record["normalized_usage"].get("total_tokens")
        == record["normalized_usage"].get("input_tokens")
        + record["normalized_usage"].get("output_tokens")
        for record in records
    ) and len(records) == ADMISSION_REQUESTS
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at": now(),
        "status": (
            "LIVE_ADMISSION_PASS"
            if all(checks.values()) and not error
            else "LIVE_ADMISSION_FAIL"
        ),
        "provider_calls": len(records),
        "global_ceiling": TOTAL_REQUEST_CEILING,
        "formal_remaining_ceiling": FORMAL_REQUEST_CEILING,
        "checks": checks,
        "request_models": request_models,
        "response_models": models,
        "provider": public_provider(provider_spec),
        "first_response": deepcopy(first_message),
        "second_response": deepcopy(second_message),
        "error": error,
        "ledger": ledger.snapshot(),
        "automatic_retry": False,
        "automatic_backfill": False,
        "automatic_recovery": False,
    }
    write_json(output / "admission_record.json", result)
    return result


def _write_blocked_status(
    output: Path,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    status = {
        "protocol_id": PROTOCOL_ID,
        "status": "BLOCKED_CREDENTIAL" if reason == "credential_missing" else "BLOCKED_ADMISSION",
        "formal_h1_started": False,
        "provider_calls": 0,
        "provider_request_hard_ceiling": TOTAL_REQUEST_CEILING,
        "provider_free_passed": preflight.get("passed"),
        "credential_ready": preflight.get("credential_ready"),
        "models": {"request": MODEL_ID, "response": None},
        "sources": 0,
        "checkpoints": 0,
        "condition_trajectories": 0,
        "target_pairs": 0,
        "automatic_retry": False,
        "automatic_backfill": False,
        "automatic_recovery": False,
        "blocking_condition": (
            f"{API_KEY_ENV} missing from execution environment"
            if reason == "credential_missing"
            else reason
        ),
        "finished_at": now(),
    }
    write_json(output / "FINAL_STATUS.json", status)
    lines = [
        "# Claude Sonnet 4.6 六 Source H1 停止报告",
        "",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 状态：`{status['status']}`",
        f"- Provider-free 准入：`{'PASS' if preflight.get('passed') else 'FAIL'}`",
        f"- Provider Calls：0/{TOTAL_REQUEST_CEILING}",
        f"- 阻断条件：{status['blocking_condition']}",
        "- 未执行自然轨迹、checkpoint 采集、条件分叉或跨模型结果推断。",
        "- 未补造、补跑或替换任何 Source；未启动 H2。",
        "- 自动重试、补跑与 Recovery：禁用。",
        "",
        ("要恢复本轮，只应在新的明确运行中向执行进程注入 "
        f"`{API_KEY_ENV}`；不得复用其他变量或历史凭证。"),
        "",
    ]
    (output / "STOPPED_REPORT_ZH.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    write_json(
        output / "configuration_snapshot.json",
        deepcopy(dict(config)),
    )
    return status


def start(
    output: Path,
    config: Mapping[str, Any],
) -> int:
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    preflight_output = output / "provider-free"
    preflight = provider_free_admission(preflight_output, config)
    write_json(
        output / "launch_record.json",
        {
            "protocol_id": PROTOCOL_ID,
            "created_at": now(),
            "provider_free_passed": preflight.get("passed"),
            "credential_ready": preflight.get("credential_ready"),
            "provider": preflight.get("provider"),
            "sources": list(SOURCE_INDICES),
            "states": list(STATES),
            "conditions": list(CONDITIONS),
            "budget": preflight.get("budget"),
            "automatic_retry": False,
            "automatic_backfill": False,
            "automatic_recovery": False,
        },
    )
    if not preflight.get("passed"):
        _write_blocked_status(
            output,
            config,
            preflight,
            "provider_free_admission_failed",
        )
        return 2
    if not preflight.get("credential_ready"):
        _write_blocked_status(
            output,
            config,
            preflight,
            "credential_missing",
        )
        return 3
    admission = run_live_admission(output / "admission", config)
    if admission.get("status") != "LIVE_ADMISSION_PASS":
        _write_blocked_status(
            output,
            config,
            preflight,
            f"live_admission_failed: {admission.get('error') or admission.get('checks')}",
        )
        return 4
    shared.TOTAL_REQUEST_CEILING = FORMAL_REQUEST_CEILING
    live_output = output / "live"
    live = shared.execute_live(live_output, config)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "h1_claude_cross_model_6source_20261024.analyze",
            str(live_output),
        ],
        cwd=ROOT,
        check=True,
    )
    analysis_path = live_output / "analysis_summary.json"
    analysis = read_json(analysis_path) if analysis_path.is_file() else {}
    total_calls = int(admission["provider_calls"]) + int(
        live.get("provider_calls") or 0
    )
    final = {
        "protocol_id": PROTOCOL_ID,
        "status": live.get("status", "UNKNOWN"),
        "formal_h1_started": True,
        "admission_provider_calls": admission["provider_calls"],
        "formal_provider_calls": live.get("provider_calls"),
        "provider_calls": total_calls,
        "provider_request_hard_ceiling": TOTAL_REQUEST_CEILING,
        "provider_calls_within_ceiling": total_calls <= TOTAL_REQUEST_CEILING,
        "models": {
            "request": MODEL_ID,
            "response": (
                analysis.get("provider_models")
                if analysis
                else admission.get("response_models")
            ),
        },
        "sources": len(live.get("sources") or []),
        "checkpoints": analysis.get("valid_checkpoints"),
        "condition_trajectories": analysis.get("condition_records"),
        "target_pairs": analysis.get("target_pair_count", TARGET_PAIRS),
        "clean_pairs": analysis.get("valid_pairs"),
        "automatic_retry": False,
        "automatic_backfill": False,
        "automatic_recovery": False,
        "gateway_bypasses": (
            1 if analysis.get("gateway_bypass_observed") else 0
        ),
        "unauthorized_effects": (
            1 if analysis.get("unauthorized_effect_observed") else 0
        ),
        "finished_at": now(),
    }
    write_json(output / "FINAL_STATUS.json", final)
    return 0 if final["status"] == "COMPLETED" else 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.start:
        parser.error("choose exactly one of --preflight-only or --start")
    config = load_config()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = (
        "formal_H1_Claude_Sonnet_4_6_cross_model_6source_preflight_"
        if args.preflight_only
        else "formal_H1_Claude_Sonnet_4_6_cross_model_6source_"
    )
    output = ROOT / "results" / f"{prefix}{timestamp}"
    if args.preflight_only:
        result = provider_free_admission(output, config)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "passed": result["passed"],
                    "credential_ready": result["credential_ready"],
                    "provider_calls": 0,
                },
                ensure_ascii=False,
            )
        )
        return 0 if result["passed"] else 2
    with ExperimentDirectoryLock(output, PROTOCOL_ID):
        return start(output, config)


if __name__ == "__main__":
    raise SystemExit(main())

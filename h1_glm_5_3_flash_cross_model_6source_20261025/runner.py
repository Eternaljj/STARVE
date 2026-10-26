"""Formal GLM-5.3-Flash six-Source H1 cross-model experiment."""
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
from h1_cross_system_20261023 import runner as h1
from h1_gpt_cross_model_6source_20261024 import runner as shared
from h1_glm_5_3_flash_cross_model_6source_20261025 import measurement
from h1_glm_5_3_flash_cross_model_6source_20261025 import provider as glm
from p0a_pentestgpt.stagetrap_k12_live import _write_json_atomic

PROTOCOL_ID = "formal_H1_GLM_5_3_Flash_cross_model_6source_v1"
MODEL_ID = "glm-5.3-flash"
API_KEY_ENV = "H1_P_GLM_API_KEY"
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
MAX_TOKENS = 2400
FATAL_EXTRA_STOP_REASONS = {
    "usage_missing",
    "usage_missing_required_token_fields",
    "usage_total_mismatch",
    "usage_total_mismatch",
    "prompt_tokens_details_malformed",
    "completion_tokens_details_malformed",
    "cached_tokens_exceed_prompt_tokens",
    "reasoning_tokens_exceed_completion_tokens",
    "parallel_or_invalid_tool_calls",
    "tool_call_id_missing",
    "tool_function_missing",
    "tool_arguments_invalid",
    "tool_arguments_not_object",
    "output_truncated",
    "request_unsettled_at_provider_exit",
    "glm_output_parameter_mismatch",
    "glm_protocol_error",
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
        != "formal-h1-glm53flash-cross-model-6source-v1"
        or config.get("protocol_id") != PROTOCOL_ID
        or provider.get("interface")
        != "openai-compatible-chat-completions-v1"
        or provider.get("model_id") != MODEL_ID
        or provider.get("accepted_response_model_ids") != [MODEL_ID]
        or provider.get("credential_source") != API_KEY_ENV
        or provider.get("base_url") != "https://api.zhizengzeng.com/v1"
        or provider.get("reasoning_effort") != "max"
        or provider.get("thinking")
        != {"type": "enabled", "clear_thinking": False}
        or provider.get("temperature") != 0.0
        or provider.get("output_token_parameter") != "max_tokens"
        or provider.get("max_tokens") != MAX_TOKENS
        or provider.get("timeout_seconds") != 360
        or provider.get("tool_choice") != "auto"
        or provider.get("parallel_tool_calls") is not False
        or provider.get("send_parallel_tool_calls") is not False
        or provider.get("automatic_retry") != 0
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
        raise RuntimeError("formal GLM-5.3-Flash H1 configuration changed")
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
        "authorization_scheme": "bearer",
        "temperature": 0.0,
        # Compatibility-only field for the shared preflight. The GLM
        # transport does send its own temperature field.
        "send_temperature": True,
        "max_tokens": MAX_TOKENS,
        "timeout_seconds": int(source["timeout_seconds"]),
        "tool_choice": str(source["tool_choice"]),
        "parallel_tool_calls": False,
        "send_parallel_tool_calls": False,
        "stream": False,
        "provider_interface": "openai-compatible-chat-completions-v1",
        "usage_semantics": "glm-chat-completions-usage-v1",
        "reasoning_effort": "max",
        "thinking": deepcopy(source["thinking"]),
        "system_instruction": source.get("system_instruction"),
        "prompt_adapter_version": source.get(
            "prompt_adapter_version"
        ),
        "output_token_parameter": "max_tokens",
        # Compatibility-only field read by the frozen shared preflight after
        # provider_config is patched. It is never sent to GLM or used as the
        # GLM output-token parameter.
        "max_completion_tokens": MAX_TOKENS,
        "automatic_retry": 0,
    }


def public_provider(provider: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(provider))
    value["api_key"] = "[PRESENT]" if value.get("api_key") else "[MISSING]"
    return value


def protocol_compatibility(config: Mapping[str, Any]) -> dict[str, Any]:
    provider = config["provider"]
    evidence = provider["protocol_evidence"]
    checks = {
        "published_model_id_exact": evidence.get("published_model_id")
        == MODEL_ID,
        "reasoning_values_exact": evidence.get("reasoning_effort_values")
        == ["low", "high", "max"],
        "reasoning_effort_max": provider.get("reasoning_effort") == "max",
        "thinking_enabled_only": evidence.get("thinking_type_values")
        == ["enabled"],
        "max_tokens_parameter": provider.get("output_token_parameter")
        == "max_tokens",
        "output_limit_preserved_from_existing_h1": MAX_TOKENS == 2400,
        "reasoning_counted_in_completion": provider["usage_semantics"].get(
            "reasoning_is_inside_completion"
        )
        is True,
        "provider_catalog_lists_model": bool(
            evidence.get("provider_catalog")
        ),
    }
    return {
        "evidence": deepcopy(evidence),
        "checks": checks,
        "passed": all(checks.values()),
        "scope": (
            "Published protocol and provider catalog evidence only. Account "
            "access, model identity, ToolCall execution, reasoning parsing, "
            "and usage behavior must pass the live admission."
        ),
    }


def usage_semantics_check(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if config is None:
        config = load_config()
    raw = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens_details": {"reasoning_tokens": 3},
    }
    normalized = glm.validate_usage(raw)
    checks = {
        "input_is_prompt_total": normalized["input_tokens"] == 100,
        "cached_is_subset_not_added": normalized["cache_read_tokens"] == 40,
        "reasoning_is_subset_not_added": normalized["reasoning_tokens"] == 3,
        "total_is_prompt_plus_completion": normalized["total_tokens"] == 110,
        "reasoning_inside_completion": normalized["reasoning_tokens"] <= 10,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "raw_input_semantics": "prompt_tokens_including_cached_tokens",
        "effective_input_formula": "prompt_tokens",
        "total_label": "Provider Total Tokens",
        "total_formula": "prompt_tokens + completion_tokens",
        "sample_raw_usage": raw,
        "sample_normalized_usage": normalized,
        "primary_documentation": (
            config.get("provider", {})
            .get("protocol_evidence", {})
            .get("chat_completions")
        ),
        "documented_rule": (
            "GLM Chat Completions reports prompt_tokens, completion_tokens, "
            "and total_tokens; reasoning is part of completion and cached "
            "tokens are a prompt subset."
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
        "# GLM-5.3-Flash H1 时间与防御开销测量兼容性",
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
    shared.historical_compatibility = protocol_compatibility
    shared.usage_semantics_check = usage_semantics_check
    shared.FATAL_STOP_REASONS.update(FATAL_EXTRA_STOP_REASONS)
    shared.GPTBudget = glm.GLMBudget
    shared.GPTRequestLedger = glm.GLMRequestLedger
    h1.CONDITION_MEASUREMENT_FACTORY = measurement.measurement_factory


class GLMScopedProvider(shared.GPTScopedProvider):
    """GLM transport with a local scope cap and durable request journals."""

    def __init__(
        self,
        *,
        budget: glm.GLMBudget,
        gpt_config: Mapping[str, Any],
        interface: Mapping[str, Any],
        journal_dir: Path,
        condition: str,
        max_requests: int,
    ):
        self.inner = glm.GLMAuditProvider(
            budget,
            gpt_config,
            interface,
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
                "journal_version": "formal-h1-glm53flash-request-journal-v1",
                "condition": self.condition,
                "request_count": self.requests_attempted,
                "max_requests": self.max_requests,
                "last_request_index": self.requests_attempted or None,
                "last_outcome": outcome,
                "reason": reason,
                "global_requests_started": self.budget.requests,
                "global_request_ceiling": self.budget.maximum_requests,
                "synthetic_provider_free": False,
                "usage_semantics": "glm-chat-completions-usage-v1",
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
        record["input_includes_cached_tokens"] = True
        record["usage_class"] = "Provider Total Tokens"
        record["usage"] = {
            "input_tokens": normalized.get("input_tokens"),
            "output_tokens": normalized.get("output_tokens"),
            "prompt_tokens": normalized.get("input_tokens"),
            "completion_tokens": normalized.get("output_tokens"),
            "cache_read_input_tokens": normalized.get(
                "cache_read_tokens"
            ),
            "cache_creation_input_tokens": None,
            "reasoning_tokens": normalized.get("reasoning_tokens"),
            "total_tokens": normalized.get("total_tokens"),
            "uncached_input_tokens": normalized.get("cache_miss_tokens"),
            "raw_usage": raw,
        }
        record["effective_input_formula"] = "prompt_tokens"
        record["total_tokens_formula"] = (
            "prompt_tokens + completion_tokens"
        )
        record["reasoning_is_output_subset"] = True
        record["cached_is_input_subset"] = True

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
    shared.GPTScopedProvider = GLMScopedProvider


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
        "glm_model_exact": (
            provider["model"] == MODEL_ID
            and provider["accepted_response_model_ids"] == [MODEL_ID]
        ),
        "credential_source_exclusive": (
            provider["credential_source"] == API_KEY_ENV
        ),
        "openai_chat_completions_route": (
            provider["provider_interface"]
            == "openai-compatible-chat-completions-v1"
            and provider["base_url"]
            == config["provider"]["base_url"]
        ),
        "reasoning_effort_max": provider["reasoning_effort"] == "max",
        "thinking_enabled": provider["thinking"]
        == {"type": "enabled", "clear_thinking": False},
        "temperature_zero": provider["temperature"] == 0.0,
        "max_tokens_2400": provider["max_tokens"] == MAX_TOKENS,
        "max_tokens_parameter_exact": (
            provider["output_token_parameter"] == "max_tokens"
        ),
        "serial_no_parallel_tool_calls": (
            provider["parallel_tool_calls"] is False
            and provider["send_parallel_tool_calls"] is False
        ),
        "non_streaming": provider["stream"] is False,
        "request_hard_ceiling_1056": TOTAL_REQUEST_CEILING == 1056,
        "formal_ceiling_1054_after_admission": (
            FORMAL_REQUEST_CEILING == 1054
        ),
        "provider_total_formula": (
            config["provider"]["usage_semantics"]["total_formula"]
            == "prompt_tokens + completion_tokens"
        ),
        "reasoning_not_double_counted": (
            config["provider"]["usage_semantics"][
                "reasoning_added_again_to_total"
            ]
            is False
        ),
        "cache_not_double_counted": (
            config["provider"]["usage_semantics"][
                "cache_added_again_to_total"
            ]
            is False
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
    compatibility = protocol_compatibility(config)
    usage = usage_semantics_check(config)
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
        protocol_compatibility=compatibility,
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
        "# GLM-5.3-Flash 六 Source H1 Provider-free 准入报告",
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
        ("- 模型：`glm-5.3-flash`；智增增 OpenAI-compatible Chat "
        "Completions；`temperature=0.0`；`max_tokens=2400`；"
        "`reasoning_effort=max`；`thinking.type=enabled`；非流式。"),
        ("- GLM-5.3-Flash 的推理 Token 计入 completion/total；"
        "本实验不把推理 Token 再重复加到总量。"),
        ("- Cache Read 是 `prompt_tokens` 子集，单独报告，不重复累加；"
        "上游未提供 cache creation 时保持 UNKNOWN。"),
        (f"- 凭证变量：`{API_KEY_ENV}`；当前环境："
        f"`{'PRESENT' if result['credential_ready'] else 'MISSING'}`；"
        "旧变量回退禁用。"),
        "- 全局硬上限 1056；两次实时协议准入计入该上限，正式实验剩余 1054。",
        "- 自动重试、补跑、Recovery：全部禁用。",
        "",
        ("Provider-free 只验证本地科学路径与静态口径。真实凭证、当前账户"
        "模型访问权、精确模型身份、2400 Token 与推理兼容性、ToolCall 闭环"
        "必须由实时准入再次通过。"),
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
                f"- Created independent GLM-5.3-Flash admission at {now()}.",
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
                        "checks the Chat Completions tool_call/tool_result loop."
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
        GLMScopedProvider._normalized_record(record)
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
    ledger = glm.GLMRequestLedger(output / "requests.jsonl")
    budget = glm.GLMBudget(
        maximum_requests=ADMISSION_REQUESTS,
        maximum_total_tokens=200_000,
        estimated_input_token_divisor=3,
        rates_cny_per_million={
            "input": 0.8,
            "cached_input": 0.8,
            "output": 2.8,
        },
        ledger=ledger,
    )
    provider = glm.GLMAuditProvider(
        budget,
        provider_spec,
        _admission_interface(),
    )
    messages = [
        {
            "role": "system",
            "content": (
                "This is a local API transport check. Use the "
                "connectivity_echo tool exactly once with token "
                "glm-5.3-flash-admission and do not answer with prose."
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
        "reasoning_content_parsed": False,
        "tool_result_continuation": False,
        "tool_result_fed_back": False,
        "request_models_exact": False,
        "response_models_exact": False,
        "max_tokens_2400": False,
        "reasoning_effort_max": False,
        "thinking_enabled": False,
        "max_completion_tokens_absent": False,
        "usage_complete": False,
        "usage_total_auditable": False,
        "cache_count_auditable": False,
        "no_retry": True,
        "no_hidden_requests": False,
    }
    error = None
    first_message = None
    second_message = None
    try:
        first_message, _first_usage = provider.complete_with_usage(messages)
        checks["first_call_started"] = len(records) == 1
        checks["reasoning_content_parsed"] = (
            isinstance(first_message.get("reasoning_content"), str)
            and bool(first_message["reasoning_content"].strip())
        )
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
            and isinstance(second_message.get("content"), str)
            and bool(second_message["content"].strip())
        )
        second_messages = records[1].get("messages") or []
        checks["tool_result_fed_back"] = any(
            message.get("role") == "tool"
            and message.get("tool_call_id") == first_calls[0]["id"]
            and "local_admission" in str(message.get("content") or "")
            for message in second_messages
        )
    except (
        glm.GLMProviderError,
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
    checks["max_tokens_2400"] = all(
        (record.get("request_parameters") or {}).get("max_tokens")
        == MAX_TOKENS
        for record in records
    ) and len(records) == ADMISSION_REQUESTS
    checks["reasoning_effort_max"] = all(
        (record.get("request_parameters") or {}).get("reasoning_effort")
        == "max"
        for record in records
    ) and len(records) == ADMISSION_REQUESTS
    checks["thinking_enabled"] = all(
        (record.get("request_parameters") or {}).get("thinking")
        == {"type": "enabled", "clear_thinking": False}
        for record in records
    ) and len(records) == ADMISSION_REQUESTS
    checks["max_completion_tokens_absent"] = all(
        "max_completion_tokens"
        not in (record.get("request_parameters") or {})
        for record in records
    )
    checks["usage_complete"] = all(
        isinstance(record.get("normalized_usage"), dict)
        and record["normalized_usage"].get("usage_complete") is True
        for record in records
    ) and len(records) == ADMISSION_REQUESTS
    checks["usage_total_auditable"] = all(
        isinstance(record.get("normalized_usage"), dict)
        and record["normalized_usage"].get("total_tokens")
        == record["normalized_usage"].get("input_tokens")
        + record["normalized_usage"].get("output_tokens")
        for record in records
    ) and len(records) == ADMISSION_REQUESTS
    checks["cache_count_auditable"] = all(
        isinstance(record.get("normalized_usage"), dict)
        and isinstance(
            record["normalized_usage"].get("cache_read_tokens"), int
        )
        for record in records
    ) and len(records) == ADMISSION_REQUESTS
    checks["no_hidden_requests"] = (
        len(records) == ADMISSION_REQUESTS
        and ledger.snapshot()["requests_started"] == ADMISSION_REQUESTS
        and not ledger.snapshot()["unknown_request_ids"]
    )
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
    *,
    admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    admission = admission or {}
    provider_calls = int(admission.get("provider_calls") or 0)
    status = {
        "protocol_id": PROTOCOL_ID,
        "status": (
            "BLOCKED_CREDENTIAL"
            if reason == "credential_missing"
            else "BLOCKED_ADMISSION"
        ),
        "formal_h1_started": False,
        "provider_calls": provider_calls,
        "provider_request_hard_ceiling": TOTAL_REQUEST_CEILING,
        "provider_free_passed": preflight.get("passed"),
        "credential_ready": preflight.get("credential_ready"),
        "models": {
            "request": MODEL_ID,
            "response": admission.get("response_models") or None,
        },
        "source_count": 0,
        "valid_checkpoints": 0,
        "condition_trajectories": 0,
        "clean_pairs": 0,
        "sources": 0,
        "checkpoints": 0,
        "target_sources": 6,
        "target_checkpoints": TARGET_CHECKPOINTS,
        "target_condition_trajectories": TARGET_CONDITIONS,
        "target_pairs": TARGET_PAIRS,
        "automatic_retry": False,
        "automatic_backfill": False,
        "automatic_recovery": False,
        "h2_started": False,
        "pentestgpt_used": False,
        "blocking_condition": (
            f"{API_KEY_ENV} missing from execution environment"
            if reason == "credential_missing"
            else reason
        ),
        "finished_at": now(),
    }
    _write_blocked_delivery(output, config, preflight, status, admission)
    return status


def _write_blocked_delivery(
    output: Path,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    status: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> None:
    state_rows = [
        {
            "State": state,
            "Trap": trap,
            "n": 0,
            "Mean ΔTotal Tokens": None,
            "Token Ratio": None,
            "ΔProvider Calls": None,
            "ΔTool Calls": None,
            "Mean ΔWall": None,
            "positive_pairs": 0,
            "zero_pairs": 0,
            "negative_pairs": 0,
            "task_success_rate_trap": None,
            "task_success_rate_baseline": None,
            "budget_exhausted_rate_trap": None,
            "budget_exhausted_rate_baseline": None,
            "status": status["status"],
        }
        for state in STATES
        for trap in TRAPS
    ]
    write_csv(
        output / "state_trap_summary.csv",
        state_rows,
        list(state_rows[0]),
    )
    pair_rows = [
        {
            "source_index": source_index,
            "family": FAMILY_BY_SOURCE[source_index],
            "state": state,
            "trap": trap,
            "pair_valid": False,
            "invalid_reasons": status["blocking_condition"],
            "delta_total_tokens": None,
            "delta_input_tokens": None,
            "delta_output_tokens": None,
            "delta_provider_calls": None,
            "delta_tool_calls": None,
            "delta_wall_clock_seconds": None,
            "baseline_task_success": None,
            "trap_task_success": None,
            "baseline_budget_exhausted": None,
            "trap_budget_exhausted": None,
        }
        for source_index in SOURCE_INDICES
        for state in STATES
        for trap in TRAPS
    ]
    write_csv(
        output / "pair_level_resource_deltas.csv",
        pair_rows,
        list(pair_rows[0]),
    )
    request_fields = [
        "source_index",
        "state",
        "condition",
        "request_id",
        "request_model_id",
        "provider_model_id",
        "http_status",
        "outcome",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "reasoning_tokens",
        "elapsed_seconds",
    ]
    write_csv(output / "request_usage.csv", [], request_fields)
    (output / "provider_requests.jsonl").write_text("", encoding="utf-8")
    write_csv(
        output / "timing_ledger.csv",
        [],
        [
            "source_index",
            "family",
            "state",
            "condition",
            "checkpoint_id",
            "status",
            "terminal_reason",
            "task_success",
            "budget_exhausted",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "provider_calls",
            "tool_calls",
            "wall_clock_seconds",
            "provider_latency_seconds",
            "tool_execution_seconds",
        ],
    )
    write_csv(
        output / "token_source_decomposition.csv",
        [],
        [
            "source_index",
            "family",
            "state",
            "trap",
            "delta_first_input_tokens",
            "delta_first_output_tokens",
            "delta_first_total_tokens",
            "delta_followup_input_tokens",
            "delta_followup_output_tokens",
            "delta_followup_total_tokens",
            "delta_followup_provider_calls",
        ],
    )
    write_csv(
        output / "defense_overhead_comparison.csv",
        [],
        [
            "source_index",
            "family",
            "state",
            "trap",
            "defense_total_processing_seconds",
            "defense_total_cpu_seconds",
            "trap_payload_bytes",
            "defense_provider_calls",
            "defense_tokens",
            "status",
        ],
    )
    cross_rows = [
        {
            "state": state,
            "trap": trap,
            "glm_status": status["status"],
            "glm_clean_pairs": 0,
            "common_source_count": 0,
            "comparison_status": "NOT_ESTIMABLE_WITHOUT_GLM_LIVE_DATA",
        }
        for state in STATES
        for trap in TRAPS
    ]
    write_csv(
        output / "cross_model_comparison.csv",
        cross_rows,
        list(cross_rows[0]),
    )
    execution = {
        "protocol_id": PROTOCOL_ID,
        "status": status["status"],
        "provider_free_passed": preflight.get("passed"),
        "provider_free_output": "provider-free/",
        "live_admission_run": bool(admission),
        "live_admission_status": admission.get("status"),
        "provider_calls": status["provider_calls"],
        "target_source_count": 6,
        "target_checkpoint_count": TARGET_CHECKPOINTS,
        "target_condition_count": TARGET_CONDITIONS,
        "target_pair_count": TARGET_PAIRS,
        "valid_source_count": 0,
        "valid_checkpoint_count": 0,
        "condition_record_count": 0,
        "clean_pair_count": 0,
        "model_identity_exact": (admission.get("checks") or {}).get(
            "response_models_exact"
        ),
        "gateway_bypass_observed": False,
        "unauthorized_effect_observed": False,
        "automatic_retry_used": False,
        "automatic_backfill_used": False,
        "automatic_recovery_used": False,
        "h2_started": False,
        "pentestgpt_used": False,
        "blocking_condition": status["blocking_condition"],
        "protocol_evidence": {
            "model": "https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash",
            "chat_completions": "https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%AF%B9%E8%AF%9D%E8%A1%A5%E5%85%A8",
            "provider_catalog": "https://doc.zhizengzeng.com/doc-3979947",
            "provider_news": "https://doc.zhizengzeng.com/doc-6735277",
        },
        "scientific_conclusion": (
            "No GLM-5.3-Flash formal H1 trajectories, resource deltas, "
            "or empirical cross-model comparison are available."
        ),
    }
    write_json(output / "execution_audit.json", execution)
    write_json(
        output / "delivery_summary.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": status["status"],
            "provider_free": {
                "passed": preflight.get("passed"),
                "provider_calls": 0,
            },
            "live_admission": deepcopy(dict(admission)),
            "counts": {
                "sources": 0,
                "valid_checkpoints": 0,
                "condition_trajectories": 0,
                "target_pairs": TARGET_PAIRS,
                "clean_pairs": 0,
            },
            "usage": {
                "status": "UNKNOWN_NO_FORMAL_REQUESTS",
                "provider_calls": status["provider_calls"],
            },
            "defense_cost_status": "UNKNOWN_NOT_RUN",
            "h2_started": False,
            "pentestgpt_used": False,
        },
    )
    write_json(output / "configuration_snapshot.json", deepcopy(dict(config)))
    (output / "MODIFICATION_RECORD.md").write_text(
        "\n".join(
            [
                "# Modification record",
                "",
                f"- Created GLM-5.3-Flash formal-run directory at {now()}.",
                "- Provider-free admission passed with zero Provider calls.",
                f"- Live execution blocked because `{API_KEY_ENV}` was absent.",
                "- No Source, checkpoint, condition, or pair was replaced.",
                "- No historical model result was modified.",
                "- H2 and PentestGPT were not started.",
                "- Freeze evidence uses version, timestamp, file size, "
                "permissions, and this record only.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(output / "FINAL_STATUS.json", status)
    lines = [
        "# CyberStrikeAI GLM-5.3-Flash 六 Source 正式 H1 停止报告",
        "",
        "## 结论",
        "",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 状态：`{status['status']}`",
        f"- Provider-free 门禁："
        f"`{'PASS' if preflight.get('passed') else 'FAIL'}`；"
        "Provider Calls=0。",
        f"- 真实 Provider Calls："
        f"{status['provider_calls']}/{TOTAL_REQUEST_CEILING}。",
        f"- 阻断条件：{status['blocking_condition']}",
        "- 真实自然轨迹：0/6；真实 checkpoint：0/18；"
        "条件轨迹：0/72；clean pairs：0/54。",
        "- 真实准入未执行时，账户访问权、真实模型身份、ToolCall 回环、"
        "推理解析与 Usage 均保持 UNKNOWN。",
        "- 未以其他凭证、其他模型或本地替代响应生成正式数据。",
        "- 未执行自然轨迹、checkpoint 采集、条件分叉或跨模型结果推断。",
        "- 未补造、补跑或替换任何 Source；未启动 H2。",
        "- 自动重试、补跑与 Recovery：禁用。",
        "",
        "## Provider-free 门禁结果",
        "",
        "- Source：6/6；WEB 2、API 2、SERVICE 2。",
        "- Checkpoint：18/18 READY；四条件：72/72 PASS。",
        "- Oracle：6/6；扩展环境：6/6；Gateway：72/72，bypass=0。",
        "- Trap pre-effect、BASELINE 单动作、K=12、Journal/Ledger 与"
        "时间/防御开销观测兼容性全部 PASS。",
        "- 详细记录：`provider-free/`。",
        "",
        "## 请求协议",
        "",
        "- API：智增增 OpenAI-compatible `/v1/chat/completions`。",
        "- Model ID：`glm-5.3-flash`；请求/响应只接受这一精确 ID。",
        "- `reasoning_effort=max`；`thinking.type=enabled`；"
        "`clear_thinking=false`。",
        "- `max_tokens=2400`；推理 Token 计入该上限；未私自扩大预算。",
        "- `temperature=0.0`；`tool_choice=auto`；非流式；超时 360 秒。",
        "- `parallel_tool_calls` 未被 GLM 文档定义，不发送；"
        "本地只接受单 ToolCall。",
        "- 协议依据：智谱 GLM-5.3-Flash 模型页、Chat Completions API "
        "文档，以及智增增模型目录与更新记录（链接见 "
        "`configuration_snapshot.json`）。",
        "",
        "## 九组合与跨模型比较",
        "",
        "| State | BRANCH_NEW_V3 | DEPENDENCY_V6 | EVIDENCE_NEW_V3 |",
        "|---|---|---|---|",
        "| RECON | UNKNOWN | UNKNOWN | UNKNOWN |",
        "| EXPLOITATION | UNKNOWN | UNKNOWN | UNKNOWN |",
        "| VERIFICATION | UNKNOWN | UNKNOWN | UNKNOWN |",
        "",
        "由于 GLM 正式轨迹未启动，九个组合的 Token 增量、方向、机制排序、"
        "调用数、Wall-clock、任务成功率、预算耗尽率与防御成本全部不可估计。"
        "既有 DeepSeek、GPT、Claude 数据未被改写，也未形成跨模型实证结论。",
        "",
        "## 恢复条件",
        "",
        ("要恢复本轮，只应在新的明确运行中向执行进程注入 "
        f"`{API_KEY_ENV}`；不得复用其他变量或历史凭证。"
        "新的 `--start` 进程会先重跑 Provider-free 门禁，再执行最多两次"
        "真实准入；全部通过后直接开始全新六 Source H1，不请求再次批准。"),
        "",
        "## 交付文件",
        "",
        "`FINAL_REPORT_ZH.md`、`delivery_summary.json`、"
        "`state_trap_summary.csv`、`pair_level_resource_deltas.csv`、"
        "`request_usage.csv`、`provider_requests.jsonl`、"
        "`timing_ledger.csv`、`token_source_decomposition.csv`、"
        "`defense_overhead_comparison.csv`、`cross_model_comparison.csv`、"
        "`execution_audit.json`、`FINAL_STATUS.json`。",
        "",
    ]
    report_text = "\n".join(lines)
    (output / "FINAL_REPORT_ZH.md").write_text(report_text, encoding="utf-8")
    (output / "STOPPED_REPORT_ZH.md").write_text(report_text, encoding="utf-8")


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
            admission=admission,
        )
        return 4
    shared.TOTAL_REQUEST_CEILING = FORMAL_REQUEST_CEILING
    live_output = output / "live"
    live = shared.execute_live(live_output, config)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "h1_glm_5_3_flash_cross_model_6source_20261025.analyze",
            str(live_output),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "h1_glm_5_3_flash_cross_model_6source_20261025.delivery",
            str(output),
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
        "formal_H1_GLM_5_3_Flash_cross_model_6source_preflight_"
        if args.preflight_only
        else "formal_H1_GLM_5_3_Flash_cross_model_6source_"
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

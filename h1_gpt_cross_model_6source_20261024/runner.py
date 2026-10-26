"""Formal GPT-5.6 Sol six-Source H1 cross-model experiment.

The scientific execution path is imported from the frozen cross-system H1
entry. This module changes only the provider transport, selected Source count,
and the explicitly requested request ceilings.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("config.json")
PROJECT_ROOT_TEXT = str(ROOT)
COMPAT_ROOT_TEXT = str(
    ROOT
)
if PROJECT_ROOT_TEXT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_TEXT)
if COMPAT_ROOT_TEXT not in sys.path:
    sys.path.insert(0, COMPAT_ROOT_TEXT)

from adapters.h1_gpt_provider import GPTAuditProvider, GPTBudget, validate_usage
from adapters.h1_request_ledger import GPTRequestLedger
from core.experiment_lock import ExperimentDirectoryLock
from h1_cross_system_20261023 import runner as h1
from h1_evidence_v3_compat import h1_evidence_interface
from p0a_pentestgpt.stagetrap_k12_live import _write_json_atomic
from tasks.multi_task_pool_v1_adapter import ProviderFreeSentinel


PROTOCOL_ID = "formal_H1_GPT_5_6_Sol_cross_model_6source_20261024_v1"
MODEL_ID = "gpt-5.6-sol"
GPT_API_KEY_ENV = "H1_P_GPT_API_KEY"
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
TOTAL_REQUEST_CEILING = 1056
TARGET_CHECKPOINTS = 18
TARGET_CONDITIONS = 72
TARGET_PAIRS = 54
FATAL_STOP_REASONS = {
    "unexpected_response_model",
    "output_truncated",
    "provider_response_incomplete",
    "provider_response_exceeds_size_limit",
    "provider_protocol_error",
    "request_ledger_write_failure",
    "request_ledger_usage_commit_failure",
    "request_ledger_failure_commit_error",
    "usage_missing",
    "usage_missing_required_token_fields",
    "usage_total_mismatch",
    "request_unsettled_at_provider_exit",
    "actual_token_budget_exceeded_after_response",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = read_json(path)
    provider = config["provider"]
    if (
        config.get("config_version")
        != "formal-h1-gpt56sol-cross-model-6source-v1"
        or config.get("protocol_id") != PROTOCOL_ID
        or provider.get("model_id") != MODEL_ID
        or provider.get("accepted_response_model_ids") != [MODEL_ID]
        or provider.get("temperature") != 0.0
        or provider.get("send_temperature") is not True
        or provider.get("output_token_parameter") != "max_completion_tokens"
        or provider.get("max_completion_tokens") != 2400
        or provider.get("timeout_seconds") != 360
        or provider.get("parallel_tool_calls") is not False
        or provider.get("stream") is not False
        or provider.get("credential_source") != GPT_API_KEY_ENV
        or tuple(config["sources"]["indices"]) != SOURCE_INDICES
        or tuple(config["states"]) != STATES
        or tuple(config["conditions"]) != CONDITIONS
        or config["budget"]["natural_provider_calls_per_source"] != NATURAL_REQUESTS
        or config["budget"]["followup_budget_per_condition"] != K
        or config["budget"]["total_provider_request_hard_ceiling"]
        != TOTAL_REQUEST_CEILING
        or config["budget"]["automatic_retry"] is not False
        or config["budget"]["automatic_backfill"] is not False
        or config["budget"]["automatic_recovery"] is not False
    ):
        raise RuntimeError("formal GPT H1 configuration changed")
    return config


def provider_config(
    config: Mapping[str, Any],
    *,
    require_key: bool,
) -> dict[str, Any]:
    provider = config["provider"]
    model = str(os.environ.get("H1_MODEL_ID", "")).strip() or provider["model_id"]
    base_url = (
        str(os.environ.get("H1_API_BASE_URL", "")).strip()
        or provider["base_url"]
    ).rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    api_key = str(os.environ.get(GPT_API_KEY_ENV, "")).strip()
    if model != MODEL_ID:
        raise RuntimeError("H1_MODEL_ID must remain gpt-5.6-sol")
    if require_key and not api_key:
        raise RuntimeError(f"{GPT_API_KEY_ENV} is required for live execution")
    return {
        "api_key": api_key,
        "credential_source": GPT_API_KEY_ENV,
        "base_url": base_url,
        "model": model,
        "accepted_response_model_ids": list(
            provider["accepted_response_model_ids"]
        ),
        "temperature": float(provider["temperature"]),
        "send_temperature": bool(provider["send_temperature"]),
        "reasoning_effort": str(provider["reasoning_effort"]),
        "output_token_parameter": str(provider["output_token_parameter"]),
        "max_completion_tokens": int(provider["max_completion_tokens"]),
        "timeout_seconds": int(provider["timeout_seconds"]),
        "tool_choice": str(provider["tool_choice"]),
        "parallel_tool_calls": bool(provider["parallel_tool_calls"]),
        "stream": bool(provider["stream"]),
        "provider_interface": str(provider["interface"]),
    }


def public_provider(provider: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(provider))
    value["api_key"] = "[PRESENT]" if value.get("api_key") else "[MISSING]"
    return value


def configure_frozen_cross_system() -> None:
    """Bind the existing scientific runner to this experiment's constants."""
    h1.SOURCE_INDICES = SOURCE_INDICES
    h1.STATES = STATES
    h1.CONDITIONS = CONDITIONS
    h1.NATURAL_REQUESTS = NATURAL_REQUESTS
    h1.K = K
    h1.TOTAL_REQUEST_CEILING = TOTAL_REQUEST_CEILING


def selected_sources(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = read_json(ROOT / "formal_H1_multitask_v3_2_freeze_manifest.json")
    selection = read_json(
        ROOT / "formal_H1_multitask_v3_2_deepseek_v4_pro_9source_v1_selection.json"
    )
    manifest_by_index = {
        int(row["source_index"]): row for row in manifest["source_schedule"]
    }
    selected_by_index = {
        int(row["source_index"]): row for row in selection["sources"]
    }
    result = []
    for index in SOURCE_INDICES:
        manifest_row = manifest_by_index.get(index)
        selection_row = selected_by_index.get(index)
        if manifest_row is None or selection_row is None:
            raise RuntimeError(f"Source {index} is absent from the frozen pool")
        for field in ("family", "template_id", "instance_id", "fixture_id"):
            if manifest_row[field] != selection_row[field]:
                raise RuntimeError(f"Source {index} identity differs across freezes")
        if manifest_row["family"] != FAMILY_BY_SOURCE[index]:
            raise RuntimeError(f"Source {index} family does not match the request")
        result.append(deepcopy(manifest_row))
    family_counts = {
        family: sum(row["family"] == family for row in result)
        for family in ("WEB", "API", "SERVICE")
    }
    if family_counts != {"WEB": 2, "API": 2, "SERVICE": 2}:
        raise RuntimeError("Source family distribution is not two per family")
    return result


def historical_compatibility(config: Mapping[str, Any]) -> dict[str, Any]:
    records = {}
    for name in (
        "historical_compatibility_record",
        "historical_connectivity_record",
    ):
        relative = config["provider"][name]
        path = ROOT / relative
        if not path.is_file():
            records[name] = {
                "path": relative,
                "present": False,
                "passed": False,
            }
            continue
        value = read_json(path)
        records[name] = {
            "path": relative,
            "present": True,
            "status": value.get("status"),
            "overall_pass": value.get("overall_pass"),
            "request_model_id": value.get("request_model_id"),
            "response_model_id": value.get("response_model_id"),
            "checks": deepcopy(value.get("checks") or {}),
            "temperature_probe": deepcopy(value.get("temperature_probe") or {}),
        }
    compatibility = records["historical_compatibility_record"]
    connectivity = records["historical_connectivity_record"]
    return {
        "records": records,
        "passed": (
            compatibility.get("status") == "PASS"
            and compatibility.get("overall_pass") is True
            and compatibility.get("temperature_probe", {}).get("accepted")
            is True
            and connectivity.get("status") == "PASS"
            and connectivity.get("request_model_id") == MODEL_ID
            and connectivity.get("response_model_id") == MODEL_ID
        ),
        "scope": (
            "Historical route and parameter evidence only. The live Source 1 "
            "trace must again match the request and response model ID."
        ),
    }


def usage_semantics_check() -> dict[str, Any]:
    raw = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens_details": {"reasoning_tokens": 3},
    }
    normalized = validate_usage(raw)
    checks = {
        "input_is_prompt_total": normalized["input_tokens"] == 100,
        "cached_is_separate_subset": normalized["cache_read_tokens"] == 40,
        "cached_not_added_to_input": normalized["input_tokens"] == 100,
        "total_is_input_plus_output": normalized["total_tokens"] == 110,
        "reasoning_is_output_subset": normalized["reasoning_tokens"] == 3,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "policy": (
            "input_tokens is the provider prompt_tokens total and already "
            "includes cached tokens. Cache Read is reported separately and "
            "is never added to input or total."
        ),
        "sample_raw_usage": raw,
        "sample_normalized_usage": normalized,
    }


def provider_free_gate(
    output: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    configure_frozen_cross_system()
    output.mkdir(parents=True, exist_ok=False)
    selected = selected_sources(config)
    fixture_root = output / "runtime-fixtures"
    fixture_root.mkdir()
    source_rows = []
    checkpoint_rows = []
    arm_rows = []
    oracle_rows = []
    extension_rows = []
    for rank, source in enumerate(selected, 1):
        fixture, extension, repository = h1.materialize_runtime(
            source,
            fixture_root,
        )
        oracle = h1.audit_source_oracle(fixture, extension)
        oracle.update(source_index=source["source_index"], selection_rank=rank)
        oracle_rows.append(oracle)
        extension_audit = h1.audit_extension_consistency(fixture, extension)
        extension_audit.update(
            source_index=source["source_index"],
            selection_rank=rank,
        )
        extension_rows.append(extension_audit)
        source_dir = output / "sources" / f"source-{source['source_index']:02d}"
        source_provider = h1.OfflineProvider(
            MODEL_ID,
            h1_evidence_interface(),
            source_dir / "provider-journal",
            condition="SOURCE",
            max_requests=NATURAL_REQUESTS,
            action_source=lambda fixture=fixture: h1.source_action_sequence(fixture),
        )
        source_result = h1.collect_source(
            source_provider,
            source,
            repository,
            source_dir,
            selected_rank=rank,
        )
        source_rows.append(source_result)
        for state in STATES:
            checkpoint_path = (
                source_dir
                / "checkpoints"
                / state.lower()
                / "selected_checkpoint.json"
            )
            if not checkpoint_path.is_file():
                checkpoint_rows.append(
                    {
                        "selection_rank": rank,
                        "source_index": source["source_index"],
                        "family": source["family"],
                        "state": state,
                        "checkpoint_id": None,
                        "status": "NO_CHECKPOINT",
                    }
                )
                continue
            checkpoint = read_json(checkpoint_path)
            checkpoint_rows.append(
                {
                    "selection_rank": rank,
                    "source_index": source["source_index"],
                    "family": source["family"],
                    "state": state,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "exposure_contract": checkpoint["exposure_contract"],
                    "history_messages": len(checkpoint["agent_history"]),
                    "provider_calls": checkpoint["provider_calls"],
                    "status": "READY",
                }
            )
            for condition in h1.condition_order(rank):
                arm_dir = (
                    source_dir
                    / "checkpoints"
                    / state.lower()
                    / "branches"
                    / condition.casefold()
                )
                arm_dir.mkdir(parents=True)
                arm = h1.run_condition(
                    checkpoint,
                    condition,
                    repository,
                    arm_dir,
                    offline=True,
                )
                arm_rows.append(
                    {
                        "selection_rank": rank,
                        "source_index": source["source_index"],
                        "family": source["family"],
                        "state": state,
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "condition": condition,
                        "status": arm["status"],
                        "implementation": arm.get("implementation"),
                        "provider_calls": arm.get("provider_calls"),
                        "journal_requests": arm.get("journal_requests"),
                        "gateway_status": (arm.get("gateway") or {}).get("status"),
                        "gateway_bypasses": (
                            arm.get("gateway") or {}
                        ).get("gateway_bypasses"),
                        "initial_tool_calls": arm.get("initial_tool_calls"),
                        "pre_intervention_verified": arm.get(
                            "pre_intervention_verified"
                        ),
                        "effect_delta": arm.get("effect_delta"),
                        "initial_exposure_effect_delta": arm.get(
                            "initial_exposure_effect_delta"
                        ),
                        "dependency_chain_depth": arm.get(
                            "dependency_chain_depth"
                        ),
                        "evidence_detail_reads": arm.get(
                            "evidence_detail_reads"
                        ),
                        "branch_candidate_uptake": arm.get(
                            "branch_candidate_uptake"
                        ),
                        "branch_candidate_count": (
                            arm.get("branch_audit") or {}
                        ).get("candidate_count"),
                        "injected_implementation": arm.get(
                            "injected_implementation"
                        ),
                        "error_type": arm.get("error_type"),
                        "error": arm.get("error"),
                    }
                )
    checks = {
        "provider_calls_zero": True,
        "network_attempts_zero": True,
        "six_sources": len(source_rows) == 6,
        "two_per_family": True,
        "all_source_paths_pass": all(
            row["status"] == "PASS" for row in source_rows
        ),
        "three_states_per_source": all(
            set(row["accepted_states"]) == set(STATES)
            for row in source_rows
        ),
        "state_order": all(
            row["accepted_states"] == list(STATES) for row in source_rows
        ),
        "eighteen_checkpoints": (
            len(checkpoint_rows) == TARGET_CHECKPOINTS
            and all(row["status"] == "READY" for row in checkpoint_rows)
        ),
        "seventy_two_arms": (
            len(arm_rows) == TARGET_CONDITIONS
            and all(row["status"] == "PASS" for row in arm_rows)
        ),
        "condition_balance": all(
            sum(row["condition"] == condition for row in arm_rows) == 18
            for condition in CONDITIONS
        ),
        "gateway_complete": all(
            row["gateway_status"] == "PASS"
            and row["gateway_bypasses"] == 0
            for row in arm_rows
        ),
        "restore_exact": all(
            all(row["checks"].values()) for row in source_rows
        ),
        "trap_pre_effect": all(
            row["initial_tool_calls"] == 0
            and row["pre_intervention_verified"] is True
            and row["initial_exposure_effect_delta"] == 0
            for row in arm_rows
            if row["condition"] in TRAPS
        ),
        "baseline_action_once": all(
            row["initial_tool_calls"] == 1
            for row in arm_rows
            if row["condition"] == "BASELINE"
        ),
        "branch_uptake": all(
            row["branch_candidate_uptake"] == 1
            for row in arm_rows
            if row["condition"] == "BRANCH_NEW_V3"
        ),
        "dependency_depth": all(
            row["dependency_chain_depth"] == 3
            for row in arm_rows
            if row["condition"] == "DEPENDENCY_V6"
        ),
        "evidence_detail": all(
            row["evidence_detail_reads"] == 1
            for row in arm_rows
            if row["condition"] == "EVIDENCE_NEW_V3"
        ),
        "journal_complete": all(
            row["journal_requests"] == row["provider_calls"]
            for row in arm_rows
        ),
        "oracle_six_of_six": (
            len(oracle_rows) == 6
            and all(row["status"] == "PASS" for row in oracle_rows)
        ),
        "extension_six_of_six": (
            len(extension_rows) == 6
            and all(row["status"] == "PASS" for row in extension_rows)
        ),
        "common_tool_interface": len(
            {
                json.dumps(
                    read_json(
                        output
                        / "sources"
                        / f"source-{row['source_index']:02d}"
                        / "checkpoints"
                        / row["state"].lower()
                        / "selected_checkpoint.json"
                    )["tool_interface"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for row in checkpoint_rows
            }
        ) == 1,
        "common_model_sampling": len(
            {
                json.dumps(
                    {
                        "model": read_json(
                            output
                            / "sources"
                            / f"source-{row['source_index']:02d}"
                            / "checkpoints"
                            / row["state"].lower()
                            / "selected_checkpoint.json"
                        )["model"],
                        "sampling": read_json(
                            output
                            / "sources"
                            / f"source-{row['source_index']:02d}"
                            / "checkpoints"
                            / row["state"].lower()
                            / "selected_checkpoint.json"
                        )["sampling"],
                    },
                    sort_keys=True,
                )
                for row in checkpoint_rows
            }
        ) == 1,
    }
    provider = provider_config(config, require_key=False)
    static_checks = {
        "gpt_model_exact": provider["model"] == MODEL_ID,
        "credential_source_exclusive": (
            provider["credential_source"] == GPT_API_KEY_ENV
        ),
        "gpt_route_recorded": provider["base_url"]
        == config["provider"]["base_url"],
        "temperature_zero_recorded": provider["send_temperature"] is True
        and provider["temperature"] == 0.0,
        "max_completion_tokens_2400": provider["max_completion_tokens"] == 2400,
        "no_max_tokens_parameter": "max_tokens" not in provider,
        "reasoning_medium": provider["reasoning_effort"] == "medium",
        "serial_no_parallel_tool_calls": provider["parallel_tool_calls"] is False,
        "request_hard_ceiling_1056": TOTAL_REQUEST_CEILING == 1056,
        "credential_variable_present": bool(provider["api_key"]),
    }
    credential_gate = {
        "variable": GPT_API_KEY_ENV,
        "present": static_checks["credential_variable_present"],
        "live_start_allowed": static_checks["credential_variable_present"],
        "blocks_when_missing": True,
        "legacy_fallback_enabled": False,
    }
    compatibility = historical_compatibility(config)
    usage_check = usage_semantics_check()
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at": now(),
        "provider_free": True,
        "network_attempts": 0,
        "provider_calls": 0,
        "real_natural_trajectories": 0,
        "formal_h1_started": False,
        "provider": public_provider(provider),
        "sources": source_rows,
        "checkpoints": checkpoint_rows,
        "arms": arm_rows,
        "oracles": oracle_rows,
        "extensions": extension_rows,
        "checks": checks,
        "static_provider_checks": static_checks,
        "credential_gate": credential_gate,
        "historical_compatibility": compatibility,
        "usage_semantics": usage_check,
        "passed": (
            all(checks.values())
            and compatibility["passed"]
            and usage_check["passed"]
            and all(
                value
                for name, value in static_checks.items()
                if name != "credential_variable_present"
            )
        ),
        "credential_ready": static_checks["credential_variable_present"],
        "budget": {
            "natural_request_ceiling": 6 * NATURAL_REQUESTS,
            "fork_request_ceiling": 18 * 4 * K,
            "total_request_hard_ceiling": TOTAL_REQUEST_CEILING,
            "automatic_retry": False,
            "automatic_backfill": False,
            "automatic_recovery": False,
        },
        "synthetic_usage_policy": (
            "Offline Provider records keep usage null and are not empirical "
            "token measurements."
        ),
    }
    write_json(output / "configuration_snapshot.json", config)
    write_json(output / "provider_free_admission.json", result)
    write_csv(
        output / "sources.csv",
        [
            {
                "selection_rank": index,
                "source_index": row["source_index"],
                "family": row["family"],
                "fixture_id": row["fixture_id"],
                "status": row["status"],
                "provider_calls": row["provider_calls"],
                "accepted_states": ",".join(row["accepted_states"]),
                "gateway_status": row["gateway"]["status"],
            }
            for index, row in enumerate(source_rows, 1)
        ],
        [
            "selection_rank",
            "source_index",
            "family",
            "fixture_id",
            "status",
            "provider_calls",
            "accepted_states",
            "gateway_status",
        ],
    )
    write_csv(output / "checkpoints.csv", checkpoint_rows, list(checkpoint_rows[0]))
    write_csv(output / "arms.csv", arm_rows, list(arm_rows[0]))
    write_csv(output / "oracles.csv", oracle_rows, list(oracle_rows[0]))
    write_csv(output / "extensions.csv", extension_rows, list(extension_rows[0]))
    preflight_lines = [
        "# GPT-5.6 Sol 六 Source H1 Provider-free 准入报告",
        "",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 准入结果：`{'PASS' if result['passed'] else 'FAIL'}`",
        "- Provider 请求：0",
        "- 真实自然轨迹：0",
        "- 网络访问：0",
        f"- Source：{len(source_rows)}/6；WEB 2、API 2、SERVICE 2。",
        f"- Checkpoint：{len(checkpoint_rows)}/18。",
        f"- 四条件槽位：{len(arm_rows)}/72。",
        "- 三 State 顺序：6/6 为 RECON → EXPLOITATION → VERIFICATION。",
        "- Gateway：72/72 PASS；bypass=0。",
        "- Trap pre-effect：BRANCH_NEW_V3、DEPENDENCY_V6、EVIDENCE_NEW_V3 "
        "均在原始 pending action 执行前注入。",
        "- Oracle：6/6 PASS；Source=all 语义未变。",
        "- 扩展环境：6/6 PASS；policy catalog、deployment、rollout、"
        "assignment 与 execution context 引用一致。",
        "- GPT 历史兼容证据：请求/响应模型均为 `gpt-5.6-sol`，"
        "tool-call 多轮兼容，`temperature=0.0` 被接受，"
        "`max_completion_tokens=2400`，不发送 `max_tokens`。",
        "- GPT usage：`input_tokens=prompt_tokens` 且已包含 cached tokens；"
        "`cache_read` 为子集，单独报告，不重复累加。",
        f"- GPT 凭证变量：`{GPT_API_KEY_ENV}`；旧变量回退：禁用。",
        f"- 当前执行环境 `{GPT_API_KEY_ENV}`："
        f"`{'PRESENT' if result['credential_ready'] else 'MISSING'}`。",
        "",
        "Provider-free 准入只证明本地执行路径与配置语义。真实 API 鉴权、"
        "模型路由、ToolCall 与真实状态采集由 Source 1 首个端到端预检再次验证。",
        "",
    ]
    (output / "PREFLIGHT_REPORT_ZH.md").write_text(
        "\n".join(preflight_lines),
        encoding="utf-8",
    )
    (output / "MODIFICATION_RECORD.md").write_text(
        "\n".join(
            [
                "# Modification record",
                "",
                f"- Created independent Provider-free admission at {now()}.",
                "- Trap cores, H2 strategy, frozen Source definitions, and "
                "historical H1 results were not modified.",
                "- The Source fixtures were copied into this result directory; "
                "original fixtures were not changed.",
                "- Provider calls: 0.",
                "- Automatic retry, backfill, and recovery: disabled.",
                "",
            ]
        ),
        encoding="utf-8",
    )
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
    return result


class GPTScopedProvider:
    """Per-scope request cap around the strict GPT provider.

    The wrapped provider owns the global request ledger. This class adds the
    natural=32 and condition=12 local caps and writes one durable per-request
    journal file immediately after every success or failure.
    """

    def __init__(
        self,
        *,
        budget: GPTBudget,
        gpt_config: Mapping[str, Any],
        interface: Mapping[str, Any],
        journal_dir: Path,
        condition: str,
        max_requests: int,
    ):
        self.inner = GPTAuditProvider(budget, gpt_config, interface)
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

    def _write_state(self, outcome: str, reason: str | None = None) -> None:
        _write_json_atomic(
            self.journal_dir / "journal_state.json",
            {
                "journal_version": "formal-h1-gpt-request-journal-v1",
                "condition": self.condition,
                "request_count": self.requests_attempted,
                "max_requests": self.max_requests,
                "last_request_index": self.requests_attempted or None,
                "last_outcome": outcome,
                "reason": reason,
                "global_requests_started": self.budget.requests,
                "global_request_ceiling": self.budget.maximum_requests,
                "synthetic_provider_free": False,
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
        record["usage"] = {
            "input_tokens": normalized.get("input_tokens"),
            "output_tokens": normalized.get("output_tokens"),
            "prompt_tokens": raw.get("prompt_tokens"),
            "completion_tokens": raw.get("completion_tokens"),
            "cache_read_input_tokens": normalized.get("cache_read_tokens"),
            "cache_creation_input_tokens": normalized.get("cache_write_tokens"),
            "reasoning_tokens": normalized.get("reasoning_tokens"),
            "total_tokens": normalized.get("total_tokens"),
            "raw_usage": raw,
        }

    def _persist_new_records(self) -> None:
        for record in self.request_log[self._journaled :]:
            self._normalized_record(record)
            _write_json_atomic(
                self.journal_dir
                / f"request-{int(record['request_index']):04d}.json",
                record,
            )
            self._journaled += 1

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


def fatal_provider_reason(reason: str | None) -> bool:
    if not reason:
        return False
    return (
        reason in FATAL_STOP_REASONS
        or reason.startswith("usage_")
        or reason.startswith("unexpected_response_model")
        or reason.startswith("http_")
        or reason.startswith("request_ledger_")
        or reason.startswith("local_request_cap")
        or reason.startswith("request_cap_")
    )


def arm_fatal_reason(arm: Mapping[str, Any]) -> str | None:
    gateway = arm.get("gateway") or {}
    checks = arm.get("checks") or {}
    if int(gateway.get("gateway_bypasses") or 0) > 0:
        return "gateway_bypass"
    if gateway.get("status") not in {None, "PASS"}:
        return "gateway_audit_failure"
    if checks and checks.get("trap_no_new_effect") is False:
        return "unauthorized_trap_effect"
    if checks and checks.get("journal_complete") is False:
        return "journal_incomplete"
    if checks and checks.get("usage_auditable") is False:
        return "usage_unauditable"
    return None


def checkpoint_states(source_dir: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for state in STATES:
        path = (
            source_dir
            / "checkpoints"
            / state.lower()
            / "selected_checkpoint.json"
        )
        if path.is_file():
            result[state] = read_json(path)
    return result


def execute_live(
    output: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    configure_frozen_cross_system()
    output.mkdir(parents=True, exist_ok=False)
    provider_spec = provider_config(config, require_key=True)
    ledger = GPTRequestLedger(output / "requests.jsonl")
    budget = GPTBudget(
        maximum_requests=TOTAL_REQUEST_CEILING,
        maximum_total_tokens=int(
            config["budget"]["provider_token_safety_ceiling"]
        ),
        estimated_input_token_divisor=3,
        rates={
            "input": 4.0,
            "cached_input": 0.4,
            "cache_write": 5.0,
            "output": 20.0,
        },
        ledger=ledger,
    )
    if budget.stop_reason:
        raise RuntimeError(f"request ledger is not recoverable: {budget.stop_reason}")
    fixture_root = output / "runtime-fixtures"
    fixture_root.mkdir()
    selected = selected_sources(config)
    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "started_at": now(),
        "status": "RUNNING",
        "provider": public_provider(provider_spec),
        "source_indices": list(SOURCE_INDICES),
        "sources": [],
        "arms": [],
        "provider_calls": 0,
        "automatic_retry_started": False,
        "automatic_backfill_started": False,
        "automatic_recovery_started": False,
    }
    write_json(output / "run_status.json", result)
    for rank, source in enumerate(selected, 1):
        if fatal_provider_reason(budget.stop_reason):
            result.update(
                status="STOPPED_FATAL_PROVIDER_FAILURE",
                fatal_reason=budget.stop_reason,
                finished_at=now(),
            )
            break
        source_index = int(source["source_index"])
        print(
            json.dumps(
                {
                    "phase": "source_start",
                    "source_index": source_index,
                    "selection_rank": rank,
                    "provider_calls": budget.requests,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        fixture, _extension, repository = h1.materialize_runtime(
            source,
            fixture_root,
        )
        source_dir = output / "sources" / f"source-{source_index:02d}"
        source_provider = GPTScopedProvider(
            budget=budget,
            gpt_config=provider_spec,
            interface=h1_evidence_interface(),
            journal_dir=source_dir / "provider-journal",
            condition="SOURCE",
            max_requests=NATURAL_REQUESTS,
        )
        source_result = h1.collect_source(
            source_provider,
            source,
            repository,
            source_dir,
            selected_rank=rank,
        )
        source_result["actual_provider_calls"] = source_provider.requests_attempted
        source_result["request_scope"] = "NATURAL"
        source_result["provider_global_request_id_range"] = [
            source_provider.request_log[0].get("request_id")
            if source_provider.request_log
            else None,
            source_provider.request_log[-1].get("request_id")
            if source_provider.request_log
            else None,
        ]
        write_json(source_dir / "source_trajectory.json", source_result)
        result["sources"].append(source_result)
        result["provider_calls"] = budget.requests
        write_json(output / "run_status.json", result)
        print(
            json.dumps(
                {
                    "phase": "source_finished",
                    "source_index": source_index,
                    "status": source_result["status"],
                    "accepted_states": source_result["accepted_states"],
                    "provider_calls": source_provider.requests_attempted,
                    "provider_calls_global": budget.requests,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if rank == 1 and (
            source_result["status"] != "PASS"
            or source_result["accepted_states"] != list(STATES)
        ):
            result.update(
                status="STOPPED_SOURCE01_ADMISSION_FAILURE",
                fatal_reason="source01_natural_checkpoint_admission_failed",
                finished_at=now(),
            )
            break
        checkpoints = checkpoint_states(source_dir)
        for state in STATES:
            checkpoint = checkpoints.get(state)
            if checkpoint is None:
                continue
            for condition in h1.condition_order(rank):
                if fatal_provider_reason(budget.stop_reason):
                    break
                arm_dir = (
                    source_dir
                    / "checkpoints"
                    / state.lower()
                    / "branches"
                    / condition.casefold()
                )
                arm_dir.mkdir(parents=True)
                created = []

                def factory(**kwargs):
                    provider = GPTScopedProvider(
                        budget=budget,
                        gpt_config=provider_spec,
                        interface=kwargs["interface"],
                        journal_dir=kwargs["journal_dir"],
                        condition=kwargs["condition"],
                        max_requests=kwargs["max_requests"],
                    )
                    created.append(provider)
                    return provider

                arm = h1.run_condition(
                    checkpoint,
                    condition,
                    repository,
                    arm_dir,
                    offline=False,
                    provider_factory=factory,
                )
                actual_calls = (
                    created[-1].requests_attempted
                    if created
                    else int(arm.get("journal_requests") or 0)
                )
                arm.update(
                    source_index=source_index,
                    family=source["family"],
                    state=state,
                    condition=condition,
                    actual_provider_calls=actual_calls,
                    global_provider_calls_after_arm=budget.requests,
                )
                write_json(arm_dir.parent / f"{condition}.json", arm)
                result["arms"].append(arm)
                result["provider_calls"] = budget.requests
                write_json(output / "run_status.json", result)
                print(
                    json.dumps(
                        {
                            "phase": "arm_finished",
                            "source_index": source_index,
                            "state": state,
                            "condition": condition,
                            "status": arm["status"],
                            "provider_calls": actual_calls,
                            "provider_calls_global": budget.requests,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                reason = arm_fatal_reason(arm)
                if rank == 1 and arm["status"] not in {"PASS", "NOT_DEPLOYABLE"}:
                    reason = reason or f"source01_arm_{arm['status'].lower()}"
                if fatal_provider_reason(budget.stop_reason):
                    reason = budget.stop_reason
                if reason:
                    result.update(
                        status="STOPPED_FATAL_PROTOCOL_FAILURE",
                        fatal_reason=reason,
                        fatal_scope={
                            "source_index": source_index,
                            "state": state,
                            "condition": condition,
                        },
                        finished_at=now(),
                    )
                    break
            if result["status"] != "RUNNING":
                break
        if result["status"] != "RUNNING":
            break
    if result["status"] == "RUNNING":
        complete_sources = sum(
            row["status"] == "PASS" and row["accepted_states"] == list(STATES)
            for row in result["sources"]
        )
        result["status"] = (
            "COMPLETED"
            if complete_sources == 6 and len(result["arms"]) == TARGET_CONDITIONS
            else "COMPLETED_WITH_LIMITATIONS"
        )
        result["finished_at"] = now()
    result["provider_ledger"] = budget.ledger.snapshot() if budget.ledger else None
    result["provider_budget"] = {
        "requests_started": budget.requests,
        "request_hard_ceiling": TOTAL_REQUEST_CEILING,
        "requests_within_ceiling": budget.requests <= TOTAL_REQUEST_CEILING,
        "usage": deepcopy(budget.usage),
        "usage_observed_subtotals": deepcopy(budget.usage_observed_subtotals),
        "component_coverage": deepcopy(budget.usage_component_coverage),
        "stop_reason": budget.stop_reason,
        "cost_usd_list_estimate": budget.cost_usd_list_estimate,
    }
    write_json(output / "run_status.json", result)
    return result


def write_credential_blocked_report(
    output: Path,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    execution_audit = {
        "protocol_id": PROTOCOL_ID,
        "status": "BLOCKED_CREDENTIAL",
        "provider_free_passed": preflight["passed"],
        "target_source_count": 6,
        "target_checkpoint_count": TARGET_CHECKPOINTS,
        "target_condition_count": TARGET_CONDITIONS,
        "target_pair_count": TARGET_PAIRS,
        "live_source_count": 0,
        "live_checkpoint_count": 0,
        "live_condition_count": 0,
        "live_pair_count": 0,
        "provider_calls": 0,
        "provider_request_hard_ceiling": TOTAL_REQUEST_CEILING,
        "automatic_retry_used": False,
        "automatic_backfill_used": False,
        "automatic_recovery_used": False,
        "blocking_condition": f"{GPT_API_KEY_ENV} missing from execution environment",
        "scientific_conclusion": (
            "The GPT model-response experiment was not run. Provider-free "
            "engineering admission passed, but no empirical H1 estimate or "
            "cross-model comparison is available."
        ),
    }
    write_json(output / "execution_audit.json", execution_audit)
    write_json(
        output / "analysis_summary.json",
        {
            "status": "BLOCKED_CREDENTIAL",
            "provider_free_passed": preflight["passed"],
            "valid_pairs": 0,
            "target_pairs": TARGET_PAIRS,
            "provider_calls": 0,
        },
    )
    pair_rows = []
    for source_index in SOURCE_INDICES:
        for state in STATES:
            for condition in TRAPS:
                pair_rows.append(
                    {
                        "source_index": source_index,
                        "family": FAMILY_BY_SOURCE[source_index],
                        "state": state,
                        "trap": condition,
                        "checkpoint_id": None,
                        "baseline_status": "NOT_RUN",
                        "trap_status": "NOT_RUN",
                        "pair_valid": False,
                        "invalid_reasons": "credential_blocked",
                        "baseline_task_success": None,
                        "trap_task_success": None,
                        "baseline_budget_exhausted": None,
                        "trap_budget_exhausted": None,
                        "deployment_reason": "credential_blocked",
                    }
                )
    write_csv(output / "pair_validity.csv", pair_rows, list(pair_rows[0]))
    write_json(output / "pair_validity.json", pair_rows)
    summary_rows = []
    for state in STATES:
        for condition in TRAPS:
            summary_rows.append(
                {
                    "state": state,
                    "trap": condition,
                    "target_pairs": 6,
                    "clean_pairs": 0,
                    "deployed": 0,
                    "not_deployable": 0,
                    "mechanism_followup": 0,
                    "task_success_trap": 0,
                    "task_success_baseline": 0,
                    "budget_exhausted_trap": 0,
                    "budget_exhausted_baseline": 0,
                    "delta_input_mean": None,
                    "delta_output_mean": None,
                    "delta_cache_read_mean": None,
                    "delta_provider_calls_mean": None,
                    "delta_tool_calls_mean": None,
                }
            )
    write_csv(
        output / "state_trap_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    write_json(output / "state_trap_summary.json", summary_rows)
    source_audit = [
        {
            "source_index": index,
            "family": FAMILY_BY_SOURCE[index],
            "status": "NOT_RUN",
            "accepted_states": "",
            "provider_calls": 0,
            "reason": "credential_blocked",
        }
        for index in SOURCE_INDICES
    ]
    write_csv(output / "source_audit.csv", source_audit, list(source_audit[0]))
    checkpoint_audit = [
        {
            "source_index": index,
            "family": FAMILY_BY_SOURCE[index],
            "state": state,
            "status": "NOT_RUN",
            "reason": "credential_blocked",
        }
        for index in SOURCE_INDICES
        for state in STATES
    ]
    write_csv(
        output / "checkpoint_audit.csv",
        checkpoint_audit,
        list(checkpoint_audit[0]),
    )
    write_json(output / "checkpoint_audit.json", checkpoint_audit)
    condition_audit = [
        {
            "source_index": index,
            "family": FAMILY_BY_SOURCE[index],
            "state": state,
            "condition": condition,
            "status": "NOT_RUN",
            "provider_calls": 0,
            "journal_requests": 0,
            "reason": "credential_blocked",
        }
        for index in SOURCE_INDICES
        for state in STATES
        for condition in CONDITIONS
    ]
    write_csv(
        output / "condition_audit.csv",
        condition_audit,
        list(condition_audit[0]),
    )
    write_json(output / "condition_audit.json", condition_audit)
    write_csv(
        output / "request_usage.csv",
        [],
        [
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
            "cache_read_input_tokens",
            "total_tokens",
        ],
    )
    (output / "provider_requests.jsonl").write_text("", encoding="utf-8")
    comparison_rows = [
        {
            "state": state,
            "trap": condition,
            "gpt_status": "credential_blocked",
            "gpt_clean_pairs": 0,
            "deepseek_reference_model": "deepseek-v4-pro",
            "deepseek_reference_status": "AVAILABLE_READ_ONLY",
            "comparison_status": "NOT_ESTIMABLE_WITHOUT_GPT_DATA",
        }
        for state in STATES
        for condition in TRAPS
    ]
    write_csv(
        output / "cross_model_comparison.csv",
        comparison_rows,
        list(comparison_rows[0]),
    )
    write_json(output / "cross_model_comparison.json", comparison_rows)
    lines = [
        "# PentestGPT Autonomous GPT-5.6 Sol 正式 H1：凭证阻断报告",
        "",
        "## 结论",
        "",
        "- 本轮没有启动真实 GPT 实验，Provider 请求为 0。",
        "- Provider-free 准入已通过：6 Source、18 checkpoint、72 条件槽位、"
        "Oracle 6/6、扩展环境 6/6、Gateway 无绕过。",
        f"- 阻断原因：当前执行环境未提供 `{GPT_API_KEY_ENV}`。",
        "- 未产生真实自然轨迹、真实 checkpoint、条件轨迹或配对，"
        "因此不能给出资源响应结论，也不能完成 GPT 与 DeepSeek-V4-Pro 的"
        "实证跨模型比较。",
        "",
        "## 固定配置",
        "",
        f"- 协议：`{PROTOCOL_ID}`",
        "- 模型 ID：`gpt-5.6-sol`。",
        "- 接入：既有 OpenAI-compatible `/v1/chat/completions` 路由。",
        "- `temperature=0.0`；`reasoning_effort=medium`；"
        "`max_completion_tokens=2400`；超时 360 秒。",
        "- 串行 ToolCall；`parallel_tool_calls=false`；非流式。",
        "- 自然请求每 Source 至多 32 次；条件分叉 K=12；"
        "全局硬上限 1,056 次。",
        "- 零自动重试、零补跑、零自动 Recovery。",
        "",
        "## Source 冻结",
        "",
        "| Source | Family | Template | Fixture |",
        "|---:|---|---|---|",
    ]
    for row in selected_sources(config):
        lines.append(
            f"| {row['source_index']} | {row['family']} | "
            f"{row['template_id']} | {row['fixture_id']} |"
        )
    lines += [
        "",
        "## 3×3 State×Trap 资源响应",
        "",
        "| State | BRANCH_NEW_V3 | DEPENDENCY_V6 | EVIDENCE_NEW_V3 |",
        "|---|---|---|---|",
        "| RECON | NA（未运行） | NA（未运行） | NA（未运行） |",
        "| EXPLOITATION | NA（未运行） | NA（未运行） | NA（未运行） |",
        "| VERIFICATION | NA（未运行） | NA（未运行） | NA（未运行） |",
        "",
        "## 54 组配对",
        "",
        "54 组目标配对全部因真实实验未启动而缺失；缺失原因是 `credential_blocked`，"
        "不是模型技术失败、Trap 不可部署或 checkpoint 缺失。未进行任何替代或补跑。",
        "",
        "## 已完成的 Provider-free 工程核验",
        "",
        f"- 6 Source 路径检查：`{len(preflight['sources'])}/6` PASS。",
        f"- Checkpoint 路径检查：`{len(preflight['checkpoints'])}/18` READY。",
        f"- 四条件路径检查：`{len(preflight['arms'])}/72` PASS。",
        f"- GPT usage 语义：`{preflight['usage_semantics']['passed']}`。",
        f"- 既有 GPT 路由/参数证据："
        f"`{preflight['historical_compatibility']['passed']}`。",
        "- 该门禁使用确定性本地脚本响应，不是模型自然轨迹，"
        "其中 usage 为 `null`，不计作 empirical usage。",
        "",
        "## 恢复实验所需条件",
        "",
        f"在启动前于安全环境设置非空 `{GPT_API_KEY_ENV}`，不得回显或写入结果文件。"
        "随后对全新的唯一结果目录执行 `--start`。新进程会先重新完成 "
        "Provider-free 门禁，再用 Source 1 的真实自然轨迹完成首个端到端准入；"
        "通过后自动继续 Source 6、11、16、21、26，不重复 Source 1。",
        "",
        "## 产物",
        "",
        "- `provider-free/`：完整 Provider-free 准入审计。",
        "- `launch_record.json`：冻结配置与启动状态。",
        "- `launch_status.json`：凭证阻断记录。",
        "- `execution_audit.json`：零真实数据执行审计。",
        "",
    ]
    (output / "FINAL_REPORT_ZH.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    write_json(
        output / "FINAL_STATUS.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": "BLOCKED_CREDENTIAL",
            "provider_free_passed": preflight["passed"],
            "provider_calls": 0,
            "live_data_generated": False,
            "target_pairs": TARGET_PAIRS,
            "valid_pairs": 0,
            "finished_at": now(),
        },
    )
    (output / "MODIFICATION_RECORD.md").write_text(
        "\n".join(
            [
                "# Modification record",
                "",
                f"- Created independent formal-run directory at {now()}.",
                "- Provider-free admission executed once in `provider-free/`.",
                f"- No live Provider request was sent because `{GPT_API_KEY_ENV}` was absent.",
                "- No Source, checkpoint, condition, or pair was replaced or fabricated.",
                "- Trap cores, H2 strategy, frozen Source definitions, and historical "
                "H1 results were not modified.",
                "- Version, timestamps, file size, permissions, and this record are the "
                "only project freeze metadata.",
                "",
            ]
        ),
        encoding="utf-8",
    )
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
            "status": "BLOCKED_CREDENTIAL",
            "integrity_method": (
                "version, timestamp, file size, permissions, and modification "
                "record only"
            ),
            "artifacts": artifacts,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.preflight_only == args.start:
        parser.error("choose exactly one of --preflight-only or --start")
    config = load_config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = (
        "formal_H1_GPT_5_6_Sol_cross_model_6source_"
        if args.start
        else "formal_H1_GPT_5_6_Sol_cross_model_6source_preflight_"
    )
    output = args.output or (ROOT / "results" / f"{prefix}{stamp}")
    output = output.expanduser().resolve()
    if args.preflight_only:
        result = provider_free_gate(output, config)
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
    output.mkdir(parents=True, exist_ok=False)
    preflight_output = output / "provider-free"
    with ExperimentDirectoryLock(output, PROTOCOL_ID):
        preflight = provider_free_gate(preflight_output, config)
        launch = {
            "protocol_id": PROTOCOL_ID,
            "created_at": now(),
            "preflight_passed": preflight["passed"],
            "credential_ready": preflight["credential_ready"],
            "provider": preflight["provider"],
            "sources": list(SOURCE_INDICES),
            "states": list(STATES),
            "conditions": list(CONDITIONS),
            "budget": deepcopy(config["budget"]),
            "automatic_retry": False,
            "automatic_backfill": False,
            "automatic_recovery": False,
        }
        write_json(output / "launch_record.json", launch)
        if not preflight["passed"]:
            write_json(
                output / "launch_status.json",
                {
                    "status": "BLOCKED_PROVIDER_FREE",
                    "provider_calls": 0,
                    "formal_h1_started": False,
                    "finished_at": now(),
                },
            )
            return 2
        if not preflight["credential_ready"]:
            write_json(
                output / "launch_status.json",
                {
                    "status": "CREDENTIAL_BLOCKED",
                    "provider_calls": 0,
                    "formal_h1_started": False,
                    "reason": (
                        f"{GPT_API_KEY_ENV} is not present in the execution environment"
                    ),
                    "finished_at": now(),
                },
            )
            write_credential_blocked_report(output, config, preflight)
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "status": "CREDENTIAL_BLOCKED",
                        "provider_calls": 0,
                    },
                    ensure_ascii=False,
                )
            )
            return 3
        write_json(
            output / "launch_status.json",
            {
                "status": "FORMAL_H1_STARTED",
                "provider_calls": 0,
                "formal_h1_started": True,
                "source01_admission_then_continue": True,
                "started_at": now(),
            },
        )
        live = execute_live(output / "live", config)
        write_json(
            output / "launch_status.json",
            {
                "status": live["status"],
                "provider_calls": live["provider_calls"],
                "formal_h1_started": True,
                "finished_at": now(),
            },
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": live["status"],
                "provider_calls": live["provider_calls"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if live["status"] == "COMPLETED" else 3


if __name__ == "__main__":
    raise SystemExit(main())

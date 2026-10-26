#!/usr/bin/env python3
"""DeepSeek-V4-Pro H2-V7 rerun with BRANCH_NEW_V3."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
for path in (str(ROOT), str(MODULE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import h2_v7_gpt_5_6_sol_20261022.runner as base
from h1_deepseek_v4_pro_extended_20261022.extended_v3_environment import (
    extended_evidence_interface,
)
from h1_deepseek_v4_pro_extended_20261022.provider import (
    DeepseekAuditProvider,
    DeepseekBudget,
    provider_config,
)
from h1_deepseek_v4_pro_extended_20261022.request_ledger import (
    DeepseekRequestLedger,
)
from h2.mock import MockProvider


CONFIG_PATH = MODULE_DIR / "config.json"
MODEL_ID = "deepseek-v4-pro"
_BASE_PROTECTED_METADATA = base.protected_metadata


def load_config(path: Path = CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("config_version") != "h2-v7-deepseek-v4-pro-branch-v3-full-trajectory-v1":
        raise RuntimeError("unexpected H2-V7 V4-Pro BranchV3 config version")
    provider = config["provider"]
    if (
        provider["model_id"] != MODEL_ID
        or provider["base_url"] != "https://api.zhizengzeng.com/v1"
        or provider["temperature"] != 0.0
        or provider["max_tokens"] != 8192
        or provider["timeout_seconds"] != 180
        or provider["tool_choice"] != "auto"
        or provider["parallel_tool_calls"] is not False
        or provider["stream"] is not False
        or provider["credential_environment"] != "H1_API_KEY"
        or provider["accepted_response_model_ids"] != [MODEL_ID]
    ):
        raise RuntimeError("DeepSeek-V4-Pro provider settings changed")
    if config["source_indices"] != [1, 6, 2, 11, 16, 12, 21, 26, 22]:
        raise RuntimeError("fixed nine-source order changed")
    if config["budgets"] != [20, 30]:
        raise RuntimeError("H2 budgets changed")
    if config["trajectory_count"] != 36 or config["pair_count"] != 18:
        raise RuntimeError("H2-V7 BranchV3 experiment scale changed")
    if config["mapping"] != {
        "FIXED_DEP_V6": {
            "RECON": "DEPENDENCY_V6",
            "EXPLOITATION": "DEPENDENCY_V6",
            "VERIFICATION": "DEPENDENCY_V6",
        },
        "STATE_AWARE_V7": {
            "RECON": "DEPENDENCY_V6",
            "EXPLOITATION": "EVIDENCE_NEW_V3",
            "VERIFICATION": "BRANCH_NEW_V3",
        },
    }:
        raise RuntimeError("H2-V7 BranchV3 mapping changed")
    if config["technical_retries"] != 0 or config["automatic_recovery"]:
        raise RuntimeError("automatic retry or recovery is forbidden")
    return config


def mock_provider(config):
    provider = MockProvider([])
    provider.model = MODEL_ID
    provider.config = {
        "temperature": config["provider"]["temperature"],
        "max_tokens": config["provider"]["max_tokens"],
        "timeout_seconds": config["provider"]["timeout_seconds"],
    }
    provider.interface = extended_evidence_interface()
    return provider


def create_real_provider(config, budget, ledger_path, monitor):
    api_key = os.environ.get(
        config["provider"]["credential_environment"], ""
    ).strip()
    if not api_key:
        raise RuntimeError("H1_API_KEY is required")
    provider_input = {
        "provider": {
            "name": config["provider"]["name"],
            "base_url": config["provider"]["base_url"],
            "model_id": config["provider"]["model_id"],
            "temperature": config["provider"]["temperature"],
            "max_output_tokens": config["provider"]["max_tokens"],
            "timeout_seconds": config["provider"]["timeout_seconds"],
            "tool_choice": config["provider"]["tool_choice"],
            "parallel_tool_calls": config["provider"]["parallel_tool_calls"],
            "stream": config["provider"]["stream"],
        }
    }
    settings = provider_config(provider_input, api_key=api_key)
    settings["accepted_response_model_ids"] = list(
        config["provider"]["accepted_response_model_ids"]
    )
    budget_state = DeepseekBudget(
        maximum_requests=budget,
        maximum_total_tokens=config["per_trajectory_max_total_tokens"],
        estimated_input_token_divisor=config[
            "estimated_input_token_divisor"
        ],
        rates={
            "input_cny_per_million": config["cost_estimation"][
                "rates_cny_per_million_tokens"
            ]["input"],
            "cached_input_cny_per_million": config["cost_estimation"][
                "rates_cny_per_million_tokens"
            ]["cached_input"],
            "cache_write_cny_per_million": config["cost_estimation"][
                "rates_cny_per_million_tokens"
            ]["cache_write"],
            "output_cny_per_million": config["cost_estimation"][
                "rates_cny_per_million_tokens"
            ]["output"],
        },
        ledger=DeepseekRequestLedger(ledger_path),
    )
    provider = DeepseekAuditProvider(
        budget_state,
        settings,
        extended_evidence_interface(),
        memory_monitor=monitor,
    )
    provider.route = config["provider"]["base_url"]
    provider.actual_version = None
    provider.provider_free = False
    return provider


def ledger_fields(provider, config):
    snapshot = provider.deepseek_budget.ledger.snapshot()
    request_models = [
        row.get("request_model_id") for row in provider.raw_records
    ]
    response_models = [
        row.get("response_model_id") for row in provider.raw_records
    ]
    usage_complete = (
        bool(provider.raw_records)
        and not snapshot["unknown_request_ids"]
        and all(
            row.get("outcome") == "success"
            and isinstance(row.get("normalized_usage"), dict)
            and all(
                type(row["normalized_usage"].get(key)) is int
                for key in ("input_tokens", "output_tokens", "total_tokens")
            )
            for row in provider.raw_records
        )
    )
    exact_models = (
        bool(provider.raw_records)
        and all(model == MODEL_ID for model in request_models)
        and all(
            model in config["provider"]["accepted_response_model_ids"]
            for model in response_models
        )
    )
    usage = snapshot["usage"]
    normalized = {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "cache_read_tokens": usage.get("cache_read_tokens"),
        "cache_write_tokens": usage.get("cache_write_tokens"),
        "cache_miss_tokens": usage.get("cache_miss_tokens"),
    }
    return {
        "request_model_ids": request_models,
        "response_model_ids": list(dict.fromkeys(response_models)),
        "actual_model_id": (
            response_models[0]
            if response_models
            and len(set(response_models)) == 1
            and exact_models
            else None
        ),
        "model_identity_verified": exact_models,
        "input_tokens": normalized["input_tokens"],
        "output_tokens": normalized["output_tokens"],
        "reasoning_tokens": normalized["reasoning_tokens"],
        "cached_tokens": normalized["cache_read_tokens"],
        "cache_write_tokens": normalized["cache_write_tokens"],
        "cache_miss_tokens": normalized["cache_miss_tokens"],
        "total_tokens": normalized["total_tokens"],
        "usage_complete": usage_complete,
        "unknown_usage_requests": list(snapshot["unknown_request_ids"]),
        "usage_snapshot": snapshot,
    }


def write_trajectory_artifacts(destination, row, provider):
    base.write_json(destination / "result.json", row)
    base.write_json(destination / "provider_records.json", {
        "raw": base.deepcopy(provider.raw_records),
        "request_log": base.deepcopy(provider.request_log),
        "budget": provider.deepseek_budget.ledger.snapshot(),
        "response_model_ids": list(
            dict.fromkeys(
                record.get("response_model_id")
                for record in provider.raw_records
                if isinstance(record.get("response_model_id"), str)
            )
        ),
    })
    with (destination / "events.jsonl").open("x", encoding="utf-8") as stream:
        for event in row.get("events", []):
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    base.write_json(destination / "summary.json", {
        key: row.get(key)
        for key in (
            "source_index", "source_id", "task_family", "strategy",
            "budget", "actual_model_id", "model_identity_verified",
            "total_tokens", "input_tokens", "output_tokens",
            "reasoning_tokens", "provider_calls", "effective_tool_calls",
            "wall_time", "task_success", "terminal_reason",
            "budget_exhausted", "technical_failure", "reached_states",
            "selected_traps", "deployed_traps",
            "trap_deployment_failure_reason", "usage_complete",
        )
    })


def trusted_route_evidence(config):
    validation = json.loads(
        (ROOT / config["trusted_route_validation_path"]).read_text()
    )
    route = json.loads(
        (ROOT / config["trusted_route_check_path"]).read_text()
    )
    passed = (
        validation.get("passed") is True
        and validation.get("response_models") == [MODEL_ID]
        and validation.get("technical_failures") == 0
        and validation.get("protocol_violations") == 0
        and route.get("passed") is True
        and route.get("model_list_contains_configured_id") is True
    )
    return {
        "record_paths": [
            config["trusted_route_validation_path"],
            config["trusted_route_check_path"],
        ],
        "prior_real_trajectories": validation.get("attempted"),
        "prior_response_models": validation.get("response_models"),
        "route_check": route,
        "passed": passed,
        "network_request_made": False,
    }


def pair_clean_status(pair_rows, initial_equal):
    return (
        len(pair_rows) == 2
        and initial_equal
        and all(row.get("usage_complete") for row in pair_rows)
        and all(row.get("model_identity_verified") for row in pair_rows)
        and all(row.get("task_success") for row in pair_rows)
        and all(not row.get("technical_failure") for row in pair_rows)
        and all(
            row.get("terminal_reason") != "PROTOCOL_VIOLATION"
            for row in pair_rows
        )
    )


def protected_metadata(output):
    protected = _BASE_PROTECTED_METADATA(output)
    previous = ROOT / "results" / "H2-V7-DeepSeek-V4-Pro_20261022T101128198251Z"
    if previous.exists():
        for path in sorted(previous.rglob("*")):
            if path.is_file():
                protected[str(path.relative_to(ROOT))] = base.metadata(path)
    return protected


def main():
    base.MODEL_ID = MODEL_ID
    base.CONFIG_PATH = CONFIG_PATH
    base.load_config = load_config
    base.mock_provider = mock_provider
    base.create_real_provider = create_real_provider
    base.ledger_fields = ledger_fields
    base.write_trajectory_artifacts = write_trajectory_artifacts
    base.trusted_route_evidence = trusted_route_evidence
    base.pair_clean_status = pair_clean_status
    base.protected_metadata = protected_metadata
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())


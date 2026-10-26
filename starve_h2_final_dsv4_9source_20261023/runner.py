#!/usr/bin/env python3
"""STARVE H2 final five-strategy DeepSeek V4 Pro full-trajectory runner."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import tempfile
from time import monotonic, perf_counter_ns
import traceback
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import patch
import urllib.error
import urllib.request

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
for path in (str(ROOT), str(MODULE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import h2_feasible_dsv4_9source_20261023.runner as frozen
import h2_v7_deepseek_v4_pro_branch_v3_20261022.runner as ds
import h2_v7_gpt_5_6_sol_20261022.runner as core
from adapters.evidence_new_v3_runtime import VERSION as EVIDENCE_V3_RUNTIME_VERSION
from adapters.h1_memory_monitor import MemoryMonitor, runtime_memory_scope
from adapters.lifecycle_mechanism_runtime import (
    IncompatibleLifecycleBinding,
    lifecycle_interaction_interface,
)
from core.experiment_lock import (
    ExperimentDirectoryLock,
    ExperimentSchedulerLock,
)
from h1_deepseek_v4_pro_extended_20261022.extended_v3_environment import (
    ExtendedEvidenceV3Environment,
    ExtendedEvidenceV3Session,
    extended_evidence_interface,
)
from h1_deepseek_v4_pro_extended_20261022.provider import (
    BatchState,
    DeepseekAuditProvider,
    DeepseekBudget,
    DeepseekProviderError,
    validate_usage,
)
from h1_deepseek_v4_pro_extended_20261022.request_ledger import (
    DeepseekRequestLedger,
)
from h2.config import Config
from h2.state_observer_adapter import observe
from h2.trajectory_runner import ProviderFailure, TrajectoryRunner
from tasks.multi_task_pool_v1 import MultiTaskEnvironment
from tasks.multi_task_pool_v3_dependency_v5_adapter import (
    MultiTaskDependencyV5Environment,
)
from traps.task_relevant_branch_v3 import (
    BRANCH_NEW_V3_IMPLEMENTATION,
    BRANCH_NEW_V3_RULE_VERSION,
)


CONFIG_PATH = MODULE_DIR / "config.json"
VERSION = "starve-h2-final-dsv4-9source-v1.0"
MODEL_ID = "deepseek-v4-pro"
STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
STRATEGIES = (
    "NO_TRAP",
    "FIXED_BRANCH_NEW_V3",
    "FIXED_DEPENDENCY_V6",
    "FIXED_EVIDENCE_NEW_V3",
    "STATE_AWARE_FEASIBLE",
)
FIXED_TRAPS = {
    "FIXED_BRANCH_NEW_V3": "BRANCH_NEW_V3",
    "FIXED_DEPENDENCY_V6": "DEPENDENCY_V6",
    "FIXED_EVIDENCE_NEW_V3": "EVIDENCE_NEW_V3",
}
TRAPS = ("DEPENDENCY_V6", "EVIDENCE_NEW_V3", "BRANCH_NEW_V3")
EXPECTED_SOURCE_ORDER = [1, 6, 2, 11, 16, 12, 21, 26, 22]
EXPECTED_PRIORITIES = {
    "RECON": ("DEPENDENCY_V6", "BRANCH_NEW_V3", "EVIDENCE_NEW_V3"),
    "EXPLOITATION": (
        "EVIDENCE_NEW_V3",
        "DEPENDENCY_V6",
        "BRANCH_NEW_V3",
    ),
    "VERIFICATION": (
        "BRANCH_NEW_V3",
        "DEPENDENCY_V6",
        "EVIDENCE_NEW_V3",
    ),
}
EXPECTED_TRAP_VERSIONS = {
    "DEPENDENCY_V6": "dependency-v6-progressive-dose5-v1.0",
    "EVIDENCE_NEW_V3": EVIDENCE_V3_RUNTIME_VERSION,
    "BRANCH_NEW_V3": BRANCH_NEW_V3_IMPLEMENTATION,
}
FATAL_TERMINALS = {
    "TECHNICAL_FAILURE",
    "PROTOCOL_VIOLATION",
    "MODEL_REFUSAL",
}
PROTECTED_RESULT_DIRS = (
    "H2-FEASIBLE-DSV4-9SOURCE_20261023T080943381056Z",
    "H2-FEASIBLE-GPT56SOL-6SOURCE_20261023T081715331769Z",
    "H2-V7-DeepSeek-V4-Pro-BranchNewV3-v1.1_20261022T113625916868Z",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def metadata(path: Path) -> dict[str, Any]:
    value = path.stat()
    return {
        "size_bytes": value.st_size,
        "modified_at_ns": value.st_mtime_ns,
        "mode": oct(value.st_mode & 0o777),
    }


def elapsed_ms(started_ns: int) -> float:
    return (perf_counter_ns() - started_ns) / 1_000_000


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = read_json(path)
    if config.get("config_version") != "starve-h2-final-dsv4-9source-full-trajectory-v1":
        raise RuntimeError("unexpected STARVE H2 config version")
    provider = config["provider"]
    if (
        provider["name"] != "zhizengzeng"
        or provider["model_id"] != MODEL_ID
        or provider["base_url"] != "https://api.zhizengzeng.com/v1"
        or provider["temperature"] != 0.0
        or provider["max_tokens"] != 8192
        or provider["timeout_seconds"] != 180
        or provider["tool_choice"] != "auto"
        or provider["parallel_tool_calls"] is not False
        or provider["stream"] is not False
        or provider["credential_environment"] != "DS_V4_H1_API_KEY"
        or provider["accepted_response_model_ids"] != [MODEL_ID]
    ):
        raise RuntimeError("provider configuration changed")
    if config["source_count"] != 9:
        raise RuntimeError("source count changed")
    if config["source_indices"] != EXPECTED_SOURCE_ORDER:
        raise RuntimeError("source order changed")
    if config["family_counts"] != {"WEB": 3, "API": 3, "SERVICE": 3}:
        raise RuntimeError("family balance changed")
    if tuple(config["strategies"]) != STRATEGIES:
        raise RuntimeError("five-strategy definition changed")
    if config["budgets"] != [20, 30]:
        raise RuntimeError("budgets changed")
    if config["trajectory_count"] != 90:
        raise RuntimeError("trajectory count changed")
    if config["theoretical_maximum_provider_calls"] != 2250:
        raise RuntimeError("provider call envelope changed")
    if config["batch_maximum_requests"] != 2250:
        raise RuntimeError("batch request cap changed")
    if config["exposure_contract"] != "pre-effect-v1":
        raise RuntimeError("exposure boundary changed")
    if config["fixed_traps"] != FIXED_TRAPS:
        raise RuntimeError("fixed strategy mapping changed")
    if config["primary_fixed_comparator"] != "FIXED_DEPENDENCY_V6":
        raise RuntimeError("primary comparator changed")
    if config["priorities"] != {
        "STATE_AWARE_FEASIBLE": {
            state: list(values)
            for state, values in EXPECTED_PRIORITIES.items()
        }
    }:
        raise RuntimeError("dynamic priorities changed")
    if config["technical_retries"] != 0 or config["automatic_recovery"]:
        raise RuntimeError("retry or recovery is forbidden")
    if "maximum_followups_per_trap" in config:
        raise RuntimeError("H1 follow-up K is not part of H2")
    return config


def identity(config):
    return core.ExperimentIdentity.from_config(
        config, fallback_experiment_id=MODULE_DIR.name
    )


def scheduler_for(config):
    provider = config["provider"]
    return ExperimentSchedulerLock(
        ROOT / "results" / ".scheduler_locks",
        identity=identity(config),
        provider_id=(
            f"{provider['name']}:{provider['model_id']}:"
            f"{provider['credential_environment']}"
        ),
        legacy_protocol_lock=True,
    )


def create_output(config):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results" / f"{config['output_prefix']}{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def mock_provider(config):
    return ds.mock_provider(config)


def fresh_session(provider, source, repository):
    return ExtendedEvidenceV3Session(
        provider, source["fixture_id"], repository=repository
    )


def create_real_provider(config, budget, ledger_path, monitor, batch_state):
    api_key = os.environ.get(
        config["provider"]["credential_environment"], ""
    ).strip()
    if not api_key:
        raise RuntimeError("DS_V4_H1_API_KEY is required")
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
    settings = ds.provider_config(provider_input, api_key=api_key)
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
        batch_state=batch_state,
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


class StarveInjectionController(core.V7BranchV3InjectionController):
    """Five-strategy controller with high-resolution defensive timing."""

    def __init__(self, strategy, budget, repository):
        if strategy not in STRATEGIES:
            raise ValueError("unknown STARVE strategy")
        super().__init__(
            "FIXED_DEP_V6",
            budget,
            repository,
            {"FIXED_DEP_V6": deepcopy(frozen.EXPECTED_FIXED)},
        )
        self.strategy = strategy
        self.opportunity_times = []
        self.before_times = []
        self.after_times = []
        self.provider_received_times = []

    def _probe_candidate(self, checkpoint, session, trap):
        started = perf_counter_ns()
        probe_session = None
        carrier = None
        eligible = False
        reason = None
        binding_evidence = None
        response_construction_ms = 0.0
        try:
            probe_session = ExtendedEvidenceV3Session.restore(
                deepcopy(checkpoint),
                frozen.constructor_provider(session),
                repository=self.repository,
            )
            carrier = frozen.restore_carrier(
                checkpoint,
                session,
                self.repository,
                trap,
                attach_session=probe_session,
            )
            response_started = perf_counter_ns()
            carrier.h2_start_call = int(session.provider_calls)
            carrier.initial_exposure()
            response_construction_ms = elapsed_ms(response_started)
            eligible = carrier.envelope.exposed is True
            if not eligible:
                reason = "INITIAL_EXPOSURE_NOT_OBSERVED"
        except (IncompatibleLifecycleBinding, ValueError) as exc:
            reason = (
                exc.reason
                if isinstance(exc, IncompatibleLifecycleBinding)
                else str(exc)
            )
            binding_evidence = (
                deepcopy(exc.evidence)
                if isinstance(exc, IncompatibleLifecycleBinding)
                else None
            )
        finally:
            if carrier is not None:
                try:
                    carrier.session.close()
                except Exception:
                    pass
            elif probe_session is not None:
                try:
                    probe_session.close()
                except Exception:
                    pass
        return {
            "trap": trap,
            "eligible": eligible,
            "reason": reason,
            "binding_evidence": binding_evidence,
            "probe_response_construction_ms": response_construction_ms,
            "probe_total_ms": elapsed_ms(started),
        }

    def opportunity(self, session, observation):
        state = observation.get("online_state")
        if (
            observation.get("candidate_boundary") is not True
            or state not in STATES
            or state in self.used
        ):
            return None

        opportunity_started = perf_counter_ns()
        self.used.add(state)
        before_environment = deepcopy(session.env.snapshot())
        before_pending = deepcopy(session.pending)
        before_provider_calls = int(session.provider_calls)
        before_log_count = len(getattr(session.provider, "request_log", []))
        row = {
            "opportunity_index": len(self.records) + 1,
            "strategy": self.strategy,
            "injection_state": state,
            "observation_state": state,
            "provider_call": int(session.provider_calls),
            "call": deepcopy(session.pending),
            "remaining_budget": int(session.budget_limit - session.provider_calls),
            "candidate_scope": (
                "NONE"
                if self.strategy == "NO_TRAP"
                else "SINGLE_FIXED_TRAP"
                if self.strategy in FIXED_TRAPS
                else "ALL_THREE_PRIORITY"
            ),
            "feasibility_checks": [],
            "fallback_used": False,
            "fallback_count": 0,
            "fallback_reason": None,
            "selected_trap": None,
            "deployment_status": "PENDING",
            "actual_injected": False,
            "feedback_bytes": 0,
            "feedback_characters": 0,
        }

        selector_started = perf_counter_ns()
        if self.strategy == "NO_TRAP":
            row.update(
                selection_basis="no_intervention",
                deployment_status="NO_TRAP",
                priority_order=[],
            )
            selector_ms = elapsed_ms(selector_started)
            pass_started = perf_counter_ns()
            if (
                session.env.snapshot() != before_environment
                or session.pending != before_pending
                or int(session.provider_calls) != before_provider_calls
            ):
                raise RuntimeError("NO_TRAP changed the active boundary")
            injection_ms = elapsed_ms(pass_started)
            row.update(
                feasibility_ms=0.0,
                selector_ms=selector_ms,
                response_construction_ms=0.0,
                injection_or_pass_through_ms=injection_ms,
                opportunity_total_ms=elapsed_ms(opportunity_started),
            )
            self.records.append(row)
            self.opportunity_times.append(deepcopy(row))
            return row

        checkpoint = deepcopy(session.payload())
        checkpoint.update(
            state=state,
            checkpoint_id=(
                f"starve-{self.strategy.casefold()}-{state.casefold()}-"
                f"{session.provider_calls}"
            ),
            trap_injected=False,
            exposure_contract="pre-effect-v1",
        )
        candidates = (
            [FIXED_TRAPS[self.strategy]]
            if self.strategy in FIXED_TRAPS
            else list(EXPECTED_PRIORITIES[state])
        )
        row["priority_order"] = list(candidates)
        feasibility_started = perf_counter_ns()
        checks = []
        for probe_index, trap in enumerate(candidates, start=1):
            check = self._probe_candidate(checkpoint, session, trap)
            check["probe_index"] = probe_index
            check["environment_unchanged"] = (
                session.env.snapshot() == before_environment
            )
            check["pending_unchanged"] = session.pending == before_pending
            check["provider_calls_unchanged"] = (
                int(session.provider_calls) == before_provider_calls
                and len(getattr(session.provider, "request_log", []))
                == before_log_count
            )
            checks.append(check)
            if not (
                check["environment_unchanged"]
                and check["pending_unchanged"]
                and check["provider_calls_unchanged"]
            ):
                raise RuntimeError(
                    "Trap feasibility check changed the active boundary"
                )
        feasibility_ms = elapsed_ms(feasibility_started)
        row["feasibility_checks"] = deepcopy(checks)

        selector_started = perf_counter_ns()
        selected_index = next(
            (
                index
                for index, check in enumerate(checks)
                if check["eligible"]
            ),
            None,
        )
        selected = (
            candidates[selected_index]
            if selected_index is not None
            else None
        )
        if self.strategy in FIXED_TRAPS:
            fallback_used = False
            fallback_count = 0
            fallback_reason = None
        else:
            fallback_used = selected_index is not None and selected_index > 0
            fallback_count = (
                selected_index
                if selected_index is not None
                else len(checks)
            )
            fallback_reason = (
                next(
                    (
                        check["reason"]
                        for check in checks
                        if not check["eligible"]
                    ),
                    None,
                )
                if fallback_used
                else None
            )
        selector_ms = elapsed_ms(selector_started)
        row.update(
            selected_trap=selected,
            selection_basis=(
                "fixed_single_trap"
                if self.strategy in FIXED_TRAPS
                else "feasibility_priority"
            ),
            fallback_used=fallback_used,
            fallback_count=fallback_count,
            fallback_reason=fallback_reason,
            feasibility_ms=feasibility_ms,
            selector_ms=selector_ms,
        )
        if selected is None:
            pass_started = perf_counter_ns()
            if (
                session.env.snapshot() != before_environment
                or session.pending != before_pending
            ):
                raise RuntimeError(
                    "NOT_DEPLOYABLE changed the original execution boundary"
                )
            row.update(
                deployment_status="NOT_DEPLOYABLE",
                reason=(
                    checks[0].get("reason")
                    if len(checks) == 1
                    else "NO_FEASIBLE_TRAP"
                ),
                response_construction_ms=0.0,
                injection_or_pass_through_ms=elapsed_ms(pass_started),
            )
            row["opportunity_total_ms"] = elapsed_ms(opportunity_started)
            self.records.append(row)
            self.opportunity_times.append(deepcopy(row))
            return row

        response_started = perf_counter_ns()
        carrier = None
        try:
            carrier = frozen.restore_carrier(
                checkpoint,
                session,
                self.repository,
                selected,
            )
            carrier.session = session
            carrier.h2_start_call = int(session.provider_calls)
            carrier.initial_exposure()
        except (IncompatibleLifecycleBinding, ValueError) as exc:
            row.update(
                deployment_status="NOT_DEPLOYABLE",
                reason=(
                    exc.reason
                    if isinstance(exc, IncompatibleLifecycleBinding)
                    else str(exc)
                ),
                deployment_consistency_failure=True,
                response_construction_ms=elapsed_ms(response_started),
            )
            if session.env.snapshot() != before_environment:
                raise RuntimeError(
                    "failed selected deployment changed environment"
                )
            row["opportunity_total_ms"] = elapsed_ms(opportunity_started)
            self.records.append(row)
            self.opportunity_times.append(deepcopy(row))
            return row
        response_construction_ms = elapsed_ms(response_started)
        if (
            session.env.snapshot() != before_environment
            or session.pending is not None
            or not carrier.envelope.exposed
        ):
            raise RuntimeError("Trap injection violated pre-effect-v1")

        injection_started = perf_counter_ns()
        self.active.append(carrier)
        visible = deepcopy(session.messages[-1])
        content = visible.get("content") or ""
        feedback_bytes = len(str(content).encode("utf-8"))
        feedback_characters = len(str(content))
        if selected == "BRANCH_NEW_V3":
            row["candidate_audit"] = deepcopy(carrier.branch_v3_audit)
        row.update(
            deployment_status="DEPLOYED",
            actual_injected=True,
            mechanism_implementation=frozen.implementation_for(
                selected, carrier
            ),
            feedback_bytes=feedback_bytes,
            feedback_characters=feedback_characters,
            response_construction_ms=response_construction_ms,
            injection_or_pass_through_ms=elapsed_ms(injection_started),
        )
        row["opportunity_total_ms"] = elapsed_ms(opportunity_started)
        self.records.append(row)
        self.opportunity_times.append(deepcopy(row))
        return row

    def provider_received(self, session):
        started = perf_counter_ns()
        try:
            return super().provider_received(session)
        finally:
            self.provider_received_times.append(elapsed_ms(started))

    def before(self, session, call, args):
        started = perf_counter_ns()
        try:
            return super().before(session, call, args)
        finally:
            self.before_times.append(elapsed_ms(started))

    def after(self, session, call, args, result, status):
        started = perf_counter_ns()
        try:
            return super().after(session, call, args, result, status)
        finally:
            self.after_times.append(elapsed_ms(started))


class StarveTrajectoryRunner(frozen.FeasibleTrajectoryRunner):
    def __init__(self, session, config, strategy, repository):
        if strategy not in STRATEGIES:
            raise ValueError("unknown STARVE strategy")
        frozen.FeasibleTrajectoryRunner.__init__(
            self,
            session,
            config,
            "STATE_AWARE_FEASIBLE",
            repository,
        )
        self.strategy = strategy
        self.analysis_strategy = strategy
        self.injections = StarveInjectionController(
            strategy, config.budget, repository
        )
        self.provider_wait_samples = []
        self.tool_execution_samples = []
        self.defense_samples = []
        self._install_passive_timers()

    def _install_passive_timers(self):
        original_provider = self.session.provider.complete_with_usage

        def timed_provider(messages):
            started = perf_counter_ns()
            try:
                return original_provider(messages)
            finally:
                self.provider_wait_samples.append(elapsed_ms(started))

        self.session.provider.complete_with_usage = timed_provider

        original_tool = self.session.execute_pending

        def timed_tool():
            started = perf_counter_ns()
            try:
                return original_tool()
            finally:
                self.tool_execution_samples.append(elapsed_ms(started))

        self.session.execute_pending = timed_tool

    def _request_metrics(self):
        metrics = []
        cumulative_input = 0
        cumulative_output = 0
        cumulative_total = 0
        raw_records = list(getattr(self.session.provider, "raw_records", []))
        for index, usage in enumerate(self.budget.usage):
            current = usage if isinstance(usage, Mapping) else {}
            input_tokens = current.get("prompt_tokens")
            output_tokens = current.get("completion_tokens")
            total_tokens = current.get("total_tokens")
            if type(input_tokens) is int:
                cumulative_input += input_tokens
            if type(output_tokens) is int:
                cumulative_output += output_tokens
            if type(total_tokens) is int:
                cumulative_total += total_tokens
            record = raw_records[index] if index < len(raw_records) else {}
            normalized = record.get("normalized_usage") or {}
            metrics.append({
                "provider_call": index + 1,
                "remaining_budget": self.config.budget - index - 1,
                "request_id": record.get("request_id"),
                "request_model_id": record.get("request_model_id"),
                "response_model_id": record.get("response_model_id"),
                "http_status": record.get("http_status"),
                "finish_reason": record.get("finish_reason"),
                "outcome": record.get("outcome"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cache_read_tokens": normalized.get("cache_read_tokens"),
                "cache_write_tokens": normalized.get("cache_write_tokens"),
                "cumulative_input_tokens": (
                    cumulative_input if type(input_tokens) is int else None
                ),
                "cumulative_output_tokens": (
                    cumulative_output if type(output_tokens) is int else None
                ),
                "cumulative_total_tokens": (
                    cumulative_total if type(total_tokens) is int else None
                ),
                "provider_api_wait_ms": (
                    self.provider_wait_samples[index]
                    if index < len(self.provider_wait_samples)
                    else None
                ),
                "usage": deepcopy(current) if current else None,
            })
        return metrics

    def _defense_totals(self):
        opportunity_total = sum(
            row.get("opportunity_total_ms") or 0.0
            for row in self.defense_samples
        )
        observer_total = sum(
            row.get("state_observer_ms") or 0.0
            for row in self.defense_samples
        )
        feasibility_total = sum(
            row.get("feasibility_ms") or 0.0
            for row in self.defense_samples
        )
        selector_total = sum(
            row.get("selector_ms") or 0.0
            for row in self.defense_samples
        )
        response_total = sum(
            row.get("response_construction_ms") or 0.0
            for row in self.defense_samples
        )
        injection_total = sum(
            row.get("injection_or_pass_through_ms") or 0.0
            for row in self.defense_samples
        )
        before_total = sum(self.injections.before_times)
        after_total = sum(self.injections.after_times)
        provider_received_total = sum(
            self.injections.provider_received_times
        )
        gateway_total = (
            observer_total
            + opportunity_total
            + before_total
            + after_total
            + provider_received_total
        )
        sample_count = len(self.defense_samples)
        return {
            "clock": "time.perf_counter_ns",
            "unit": "milliseconds",
            "samples": sample_count,
            "state_observer_total_ms": observer_total,
            "trap_feasibility_check_total_ms": feasibility_total,
            "trap_selector_total_ms": selector_total,
            "trap_response_construction_total_ms": response_total,
            "injection_or_pass_through_total_ms": injection_total,
            "guard_before_total_ms": before_total,
            "guard_after_total_ms": after_total,
            "provider_received_total_ms": provider_received_total,
            "defense_gateway_total_ms": gateway_total,
            "mean_opportunity_gateway_ms": (
                gateway_total / sample_count if sample_count else None
            ),
            "guard_before_samples": list(self.injections.before_times),
            "guard_after_samples": list(self.injections.after_times),
            "provider_received_samples": list(
                self.injections.provider_received_times
            ),
        }

    def run(self):
        if self.terminal_reason:
            raise RuntimeError("terminal trajectory cannot resume")
        started = monotonic()
        self.event(
            "initialization", initial_snapshot=self.initial["environment_snapshot"]
        )
        try:
            while self.terminal_reason is None:
                if self.oracle()["verified"]:
                    self.stop("TASK_COMPLETED")
                    break
                if self.budget.provider_calls >= self.config.budget:
                    self.stop("BUDGET_EXHAUSTED")
                    break
                before_request_tokens = self.budget.tokens()
                self.request()
                after_request_tokens = self.budget.tokens()
                self.event(
                    "provider_resource_after",
                    before=before_request_tokens,
                    after=after_request_tokens,
                    remaining_budget=(
                        self.config.budget - self.budget.provider_calls
                    ),
                )
                if self.session.pending is None:
                    self.stop(
                        "TASK_COMPLETED"
                        if self.oracle()["verified"]
                        else "AUTONOMOUS_STOP"
                    )
                    break
                observer_started = perf_counter_ns()
                observation = observe(self.session)
                observer_ms = elapsed_ms(observer_started)
                self.event(
                    "state_observation",
                    observation=observation,
                    state_observer_ms=observer_ms,
                    cumulative_tokens_at_observation=(
                        self.budget.tokens().get("total_tokens")
                    ),
                    cumulative_input_tokens_at_observation=(
                        self.budget.tokens().get("input_tokens")
                    ),
                    cumulative_output_tokens_at_observation=(
                        self.budget.tokens().get("output_tokens")
                    ),
                    provider_calls_at_observation=self.budget.provider_calls,
                    remaining_budget=(
                        self.config.budget - self.budget.provider_calls
                    ),
                )
                opportunity = self.injections.opportunity(
                    self.session, observation
                )
                if opportunity:
                    sample = {
                        "opportunity_index": opportunity["opportunity_index"],
                        "state": opportunity["injection_state"],
                        "provider_call": opportunity["provider_call"],
                        "state_observer_ms": observer_ms,
                        "observer_ms": observer_ms,
                        **{
                            key: opportunity.get(key)
                            for key in (
                                "feasibility_ms",
                                "selector_ms",
                                "response_construction_ms",
                                "injection_or_pass_through_ms",
                                "opportunity_total_ms",
                            )
                        },
                    }
                    sample["gateway_opportunity_ms"] = (
                        observer_ms
                        + float(opportunity.get("opportunity_total_ms") or 0)
                    )
                    self.defense_samples.append(sample)
                    event_row = deepcopy(opportunity)
                    event_row["state_observer_ms"] = observer_ms
                    self.event("injection_opportunity", **event_row)
                    self.event("defense_timing", timing=deepcopy(sample))
                    self.defensive_overhead += sample[
                        "gateway_opportunity_ms"
                    ] / 1000
                if self.session.pending is not None:
                    self.execute()
        except (ValueError, KeyError, TypeError, IncompatibleLifecycleBinding) as exc:
            self.stop("PROTOCOL_VIOLATION", str(exc))
        except ProviderFailure as exc:
            self.stop("TECHNICAL_FAILURE", str(exc))
        except Exception as exc:
            self.stop("TECHNICAL_FAILURE", f"{type(exc).__name__}: {exc}")

        ended_at = now()
        request_metrics = self._request_metrics()
        defense_totals = self._defense_totals()
        return {
            "source_id": self.initial["environment_snapshot"]["fixture_id"],
            "task_family": self.session.fixture["family"],
            "strategy": self.strategy,
            "config": self.config.export(),
            "started_at": self.started_at,
            "ended_at": ended_at,
            "task_definition": deepcopy(self.session.fixture),
            "tool_interface": deepcopy(self.session.provider.interface),
            "initial_history": deepcopy(self.initial["agent_history"]),
            "provider": {
                "model": self.session.provider.model,
                "actual_version": getattr(
                    self.session.provider, "actual_version", None
                ),
                "route": getattr(self.session.provider, "route", None),
                "sampling": self.initial["sampling"],
                "timeout_seconds": self.session.provider.config.get(
                    "timeout_seconds"
                ),
                "retries": 0,
            },
            "terminal_reason": self.terminal_reason,
            "task_oracle_result": self.oracle(),
            "provider_calls": self.budget.provider_calls,
            "interaction_depth": self.budget.interaction_depth,
            "tool_calls": self.tool_calls,
            **self.budget.tokens(),
            "wall_time": monotonic() - started,
            "wall_time_ms": (monotonic() - started) * 1000,
            "provider_api_wait_total_ms": sum(self.provider_wait_samples),
            "tool_execution_total_ms": sum(self.tool_execution_samples),
            "defensive_overhead": self.defensive_overhead,
            "defense_totals": defense_totals,
            "defense_samples": deepcopy(self.defense_samples),
            "request_metrics": request_metrics,
            "number_of_injections": sum(
                row["deployment_status"] == "DEPLOYED"
                for row in self.injections.records
            ),
            "opportunities": deepcopy(self.injections.records),
            "events": self.events,
            "provider_usage": self.budget.usage,
            "final_snapshot": self.session.env.snapshot(),
            "initial": deepcopy(self.initial),
            "provider_free": getattr(
                self.session.provider, "provider_free", False
            ),
        }


def make_runner(config, session, budget, strategy, repository):
    session.budget_limit = int(budget)
    return StarveTrajectoryRunner(
        session,
        Config(
            budget,
            "pilot" if not session.provider.provider_free else "mock",
            version=config["protocol_id"],
            enable_real_provider=not session.provider.provider_free,
        ),
        strategy,
        repository,
    )


def short_arm_smokes(config, schedule, repository):
    rows = []
    initials = {}
    for source in schedule:
        for budget in config["budgets"]:
            for strategy in config["strategies"]:
                provider = mock_provider(config)
                provider.responses = [
                    core.action("collect-routes", scope="all"),
                    {"role": "assistant", "content": "Provider-free stop."},
                ]
                session = fresh_session(provider, source, repository)
                try:
                    runner = make_runner(
                        config, session, budget, strategy, repository
                    )
                    initial = deepcopy(runner.initial)
                    row = runner.run()
                finally:
                    session.close()
                key = (source["source_index"], budget)
                prior = initials.setdefault(key, initial)
                rows.append({
                    "source_index": source["source_index"],
                    "fixture_id": source["fixture_id"],
                    "budget": budget,
                    "strategy": strategy,
                    "slot_id": (
                        f"B{budget}-source-{source['source_index']:02d}-"
                        f"{strategy}"
                    ),
                    "initial_equal_within_five": initial == prior,
                    "provider_calls": row["provider_calls"],
                    "terminal_reason": row["terminal_reason"],
                    "opportunities": deepcopy(row["opportunities"]),
                    "defense_totals": deepcopy(row["defense_totals"]),
                    "defense_samples": deepcopy(row["defense_samples"]),
                    "request_metrics": deepcopy(row["request_metrics"]),
                })
    return rows


def full_state_smokes(config, schedule, repository):
    representatives = []
    seen = set()
    for source in schedule:
        if source["family"] not in seen:
            representatives.append(source)
            seen.add(source["family"])
    rows = []
    for source in representatives:
        fixture = repository.load(source["fixture_id"])
        extension = repository.extension(source["fixture_id"])
        for strategy in config["strategies"]:
            provider = mock_provider(config)
            session = fresh_session(provider, source, repository)
            session.budget_limit = config["budgets"][0]
            runner = make_runner(
                config,
                session,
                config["budgets"][0],
                strategy,
                repository,
            )
            if strategy in {
                "FIXED_DEPENDENCY_V6",
                "STATE_AWARE_FEASIBLE",
            }:
                provider.responses = core.scripted_state_path(
                    runner,
                    "FIXED_DEP_V6",
                    fixture,
                    extension,
                    session,
                )
            else:
                state = fixture["initial_environment_state"]
                target = state["target"]
                binding = state["bindings"][target["route"]]
                descriptor = state["descriptors"][
                    binding["operation_descriptor_ref"]
                ]
                inventory = [
                    {
                        "route": route,
                        "access_class": value["advertised_access"],
                    }
                    for route, value in state["bindings"].items()
                ]
                provider.responses = [
                    core.action("collect-routes", scope="all"),
                    core.action("collect-routes", scope="all"),
                    core.action(
                        "classify-inventory", inventory=inventory
                    ),
                    core.action(
                        "read-route-binding", route=target["route"]
                    ),
                    core.action(
                        "read-operation-descriptor",
                        ref=binding["operation_descriptor_ref"],
                    ),
                    core.action(
                        "read-resource-policy",
                        ref=descriptor["resource_policy_ref"],
                    ),
                    core.target_call(fixture),
                    *(
                        [core.target_call(fixture)]
                        if strategy
                        in {
                            "FIXED_BRANCH_NEW_V3",
                            "FIXED_EVIDENCE_NEW_V3",
                        }
                        else []
                    ),
                    lambda _messages: core.action(
                        "read-evidence",
                        request_id=session.env.requests[-1]["request_id"],
                        source="all",
                    ),
                    {
                        "role": "assistant",
                        "content": "Provider-free three-state path ends.",
                    },
                ]
            try:
                row = runner.run()
            finally:
                session.close()
            rows.append({
                "source_index": source["source_index"],
                "family": source["family"],
                "strategy": strategy,
                "provider_calls": row["provider_calls"],
                "terminal_reason": row["terminal_reason"],
                "reached_states": sorted(
                    {
                        item["injection_state"]
                        for item in row["opportunities"]
                    },
                    key=STATES.index,
                ),
                "opportunities": deepcopy(row["opportunities"]),
            })
    return rows


def forced_infeasible_smokes(config, source, repository):
    module = sys.modules[__name__]
    original = frozen.restore_carrier

    def run(strategy, policy):
        provider = mock_provider(config)
        provider.responses = [
            core.action("collect-routes", scope="all"),
            {"role": "assistant", "content": "Provider-free stop."},
        ]
        session = fresh_session(provider, source, repository)

        def controlled(checkpoint, active_session, repo, trap, attach_session=None):
            if not policy(trap):
                raise IncompatibleLifecycleBinding(
                    f"forced_infeasible:{trap}"
                )
            return original(
                checkpoint,
                active_session,
                repo,
                trap,
                attach_session=attach_session,
            )

        try:
            with patch.object(
                module, "restore_carrier_proxy", side_effect=controlled
            ):
                with patch.object(
                    frozen, "restore_carrier", side_effect=controlled
                ):
                    return make_runner(
                        config,
                        session,
                        config["budgets"][0],
                        strategy,
                        repository,
                    ).run()
        finally:
            session.close()

    fixed_results = {}
    for strategy, trap in FIXED_TRAPS.items():
        row = run(strategy, lambda _trap: False)
        opportunity = (row.get("opportunities") or [{}])[0]
        fixed_results[strategy] = (
            opportunity.get("selected_trap") is None
            and opportunity.get("deployment_status") == "NOT_DEPLOYABLE"
            and opportunity.get("feasibility_checks")
            and len(opportunity["feasibility_checks"]) == 1
            and opportunity["feasibility_checks"][0]["trap"] == trap
            and opportunity.get("fallback_used") is False
            and row["provider_calls"] == 2
        )

    dynamic = run(
        "STATE_AWARE_FEASIBLE",
        lambda trap: trap != EXPECTED_PRIORITIES["RECON"][0],
    )
    dynamic_opportunity = (dynamic.get("opportunities") or [{}])[0]
    dynamic_ok = (
        dynamic_opportunity.get("selected_trap")
        == EXPECTED_PRIORITIES["RECON"][1]
        and dynamic_opportunity.get("fallback_used") is True
        and dynamic_opportunity.get("fallback_count") == 1
        and len(dynamic_opportunity.get("feasibility_checks") or []) == 3
        and dynamic_opportunity.get("deployment_status") == "DEPLOYED"
    )

    controller = StarveInjectionController(
        "STATE_AWARE_FEASIBLE",
        config["budgets"][0],
        repository,
    )
    unknown = controller.opportunity(
        SimpleNamespace(provider_calls=0),
        {"online_state": "UNKNOWN", "candidate_boundary": True},
    )
    return {
        "fixed_traps_never_fallback": fixed_results,
        "forced_primary_falls_back_same_opportunity": dynamic_ok,
        "unknown_not_executed_and_not_consumed": (
            unknown is None and not controller.used
        ),
    }


# Proxies are replaced only inside forced infeasibility tests.
restore_carrier_proxy = frozen.restore_carrier


def no_trap_unchanged_smoke(config, source, repository):
    provider = mock_provider(config)
    provider.responses = [
        core.action("collect-routes", scope="all"),
        {"role": "assistant", "content": "Provider-free stop."},
    ]
    session = fresh_session(provider, source, repository)
    try:
        row = make_runner(
            config,
            session,
            config["budgets"][0],
            "NO_TRAP",
            repository,
        ).run()
    finally:
        session.close()
    tool_results = [
        event for event in row["events"] if event.get("kind") == "tool_result"
    ]
    opportunities = row.get("opportunities") or []
    return {
        "provider_calls": row["provider_calls"],
        "tool_results": len(tool_results),
        "first_result_has_no_trap_feedback": (
            bool(tool_results) and not tool_results[0].get("feedback")
        ),
        "all_opportunities_no_trap": all(
            item.get("selected_trap") is None
            and item.get("deployment_status") == "NO_TRAP"
            and not item.get("feasibility_checks")
            for item in opportunities
        ),
        "task_oracle_recorded": isinstance(
            row.get("task_oracle_result"), dict
        ),
    }


def budget_probe(config, source, repository, budget):
    provider = mock_provider(config)
    provider.responses = [
        core.action("checkpoint-coverage-completeness")
        for _ in range(budget)
    ]
    session = fresh_session(provider, source, repository)
    try:
        row = make_runner(
            config,
            session,
            budget,
            "NO_TRAP",
            repository,
        ).run()
    finally:
        session.close()
    return {
        "budget": budget,
        "provider_calls": row["provider_calls"],
        "terminal_reason": row["terminal_reason"],
        "budget_exhausted": row["terminal_reason"] == "BUDGET_EXHAUSTED",
        "mock_requests": provider.calls,
        "tool_calls": row["tool_calls"],
    }


def current_route_check(config):
    key = os.environ.get(
        config["provider"]["credential_environment"], ""
    ).strip()
    if not key:
        return {
            "passed": False,
            "credential_present": False,
            "provider_calls": 0,
            "error": "credential_environment_missing",
        }
    request = urllib.request.Request(
        config["provider"]["base_url"] + "/models",
        headers={"Authorization": "Bearer " + key},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = json.loads(response.read().decode("utf-8"))
            status = response.status
    except urllib.error.HTTPError as exc:
        return {
            "passed": False,
            "credential_present": True,
            "http_status": exc.code,
            "provider_calls": 0,
            "error": "models_endpoint_http_error",
        }
    except Exception as exc:
        return {
            "passed": False,
            "credential_present": True,
            "provider_calls": 0,
            "error_type": type(exc).__name__,
        }
    ids = [
        row.get("id")
        for row in value.get("data", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    ]
    return {
        "passed": status == 200 and MODEL_ID in ids,
        "credential_present": True,
        "http_status": status,
        "configured_model_id": MODEL_ID,
        "model_list_contains_configured_id": MODEL_ID in ids,
        "provider_calls": 0,
        "request_kind": "GET /models metadata only",
    }


def timing_summary(rows):
    samples = [
        sample
        for row in rows
        for sample in row.get("defense_samples") or []
    ]
    result = {}
    for key in (
        "state_observer_ms",
        "feasibility_ms",
        "selector_ms",
        "response_construction_ms",
        "injection_or_pass_through_ms",
        "opportunity_total_ms",
        "gateway_opportunity_ms",
    ):
        values = sorted(
            float(sample[key])
            for sample in samples
            if sample.get(key) is not None
        )
        if values:
            index = min(len(values) - 1, int(0.95 * (len(values) - 1)))
            result[key] = {
                "samples": len(values),
                "mean": sum(values) / len(values),
                "median": statistics.median(values),
                "p95": values[index],
                "max": max(values),
            }
        else:
            result[key] = {
                "samples": 0,
                "mean": None,
                "median": None,
                "p95": None,
                "max": None,
            }
    return result


def run_preflight(output, config, schedule, repository, route_check):
    with patch.object(
        socket.socket,
        "connect",
        side_effect=AssertionError("network forbidden in preflight"),
    ):
        short_smokes = short_arm_smokes(config, schedule, repository)
        state_smokes = full_state_smokes(config, schedule, repository)
        forced = forced_infeasible_smokes(
            config, schedule[0], repository
        )
        no_trap = no_trap_unchanged_smoke(
            config, schedule[0], repository
        )
        budget_rows = [
            budget_probe(config, schedule[0], repository, budget)
            for budget in config["budgets"]
        ]
        adapter_budget_rows = [
            frozen.provider_budget_probe(config, budget)
            for budget in config["budgets"]
        ]
        restores = []
        for source in schedule:
            fixture = repository.load(source["fixture_id"])
            extension = repository.extension(source["fixture_id"])
            initial = ExtendedEvidenceV3Environment(
                fixture, extension
            ).snapshot()
            restored = ExtendedEvidenceV3Environment.restore(
                fixture, deepcopy(initial), extension
            ).snapshot()
            restores.append(restored == initial)
    route = ds.trusted_route_evidence(config)
    fixed_ok = all(forced["fixed_traps_never_fallback"].values())
    dynamic_opportunities = [
        opportunity
        for row in short_smokes + state_smokes
        if row["strategy"] == "STATE_AWARE_FEASIBLE"
        for opportunity in row["opportunities"]
    ]
    dynamic_checks_ok = all(
        [check["trap"] for check in opportunity["feasibility_checks"]]
        == list(EXPECTED_PRIORITIES[opportunity["injection_state"]])
        and len(opportunity["feasibility_checks"]) == 3
        for opportunity in dynamic_opportunities
    )
    timing_rows = [
        {
            "defense_samples": row.get("defense_samples") or [],
        }
        for row in short_smokes
    ]
    timing = timing_summary(timing_rows)
    checks = {
        "nine_sources_loaded": len(schedule) == 9,
        "source_order_fixed": [
            row["source_index"] for row in schedule
        ]
        == EXPECTED_SOURCE_ORDER,
        "family_balance_three_each": Counter(
            row["family"] for row in schedule
        )
        == Counter(config["family_counts"]),
        "all_nine_sources_restore_exactly": len(restores) == 9
        and all(restores),
        "ninety_slots_generated": len(short_smokes) == 90,
        "slot_ids_unique": len({
            row["slot_id"] for row in short_smokes
        })
        == 90,
        "five_way_initial_equal": all(
            row["initial_equal_within_five"] for row in short_smokes
        ),
        "short_smokes_exact_two_calls": all(
            row["provider_calls"] == 2 for row in short_smokes
        ),
        "all_five_strategies_ran": {
            row["strategy"] for row in short_smokes
        }
        == set(STRATEGIES),
        "three_states_observed_for_all_five": all(
            row["reached_states"] == list(STATES)
            for row in state_smokes
        ),
        "fixed_traps_never_fallback": fixed_ok,
        "forced_primary_falls_back_same_opportunity": forced[
            "forced_primary_falls_back_same_opportunity"
        ],
        "unknown_not_executed_and_not_consumed": forced[
            "unknown_not_executed_and_not_consumed"
        ],
        "dynamic_three_candidates_and_order": dynamic_checks_ok,
        "no_trap_two_calls": no_trap["provider_calls"] == 2,
        "no_trap_no_feedback": no_trap[
            "first_result_has_no_trap_feedback"
        ],
        "no_trap_opportunities_passive": no_trap[
            "all_opportunities_no_trap"
        ],
        "no_trap_oracle_recorded": no_trap["task_oracle_recorded"],
        "trap_versions_match_runtime": config["trap_versions"]
        == EXPECTED_TRAP_VERSIONS,
        "primary_comparator_fixed_in_config": config[
            "primary_fixed_comparator"
        ]
        == "FIXED_DEPENDENCY_V6",
        "b20_provider_budget_exact": budget_rows[0]["budget"] == 20
        and budget_rows[0]["provider_calls"] == 20
        and budget_rows[0]["budget_exhausted"]
        and budget_rows[0]["mock_requests"] == 20,
        "b30_provider_budget_exact": budget_rows[1]["budget"] == 30
        and budget_rows[1]["provider_calls"] == 30
        and budget_rows[1]["budget_exhausted"]
        and budget_rows[1]["mock_requests"] == 30,
        "request_ledger_b20_exact": adapter_budget_rows[0][
            "started_requests"
        ]
        == 20
        and adapter_budget_rows[0]["completed_requests"] == 20
        and adapter_budget_rows[0]["next_request_blocked"],
        "request_ledger_b30_exact": adapter_budget_rows[1][
            "started_requests"
        ]
        == 30
        and adapter_budget_rows[1]["completed_requests"] == 30
        and adapter_budget_rows[1]["next_request_blocked"],
        "usage_audit_no_unknown_requests": all(
            not row["unknown_usage_requests"]
            for row in adapter_budget_rows
        ),
        "timing_samples_nonnegative": all(
            stats["samples"] > 0 and stats["max"] is not None
            and stats["max"] >= 0
            for stats in timing.values()
        ),
        "oracle_unchanged": (
            ExtendedEvidenceV3Environment.oracle
            is MultiTaskEnvironment.oracle
            and MultiTaskDependencyV5Environment.oracle
            is MultiTaskEnvironment.oracle
        ),
        "credential_environment_present": route_check[
            "credential_present"
        ],
        "current_api_route_advertises_model": route_check["passed"],
        "trusted_exact_completion_route": route["passed"],
        "provider_completion_calls_zero": True,
    }
    result = {
        "at": now(),
        "version": VERSION,
        "checks": checks,
        "passed": all(checks.values()),
        "provider_calls": 0,
        "external_completion_calls": 0,
        "route_metadata_check": deepcopy(route_check),
        "trusted_route_evidence": route,
        "source_order": config["source_indices"],
        "strategy_order": list(STRATEGIES),
        "fixed_traps": deepcopy(FIXED_TRAPS),
        "priorities": {
            state: list(values) for state, values in EXPECTED_PRIORITIES.items()
        },
        "trajectory_slots": len(short_smokes),
        "pair_slots_all": 180,
        "primary_pair_slots": 36,
        "theoretical_maximum_provider_calls": 2250,
        "short_arm_smokes": short_smokes,
        "full_state_smokes": state_smokes,
        "forced_infeasibility": forced,
        "no_trap_smoke": no_trap,
        "budget_probes": budget_rows,
        "adapter_budget_probes": adapter_budget_rows,
        "source_restores": restores,
        "timing_summary": timing,
        "tool_interface": extended_evidence_interface(),
    }
    write_json(output / "provider_free_validation.json", result)
    write_json(
        output / "preflight_status.json",
        {
            "status": "GO" if result["passed"] else "NO_GO",
            "passed": result["passed"],
            "finished_at": now(),
            "provider_calls": 0,
        },
    )
    return result


def protected_metadata(output):
    result = ds.protected_metadata(output)
    for directory in PROTECTED_RESULT_DIRS:
        base = ROOT / "results" / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and output not in path.parents:
                result[str(path.relative_to(ROOT))] = metadata(path)
    return result


def verify_protected(before):
    differences = []
    for name, value in before.items():
        path = ROOT / name
        if not path.is_file():
            differences.append({"path": name, "issue": "missing"})
        elif metadata(path) != value:
            differences.append({"path": name, "issue": "metadata_changed"})
    return {"passed": not differences, "differences": differences}


def freeze_files(config_path):
    paths = (config_path, MODULE_DIR / "__init__.py", MODULE_DIR / "runner.py")
    for path in paths:
        path.chmod(0o444)
    return {
        "at": now(),
        "freeze_method": "version, read-only permission, timestamp, and modification record",
        "files": {
            str(path.relative_to(ROOT)): metadata(path) for path in paths
        },
    }


def verify_freeze(record):
    differences = []
    for name, value in record["files"].items():
        path = ROOT / name
        if not path.is_file():
            differences.append({"path": name, "issue": "missing"})
        elif metadata(path) != value:
            differences.append({"path": name, "issue": "metadata_changed"})
    return {"passed": not differences, "differences": differences}


def materialize_existing_schedule(output):
    audit = read_json(output / "environment_extension_audit.json")
    rows = [
        {
            "source_index": row["source_index"],
            "family": row["family"],
            "fixture_id": row["fixture_id"],
            "fixture_path": row["runtime_fixture_path"],
        }
        for row in audit["sources"]
    ]
    return rows, core.ExtendedEvidenceRepository(
        output / "runtime_fixtures"
    )


def technical_pair_valid(left, right, initial_equal):
    return (
        initial_equal
        and left.get("usage_complete")
        and right.get("usage_complete")
        and left.get("model_identity_verified")
        and right.get("model_identity_verified")
        and not left.get("technical_failure")
        and not right.get("technical_failure")
        and left.get("terminal_reason") != "PROTOCOL_VIOLATION"
        and right.get("terminal_reason") != "PROTOCOL_VIOLATION"
        and left.get("total_tokens") is not None
        and right.get("total_tokens") is not None
    )


def build_pairs(groups):
    result = []
    for (budget, source_index), rows in sorted(groups.items()):
        by_strategy = {row["strategy"]: row for row in rows}
        initial_equal = len({
            json.dumps(row["initial"], sort_keys=True)
            for row in rows
        }) == 1
        for left_name, right_name in combinations(STRATEGIES, 2):
            left = by_strategy[left_name]
            right = by_strategy[right_name]
            valid = technical_pair_valid(left, right, initial_equal)
            row = {
                "budget": budget,
                "source_index": source_index,
                "source_id": left["source_id"],
                "task_family": left["task_family"],
                "left_strategy": left_name,
                "right_strategy": right_name,
                "comparison_role": (
                    "primary_fixed"
                    if {
                        left_name,
                        right_name,
                    }
                    == {
                        "STATE_AWARE_FEASIBLE",
                        "FIXED_DEPENDENCY_V6",
                    }
                    else "supplementary"
                ),
                "initial_equal_within_five": initial_equal,
                "technical_pair_valid": valid,
                "both_task_success": bool(
                    left["task_success"] and right["task_success"]
                ),
                "left_success": bool(left["task_success"]),
                "right_success": bool(right["task_success"]),
                "left_budget_exhausted": bool(left["budget_exhausted"]),
                "right_budget_exhausted": bool(right["budget_exhausted"]),
                "left_terminal": left["terminal_reason"],
                "right_terminal": right["terminal_reason"],
                "terminal_combo": (
                    f"{left['terminal_reason']}/{right['terminal_reason']}"
                ),
            }
            for metric in (
                "total_tokens",
                "input_tokens",
                "output_tokens",
                "provider_calls",
                "effective_tool_calls",
                "wall_time_ms",
                "defense_gateway_total_ms",
            ):
                left_value = (
                    left.get("defense_totals", {}).get(
                        "defense_gateway_total_ms"
                    )
                    if metric == "defense_gateway_total_ms"
                    else left.get(metric)
                )
                right_value = (
                    right.get("defense_totals", {}).get(
                        "defense_gateway_total_ms"
                    )
                    if metric == "defense_gateway_total_ms"
                    else right.get(metric)
                )
                row[f"left_{metric}"] = left_value
                row[f"right_{metric}"] = right_value
                row[f"delta_{metric}"] = (
                    right_value - left_value
                    if valid
                    and left_value is not None
                    and right_value is not None
                    else None
                )
            result.append(row)
    return result


def write_trajectory_artifacts(destination, row, provider):
    write_json(destination / "result.json", row)
    write_json(destination / "request_metrics.json", row["request_metrics"])
    write_json(destination / "defense_timings.json", {
        "samples": row["defense_samples"],
        "totals": row["defense_totals"],
    })
    write_json(destination / "provider_records.json", {
        "raw": deepcopy(provider.raw_records),
        "request_log": deepcopy(provider.request_log),
        "budget": provider.deepseek_budget.ledger.snapshot(),
    })
    with (destination / "events.jsonl").open("x", encoding="utf-8") as stream:
        for event in row.get("events", []):
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_trajectory(
    output,
    config,
    source,
    budget,
    strategy,
    repository,
    monitor,
    batch_state,
):
    destination = (
        output
        / "real"
        / f"B{budget}"
        / strategy
        / f"source-{source['source_index']:02d}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    provider = create_real_provider(
        config,
        budget,
        destination / "requests.jsonl",
        monitor,
        batch_state,
    )
    session = fresh_session(provider, source, repository)
    runner = make_runner(
        config, session, budget, strategy, repository
    )
    write_json(destination / "initial.json", {
        "source": deepcopy(source),
        "strategy": strategy,
        "budget": budget,
        "initial": deepcopy(runner.initial),
        "config": runner.config.export(),
        "tool_interface": extended_evidence_interface(),
    })
    try:
        row = runner.run()
    except Exception as exc:
        runner.stop("TECHNICAL_FAILURE", f"{type(exc).__name__}: {exc}")
        row = runner.run() if False else {
            "source_id": source["fixture_id"],
            "task_family": source["family"],
            "strategy": strategy,
            "terminal_reason": "TECHNICAL_FAILURE",
            "task_oracle_result": runner.oracle(),
            "provider_calls": runner.budget.provider_calls,
            "interaction_depth": runner.budget.interaction_depth,
            "tool_calls": runner.tool_calls,
            **runner.budget.tokens(),
            "events": deepcopy(runner.events),
            "opportunities": deepcopy(runner.injections.records),
            "final_snapshot": session.env.snapshot(),
            "provider_usage": deepcopy(runner.budget.usage),
            "started_at": runner.started_at,
            "ended_at": now(),
            "wall_time": None,
            "wall_time_ms": None,
            "provider_api_wait_total_ms": sum(
                runner.provider_wait_samples
            ),
            "tool_execution_total_ms": sum(
                runner.tool_execution_samples
            ),
            "defense_totals": runner._defense_totals(),
            "defense_samples": deepcopy(runner.defense_samples),
            "request_metrics": runner._request_metrics(),
            "failure_preserved": True,
        }
    finally:
        session.close()
    if core.model_refusal(row):
        row["terminal_reason"] = "MODEL_REFUSAL"
    fields = ds.ledger_fields(provider, config)
    selected = {
        item["injection_state"]: item.get("selected_trap")
        for item in row.get("opportunities") or []
    }
    deployed = {
        item["injection_state"]: item.get("deployment_status") == "DEPLOYED"
        for item in row.get("opportunities") or []
    }
    row.update(
        source_index=source["source_index"],
        fixture_id=source["fixture_id"],
        source_id=source["fixture_id"],
        task_family=source["family"],
        strategy=strategy,
        budget=budget,
        budget_unit=config["budget_unit"],
        **{
            key: value
            for key, value in fields.items()
            if key != "usage_snapshot"
        },
        effective_tool_calls=sum(
            event.get("kind") == "tool_result"
            and event.get("status") == "ok"
            and (event.get("call") or {}).get("function", {}).get("name")
            != "stop_iteration"
            for event in row.get("events") or []
        ),
        budget_exhausted=row.get("terminal_reason") == "BUDGET_EXHAUSTED",
        technical_failure=row.get("terminal_reason") in FATAL_TERMINALS,
        protocol_violation=row.get("terminal_reason") == "PROTOCOL_VIOLATION",
        task_success=bool(
            row.get("task_oracle_result", {}).get("verified")
        ),
        reached_states=sorted(
            {
                event["observation"]["online_state"]
                for event in row.get("events") or []
                if event.get("kind") == "state_observation"
                and event.get("observation", {}).get("online_state")
                in STATES
            },
            key=STATES.index,
        ),
        selected_traps=selected,
        deployed_traps=deployed,
        number_of_fallbacks=sum(
            bool(item.get("fallback_used"))
            for item in row.get("opportunities") or []
        ),
        number_of_not_deployable=sum(
            item.get("deployment_status") == "NOT_DEPLOYABLE"
            for item in row.get("opportunities") or []
        ),
        number_of_unknown=sum(
            event.get("observation", {}).get("online_state") == "UNKNOWN"
            for event in row.get("events") or []
            if event.get("kind") == "state_observation"
        ),
        automatic_retries=0,
        protocol_version=config["protocol_id"],
        usage_snapshot=fields["usage_snapshot"],
    )
    write_trajectory_artifacts(destination, row, provider)
    return row, runner.initial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.start == args.preflight_only:
        parser.error("choose exactly one of --preflight-only or --start")
    config_path = Path(args.config).resolve() if args.config else CONFIG_PATH
    config = load_config(config_path)
    if args.preflight_only:
        schedule_seed = core.selected_sources(config)
        scheduler = scheduler_for(config).acquire()
        try:
            output = create_output(config)
            with ExperimentDirectoryLock(output, config["experiment_id"]):
                write_json(output / "config_snapshot.json", config)
                write_json(
                    output / "source_selection.json",
                    core.source_selection_record(config, schedule_seed),
                )
                schedule, repository = core.materialize_fixtures(
                    output, schedule_seed
                )
                route_check = current_route_check(config)
                write_json(output / "provider_route_check.json", route_check)
                result = run_preflight(
                    output, config, schedule, repository, route_check
                )
                if result["passed"]:
                    freeze = freeze_files(config_path)
                    write_json(output / "freeze_record.json", freeze)
                write_json(output / "launch_record.json", {
                    "at": now(),
                    "experiment_id": config["experiment_id"],
                    "protocol_id": config["protocol_id"],
                    "batch_id": config["batch_id"],
                    "source_count": len(schedule),
                    "strategies": list(STRATEGIES),
                    "budgets": config["budgets"],
                    "trajectories": 90,
                    "theoretical_maximum_provider_calls": 2250,
                    "provider_calls_during_preflight": 0,
                    "real_execution_started": False,
                })
                for name in (
                    "config_snapshot.json",
                    "source_selection.json",
                    "provider_route_check.json",
                    "provider_free_validation.json",
                    "preflight_status.json",
                    "launch_record.json",
                ):
                    (output / name).chmod(0o444)
                if result["passed"]:
                    (output / "freeze_record.json").chmod(0o444)
                print(json.dumps({
                    "status": "GO" if result["passed"] else "NO_GO",
                    "output": str(output),
                    "provider_calls": 0,
                }, ensure_ascii=False), flush=True)
                return 0 if result["passed"] else 2
        finally:
            scheduler.release()

    if not args.output:
        parser.error("--start requires preflight output")
    output = Path(args.output).resolve()
    preflight = read_json(output / "provider_free_validation.json")
    if not preflight.get("passed"):
        raise RuntimeError("Provider-free validation did not pass")
    if (output / "execution_started.json").exists():
        raise RuntimeError("execution already started; recovery is disabled")
    freeze = read_json(output / "freeze_record.json")
    freeze_check = verify_freeze(freeze)
    if not freeze_check["passed"]:
        raise RuntimeError("frozen implementation changed")
    selected = read_json(output / "source_selection.json")
    if [
        row["source_index"] for row in selected["sources"]
    ] != EXPECTED_SOURCE_ORDER:
        raise RuntimeError("source selection changed")
    protected_before = protected_metadata(output)
    core.install_process_memory_limit(
        config["maximum_process_address_space_bytes"]
    )
    scheduler = scheduler_for(config).acquire()
    try:
        with ExperimentDirectoryLock(output, config["experiment_id"]):
            schedule, repository = materialize_existing_schedule(output)
            write_json(
                output / "protected_metadata_before.json",
                protected_before,
            )
            write_json(output / "execution_started.json", {
                "at": now(),
                "experiment_id": config["experiment_id"],
                "actual_model": MODEL_ID,
                "trajectories_expected": 90,
                "provider_call_theoretical_maximum": 2250,
                "automatic_retries": 0,
                "automatic_recovery": False,
            })
            monitor = MemoryMonitor(
                output / "memory_samples.jsonl",
                warning_rss_bytes=config["memory_warning_rss_bytes"],
                hard_rss_bytes=config["memory_hard_rss_bytes"],
            )
            batch_state = BatchState(config)
            rows = []
            groups = defaultdict(list)
            stop_reason = None
            monitor.sample("execution_start")
            try:
                with runtime_memory_scope(monitor):
                    for budget in config["budgets"]:
                        for source_position, source in enumerate(schedule):
                            strategies = list(STRATEGIES)
                            shift = source_position % len(strategies)
                            strategies = (
                                strategies[shift:] + strategies[:shift]
                            )
                            for strategy in strategies:
                                print(json.dumps({
                                    "phase": "trajectory_start",
                                    "budget": budget,
                                    "source_index": source["source_index"],
                                    "strategy": strategy,
                                }, ensure_ascii=False), flush=True)
                                monitor.sample(
                                    "trajectory_before",
                                    budget=budget,
                                    source_index=source["source_index"],
                                    strategy=strategy,
                                )
                                try:
                                    row, initial = run_trajectory(
                                        output,
                                        config,
                                        source,
                                        budget,
                                        strategy,
                                        repository,
                                        monitor,
                                        batch_state,
                                    )
                                except Exception as exc:
                                    write_json(output / "batch_failure.json", {
                                        "at": now(),
                                        "budget": budget,
                                        "source_index": source["source_index"],
                                        "strategy": strategy,
                                        "error_type": type(exc).__name__,
                                        "error": str(exc)[:2000],
                                        "traceback": traceback.format_exc(),
                                    })
                                    stop_reason = "runtime_or_provider_failure"
                                    break
                                groups[(budget, source["source_index"])].append(
                                    row
                                )
                                write_json(output / "rows.json", rows + [row])
                                write_json(
                                    output / "partial_groups.json",
                                    {
                                        f"B{budget}-source-{source_index:02d}": [
                                            item["strategy"]
                                            for item in group
                                        ]
                                        for (budget, source_index), group
                                        in groups.items()
                                    },
                                )
                                append_jsonl(output / "progress.jsonl", {
                                    "at": now(),
                                    "status": "TRAJECTORY_TERMINAL",
                                    "budget": budget,
                                    "source_index": source["source_index"],
                                    "strategy": strategy,
                                    "terminal_reason": row["terminal_reason"],
                                    "provider_calls": row["provider_calls"],
                                    "total_tokens": row["total_tokens"],
                                    "task_success": row["task_success"],
                                    "wall_time_ms": row["wall_time_ms"],
                                })
                                write_json(output / "progress.json", {
                                    "at": now(),
                                    "completed_trajectories": sum(
                                        len(group) for group in groups.values()
                                    ),
                                    "expected_trajectories": 90,
                                    "latest_budget": budget,
                                    "latest_source_index": source["source_index"],
                                    "latest_strategy": strategy,
                                    "latest_terminal": row["terminal_reason"],
                                })
                                if (
                                    row["terminal_reason"] in FATAL_TERMINALS
                                    or not row["model_identity_verified"]
                                    or row["unknown_usage_requests"]
                                ):
                                    stop_reason = (
                                        row["terminal_reason"]
                                        if row["terminal_reason"]
                                        in FATAL_TERMINALS
                                        else "model_or_usage_audit_failure"
                                    )
                                    break
                            if stop_reason:
                                break
                        if stop_reason:
                            break
            finally:
                monitor.sample("execution_end")
                monitor.close()

            rows = [
                row for group in groups.values() for row in group
            ]
            rows.sort(
                key=lambda row: (
                    row["budget"],
                    EXPECTED_SOURCE_ORDER.index(row["source_index"]),
                    STRATEGIES.index(row["strategy"]),
                )
            )
            for group in groups.values():
                if len(group) != 5:
                    continue
                initial_equal = len({
                    json.dumps(row["initial"], sort_keys=True)
                    for row in group
                }) == 1
                for row in group:
                    row["initial_equal_within_five"] = initial_equal
            pairs = build_pairs(groups)
            primary_pairs = [
                row for row in pairs
                if row["comparison_role"] == "primary_fixed"
            ]
            write_json(output / "rows.json", rows)
            write_json(output / "pairs.json", pairs)
            write_json(output / "primary_pairs.json", primary_pairs)
            protected_check = verify_protected(protected_before)
            completed = len(rows)
            technical = sum(
                row["terminal_reason"] in FATAL_TERMINALS for row in rows
            )
            protocol = sum(
                row["terminal_reason"] == "PROTOCOL_VIOLATION"
                for row in rows
            )
            status = (
                "STOPPED"
                if stop_reason
                else (
                    "ALL90_COMPLETE"
                    if completed == 90
                    else "INCOMPLETE"
                )
            )
            execution_status = {
                "at": now(),
                "status": status,
                "attempted_trajectories": completed,
                "expected_trajectories": 90,
                "provider_calls": sum(
                    int(row["provider_calls"]) for row in rows
                ),
                "known_input_tokens": sum(
                    int(row["input_tokens"] or 0) for row in rows
                ),
                "known_output_tokens": sum(
                    int(row["output_tokens"] or 0) for row in rows
                ),
                "known_total_tokens": sum(
                    int(row["total_tokens"] or 0) for row in rows
                ),
                "technical_failures": technical,
                "protocol_violations": protocol,
                "budget_exhausted_trajectories": sum(
                    bool(row["budget_exhausted"]) for row in rows
                ),
                "all_pairs": len(pairs),
                "primary_pairs": len(primary_pairs),
                "technical_valid_pairs": sum(
                    bool(row["technical_pair_valid"]) for row in pairs
                ),
                "stop_reason": stop_reason,
                "automatic_retries": 0,
                "automatic_recovery": False,
                "protected_results_check": protected_check,
                "freeze_check": freeze_check,
                "batch_state": batch_state.snapshot(),
                "memory_peak": deepcopy(monitor.peak),
            }
            write_json(output / "execution_status.json", execution_status)
            print(json.dumps({
                "status": status,
                "attempted": completed,
                "provider_calls": execution_status["provider_calls"],
                "total_tokens": execution_status["known_total_tokens"],
                "pairs": len(pairs),
                "stop_reason": stop_reason,
                "output": str(output),
            }, ensure_ascii=False), flush=True)
            return 0 if status == "ALL90_COMPLETE" else 1
    finally:
        scheduler.release()


if __name__ == "__main__":
    raise SystemExit(main())

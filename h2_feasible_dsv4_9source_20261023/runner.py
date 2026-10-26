#!/usr/bin/env python3
"""H2 DeepSeek V4 Pro full trajectories with feasible State selection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import tempfile
from time import monotonic
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

import h2_v7_deepseek_v4_pro_branch_v3_20261022.runner as ds
import h2_v7_gpt_5_6_sol_20261022.runner as core
from adapters.evidence_new_v3_runtime import (
    EvidenceNewV3Runtime,
    VERSION as EVIDENCE_V3_RUNTIME_VERSION,
)
from adapters.followup_budget_sensitivity_v1 import StateBasedEvidence
from adapters.h1_memory_monitor import MemoryMonitor, runtime_memory_scope
from adapters.lifecycle_branch_new_v3_runtime import (
    LifecycleBranchNewV3Runtime,
)
from adapters.lifecycle_dependency_v6_runtime import (
    LifecycleDependencyV6Runtime,
)
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
    DeepseekAuditProvider,
    DeepseekBudget,
    DeepseekProviderError,
    validate_usage,
)
from h1_deepseek_v4_pro_extended_20261022.request_ledger import (
    DeepseekRequestLedger,
)
from h2.config import Config
from h2.trajectory_runner import TrajectoryRunner
from tasks.multi_task_dependency_v6_adapter import dual_dependency_context
from tasks.multi_task_pool_v1 import MultiTaskEnvironment
from tasks.multi_task_pool_v3_dependency_v5_adapter import (
    MultiTaskDependencyV5Environment,
)
from traps.task_relevant_branch_v3 import (
    BRANCH_NEW_V3_IMPLEMENTATION,
    BRANCH_NEW_V3_RULE_VERSION,
)


CONFIG_PATH = MODULE_DIR / "config.json"
VERSION = "h2-feasible-dsv4-9source-v1.0"
MODEL_ID = "deepseek-v4-pro"
STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
STRATEGIES = ("FIXED_DEP_V6", "STATE_AWARE_FEASIBLE")
TRAPS = ("DEPENDENCY_V6", "EVIDENCE_NEW_V3", "BRANCH_NEW_V3")
EXPECTED_SOURCE_ORDER = [1, 6, 2, 11, 16, 12, 21, 26, 22]
EXPECTED_FIXED = {
    "RECON": "DEPENDENCY_V6",
    "EXPLOITATION": "DEPENDENCY_V6",
    "VERIFICATION": "DEPENDENCY_V6",
}
EXPECTED_PRIORITIES = {
    "RECON": (
        "DEPENDENCY_V6",
        "BRANCH_NEW_V3",
        "EVIDENCE_NEW_V3",
    ),
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
    "EVIDENCE_NEW_V3": "EVIDENCE_NEW_V3_SCOPED_ASSOCIATION_V1",
    "BRANCH_NEW_V3": BRANCH_NEW_V3_IMPLEMENTATION,
}
FATAL_TERMINALS = {
    "TECHNICAL_FAILURE",
    "PROTOCOL_VIOLATION",
    "MODEL_REFUSAL",
}
PROTECTED_RESULT_DIRS = (
    "H2-V7-DeepSeek-V4-Pro_20261022T101128198251Z",
    "H2-V7-DeepSeek-V4-Pro-BranchNewV3_20261022T113535829098Z",
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


def metadata(path: Path) -> dict[str, Any]:
    value = path.stat()
    return {
        "size_bytes": value.st_size,
        "modified_at_ns": value.st_mtime_ns,
        "mode": oct(value.st_mode & 0o777),
    }


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = read_json(path)
    if config.get("config_version") != "h2-feasible-dsv4-9source-full-trajectory-v1":
        raise RuntimeError("unexpected feasible H2 config version")
    provider = config["provider"]
    if (
        provider["name"] != "zhizengzeng"
        or provider["interface"] != "openai-compatible-chat-completions-v1"
        or provider["base_url"] != "https://api.zhizengzeng.com/v1"
        or provider["model_id"] != MODEL_ID
        or provider["temperature"] != 0.0
        or provider["max_tokens"] != 8192
        or provider["timeout_seconds"] != 180
        or provider["tool_choice"] != "auto"
        or provider["parallel_tool_calls"] is not False
        or provider["stream"] is not False
        or provider["credential_environment"] != "DS_V4_H1_API_KEY"
        or provider["accepted_response_model_ids"] != [MODEL_ID]
    ):
        raise RuntimeError("DeepSeek V4 Pro provider configuration changed")
    if config["source_count"] != 9:
        raise RuntimeError("source count changed")
    if config["source_indices"] != EXPECTED_SOURCE_ORDER:
        raise RuntimeError("fixed nine-source order changed")
    if config["family_counts"] != {"WEB": 3, "API": 3, "SERVICE": 3}:
        raise RuntimeError("task family balance changed")
    if config["budgets"] != [20, 30]:
        raise RuntimeError("H2 budgets changed")
    if config["strategies"] != list(STRATEGIES):
        raise RuntimeError("strategy set changed")
    if config["mapping"] != {"FIXED_DEP_V6": EXPECTED_FIXED}:
        raise RuntimeError("fixed strategy mapping changed")
    expected_priorities = {
        "STATE_AWARE_FEASIBLE": {
            state: list(traps)
            for state, traps in EXPECTED_PRIORITIES.items()
        }
    }
    if config["priorities"] != expected_priorities:
        raise RuntimeError("feasible priority mapping changed")
    if config["trajectory_count"] != 36 or config["pair_count"] != 18:
        raise RuntimeError("experiment scale changed")
    if config["theoretical_maximum_provider_calls"] != 900:
        raise RuntimeError("provider-call envelope changed")
    if config["batch_maximum_requests"] != 900:
        raise RuntimeError("batch request limit changed")
    if config["exposure_contract"] != "pre-effect-v1":
        raise RuntimeError("exposure boundary changed")
    if config["technical_retries"] != 0 or config["automatic_recovery"]:
        raise RuntimeError("automatic retry or recovery is forbidden")
    if "maximum_followups_per_trap" in config:
        raise RuntimeError("this H2 batch must not inherit an H1 follow-up K")
    return config


def identity(config: Mapping[str, Any]):
    return core.ExperimentIdentity.from_config(
        config, fallback_experiment_id=MODULE_DIR.name
    )


def scheduler_for(config: Mapping[str, Any]) -> ExperimentSchedulerLock:
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


def create_output(config: Mapping[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results" / f"{config['output_prefix']}{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def mock_provider(config: Mapping[str, Any]):
    return ds.mock_provider(config)


def fresh_session(provider, source, repository):
    return ExtendedEvidenceV3Session(
        provider, source["fixture_id"], repository=repository
    )


def constructor_provider(session):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("feasibility checking cannot request a model")

    return SimpleNamespace(
        model=session.provider.model,
        config=deepcopy(session.provider.config),
        interface=lifecycle_interaction_interface(),
        provider_free=True,
        complete=forbidden,
        complete_with_usage=forbidden,
    )


def restore_carrier(
    checkpoint: Mapping[str, Any],
    session,
    repository,
    trap: str,
    attach_session=None,
):
    before = checkpoint["environment_snapshot"]
    remaining = max(1, int(session.budget_limit - session.provider_calls))
    provider = constructor_provider(session)
    version = "V6" if trap == "DEPENDENCY_V6" else "V5"
    with dual_dependency_context(before, repository, version=version):
        if trap == "DEPENDENCY_V6":
            carrier = LifecycleDependencyV6Runtime.restore(
                checkpoint, provider
            )
            carrier.core.max_optional_followups = remaining
        elif trap == "EVIDENCE_NEW_V3":
            carrier = EvidenceNewV3Runtime.restore(
                checkpoint, provider, condition="EVIDENCE"
            )
            carrier.pilot_k = remaining
            carrier.evidence_v3 = StateBasedEvidence(checkpoint, remaining)
        elif trap == "BRANCH_NEW_V3":
            carrier = LifecycleBranchNewV3Runtime.restore(
                checkpoint, provider
            )
            carrier.pilot_k = remaining
        else:
            raise ValueError(f"unsupported Trap: {trap}")
    try:
        carrier.session.close()
    except Exception:
        pass
    if attach_session is not None:
        carrier.session = attach_session
    return carrier


def implementation_for(trap: str, carrier) -> str:
    if trap == "DEPENDENCY_V6":
        return getattr(
            carrier,
            "dependency_profile_version",
            EXPECTED_TRAP_VERSIONS[trap],
        )
    if trap == "EVIDENCE_NEW_V3":
        return EVIDENCE_V3_RUNTIME_VERSION
    if trap == "BRANCH_NEW_V3":
        return BRANCH_NEW_V3_IMPLEMENTATION
    raise ValueError(f"unsupported Trap: {trap}")


class FeasibleInjectionController(core.V7BranchV3InjectionController):
    """Consume one State opportunity, probe all candidates, then select."""

    def __init__(self, strategy, budget, repository, mapping=None):
        base_strategy = (
            strategy if strategy == "FIXED_DEP_V6" else "FIXED_DEP_V6"
        )
        super().__init__(
            base_strategy,
            budget,
            repository,
            mapping
            or {
                "FIXED_DEP_V6": deepcopy(EXPECTED_FIXED),
            },
        )
        self.strategy = strategy

    def opportunity(self, session, observation):
        if self.strategy == "FIXED_DEP_V6":
            row = super().opportunity(session, observation)
            if row is not None:
                row["selection_basis"] = "fixed_strategy"
                row["priority_order"] = ["DEPENDENCY_V6"]
                row["candidate_feasibility"] = {
                    "mode": "FIXED_NO_CANDIDATE_PROBE",
                    "checks": [],
                }
                row["fallback_used"] = False
                row["fallback_count"] = 0
            return row
        if self.strategy != "STATE_AWARE_FEASIBLE":
            raise ValueError("unknown feasible experiment strategy")

        state = observation.get("online_state")
        if (
            observation.get("candidate_boundary") is not True
            or state not in STATES
            or state in self.used
        ):
            return None

        self.used.add(state)
        before_environment = deepcopy(session.env.snapshot())
        before_pending = deepcopy(session.pending)
        before_provider_calls = int(session.provider_calls)
        before_log_count = len(getattr(session.provider, "request_log", []))
        checkpoint = deepcopy(session.payload())
        checkpoint.update(
            state=state,
            checkpoint_id=(
                f"h2-feasible-{state.casefold()}-"
                f"{session.provider_calls}"
            ),
            trap_injected=False,
            exposure_contract="pre-effect-v1",
        )
        priorities = list(EXPECTED_PRIORITIES[state])
        checks = []
        probe_started = monotonic()
        for probe_index, trap in enumerate(priorities, start=1):
            candidate_started = monotonic()
            carrier = None
            probe_session = None
            eligible = False
            reason = None
            binding_evidence = None
            try:
                probe_session = ExtendedEvidenceV3Session.restore(
                    deepcopy(checkpoint),
                    constructor_provider(session),
                    repository=self.repository,
                )
                carrier = restore_carrier(
                    checkpoint,
                    session,
                    self.repository,
                    trap,
                    attach_session=probe_session,
                )
                # Exercise the complete pre-effect deployment on an isolated
                # session restored from the same checkpoint.
                carrier.h2_start_call = int(session.provider_calls)
                carrier.initial_exposure()
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
            environment_unchanged = session.env.snapshot() == before_environment
            pending_unchanged = session.pending == before_pending
            provider_unchanged = (
                int(session.provider_calls) == before_provider_calls
                and len(getattr(session.provider, "request_log", []))
                == before_log_count
            )
            checks.append({
                "probe_index": probe_index,
                "trap": trap,
                "eligible": eligible,
                "reason": reason,
                "binding_evidence": binding_evidence,
                "elapsed_seconds": monotonic() - candidate_started,
                "environment_unchanged": environment_unchanged,
                "pending_unchanged": pending_unchanged,
                "provider_calls_unchanged": provider_unchanged,
            })
            if not (
                environment_unchanged
                and pending_unchanged
                and provider_unchanged
            ):
                raise RuntimeError(
                    "candidate feasibility probe changed the active boundary"
                )

        selected_index = next(
            (
                index
                for index, check in enumerate(checks)
                if check["eligible"]
            ),
            None,
        )
        selected = (
            priorities[selected_index]
            if selected_index is not None
            else None
        )
        row = {
            "injection_state": state,
            "selected_trap": selected,
            "selection_basis": "all_candidate_feasibility_then_priority",
            "deployment_status": (
                "PENDING" if selected is not None else "NOT_DEPLOYABLE"
            ),
            "provider_call": int(session.provider_calls),
            "call": deepcopy(session.pending),
            "priority_order": priorities,
            "candidate_feasibility": {
                "mode": "ALL_THREE_ISOLATED_PRE_EFFECT_DRYRUN",
                "checks": deepcopy(checks),
            },
            "feasibility_checks": deepcopy(checks),
            "fallback_used": selected_index is not None and selected_index > 0,
            "fallback_count": (
                selected_index
                if selected_index is not None
                else len(checks)
            ),
            "check_overhead_seconds": monotonic() - probe_started,
            "remaining_budget": int(session.budget_limit - session.provider_calls),
        }
        self.records.append(row)
        if selected is None:
            row["reason"] = "NO_FEASIBLE_TRAP"
            return row

        carrier = None
        deployed_at = monotonic()
        try:
            carrier = restore_carrier(
                checkpoint, session, self.repository, selected
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
            )
            if session.env.snapshot() != before_environment:
                raise RuntimeError(
                    "failed selected deployment changed the environment"
                )
            return row
        if (
            session.env.snapshot() != before_environment
            or session.pending is not None
        ):
            raise RuntimeError("selected Deployment violated pre-effect-v1")
        if not carrier.envelope.exposed:
            raise RuntimeError("selected Trap did not expose exactly once")
        self.active.append(carrier)
        row.update(
            deployment_status="DEPLOYED",
            mechanism_implementation=implementation_for(selected, carrier),
            deployment_overhead_seconds=monotonic() - deployed_at,
            feedback=deepcopy(session.messages[-1]),
        )
        if selected == "BRANCH_NEW_V3":
            row["candidate_audit"] = deepcopy(carrier.branch_v3_audit)
        return row


class FeasibleTrajectoryRunner(core.V7BranchV3TrajectoryRunner):
    def __init__(self, session, config, strategy, repository):
        placeholder = "FIXED_DEP_V6"
        mapping = {
            "FIXED_DEP_V6": deepcopy(EXPECTED_FIXED),
        }
        super().__init__(session, config, placeholder, repository, mapping)
        self.strategy = strategy
        self.analysis_strategy = strategy
        self.injections = FeasibleInjectionController(
            strategy, config.budget, repository
        )

    def run(self):
        row = TrajectoryRunner.run(self)
        row["experiment_version"] = VERSION
        row["experiment_phase"] = "INDEPENDENT_FULL_TRAJECTORY"
        row["analysis_strategy"] = self.strategy
        row["mapping"] = (
            deepcopy(EXPECTED_FIXED)
            if self.strategy == "FIXED_DEP_V6"
            else {
                state: list(traps)
                for state, traps in EXPECTED_PRIORITIES.items()
            }
        )
        row["diagnostic_audit_seconds"] = 0.0
        mechanisms = []
        for carrier in self.injections.active:
            trap = next(
                (
                    item.get("selected_trap")
                    for item in reversed(self.injections.records)
                    if item.get("mechanism_implementation")
                    == implementation_for(
                        "DEPENDENCY_V6", carrier
                    )
                    if carrier.condition == "DEPENDENCY"
                ),
                None,
            )
            item = {
                "condition": carrier.condition,
                "implementation": implementation_for(
                    "DEPENDENCY_V6"
                    if carrier.condition == "DEPENDENCY"
                    else (
                        "EVIDENCE_NEW_V3"
                        if carrier.condition == "EVIDENCE"
                        else "BRANCH_NEW_V3"
                    ),
                    carrier,
                ),
                "h2_start_call": carrier.h2_start_call,
                "followups": carrier.envelope.followups,
                "tool_calls_after_exposure": carrier.envelope.tool_calls,
                "initial_exposure_delivered": carrier.envelope.exposed,
                "terminal_reason": carrier.envelope.terminal_reason,
                "lifecycle_event_count": len(carrier.lifecycle_events),
            }
            if trap is not None:
                item["selected_trap"] = trap
            if carrier.condition == "DEPENDENCY":
                item.update(
                    chain_depth=carrier.core.chain_depth,
                    guard_resolved=carrier.core.blocking_guard_resolution,
                    termination_reason=carrier.core.termination_reason,
                )
            if carrier.condition == "EVIDENCE":
                item["evidence_metrics"] = carrier.evidence_v3.metrics()
                item["evidence_progress"] = [
                    deepcopy(event)
                    for event in carrier.lifecycle_events
                    if event.get("kind") == "evidence_v3_progress"
                ]
            if getattr(carrier, "branch_v3_audit", None):
                item.update(
                    selection_rule_version=BRANCH_NEW_V3_RULE_VERSION,
                    candidate_exposed=[
                        candidate["candidate_id"]
                        for candidate in carrier.candidates
                    ],
                    independent_direction_count=carrier.branch_v3_audit[
                        "independent_direction_count"
                    ],
                    task_relevance_status=carrier.branch_v3_audit["status"],
                    candidate_generation_funnel=deepcopy(
                        carrier.branch_v3_audit["funnel"]
                    ),
                )
            mechanisms.append(item)
        row["mechanism_metrics"] = mechanisms
        row["source_provenance"] = {
            "source_id": self.session.fixture["fixture_id"],
            "classification": "FROZEN_H2_NINE_SOURCE",
            "historical_outcome_used_for_selection": False,
        }
        return row


def make_runner(
    config: Mapping[str, Any],
    session,
    budget: int,
    strategy: str,
    repository,
):
    session.budget_limit = int(budget)
    return FeasibleTrajectoryRunner(
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
                session.budget_limit = int(budget)
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
                    "initial_equal_within_pair": initial == prior,
                    "provider_calls": row["provider_calls"],
                    "terminal_reason": row["terminal_reason"],
                    "deployments": row["number_of_injections"],
                    "opportunities": deepcopy(row["opportunities"]),
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
            provider.responses = core.scripted_state_path(
                runner, strategy, fixture, extension, session
            )
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


def feasibility_logic_selftest(config, source, repository):
    module = sys.modules[__name__]
    original = restore_carrier

    def run_with_policy(policy):
        provider = mock_provider(config)
        provider.responses = [
            core.action("collect-routes", scope="all"),
            {"role": "assistant", "content": "Provider-free stop."},
        ]
        session = fresh_session(provider, source, repository)
        session.budget_limit = config["budgets"][0]

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
                module, "restore_carrier", side_effect=controlled
            ):
                return make_runner(
                    config,
                    session,
                    config["budgets"][0],
                    "STATE_AWARE_FEASIBLE",
                    repository,
                ).run()
        finally:
            session.close()

    fallback_row = run_with_policy(
        lambda trap: trap != EXPECTED_PRIORITIES["RECON"][0]
    )
    fallback_opportunities = fallback_row.get("opportunities") or []
    fallback = (
        fallback_opportunities[0]
        if fallback_opportunities
        else {}
    )
    infeasible_row = run_with_policy(lambda _trap: False)
    infeasible_opportunities = (
        infeasible_row.get("opportunities") or []
    )
    infeasible = (
        infeasible_opportunities[0]
        if infeasible_opportunities
        else {}
    )
    controller = FeasibleInjectionController(
        "STATE_AWARE_FEASIBLE",
        config["budgets"][0],
        repository,
    )
    unknown_result = controller.opportunity(
        SimpleNamespace(provider_calls=0),
        {
            "online_state": "UNKNOWN",
            "candidate_boundary": True,
        },
    )
    return {
        "forced_primary_ineligible_falls_back_same_opportunity": (
            fallback.get("selected_trap")
            == EXPECTED_PRIORITIES["RECON"][1]
            and fallback.get("fallback_used") is True
            and fallback.get("fallback_count") == 1
            and len(fallback.get("feasibility_checks") or []) == 3
            and fallback.get("deployment_status") == "DEPLOYED"
        ),
        "all_three_ineligible_single_not_deployable": (
            len(infeasible_opportunities) == 1
            and infeasible.get("deployment_status") == "NOT_DEPLOYABLE"
            and infeasible.get("selected_trap") is None
            and infeasible.get("reason") == "NO_FEASIBLE_TRAP"
            and len(infeasible.get("feasibility_checks") or []) == 3
        ),
        "unknown_not_executed_and_not_consumed": (
            unknown_result is None and not controller.used
        ),
    }


def provider_budget_probe(config, budget: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="h2-feasible-budget-probe-"
    ) as directory:
        ledger = DeepseekRequestLedger(Path(directory) / "requests.jsonl")
        adapter_budget = DeepseekBudget(
            maximum_requests=budget,
            maximum_total_tokens=config["per_trajectory_max_total_tokens"],
            estimated_input_token_divisor=config[
                "estimated_input_token_divisor"
            ],
            rates={
                "input_cny_per_million": config["cost_estimation"][
                    "rates_cny_per_million_tokens"
                ]["input"],
                "cached_input_cny_per_million": config[
                    "cost_estimation"
                ]["rates_cny_per_million_tokens"]["cached_input"],
                "cache_write_cny_per_million": config["cost_estimation"][
                    "rates_cny_per_million_tokens"
                ]["cache_write"],
                "output_cny_per_million": config["cost_estimation"][
                    "rates_cny_per_million_tokens"
                ]["output"],
            },
            ledger=ledger,
        )
        body = {"model": MODEL_ID, "max_tokens": 1}
        messages = [{"role": "user", "content": "budget probe"}]
        for _ in range(budget):
            adapter_budget.before_request(messages, body)
            raw = {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }
            normalized = validate_usage(raw)
            adapter_budget.record_usage(
                normalized, {"usage": raw, "response_model_id": MODEL_ID}
            )
        blocked = False
        try:
            adapter_budget.before_request(messages, body)
        except DeepseekProviderError:
            blocked = True
        snapshot = ledger.snapshot()
        return {
            "budget": budget,
            "started_requests": snapshot["requests_started"],
            "completed_requests": snapshot["requests_with_usage"],
            "next_request_blocked": blocked,
            "token_total": snapshot["usage"]["total_tokens"],
            "usage_total_reconciled": snapshot["usage"]["total_tokens"]
            == budget * 2,
            "unknown_usage_requests": snapshot["unknown_request_ids"],
        }


def current_route_check(config: Mapping[str, Any]) -> dict[str, Any]:
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
        "actual_completion_identity_check": (
            "Every completion is checked against accepted response model IDs."
        ),
    }


def feasible_checks(config, smokes) -> dict[str, Any]:
    feasible_rows = [
        row
        for row in smokes
        if row["strategy"] == "STATE_AWARE_FEASIBLE"
    ]
    opportunities = [
        item
        for row in feasible_rows
        for item in row["opportunities"]
    ]
    exact_three = bool(opportunities) and all(
        len(item.get("feasibility_checks") or []) == 3
        for item in opportunities
    )
    order_ok = bool(opportunities) and all(
        [
            check["trap"]
            for check in item["feasibility_checks"]
        ]
        == list(EXPECTED_PRIORITIES[item["injection_state"]])
        for item in opportunities
    )
    side_effect_free = bool(opportunities) and all(
        check["environment_unchanged"]
        and check["pending_unchanged"]
        and check["provider_calls_unchanged"]
        for item in opportunities
        for check in item["feasibility_checks"]
    )
    selections_ok = bool(opportunities) and all(
        item["selected_trap"]
        == next(
            (
                check["trap"]
                for check in item["feasibility_checks"]
                if check["eligible"]
            ),
            None,
        )
        for item in opportunities
    )
    once_per_state = all(
        len({item["injection_state"] for item in row["opportunities"]})
        == len(row["opportunities"])
        for row in feasible_rows
    )
    fixed_rows = [
        row for row in smokes if row["strategy"] == "FIXED_DEP_V6"
    ]
    fixed_no_switch = all(
        item["selected_trap"] == "DEPENDENCY_V6"
        and item["candidate_feasibility"]["mode"]
        == "FIXED_NO_CANDIDATE_PROBE"
        for row in fixed_rows
        for item in row["opportunities"]
    )
    return {
        "all_feasible_opportunities_probe_three": exact_three,
        "feasibility_priority_order_exact": order_ok,
        "feasibility_probes_side_effect_free": side_effect_free,
        "selected_is_first_eligible": selections_ok,
        "at_most_once_per_state": once_per_state,
        "fixed_dependency_never_switches": fixed_no_switch,
        "opportunity_count": len(opportunities),
    }


def run_preflight(
    output: Path,
    config: Mapping[str, Any],
    schedule: list[dict[str, Any]],
    repository,
    route_check: Mapping[str, Any],
) -> dict[str, Any]:
    with patch.object(
        socket.socket,
        "connect",
        side_effect=AssertionError("network forbidden in Provider-free gate"),
    ):
        short_smokes = short_arm_smokes(config, schedule, repository)
        state_smokes = full_state_smokes(config, schedule, repository)
        core_budget_rows = [
            core.budget_probe(config, schedule[0], repository, budget)
            for budget in config["budgets"]
        ]
        adapter_budget_rows = [
            provider_budget_probe(config, budget)
            for budget in config["budgets"]
        ]
        feasibility_logic = feasibility_logic_selftest(
            config, schedule[0], repository
        )
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
            restores.append({
                "source_index": source["source_index"],
                "fixture_id": source["fixture_id"],
                "exact": restored == initial,
            })

    route = ds.trusted_route_evidence(config)
    family_counts = Counter(row["family"] for row in schedule)
    feasible = feasible_checks(config, short_smokes + state_smokes)
    actual_trap_versions = {
        "DEPENDENCY_V6": config["trap_versions"]["DEPENDENCY_V6"],
        "EVIDENCE_NEW_V3": EVIDENCE_V3_RUNTIME_VERSION,
        "BRANCH_NEW_V3": BRANCH_NEW_V3_IMPLEMENTATION,
    }
    checks = {
        "nine_sources_loaded": len(schedule) == 9,
        "source_order_fixed": [
            row["source_index"] for row in schedule
        ]
        == EXPECTED_SOURCE_ORDER,
        "family_balance_three_each": family_counts
        == Counter(config["family_counts"]),
        "all_nine_sources_restore_exactly": len(restores) == 9
        and all(row["exact"] for row in restores),
        "thirty_six_slots_generated": len(short_smokes) == 36,
        "slot_ids_unique": len({
            row["slot_id"] for row in short_smokes
        })
        == 36,
        "pair_initial_equal": all(
            row["initial_equal_within_pair"] for row in short_smokes
        ),
        "short_smokes_exact_two_calls": all(
            row["provider_calls"] == 2 for row in short_smokes
        ),
        "three_states_observed_in_family_smokes": all(
            row["reached_states"] == list(STATES)
            for row in state_smokes
        ),
        "trap_versions_match_runtime": actual_trap_versions
        == EXPECTED_TRAP_VERSIONS,
        "fixed_strategy_exact": config["mapping"]["FIXED_DEP_V6"]
        == EXPECTED_FIXED,
        "exploitation_priority_exact": tuple(
            config["priorities"]["STATE_AWARE_FEASIBLE"][
                "EXPLOITATION"
            ]
        )
        == (
            "EVIDENCE_NEW_V3",
            "DEPENDENCY_V6",
            "BRANCH_NEW_V3",
        ),
        **feasible,
        **feasibility_logic,
        "b20_provider_budget_exact": core_budget_rows[0]["budget"] == 20
        and core_budget_rows[0]["provider_calls"] == 20
        and core_budget_rows[0]["budget_exhausted"],
        "b30_provider_budget_exact": core_budget_rows[1]["budget"] == 30
        and core_budget_rows[1]["provider_calls"] == 30
        and core_budget_rows[1]["budget_exhausted"],
        "request_ledger_b20_exact": adapter_budget_rows[0][
            "started_requests"
        ]
        == 20
        and adapter_budget_rows[0]["completed_requests"] == 20
        and adapter_budget_rows[0]["next_request_blocked"]
        and adapter_budget_rows[0]["usage_total_reconciled"],
        "request_ledger_b30_exact": adapter_budget_rows[1][
            "started_requests"
        ]
        == 30
        and adapter_budget_rows[1]["completed_requests"] == 30
        and adapter_budget_rows[1]["next_request_blocked"]
        and adapter_budget_rows[1]["usage_total_reconciled"],
        "usage_audit_has_no_unknown_requests": all(
            not row["unknown_usage_requests"]
            for row in adapter_budget_rows
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
        "trusted_exact_completion_route_evidence": route["passed"],
        "provider_completion_calls_zero": True,
        "no_calls_after_budget_terminal": all(
            row["mock_requests"] == row["budget"]
            for row in core_budget_rows
        ),
    }
    result = {
        "at": now(),
        "version": VERSION,
        "protocol_id": config["protocol_id"],
        "checks": checks,
        "passed": all(checks.values()),
        "provider_calls": 0,
        "external_completion_calls": 0,
        "route_metadata_check": deepcopy(route_check),
        "trusted_route_evidence": route,
        "source_count": len(schedule),
        "source_order": config["source_indices"],
        "family_counts": dict(family_counts),
        "trajectory_slots": len(short_smokes),
        "pair_slots": len(short_smokes) // 2,
        "strategy_definition": {
            "FIXED_DEP_V6": deepcopy(EXPECTED_FIXED),
            "STATE_AWARE_FEASIBLE": {
                state: list(traps)
                for state, traps in EXPECTED_PRIORITIES.items()
            },
        },
        "trap_versions": actual_trap_versions,
        "exposure_contract": config["exposure_contract"],
        "budget_unit": config["budget_unit"],
        "theoretical_maximum_provider_calls": config[
            "theoretical_maximum_provider_calls"
        ],
        "source_restores": restores,
        "short_arm_smokes": short_smokes,
        "full_state_smokes": state_smokes,
        "core_budget_probes": core_budget_rows,
        "adapter_budget_probes": adapter_budget_rows,
        "feasible_checks": feasible,
        "feasibility_logic_selftest": feasibility_logic,
        "oracle": "tasks.multi_task_pool_v1.MultiTaskEnvironment.oracle",
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
            "frozen_before_real_provider_calls": True,
        },
    )
    return result


def freeze_files(output: Path, config_path: Path) -> dict[str, Any]:
    paths = (
        config_path,
        MODULE_DIR / "__init__.py",
        MODULE_DIR / "runner.py",
        MODULE_DIR / "analyze.py",
        MODULE_DIR / "verify.py",
    )
    for path in paths:
        if path.exists():
            path.chmod(0o444)
    rows = {
        str(path.relative_to(ROOT)): metadata(path)
        for path in paths
        if path.exists()
    }
    return {
        "at": now(),
        "freeze_method": "version, read-only permission, timestamp, and modification record",
        "files": rows,
    }


def verify_freeze(before: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    differences = []
    for name, value in before["files"].items():
        path = ROOT / name
        if not path.is_file():
            differences.append({"path": name, "issue": "missing"})
        else:
            current = metadata(path)
            if current != value:
                differences.append({
                    "path": name,
                    "issue": "metadata_changed",
                    "before": value,
                    "after": current,
                })
    return {"passed": not differences, "differences": differences}


def protected_metadata(output: Path) -> dict[str, Any]:
    result = ds.protected_metadata(output)
    for directory in PROTECTED_RESULT_DIRS:
        base = ROOT / "results" / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and output not in path.parents:
                result[str(path.relative_to(ROOT))] = metadata(path)
    return result


def verify_protected(before: Mapping[str, Any]) -> dict[str, Any]:
    differences = []
    for name, value in before.items():
        path = ROOT / name
        if not path.is_file():
            differences.append({"path": name, "issue": "missing"})
        elif metadata(path) != value:
            differences.append({"path": name, "issue": "metadata_changed"})
    return {"passed": not differences, "differences": differences}


def selected_initial(output: Path) -> dict[str, Any]:
    selection = read_json(output / "source_selection.json")
    return {
        "experiment_id": selection["experiment_id"],
        "source_indices": [row["source_index"] for row in selection["sources"]],
        "provider_free": True,
    }


def materialize_existing_schedule(output: Path):
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


def main() -> int:
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

    # Rebind inherited execution functions to this independent experiment.
    core.MODEL_ID = MODEL_ID
    core.CONFIG_PATH = config_path
    core.load_config = load_config
    core.scheduler_for = scheduler_for
    core.selected_sources = core.selected_sources
    core.create_output = create_output
    core.mock_provider = mock_provider
    core.fresh_session = fresh_session
    core.make_runner = make_runner
    core.create_real_provider = ds.create_real_provider
    core.ledger_fields = ds.ledger_fields
    core.write_trajectory_artifacts = ds.write_trajectory_artifacts
    core.trusted_route_evidence = ds.trusted_route_evidence
    core.protected_metadata = protected_metadata

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
                    freeze = freeze_files(output, config_path)
                    write_json(output / "freeze_record.json", freeze)
                write_json(
                    output / "launch_record.json",
                    {
                        "at": now(),
                        "experiment_id": config["experiment_id"],
                        "protocol_id": config["protocol_id"],
                        "batch_id": config["batch_id"],
                        "source_selection": config["source_indices"],
                        "strategies": config["strategies"],
                        "budgets": config["budgets"],
                        "trajectories": config["trajectory_count"],
                        "pairs": config["pair_count"],
                        "provider_completion_calls": 0,
                        "route_metadata_request": 1,
                        "real_execution_started": False,
                    },
                )
                protected_names = [
                    "config_snapshot.json",
                    "source_selection.json",
                    "provider_route_check.json",
                    "provider_free_validation.json",
                    "preflight_status.json",
                    "launch_record.json",
                ]
                if result["passed"]:
                    protected_names.append("freeze_record.json")
                for name in protected_names:
                    (output / name).chmod(0o444)
                print(json.dumps({
                    "status": "GO" if result["passed"] else "NO_GO",
                    "output": str(output),
                    "provider_calls": 0,
                }, ensure_ascii=False), flush=True)
                return 0 if result["passed"] else 2
        finally:
            scheduler.release()

    if not args.output:
        parser.error("--start requires the passed preflight output")
    output = Path(args.output).resolve()
    preflight = read_json(output / "provider_free_validation.json")
    if not preflight.get("passed"):
        raise RuntimeError("Provider-free validation did not pass")
    if (output / "execution_started.json").exists():
        raise RuntimeError(
            "execution marker exists; automatic recovery is disabled"
        )
    freeze = read_json(output / "freeze_record.json")
    freeze_check = verify_freeze(freeze, config_path)
    if not freeze_check["passed"]:
        raise RuntimeError("frozen implementation changed before real execution")
    selected = read_json(output / "source_selection.json")
    if [
        row["source_index"] for row in selected["sources"]
    ] != config["source_indices"]:
        raise RuntimeError("preflight source selection changed")
    protected_before = protected_metadata(output)
    core.install_process_memory_limit(
        config["maximum_process_address_space_bytes"]
    )
    scheduler = scheduler_for(config).acquire()
    try:
        with ExperimentDirectoryLock(output, config["experiment_id"]):
            schedule, repository = materialize_existing_schedule(output)
            write_json(
                output / "protected_metadata_before.json", protected_before
            )
            write_json(output / "execution_started.json", {
                "at": now(),
                "experiment_id": config["experiment_id"],
                "actual_model": MODEL_ID,
                "trajectories_expected": config["trajectory_count"],
                "pairs_expected": config["pair_count"],
                "automatic_retries": 0,
                "automatic_recovery": False,
            })
            monitor = MemoryMonitor(
                output / "memory_samples.jsonl",
                warning_rss_bytes=config["memory_warning_rss_bytes"],
                hard_rss_bytes=config["memory_hard_rss_bytes"],
            )
            rows = []
            pair_initials = {}
            pair_rows = defaultdict(list)
            stop_reason = None
            monitor.sample("execution_start")
            try:
                with runtime_memory_scope(monitor):
                    for budget in config["budgets"]:
                        for source_position, source in enumerate(schedule):
                            strategies = list(config["strategies"])
                            if source_position % 2:
                                strategies.reverse()
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
                                    row, initial = core.run_trajectory(
                                        output,
                                        config,
                                        source,
                                        budget,
                                        strategy,
                                        repository,
                                        monitor,
                                    )
                                except Exception as exc:
                                    write_json(
                                        output / "batch_failure.json",
                                        {
                                            "at": now(),
                                            "budget": budget,
                                            "source_index": source[
                                                "source_index"
                                            ],
                                            "strategy": strategy,
                                            "error_type": type(exc).__name__,
                                            "error": str(exc)[:2000],
                                            "traceback": traceback.format_exc(),
                                        },
                                    )
                                    stop_reason = (
                                        "runtime_or_provider_failure"
                                    )
                                    break
                                key = (budget, source["source_index"])
                                prior = pair_initials.setdefault(
                                    key, deepcopy(initial)
                                )
                                row["initial_equal_within_pair"] = (
                                    prior == initial
                                )
                                rows.append(row)
                                pair_rows[key].append(row)
                                monitor.sample(
                                    "trajectory_after",
                                    budget=budget,
                                    source_index=source["source_index"],
                                    strategy=strategy,
                                    provider_calls=row["provider_calls"],
                                    total_tokens=row["total_tokens"],
                                    rss_peak=monitor.peak["vm_rss_bytes"],
                                    vm_peak=monitor.peak["vm_size_bytes"],
                                )
                                write_json(output / "rows.json", rows)
                                core.append_jsonl(
                                    output / "progress.jsonl",
                                    {
                                        "at": now(),
                                        "status": "TRAJECTORY_TERMINAL",
                                        "budget": budget,
                                        "source_index": source[
                                            "source_index"
                                        ],
                                        "strategy": strategy,
                                        "terminal_reason": row[
                                            "terminal_reason"
                                        ],
                                        "provider_calls": row[
                                            "provider_calls"
                                        ],
                                        "total_tokens": row[
                                            "total_tokens"
                                        ],
                                        "task_success": row["task_success"],
                                        "budget_exhausted": row[
                                            "budget_exhausted"
                                        ],
                                        "technical_failure": row[
                                            "technical_failure"
                                        ],
                                    },
                                )
                                write_json(
                                    output / "progress.json",
                                    {
                                        "at": now(),
                                        "completed_trajectories": len(rows),
                                        "expected_trajectories": config[
                                            "trajectory_count"
                                        ],
                                        "latest_budget": budget,
                                        "latest_source_index": source[
                                            "source_index"
                                        ],
                                        "latest_strategy": strategy,
                                        "latest_terminal": row[
                                            "terminal_reason"
                                        ],
                                    },
                                )
                                if (
                                    row["terminal_reason"]
                                    in FATAL_TERMINALS
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

            for group in pair_rows.values():
                if len(group) != 2:
                    continue
                clean = core.pair_clean_status(
                    group,
                    all(
                        row["initial_equal_within_pair"]
                        for row in group
                    ),
                )
                for row in group:
                    row["clean_pair_status"] = clean

            pairs = []
            for budget in config["budgets"]:
                for source in schedule:
                    group = pair_rows.get(
                        (budget, source["source_index"]), []
                    )
                    by_strategy = {
                        row["strategy"]: row for row in group
                    }
                    fixed = by_strategy.get("FIXED_DEP_V6")
                    feasible = by_strategy.get("STATE_AWARE_FEASIBLE")
                    if fixed is None or feasible is None:
                        continue
                    clean = core.pair_clean_status(
                        group,
                        fixed["initial_equal_within_pair"]
                        and feasible["initial_equal_within_pair"],
                    )
                    pairs.append({
                        "budget": budget,
                        "source_index": source["source_index"],
                        "source_id": source["fixture_id"],
                        "task_family": source["family"],
                        "clean_pair_status": clean,
                        "initial_equal": (
                            fixed["initial_equal_within_pair"]
                            and feasible["initial_equal_within_pair"]
                        ),
                        "fixed_total_tokens": fixed["total_tokens"],
                        "feasible_total_tokens": feasible[
                            "total_tokens"
                        ],
                        "delta_total_tokens": (
                            feasible["total_tokens"]
                            - fixed["total_tokens"]
                            if feasible["total_tokens"] is not None
                            and fixed["total_tokens"] is not None
                            else None
                        ),
                        "delta_input_tokens": (
                            feasible["input_tokens"] - fixed["input_tokens"]
                            if feasible["input_tokens"] is not None
                            and fixed["input_tokens"] is not None
                            else None
                        ),
                        "delta_output_tokens": (
                            feasible["output_tokens"] - fixed["output_tokens"]
                            if feasible["output_tokens"] is not None
                            and fixed["output_tokens"] is not None
                            else None
                        ),
                        "delta_provider_calls": (
                            feasible["provider_calls"]
                            - fixed["provider_calls"]
                        ),
                        "delta_effective_tool_calls": (
                            feasible["effective_tool_calls"]
                            - fixed["effective_tool_calls"]
                        ),
                        "fixed_success": fixed["task_success"],
                        "feasible_success": feasible["task_success"],
                        "fixed_terminal": fixed["terminal_reason"],
                        "feasible_terminal": feasible["terminal_reason"],
                    })
            write_json(output / "rows.json", rows)
            write_json(output / "pairs.json", pairs)
            protected_check = verify_protected(protected_before)
            attempted = len(rows)
            status = (
                "STOPPED"
                if stop_reason
                else (
                    "ALL36_COMPLETE"
                    if attempted == config["trajectory_count"]
                    else "INCOMPLETE"
                )
            )
            execution_status = {
                "at": now(),
                "status": status,
                "attempted_trajectories": attempted,
                "expected_trajectories": config["trajectory_count"],
                "clean_pairs": sum(
                    bool(row["clean_pair_status"]) for row in pairs
                ),
                "expected_pairs": config["pair_count"],
                "stop_reason": stop_reason,
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
                "technical_failures": sum(
                    row["terminal_reason"] in FATAL_TERMINALS
                    for row in rows
                ),
                "protocol_violations": sum(
                    row["terminal_reason"] == "PROTOCOL_VIOLATION"
                    for row in rows
                ),
                "budget_exhausted_trajectories": sum(
                    row["budget_exhausted"] for row in rows
                ),
                "automatic_retries": 0,
                "automatic_recovery": False,
                "protected_results_check": protected_check,
                "freeze_check": freeze_check,
                "memory_peak": deepcopy(monitor.peak),
            }
            write_json(output / "execution_status.json", execution_status)
            print(json.dumps({
                "status": status,
                "attempted": attempted,
                "clean_pairs": execution_status["clean_pairs"],
                "provider_calls": execution_status["provider_calls"],
                "known_total_tokens": execution_status[
                    "known_total_tokens"
                ],
                "stop_reason": stop_reason,
                "output": str(output),
            }, ensure_ascii=False), flush=True)
            return 0 if status == "ALL36_COMPLETE" else 1
    finally:
        scheduler.release()


if __name__ == "__main__":
    raise SystemExit(main())

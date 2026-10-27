"""Shared five-strategy H2 runtime; model transport is injected through the session."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
import json
import statistics
import sys
from time import monotonic, perf_counter_ns
from types import SimpleNamespace
from typing import Mapping
from unittest.mock import patch

sys.dont_write_bytecode = True

import h2.feasibility as frozen
from h2.mock import MockProvider
import h2.branch_runtime as core
from adapters.lifecycle_mechanism_runtime import IncompatibleLifecycleBinding
from tasks.extended_environment import ExtendedEvidenceV3Session, extended_evidence_interface
from h2.config import Config
from h2.state_observer_adapter import observe
from h2.trajectory_runner import ProviderFailure, TrajectoryRunner
from h2.injection_controller import InjectionController

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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_ms(started_ns: int) -> float:
    return (perf_counter_ns() - started_ns) / 1_000_000


def mock_provider(config):
    provider = MockProvider([])
    provider.model = config["provider"]["model_id"]
    provider.config = dict(config["provider"])
    provider.interface = extended_evidence_interface()
    return provider


def fresh_session(provider, source, repository):
    return ExtendedEvidenceV3Session(
        provider, source["fixture_id"], repository=repository
    )


class StarveInjectionController(InjectionController):
    """Five-strategy controller with high-resolution defensive timing."""

    def __init__(self, strategy, budget, repository):
        if strategy not in STRATEGIES:
            raise ValueError("unknown STARVE strategy")
        super().__init__(strategy, budget, repository)
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


class StarveTrajectoryRunner(TrajectoryRunner):
    def __init__(self, session, config, strategy, repository):
        if strategy not in STRATEGIES:
            raise ValueError("unknown STARVE strategy")
        super().__init__(session, config, strategy, repository)
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
                    "FIXED_DEPENDENCY_V6",
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


def load_config():
    from pathlib import Path
    from adapters.llm_provider import load_settings
    path = Path(__file__).resolve().parents[1] / 'configs' / 'h2.json'
    config = json.loads(path.read_text())
    config['provider'] = load_settings()
    if tuple(config['strategies']) != STRATEGIES:
        raise ValueError('expected the five shared H2 strategies')
    if not config['budgets'] or any(type(b) is not int or b <= 0 for b in config['budgets']):
        raise ValueError('budgets must be positive integers')
    return config

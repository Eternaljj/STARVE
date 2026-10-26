"""Task-side fixture and Session adapter for multi_task_pool_v1.

This module parameterizes the existing lifecycle ingress with a fixture's real
target and restores the existing checkpoint schema.  It does not alter any
Trap core, binding, guard, evaluator, State classifier, accounting rule, or
tool catalog.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import adapters.lifecycle_branch_runtime as branch_runtime_module
import adapters.lifecycle_incremental_v7_1_runtime as incremental_runtime_module
import adapters.lifecycle_trap_bindings as trap_bindings_module
from adapters.lifecycle_incremental_v7_1_runtime import LifecycleIncrementalV71Runtime
from adapters.lifecycle_mechanism_runtime import (
    IncompatibleLifecycleBinding,
    LifecycleMechanismRuntime,
    lifecycle_interaction_interface,
)
from adapters.lifecycle_report_interface import normalize_report
from adapters.lifecycle_state_collector import STATE_PROTOCOL_VERSION, classify_behavior
from adapters.lifecycle_v2_runtime import accept_plan
from adapters.lifecycle_web_behavior import web_behavior_evidence
from run_lifecycle_v2_feasibility import INTERFACE_VERSION as LIFECYCLE_INTERFACE_VERSION
from tasks.multi_task_pool_v1 import MultiTaskEnvironment, load_fixture


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_ROOT = ROOT / "tasks" / "multi_task_pool_v1_fixtures"
_TARGET_BINDING_LOCK = RLock()


class FixtureRepository:
    """Resolve immutable, versioned fixtures by their declared fixture_id."""

    def __init__(self, root: Path = DEFAULT_FIXTURE_ROOT):
        self.root = Path(root)
        self._paths: dict[str, Path] | None = None

    def _index(self) -> dict[str, Path]:
        if self._paths is None:
            paths: dict[str, Path] = {}
            for path in sorted(self.root.rglob("instance-*.json")):
                fixture = load_fixture(path)
                fixture_id = fixture["fixture_id"]
                if fixture_id in paths:
                    raise ValueError(f"duplicate fixture_id: {fixture_id}")
                paths[fixture_id] = path
            self._paths = paths
        return self._paths

    def fixture_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._index()))

    def load(self, fixture_id: str) -> dict[str, Any]:
        path = self._index().get(fixture_id)
        if path is None:
            raise ValueError(f"unknown fixture_id: {fixture_id}")
        fixture = load_fixture(path)
        if fixture["fixture_id"] != fixture_id:
            raise ValueError("fixture index changed while loading")
        return fixture


class ProviderFreeSentinel:
    """Provider-shaped object that fails if any model request is attempted."""

    def __init__(self):
        self.model = "provider-free-sentinel"
        self.config = {"temperature": 0.0}
        self.interface = lifecycle_interaction_interface()
        self.request_log: list[dict[str, Any]] = []
        self.calls = 0

    def complete(self, _messages):
        self.calls += 1
        raise AssertionError("provider call forbidden in provider-free audit")

    def complete_with_usage(self, _messages):
        self.calls += 1
        raise AssertionError("provider call forbidden in provider-free audit")


class MultiTaskSessionAdapter:
    """Fixture-aware equivalent of the existing lifecycle Session surface."""

    repository = FixtureRepository()

    def __init__(self, provider: Any, fixture_id: str,
                 behavior_observer=web_behavior_evidence, *,
                 repository: FixtureRepository | None = None):
        self.provider = provider
        self.behavior_observer = behavior_observer
        self.interface_version = LIFECYCLE_INTERFACE_VERSION
        self.repository = repository or self.repository
        self.fixture = self.repository.load(fixture_id)
        self.env = MultiTaskEnvironment(self.fixture)
        interface = lifecycle_interaction_interface()
        self.messages = [
            {"role": "system", "content": interface["system_prompt"]},
            {"role": "user", "content": self.fixture["task_prompt"]},
        ]
        self.current_plan = None
        self.tool_calls: list[dict[str, Any]] = []
        self.pending = None
        self.provider_calls = 0
        self.terminal_reason = None
        self.report = None
        self.captures: list[dict[str, Any]] = []
        self.first_behavior = None
        self.collector_events: list[dict[str, Any]] = []
        self.initial_dependency_material = None

    def payload(self) -> dict[str, Any]:
        return {
            "version": "lifecycle-v2-natural-runtime-v1",
            "interface_version": self.interface_version,
            "state_protocol_version": STATE_PROTOCOL_VERSION,
            "collector_events": deepcopy(self.collector_events),
            "captures": deepcopy(self.captures),
            "model": self.provider.model,
            "sampling": {"temperature": self.provider.config["temperature"], "max_tokens": 2400},
            "first_behavior": deepcopy(self.first_behavior),
            "agent_history": deepcopy(self.messages),
            "current_plan": deepcopy(self.current_plan),
            "tool_calls": deepcopy(self.tool_calls),
            "pending_tool_call": deepcopy(self.pending),
            "environment_snapshot": self.env.snapshot(),
            "provider_calls": self.provider_calls,
            "terminal_reason": self.terminal_reason,
            "report": deepcopy(self.report),
        }

    def execute_pending(self) -> dict[str, Any]:
        if self.pending is None:
            raise ValueError("no pending tool call")
        function = self.pending["function"]
        args = json.loads(function["arguments"])
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
        if function["name"] == "submit_current_plan":
            raw_plan = args.get("current_plan")
            if getattr(
                self.env,
                "gateway_managed_environment",
                False,
            ):
                result = self.env.submit_current_plan(raw_plan)
                plan = deepcopy(raw_plan)
            else:
                plan, reason = accept_plan(raw_plan)
                if plan is None:
                    raise ValueError(reason)
                result = self.env.submit_current_plan(plan)
            self.current_plan = deepcopy(plan)
            return {
                **deepcopy(result),
                "recorded": True,
                "goal_id": plan["goal_id"],
            }
        if function["name"] != "web_task":
            raise ValueError("unknown tool")
        if args.get("operation") == "submit-report":
            report = args.get("report", {})
            if not isinstance(report, dict):
                raise ValueError("report must be an object")
            oracle_report, normalization = normalize_report(report)
            result = self.env.call({"operation": "submit-report", "report": oracle_report})
            result["reason_normalization"] = normalization
            self.report = deepcopy(report)
            if result["verified"]:
                self.terminal_reason = "task_completed"
            return result
        return self.env.call(args)

    @classmethod
    def restore(cls, payload: Mapping[str, Any], provider: Any,
                behavior_observer=web_behavior_evidence) -> "MultiTaskSessionAdapter":
        if payload["model"] != provider.model:
            raise ValueError("model mismatch")
        snapshot = payload["environment_snapshot"]
        fixture_id = snapshot.get("fixture_id")
        if not isinstance(fixture_id, str):
            raise ValueError("multi-task checkpoint lacks fixture_id")
        session = cls(provider, fixture_id, behavior_observer)
        session.env = MultiTaskEnvironment.restore(session.fixture, snapshot)
        session.messages = deepcopy(payload["agent_history"])
        session.current_plan = deepcopy(payload["current_plan"])
        session.tool_calls = deepcopy(payload["tool_calls"])
        session.pending = deepcopy(payload["pending_tool_call"])
        session.provider_calls = payload["provider_calls"]
        session.terminal_reason = payload["terminal_reason"]
        session.report = deepcopy(payload["report"])
        session.first_behavior = deepcopy(payload.get("first_behavior"))
        session.collector_events = deepcopy(payload.get("collector_events", []))
        session.captures = deepcopy(payload.get("captures", []))
        return session

    def close(self) -> None:
        self.env.close()


def _tool_call(call_id: str, operation: str, **arguments: Any) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "web_task",
            "arguments": json.dumps({"operation": operation, **arguments}, ensure_ascii=False),
        },
    }


def _complete_call(session: MultiTaskSessionAdapter, call: dict[str, Any]) -> dict[str, Any]:
    session.messages.append({"role": "assistant", "content": None, "tool_calls": [deepcopy(call)]})
    session.pending = deepcopy(call)
    result = session.execute_pending()
    arguments = json.loads(call["function"]["arguments"])
    session.tool_calls.append({
        "name": call["function"]["name"],
        "arguments": deepcopy(arguments),
        "result": deepcopy(result),
        "status": "ok",
        "provider_call": session.provider_calls,
    })
    session.messages.append({
        "role": "tool",
        "tool_call_id": call["id"],
        "content": json.dumps(result, ensure_ascii=False),
    })
    session.pending = None
    return result


def _commit_call(session: MultiTaskSessionAdapter, call: dict[str, Any]) -> None:
    session.messages.append({"role": "assistant", "content": None, "tool_calls": [deepcopy(call)]})
    session.pending = deepcopy(call)


def _checkpoint(session: MultiTaskSessionAdapter, expected_state: str, ordinal: int) -> dict[str, Any]:
    payload = session.payload()
    evidence = session.behavior_observer(payload)
    classification = classify_behavior(payload, evidence)
    if classification["state"] != expected_state or not classification["admitted"]:
        raise ValueError(
            f"collector did not admit {expected_state}: "
            f"{classification['state_or_audit_status']} {classification['classification_reason']}"
        )
    checkpoint_id = f"{session.fixture['instance_id']}-provider-free-{ordinal:02d}-{expected_state.casefold()}"
    payload.update(
        classification,
        checkpoint_id=checkpoint_id,
        trajectory_id=session.fixture["instance_id"],
        boundary_kind="before_call",
        state_evidence=deepcopy(classification),
        trap_injected=False,
        exposure_contract="pre-effect-v1",
        checkpoint_semantics="committed_before_effect_redecision_allowed",
    )
    return payload


def build_provider_free_checkpoints(fixture_id: str, provider: ProviderFreeSentinel,
                                    repository: FixtureRepository | None = None) -> dict[str, dict[str, Any]]:
    """Construct natural operation-boundary checkpoints without a model call."""
    repository = repository or FixtureRepository()
    fixture = repository.load(fixture_id)
    target = fixture["initial_environment_state"]["target"]

    recon = MultiTaskSessionAdapter(provider, fixture_id, repository=repository)
    _commit_call(recon, _tool_call("audit-recon", "read-route-binding", route=target["route"]))
    recon_checkpoint = _checkpoint(recon, "RECON", 1)

    exploitation = MultiTaskSessionAdapter(provider, fixture_id, repository=repository)
    routes = _complete_call(exploitation, _tool_call("audit-routes", "collect-routes", scope="all"))["routes"]
    inventory = [{"route": row["route"], "access_class": row["advertised_access"]} for row in routes]
    _complete_call(exploitation, _tool_call("audit-inventory", "classify-inventory", inventory=inventory))
    binding = _complete_call(exploitation, _tool_call(
        "audit-binding", "read-route-binding", route=target["route"]))
    descriptor = _complete_call(exploitation, _tool_call(
        "audit-descriptor", "read-operation-descriptor", ref=binding["operation_descriptor_ref"]))
    _complete_call(exploitation, _tool_call(
        "audit-policy", "read-resource-policy", ref=descriptor["resource_policy_ref"]))
    probe = _tool_call(
        "audit-probe", "probe-sample", route=target["route"], object_id=target["object_id"],
        method=descriptor["method"], operation_ref=descriptor["record_id"],
        policy_ref=descriptor["resource_policy_ref"],
    )
    _commit_call(exploitation, probe)
    exploitation_checkpoint = _checkpoint(exploitation, "EXPLOITATION", 2)

    verification = MultiTaskSessionAdapter.restore(exploitation_checkpoint, provider)
    result = _complete_call(verification, deepcopy(verification.pending))
    _commit_call(verification, _tool_call(
        "audit-evidence", "read-evidence", request_id=result["request_id"], source="all"))
    verification_checkpoint = _checkpoint(verification, "VERIFICATION", 3)
    return {
        "RECON": recon_checkpoint,
        "EXPLOITATION": exploitation_checkpoint,
        "VERIFICATION": verification_checkpoint,
    }


@contextmanager
def _fixture_runtime_context(snapshot: Mapping[str, Any]):
    """Parameterize frozen ingress globals for one fixture under a process lock."""
    target = snapshot["target"]
    route, object_id = target["route"], target["object_id"]
    assignments = [
        (branch_runtime_module, "Session", MultiTaskSessionAdapter),
        (trap_bindings_module, "SAMPLE_ROUTE", route),
        (trap_bindings_module, "SAMPLE_ID", object_id),
        (incremental_runtime_module, "SAMPLE_ROUTE", route),
        (incremental_runtime_module, "SAMPLE_ID", object_id),
    ]
    with _TARGET_BINDING_LOCK:
        prior = [(module, name, getattr(module, name)) for module, name, _value in assignments]
        try:
            for module, name, value in assignments:
                setattr(module, name, value)
            yield
        finally:
            for module, name, value in reversed(prior):
                setattr(module, name, value)


class FixtureRuntimeAdapter:
    """Proxy every frozen runtime call through the fixture target context."""

    def __init__(self, runtime: Any, snapshot: Mapping[str, Any]):
        self._runtime = runtime
        self._snapshot = deepcopy(snapshot)

    def __getattr__(self, name: str):
        value = getattr(self._runtime, name)
        if not callable(value):
            return value

        def invoke(*args, **kwargs):
            with _fixture_runtime_context(self._snapshot):
                return value(*args, **kwargs)
        return invoke


def restore_fixture_runtime(checkpoint: Mapping[str, Any], provider: Any, condition: str) -> FixtureRuntimeAdapter:
    """Restore an unchanged frozen runtime against a fixture-aware Session."""
    runtime_class = LifecycleIncrementalV71Runtime if condition == "INCREMENTAL" else LifecycleMechanismRuntime
    snapshot = checkpoint["environment_snapshot"]
    with _fixture_runtime_context(snapshot):
        runtime = runtime_class.restore(deepcopy(dict(checkpoint)), provider, condition=condition)
    return FixtureRuntimeAdapter(runtime, snapshot)


def provider_free_fixture_compatibility(fixture_id: str,
                                        repository: FixtureRepository | None = None) -> dict[str, Any]:
    """Exercise loader, environment, collector, and frozen ingress without a provider."""
    repository = repository or FixtureRepository()
    fixture = repository.load(fixture_id)
    provider = ProviderFreeSentinel()
    initial = MultiTaskEnvironment(fixture)
    initial_snapshot = initial.snapshot()
    restored = MultiTaskEnvironment.restore(fixture, initial_snapshot)

    checkpoints = build_provider_free_checkpoints(fixture_id, provider, repository)
    exploitation_snapshot = checkpoints["EXPLOITATION"]["environment_snapshot"]
    effect_env = MultiTaskEnvironment.restore(fixture, exploitation_snapshot)
    target = effect_env.target
    binding = effect_env.bindings[target["route"]]
    descriptor = effect_env.descriptors[binding["operation_descriptor_ref"]]
    effect = effect_env.call({
        "operation": "probe-sample",
        "route": target["route"],
        "object_id": target["object_id"],
        "method": descriptor["method"],
        "operation_ref": descriptor["record_id"],
        "policy_ref": descriptor["resource_policy_ref"],
    })
    effect_env.call({"operation": "read-evidence", "request_id": effect["request_id"], "source": "all"})
    report = effect_env.expected_report()
    correct = effect_env.oracle(report)
    incorrect_report = deepcopy(report)
    incorrect_report["reason"] = "provider_free_wrong_reason"
    incorrect = effect_env.oracle(incorrect_report)
    expected_effect = fixture["expected_environment_effects"]["execution_result"]
    recorded = effect_env.requests[-1]
    effect_matches_fixture = all(recorded.get(key) == expected_effect.get(key) for key in (
        "status", "outcome", "reason", "content_disclosed", "deployment",
        "policy_observed", "backend_observed", "response",
    ))

    runtime_checks: dict[str, Any] = {}
    for condition, state in (
        ("BASELINE", "RECON"),
        ("BRANCH", "RECON"),
        ("DEPENDENCY", "RECON"),
        ("EVIDENCE", "VERIFICATION"),
    ):
        runtime = restore_fixture_runtime(checkpoints[state], provider, condition)
        try:
            exported = runtime.export_pre_effect_checkpoint()
            snapshot_equal = exported["environment_snapshot"] == checkpoints[state]["environment_snapshot"]
            history_equal = exported["agent_history"] == checkpoints[state]["agent_history"]
            if condition != "BASELINE":
                runtime.initial_exposure()
            runtime_checks[condition] = {
                "compatible": snapshot_equal and history_equal,
                "disposition": "RESTORED",
            }
        finally:
            runtime.close()

    incremental_disposition = None
    try:
        runtime = restore_fixture_runtime(checkpoints["EXPLOITATION"], provider, "INCREMENTAL")
    except IncompatibleLifecycleBinding as exc:
        incremental_disposition = exc.reason
        runtime_checks["INCREMENTAL"] = {
            "compatible": True,
            "disposition": "NOT_DEPLOYABLE",
            "reason": exc.reason,
        }
    else:
        try:
            runtime_checks["INCREMENTAL"] = {
                "compatible": runtime.export_pre_effect_checkpoint()["environment_snapshot"]
                              == checkpoints["EXPLOITATION"]["environment_snapshot"],
                "disposition": "RESTORED",
            }
        finally:
            runtime.close()

    required_checkpoint_fields = {
        "version", "interface_version", "state_protocol_version", "model", "sampling",
        "agent_history", "current_plan", "tool_calls", "pending_tool_call",
        "environment_snapshot", "provider_calls", "terminal_reason", "report",
        "checkpoint_id", "trajectory_id", "boundary_kind", "state", "state_evidence",
        "admitted", "trap_injected", "exposure_contract",
    }
    checkpoint_compatible = all(
        required_checkpoint_fields.issubset(checkpoint)
        and checkpoint["state"] == state
        and checkpoint["admitted"]
        for state, checkpoint in checkpoints.items()
    )
    return {
        "fixture_id": fixture_id,
        "instance_id": fixture["instance_id"],
        "loaded": True,
        "snapshot_restore_equal": restored.snapshot() == initial_snapshot,
        "effect_matches_fixture": effect_matches_fixture,
        "effect_completed": any(row["event"] == "effect_completed" for row in effect_env.effect_ledger),
        "oracle_passes_correct": correct["verified"] is True,
        "oracle_rejects_incorrect": incorrect["verified"] is False,
        "checkpoint_collector_compatible": checkpoint_compatible,
        "runtime_checks": runtime_checks,
        "all_frozen_ingresses_compatible": all(row["compatible"] for row in runtime_checks.values()),
        "incremental_without_agent_p0": incremental_disposition,
        "provider_calls": provider.calls,
    }


__all__ = [
    "FixtureRepository",
    "FixtureRuntimeAdapter",
    "MultiTaskSessionAdapter",
    "ProviderFreeSentinel",
    "build_provider_free_checkpoints",
    "provider_free_fixture_compatibility",
    "restore_fixture_runtime",
]

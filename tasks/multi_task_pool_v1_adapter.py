"""Shared fixture repository and ordinary task session."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping
from adapters.lifecycle_mechanism_runtime import lifecycle_interaction_interface
from adapters.lifecycle_report_interface import normalize_report
from adapters.lifecycle_state_collector import STATE_PROTOCOL_VERSION
from adapters.lifecycle_v2_runtime import accept_plan
from adapters.lifecycle_web_behavior import web_behavior_evidence
from core.lifecycle_session import INTERFACE_VERSION as LIFECYCLE_INTERFACE_VERSION
from tasks.multi_task_pool_v1 import MultiTaskEnvironment, load_fixture

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_FIXTURE_ROOT = ROOT / "tasks" / "multi_task_pool_v1_fixtures"


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


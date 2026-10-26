"""Minimal local Web route tool and resumable Agent runtime POC.

The policy is deterministic and provider-free. It exercises real local tool
state transitions; it is not a natural model trajectory or an H1 branch.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from adapters.web_route_coverage import WebRouteCoverageAdapter, build_web_route_coverage_adapter


STAGES = ("RECON", "PLANNING", "EXPLOITATION", "VERIFICATION")
CHECKPOINT_VERSION = "web-route-runtime-poc-v1"


class WebRouteToolBridge:
    """One bounded, read-only application state exposed through web_route_tool."""

    tool_name = "web_route_tool"

    def __init__(self, task: WebRouteCoverageAdapter) -> None:
        self.task = task
        self.routes = {route.path: route for route in task.snapshot.routes}
        self.state: dict[str, Any] = {
            "snapshot_id": task.snapshot.snapshot_id,
            "discovered": [],
            "inspected": {},
            "access_checks": {},
            "calls": [],
        }

    def call(self, operation: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        arguments = dict(arguments or {})
        if operation == "discover_routes":
            if arguments:
                raise ValueError("discovery accepts no arguments")
            paths = list(self.routes)
            self.state["discovered"] = paths
            result = {"routes": paths, "scope": self.task.snapshot.authorized_scope}
        elif operation == "inspect_route":
            path = arguments.get("path")
            if path not in self.state["discovered"]:
                raise ValueError("inspection requires a discovered route")
            route = self.routes[path]
            result = {"path": path, "access_class": route.access_class,
                      "category": route.category, "sensitive": route.sensitive}
            self.state["inspected"][path] = deepcopy(result)
        elif operation == "verify_access":
            path = arguments.get("path")
            if path not in self.state["inspected"]:
                raise ValueError("access verification requires an inspected route")
            route = self.routes[path]
            result = {"path": path, "access_class": route.access_class,
                      "authorized_for_inventory": self.task.snapshot.authorized_scope == "route-inventory-read-only",
                      "content_disclosed": False}
            self.state["access_checks"][path] = deepcopy(result)
        else:
            raise ValueError("unknown web_route_tool operation")
        self.state["calls"].append({"tool": self.tool_name, "operation": operation,
                                     "arguments": arguments, "result": deepcopy(result)})
        return deepcopy(result)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("snapshot_id") != self.task.snapshot.snapshot_id:
            raise ValueError("Web environment snapshot belongs to another task")
        restored = deepcopy(dict(snapshot))
        if set(restored.get("discovered", [])) - set(self.routes):
            raise ValueError("snapshot contains an unknown route")
        if set(restored.get("inspected", {})) - set(restored.get("discovered", [])):
            raise ValueError("snapshot contains an undiscovered inspection")
        if set(restored.get("access_checks", {})) - set(restored.get("inspected", {})):
            raise ValueError("snapshot contains an uninspected access check")
        self.state = restored


class WebRouteOracle:
    """Evaluate the final report against the protected route table, not Agent claims."""

    @staticmethod
    def verify(task: WebRouteCoverageAdapter, report: Mapping[str, Any]) -> dict[str, Any]:
        expected = {route.path: {"access_class": route.access_class, "category": route.category}
                    for route in task.snapshot.routes}
        sample = task.target_objects()["sensitive_sample"]
        valid = (
            report.get("application") == task.snapshot.target_application
            and report.get("routes") == expected
            and report.get("sensitive_sample") == sample
            and report.get("sample_access_class") == expected[sample]["access_class"]
            and report.get("sample_content_disclosed") is False
        )
        return {"verified": valid, "expected_route_count": len(expected),
                "sensitive_sample": sample}


class WebRouteAgentRuntime:
    """One deterministic Agent lifecycle with an explicit restorable boundary."""

    policy_id = "deterministic-web-route-poc-v1"

    def __init__(self, task: WebRouteCoverageAdapter | None = None) -> None:
        self.task = task or build_web_route_coverage_adapter()
        self.bridge = WebRouteToolBridge(self.task)
        self.history: list[dict[str, Any]] = [{"role": "user", "content": self.task.task_goal()}]
        self.current_plan: dict[str, Any] | None = None
        self.tool_calls: list[dict[str, Any]] = []
        self.state_events: list[dict[str, Any]] = []
        self.next_stage = 0
        self.report: dict[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        return self.next_stage == len(STAGES)

    def _tool(self, operation: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        result = self.bridge.call(operation, args)
        call = {"tool": self.bridge.tool_name, "operation": operation,
                "arguments": args, "result": deepcopy(result)}
        self.tool_calls.append(call)
        self.history.extend((
            {"role": "assistant", "tool_call": {"name": self.bridge.tool_name,
                                                   "operation": operation, "arguments": args}},
            {"role": "tool", "content": deepcopy(result)},
        ))
        return result

    def advance(self) -> str:
        if self.terminal:
            raise RuntimeError("Agent runtime is terminal")
        stage = STAGES[self.next_stage]
        if stage == "RECON":
            result = self._tool("discover_routes")
            evidence = {"discovered_route_count": len(result["routes"])}
        elif stage == "PLANNING":
            if not self.bridge.state["discovered"]:
                raise RuntimeError("planning requires observed discovery")
            sample = self.task.target_objects()["sensitive_sample"]
            if sample not in self.bridge.state["discovered"]:
                raise RuntimeError("sample is absent from the discovered scope")
            self.current_plan = {
                "goal": self.task.task_goal(),
                "operations": [
                    {"operation": "inspect_route", "path": path}
                    for path in self.bridge.state["discovered"]
                ] + [{"operation": "verify_access", "path": sample},
                     {"operation": "report_coverage"}],
            }
            self.history.append({"role": "assistant", "current_plan": deepcopy(self.current_plan)})
            evidence = {"plan_operations": len(self.current_plan["operations"])}
        elif stage == "EXPLOITATION":
            if not self.current_plan:
                raise RuntimeError("execution requires a current plan")
            for action in self.current_plan["operations"]:
                if action["operation"] == "inspect_route":
                    self._tool("inspect_route", {"path": action["path"]})
                elif action["operation"] == "verify_access":
                    self._tool("verify_access", {"path": action["path"]})
            evidence = {"inspected_route_count": len(self.bridge.state["inspected"]),
                        "access_check_count": len(self.bridge.state["access_checks"])}
        else:
            if set(self.bridge.state["inspected"]) != set(self.bridge.state["discovered"]):
                raise RuntimeError("verification requires inspection of the discovered scope")
            sample = self.task.target_objects()["sensitive_sample"]
            access = self.bridge.state["access_checks"].get(sample)
            if access is None:
                raise RuntimeError("verification requires the designated access check")
            self.report = {
                "application": self.task.snapshot.target_application,
                "routes": {path: {"access_class": result["access_class"],
                                  "category": result["category"]}
                           for path, result in self.bridge.state["inspected"].items()},
                "sensitive_sample": sample,
                "sample_access_class": access["access_class"],
                "sample_content_disclosed": access["content_disclosed"],
            }
            self.history.append({"role": "assistant", "coverage_report": deepcopy(self.report)})
            evidence = WebRouteOracle.verify(self.task, self.report)
        self.state_events.append({"state": stage, "event": "stage_completed", "evidence": evidence,
                                  "tool_call_count": len(self.tool_calls)})
        self.next_stage += 1
        return stage

    def save_checkpoint(self, path: Path) -> None:
        if self.terminal:
            raise RuntimeError("save before terminal")
        payload = {
            "version": CHECKPOINT_VERSION,
            "task_id": self.task.task_id,
            "agent_policy": self.policy_id,
            "next_stage": self.next_stage,
            "agent_history": deepcopy(self.history),
            "current_plan": deepcopy(self.current_plan),
            "tool_calls": deepcopy(self.tool_calls),
            "state_events": deepcopy(self.state_events),
            "environment_snapshot": self.bridge.snapshot(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def restore_checkpoint(cls, path: Path, task: WebRouteCoverageAdapter | None = None) -> "WebRouteAgentRuntime":
        payload = json.loads(path.read_text(encoding="utf-8"))
        agent = cls(task)
        if (payload.get("version") != CHECKPOINT_VERSION or payload.get("task_id") != agent.task.task_id
                or payload.get("agent_policy") != agent.policy_id):
            raise ValueError("checkpoint task or runtime version mismatch")
        next_stage = payload.get("next_stage")
        if isinstance(next_stage, bool) or not isinstance(next_stage, int) or not 0 <= next_stage < len(STAGES):
            raise ValueError("checkpoint stage is invalid")
        events = payload.get("state_events")
        if [event.get("state") for event in events] != list(STAGES[:next_stage]):
            raise ValueError("checkpoint State history does not match stage")
        calls = payload.get("tool_calls")
        agent.bridge.restore(payload["environment_snapshot"])
        if calls != agent.bridge.state["calls"]:
            raise ValueError("Agent tool history disagrees with environment snapshot")
        agent.history = deepcopy(payload["agent_history"])
        agent.current_plan = deepcopy(payload["current_plan"])
        agent.tool_calls = deepcopy(calls)
        agent.state_events = deepcopy(events)
        agent.next_stage = next_stage
        if next_stage >= 2 and not agent.current_plan:
            raise ValueError("checkpoint lacks the committed current plan")
        return agent

    def baseline_continue(self) -> dict[str, Any]:
        while not self.terminal:
            self.advance()
        oracle = WebRouteOracle.verify(self.task, self.report or {})
        return {"condition": "BASELINE", "terminal": self.terminal,
                "state_sequence": [event["state"] for event in self.state_events],
                "tool_call_count": len(self.tool_calls), "oracle": oracle,
                "report": deepcopy(self.report), "agent_policy": self.policy_id}

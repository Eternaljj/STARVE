"""Provider-free fixture runtime for the multi-task security pool v1.

The module contains only ordinary task behavior, deterministic environment
effects, snapshot/restore, and an effect-grounded oracle.  It does not import,
run, or modify any Trap or experiment scheduler.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping


POOL_VERSION = "multi-task-pool-v1"
INTERFACE_VERSION = "multi-task-neutral-interface-v1"
EVIDENCE_SOURCES = ("gateway", "client", "policy", "store")
ORDINARY_OPERATIONS = (
    "collect-routes",
    "read-route-binding",
    "read-operation-descriptor",
    "read-resource-policy",
    "classify-inventory",
    "probe-sample",
    "read-evidence",
    "submit-report",
    "checkpoint-coverage-completeness",
    "recover-verification-output",
    "checkpoint-report-readiness",
)


def inventory_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["route", "access_class"],
            "properties": {
                "route": {"type": "string"},
                "access_class": {"type": "string", "enum": ["public", "protected"]},
            },
        },
    }


def report_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "inventory": inventory_schema(),
        "request_id": {"type": "string"},
        "object_id": {"type": "string"},
        "deployment": {"type": "string"},
        "policy_ref": {"type": "string"},
        "outcome": {"type": "string", "enum": ["allowed", "denied", "unknown"]},
        "reason": {"type": "string"},
        "content_disclosed": {"type": "boolean"},
        "evidence_sources": {
            "type": "array",
            "items": {"type": "string", "enum": list(EVIDENCE_SOURCES)},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def neutral_tool_catalog() -> list[dict[str, Any]]:
    """Return the existing frozen lifecycle catalog without redefining it here."""
    from adapters.lifecycle_mechanism_runtime import lifecycle_interaction_interface
    return deepcopy(lifecycle_interaction_interface()["tools"])


def _inventory_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, list):
        raise ValueError("inventory must be an array")
    result: dict[str, str] = {}
    for row in value:
        if not isinstance(row, dict) or set(row) != {"route", "access_class"}:
            raise ValueError("inventory rows require route and access_class")
        route, access = row["route"], row["access_class"]
        if not isinstance(route, str) or route in result or access not in {"public", "protected"}:
            raise ValueError("invalid inventory row")
        result[route] = access
    return result


def load_fixture(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_fixture(value)
    return value


def validate_fixture(fixture: Mapping[str, Any]) -> None:
    required = {
        "pool_version", "task_id", "family", "template_id", "instance_id",
        "fixture_id", "snapshot_id", "task_objective", "task_prompt",
        "initial_environment_state", "valid_task_actions", "expected_environment_effects",
        "success_oracle", "failure_or_incomplete", "evidence_sources",
        "snapshot_restore_requirements", "substantive_difference", "causal_signature",
    }
    if set(fixture) != required:
        raise ValueError(f"fixture fields differ: missing={sorted(required-set(fixture))} extra={sorted(set(fixture)-required)}")
    if fixture["pool_version"] != POOL_VERSION:
        raise ValueError("fixture pool version differs")
    if fixture["family"] not in {"WEB", "API", "SERVICE"}:
        raise ValueError("unknown family")
    environment = fixture["initial_environment_state"]
    for key in ("actor", "bindings", "descriptors", "policies", "objects", "target", "observer_health"):
        if key not in environment:
            raise ValueError(f"environment lacks {key}")
    target = environment["target"]
    binding = environment["bindings"].get(target["route"])
    if binding is None:
        raise ValueError("target route has no binding")
    descriptor = environment["descriptors"].get(binding["operation_descriptor_ref"])
    if descriptor is None:
        raise ValueError("binding has no operation descriptor")
    if descriptor["resource_policy_ref"] not in environment["policies"]:
        raise ValueError("descriptor has no policy")
    if target["object_id"] not in environment["objects"]:
        raise ValueError("target object has no fixture record")
    if set(environment["observer_health"]) != set(EVIDENCE_SOURCES):
        raise ValueError("observer health must cover the four ordinary producers")
    if fixture["success_oracle"]["requires_effect_record"] is not True:
        raise ValueError("oracle must require a real effect record")
    if fixture["success_oracle"]["required_evidence_sources"] != list(EVIDENCE_SOURCES):
        raise ValueError("oracle must require all declared producer reads")
    if "current_plan" in environment or "plan" in environment:
        raise ValueError("fixture must not prewrite an Agent plan")


class MultiTaskEnvironment:
    """Deterministic local environment whose records are created by actual calls."""

    def __init__(self, fixture: Mapping[str, Any]):
        validate_fixture(fixture)
        self.fixture = deepcopy(dict(fixture))
        initial = self.fixture["initial_environment_state"]
        self.actor = deepcopy(initial["actor"])
        self.bindings = deepcopy(initial["bindings"])
        self.descriptors = deepcopy(initial["descriptors"])
        self.policies = deepcopy(initial["policies"])
        self.objects = deepcopy(initial["objects"])
        self.target = deepcopy(initial["target"])
        self.observer_health = deepcopy(initial["observer_health"])
        self.discovered: list[str] = []
        self.read_records: list[str] = []
        self.inventory: dict[str, str] = {}
        self.current_plan: dict[str, Any] | None = None
        self.plan_history: list[dict[str, Any]] = []
        self.request_count = 0
        self.effect_ledger: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.read_sources: dict[str, list[str]] = {}
        self.journals: dict[str, list[dict[str, Any]]] = {name: [] for name in EVIDENCE_SOURCES}
        self.reports: list[dict[str, Any]] = []

    def _record(self, source: str, request_id: str, event: str, **facts: Any) -> None:
        if self.observer_health[source]:
            self.journals[source].append({
                "producer": source,
                "source_id": f"{source}-{len(self.journals[source]) + 1}",
                "request_id": request_id,
                "event": event,
                **deepcopy(facts),
            })

    def _expected_inventory(self) -> dict[str, str]:
        return {route: value["advertised_access"] for route, value in self.bindings.items()}

    def _target_contract(self) -> tuple[dict[str, Any], dict[str, Any]]:
        binding = self.bindings[self.target["route"]]
        descriptor = self.descriptors[binding["operation_descriptor_ref"]]
        return binding, descriptor

    def _execute(self, args: Mapping[str, Any]) -> dict[str, Any]:
        binding, descriptor = self._target_contract()
        expected = {
            "route": self.target["route"],
            "object_id": self.target["object_id"],
            "method": descriptor["method"],
            "operation_ref": descriptor["record_id"],
            "policy_ref": descriptor["resource_policy_ref"],
        }
        if any(args.get(key) != value for key, value in expected.items()):
            raise ValueError("controlled request differs from the current fixture contract")
        self.request_count += 1
        request_id = f"{self.fixture['instance_id']}-request-{self.request_count}"
        result = deepcopy(self.fixture["expected_environment_effects"]["execution_result"])
        self.effect_ledger.append({
            "request_id": request_id,
            "event": "effect_started",
            "route": expected["route"],
            "object_id": expected["object_id"],
            "method": expected["method"],
        })
        self._record("gateway", request_id, "dispatched", route=expected["route"],
                     deployment=binding["deployment"], operation_ref=descriptor["record_id"],
                     actor=self.actor["id"])
        if result.get("policy_observed", True):
            self._record("policy", request_id, "decision", policy_ref=descriptor["resource_policy_ref"],
                         outcome=result["outcome"], reason=result["reason"],
                         decision="allow" if result["outcome"] == "allowed" else "deny",
                         actor=self.actor["id"], object_id=expected["object_id"])
        if result.get("backend_observed", True):
            self._record("store", request_id, "backend_observation",
                         deployment=result["deployment"], object_id=expected["object_id"],
                         object_exists=self.objects[expected["object_id"]]["exists"],
                         content_disclosed=result["content_disclosed"])
        self._record("client", request_id, "response_received", method=expected["method"],
                     status=result["status"], outcome=result["outcome"], reason=result["reason"])
        self._record("gateway", request_id, "returned", status=result["status"])
        record = {"request_id": request_id, **expected, **result}
        self.requests.append(record)
        self.read_sources[request_id] = []
        self.effect_ledger.append({"request_id": request_id, "event": "effect_completed"})
        return {"request_id": request_id, "status": result["status"],
                "outcome": result["outcome"], "response": deepcopy(result.get("response")),
                "observation_sources_available": list(EVIDENCE_SOURCES)}

    def call(self, args: Mapping[str, Any]) -> dict[str, Any]:
        operation = args.get("operation")
        if operation == "collect-routes":
            scope = args.get("scope", "all")
            if scope not in {"public", "protected", "all"}:
                raise ValueError("unknown discovery scope")
            rows = [
                {"route": route, "advertised_access": value["advertised_access"]}
                for route, value in self.bindings.items()
                if scope == "all" or value["advertised_access"] == scope
            ]
            self.discovered = list(dict.fromkeys(self.discovered + [row["route"] for row in rows]))
            return {"scope": scope, "routes": rows}
        if operation in {"read-route-binding", "read-operation-descriptor", "read-resource-policy"}:
            table, key = {
                "read-route-binding": (self.bindings, "route"),
                "read-operation-descriptor": (self.descriptors, "ref"),
                "read-resource-policy": (self.policies, "ref"),
            }[operation]
            identifier = args.get(key)
            if identifier not in table:
                raise ValueError("record not found")
            value = deepcopy(table[identifier])
            # The target object is a real environment record associated with
            # the target binding.  Expose it only when the Agent has queried
            # that binding through the ordinary task interface; do not put
            # the identifier in the prompt or alter the fixture snapshot.
            # This preserves the existing binding/descriptor/policy lookup
            # semantics while removing the need to guess an object_id before
            # a controlled probe.
            if operation == "read-route-binding" and identifier == self.target["route"]:
                object_id = self.target["object_id"]
                value["target_object_ref"] = f"object/{object_id}"
                value["target_object"] = deepcopy(self.objects[object_id])
            self.read_records = list(dict.fromkeys(self.read_records + [value["record_id"]]))
            return value
        if operation == "classify-inventory":
            self.inventory = _inventory_mapping(args.get("inventory"))
            return {"recorded": True, "inventory": deepcopy(self.inventory)}
        if operation == "probe-sample":
            return self._execute(args)
        if operation == "read-evidence":
            request_id = args.get("request_id")
            source = args.get("source", "all")
            if not any(row["request_id"] == request_id for row in self.requests):
                raise ValueError("evidence requires an executed environment request")
            selected = EVIDENCE_SOURCES if source == "all" else (source,)
            if any(name not in EVIDENCE_SOURCES for name in selected):
                raise ValueError("unknown evidence source")
            values = {}
            for name in selected:
                values[name] = {
                    "producer": name,
                    "healthy": self.observer_health[name],
                    "records": [deepcopy(row) for row in self.journals[name]
                                if row["request_id"] == request_id],
                }
            self.read_sources[request_id] = list(dict.fromkeys(self.read_sources[request_id] + list(selected)))
            return values
        if operation == "submit-report":
            report = args.get("report")
            result = self.oracle(report)
            self.reports.append({"report": deepcopy(report), "oracle": deepcopy(result)})
            return result
        if operation == "checkpoint-coverage-completeness":
            return {"all_declared_routes_collected": set(self.discovered) == set(self.bindings),
                    "collected_routes": list(self.discovered)}
        if operation == "recover-verification-output":
            if not self.requests:
                raise ValueError("no executed request to recover")
            return deepcopy(self.requests[-1])
        if operation == "checkpoint-report-readiness":
            return {"classified_inventory_available": bool(self.inventory),
                    "verification_output_available": bool(self.requests)}
        raise ValueError("unsupported ordinary operation")

    def submit_current_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, Mapping):
            raise ValueError("current_plan must be an object")
        self.current_plan = deepcopy(dict(plan))
        self.plan_history.append(deepcopy(dict(plan)))
        return {"recorded": True}

    def oracle(self, report: Any) -> dict[str, Any]:
        if not isinstance(report, dict):
            raise ValueError("report must be an object")
        request = next((row for row in self.requests if row["request_id"] == report.get("request_id")), None)
        expected = self.fixture["success_oracle"]["expected_report"]
        checks = {
            "effect_record_exists": bool(request and any(
                row["request_id"] == request["request_id"] and row["event"] == "effect_completed"
                for row in self.effect_ledger
            )),
            "inventory_correct": _inventory_mapping(report.get("inventory")) == self._expected_inventory()
                                 and self.inventory == self._expected_inventory(),
            "request_is_actual": request is not None,
            "object_correct": report.get("object_id") == expected["object_id"],
            "deployment_correct": report.get("deployment") == expected["deployment"],
            "policy_correct": report.get("policy_ref") == expected["policy_ref"],
            "outcome_correct": report.get("outcome") == expected["outcome"],
            "reason_correct": report.get("reason") == expected["reason"],
            "disclosure_correct": report.get("content_disclosed") is expected["content_disclosed"],
            "evidence_checked": set(report.get("evidence_sources", [])) == set(EVIDENCE_SOURCES)
                                and bool(request)
                                and set(self.read_sources.get(request["request_id"], [])) == set(EVIDENCE_SOURCES),
        }
        return {"verified": all(checks.values()), "checks": checks}

    def expected_report(self) -> dict[str, Any]:
        if not self.requests:
            raise ValueError("an executed request is required")
        expected = deepcopy(self.fixture["success_oracle"]["expected_report"])
        expected.update(
            request_id=self.requests[-1]["request_id"],
            inventory=[{"route": route, "access_class": access}
                       for route, access in self._expected_inventory().items()],
            evidence_sources=list(EVIDENCE_SOURCES),
        )
        return expected

    @property
    def probe_results(self) -> list[dict[str, Any]]:
        """Existing lifecycle name for completed controlled request results."""
        return self.requests

    @property
    def execution_ledger(self) -> list[dict[str, Any]]:
        """Existing lifecycle name for the environment effect ledger."""
        return self.effect_ledger

    @property
    def dependency_material(self) -> None:
        """No task fixture pre-provisions dependency material for a mechanism."""
        return None

    def close(self) -> None:
        """Match TaskEnvironment's lifecycle without allocating external resources."""

    def snapshot(self) -> dict[str, Any]:
        return deepcopy({
            "pool_version": POOL_VERSION,
            "interface_version": INTERFACE_VERSION,
            "fixture_id": self.fixture["fixture_id"],
            "snapshot_id": self.fixture["snapshot_id"],
            "instance_id": self.fixture["instance_id"],
            "actor": self.actor,
            "bindings": self.bindings,
            "descriptors": self.descriptors,
            "policies": self.policies,
            "objects": self.objects,
            "target": self.target,
            "observer_health": self.observer_health,
            "discovered": self.discovered,
            "read_records": self.read_records,
            "inventory": self.inventory,
            "current_plan": self.current_plan,
            "plan_history": self.plan_history,
            "request_count": self.request_count,
            "effect_ledger": self.effect_ledger,
            "execution_ledger": self.effect_ledger,
            "requests": self.requests,
            "probe_results": self.requests,
            "oracle_observations": self.requests,
            "read_sources": self.read_sources,
            "journals": {
                name: {"records": records, "enabled": self.observer_health[name]}
                for name, records in self.journals.items()
            },
            "reports": self.reports,
        })

    @classmethod
    def restore(cls, fixture: Mapping[str, Any], snapshot: Mapping[str, Any]) -> "MultiTaskEnvironment":
        env = cls(fixture)
        if (snapshot.get("fixture_id") != fixture["fixture_id"]
                or snapshot.get("snapshot_id") != fixture["snapshot_id"]
                or snapshot.get("instance_id") != fixture["instance_id"]):
            raise ValueError("snapshot identity differs from fixture")
        if (snapshot.get("execution_ledger") != snapshot.get("effect_ledger")
                or snapshot.get("probe_results") != snapshot.get("requests")):
            raise ValueError("lifecycle compatibility aliases differ from task state")
        for key in (
            "actor", "bindings", "descriptors", "policies", "objects", "target", "observer_health",
            "discovered", "read_records", "inventory", "current_plan", "plan_history", "request_count",
            "effect_ledger", "requests", "read_sources", "reports",
        ):
            setattr(env, key, deepcopy(snapshot[key]))
        env.journals = {
            name: deepcopy(snapshot["journals"][name]["records"])
            for name in EVIDENCE_SOURCES
        }
        if any(snapshot["journals"][name]["enabled"] != env.observer_health[name]
               for name in EVIDENCE_SOURCES):
            raise ValueError("journal availability differs from fixture observer state")
        return env

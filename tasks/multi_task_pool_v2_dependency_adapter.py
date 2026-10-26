"""Task-side DEPENDENCY_V4 compatibility for the frozen multi-task pool.

This module is deliberately separate from the v1 pool adapter.  It derives
the dependency material from each fixture's actual binding, descriptor,
policy, and request journals, then temporarily supplies that material to the
unchanged lifecycle DEPENDENCY ingress.  No Trap core, guard, State rule,
pre-effect rule, or tool operation is changed here.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import adapters.lifecycle_branch_runtime as branch_runtime_module
import adapters.lifecycle_mechanism_runtime as mechanism_runtime_module
import adapters.lifecycle_trap_bindings as trap_bindings_module
from adapters.lifecycle_mechanism_runtime import LifecycleMechanismRuntime
from adapters.lifecycle_state_collector import STATE_PROTOCOL_VERSION
from adapters.lifecycle_trap_bindings import DependencyBindingUnavailable
from adapters.lifecycle_v2_runtime import SAMPLE_ID as LEGACY_SAMPLE_ID
from adapters.lifecycle_v2_runtime import SAMPLE_ROUTE as LEGACY_SAMPLE_ROUTE
from adapters.lifecycle_v2_runtime import SOURCES
from adapters.lifecycle_web_behavior import web_behavior_evidence
from run_lifecycle_v2_feasibility import INTERFACE_VERSION as LIFECYCLE_INTERFACE_VERSION
from tasks.multi_task_pool_v1 import EVIDENCE_SOURCES, MultiTaskEnvironment, load_fixture
from tasks.multi_task_pool_v1_adapter import (
    FixtureRepository,
    MultiTaskSessionAdapter,
)
from traps.dependency_v4 import (
    ActionContract,
    FactNode,
    StateGroundedFactGraph,
    STATE_ROLES,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_ROOT = ROOT / "tasks" / "multi_task_pool_v1_fixtures"
ADAPTER_VERSION = "multi-task-dependency-compatibility-v2.0"
MULTITASK_MATERIAL_PROFILE = "multi-task-state-dependency-material-v2.0"
MATERIAL_READ_OPERATIONS = {
    "read-task-entry": "task_dependency_entry",
    "read-task-contract": "task_dependency_contract",
    "read-task-rule": "task_dependency_rule",
}
TASK_FAMILY_NAMES = {
    "WEB": "multi_task_web",
    "API": "multi_task_api",
    "SERVICE": "multi_task_service",
}
_TARGET_BINDING_LOCK = RLock()


def dependency_identity(fixture: Mapping[str, Any]) -> dict[str, str]:
    """Return the immutable V4 binding identity for one pool fixture."""
    family = str(fixture["family"])
    instance_id = str(fixture["instance_id"])
    template_id = str(fixture["template_id"])
    return {
        "task_instance_id": instance_id,
        "family": family,
        "task_family": TASK_FAMILY_NAMES[family],
        "variant": f"{template_id}:{instance_id}",
        "namespace": str(fixture["fixture_id"]),
        "fixture_namespace": str(fixture["fixture_id"]),
    }


def snapshot_dependency_identity(snapshot: Mapping[str, Any]) -> dict[str, str]:
    """Read the saved identity without requiring the fixture JSON itself."""
    required = ("task_instance_id", "family", "variant", "namespace")
    if not all(isinstance(snapshot.get(key), str) and snapshot.get(key) for key in required):
        raise ValueError("dependency snapshot lacks its saved identity")
    family = str(snapshot["family"])
    if family not in TASK_FAMILY_NAMES:
        raise ValueError("unknown multi-task dependency family")
    return {
        "task_instance_id": str(snapshot["task_instance_id"]),
        "family": family,
        "task_family": TASK_FAMILY_NAMES[family],
        "variant": str(snapshot["variant"]),
        "namespace": str(snapshot["namespace"]),
        "fixture_namespace": str(snapshot["namespace"]),
    }


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("record_id")
    if not isinstance(value, str) or not value:
        raise ValueError("multi-task dependency material requires record identity")
    return value


def _material_record(record: Mapping[str, Any], record_type: str, next_ref: str = "") -> dict[str, Any]:
    """Make a typed material view while preserving the real source record."""
    result = {
        "record_id": _record_id(record),
        "record_type": record_type,
        "source_record_type": record.get("record_type"),
        "source_record": deepcopy(dict(record)),
    }
    if next_ref:
        result["next_ref"] = next_ref
    return result


def _scope_record(identity: Mapping[str, str], scope: str, kind: str,
                  source_records: list[Mapping[str, Any]], next_ref: str = "") -> dict[str, Any]:
    """Build a real aggregate view from the selected fixture records.

    The scope chain is not a fabricated policy or backend.  It is an
    immutable projection of the route bindings, operation descriptors, and
    policies that are already present in the fixture.
    """
    suffix = {"entry": "bindings", "contract": "descriptors", "rule": "policies"}[kind]
    record = {
        "record_id": f"{identity['namespace']}/scope/{scope}/{suffix}",
        "record_type": {
            "entry": "task_dependency_entry",
            "contract": "task_dependency_contract",
            "rule": "task_dependency_rule",
        }[kind],
        "scope": scope,
        "source_record_ids": [_record_id(item) for item in source_records],
        "source_facts": deepcopy([dict(item) for item in source_records]),
        "projection": "fixture_binding_descriptor_policy_records",
    }
    if next_ref:
        record["next_ref"] = next_ref
    return record


def _add_candidate(records: dict[str, dict[str, Any]], candidates: list[dict[str, Any]],
                   material_id: str, state: str, operations: tuple[str, ...],
                   selector: Mapping[str, Any], consumer: str,
                   chain: list[dict[str, Any]], **metadata: Any) -> None:
    ids = [_record_id(row) for row in chain]
    for row in chain:
        existing = records.get(row["record_id"])
        if existing is not None and existing != row:
            raise ValueError(f"dependency material record collision: {row['record_id']}")
        records[row["record_id"]] = deepcopy(row)
    candidates.append({
        "material_id": material_id,
        "state": state,
        "operations": list(operations),
        "selector": deepcopy(dict(selector)),
        "consumer": consumer,
        "record_ids": ids,
        **deepcopy(metadata),
    })


def _target_chain(snapshot: Mapping[str, Any], route: str) -> list[dict[str, Any]]:
    bindings = snapshot["bindings"]
    binding = bindings.get(route)
    if not isinstance(binding, dict):
        return []
    descriptor = snapshot["descriptors"].get(binding.get("operation_descriptor_ref"))
    if not isinstance(descriptor, dict):
        return []
    policy = snapshot["policies"].get(descriptor.get("resource_policy_ref"))
    if not isinstance(policy, dict):
        return []
    return [
        _material_record(binding, "task_dependency_entry", _record_id(descriptor)),
        _material_record(descriptor, "task_dependency_contract", _record_id(policy)),
        _material_record(policy, "task_dependency_rule"),
    ]


def _scope_chain(snapshot: Mapping[str, Any], identity: Mapping[str, str], scope: str) -> list[dict[str, Any]]:
    routes = [
        row for row in snapshot["bindings"].values()
        if scope == "all" or row.get("advertised_access") == scope
    ]
    if not routes:
        return []
    descriptors = []
    policies = []
    for binding in routes:
        descriptor = snapshot["descriptors"].get(binding.get("operation_descriptor_ref"))
        if not isinstance(descriptor, dict):
            return []
        policy = snapshot["policies"].get(descriptor.get("resource_policy_ref"))
        if not isinstance(policy, dict):
            return []
        descriptors.append(descriptor)
        policies.append(policy)
    entry = _scope_record(identity, scope, "entry", routes)
    contract = _scope_record(identity, scope, "contract", descriptors)
    rule = _scope_record(identity, scope, "rule", policies)
    entry["next_ref"] = contract["record_id"]
    contract["next_ref"] = rule["record_id"]
    return [entry, contract, rule]


def build_dependency_material(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Build only material that can be traced to the current fixture state."""
    identity = dependency_identity(snapshot["fixture"] if "fixture" in snapshot else snapshot)
    target = snapshot["target"]
    route = target["route"]
    object_id = target["object_id"]
    records: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []

    for scope in ("public", "protected", "all"):
        chain = _scope_chain(snapshot, identity, scope)
        if chain:
            _add_candidate(
                records, candidates, f"recon-scope-{scope}", "RECON",
                ("collect-routes",), {"scope": scope}, "inventory_discovery", chain,
                real_scope_level=True,
            )

    for candidate_route in snapshot["bindings"]:
        chain = _target_chain(snapshot, candidate_route)
        if chain:
            _add_candidate(
                records, candidates, f"recon-route-{len(candidates) + 1}", "RECON",
                ("read-route-binding",), {"route": candidate_route}, "route_discovery", chain,
                real_single_target=True,
            )

    target_chain = _target_chain(snapshot, route)
    if target_chain and object_id in snapshot.get("objects", {}):
        _add_candidate(
            records, candidates, "probe-execution", "EXPLOITATION", ("probe-sample",),
            {"route": route, "object_id": object_id}, "probe_execution", target_chain,
            real_target=True,
        )
        _add_candidate(
            records, candidates, "request-verification", "VERIFICATION",
            ("read-evidence", "submit-report"), {"route": route, "object_id": object_id},
            "request_verification", target_chain, real_target=True,
        )

    return {
        "adapter_version": ADAPTER_VERSION,
        "profile_version": MULTITASK_MATERIAL_PROFILE,
        "provisioned_before_source": True,
        **identity,
        "records": records,
        "candidates": candidates,
        "source_fact_policy": "binding_descriptor_policy_request_journal_only",
        "request_journal_is_runtime_observation": True,
    }


def _candidate(material: Mapping[str, Any], consumer: str, selector: Mapping[str, Any],
               state: str, operation: str) -> dict[str, Any] | None:
    matches = [row for row in material["candidates"]
               if row.get("consumer") == consumer
               and row.get("state") == state
               and row.get("selector") == dict(selector)
               and operation in row.get("operations", [])]
    return matches[0] if len(matches) == 1 else None


def validate_dependency_consumer(snapshot: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    """Validate material against current real fixture facts and journals."""
    material = snapshot.get("dependency_material")
    if not isinstance(material, dict):
        return False
    identity = snapshot_dependency_identity(snapshot)
    if any(material.get(key) != value for key, value in (
            ("task_instance_id", identity["task_instance_id"]),
            ("family", identity["family"]),
            ("task_family", identity["task_family"]),
            ("variant", identity["variant"]),
            ("namespace", identity["namespace"]),
            ("fixture_namespace", identity["fixture_namespace"]),
    )):
        return False
    rows = [material["records"].get(identifier) for identifier in candidate.get("record_ids", [])]
    if len(rows) != 3 or any(not isinstance(row, dict) for row in rows):
        return False
    if any(row.get("record_type") != expected for row, expected in zip(
            rows, MATERIAL_READ_OPERATIONS.values())):
        return False
    if rows[0].get("next_ref") != rows[1].get("record_id") or rows[1].get("next_ref") != rows[2].get("record_id"):
        return False
    if "next_ref" in rows[2]:
        return False

    selector = candidate.get("selector", {})
    consumer = candidate.get("consumer")
    if consumer == "inventory_discovery":
        scope = selector.get("scope")
        expected = [row for row in snapshot["bindings"].values()
                    if scope == "all" or row.get("advertised_access") == scope]
        if scope not in {"public", "protected", "all"} or not expected:
            return False
        return bool(candidate.get("real_scope_level")) and rows[0].get("source_record_ids") == [
            _record_id(row) for row in expected
        ]
    if consumer in {"route_discovery", "probe_execution", "request_verification"}:
        route = selector.get("route")
        chain = _target_chain(snapshot, route) if isinstance(route, str) else []
        if not chain or [row.get("record_id") for row in rows] != [row.get("record_id") for row in chain]:
            return False
        if consumer in {"probe_execution", "request_verification"}:
            return selector.get("object_id") == snapshot["target"].get("object_id")
        return selector == {"route": route}
    return False


def _base_graph(snapshot: Mapping[str, Any], route: str, request_id: str | None = None) -> StateGroundedFactGraph:
    identity = snapshot_dependency_identity(snapshot)
    binding = snapshot["bindings"][route]
    descriptor = snapshot["descriptors"][binding["operation_descriptor_ref"]]
    policy = snapshot["policies"][descriptor["resource_policy_ref"]]
    nodes = (
        FactNode("D1", "route_binding", "read-route-binding", deepcopy(binding), "operation_descriptor_ref"),
        FactNode("D2", "operation_descriptor", "read-operation-descriptor", deepcopy(descriptor), "resource_policy_ref"),
        FactNode("D3", "resource_policy", "read-resource-policy", deepcopy(policy)),
    )
    contracts = {
        "RECON": ActionContract(identity["task_family"], "RECON", "web_task",
            "read-route-binding", "route", route, STATE_ROLES["RECON"], False,
            "route_contract", (), ("route",), {"route": route}, "committed_before_effect"),
        "PLANNING": ActionContract(identity["task_family"], "PLANNING", "web_task",
            "read-operation-descriptor", "ref", descriptor["record_id"], STATE_ROLES["PLANNING"], False,
            "planning_constraint", (), ("ref",), {"ref": descriptor["record_id"]}, "committed_before_effect"),
        "EXPLOITATION": ActionContract(identity["task_family"], "EXPLOITATION", "web_task",
            "probe-sample", "route", route, STATE_ROLES["EXPLOITATION"], True,
            "route_contract", ("operation_ref", "policy_ref"), ("route",), {"route": route},
            "committed_before_effect"),
        "VERIFICATION": ActionContract(identity["task_family"], "VERIFICATION", "web_task",
            "read-evidence", "request_id", request_id or "no-request-yet", STATE_ROLES["VERIFICATION"], False,
            "verification_evidence", (), ("request_id",), {"request_id": request_id or "no-request-yet"},
            "committed_before_effect"),
    }
    return StateGroundedFactGraph(
        task_instance_id=identity["task_instance_id"],
        task_family=identity["task_family"],
        declared_variant=identity["variant"],
        fixture_namespace=identity["fixture_namespace"],
        root_fact={"object": binding["record_id"], "route": route},
        nodes=nodes,
        contracts=contracts,
    )


class MultiTaskDependencyEnvironment(MultiTaskEnvironment):
    """v2 environment that adds only fixture-derived dependency material."""

    def __init__(self, fixture: Mapping[str, Any]):
        super().__init__(fixture)
        base_snapshot = super().snapshot()
        self._dependency_material = build_dependency_material({**base_snapshot, "fixture": self.fixture})

    @property
    def dependency_material(self) -> dict[str, Any]:
        return deepcopy(self._dependency_material)

    @property
    def dependency_context(self) -> dict[str, str]:
        identity = dependency_identity(self.fixture)
        return {
            "task_instance_id": identity["task_instance_id"],
            "family": identity["family"],
            "variant": identity["variant"],
            "namespace": identity["namespace"],
        }

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()
        identity = dependency_identity(self.fixture)
        snapshot.update({
            "task_instance_id": identity["task_instance_id"],
            "family": identity["family"],
            "variant": identity["variant"],
            "namespace": identity["namespace"],
            "dependency_context": self.dependency_context,
            "dependency_material": self.dependency_material,
            "initial_dependency_material": self.dependency_material,
        })
        return snapshot

    @classmethod
    def restore(cls, fixture: Mapping[str, Any], snapshot: Mapping[str, Any]) -> "MultiTaskDependencyEnvironment":
        base = MultiTaskEnvironment.restore(fixture, snapshot)
        env = cls(fixture)
        for key in (
            "actor", "bindings", "descriptors", "policies", "objects", "target", "observer_health",
            "discovered", "read_records", "inventory", "current_plan", "plan_history", "request_count",
            "effect_ledger", "requests", "read_sources", "reports",
        ):
            setattr(env, key, deepcopy(getattr(base, key)))
        env.journals = deepcopy(base.journals)
        expected = env.snapshot()
        for key in ("task_instance_id", "family", "variant", "namespace", "dependency_context"):
            if snapshot.get(key) != expected[key]:
                raise ValueError(f"dependency snapshot identity differs: {key}")
        if snapshot.get("dependency_material") != expected["dependency_material"]:
            raise ValueError("dependency material differs from fixture-derived material")
        if snapshot.get("initial_dependency_material") != expected["initial_dependency_material"]:
            raise ValueError("initial dependency material differs from fixture-derived material")
        return env


class MultiTaskDependencySessionAdapter(MultiTaskSessionAdapter):
    """Existing Session surface with a fixture-derived dependency payload."""

    adapter_version = ADAPTER_VERSION

    def __init__(self, provider: Any, fixture_id: str,
                 behavior_observer=web_behavior_evidence, *, repository: FixtureRepository | None = None):
        super().__init__(provider, fixture_id, behavior_observer, repository=repository)
        self.env = MultiTaskDependencyEnvironment(self.fixture)
        self.initial_dependency_material = self.env.dependency_material
        self.dependency_context = self.env.dependency_context

    def payload(self) -> dict[str, Any]:
        payload = super().payload()
        payload.update({
            "adapter_version": ADAPTER_VERSION,
            "task_instance_id": self.dependency_context["task_instance_id"],
            "family": self.dependency_context["family"],
            "variant": self.dependency_context["variant"],
            "namespace": self.dependency_context["namespace"],
            "dependency_context": deepcopy(self.dependency_context),
            "initial_dependency_material": deepcopy(self.initial_dependency_material),
        })
        return payload

    @classmethod
    def restore(cls, payload: Mapping[str, Any], provider: Any,
                behavior_observer=web_behavior_evidence) -> "MultiTaskDependencySessionAdapter":
        session = super().restore(payload, provider, behavior_observer)
        session.env = MultiTaskDependencyEnvironment.restore(session.fixture, payload["environment_snapshot"])
        session.initial_dependency_material = deepcopy(payload["initial_dependency_material"])
        session.dependency_context = deepcopy(payload["dependency_context"])
        if session.initial_dependency_material != session.env.dependency_material:
            raise ValueError("Session initial dependency material differs after restore")
        if session.dependency_context != session.env.dependency_context:
            raise ValueError("Session dependency context differs after restore")
        return session


def upgrade_checkpoint(fixture: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Attach v2 dependency fields without changing historical task state."""
    upgraded = deepcopy(dict(checkpoint))
    environment = MultiTaskDependencyEnvironment(fixture)
    snapshot = deepcopy(dict(checkpoint["environment_snapshot"]))
    identity = dependency_identity(fixture)
    snapshot.update({
        "task_instance_id": identity["task_instance_id"],
        "family": identity["family"],
        "variant": identity["variant"],
        "namespace": identity["namespace"],
        "dependency_context": environment.dependency_context,
        "dependency_material": environment.dependency_material,
        "initial_dependency_material": environment.dependency_material,
    })
    MultiTaskDependencyEnvironment.restore(fixture, snapshot)
    upgraded["environment_snapshot"] = snapshot
    upgraded["adapter_version"] = ADAPTER_VERSION
    upgraded["task_instance_id"] = identity["task_instance_id"]
    upgraded["family"] = identity["family"]
    upgraded["variant"] = identity["variant"]
    upgraded["namespace"] = identity["namespace"]
    upgraded["dependency_context"] = environment.dependency_context
    upgraded["initial_dependency_material"] = environment.dependency_material
    return upgraded


@contextmanager
def dependency_compatibility_context(snapshot: Mapping[str, Any]):
    """Route frozen lifecycle ingress through this task-side material bridge."""
    identity = {
        "task_instance_id": snapshot["task_instance_id"],
        "family": snapshot["family"],
        "task_family": TASK_FAMILY_NAMES[snapshot["family"]],
        "variant": snapshot["variant"],
        "namespace": snapshot["namespace"],
        "fixture_namespace": snapshot["namespace"],
    }
    route = snapshot["target"]["route"]
    object_id = snapshot["target"]["object_id"]
    material = deepcopy(snapshot["dependency_material"])

    def provision(_bindings, sample_route, sample_id):
        if sample_route != route or sample_id != object_id:
            raise ValueError("multi-task dependency target context differs")
        return deepcopy(material)

    def validate(snapshot_value, candidate):
        return validate_dependency_consumer(snapshot_value, candidate)

    def graph_for(snapshot_value, *, route=route, request_id=None):
        return _base_graph(snapshot_value, route=route, request_id=request_id)

    assignments = [
        (branch_runtime_module, "Session", MultiTaskDependencySessionAdapter),
        (trap_bindings_module, "SAMPLE_ROUTE", route),
        (trap_bindings_module, "SAMPLE_ID", object_id),
        (trap_bindings_module, "MATERIAL_PROFILE", MULTITASK_MATERIAL_PROFILE),
        (trap_bindings_module, "provision_material", provision),
        (trap_bindings_module, "validate_consumer", validate),
        (trap_bindings_module, "graph_for", graph_for),
        (mechanism_runtime_module, "dependency_binding", trap_bindings_module.dependency_binding),
    ]
    with _TARGET_BINDING_LOCK:
        previous = [(module, name, getattr(module, name)) for module, name, _ in assignments]
        try:
            for module, name, value in assignments:
                setattr(module, name, value)
            yield identity
        finally:
            for module, name, value in reversed(previous):
                setattr(module, name, value)


def restore_dependency_runtime(checkpoint: Mapping[str, Any], provider: Any):
    """Restore the unchanged DEPENDENCY runtime through the task adapter."""
    snapshot = checkpoint["environment_snapshot"]
    with dependency_compatibility_context(snapshot):
        return LifecycleMechanismRuntime.restore(
            deepcopy(dict(checkpoint)), provider, condition="DEPENDENCY",
        )


__all__ = [
    "ADAPTER_VERSION",
    "MULTITASK_MATERIAL_PROFILE",
    "FixtureRepository",
    "MultiTaskDependencyEnvironment",
    "MultiTaskDependencySessionAdapter",
    "build_dependency_material",
    "dependency_compatibility_context",
    "dependency_identity",
    "restore_dependency_runtime",
    "upgrade_checkpoint",
    "validate_dependency_consumer",
]

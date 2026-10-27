"""Full-provenance task records and session context for the current scoped environment."""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
from typing import Any, Mapping
import adapters.lifecycle_branch_runtime as branch_runtime_module
import adapters.lifecycle_mechanism_runtime as mechanism_runtime_module
import adapters.lifecycle_trap_bindings as trap_bindings_module
from adapters.lifecycle_dependency_material import material_records
from adapters.lifecycle_web_behavior import web_behavior_evidence
from tasks.multi_task_pool_v1 import EVIDENCE_SOURCES, MultiTaskEnvironment
from tasks.multi_task_pool_v1_adapter import FixtureRepository, MultiTaskSessionAdapter
from tasks.multi_task_pool_v2_dependency_adapter import dependency_identity, snapshot_dependency_identity
from core.dependency import ActionContract, FactNode, STATE_ROLES, StateGroundedFactGraph
from tasks.dependency_material import FULL_RECORD_MATERIAL_VERSION, DEPENDENCY_PROFILE_VERSION, STATE_MATERIAL_SEMANTICS, validate_v5_graph

ADAPTER_VERSION = "multi-task-dependency-v5-adapter-v1.0"


MATERIAL_PROFILE = FULL_RECORD_MATERIAL_VERSION


READ_OPERATIONS = {
    "read-task-entry": "task_dependency_entry",
    "read-task-contract": "task_dependency_contract",
    "read-task-rule": "task_dependency_rule",
}


_TARGET_BINDING_LOCK = RLock()


def _stable_source(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return only pre-source fixture configuration used to provision V5."""
    return {
        "fixture_id": snapshot["fixture_id"],
        "instance_id": snapshot["instance_id"],
        "actor": deepcopy(snapshot["actor"]),
        "bindings": deepcopy(snapshot["bindings"]),
        "descriptors": deepcopy(snapshot["descriptors"]),
        "policies": deepcopy(snapshot["policies"]),
        "objects": deepcopy(snapshot["objects"]),
        "target": deepcopy(snapshot["target"]),
        "observer_health": deepcopy(snapshot["observer_health"]),
    }


def _fact_ref(record: Mapping[str, Any]) -> str:
    value = record.get("record_id")
    if not isinstance(value, str) or not value:
        raise ValueError("V5 source record lacks record_id")
    return value


def _record(
    record_id: str,
    record_type: str,
    semantic: str,
    source_refs: list[str],
    source_facts: Any,
    values: Mapping[str, Any],
    next_ref: str = "",
) -> dict[str, Any]:
    row = {
        "record_id": record_id,
        "record_type": record_type,
        "v5_semantic": semantic,
        "preprovisioned_before_source": True,
        "source_fact_refs": list(source_refs),
        "source_facts": deepcopy(source_facts),
        **deepcopy(dict(values)),
    }
    if next_ref:
        row["next_ref"] = next_ref
    return row


def _add_chain(
    material: dict[str, Any],
    *,
    name: str,
    state: str,
    operations: tuple[str, ...],
    selector: Mapping[str, Any],
    consumer: str,
    nodes: tuple[tuple[list[str], Any, Mapping[str, Any]], ...],
) -> None:
    semantics = STATE_MATERIAL_SEMANTICS[state]
    ids = [f"{material['namespace']}/dependency-v5/{name}/{suffix}"
           for suffix in ("d1", "d2", "d3")]
    for index, ((operation, record_type), semantic, node) in enumerate(
        zip(READ_OPERATIONS.items(), semantics, nodes)
    ):
        source_refs, source_facts, values = node
        row = _record(
            ids[index], record_type, semantic, source_refs, source_facts, values,
            ids[index + 1] if index < 2 else "",
        )
        material["records"][row["record_id"]] = row
    material["candidates"].append({
        "material_id": name,
        "state": state,
        "operations": list(operations),
        "selector": deepcopy(dict(selector)),
        "consumer": consumer,
        "record_ids": ids,
    })


def build_v5_dependency_material(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Provision State-specific chains solely from stable fixture facts."""
    source = _stable_source(snapshot)
    identity = snapshot_dependency_identity(snapshot)
    material: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION,
        "profile_version": MATERIAL_PROFILE,
        "mechanism_profile": DEPENDENCY_PROFILE_VERSION,
        "provisioned_before_source": True,
        **identity,
        "records": {},
        "candidates": [],
        "source_policy": "stable_fixture_configuration_only",
        "runtime_request_ids_preprovisioned": False,
    }

    for scope in ("public", "protected", "all"):
        bindings = [row for row in source["bindings"].values()
                    if scope == "all" or row["advertised_access"] == scope]
        if not bindings:
            continue
        descriptors = [source["descriptors"][row["operation_descriptor_ref"]] for row in bindings]
        policies = [source["policies"][row["resource_policy_ref"]] for row in descriptors]
        _add_chain(
            material,
            name=f"recon-scope-{scope}",
            state="RECON",
            operations=("collect-routes",),
            selector={"scope": scope},
            consumer="inventory_discovery",
            nodes=(
                ([_fact_ref(row) for row in bindings], bindings,
                 {"scope": scope, "registry": "bindings", "selector": "advertised_access"}),
                ([_fact_ref(row) for row in descriptors], descriptors,
                 {"namespace": source["fixture_id"], "projection": ["route", "advertised_access"]}),
                ([_fact_ref(row) for row in policies], policies,
                 {"allowed_access_classes": ["public", "protected"],
                  "require_unique_routes": True, "require_live_descriptor": True}),
            ),
        )

    route = source["target"]["route"]
    object_id = source["target"]["object_id"]
    binding = source["bindings"][route]
    descriptor = source["descriptors"][binding["operation_descriptor_ref"]]
    policy = source["policies"][descriptor["resource_policy_ref"]]
    object_record = source["objects"][object_id]
    actor_ref = f"{source['fixture_id']}#actor/{source['actor']['id']}"
    observer_refs = [f"{source['fixture_id']}#observer/{name}" for name in EVIDENCE_SOURCES]

    _add_chain(
        material,
        name="execution-target",
        state="EXPLOITATION",
        operations=("probe-sample",),
        selector={"route": route, "object_id": object_id},
        consumer="probe_execution",
        nodes=(
            ([_fact_ref(binding), f"{source['fixture_id']}#object/{object_id}"],
             {"binding": binding, "object": object_record},
             {"route": route, "object_id": object_id, "deployment": binding["deployment"]}),
            ([_fact_ref(descriptor), _fact_ref(policy)],
             {"descriptor": descriptor, "policy": policy},
             {"method": descriptor["method"], "operation_ref": descriptor["record_id"],
              "policy_ref": policy["record_id"]}),
            ([actor_ref, *observer_refs],
             {"actor": source["actor"], "observer_health": source["observer_health"]},
             {"actor_id": source["actor"]["id"], "object_must_exist": True,
              "controlled_effect": "fixture_request_dispatch",
              "required_observers": list(EVIDENCE_SOURCES)}),
        ),
    )

    _add_chain(
        material,
        name="verification-trace",
        state="VERIFICATION",
        operations=("read-evidence", "submit-report"),
        selector={"route": route, "object_id": object_id},
        consumer="request_verification",
        nodes=(
            ([_fact_ref(binding), f"{source['fixture_id']}#object/{object_id}"],
             {"binding": binding, "target": source["target"]},
             {"route": route, "object_id": object_id,
              "request_origin": "completed_fixture_request"}),
            (observer_refs,
             {"observer_health": source["observer_health"]},
             {"producers": list(EVIDENCE_SOURCES), "request_join_key": "request_id",
              "require_enabled_producer": True}),
            ([_fact_ref(descriptor), _fact_ref(policy), actor_ref],
             {"descriptor": descriptor, "policy": policy, "actor": source["actor"]},
             {"require_same_request": True, "require_same_actor": True,
              "require_same_object": True, "require_current_deployment": True,
              "required_grounding_events": {"gateway": "dispatched", "policy": "decision"}}),
        ),
    )
    return material


def _candidate(
    material: Mapping[str, Any], state: str, operation: str, selector: Mapping[str, Any]
) -> dict[str, Any]:
    matches = [row for row in material["candidates"]
               if row["state"] == state and operation in row["operations"]
               and row["selector"] == dict(selector)]
    if len(matches) != 1:
        raise ValueError("ordinary operation lacks one exact V5 configuration chain")
    return matches[0]


def validate_v5_consumer(snapshot: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    """Prove the selected records equal a fresh derivation from fixture state."""
    try:
        expected = build_v5_dependency_material(snapshot)
        actual = snapshot["dependency_material"]
        if actual != expected or snapshot.get("initial_dependency_material") != actual:
            return False
        expected_candidate = next(row for row in expected["candidates"]
                                  if row["material_id"] == candidate.get("material_id"))
        if expected_candidate != candidate:
            return False
        records = material_records(actual, candidate)
        return (
            tuple(row["v5_semantic"] for row in records)
            == STATE_MATERIAL_SEMANTICS[candidate["state"]]
            and all(row["preprovisioned_before_source"] is True for row in records)
            and all(row["source_fact_refs"] for row in records)
        )
    except (KeyError, StopIteration, TypeError, ValueError):
        return False


class MultiTaskDependencyV5Environment(MultiTaskEnvironment):
    """Normal environment plus immutable, operation-consumed V5 configuration."""

    def __init__(self, fixture: Mapping[str, Any]):
        super().__init__(fixture)
        base = super().snapshot()
        identity = dependency_identity(fixture)
        base.update({
            "task_instance_id": identity["task_instance_id"],
            "family": identity["family"],
            "variant": identity["variant"],
            "namespace": identity["namespace"],
        })
        self._dependency_material = build_v5_dependency_material(base)
        self.dependency_reads: list[dict[str, Any]] = []
        self.dependency_consumption_ledger: list[dict[str, Any]] = []

    @property
    def dependency_material(self) -> dict[str, Any]:
        return deepcopy(self._dependency_material)

    @property
    def dependency_context(self) -> dict[str, str]:
        identity = dependency_identity(self.fixture)
        return {key: identity[key] for key in ("task_instance_id", "family", "variant", "namespace")}

    def _snapshot_for_validation(self) -> dict[str, Any]:
        snapshot = super().snapshot()
        snapshot.update(self.dependency_context)
        snapshot["dependency_material"] = self.dependency_material
        snapshot["initial_dependency_material"] = self.dependency_material
        return snapshot

    def _consume(self, state: str, operation: str, selector: Mapping[str, Any]) -> None:
        candidate = _candidate(self._dependency_material, state, operation, selector)
        if not validate_v5_consumer(self._snapshot_for_validation(), candidate):
            raise ValueError("V5 configuration no longer matches fixture state")
        self.dependency_consumption_ledger.append({
            "state": state,
            "operation": operation,
            "selector": deepcopy(dict(selector)),
            "material_id": candidate["material_id"],
            "record_ids": deepcopy(candidate["record_ids"]),
        })

    def call(self, args: Mapping[str, Any]) -> dict[str, Any]:
        operation = args.get("operation")
        if operation in READ_OPERATIONS:
            ref = args.get("ref")
            record = self._dependency_material["records"].get(ref)
            if not isinstance(record, dict) or record.get("record_type") != READ_OPERATIONS[operation]:
                raise ValueError("V5 dependency record not found for this read operation")
            self.dependency_reads.append({"operation": operation, "record_id": ref})
            return deepcopy(record)
        if operation == "collect-routes":
            scope = args.get("scope", "all")
            self._consume("RECON", operation, {"scope": scope})
        elif operation == "probe-sample":
            self._consume("EXPLOITATION", operation, {
                "route": args.get("route"), "object_id": args.get("object_id"),
            })
        elif operation in {"read-evidence", "submit-report"}:
            self._consume("VERIFICATION", operation, deepcopy(self.target))
        return super().call(args)

    def snapshot(self) -> dict[str, Any]:
        snapshot = super().snapshot()
        snapshot.update(self.dependency_context)
        snapshot.update({
            "adapter_version": ADAPTER_VERSION,
            "dependency_context": self.dependency_context,
            "dependency_material": self.dependency_material,
            "initial_dependency_material": self.dependency_material,
            "dependency_reads": deepcopy(self.dependency_reads),
            "dependency_consumption_ledger": deepcopy(self.dependency_consumption_ledger),
        })
        return snapshot

    @classmethod
    def restore(
        cls, fixture: Mapping[str, Any], snapshot: Mapping[str, Any]
    ) -> "MultiTaskDependencyV5Environment":
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
        for key in (
            "task_instance_id", "family", "variant", "namespace", "dependency_context",
            "dependency_material", "initial_dependency_material",
        ):
            if snapshot.get(key) != expected[key]:
                raise ValueError(f"V5 dependency snapshot differs: {key}")
        env.dependency_reads = deepcopy(snapshot.get("dependency_reads", []))
        env.dependency_consumption_ledger = deepcopy(snapshot.get("dependency_consumption_ledger", []))
        return env


class MultiTaskDependencyV5SessionAdapter(MultiTaskSessionAdapter):
    adapter_version = ADAPTER_VERSION

    def __init__(self, provider: Any, fixture_id: str,
                 behavior_observer=web_behavior_evidence, *, repository: FixtureRepository | None = None):
        super().__init__(provider, fixture_id, behavior_observer, repository=repository)
        self.env = MultiTaskDependencyV5Environment(self.fixture)
        self.initial_dependency_material = self.env.dependency_material
        self.dependency_context = self.env.dependency_context

    def payload(self) -> dict[str, Any]:
        payload = super().payload()
        payload.update({
            "adapter_version": ADAPTER_VERSION,
            **deepcopy(self.dependency_context),
            "dependency_context": deepcopy(self.dependency_context),
            "initial_dependency_material": deepcopy(self.initial_dependency_material),
        })
        return payload

    @classmethod
    def restore(cls, payload: Mapping[str, Any], provider: Any,
                behavior_observer=web_behavior_evidence, *,
                repository: FixtureRepository | None = None
                ) -> "MultiTaskDependencyV5SessionAdapter":
        if payload["model"] != provider.model:
            raise ValueError("model mismatch")
        snapshot = payload["environment_snapshot"]
        fixture_id = snapshot.get("fixture_id")
        if not isinstance(fixture_id, str):
            raise ValueError("V5 checkpoint lacks fixture_id")
        session = cls(provider, fixture_id, behavior_observer, repository=repository)
        session.env = MultiTaskDependencyV5Environment.restore(
            session.fixture, snapshot
        )
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
        session.initial_dependency_material = deepcopy(payload["initial_dependency_material"])
        session.dependency_context = deepcopy(payload["dependency_context"])
        if session.initial_dependency_material != session.env.dependency_material:
            raise ValueError("V5 initial material differs after Session restore")
        return session


def _base_graph(
    snapshot: Mapping[str, Any], route: str, request_id: str | None = None
) -> StateGroundedFactGraph:
    identity = snapshot_dependency_identity(snapshot)
    binding = snapshot["bindings"][route]
    descriptor = snapshot["descriptors"][binding["operation_descriptor_ref"]]
    policy = snapshot["policies"][descriptor["resource_policy_ref"]]
    nodes = (
        FactNode("D1", "route_binding", "read-route-binding", deepcopy(binding), "operation_descriptor_ref"),
        FactNode("D2", "operation_descriptor", "read-operation-descriptor", deepcopy(descriptor),
                 "resource_policy_ref"),
        FactNode("D3", "resource_policy", "read-resource-policy", deepcopy(policy)),
    )
    contracts = {
        "RECON": ActionContract(identity["task_family"], "RECON", "web_task", "collect-routes",
            "scope", "all", STATE_ROLES["RECON"], False, "discovery_configuration", (),
            ("scope",), {"scope": "all"}, "committed_before_effect"),
        "PLANNING": ActionContract(identity["task_family"], "PLANNING", "web_task",
            "read-operation-descriptor", "ref", descriptor["record_id"], STATE_ROLES["PLANNING"],
            False, "planning_constraint", (), ("ref",), {"ref": descriptor["record_id"]},
            "committed_before_effect"),
        "EXPLOITATION": ActionContract(identity["task_family"], "EXPLOITATION", "web_task",
            "probe-sample", "route", route, STATE_ROLES["EXPLOITATION"], True,
            "execution_configuration", ("probe_execution_configuration",), ("route",),
            {"route": route}, "committed_before_effect"),
        "VERIFICATION": ActionContract(identity["task_family"], "VERIFICATION", "web_task",
            "read-evidence", "request_id", request_id or "no-request-yet",
            STATE_ROLES["VERIFICATION"], False, "trace_configuration", (), ("request_id",),
            {"request_id": request_id or "no-request-yet"}, "committed_before_effect"),
    }
    return StateGroundedFactGraph(
        identity["task_instance_id"], identity["task_family"], identity["variant"],
        identity["fixture_namespace"], {"object": binding["record_id"], "route": route},
        nodes, contracts,
    )


@contextmanager
def dependency_v5_context(
    snapshot: Mapping[str, Any], repository: FixtureRepository | None = None
):
    """Route frozen binding/guard logic through the V5 task-side records."""
    route = snapshot["target"]["route"]
    object_id = snapshot["target"]["object_id"]
    material = deepcopy(snapshot["dependency_material"])
    frozen_binding = trap_bindings_module.dependency_binding

    def provision(_bindings, sample_route, sample_id):
        if sample_route != route or sample_id != object_id:
            raise ValueError("V5 target context differs from fixture")
        return deepcopy(material)

    def graph_for(snapshot_value, *, route=route, request_id=None):
        return _base_graph(snapshot_value, route, request_id)

    def v5_binding(checkpoint, *, active=False):
        graph, action = frozen_binding(checkpoint, active=active)
        validate_v5_graph(graph, checkpoint["state"])
        return graph, action

    assignments = [
        (branch_runtime_module, "Session", MultiTaskDependencyV5SessionAdapter),
        (trap_bindings_module, "SAMPLE_ROUTE", route),
        (trap_bindings_module, "SAMPLE_ID", object_id),
        (trap_bindings_module, "MATERIAL_PROFILE", MATERIAL_PROFILE),
        (trap_bindings_module, "READ_OPERATIONS", READ_OPERATIONS),
        (trap_bindings_module, "provision_material", provision),
        (trap_bindings_module, "validate_consumer", validate_v5_consumer),
        (trap_bindings_module, "graph_for", graph_for),
        (mechanism_runtime_module, "dependency_binding", v5_binding),
        
    ]
    with _TARGET_BINDING_LOCK:
        previous_repository = getattr(MultiTaskDependencyV5SessionAdapter, "repository", None)
        previous = [(module, name, getattr(module, name)) for module, name, _ in assignments]
        try:
            if repository is not None:
                MultiTaskDependencyV5SessionAdapter.repository = repository
            for module, name, value in assignments:
                setattr(module, name, value)
            yield
        finally:
            for module, name, value in reversed(previous):
                setattr(module, name, value)
            if repository is not None:
                MultiTaskDependencyV5SessionAdapter.repository = previous_repository


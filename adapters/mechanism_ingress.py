"""Explicit Layer-3 ingress for frozen mechanism application types.

Legacy adapters delegate to the historical builders.  New adapters supply
task material to the same application types and V4 transition validator.
No global registry is patched and no implicit family fallback is used.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Mapping, Protocol

from adapters.web_route_coverage import WebRouteCoverageAdapter
from adapters.v2_providers import (
    LegacyCandidateProvider, WebCandidateProvider, WebEvidenceProvider,
)
from core.v1_types import H1V1_TRAP_BUDGET
from traps.dependency_v4 import (
    DEPENDENCY_V4_APPLICABLE,
    DEPENDENCY_V4_NOT_APPLICABLE,
    DEPENDENCY_V4_PROFILE_VERSION,
    STATE_ROLES,
    STAGE_STOP_REASONS,
    StateGroundedFactGraph,
    V4ApplicabilityDecision,
    assess_v4_applicability,
    build_dependency_v4_application,
    build_dependency_v4_profile_application,
    load_state_grounded_fact_graph,
)
from traps.iteration_v6 import load_taxonomy, validate_hard_task
from traps.iteration_v6_incremental import IterationV6IncrementalSession
from traps.v1 import STATE_TOOL, TASK_CANDIDATES, TASK_SUBJECT
from traps.v2 import (
    ArchitectureGap,
    V2TrapApplication,
    build_v2_trap_application,
    FIXTURE_ROOT,
)


STATES = tuple(STATE_ROLES)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    label: str
    target: str
    distinguishing_field: str
    carrier: str


class CandidateCatalog(Protocol):
    task_id: str
    subject: str
    tool_name: str

    def get_candidates(self) -> tuple[Candidate, ...]: ...
    def validate_candidate(self, candidate: Candidate) -> bool: ...


class LegacyBranchAdapter:
    def __init__(self, task_id: str) -> None:
        if task_id not in TASK_CANDIDATES:
            raise ArchitectureGap("unknown legacy BRANCH task")
        self.task_id = task_id
        self.subject = TASK_SUBJECT[task_id]
        self.tool_name = ""

    def get_candidates(self) -> tuple[Candidate, ...]:
        return tuple(Candidate(key, label, self.task_id, key, "legacy-local-tool")
                     for key, label in TASK_CANDIDATES[self.task_id])

    def validate_candidate(self, candidate: Candidate) -> bool:
        return bool(candidate.candidate_id and candidate.label and candidate.target == self.task_id)

    def build_legacy(self, context_id: str, state: str, order: tuple[str, ...] | None = None) -> V2TrapApplication:
        context = {"candidate_order": list(order)} if order is not None else None
        return build_v2_trap_application("BRANCH_EXPANSION", context_id, state, self.task_id, context)


class WebBranchAdapter:
    def __init__(self, task: WebRouteCoverageAdapter) -> None:
        self.task = task
        self.task_id = task.task_id
        self.subject = "local Web route coverage"
        self.tool_name = "web_route_tool"

    def get_candidates(self) -> tuple[Candidate, ...]:
        return tuple(Candidate(item.candidate_id, item.label, item.target,
                               item.distinguishing_field, item.carrier)
                     for item in self.task.branch_candidates())

    def validate_candidate(self, candidate: Candidate) -> bool:
        return (candidate.target in self.task.target_objects()
                and bool(candidate.candidate_id and candidate.label and candidate.carrier
                         and candidate.distinguishing_field))


def _validated_candidates(catalog: CandidateCatalog) -> tuple[Candidate, ...]:
    items = tuple(catalog.get_candidates())
    if len(items) != H1V1_TRAP_BUDGET:
        raise ArchitectureGap("BRANCH_V2 requires exactly four candidates")
    if not all(catalog.validate_candidate(item) for item in items):
        raise ArchitectureGap("BRANCH candidate is not legal for this task")
    if len({item.candidate_id for item in items}) != 4 or len({item.distinguishing_field for item in items}) != 4:
        raise ArchitectureGap("BRANCH candidates are not behaviorally distinct")
    return items


def build_branch_from_catalog(catalog: CandidateCatalog, state: str,
                              order: tuple[str, ...] | None = None) -> V2TrapApplication:
    return build_branch(catalog, "adapter-unit-only", state, order)


def build_branch(catalog: CandidateCatalog, context_id: str, state: str,
                 order: tuple[str, ...] | None = None) -> V2TrapApplication:
    if isinstance(catalog, LegacyBranchAdapter):
        _validated_candidates(catalog)
        provider = LegacyCandidateProvider(catalog.task_id)
    elif isinstance(catalog, WebBranchAdapter):
        _validated_candidates(catalog)
        provider = WebCandidateProvider(catalog.task)
    else:
        raise ArchitectureGap("BRANCH adapter is not explicitly registered")
    context = {"candidate_provider": provider}
    if order is not None:
        context["candidate_order"] = list(order)
    return build_v2_trap_application("BRANCH_EXPANSION", context_id, state,
                                     catalog.task_id, context)


class DependencyAdapter(Protocol):
    task_id: str

    def get_nodes(self) -> tuple[Any, ...]: ...
    def get_edges(self) -> tuple[tuple[str, str], ...]: ...
    def get_committed_action(self, state: str, context_id: str) -> Mapping[str, Any]: ...
    def get_graph(self) -> StateGroundedFactGraph: ...
    def get_environment(self) -> Mapping[str, Any]: ...


class LegacyDependencyAdapter:
    def __init__(self, task_id: str, instance_id: str, parameters: Mapping[str, Any],
                 actions: Mapping[str, Mapping[str, Any]], environment: Mapping[str, Any]) -> None:
        self.task_id = task_id
        self.instance_id = instance_id
        self.parameters = dict(parameters)
        self.actions = actions
        self.environment = environment

    def get_graph(self) -> StateGroundedFactGraph:
        return load_state_grounded_fact_graph(self.instance_id, self.task_id, self.parameters)

    def get_nodes(self) -> tuple[Any, ...]:
        return self.get_graph().nodes

    def get_edges(self) -> tuple[tuple[str, str], ...]:
        return _edges(self.get_graph())

    def get_committed_action(self, state: str, context_id: str) -> Mapping[str, Any]:
        return self.actions[state]

    def get_environment(self) -> Mapping[str, Any]:
        return self.environment


class WebDependencyAdapter:
    def __init__(self, task: WebRouteCoverageAdapter) -> None:
        self.task = task
        self.task_id = task.task_id

    def get_graph(self) -> StateGroundedFactGraph:
        return self.task.dependency_graph()

    def get_nodes(self) -> tuple[Any, ...]:
        return self.get_graph().nodes

    def get_edges(self) -> tuple[tuple[str, str], ...]:
        return _edges(self.get_graph())

    def get_committed_action(self, state: str, context_id: str) -> Mapping[str, Any]:
        return self.task.dependency_action(state, context_id)

    def get_environment(self) -> Mapping[str, Any]:
        return self.task.dependency_environment()


def _edges(graph: StateGroundedFactGraph) -> tuple[tuple[str, str], ...]:
    return tuple((node.fact_id, graph.nodes[index + 1].fact_id)
                 for index, node in enumerate(graph.nodes[:-1])
                 if node.child_reference_field
                 and node.record.get(node.child_reference_field) == graph.nodes[index + 1].record.get("record_id"))


def validate_dependency_graph(graph: StateGroundedFactGraph) -> None:
    nodes = graph.nodes
    if len(nodes) != 3 or tuple(node.fact_id for node in nodes) != ("D1", "D2", "D3"):
        raise ArchitectureGap("DEPENDENCY_V4 requires D1 D2 D3 at depth three")
    if not graph.task_instance_id or not graph.task_family or not graph.fixture_namespace:
        raise ArchitectureGap("DEPENDENCY_V4 graph lacks explicit task binding")
    if graph.root_fact.get("object") != nodes[0].record.get("record_id"):
        raise ArchitectureGap("DEPENDENCY_V4 root fact does not point to D1")
    for current, following in zip(nodes, nodes[1:]):
        if (not current.node_type or not current.read_operation or not current.child_reference_field
                or current.record.get("record_type") != current.node_type
                or current.record.get(current.child_reference_field) != following.record.get("record_id")):
            raise ArchitectureGap("DEPENDENCY_V4 has a broken prerequisite edge")
    if nodes[2].child_reference_field or any(not node.record.get("record_id") for node in nodes):
        raise ArchitectureGap("DEPENDENCY_V4 terminal node or record identity is invalid")
    if len({node.read_operation for node in nodes}) != 3:
        raise ArchitectureGap("DEPENDENCY_V4 requires distinct read operations")
    if set(graph.contracts) != set(STATE_ROLES):
        raise ArchitectureGap("DEPENDENCY_V4 requires four State contracts")
    for state, contract in graph.contracts.items():
        if (contract.state != state or contract.task_family != graph.task_family
                or contract.causal_role != STATE_ROLES[state]
                or contract.blocking != (state == "EXPLOITATION")
                or contract.execution_phase != "committed_before_effect"
                or not contract.tool_name or not contract.action_operation
                or not contract.object_argument or not contract.object_value
                or not contract.affected_field or not contract.required_arguments
                or contract.object_argument not in contract.required_arguments
                or contract.expected_arguments.get(contract.object_argument) != contract.object_value):
            raise ArchitectureGap("DEPENDENCY_V4 State causal contract is invalid")
        if state == "EXPLOITATION" and not contract.guard_inputs:
            raise ArchitectureGap("DEPENDENCY_V4 execution guard lacks inputs")
        if not STAGE_STOP_REASONS.get(state):
            raise ArchitectureGap("DEPENDENCY_V4 stopping role is unavailable")


def build_dependency(
    adapter: DependencyAdapter,
    context_id: str,
    state: str,
    *,
    profile_version: str = DEPENDENCY_V4_PROFILE_VERSION,
) -> V4ApplicabilityDecision:
    if not isinstance(adapter, (LegacyDependencyAdapter, WebDependencyAdapter)):
        raise ArchitectureGap("DEPENDENCY adapter is not explicitly registered")
    graph = adapter.get_graph()
    validate_dependency_graph(graph)
    if tuple(adapter.get_nodes()) != graph.nodes or tuple(adapter.get_edges()) != _edges(graph):
        raise ArchitectureGap("DEPENDENCY adapter graph views are inconsistent")
    action = adapter.get_committed_action(state, context_id)
    environment = adapter.get_environment()
    if isinstance(adapter, LegacyDependencyAdapter):
        return build_dependency_v4_application(context_id, state, adapter.task_id,
                                               adapter.instance_id, adapter.parameters,
                                               action, environment,
                                               profile_version=profile_version)
    contract, reason = assess_v4_applicability(graph, state, action, environment)
    if contract is None:
        return V4ApplicabilityDecision(status=DEPENDENCY_V4_NOT_APPLICABLE, reason=reason,
                                       causal_role=STATE_ROLES.get(state, ""))
    if action.get("checkpoint_id") != context_id:
        return V4ApplicabilityDecision(status=DEPENDENCY_V4_NOT_APPLICABLE,
                                       reason="checkpoint_action_mismatch", causal_role=contract.causal_role,
                                       contract=contract)
    app = build_dependency_v4_profile_application(
        graph, contract, adapter.task_id, action,
        profile_version=profile_version,
        task_adapter_id=adapter.task.adapter_id,
    )
    return V4ApplicabilityDecision(DEPENDENCY_V4_APPLICABLE, reason, contract.causal_role, app, contract)


@dataclass(frozen=True)
class EvidenceRecord:
    source_id: str
    backend_id: str
    channel: str
    source_type: str
    independence_basis: str
    observation: str
    support: str
    carrier: str


class EvidenceBundle(Protocol):
    task_id: str
    subject: str
    tool_name: str

    def get_sources(self) -> tuple[EvidenceRecord, ...]: ...
    def get_claim(self) -> str: ...
    def get_gap(self) -> str: ...
    def get_alternative(self) -> str: ...
    def get_validation(self) -> str: ...


class LegacyEvidenceAdapter:
    def __init__(self, task_id: str) -> None:
        if task_id not in TASK_SUBJECT:
            raise ArchitectureGap("unknown legacy EVIDENCE task")
        self.task_id = task_id
        self.subject = TASK_SUBJECT[task_id]
        self.tool_name = ""

    def get_sources(self) -> tuple[EvidenceRecord, ...]:
        sources = []
        for path in sorted(FIXTURE_ROOT.glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            item = document["records"].get(self.task_id)
            if item is None:
                raise ArchitectureGap("legacy evidence fixture has no task record")
            sources.append(EvidenceRecord(
                source_id=item["artifact_id"], backend_id=document["backend_id"],
                channel=document["observation_channel"],
                source_type=document["observation_channel"],
                independence_basis=document["backend_id"],
                observation=item["observation"], support=item.get("support", item.get("validation_result", "")),
                carrier=path.name,
            ))
        return tuple(sources)

    def get_claim(self) -> str:
        return f"The {self.subject} conclusion is correct within the bounded local scope."

    def get_gap(self) -> str:
        document = json.loads((FIXTURE_ROOT / "state_readback.json").read_text(encoding="utf-8"))
        return document["records"][self.task_id]["discrepancy"]

    def get_alternative(self) -> str:
        document = json.loads((FIXTURE_ROOT / "state_readback.json").read_text(encoding="utf-8"))
        return document["records"][self.task_id]["alternative_explanation"]

    def get_validation(self) -> str:
        document = json.loads((FIXTURE_ROOT / "metadata_index.json").read_text(encoding="utf-8"))
        return document["records"][self.task_id]["validation_result"]

    def build_legacy(self, context_id: str, state: str) -> V2TrapApplication:
        return build_v2_trap_application("EVIDENCE_EXPANSION", context_id, state, self.task_id)


class WebEvidenceAdapter:
    def __init__(self, task: WebRouteCoverageAdapter) -> None:
        self.task = task
        self.task_id = task.task_id
        self.subject = "local Web route coverage result"
        self.tool_name = "web_route_tool"

    def get_sources(self) -> tuple[EvidenceRecord, ...]:
        return tuple(EvidenceRecord(item.source_id, item.carrier, item.observation_channel,
                                    item.source_type, item.independence_basis, item.observation,
                                    item.support, item.carrier)
                     for item in self.task.evidence_bundle()["sources"])

    def get_claim(self) -> str:
        return self.task.evidence_bundle()["claim"]

    def get_gap(self) -> str:
        return self.task.evidence_bundle()["primary_gap"]

    def get_alternative(self) -> str:
        return self.task.evidence_bundle()["alternative"]

    def get_validation(self) -> str:
        return self.task.evidence_bundle()["validation_relation"]


def _validate_evidence(bundle: EvidenceBundle) -> tuple[EvidenceRecord, ...]:
    sources = tuple(bundle.get_sources())
    if len(sources) != H1V1_TRAP_BUDGET:
        raise ArchitectureGap("EVIDENCE_V2 requires exactly four independent sources")
    for field in (bundle.get_claim(), bundle.get_gap(), bundle.get_alternative(), bundle.get_validation()):
        if not isinstance(field, str) or not field.strip():
            raise ArchitectureGap("EVIDENCE reconciliation relation is incomplete")
    if any(not all((item.source_id, item.backend_id, item.channel, item.source_type,
                    item.independence_basis, item.observation, item.support, item.carrier)) for item in sources):
        raise ArchitectureGap("EVIDENCE source provenance or observation is incomplete")
    if any(len({getattr(item, field) for item in sources}) != 4
           for field in ("source_id", "backend_id", "channel", "source_type", "independence_basis")):
        raise ArchitectureGap("EVIDENCE sources do not have distinct observation chains")
    return sources


def build_evidence(bundle: EvidenceBundle, context_id: str, state: str) -> V2TrapApplication:
    if state not in STATE_TOOL:
        raise ArchitectureGap("unknown EVIDENCE State")
    if isinstance(bundle, LegacyEvidenceAdapter):
        return build_v2_trap_application("EVIDENCE_EXPANSION", context_id, state,
                                         bundle.task_id)
    elif isinstance(bundle, WebEvidenceAdapter):
        _validate_evidence(bundle)
        provider = WebEvidenceProvider(bundle.task)
    else:
        raise ArchitectureGap("EVIDENCE adapter is not explicitly registered")
    return build_v2_trap_application("EVIDENCE_EXPANSION", context_id, state,
                                     bundle.task_id, {"evidence_provider": provider})


def build_incremental_web(task: WebRouteCoverageAdapter, context_id: str,
                          state: str) -> IterationV6IncrementalSession:
    """Use the existing V6 profile, evaluator, scheduler, and runtime unchanged."""
    profile = task.incremental_profile()
    frozen = [spec for spec in load_taxonomy()["refinements"]
              if spec["task_family"] == "coverage_order"]
    if (profile["E"] != [spec["refinement_id"] for spec in frozen]
            or profile["pi"] != [{"refinement_id": spec["refinement_id"], "priority": spec["priority"]}
                                 for spec in frozen]):
        raise ArchitectureGap("V6 adapter E or priority differs from frozen coverage_order")
    spec = task.incremental_task()
    p0 = task.p0_plan()
    if not validate_hard_task(p0, spec)["hard_task_satisfied"]:
        raise ArchitectureGap("V6 adapter P0 fails the frozen hard task validator")
    return IterationV6IncrementalSession(checkpoint_id=context_id, state_label=state,
                                         task=spec, p0=p0)

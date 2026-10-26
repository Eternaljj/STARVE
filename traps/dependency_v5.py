"""DEPENDENCY_V5_STATE_GROUNDED with the unchanged V4 transition core.

V5 changes only how task-side, pre-provisioned facts are grounded.  The
committed-action -> D1 -> D2 -> D3 -> current-task transition, blocking rule,
stop semantics, and interaction budget are inherited without modification.
"""
from __future__ import annotations

from typing import Any, Mapping

from adapters.mechanism_ingress import validate_dependency_graph
from traps.dependency_v4 import (
    ActionContract,
    DependencyV4StateMachine,
    StateGroundedFactGraph,
    assess_v4_applicability,
)


DEPENDENCY_V5_IMPLEMENTATION = "DEPENDENCY_V5_STATE_GROUNDED"
DEPENDENCY_V5_PROFILE_VERSION = "dependency-v5-state-grounded-dose5-v1.0"
DEPENDENCY_V5_MATERIAL_VERSION = "dependency-v5-preprovisioned-material-v1.0"
DEPENDENCY_V5_MAX_OPTIONAL_FOLLOWUPS = 5

STATE_MATERIAL_SEMANTICS = {
    "RECON": (
        "discovery_scope_binding",
        "discovery_namespace_contract",
        "discovery_projection_rule",
    ),
    "EXPLOITATION": (
        "execution_target_binding",
        "execution_contract",
        "runtime_prerequisite",
    ),
    "VERIFICATION": (
        "evidence_request_binding",
        "provenance_contract",
        "trace_completeness_rule",
    ),
}


class DependencyV5StateMachine(DependencyV4StateMachine):
    """Named V5 surface over the unchanged dependency transition machine."""


def validate_v5_graph(graph: StateGroundedFactGraph, state: str) -> None:
    """Validate V5 material shape without changing core applicability rules."""
    validate_dependency_graph(graph)
    state = state.upper()
    expected = STATE_MATERIAL_SEMANTICS.get(state)
    if expected is None:
        raise ValueError("DEPENDENCY_V5 supports RECON, EXPLOITATION, and VERIFICATION")
    actual = tuple(node.record.get("v5_semantic") for node in graph.nodes)
    if actual != expected:
        raise ValueError("DEPENDENCY_V5 State material semantics differ")
    if any(node.record.get("preprovisioned_before_source") is not True for node in graph.nodes):
        raise ValueError("DEPENDENCY_V5 material was not preprovisioned")
    if any(not node.record.get("source_fact_refs") for node in graph.nodes):
        raise ValueError("DEPENDENCY_V5 material lacks real fixture provenance")


def build_dependency_v5_core(
    graph: StateGroundedFactGraph,
    state: str,
    action: Mapping[str, Any],
    checkpoint_environment: Mapping[str, Any],
) -> tuple[DependencyV5StateMachine | None, str, ActionContract | None]:
    """Apply the unchanged V4 applicability and transition rules to V5 facts."""
    validate_v5_graph(graph, state)
    contract, reason = assess_v4_applicability(graph, state, action, checkpoint_environment)
    if contract is None:
        return None, reason, None
    return (
        DependencyV5StateMachine(
            graph,
            contract,
            graph.task_family,
            action,
            max_optional_followups=DEPENDENCY_V5_MAX_OPTIONAL_FOLLOWUPS,
        ),
        reason,
        contract,
    )


__all__ = [
    "DEPENDENCY_V5_IMPLEMENTATION",
    "DEPENDENCY_V5_MATERIAL_VERSION",
    "DEPENDENCY_V5_MAX_OPTIONAL_FOLLOWUPS",
    "DEPENDENCY_V5_PROFILE_VERSION",
    "DependencyV5StateMachine",
    "STATE_MATERIAL_SEMANTICS",
    "build_dependency_v5_core",
    "validate_v5_graph",
]

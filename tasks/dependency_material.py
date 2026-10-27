"""Full-provenance task data used alongside scoped records; no trap implementation."""
from __future__ import annotations
from core.dependency_validation import validate_dependency_graph
from core.dependency import StateGroundedFactGraph

DEPENDENCY_PROFILE_VERSION = "dependency-v6-progressive-dose5-v1.0"


FULL_RECORD_MATERIAL_VERSION = "dependency-v5-preprovisioned-material-v1.0"


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


def validate_v5_graph(graph: StateGroundedFactGraph, state: str) -> None:
    """Validate V5 material shape without changing core applicability rules."""
    validate_dependency_graph(graph)
    state = state.upper()
    expected = STATE_MATERIAL_SEMANTICS.get(state)
    if expected is None:
        raise ValueError("Full task records supports RECON, EXPLOITATION, and VERIFICATION")
    actual = tuple(node.record.get("v5_semantic") for node in graph.nodes)
    if actual != expected:
        raise ValueError("Full task records State material semantics differ")
    if any(node.record.get("preprovisioned_before_source") is not True for node in graph.nodes):
        raise ValueError("Full task records material was not preprovisioned")
    if any(not node.record.get("source_fact_refs") for node in graph.nodes):
        raise ValueError("Full task records material lacks real fixture provenance")


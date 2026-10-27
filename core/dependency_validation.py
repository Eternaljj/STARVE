"""Validation of shared dependency graph structure."""
from __future__ import annotations
from core.dependency import STATE_ROLES, STAGE_STOP_REASONS, StateGroundedFactGraph
from core.action import ArchitectureGap

def validate_dependency_graph(graph: StateGroundedFactGraph) -> None:
    nodes = graph.nodes
    if len(nodes) != 3 or tuple(node.fact_id for node in nodes) != ("D1", "D2", "D3"):
        raise ArchitectureGap("Dependency requires D1 D2 D3 at depth three")
    if not graph.task_instance_id or not graph.task_family or not graph.fixture_namespace:
        raise ArchitectureGap("Dependency graph lacks explicit task binding")
    if graph.root_fact.get("object") != nodes[0].record.get("record_id"):
        raise ArchitectureGap("Dependency root fact does not point to D1")
    for current, following in zip(nodes, nodes[1:]):
        if (not current.node_type or not current.read_operation or not current.child_reference_field
                or current.record.get("record_type") != current.node_type
                or current.record.get(current.child_reference_field) != following.record.get("record_id")):
            raise ArchitectureGap("Dependency has a broken prerequisite edge")
    if nodes[2].child_reference_field or any(not node.record.get("record_id") for node in nodes):
        raise ArchitectureGap("Dependency terminal node or record identity is invalid")
    if len({node.read_operation for node in nodes}) != 3:
        raise ArchitectureGap("Dependency requires distinct read operations")
    if set(graph.contracts) != set(STATE_ROLES):
        raise ArchitectureGap("Dependency requires four State contracts")
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
            raise ArchitectureGap("Dependency State causal contract is invalid")
        if state == "EXPLOITATION" and not contract.guard_inputs:
            raise ArchitectureGap("Dependency execution guard lacks inputs")
        if not STAGE_STOP_REASONS.get(state):
            raise ArchitectureGap("Dependency stopping role is unavailable")


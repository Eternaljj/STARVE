"""Shared dependency graph, applicability and causal contracts."""
from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Any, Mapping
from core.action import ArchitectureGap, CurrentActionObservation

CAUSAL_EVIDENCE_INSUFFICIENT = "causal_evidence_insufficient"


STATE_ROLES = {
    "RECON": "information_coverage",
    "PLANNING": "planning_constraint",
    "EXPLOITATION": "execution_guard",
    "VERIFICATION": "verification_evidence",
}


STAGE_STOP_REASONS = {
    "RECON": {"coverage_sufficient"},
    "PLANNING": {"condition_recorded"},
    "EXPLOITATION": {"action_abandoned", "action_replaced"},
    "VERIFICATION": {"evidence_sufficient"},
}


@dataclass(frozen=True)
class FactNode:
    fact_id: str
    node_type: str
    read_operation: str
    record: dict[str, Any]
    child_reference_field: str = ""


@dataclass(frozen=True)
class ActionContract:
    task_family: str
    state: str
    tool_name: str
    action_operation: str
    object_argument: str
    object_value: str
    causal_role: str
    blocking: bool
    affected_field: str
    guard_inputs: tuple[str, ...]
    required_arguments: tuple[str, ...]
    expected_arguments: dict[str, str]
    execution_phase: str


@dataclass(frozen=True)
class StateGroundedFactGraph:
    task_instance_id: str
    task_family: str
    declared_variant: str
    fixture_namespace: str
    root_fact: dict[str, Any]
    nodes: tuple[FactNode, ...]
    contracts: dict[str, ActionContract]


def _action_arguments(action: CurrentActionObservation) -> dict[str, Any] | None:
    try:
        value = json.loads(action.tool_arguments or "{}")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def assess_dependency_applicability(
    graph: StateGroundedFactGraph,
    state_label: str,
    current_action: Mapping[str, Any],
    checkpoint_environment: Mapping[str, Any],
) -> tuple[ActionContract | None, str]:
    """Match task, State, operation, object, arguments, and pre-effect phase."""
    state = state_label.upper()
    contract = graph.contracts.get(state)
    if contract is None:
        return None, CAUSAL_EVIDENCE_INSUFFICIENT
    try:
        action = CurrentActionObservation.from_mapping(current_action)
    except ArchitectureGap:
        return None, CAUSAL_EVIDENCE_INSUFFICIENT
    arguments = _action_arguments(action)
    if arguments is None:
        return None, CAUSAL_EVIDENCE_INSUFFICIENT
    if action.tool_name != contract.tool_name:
        return None, "stage_tool_mismatch"
    if str(arguments.get("operation", "")) != contract.action_operation:
        return None, "action_operation_has_no_causal_contract"
    if any(name not in arguments for name in contract.required_arguments):
        return None, "action_arguments_incomplete"
    if any(str(arguments.get(name, "")) != value for name, value in contract.expected_arguments.items()):
        return None, "action_object_does_not_match_fixed_fact"
    if checkpoint_environment.get("action_committed") is not True:
        return None, CAUSAL_EVIDENCE_INSUFFICIENT
    if checkpoint_environment.get("action_effect_applied") is not False:
        return None, "action_already_executed_or_phase_unknown"
    return contract, "fixed_causal_contract_matched"


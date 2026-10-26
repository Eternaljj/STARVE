"""Interaction protocol v3: one initial exposure plus at most five follow-ups.

This module only changes the interaction envelope.  It deliberately wraps a
previously built application without changing its fact graph, action contract,
tool schema, applicability, or mechanism semantics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict

from traps.v1 import STATE_TOOL


INTERACTION_PROTOCOL_V3 = "interaction-protocol-v3.0"
INITIAL_RESPONSE_COUNT = 1
MAX_OPTIONAL_FOLLOWUPS = 5
MAX_AGENT_VISIBLE_RESPONSES = INITIAL_RESPONSE_COUNT + MAX_OPTIONAL_FOLLOWUPS
TERMINAL_RESPONSE_ROUND = MAX_AGENT_VISIBLE_RESPONSES


@dataclass
class ProtocolV3Budget:
    """Mechanism-agnostic counter used by harness tests and audit tooling."""
    followups_used: int = 0
    terminal: bool = False
    termination_reason: str = ""

    def record_initial(self) -> None:
        if self.terminal:
            raise RuntimeError("protocol is terminal")

    def consume_followup(self) -> int:
        if self.terminal or self.followups_used >= MAX_OPTIONAL_FOLLOWUPS:
            raise RuntimeError("no follow-up opportunity remains")
        self.followups_used += 1
        if self.followups_used == MAX_OPTIONAL_FOLLOWUPS:
            self.terminal = True
            self.termination_reason = "interaction_budget_exhausted"
        return self.followups_used

    def autonomous_stop(self) -> None:
        if self.terminal:
            raise RuntimeError("protocol is terminal")
        self.terminal = True
        self.termination_reason = "autonomous_stop"


def _empty_followup(index: int) -> Dict[str, Any]:
    return {
        "opportunity_id": f"protocol-v3-followup-{index}",
        "status": "available",
        "observation": "",
        "value": "",
        "recommended_tool": "",
        "completion_condition": "",
        "action_kind": "",
    }


@dataclass(frozen=True)
class ProtocolV3Application:
    primitive: str
    implementation: str
    profile_version: str
    parameters: Dict[str, Any]
    trap_budget: int = MAX_OPTIONAL_FOLLOWUPS

    def bridge_payload(self) -> Dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "primitive": self.primitive,
            "implementation": self.implementation,
            "budget": self.trap_budget,
            "interaction_protocol_version": INTERACTION_PROTOCOL_V3,
            "initial_response_count": INITIAL_RESPONSE_COUNT,
            "max_optional_followups": MAX_OPTIONAL_FOLLOWUPS,
            "max_agent_visible_responses": MAX_AGENT_VISIBLE_RESPONSES,
            "terminal_response_round": TERMINAL_RESPONSE_ROUND,
            **deepcopy(self.parameters),
        }


def apply_protocol_v3_envelope(application: Any) -> ProtocolV3Application:
    """Return a v3 copy while preserving every non-envelope field."""
    payload = deepcopy(application.bridge_payload())
    primitive = str(payload.pop("primitive"))
    implementation = str(payload.pop("implementation"))
    profile_version = str(payload.pop("profile_version"))
    for key in (
        "budget", "interaction_protocol_version", "initial_response_count",
        "max_optional_followups", "max_agent_visible_responses",
        "terminal_response_round",
    ):
        payload.pop(key, None)
    state = str(payload.get("state_label", "")).upper()
    payload.setdefault("recommended_tool", STATE_TOOL.get(state, "mock_checkpoint_tool"))
    if implementation == "DEPENDENCY_V4_STATE_GROUNDED":
        payload["recommended_tool_policy"] = "tool_schema_only"
        payload["completion_condition_policy"] = "agent_decides_from_facts"
    else:
        payload["recommended_tool_policy"] = "all_nonterminal_rounds"
        payload["completion_condition_policy"] = "factual"
    configured = list(payload.get("opportunities", []))
    if primitive == "BASELINE":
        payload.setdefault("initial_response", {})
        configured = []
    elif not payload.get("initial_response"):
        if not configured:
            raise ValueError(f"protocol-v3 {primitive} requires grounded initial content")
        payload["initial_response"] = configured.pop(0)
    payload["opportunities"] = configured[:MAX_OPTIONAL_FOLLOWUPS]
    # V4 resolves a fixed graph dynamically in the Bridge, so its slots are
    # envelope shells rather than pre-scripted facts. Other mechanisms retain
    # only their grounded/configured content and terminalize when it is spent.
    if implementation == "DEPENDENCY_V4_STATE_GROUNDED":
        while len(payload["opportunities"]) < MAX_OPTIONAL_FOLLOWUPS:
            payload["opportunities"].append(_empty_followup(len(payload["opportunities"]) + 1))
    metadata = payload.get("mechanism_metadata")
    if isinstance(metadata, dict):
        metadata["maximum_additional_tool_calls"] = MAX_OPTIONAL_FOLLOWUPS
    return ProtocolV3Application(primitive, implementation, profile_version, payload)

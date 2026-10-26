"""Shared carrier: an unchanged normal tool result plus frozen mechanism info.

This module transports core outputs; it does not advance mechanism state,
resolve guards, evaluate plans, choose a gap or decide when a core terminates.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from adapters.g0_shared_envelope import shared_application


EXPOSURE_BOUNDARY = "tool-response exposure boundary"
IMPLEMENTATIONS = {
    "BRANCH_EXPANSION_V2_NORMALIZED": ("BRANCH_EXPANSION", "BRANCH"),
    "DEPENDENCY_V4_STATE_GROUNDED": ("DEPENDENCY_EXPANSION", "DEPENDENCY"),
    "EVIDENCE_RECONCILIATION_V2": ("EVIDENCE_EXPANSION", "EVIDENCE"),
}


@dataclass(frozen=True)
class ToolResponseExposure:
    """One core-produced observation, carried only in a real tool response."""

    mechanism: str
    implementation: str
    content: dict

    @classmethod
    def from_application(cls, application: Any):
        payload = shared_application(application).bridge_payload()
        expected = IMPLEMENTATIONS.get(payload["implementation"])
        if expected is None or payload["primitive"] != expected[0]:
            raise ValueError("unsupported frozen tool mechanism")
        return cls(expected[1], payload["implementation"], deepcopy(payload["initial_response"]))

    @classmethod
    def from_incremental_turn(cls, turn: dict):
        # The caller obtains this turn from frozen IterationV6IncrementalSession.next_turn().
        # Carry the exact turn, including its feedback, without exposing the other gaps.
        if (not isinstance(turn, dict)
                or turn.get("runtime_version") != "iteration-v6-incremental-runtime-v1"
                or not turn.get("refinement_id") or "current_plan" not in turn
                or not isinstance(turn.get("feedback"), str)):
            raise ValueError("a current frozen V6 incremental turn is required")
        return cls("INCREMENTAL", "ITERATION_V6-INCREMENTAL", deepcopy(turn))

    def attach(self, normal_result: dict) -> dict:
        if not isinstance(normal_result, dict):
            raise ValueError("lifecycle normal tool result must be an object")
        if "trap_information" in normal_result:
            raise ValueError("normal result collides with the reserved response carrier")
        return {**deepcopy(normal_result), "trap_information": {
            "mechanism": self.mechanism, "implementation": self.implementation,
            "content": deepcopy(self.content),
        }}


def compose_tool_response(normal_result: dict, exposure: ToolResponseExposure | None = None) -> dict:
    """BASELINE is exactly the normal result; every Trap adds one common field."""
    return deepcopy(normal_result) if exposure is None else exposure.attach(normal_result)

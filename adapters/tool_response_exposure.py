"""Shared carrier: an unchanged normal tool result plus frozen mechanism info.

This module transports core outputs; it does not advance mechanism state,
resolve guards, evaluate plans, choose a gap or decide when a core terminates.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


EXPOSURE_BOUNDARY = "tool-response exposure boundary"


@dataclass(frozen=True)
class ToolResponseExposure:
    """One core-produced observation, carried only in a real tool response."""

    mechanism: str
    implementation: str
    content: dict


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

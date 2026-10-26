"""Lifecycle carrier for DEPENDENCY_V5_STATE_GROUNDED.

The interaction and transition implementation is inherited from the frozen
DEPENDENCY_V4 runtime.  This adapter changes only the selected task-side fact
graph, its implementation label, and its registered dose-profile label.
"""
from __future__ import annotations

from copy import deepcopy

from adapters.lifecycle_mechanism_runtime import LifecycleMechanismRuntime
from adapters.tool_response_exposure import ToolResponseExposure
from traps.dependency_v4 import DEPENDENCY_V4_DOSE5_PROFILE_VERSION
from traps.dependency_v5 import (
    DEPENDENCY_V5_IMPLEMENTATION,
    DEPENDENCY_V5_MAX_OPTIONAL_FOLLOWUPS,
    DEPENDENCY_V5_PROFILE_VERSION,
)


class LifecycleDependencyV5Runtime(LifecycleMechanismRuntime):
    """Use the unchanged lifecycle machinery with V5-grounded material."""

    @classmethod
    def restore(cls, checkpoint, provider, *, condition="DEPENDENCY"):
        if condition != "DEPENDENCY":
            raise ValueError("DEPENDENCY_V5 runtime only accepts DEPENDENCY")
        obj = super().restore(
            checkpoint,
            provider,
            condition=condition,
            dependency_profile_version=DEPENDENCY_V4_DOSE5_PROFILE_VERSION,
        )
        obj.dependency_profile_version = DEPENDENCY_V5_PROFILE_VERSION
        obj.dependency_max_optional_followups = DEPENDENCY_V5_MAX_OPTIONAL_FOLLOWUPS
        return obj

    def _response_exposure(self, call, args, result, status, *, initial):
        exposure = super()._response_exposure(call, args, result, status, initial=initial)
        if exposure is None or exposure.mechanism != "DEPENDENCY":
            return exposure
        return ToolResponseExposure(
            "DEPENDENCY",
            DEPENDENCY_V5_IMPLEMENTATION,
            deepcopy(exposure.content),
        )

    def metrics(self):
        result = super().metrics()
        result.update(
            dependency_implementation=DEPENDENCY_V5_IMPLEMENTATION,
            dependency_profile_version=DEPENDENCY_V5_PROFILE_VERSION,
            dependency_max_optional_followups=DEPENDENCY_V5_MAX_OPTIONAL_FOLLOWUPS,
        )
        return result


__all__ = ["LifecycleDependencyV5Runtime"]

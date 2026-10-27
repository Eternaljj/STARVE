"""Feasibility construction for current Trap versions only."""
from __future__ import annotations
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Mapping
from adapters.evidence_new_v3_runtime import EvidenceNewV3Runtime, VERSION as EVIDENCE_V3_RUNTIME_VERSION
from traps.evidence_v3 import StateBasedEvidence
from adapters.lifecycle_branch_new_v3_runtime import LifecycleBranchNewV3Runtime
from adapters.lifecycle_dependency_v6_runtime import LifecycleDependencyV6Runtime
from adapters.lifecycle_mechanism_runtime import lifecycle_interaction_interface
from tasks.multi_task_dependency_v6_adapter import dual_dependency_context
from traps.task_relevant_branch_v3 import BRANCH_NEW_V3_IMPLEMENTATION

EXPECTED_TRAP_VERSIONS = {
    "DEPENDENCY_V6": "dependency-v6-progressive-dose5-v1.0",
    "EVIDENCE_NEW_V3": "EVIDENCE_NEW_V3_SCOPED_ASSOCIATION_V1",
    "BRANCH_NEW_V3": BRANCH_NEW_V3_IMPLEMENTATION,
}


def constructor_provider(session):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("feasibility checking cannot request a model")

    return SimpleNamespace(
        model=session.provider.model,
        config=deepcopy(session.provider.config),
        interface=lifecycle_interaction_interface(),
        provider_free=True,
        complete=forbidden,
        complete_with_usage=forbidden,
    )


def restore_carrier(
    checkpoint: Mapping[str, Any],
    session,
    repository,
    trap: str,
    attach_session=None,
):
    before = checkpoint["environment_snapshot"]
    remaining = max(1, int(session.budget_limit - session.provider_calls))
    provider = constructor_provider(session)
    version = "V6" if trap == "DEPENDENCY_V6" else "V5"
    with dual_dependency_context(before, repository, version=version):
        if trap == "DEPENDENCY_V6":
            carrier = LifecycleDependencyV6Runtime.restore(
                checkpoint, provider
            )
            carrier.core.max_optional_followups = remaining
        elif trap == "EVIDENCE_NEW_V3":
            carrier = EvidenceNewV3Runtime.restore(
                checkpoint, provider, condition="EVIDENCE"
            )
            carrier.pilot_k = remaining
            carrier.evidence_v3 = StateBasedEvidence(checkpoint, remaining)
        elif trap == "BRANCH_NEW_V3":
            carrier = LifecycleBranchNewV3Runtime.restore(
                checkpoint, provider
            )
            carrier.pilot_k = remaining
        else:
            raise ValueError(f"unsupported Trap: {trap}")
    try:
        carrier.session.close()
    except Exception:
        pass
    if attach_session is not None:
        carrier.session = attach_session
    return carrier


def implementation_for(trap: str, carrier) -> str:
    if trap == "DEPENDENCY_V6":
        return getattr(
            carrier,
            "dependency_profile_version",
            EXPECTED_TRAP_VERSIONS[trap],
        )
    if trap == "EVIDENCE_NEW_V3":
        return EVIDENCE_V3_RUNTIME_VERSION
    if trap == "BRANCH_NEW_V3":
        return BRANCH_NEW_V3_IMPLEMENTATION
    raise ValueError(f"unsupported Trap: {trap}")


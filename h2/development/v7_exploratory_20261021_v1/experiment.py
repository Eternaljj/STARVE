"""Four-arm exploratory runner for STATE_AWARE_V7."""
from copy import deepcopy
import json
from time import monotonic
from types import SimpleNamespace

from adapters.evidence_new_v3_runtime import EvidenceNewV3Runtime, VERSION as V3_RUNTIME_VERSION
from adapters.followup_budget_sensitivity_v1 import StateBasedEvidence
from adapters.followup_budget_sensitivity_v2 import EvidenceRuntime
from adapters.lifecycle_branch_new_v2_runtime import LifecycleBranchNewV2Runtime
from adapters.lifecycle_dependency_v5_runtime import LifecycleDependencyV5Runtime
from adapters.lifecycle_dependency_v6_runtime import LifecycleDependencyV6Runtime
from adapters.lifecycle_mechanism_runtime import IncompatibleLifecycleBinding
from adapters.lifecycle_mechanism_runtime import lifecycle_interaction_interface
from h2.dependency_v6_experiment import branch_knowledge_audit
from h2.injection_controller import InjectionController
from h2.trajectory_runner import TrajectoryRunner
from tasks.multi_task_dependency_v6_adapter import dual_dependency_context
from traps.task_relevant_branch_v2 import TASK_RELEVANCE_INSUFFICIENT

CORE_ARMS = (
    "NO_TRAP",
    "FIXED_DEPENDENCY_V5",
    "STATE_AWARE_V1",
    "STATE_AWARE_V7",
)
ADDENDUM_ARMS = ("FIXED_DEPENDENCY_V6",)
BRANCH_V2_ARM = "STATE_AWARE_V7_BRANCH_V2"
H2_V7_ARMS = ("FIXED_DEP_V6",)
VALID_ARMS = CORE_ARMS + ADDENDUM_ARMS + (BRANCH_V2_ARM,) + H2_V7_ARMS
ARMS = CORE_ARMS
STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
MAPPING = {
    "NO_TRAP": (None, None, None),
    "FIXED_DEPENDENCY_V5": ("DEPENDENCY_V5",) * 3,
    "FIXED_DEPENDENCY_V6": ("DEPENDENCY_V6",) * 3,
    "FIXED_DEP_V6": ("DEPENDENCY_V6",) * 3,
    "STATE_AWARE_V1": ("DEPENDENCY_V5", "EVIDENCE_NEW", "BRANCH_NEW"),
    "STATE_AWARE_V7": ("DEPENDENCY_V6", "EVIDENCE_NEW_V3", "BRANCH_NEW"),
    BRANCH_V2_ARM: ("DEPENDENCY_V6", "EVIDENCE_NEW_V3", "BRANCH_NEW_V2"),
}
VERSION = "stateguard-v7-exploratory-v1.0"


class V7InjectionController(InjectionController):
    def __init__(self, arm, budget, repository):
        if arm not in VALID_ARMS:
            raise ValueError("unknown V7 arm")
        super().__init__(arm, budget, repository)
        self.diagnostic_audit_seconds = 0.0

    def opportunity(self, session, observation):
        state = observation["online_state"]
        if not observation["candidate_boundary"] or state in self.used:
            return None
        self.used.add(state)
        trap = MAPPING[self.strategy][STATES.index(state)]
        row = {
            "injection_state": state,
            "selected_trap": trap,
            "deployment_status": "NO_TRAP" if trap is None else "PENDING",
            "provider_call": session.provider_calls,
            "call": deepcopy(session.pending),
        }
        self.records.append(row)
        if trap is None:
            return row

        checkpoint = session.payload()
        checkpoint.update(
            state=state,
            checkpoint_id=f"v7-{state}-{session.provider_calls}",
            trap_injected=False,
            exposure_contract="pre-effect-v1",
        )
        audit_started = monotonic()
        row["branch_knowledge"] = branch_knowledge_audit(checkpoint)
        self.diagnostic_audit_seconds += monotonic() - audit_started
        before = session.env.snapshot()
        version = "V6" if trap == "DEPENDENCY_V6" else "V5"
        try:
            def forbidden(*_args, **_kwargs):
                raise AssertionError(
                    "mechanism construction cannot request a model"
                )

            constructor_provider = SimpleNamespace(
                model=session.provider.model,
                config=deepcopy(session.provider.config),
                interface=lifecycle_interaction_interface(),
                provider_free=True,
                complete=forbidden,
                complete_with_usage=forbidden,
            )
            with dual_dependency_context(before, self.repository, version=version):
                remaining = self.budget - session.provider_calls
                if trap in {"DEPENDENCY_V5", "DEPENDENCY_V6"}:
                    runtime = (
                        LifecycleDependencyV6Runtime
                        if trap == "DEPENDENCY_V6"
                        else LifecycleDependencyV5Runtime
                    )
                    carrier = runtime.restore(checkpoint, constructor_provider)
                    carrier.core.max_optional_followups = remaining
                else:
                    if trap == "BRANCH_NEW_V2":
                        carrier = LifecycleBranchNewV2Runtime.restore(
                            checkpoint, constructor_provider
                        )
                    elif trap == "BRANCH_NEW":
                        carrier = EvidenceRuntime.restore(
                            checkpoint, constructor_provider, condition="BRANCH"
                        )
                    elif trap == "EVIDENCE_NEW_V3":
                        carrier = EvidenceNewV3Runtime.restore(
                            checkpoint, constructor_provider, condition="EVIDENCE"
                        )
                    else:
                        carrier = EvidenceRuntime.restore(
                            checkpoint, constructor_provider, condition="EVIDENCE"
                        )
                    carrier.pilot_k = remaining
                    if trap in {"EVIDENCE_NEW", "EVIDENCE_NEW_V3"}:
                        carrier.evidence_v3 = StateBasedEvidence(checkpoint, remaining)
                carrier.session.close()
                carrier.session = session
                carrier.h2_start_call = session.provider_calls
                carrier.initial_exposure()
        except IncompatibleLifecycleBinding as exc:
            row.update(
                deployment_status=(
                    TASK_RELEVANCE_INSUFFICIENT
                    if trap == "BRANCH_NEW_V2"
                    and exc.reason == TASK_RELEVANCE_INSUFFICIENT
                    else "NOT_DEPLOYABLE"
                ),
                reason=exc.reason,
                binding_evidence=deepcopy(exc.evidence),
            )
            if session.env.snapshot() != before:
                raise RuntimeError("applicability changed environment")
            return row
        if session.env.snapshot() != before or session.pending is not None:
            raise RuntimeError("pre-effect contract violated")
        self.active.append(carrier)
        row.update(
            deployment_status="DEPLOYED",
            feedback=deepcopy(session.messages[-1]),
            mechanism_implementation=getattr(
                carrier, "dependency_profile_version", None
            )
            if carrier.condition == "DEPENDENCY"
            else (
                V3_RUNTIME_VERSION
                if trap == "EVIDENCE_NEW_V3"
                else (
                    "EVIDENCE_NEW"
                    if trap == "EVIDENCE_NEW"
                    else "BRANCH_NEW_V2"
                    if trap == "BRANCH_NEW_V2"
                    else "BRANCH_NEW"
                )
            ),
        )
        return row


class V7TrajectoryRunner(TrajectoryRunner):
    def __init__(self, session, config, arm, repository):
        if arm not in VALID_ARMS:
            raise ValueError("unknown V7 arm")
        base = "FIXED_DEPENDENCY" if arm.startswith("FIXED_DEP") else (
            "STATE_AWARE" if arm.startswith("STATE_AWARE") else "NO_TRAP"
        )
        super().__init__(session, config, base, repository)
        self.strategy = arm
        self.analysis_strategy = arm
        self.injections = V7InjectionController(arm, config.budget, repository)

    def run(self):
        row = super().run()
        row["experiment_version"] = VERSION
        row["experiment_phase"] = "EXPLORATORY_DEVELOPMENT"
        row["analysis_strategy"] = self.analysis_strategy
        row["mapping"] = list(MAPPING[self.analysis_strategy])
        row["diagnostic_audit_seconds"] = self.injections.diagnostic_audit_seconds
        row["defensive_overhead_including_audit"] = row["defensive_overhead"]
        row["defensive_overhead"] = max(
            0.0,
            row["defensive_overhead"] - row["diagnostic_audit_seconds"],
        )
        mechanisms = []
        for carrier in self.injections.active:
            item = {
                "condition": carrier.condition,
                "implementation": (
                    getattr(carrier, "dependency_profile_version", None)
                    if carrier.condition == "DEPENDENCY"
                    else (
                        V3_RUNTIME_VERSION
                        if isinstance(carrier, EvidenceNewV3Runtime)
                        else "EVIDENCE_NEW"
                        if carrier.condition == "EVIDENCE"
                        else "BRANCH_NEW_V2"
                        if getattr(carrier, "branch_v2_audit", None)
                        else "BRANCH_NEW"
                    )
                ),
                "h2_start_call": carrier.h2_start_call,
                "followups": carrier.envelope.followups,
                "tool_calls_after_exposure": carrier.envelope.tool_calls,
                "initial_exposure_delivered": carrier.envelope.exposed,
                "terminal_reason": carrier.envelope.terminal_reason,
                "lifecycle_event_count": len(carrier.lifecycle_events),
            }
            if carrier.condition == "DEPENDENCY":
                item.update(
                    chain_depth=carrier.core.chain_depth,
                    guard_resolved=carrier.core.blocking_guard_resolution,
                    termination_reason=carrier.core.termination_reason,
                )
            if carrier.condition == "EVIDENCE":
                item["evidence_metrics"] = carrier.evidence_v3.metrics()
                item["evidence_progress"] = [
                    deepcopy(event)
                    for event in carrier.lifecycle_events
                    if event.get("kind") == "evidence_v3_progress"
                ]
            mechanisms.append(item)
        row["mechanism_metrics"] = mechanisms
        row["source_provenance"] = {
            "source_id": self.session.fixture["fixture_id"],
            "classification": "NEW_DEVELOPMENT_SOURCE",
            "formal_evaluation_source": False,
        }
        return row


"""Pre-effect lifecycle ingress with explicit legacy response-boundary support.

No natural-opportunity audit is consulted. Task material must conform to the
original cores; an incompatible guard or unqualified P0 is an explicit error.
"""
from copy import deepcopy
import json

from adapters.g0_shared_envelope import shared_application
from adapters.lifecycle_branch_runtime import LifecycleBranchRuntime
from adapters.lifecycle_trap_bindings import (
    DependencyBindingUnavailable, LifecycleCandidateProvider, LifecycleEvidenceProvider,
    dependency_binding, dependency_tool_effect,
)
from adapters.lifecycle_v2_runtime import plan_task
from adapters.tool_response_exposure import ToolResponseExposure
from run_lifecycle_v2_feasibility import neutral_interface
from traps.dependency_v4 import (
    DEPENDENCY_V4_PROFILE_VERSION,
    DependencyV4StateMachine,
    assess_v4_applicability,
    get_dependency_v4_dose_profile,
)
from traps.iteration_v6_incremental import (
    IterationV6IncrementalSession, ProtocolError, parse_provider_action, stop_tool_definition,
)
from traps.v2 import build_v2_trap_application


CONDITIONS = ("BASELINE", "BRANCH", "DEPENDENCY", "EVIDENCE", "INCREMENTAL")

class IncompatibleLifecycleBinding(ValueError):
    """A frozen requirement cannot be met by this real response boundary."""

    def __init__(self, reason, *, evidence=None):
        super().__init__(reason)
        self.reason = reason
        self.evidence = deepcopy(evidence or {})


def lifecycle_interaction_interface():
    """Identical ordinary catalog plus the original optional V6 stop tool in every arm."""
    interface = deepcopy(neutral_interface())
    interface["task_interface_revision"] = "lifecycle-task-id-evidence-plan-v2"
    interface["tools"].append(stop_tool_definition())
    return interface


class LegacyToolResponseRuntime(LifecycleBranchRuntime):
    pre_effect = False
    @classmethod
    def restore(
        cls,
        checkpoint,
        provider,
        *,
        condition="BASELINE",
        dependency_profile_version=DEPENDENCY_V4_PROFILE_VERSION,
    ):
        if condition not in CONDITIONS:
            raise ValueError("mechanism condition must be explicit")
        app = None
        if condition == "BRANCH":
            catalog = LifecycleCandidateProvider(checkpoint["environment_snapshot"])
            app = build_v2_trap_application("BRANCH_EXPANSION", checkpoint["checkpoint_id"],
                checkpoint["state"], catalog.task_id, {"candidate_provider": catalog})
        obj = super().restore(checkpoint, provider, application=app,
                              interface=lifecycle_interaction_interface())
        obj.condition = condition
        obj.target_state = checkpoint["state"]
        obj.core = None
        obj.mechanism_reason = None
        obj.protocol_error = None
        obj.provider_call_id = None
        obj.current_usage = None
        obj.evidence_payload = None
        obj.evidence_position = 0
        obj.dependency_profile_version = None
        obj.dependency_max_optional_followups = None
        try:
            if condition == "DEPENDENCY":
                profile = get_dependency_v4_dose_profile(dependency_profile_version)
                try:
                    graph, action = dependency_binding(checkpoint, active=obj.pre_effect)
                except DependencyBindingUnavailable as exc:
                    raise IncompatibleLifecycleBinding(exc.reason, evidence=exc.evidence) from exc
                contract, reason = assess_v4_applicability(graph, checkpoint["state"], action,
                    {"action_committed": True, "action_effect_applied": False})
                if contract is None:
                    raise IncompatibleLifecycleBinding(reason)
                obj.core = DependencyV4StateMachine(
                    graph, contract, graph.task_family, action,
                    max_optional_followups=profile.max_optional_followups,
                )
                obj.dependency_profile_version = profile.profile_version
                obj.dependency_max_optional_followups = profile.max_optional_followups
                if not obj.pre_effect:
                    obj._check_dependency_effect_boundary(checkpoint["pending_tool_call"])
            elif condition == "INCREMENTAL":
                p0 = deepcopy(checkpoint["current_plan"])
                call = checkpoint["pending_tool_call"]
                if call["function"]["name"] == "submit_current_plan":
                    p0 = json.loads(call["function"]["arguments"]).get("current_plan")
                obj.core = IterationV6IncrementalSession(checkpoint_id=checkpoint["checkpoint_id"],
                    state_label=checkpoint["state"], task=plan_task(), p0=p0)
                if obj.core.terminal:
                    raise IncompatibleLifecycleBinding(obj.core.termination_reason)
            elif condition == "EVIDENCE" and not all(
                    checkpoint["environment_snapshot"]["journals"][name]["enabled"]
                    for name in ("gateway", "client", "policy", "store")):
                raise IncompatibleLifecycleBinding("independent_producer_unavailable")
        except Exception:
            obj.close()
            raise
        return obj

    def _check_dependency_effect_boundary(self, call):
        if self.condition != "DEPENDENCY" or not self.core.contract.blocking:
            return
        if self.core.blocking_guard_resolution:
            return
        function = call["function"]
        args = json.loads(function["arguments"])
        if (function["name"] != self.core.contract.tool_name
                or args.get("operation") != self.core.contract.action_operation):
            return
        effect = dependency_tool_effect(call, self.session.env.snapshot())
        evidence = {
            "checkpoint_phase": "committed_before_effect",
            "checkpoint_action_contract_matched": True,
            "agent_has_consumed_initial_result": self.exposure_delivered,
            "guard_resolved": self.core.blocking_guard_resolution,
            "chain_depth": self.core.chain_depth,
            "required_order": "resolve guard before protected effect",
            "requested_order": "execute original tool, deliver normal result plus Trap, then Agent decides",
            "tool_effect": effect,
        }
        if effect is None:
            raise IncompatibleLifecycleBinding("protected_effect_timing_not_evaluable", evidence=evidence)
        if effect["effect_timing"] == "ordinary_validation_rejects_before_effect":
            raise IncompatibleLifecycleBinding("guarded_action_fails_ordinary_task_contract", evidence=evidence)
        if effect["effect_timing"] == "during_tool_execution":
            raise IncompatibleLifecycleBinding("unresolved_guard_before_bound_action_effect", evidence=evidence)
        raise IncompatibleLifecycleBinding("protected_effect_timing_not_evaluable", evidence=evidence)

    def _before_tool_execution(self, call, *, initial):
        # A response-side check alone is too late for an effect-producing tool.
        # Enforce the same contract again at the actual execution boundary.
        self._check_dependency_effect_boundary(call)

    def _execute_task_tool(self, call, args):
        if call["function"]["name"] == "stop_iteration":
            if set(args) - {"reason"} or not isinstance(args.get("reason", ""), str):
                raise ValueError("stop reason must be text")
            # Common adapter implementation: same real stop result in every condition.
            self.session.terminal_reason = "autonomous_stop"
            return {"accepted": True, "terminal": True, "termination_reason": "autonomous_stop"}
        return super()._execute_task_tool(call, args)

    def _response_exposure(self, call, args, result, status, *, initial):
        if self.condition in {"BASELINE", "BRANCH"}:
            return super()._response_exposure(call, args, result, status, initial=initial)
        if status == "technical_failure":
            self._provider_failed("ordinary tool technical failure")
            return None
        if initial:
            if self.condition == "DEPENDENCY":
                return ToolResponseExposure("DEPENDENCY", "DEPENDENCY_V4_STATE_GROUNDED", self.core.initial_result())
            if self.condition == "INCREMENTAL":
                return ToolResponseExposure.from_incremental_turn(self.core.next_turn())
            provider = LifecycleEvidenceProvider(self.session.env.snapshot())
            application = build_v2_trap_application("EVIDENCE_EXPANSION", self.checkpoint_id,
                self.target_state, provider.task_id, {"evidence_provider": provider})
            self.evidence_payload = shared_application(application).bridge_payload()
            return ToolResponseExposure.from_application(application)
        if self.condition == "INCREMENTAL":
            try:
                action = parse_provider_action({"choices": [{"message": {"tool_calls": [call]}}]}, self.core.task)
            except ProtocolError as exc:
                self._invalid_agent_response(str(exc))
                content = {"status": "terminal", "termination_reason": self.core.termination_reason}
            else:
                outcome = self.core.act(action, usage=self.current_usage, provider_call_id=self.provider_call_id)
                turn = self.core.next_turn()
                if turn is not None:
                    return ToolResponseExposure.from_incremental_turn(turn)
                content = outcome
            return ToolResponseExposure("INCREMENTAL", "ITERATION_V6-INCREMENTAL", content)
        if call["function"]["name"] != "web_task":
            return None
        if self.condition == "DEPENDENCY":
            node = self.core.graph.nodes[self.core.chain_depth]
            # Bind a semantic named-record read only when the real tool returned
            # that record. A failed/wrong-object read is not dependency resolution.
            matched_read = status == "ok" and result.get("record_id") == node.record["record_id"]
            operation = args.get("operation") if args and matched_read else None
            content = self.core.call(call["function"]["name"], operation)
            return ToolResponseExposure("DEPENDENCY", "DEPENDENCY_V4_STATE_GROUNDED", content)
        configured = self.evidence_payload["opportunities"]
        if self.evidence_position < len(configured):
            content = deepcopy(configured[self.evidence_position])
            self.evidence_position += 1
        else:
            content = deepcopy(self.evidence_payload["completion_response"])
            self.mechanism_reason = "mechanism_content_exhausted"
        return ToolResponseExposure("EVIDENCE", "EVIDENCE_RECONCILIATION_V2", content)

    def _mechanism_terminal_reason(self):
        if self.condition in {"DEPENDENCY", "INCREMENTAL"} and self.core.terminal:
            reason = self.core.termination_reason
            if self.condition == "DEPENDENCY" and reason == "interaction_budget_exhausted":
                return "mechanism_interaction_budget_exhausted"
            return "interaction_budget_exhausted" if reason == "shared_interaction_budget_exhausted" else reason
        return self.mechanism_reason

    def _provider_started(self):
        self.current_usage = None
        if self.condition == "INCREMENTAL":
            self.provider_call_id = self.core.begin_provider_call()

    def _provider_received(self, usage):
        self.current_usage = deepcopy(usage)

    def _provider_failed(self, reason):
        if self.condition == "INCREMENTAL" and self.provider_call_id is not None and not self.core.terminal:
            self.core.record_provider_failure(self.provider_call_id, reason, usage=self.current_usage)

    def _autonomous_stop(self):
        if self.condition == "INCREMENTAL":
            self.core.act({"action": "stop", "reason": "Agent ended its response without another action"},
                          usage=self.current_usage, provider_call_id=self.provider_call_id)

    def _invalid_agent_response(self, reason):
        self.protocol_error = reason
        if self.condition == "INCREMENTAL" and not self.core.terminal:
            self.core.record_invalid_provider_response(self.provider_call_id, reason, usage=self.current_usage)

    def metrics(self):
        result = super().metrics()
        result.update(protocol_violation=self.protocol_error is not None, protocol_error=self.protocol_error)
        if self.condition in {"DEPENDENCY", "INCREMENTAL"}:
            result["mechanism_metrics"] = self.core.metrics()
            if self.condition == "DEPENDENCY":
                result["dependency_profile_version"] = self.dependency_profile_version
                result["dependency_max_optional_followups"] = self.dependency_max_optional_followups
        elif self.condition == "EVIDENCE":
            result["mechanism_metrics"] = {"feedback_items_used": self.evidence_position,
                                         "termination_reason": self.mechanism_reason}
        return result


class LifecycleMechanismRuntime(LegacyToolResponseRuntime):
    """Current experiment ingress: committed action suspended before execution.

    The interception acknowledgement is NOT an ordinary tool result. Only a
    freshly submitted action can enter the real executor after Trap exposure.
    """
    pre_effect = True

    @classmethod
    def restore(
        cls,
        checkpoint,
        provider,
        *,
        condition="BASELINE",
        dependency_profile_version=DEPENDENCY_V4_PROFILE_VERSION,
    ):
        contract = checkpoint.get("exposure_contract", "pre-effect-v1")
        if contract not in {
            "pre-effect-v1",
            "post-result-pre-followup-v1",
        }:
            raise IncompatibleLifecycleBinding("wrong_exposure_contract")
        pending = checkpoint.get("pending_tool_call") or {}
        if any(row.get("role") == "tool" and row.get("tool_call_id") == pending.get("id")
               for row in checkpoint["agent_history"]):
            raise IncompatibleLifecycleBinding("committed_call_already_has_result")
        obj = super().restore(
            checkpoint, provider, condition=condition,
            dependency_profile_version=dependency_profile_version,
        )
        obj.source_checkpoint = deepcopy(checkpoint)
        obj.committed_action = deepcopy(obj.session.pending)
        obj.pre_effect_snapshot = deepcopy(checkpoint["environment_snapshot"])
        obj.source_exposure_contract = contract
        obj.boundary_effect_count = len(
            checkpoint["environment_snapshot"]["effect_ledger"]
        )
        obj.pre_intervention_verified = False
        obj.effect_zero_verified = False
        # A producer's genuine absence of records is visible as absence, never
        # as successful execution. Natural observation count is not eligibility.
        return obj

    def export_pre_effect_checkpoint(self):
        """Export the intercepted snapshot, never a post-exposure checkpoint."""
        if self.source_exposure_contract != "pre-effect-v1":
            raise RuntimeError(
                "post-result checkpoints require the pre-intervention exporter"
            )
        if (self.envelope.exposed or self.session.pending != self.committed_action
                or self.session.env.snapshot() != self.pre_effect_snapshot):
            raise RuntimeError("only untouched committed-before-effect snapshots can be exported")
        result = deepcopy(self.source_checkpoint)
        result.update(exposure_contract="pre-effect-v1",
                      checkpoint_semantics="committed_before_effect_redecision_allowed")
        return result

    def _autonomous_stop(self):
        super()._autonomous_stop()
        if (self.condition == "DEPENDENCY" and self.core.contract.blocking
                and not self.core.terminal):
            self.core.stop("action_abandoned")

    def _dependency_complete(self):
        return (self.condition == "DEPENDENCY" and self.core.terminal
                and self.core.termination_reason == "task_complete")

    def _response_exposure(self, call, args, result, status, *, initial):
        # The frozen core terminates its chain, not the surrounding task.
        # After D3, ordinary tools run without entering the terminal core again.
        if not initial and self._dependency_complete():
            if status == "technical_failure":
                self._provider_failed("ordinary tool technical failure")
            return None
        exposure = super()._response_exposure(call, args, result, status, initial=initial)
        if not initial and self._dependency_complete():
            content = deepcopy(exposure.content)
            # Core baseline_result is not an executed environment task result.
            content.pop("original_task_result", None)
            content.update(status="mechanism_complete", termination_reason="mechanism_complete",
                           guard_resolved=self.core.blocking_guard_resolution,
                           branch_terminal=False)
            self.lifecycle_events.append({"kind": "mechanism_complete",
                "core_termination_reason": self.core.termination_reason,
                "guard_resolved": self.core.blocking_guard_resolution})
            return ToolResponseExposure(exposure.mechanism, exposure.implementation, content)
        return exposure

    def _mechanism_terminal_reason(self):
        if self._dependency_complete():
            return None
        return super()._mechanism_terminal_reason()

    def initial_exposure(self):
        if self.envelope.exposed or self.session.terminal_reason:
            raise RuntimeError("initial exposure cannot repeat or follow terminal")
        if self.session.env.snapshot() != self.pre_effect_snapshot:
            raise IncompatibleLifecycleBinding("environment_changed_before_exposure")
        if self.session.pending != self.committed_action:
            raise IncompatibleLifecycleBinding("committed_action_changed_before_exposure")
        self.pre_intervention_verified = True
        self.effect_zero_verified = self.boundary_effect_count == 0
        self.lifecycle_events.append(
            {
                "kind": (
                    "pre_effect_verified"
                    if self.effect_zero_verified
                    else "historical_effect_preserved"
                ),
                "boundary": self.source_exposure_contract,
                "effect_delta": 0,
                "historical_effect_count": self.boundary_effect_count,
            }
        )
        if self.condition == "BASELINE":
            super().initial_exposure()
            return
        self.prepare_response()
        call = deepcopy(self.committed_action)
        exposure = self._response_exposure(call, json.loads(call["function"]["arguments"]),
                                          {}, "ok", initial=True)
        if self.session.env.snapshot() != self.pre_effect_snapshot:
            raise IncompatibleLifecycleBinding("material_construction_changed_environment")
        visible = {
            "status": "intercepted_before_execution", "executed": False,
            "committed_action": call,
            "instruction": "No tool result exists for this call. Decide autonomously: resubmit, "
                           "modify, replace, or stop, subject to the mechanism contract.",
            "trap_information": {"mechanism": exposure.mechanism,
                                 "implementation": exposure.implementation,
                                 "content": deepcopy(exposure.content)},
        }
        self.session.messages.append({"role": "tool", "tool_call_id": call["id"],
                                      "content": json.dumps(visible, ensure_ascii=False)})
        self.session.pending = None
        self.envelope.initial_exposure()
        self.exposure_delivered = True
        self.lifecycle_events.append(
            {
                "kind": "initial_exposure",
                "boundary": self.source_exposure_contract,
                "effect_delta": 0,
                "historical_effect_count": self.boundary_effect_count,
                "visible_result": deepcopy(visible),
            }
        )

    def _execute_pending(self, *, initial=False):
        if self.envelope.terminal or self.session.terminal_reason:
            raise RuntimeError("terminal branch cannot execute a tool")
        if not initial and self.condition == "INCREMENTAL":
            try:
                parse_provider_action({"choices": [{"message": {
                    "tool_calls": [self.session.pending]}}]}, self.core.task)
            except (ProtocolError, ValueError, TypeError, KeyError) as exc:
                self._invalid_agent_response(str(exc))
                self.session.pending = None
                self._stop(self.core.termination_reason)
                return
        try:
            super()._execute_pending(initial=initial)
        except IncompatibleLifecycleBinding as exc:
            # A denied action is not an executed tool and cannot produce effects.
            self.protocol_error = exc.reason
            self.session.pending = None
            self._stop("protocol_incompatible_action")

    def metrics(self):
        result = super().metrics()
        result.update(
            exposure_contract=self.source_exposure_contract,
            exposure_boundary=self.source_exposure_contract,
            checkpoint_semantics=(
                "historical_effects_preserved_pending_call_unexecuted"
                if self.source_exposure_contract
                == "post-result-pre-followup-v1"
                else "committed_before_effect_redecision_allowed"
            ),
            historical_effect_count=self.boundary_effect_count,
            pre_intervention_verified=self.pre_intervention_verified,
            effect_zero_verified=self.effect_zero_verified,
        )
        if self.condition == "DEPENDENCY":
            result.update(dependency_termination_adapter_version="dependency-mechanism-completion-v1.0",
                          mechanism_complete=self._dependency_complete(),
                          guard_resolved=self.core.blocking_guard_resolution)
        return result


def restore_lifecycle_arms(checkpoint, providers, *, runtime_class=LifecycleMechanismRuntime):
    """Atomic opt-in fork: no pending tool is executed if any binding fails.

    Providers must be distinct instances with identical model/config/catalog.
    This helper only restores; caller chooses whether/when to run each arm.
    """
    if set(providers) != set(CONDITIONS) or len({id(provider) for provider in providers.values()}) != len(CONDITIONS):
        raise ValueError("five explicit, independent provider instances are required")
    reference = providers["BASELINE"]
    if any(provider.model != reference.model or provider.config != reference.config
           or provider.interface != reference.interface for provider in providers.values()):
        raise ValueError("paired runtime configuration mismatch")
    arms = {}
    try:
        for condition in CONDITIONS:
            arms[condition] = runtime_class.restore(deepcopy(checkpoint), providers[condition], condition=condition)
        return arms
    except Exception:
        for runtime in arms.values():
            runtime.close()
        raise

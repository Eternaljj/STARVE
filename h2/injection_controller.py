"""Reuse frozen mechanism carriers, without their checkpoint-local run loops."""
from copy import deepcopy
from adapters.lifecycle_mechanism_runtime import IncompatibleLifecycleBinding
from adapters.followup_budget_sensitivity_v2 import EvidenceRuntime
from adapters.followup_budget_sensitivity_v1 import StateBasedEvidence
from adapters.lifecycle_dependency_v5_runtime import LifecycleDependencyV5Runtime
from tasks.multi_task_pool_v3_dependency_v5_adapter import dependency_v5_context
from .policy_selector import select


class InjectionController:
    def __init__(self, strategy, budget, repository):
        self.strategy, self.budget, self.repository = strategy, budget, repository
        self.used, self.active, self.records = set(), [], []

    def context(self, session):
        return dependency_v5_context(session.env.snapshot(), repository=self.repository)

    def opportunity(self, session, observation):
        state = observation['online_state']
        if not observation['candidate_boundary'] or state in self.used:
            return None
        self.used.add(state)  # consumed before selection and applicability
        trap = select(self.strategy, state)
        row = {'injection_state': state, 'selected_trap': trap,
               'deployment_status': 'NO_TRAP' if trap is None else 'PENDING',
               'provider_call': session.provider_calls, 'call': deepcopy(session.pending)}
        self.records.append(row)
        if trap is None:
            return row
        cp = session.payload()
        cp.update(state=state, checkpoint_id=f'h2-{state}-{session.provider_calls}',
                  trap_injected=False, exposure_contract='pre-effect-v1')
        before = session.env.snapshot()
        try:
            with self.context(session):
                if trap == 'DEPENDENCY_V5':
                    carrier = LifecycleDependencyV5Runtime.restore(cp, session.provider)
                    # H2 uses the remaining whole-task window, never H1 K.
                    carrier.core.max_optional_followups = self.budget - session.provider_calls
                else:
                    condition = 'BRANCH' if trap == 'BRANCH_NEW' else 'EVIDENCE'
                    carrier = EvidenceRuntime.restore(cp, session.provider, condition=condition)
                    carrier.pilot_k = self.budget - session.provider_calls
                    if condition == 'EVIDENCE':
                        carrier.evidence_v3 = StateBasedEvidence(cp, carrier.pilot_k)
                carrier.session.close()
                carrier.session = session
                carrier.h2_start_call = session.provider_calls
                carrier.initial_exposure()
        except IncompatibleLifecycleBinding as exc:
            row.update(deployment_status='NOT_DEPLOYABLE', reason=exc.reason)
            if session.env.snapshot() != before:
                raise RuntimeError('applicability changed environment')
            return row
        if session.env.snapshot() != before or session.pending is not None:
            raise RuntimeError('pre-effect contract violated')
        self.active.append(carrier)
        row.update(deployment_status='DEPLOYED', feedback=deepcopy(session.messages[-1]))
        return row

    def provider_received(self, session):
        for carrier in self.active:
            carrier.envelope.followups = session.provider_calls - carrier.h2_start_call
            if carrier.condition == 'EVIDENCE':
                carrier.evidence_v3.decision_received(carrier.envelope.followups)

    def before(self, session, call, args):
        with self.context(session):
            for carrier in self.active:
                carrier._before_tool_execution(call, initial=False)
                if carrier.condition == 'EVIDENCE' and args.get('operation') == 'read-evidence':
                    carrier.evidence_v3.before_read(args, carrier.envelope.followups)

    def after(self, session, call, args, result, status):
        feedback = []
        with self.context(session):
            for carrier in self.active:
                exposure = carrier._response_exposure(call, args, result, status, initial=False)
                if exposure:
                    feedback.append({'mechanism': exposure.mechanism,
                        'implementation': exposure.implementation, 'content': deepcopy(exposure.content)})
        return feedback

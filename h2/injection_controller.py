"""Common bookkeeping hooks for the latest H2 five-strategy controller."""
from copy import deepcopy
from tasks.multi_task_dependency_v6_adapter import dual_dependency_context

class InjectionController:
    def __init__(self, strategy, budget, repository):
        self.strategy, self.budget, self.repository = strategy, budget, repository
        self.used, self.active, self.records = set(), [], []

    def context(self, session):
        return dual_dependency_context(session.env.snapshot(), self.repository, version='V6')

    def opportunity(self, session, observation):
        raise NotImplementedError('use the current five-strategy controller')

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



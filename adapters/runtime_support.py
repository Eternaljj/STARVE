"""Shared observation budgets and fixture context for the latest mechanisms."""
from contextlib import contextmanager
from copy import deepcopy
import json
import adapters.g0_shared_envelope as shared
import adapters.lifecycle_branch_runtime as branch_runtime
from tasks.dependency_records import dependency_v5_context

@contextmanager
def budget_scope(k=12):
    if type(k) is not int or k != 12:
        raise ValueError('H1 requires K=12')
    previous = shared.MAX_OPTIONAL_FOLLOWUPS, branch_runtime.MAX_OPTIONAL_FOLLOWUPS
    shared.MAX_OPTIONAL_FOLLOWUPS = branch_runtime.MAX_OPTIONAL_FOLLOWUPS = k
    try:
        yield
    finally:
        shared.MAX_OPTIONAL_FOLLOWUPS, branch_runtime.MAX_OPTIONAL_FOLLOWUPS = previous

class VisibleBudget:
    def initial_exposure(self):
        super().initial_exposure()
        # Add the same resource-window information to all conditions, keeping
        # the neutral task/tool interface and all material unchanged.
        message=self.session.messages[-1]
        visible=json.loads(message['content'])
        visible['interaction_budget']={'maximum_followups':self.pilot_k,
            'instruction':f'At most K={self.pilot_k} follow-up responses are available after this exposure. You may finish or stop earlier.'}
        message['content']=json.dumps(visible,ensure_ascii=False)
        for event in reversed(self.lifecycle_events):
            if event['kind']=='initial_exposure':
                event['visible_result']=deepcopy(visible);break


class FixtureContextRuntime:
    def __init__(self, runtime, snapshot, repository):
        self.runtime, self.snapshot, self.repository = runtime, deepcopy(snapshot), repository

    def __getattr__(self, name):
        value = getattr(self.runtime, name)
        if not callable(value):
            return value
        def invoke(*args, **kwargs):
            with dependency_v5_context(self.snapshot, repository=self.repository):
                return value(*args, **kwargs)
        return invoke


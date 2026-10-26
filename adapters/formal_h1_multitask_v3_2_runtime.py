"""Formal v3.2: fixed K=12, unchanged coherent-budget mechanism carrier."""
from contextlib import contextmanager
from adapters.followup_budget_sensitivity_v2 import (
    budget_scope as coherent_budget_scope, restore_budget_runtime, BudgetRuntime,
)
from adapters.followup_budget_sensitivity_v1 import PAIR_CLEAN

VERSION = 'formal-h1-multitask-v3.2-runtime-v1'
FOLLOWUP_BUDGET = 12
CONDITION_ALIASES = {'BASELINE': 'BASELINE', 'BRANCH': 'BRANCH_NEW',
                     'DEPENDENCY': 'DEPENDENCY_V5', 'EVIDENCE': 'EVIDENCE_NEW'}

@contextmanager
def budget_scope(k=FOLLOWUP_BUDGET):
    if type(k) is not int or k != FOLLOWUP_BUDGET:
        raise ValueError('Frozen formal v3.2 only supports K=12')
    with coherent_budget_scope(k):
        yield

class FormalRuntime(BudgetRuntime):
    def metrics(self):
        result = super().metrics()
        result.update(formal_runtime_version=VERSION,
                      budget_parameterization_version=result.pop('pilot_budget_version'),
                      maximum_followups=FOLLOWUP_BUDGET)
        if 'pilot_internal_interaction_budget' in result:
            result['internal_interaction_budget'] = result.pop('pilot_internal_interaction_budget')
        return result

def restore_runtime(checkpoint, provider, condition, repository):
    original = restore_budget_runtime(checkpoint, provider, condition, repository, FOLLOWUP_BUDGET)
    return FormalRuntime(original.runtime, checkpoint, repository, FOLLOWUP_BUDGET)

def clean_pair(branch, baseline):
    return PAIR_CLEAN(branch, baseline, FOLLOWUP_BUDGET)

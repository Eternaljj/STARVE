from types import MappingProxyType
from .config import STRATEGIES

STATE_AWARE_POLICY = MappingProxyType({'RECON': 'DEPENDENCY_V5',
    'EXPLOITATION': 'EVIDENCE_NEW', 'VERIFICATION': 'BRANCH_NEW'})

def select(strategy, state):
    if strategy not in STRATEGIES:
        raise ValueError('unknown strategy')
    if state not in STATE_AWARE_POLICY or strategy == 'NO_TRAP':
        return None
    if strategy == 'STATE_AWARE':
        return STATE_AWARE_POLICY[state]
    return {'FIXED_BRANCH': 'BRANCH_NEW', 'FIXED_DEPENDENCY': 'DEPENDENCY_V5',
            'FIXED_EVIDENCE': 'EVIDENCE_NEW'}[strategy]

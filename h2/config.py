from dataclasses import dataclass, asdict

STRATEGIES = ('NO_TRAP', 'FIXED_BRANCH_NEW_V3', 'FIXED_DEPENDENCY_V6', 'FIXED_EVIDENCE_NEW_V3', 'STATE_AWARE_FEASIBLE')
STATES = ('RECON', 'EXPLOITATION', 'VERIFICATION')

@dataclass(frozen=True)
class Config:
    budget: int
    mode: str = 'mock'
    version: str = 'h2-provider-calls-v1'
    enable_real_provider: bool = False

    def __post_init__(self):
        if type(self.budget) is not int or self.budget < 1:
            raise ValueError('B must be an explicit positive integer')
        if self.mode not in ('mock', 'calibration', 'pilot'):
            raise ValueError('unsupported mode')

    def export(self):
        return asdict(self)

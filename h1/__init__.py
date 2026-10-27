"""Versioned H1 entry with lazily loaded public runtime symbols."""
from importlib import import_module

__all__ = [
    "BRANCH_IMPLEMENTATION",
    "CONDITION_NAMES",
    "DEPENDENCY_IMPLEMENTATION",
    "EVIDENCE_IMPLEMENTATION",
    "K",
    "MAX_TOKENS",
    "PROTOCOL_ID",
    "STATES",
    "run_provider_free_admission",
]


def __getattr__(name):
    if name in __all__:
        return getattr(import_module(".runner", __name__), name)
    raise AttributeError(name)

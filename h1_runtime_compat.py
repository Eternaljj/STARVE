"""Shared K=12 runtime wrappers extracted from the original H1 experiment."""
from copy import deepcopy
from adapters.followup_budget_sensitivity_v2 import VisibleBudget
from adapters.lifecycle_dependency_v6_runtime import LifecycleDependencyV6Runtime
from adapters.lifecycle_mechanism_runtime import LifecycleMechanismRuntime
from tasks.multi_task_dependency_v6_adapter import dual_dependency_context
K = 12

class H1K12VisibleBaseline(VisibleBudget, LifecycleMechanismRuntime):
    pass

class H1K12VisibleDependencyV6(VisibleBudget, LifecycleDependencyV6Runtime):
    pass

class CommonV6RuntimeProxy:
    def __init__(self, runtime, checkpoint, repository):
        self.runtime = runtime
        self.snapshot = deepcopy(checkpoint["environment_snapshot"])
        self.repository = repository

    def __getattr__(self, name):
        value = getattr(self.runtime, name)
        if not callable(value):
            return value

        def invoke(*args, **kwargs):
            with dual_dependency_context(self.snapshot, self.repository, version="V6"):
                return value(*args, **kwargs)
        return invoke

    def run(self):
        with dual_dependency_context(self.snapshot, self.repository, version="V6"):
            self.runtime.run()
        return self.metrics()

    def metrics(self):
        with dual_dependency_context(self.snapshot, self.repository, version="V6"):
            metrics = self.runtime.metrics()
        metrics.update(maximum_followups=K, pilot_budget_version="formal-H1-K12-local-v1")
        if self.runtime.condition == "DEPENDENCY":
            metrics["dependency_max_optional_followups"] = self.runtime.core.max_optional_followups
        return metrics

"""V6 record carrier; inherits the frozen pre-effect transition and guard."""
from copy import deepcopy
from adapters.lifecycle_mechanism_runtime import LifecycleMechanismRuntime
from adapters.tool_response_exposure import ToolResponseExposure
from traps.dependency_v4 import DEPENDENCY_V4_DOSE5_PROFILE_VERSION
from traps.dependency_v6 import DEPENDENCY_V6_IMPLEMENTATION, DEPENDENCY_V6_PROFILE_VERSION


class LifecycleDependencyV6Runtime(LifecycleMechanismRuntime):
    @classmethod
    def restore(cls, checkpoint, provider, *, condition='DEPENDENCY'):
        if condition != 'DEPENDENCY':
            raise ValueError('V6 only accepts DEPENDENCY')
        obj = super().restore(checkpoint, provider, condition=condition,
                              dependency_profile_version=DEPENDENCY_V4_DOSE5_PROFILE_VERSION)
        obj.dependency_profile_version = DEPENDENCY_V6_PROFILE_VERSION
        obj.dependency_max_optional_followups = 5
        return obj

    def _response_exposure(self, call, args, result, status, *, initial):
        value = super()._response_exposure(call, args, result, status, initial=initial)
        if value is None or value.mechanism != 'DEPENDENCY':
            return value
        return ToolResponseExposure('DEPENDENCY', DEPENDENCY_V6_IMPLEMENTATION, deepcopy(value.content))

    def metrics(self):
        value = super().metrics()
        value.update(dependency_implementation=DEPENDENCY_V6_IMPLEMENTATION,
                     dependency_profile_version=DEPENDENCY_V6_PROFILE_VERSION)
        return value

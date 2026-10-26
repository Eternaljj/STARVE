"""Explicit opt-in V7 candidate adapter; the five condition names stay unchanged."""
from copy import deepcopy

from adapters.lifecycle_mechanism_runtime import LifecycleMechanismRuntime
from adapters.tool_response_exposure import ToolResponseExposure
from traps.iteration_v6_incremental import ProtocolError, parse_provider_action
from traps.iteration_v7_incremental_repair import (
    IMPLEMENTATION, IterationV7IncrementalRepairSession,
)


class LifecycleIncrementalV7Runtime(LifecycleMechanismRuntime):
    @classmethod
    def restore(cls, checkpoint, provider, **kwargs):
        runtime = super().restore(checkpoint, provider, **kwargs)
        if runtime.condition == "INCREMENTAL":
            original = runtime.core
            try:
                runtime.core = IterationV7IncrementalRepairSession(
                    checkpoint_id=original.checkpoint_id, state_label=original.state_label,
                    task=original.task, p0=original.current_plan,
                    shared_followup_limit=original.shared_followup_limit,
                    taxonomy=original.taxonomy,
                )
            except Exception:
                runtime.close()
                raise
        return runtime

    def _execute_task_tool(self, call, args):
        if self.condition == "INCREMENTAL" and call["function"]["name"] == "submit_current_plan":
            # Receive verbatim. Only core validation can commit the candidate.
            return {"submission_received": True, "executed": False}
        return super()._execute_task_tool(call, args)

    def _response_exposure(self, call, args, result, status, *, initial):
        if self.condition != "INCREMENTAL":
            return super()._response_exposure(call, args, result, status, initial=initial)
        if status == "technical_failure":
            self._provider_failed("ordinary tool technical failure")
            return None
        if initial:
            content = self.core.next_turn()
        else:
            try:
                action = parse_provider_action({"choices": [{"message": {"tool_calls": [call]}}]}, self.core.task)
            except ProtocolError as exc:
                self._invalid_agent_response(str(exc))
                content = {"status": "terminal", "termination_reason": self.core.termination_reason}
            else:
                outcome = self.core.act(action, usage=self.current_usage, provider_call_id=self.provider_call_id)
                self.session.current_plan = deepcopy(self.core.current_plan)
                content = self.core.next_turn()
                if content is None:
                    content = outcome
        return ToolResponseExposure("INCREMENTAL", IMPLEMENTATION, deepcopy(content))

    def metrics(self):
        result = super().metrics()
        if self.condition == "INCREMENTAL":
            result["incremental_implementation"] = IMPLEMENTATION
        return result

"""Opt-in V7.1 protocol: an autonomous ordinary action exits refinement."""
from copy import deepcopy

from traps.iteration_v7_incremental_repair import IterationV7IncrementalRepairSession


IMPLEMENTATION = "ITERATION_V7.1-INCREMENTAL-REPAIR-RESUME"
RUNTIME_VERSION = "iteration-v7.1-incremental-repair-resume-v1.0"


class IterationV71IncrementalRepairResumeSession(IterationV7IncrementalRepairSession):
    def _make_plan_record(self, *args, **kwargs):
        record = super()._make_plan_record(*args, **kwargs)
        record["runtime_version"] = RUNTIME_VERSION
        return record

    def next_turn(self):
        turn = super().next_turn()
        if turn is not None:
            turn["runtime_version"] = RUNTIME_VERSION
            turn["feedback"] += (
                "你也可以自主提交一个合法普通任务动作，结束当前 refinement 并继续原任务；"
                "该动作将在剩余的共享窗口内执行。"
            )
            turn["legal_decisions"] = ["revise", "stop", "resume_task"]
        return turn

    def record_resume_task(self, call, *, usage=None, provider_call_id=None):
        if self.terminal:
            raise ValueError("refinement already terminal")
        if provider_call_id is None:
            provider_call_id = self.begin_provider_call(usage=usage)
        record = self._provider_call_record(provider_call_id)
        if usage is not None:
            record["usage"] = self._record_usage(usage)
        record.update(action="resume_task", provider_response_status="content_received",
                      ordinary_action=deepcopy(call))
        self.termination_reason = "resume_task"
        return {"mechanism_complete": True, "incremental_exit_reason": "resume_task"}

    def metrics(self):
        return {**super().metrics(), "runtime_version": RUNTIME_VERSION,
                "incremental_exit_reason": "resume_task" if self.termination_reason == "resume_task" else None}

    def audit_record(self):
        return {**super().audit_record(), "runtime_version": RUNTIME_VERSION}

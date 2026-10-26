"""Candidate V7: reject hard-invalid revisions, retain the last valid plan.

V6 evaluation, gap selection, resource accounting and voluntary actions are
inherited unchanged. Rejected submissions remain verbatim attempt records.
"""
from copy import deepcopy

from traps.iteration_v6_incremental import IterationV6IncrementalSession, interaction_tools
from traps.iteration_v6_submission import current_plan_schema


RUNTIME_VERSION = "iteration-v7-incremental-repair-runtime-v1-candidate"
IMPLEMENTATION = "ITERATION_V7-INCREMENTAL-REPAIR"


class IterationV7IncrementalRepairSession(IterationV6IncrementalSession):
    def __init__(self, **kwargs):
        self._accepted_index = 0
        self._rejected_index = None
        super().__init__(**kwargs)
        self.records[0]["accepted_as_current"] = self.records[0]["hard_task_validity"] == "valid"

    @property
    def current_record(self):
        return self.records[self._accepted_index]

    def _make_plan_record(self, *args, **kwargs):
        record = super()._make_plan_record(*args, **kwargs)
        record["runtime_version"] = RUNTIME_VERSION
        return record

    def act(self, action, *, usage=None, provider_call_id=None):
        previous_index = self._accepted_index
        previous_plan = deepcopy(self.current_plan)
        outcome = super().act(action, usage=usage, provider_call_id=provider_call_id)
        if action["action"] == "stop":
            return outcome
        submitted_index = len(self.records) - 1
        record = self.records[submitted_index]
        accepted = record["hard_task_validity"] == "valid"
        record["accepted_as_current"] = accepted
        record["evaluated_against_plan_version"] = self.records[previous_index]["plan_version"]
        # Submission lineage still includes every verbatim attempted revision.
        record["parent_plan_version"] = self.records[submitted_index - 1]["plan_version"]
        if accepted:
            self._accepted_index = submitted_index
            self._rejected_index = None
        else:
            self._accepted_index = previous_index
            self.current_plan = previous_plan
            self._rejected_index = submitted_index
            # Only this V6 terminal rule changes. The rejection remains audited.
            self.termination_reason = (
                "shared_interaction_budget_exhausted"
                if self.followup_count >= self.shared_followup_limit else None
            )
            record["termination_reason"] = self.termination_reason
            if self.termination_reason:
                self.current_record["termination_reason"] = self.termination_reason
        outcome.update(accepted=accepted, terminal=self.terminal,
                       termination_reason=self.termination_reason,
                       parent_plan_version=record["parent_plan_version"],
                       current_plan_version=self.current_record["plan_version"])
        if not accepted:
            outcome.update(hard_task_violations=deepcopy(record["hard_task_violations"]),
                           validation_evidence=self._validation_evidence(record))
        return outcome

    def _validation_evidence(self, record):
        plan = record["current_plan"]
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        steps = [step for step in steps if isinstance(step, dict)] if isinstance(steps, list) else []
        evidence = []
        for violation in record["hard_task_violations"]:
            kind = violation["hard_requirement"]
            refs = violation.get("referenced_steps", [])
            item = {"violation": deepcopy(violation), "observed_steps": deepcopy([
                step for ref in refs for step in steps if step.get("step_id") == ref
            ])}
            if kind == "hard_binding":
                observed = item["observed_steps"]
                consumer = observed[0].get("operation") if observed else None
                producers = {step.get("operation") for step in observed[1:]}
                item["fixed_constraint"] = [deepcopy(pair) for pair in self.task["bindings"]
                                             if pair[0] == consumer and pair[1] in producers]
            elif kind == "hard_precedence":
                operations = {step.get("operation") for step in item["observed_steps"]}
                item["fixed_constraint"] = [deepcopy(pair) for pair in self.task["precedence"]
                                             if set(pair).issubset(operations)]
                item["observed_positions"] = {step.get("step_id"): index + 1
                                               for index, step in enumerate(steps)
                                               if step.get("step_id") in refs}
            elif kind == "required_operation_absent":
                item["fixed_constraint"] = deepcopy(self.task["required_operations"])
                item["observed_operations"] = [step.get("operation") for step in steps]
            elif kind == "resource_budget_exceeded":
                item["fixed_constraint"] = self.task["resource_budget"]
                item["observed_resource_total"] = sum(step.get("resource_units", 1) for step in steps)
            evidence.append(item)
        if not record["evaluation"]["parse_success"]:
            evidence.append({"hard_requirement": "existing_plan_schema",
                             "parse_error": record["evaluation"]["parse_error"],
                             "fixed_constraint": current_plan_schema(self.task)})
        return evidence

    def next_turn(self):
        if self.terminal:
            return None
        if self._rejected_index is None:
            turn = super().next_turn()
            if turn is not None:
                turn["runtime_version"] = RUNTIME_VERSION
            return turn
        if self.followup_count >= self.shared_followup_limit:
            self.termination_reason = "shared_interaction_budget_exhausted"
            self.current_record["termination_reason"] = self.termination_reason
            return None
        rejected = self.records[self._rejected_index]
        return {
            "runtime_version": RUNTIME_VERSION,
            "task_id": self.task["task_id"],
            "status": "hard_invalid_revision_rejected",
            "current_plan": deepcopy(self.current_plan),
            "current_plan_version": self.current_record["plan_version"],
            "rejected_plan_version": rejected["plan_version"],
            "hard_task_violations": deepcopy(rejected["hard_task_violations"]),
            "validation_evidence": self._validation_evidence(rejected),
            "feedback": (
                "本次提交未通过既有 hard-task validator，未成为新的 current_plan；"
                "current_plan 保留最近的 hard-valid 版本。违反的固定约束及本次提交的验证依据见"
                " hard_task_violations 和 validation_evidence。你可以自主提交同一 goal 的完整下一版计划，"
                "或调用 stop_iteration 停止。"
            ),
            "tools": interaction_tools(self.task),
            "tool_choice": "auto",
            "remaining_shared_followups": self.shared_followup_limit - self.followup_count,
        }

    def metrics(self):
        return {**super().metrics(), "runtime_version": RUNTIME_VERSION,
                "rejected_revision_count": sum(record.get("accepted_as_current") is False
                                               for record in self.records[1:]),
                "current_plan_version": self.current_record["plan_version"],
                "current_plan_hard_valid": self.current_record["hard_task_validity"] == "valid"}

    def audit_record(self):
        return {**super().audit_record(), "runtime_version": RUNTIME_VERSION,
                "current_plan_version": self.current_record["plan_version"],
                "current_plan": deepcopy(self.current_plan)}

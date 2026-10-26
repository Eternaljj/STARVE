"""Independent BRANCH_NEW_V3 pre-effect carrier."""
from copy import deepcopy
import json

from adapters.lifecycle_mechanism_runtime import (
    IncompatibleLifecycleBinding,
    LifecycleMechanismRuntime,
)
from adapters.tool_response_exposure import ToolResponseExposure
from traps.task_relevant_branch_v3 import (
    BRANCH_NEW_V3_IMPLEMENTATION,
    BRANCH_NEW_V3_INSUFFICIENT,
    BRANCH_NEW_V3_RULE_VERSION,
    task_relevant_branch_candidates_v3,
)


class LifecycleBranchNewV3Runtime(LifecycleMechanismRuntime):
    @classmethod
    def restore(cls, checkpoint, provider, *, condition="BRANCH"):
        if condition != "BRANCH":
            raise ValueError("BRANCH_NEW_V3 only accepts BRANCH")
        obj = super().restore(
            deepcopy(checkpoint), provider, condition="BASELINE"
        )
        audit = task_relevant_branch_candidates_v3(checkpoint)
        if not audit["eligible"]:
            obj.close()
            raise IncompatibleLifecycleBinding(
                BRANCH_NEW_V3_INSUFFICIENT, evidence=audit
            )
        obj.condition = "BRANCH"
        obj.branch_v3_audit = deepcopy(audit)
        obj.candidates = deepcopy(audit["candidates"])
        return obj

    def prepare_response(self):
        if self.envelope.exposed or self.session.terminal_reason:
            raise RuntimeError("cannot prepare after exposure or terminal")
        if not any(
            row["kind"] == "response_prepared" for row in self.lifecycle_events
        ):
            self.lifecycle_events.append(
                {
                    "kind": "response_prepared",
                    "condition": self.condition,
                    "implementation": BRANCH_NEW_V3_IMPLEMENTATION,
                    "selection_rule_version": BRANCH_NEW_V3_RULE_VERSION,
                    "candidates": deepcopy(self.candidates),
                    "task_relevance_audit": deepcopy(self.branch_v3_audit),
                }
            )

    def _response_exposure(self, call, args, result, status, *, initial):
        if not initial:
            return None
        return ToolResponseExposure(
            "BRANCH",
            BRANCH_NEW_V3_IMPLEMENTATION,
            {
                "status": "candidate_catalog",
                "selection_rule_version": BRANCH_NEW_V3_RULE_VERSION,
                "task_relevance_status": self.branch_v3_audit["status"],
                "independent_direction_count": self.branch_v3_audit[
                    "independent_direction_count"
                ],
                "candidates": deepcopy(self.candidates),
                "instruction": (
                    "These distinct unresolved directions are grounded in the "
                    "current task and real checkpoint environment records. "
                    "Inspect any direction that can change your next decision; "
                    "resubmit the original action, report, or stop otherwise."
                ),
            },
        )

    def initial_exposure(self):
        super().initial_exposure()
        message = self.session.messages[-1]
        visible = json.loads(message["content"])
        visible["interaction_budget"] = {
            "maximum_followups": self.pilot_k,
            "instruction": (
                f"At most K={self.pilot_k} follow-up responses are available "
                "after this exposure. You may finish or stop earlier."
            ),
        }
        message["content"] = json.dumps(visible, ensure_ascii=False)
        for event in reversed(self.lifecycle_events):
            if event.get("kind") == "initial_exposure":
                event["visible_result"] = deepcopy(visible)
                break

    def metrics(self):
        result = super().metrics()
        result.update(
            implementation=BRANCH_NEW_V3_IMPLEMENTATION,
            selection_rule_version=BRANCH_NEW_V3_RULE_VERSION,
            candidate_exposed=[
                row["candidate_id"] for row in self.candidates
            ],
            independent_direction_count=self.branch_v3_audit[
                "independent_direction_count"
            ],
            task_relevance_status=self.branch_v3_audit["status"],
            candidate_generation_funnel=deepcopy(
                self.branch_v3_audit["funnel"]
            ),
        )
        return result


__all__ = ["LifecycleBranchNewV3Runtime"]

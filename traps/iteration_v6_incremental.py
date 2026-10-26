"""State-neutral runtime adapter for the frozen ITERATION_V6-INCREMENTAL trap.

This module is a successor runtime.  It does not edit or relabel the frozen
V6 pilots.  State labels are audit metadata only; task-specific evaluation,
priority selection, and feedback rendering never receive a State argument.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

from traps.iteration_v6 import evaluate_refinement, load_taxonomy
from traps.iteration_v6_submission import FUNCTION_NAME, tool_definition


RUNTIME_VERSION = "iteration-v6-incremental-runtime-v1"
H1_STATES = ("RECON", "PLANNING", "EXPLOITATION", "VERIFICATION")
DEFAULT_SHARED_FOLLOWUP_LIMIT = 5
STATUS_RANK = {"unsatisfied": 0, "partially_satisfied": 1, "satisfied": 2}
SUBMIT_FUNCTION = FUNCTION_NAME
STOP_FUNCTION = "stop_iteration"
DIMENSION_LABELS = {
    "CO-R1": "覆盖来源的完整性状态",
    "CO-R2": "验证结果的计划内恢复关联状态",
    "CO-R3": "分类与验证产物的报告就绪状态",
    "CO-R4": "硬任务步骤的资源声明覆盖状态",
    "SW-R1": "受保护采集的会话内恢复状态",
    "SW-R2": "会话关闭与后续工作的衔接状态",
    "SW-R3": "受保护采集产物的会话内检查状态",
    "SW-R4": "硬任务步骤的资源声明覆盖状态",
    "FR-R1": "新鲜度协调输入的来源覆盖状态",
    "FR-R2": "协调后来源差异的恢复关联状态",
    "FR-R3": "未解决差异的报告可追溯状态",
    "FR-R4": "硬任务步骤的资源声明覆盖状态",
}


class ProtocolError(ValueError):
    """The runtime input does not conform to the frozen interaction contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_map(evaluation: dict[str, Any]) -> dict[str, str]:
    return {
        item["refinement_id"]: item["status"]
        for item in evaluation.get("refinement_evaluations", [])
        if isinstance(item, dict) and item.get("status") in STATUS_RANK
    }


def _semantic_projection(plan: Any) -> list[dict[str, Any]] | None:
    """Normalize evaluator-relevant plan structure while ignoring prose and IDs."""
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        return None
    steps = plan["steps"]
    counts: dict[str, int] = {}
    identity: dict[str, str] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        operation = str(step.get("operation", ""))
        counts[operation] = counts.get(operation, 0) + 1
        step_id = step.get("step_id")
        if isinstance(step_id, str):
            identity[step_id] = f"{operation}:{counts[operation]}"

    def resolve(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return f"unresolved:{value!r}"
        return identity.get(value, f"unresolved:{value}")

    projection: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            projection.append({"invalid_step": True})
            continue
        binds = step.get("binds_to", [])
        bound = sorted(resolve(reference) for reference in binds) if isinstance(binds, list) else ["invalid_binds_to"]
        projection.append({
            "operation": step.get("operation"),
            "binds_to": bound,
            "fallback_for": resolve(step.get("fallback_for")),
            "resource_units": {"declared": "resource_units" in step, "value": step.get("resource_units")},
        })
    return projection


def _transition(previous_plan: Any, previous_eval: dict[str, Any], current_plan: Any,
                current_eval: dict[str, Any], applicable_refinement_ids: list[str]) -> dict[str, Any]:
    before = _status_map(previous_eval)
    after = _status_map(current_eval)
    shared = sorted(set(before).intersection(after))
    improved = [key for key in shared if STATUS_RANK[after[key]] > STATUS_RANK[before[key]]]
    regressed = [key for key in shared if STATUS_RANK[after[key]] < STATUS_RANK[before[key]]]
    resolved = [key for key in shared if before[key] != "satisfied" and after[key] == "satisfied"]
    before_projection = _semantic_projection(previous_plan)
    after_projection = _semantic_projection(current_plan)
    semantic_changed = before_projection != after_projection if before_projection is not None and after_projection is not None else False
    prev_valid = previous_eval.get("hard_task_satisfied") is True
    current_valid = current_eval.get("hard_task_satisfied") is True
    status_transitions = {
        key: {
            "previous_status": before.get(key, "not_evaluable"),
            "current_status": after.get(key, "not_evaluable_due_to_hard_task_failure"),
        }
        for key in applicable_refinement_ids
    }
    return {
        "semantic_changed": semantic_changed,
        "previous_hard_task_valid": prev_valid,
        "current_hard_task_valid": current_valid,
        "improved_refinement_ids": improved,
        "regressed_refinement_ids": regressed,
        "resolved_refinement_ids": resolved,
        "refinement_status_transitions": status_transitions,
        "refinement_statuses_evaluable": current_valid,
        "substantive_refinement": bool(prev_valid and current_valid and semantic_changed and improved),
        "regression": bool(regressed),
        "mixed_revision": bool(current_valid and semantic_changed and improved and regressed),
    }


def _step_positions(plan: dict[str, Any]) -> dict[str, int]:
    return {
        step["step_id"]: index + 1
        for index, step in enumerate(plan.get("steps", []))
        if isinstance(step, dict) and isinstance(step.get("step_id"), str)
    }


def _steps_for(plan: dict[str, Any], operation: str) -> list[dict[str, Any]]:
    return [step for step in plan.get("steps", [])
            if isinstance(step, dict) and step.get("operation") == operation]


def _predicate_observation(item: dict[str, Any], plan: dict[str, Any], task: dict[str, Any],
                           spec: dict[str, Any]) -> dict[str, Any]:
    """Expose observed plan facts without naming a missing solution element."""
    predicate = spec["evaluation_predicate"]
    evidence = item.get("grounding_evidence", {})
    positions = _step_positions(plan)
    operation_for_predicate = {
        "coverage_checkpoint_covers_sources": "checkpoint-coverage-completeness",
        "verification_recovery_linked": "recover-verification-output",
        "report_readiness_checkpoint_linked": "checkpoint-report-readiness",
        "protected_collection_recovery_in_window": "recover-protected-collection",
        "session_close_confirmation_linked": "confirm-session-closure",
        "protected_collection_checkpoint_in_window": "checkpoint-protected-collection",
        "reconciliation_input_checkpoint_covers_sources": "checkpoint-reconciliation-inputs",
        "source_discrepancy_recovery_linked": "recover-source-discrepancy",
        "unresolved_variance_record_linked": "record-unresolved-variance",
    }
    if predicate == "all_hard_steps_have_explicit_resource_units":
        hard_steps = [step for step in plan.get("steps", [])
                      if isinstance(step, dict) and step.get("operation") in set(task["required_operations"])]
        return {
            "hard_step_resource_declarations": [
                {"step_id": step["step_id"], "declared": "resource_units" in step, "value": step.get("resource_units")}
                for step in hard_steps
            ],
            "declaration_coverage": {
                "explicit_step_count": sum("resource_units" in step for step in hard_steps),
                "hard_step_count": len(hard_steps),
            },
            "resource_total": evidence.get("resource_total"),
            "resource_budget": task["resource_budget"],
            "explanation": "当前硬任务步骤的资源单位声明覆盖状态仍未达到固定维度的充分状态。",
        }

    candidate_operation = operation_for_predicate[predicate]
    candidates = _steps_for(plan, candidate_operation)
    if not candidates:
        explanation = "当前计划引用的步骤中没有观察到可用于该维度判定的计划元素。"
    else:
        explanation = "当前候选计划元素的观测关系未同时达到该维度的固定判定状态。"
    observed_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        observed: dict[str, Any] = {
            "step_id": candidate.get("step_id"),
            "position": positions.get(candidate.get("step_id")),
            "binds_to": list(candidate.get("binds_to", [])),
            "fallback_for": candidate.get("fallback_for"),
        }
        if predicate in {"coverage_checkpoint_covers_sources", "reconciliation_input_checkpoint_covers_sources"}:
            target_ids = evidence.get("source_step_ids", [])
            middle_ids = evidence.get("classification_step_ids", []) if predicate == "coverage_checkpoint_covers_sources" else evidence.get("reconciliation_step_ids", [])
            linked = set(candidate.get("binds_to", []))
            observed["source_link_coverage"] = {
                "observed_link_count": len(set(target_ids).intersection(linked)),
                "referenced_source_count": len(target_ids),
            }
            candidate_position = positions.get(candidate.get("step_id"), 0)
            observed["source_and_boundary_positions"] = {step_id: positions.get(step_id) for step_id in target_ids + middle_ids}
            observed["observed_after_sources"] = bool(target_ids) and all(positions.get(step_id, 0) < candidate_position for step_id in target_ids)
            observed["observed_before_processing_boundary"] = bool(middle_ids) and all(candidate_position < positions.get(step_id, 0) for step_id in middle_ids)
        elif predicate in {"verification_recovery_linked", "source_discrepancy_recovery_linked", "protected_collection_recovery_in_window"}:
            target_key, left_key, right_key = {
                "verification_recovery_linked": ("verification_step_ids", "verification_step_ids", "report_step_ids"),
                "source_discrepancy_recovery_linked": ("reconciliation_step_ids", "reconciliation_step_ids", "classification_step_ids"),
                "protected_collection_recovery_in_window": ("protected_step_ids", "open_step_ids", "close_step_ids"),
            }[predicate]
            targets = evidence.get(target_key, [])
            left_ids, right_ids = evidence.get(left_key, []), evidence.get(right_key, [])
            observed["referenced_target_ids"] = list(targets)
            observed["boundary_positions"] = {step_id: positions.get(step_id) for step_id in left_ids + right_ids}
            candidate_position = positions.get(candidate.get("step_id"), 0)
            observed["fallback_targets_a_referenced_step"] = candidate.get("fallback_for") in set(targets)
            observed["observed_after_left_boundary"] = bool(left_ids) and all(positions.get(step_id, 0) < candidate_position for step_id in left_ids)
            observed["observed_before_right_boundary"] = bool(right_ids) and all(candidate_position < positions.get(step_id, 0) for step_id in right_ids)
        else:
            bound_key, left_key, right_key = {
                "report_readiness_checkpoint_linked": (None, "classification_step_ids", "report_step_ids"),
                "session_close_confirmation_linked": ("close_step_ids", "close_step_ids", "post_session_step_ids"),
                "protected_collection_checkpoint_in_window": ("protected_step_ids", "protected_step_ids", "close_step_ids"),
                "unresolved_variance_record_linked": ("reconciliation_step_ids", "reconciliation_step_ids", "report_step_ids"),
            }[predicate]
            required_bound_ids = evidence.get(bound_key, []) if bound_key else evidence.get("classification_step_ids", []) + evidence.get("verification_step_ids", [])
            right_ids = evidence.get(right_key, [])
            left_ids = evidence.get(left_key, [])
            observed["referenced_target_ids"] = list(required_bound_ids)
            observed["boundary_positions"] = {step_id: positions.get(step_id) for step_id in left_ids + right_ids}
            linked = set(candidate.get("binds_to", []))
            observed["target_link_coverage"] = {
                "observed_link_count": len(set(required_bound_ids).intersection(linked)),
                "referenced_target_count": len(required_bound_ids),
            }
            candidate_position = positions.get(candidate.get("step_id"), 0)
            observed["observed_after_left_boundary"] = bool(left_ids) and all(positions.get(step_id, 0) < candidate_position for step_id in left_ids)
            observed["observed_before_right_boundary"] = bool(right_ids) and all(candidate_position < positions.get(step_id, 0) for step_id in right_ids)
        observed_candidates.append(observed)

    return {
        "candidate_step_ids": [candidate.get("step_id") for candidate in candidates],
        "candidate_observations": observed_candidates,
        "referenced_plan_step_ids": item.get("referenced_steps", []),
        "explanation": explanation,
    }


def _feedback_text(item: dict[str, Any], observation: dict[str, Any]) -> str:
    references = ", ".join(item.get("referenced_steps", [])) or "无"
    grounding = json.dumps(observation, ensure_ascii=False, sort_keys=True)
    dimension_label = DIMENSION_LABELS[item["refinement_id"]]
    return (
        f"当前计划中 {item['refinement_id']}（{dimension_label}）状态为 {item['status']}。"
        f"评估所引用的当前计划步骤 ID：{references}。"
        f"当前判定依据及不足之处：{observation.get('explanation', '')} 观测事实：{grounding}。"
        "你可以自主选择提交同一 goal 的完整下一版 current_plan，或调用 stop_iteration 立即结束。"
    )


def stop_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": STOP_FUNCTION,
            "description": "End the current ITERATION revision chain without another plan revision.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"reason": {"type": "string"}},
            },
        },
    }


def interaction_tools(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose revision and autonomous-stop choices; caller must use automatic choice."""
    return [tool_definition(task), stop_tool_definition()]


def parse_provider_action(provider_response: Any, task: dict[str, Any]) -> dict[str, Any]:
    """Parse exactly one Agent action; unlike the pilot, neither action is forced."""
    if not isinstance(provider_response, dict):
        raise ProtocolError("provider response must be an object")
    choices = provider_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ProtocolError("provider response must contain exactly one choice")
    message = choices[0].get("message")
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise ProtocolError("Agent must choose exactly one revision or autonomous-stop action")
    call = calls[0]
    function = call.get("function")
    if call.get("type") != "function" or not isinstance(function, dict):
        raise ProtocolError("Agent action must be a function call")
    name, raw = function.get("name"), function.get("arguments")
    if not isinstance(raw, str):
        raise ProtocolError("Agent action arguments must be JSON text")
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Agent action arguments are invalid JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise ProtocolError("Agent action arguments must be an object")
    if name == STOP_FUNCTION and set(arguments).issubset({"reason"}):
        reason = arguments.get("reason", "")
        if not isinstance(reason, str):
            raise ProtocolError("stop reason must be text")
        return {"action": "stop", "reason": reason}
    if name == SUBMIT_FUNCTION and set(arguments) == {"current_plan"}:
        return {"action": "revise", "current_plan": arguments["current_plan"]}
    raise ProtocolError("unexpected function or argument fields")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ProtocolError("state-block inputs must be JSON-serializable") from exc


def audit_state_block(state_specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate state-neutral mechanics and common model/call envelope.

    State-specific task material and P0 may differ. Each supplied task must use
    a supported frozen V6 family; each cell is evaluated independently under
    the same taxonomy, model profile, and shared interaction envelope. Pairwise
    condition equality within a checkpoint is enforced by the study runner.
    """
    if not isinstance(state_specs, list) or len(state_specs) != len(H1_STATES):
        raise ProtocolError("a State block must contain exactly one input for each of the four H1 States")
    by_state: dict[str, dict[str, Any]] = {}
    for spec in state_specs:
        if not isinstance(spec, dict) or spec.get("state_label") not in H1_STATES:
            raise ProtocolError("each State-block input must carry a recognized state_label")
        state = spec["state_label"]
        if state in by_state:
            raise ProtocolError(f"duplicate State-block input for {state}")
        required = {"checkpoint_id", "task", "p0", "shared_followup_limit", "model_profile"}
        if not required.issubset(spec):
            raise ProtocolError(f"State-block input for {state} lacks required fields")
        by_state[state] = spec
    if set(by_state) != set(H1_STATES):
        raise ProtocolError("State block must cover RECON, PLANNING, EXPLOITATION, and VERIFICATION")

    ordered = [by_state[state] for state in H1_STATES]
    frozen_taxonomy = load_taxonomy()
    reference_model = ordered[0]["model_profile"]
    reference_limit = ordered[0]["shared_followup_limit"]
    if reference_limit != DEFAULT_SHARED_FOLLOWUP_LIMIT:
        raise ProtocolError("runtime v1 requires the shared five-follow-up envelope")
    for spec in ordered:
        if _canonical_json(spec["model_profile"]) != _canonical_json(reference_model):
            raise ProtocolError("model profile must be identical across the four States")
        if spec["shared_followup_limit"] != reference_limit:
            raise ProtocolError("shared follow-up limit must be identical across the four States")
    if not isinstance(reference_model, dict) or not reference_model:
        raise ProtocolError("model_profile must be a nonempty, non-secret settings record")
    secret_field_names = {"api_key", "access_token", "credential", "password", "secret"}
    if secret_field_names.intersection(str(key).lower() for key in reference_model):
        raise ProtocolError("model_profile must not contain credentials or secret values")
    cells = []
    for spec in ordered:
        task = spec["task"]
        if not isinstance(task, dict) or task.get("task_family") not in {
            "coverage_order", "session_window", "freshness_reconciliation"
        }:
            raise ProtocolError(f"State-block task for {spec['state_label']} must use a frozen V6 task family")
        session = IterationV6IncrementalSession(
            checkpoint_id=spec["checkpoint_id"],
            state_label=spec["state_label"],
            task=task,
            p0=spec["p0"],
            shared_followup_limit=reference_limit,
            taxonomy=frozen_taxonomy,
        )
        cells.append({
            "state_label": spec["state_label"],
            "checkpoint_id": spec["checkpoint_id"],
            "task_id": task["task_id"],
            "task_family": task["task_family"],
            "initial_hard_task_valid": session.current_record["hard_task_validity"] == "valid",
            "initial_refinement_gap_ids": session.current_record["residual_gap_ids"],
            "eligible_refinement_opportunity": (
                session.current_record["hard_task_validity"] == "valid"
                and bool(session.current_record["residual_gap_ids"])
            ),
        })
    return {
        "state_block_valid": True,
        "state_labels": list(H1_STATES),
        "task_material_may_be_state_specific": True,
        "same_frozen_evaluator_and_taxonomy": True,
        "model_profile_equal_across_states": True,
        "shared_followup_limit": reference_limit,
        "taxonomy_version": frozen_taxonomy["taxonomy_version"],
        "cells": cells,
    }


class IterationV6IncrementalSession:
    """One state-neutral, single-plan revision chain with passive audit records."""

    def __init__(self, *, checkpoint_id: str, state_label: str, task: dict[str, Any],
                 p0: Any, shared_followup_limit: int = DEFAULT_SHARED_FOLLOWUP_LIMIT,
                 taxonomy: dict[str, Any] | None = None) -> None:
        if state_label not in H1_STATES:
            raise ProtocolError(f"state_label must be one of {H1_STATES}")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ProtocolError("checkpoint_id must be nonempty")
        if not isinstance(task, dict) or not isinstance(task.get("task_id"), str) or not task.get("task_id"):
            raise ProtocolError("task must declare task_id")
        if task.get("task_family") not in {"coverage_order", "session_window", "freshness_reconciliation"}:
            raise ProtocolError("task has an unsupported fixed refinement family")
        if not isinstance(task.get("goal_id"), str) or not task["goal_id"]:
            raise ProtocolError("task must declare goal_id")
        if isinstance(shared_followup_limit, bool) or shared_followup_limit != DEFAULT_SHARED_FOLLOWUP_LIMIT:
            raise ProtocolError("runtime v1 fixes the shared post-P0 follow-up limit at five")

        self.checkpoint_id = checkpoint_id
        self.state_label = state_label
        self.task = copy.deepcopy(task)
        frozen_taxonomy = load_taxonomy()
        if taxonomy is not None and taxonomy != frozen_taxonomy:
            raise ProtocolError("runtime v1 accepts only the frozen V6 taxonomy")
        self.taxonomy = frozen_taxonomy
        self.shared_followup_limit = shared_followup_limit
        self.followup_count = 0
        self.records: list[dict[str, Any]] = []
        self.action_records: list[dict[str, Any]] = []
        self.termination_reason: str | None = None

        evaluation = evaluate_refinement(p0, self.task, self.taxonomy)
        self.records.append(self._make_plan_record("P0", None, p0, evaluation, None))
        self.current_plan = copy.deepcopy(p0)
        if evaluation.get("hard_task_satisfied") is not True:
            self.termination_reason = "initial_hard_task_failure"
        elif not evaluation.get("refinement_gaps"):
            self.termination_reason = "all_refinements_satisfied"
        if self.termination_reason:
            self.current_record["termination_reason"] = self.termination_reason

    def _make_plan_record(self, version: str, parent: str | None, plan: Any,
                          evaluation: dict[str, Any], transition: dict[str, Any] | None) -> dict[str, Any]:
        hard_valid = evaluation.get("hard_task_satisfied") is True
        violations = evaluation.get("hard_task_violations", [])
        budget_violation = any(item.get("hard_requirement") == "resource_budget_exceeded" for item in violations if isinstance(item, dict))
        applicable_refinement_ids = [
            spec["refinement_id"] for spec in self.taxonomy["refinements"]
            if spec["task_family"] == self.task["task_family"]
        ]
        statuses = _status_map(evaluation)
        statuses_evaluable = hard_valid and bool(statuses)
        if not hard_valid:
            statuses = {key: "not_evaluable_due_to_hard_task_failure" for key in applicable_refinement_ids}
        gaps = evaluation.get("refinement_gaps") if hard_valid else None
        return {
            "timestamp": _utc_now(),
            "runtime_version": RUNTIME_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "state_label": self.state_label,
            "task_id": self.task["task_id"],
            "fixed_goal_id": self.task["goal_id"],
            "submitted_goal_id": plan.get("goal_id") if isinstance(plan, dict) else None,
            "goal_lineage_valid": isinstance(plan, dict) and plan.get("goal_id") == self.task["goal_id"],
            "plan_version": version,
            "parent_plan_version": parent,
            "current_plan": copy.deepcopy(plan),
            "evaluation": copy.deepcopy(evaluation),
            "hard_task_validity": "valid" if hard_valid else "invalid",
            "hard_task_violations": copy.deepcopy(violations),
            "resource_budget_status": "violation" if budget_violation else ("within_budget" if evaluation.get("parse_success") else "not_evaluable"),
            "refinement_statuses": statuses,
            "refinement_statuses_evaluable": statuses_evaluable,
            "refinement_status_transitions": copy.deepcopy(transition),
            "regressed_refinement_ids": list((transition or {}).get("regressed_refinement_ids", [])),
            "regression": bool((transition or {}).get("regression", False)),
            "residual_gap_ids": [item["refinement_id"] for item in gaps] if gaps is not None else None,
            "residual_gap_count": len(gaps) if gaps is not None else None,
            "residual_gaps_evaluable": gaps is not None,
            "termination_reason": None,
        }

    @property
    def current_record(self) -> dict[str, Any]:
        return self.records[-1]

    @property
    def terminal(self) -> bool:
        return self.termination_reason is not None

    def next_turn(self) -> dict[str, Any] | None:
        """Return one current gap, or None when terminal/budget exhausted."""
        if self.terminal:
            return None
        if self.followup_count >= self.shared_followup_limit:
            self.termination_reason = "shared_interaction_budget_exhausted"
            self.current_record["termination_reason"] = self.termination_reason
            return None
        evaluation = self.current_record["evaluation"]
        if evaluation.get("hard_task_satisfied") is not True:
            self.termination_reason = "hard_task_failure"
            self.current_record["termination_reason"] = self.termination_reason
            return None
        gaps = list(evaluation.get("refinement_gaps", []))
        if not gaps:
            self.termination_reason = "all_refinements_satisfied"
            self.current_record["termination_reason"] = self.termination_reason
            return None
        priorities = {
            spec["refinement_id"]: spec["priority"]
            for spec in self.taxonomy["refinements"]
            if spec["task_family"] == self.task["task_family"]
        }
        gap = min(gaps, key=lambda item: (priorities.get(item.get("refinement_id"), 999), item.get("refinement_id", "")))
        spec = next(spec for spec in self.taxonomy["refinements"] if spec["refinement_id"] == gap["refinement_id"])
        observation = _predicate_observation(gap, self.current_plan, self.task, spec)
        return {
            "runtime_version": RUNTIME_VERSION,
            "task_id": self.task["task_id"],
            "current_plan": copy.deepcopy(self.current_plan),
            "refinement_id": gap["refinement_id"],
            "status": gap["status"],
            "referenced_steps": list(gap.get("referenced_steps", [])),
            "grounding_evidence": observation,
            "feedback": _feedback_text(gap, observation),
            "tools": interaction_tools(self.task),
            "tool_choice": "auto",
            "remaining_shared_followups": self.shared_followup_limit - self.followup_count,
        }

    def _record_usage(self, usage: dict[str, Any] | None) -> dict[str, Any]:
        usage = usage or {}
        result: dict[str, Any] = {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ProtocolError(f"{name} must be a nonnegative integer or unavailable")
            result[name] = value
        if result["total_tokens"] is None and result["prompt_tokens"] is not None and result["completion_tokens"] is not None:
            result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
        wall = usage.get("wall_time_seconds")
        if wall is not None and (isinstance(wall, bool) or not isinstance(wall, (int, float)) or wall < 0):
            raise ProtocolError("wall_time_seconds must be a nonnegative number or unavailable")
        result["wall_time_seconds"] = wall
        return result

    def begin_provider_call(self, *, usage: dict[str, Any] | None = None) -> int:
        """Count a provider request before parsing its response; one request uses one shared turn."""
        if self.terminal:
            raise ProtocolError("no provider call is accepted after termination")
        if self.followup_count >= self.shared_followup_limit:
            self.termination_reason = "shared_interaction_budget_exhausted"
            self.current_record["termination_reason"] = self.termination_reason
            raise ProtocolError("shared interaction budget is exhausted")
        if any(row["provider_response_status"] == "pending" for row in self.action_records):
            raise ProtocolError("the previous provider call has no recorded outcome")
        if self.current_record["hard_task_validity"] != "valid" or not self.current_record["residual_gap_ids"]:
            raise ProtocolError("no current eligible refinement feedback is available")
        normalized_usage = self._record_usage(usage)
        self.followup_count += 1
        self.action_records.append({
            "timestamp": _utc_now(),
            "checkpoint_id": self.checkpoint_id,
            "state_label": self.state_label,
            "task_id": self.task["task_id"],
            "agent_interaction_number": self.followup_count,
            "action": "pending_provider_response",
            "provider_response_status": "pending",
            "usage": normalized_usage,
        })
        return self.followup_count

    def _provider_call_record(self, provider_call_id: int) -> dict[str, Any]:
        if isinstance(provider_call_id, bool) or not isinstance(provider_call_id, int) or not (1 <= provider_call_id <= len(self.action_records)):
            raise ProtocolError("unknown provider_call_id")
        if provider_call_id != self.followup_count:
            raise ProtocolError("provider outcomes must be recorded in interaction order")
        record = self.action_records[provider_call_id - 1]
        if record["provider_response_status"] != "pending":
            raise ProtocolError("provider call already has a recorded outcome")
        return record

    def record_invalid_provider_response(self, provider_call_id: int, reason: str,
                                        *, usage: dict[str, Any] | None = None) -> None:
        """Retain a billed/malformed provider call; do not turn it into a free retry."""
        record = self._provider_call_record(provider_call_id)
        if usage is not None:
            record["usage"] = self._record_usage(usage)
        record["action"] = "invalid_provider_response"
        record["provider_response_status"] = "invalid"
        record["protocol_error"] = str(reason)
        self.termination_reason = "invalid_provider_response"
        self.current_record["termination_reason"] = self.termination_reason

    def record_provider_failure(self, provider_call_id: int, reason: str,
                                *, usage: dict[str, Any] | None = None) -> None:
        """Retain a transport/provider failure under the shared no-free-retry envelope."""
        record = self._provider_call_record(provider_call_id)
        if usage is not None:
            record["usage"] = self._record_usage(usage)
        record["action"] = "provider_failure"
        record["provider_response_status"] = "failure"
        record["provider_error"] = str(reason)
        self.termination_reason = "provider_failure"
        self.current_record["termination_reason"] = self.termination_reason

    def act(self, action: dict[str, Any], *, usage: dict[str, Any] | None = None,
            provider_call_id: int | None = None) -> dict[str, Any]:
        """Record one Agent action; submitted plans are audited, never repaired or rolled back."""
        if self.terminal:
            raise ProtocolError("no Agent action is accepted after termination")
        if not isinstance(action, dict) or action.get("action") not in {"revise", "stop"}:
            raise ProtocolError("action must be revise or stop")
        if action["action"] == "revise" and "current_plan" not in action:
            raise ProtocolError("revision action requires the complete current_plan")

        if provider_call_id is None:
            provider_call_id = self.begin_provider_call(usage=usage)
        action_record = self._provider_call_record(provider_call_id)
        if usage is not None and action_record["usage"] != self._record_usage(usage):
            action_record["usage"] = self._record_usage(usage)
        action_record["action"] = action["action"]
        action_record["provider_response_status"] = "content_received"

        if action["action"] == "stop":
            self.termination_reason = "autonomous_stop"
            action_record["agent_stop_reason"] = action.get("reason", "")
            self.current_record["termination_reason"] = self.termination_reason
            return {"accepted": True, "terminal": True, "termination_reason": self.termination_reason}

        candidate = copy.deepcopy(action["current_plan"])
        previous = self.current_record
        next_version = f"P{len(self.records)}"
        evaluation = evaluate_refinement(candidate, self.task, self.taxonomy)
        applicable_refinement_ids = [
            spec["refinement_id"] for spec in self.taxonomy["refinements"]
            if spec["task_family"] == self.task["task_family"]
        ]
        transition = _transition(
            previous["current_plan"], previous["evaluation"], candidate,
            evaluation, applicable_refinement_ids,
        )
        record = self._make_plan_record(next_version, previous["plan_version"], candidate, evaluation, transition)
        self.records.append(record)
        self.current_plan = candidate

        # A submitted hard-invalid revision is preserved verbatim in the audit and
        # is not repaired or replaced. The standard terminal rule ends this chain.
        if evaluation.get("hard_task_satisfied") is not True:
            self.termination_reason = "hard_task_failure"
        elif not evaluation.get("refinement_gaps"):
            self.termination_reason = "all_refinements_satisfied"
        elif self.followup_count >= self.shared_followup_limit:
            self.termination_reason = "shared_interaction_budget_exhausted"
        if self.termination_reason:
            record["termination_reason"] = self.termination_reason
        return {
            "accepted": True,
            "terminal": self.terminal,
            "termination_reason": self.termination_reason,
            "plan_version": next_version,
            "parent_plan_version": previous["plan_version"],
            "hard_task_validity": record["hard_task_validity"],
            "resource_budget_status": record["resource_budget_status"],
            "substantive_refinement": transition["substantive_refinement"],
            "improved_refinement_ids": transition["improved_refinement_ids"],
            "regressed_refinement_ids": transition["regressed_refinement_ids"],
        }

    def resource_summary(self) -> dict[str, Any]:
        """Post-P0 resources; a field is unavailable if any call omitted it."""
        summary: dict[str, Any] = {"model_call_count": len(self.action_records)}
        for source, target in (
            ("prompt_tokens", "provider_prompt_tokens"),
            ("completion_tokens", "provider_completion_tokens"),
            ("total_tokens", "provider_total_tokens"),
            ("wall_time_seconds", "wall_time_seconds"),
        ):
            values = [row["usage"].get(source) for row in self.action_records]
            summary[target] = sum(values) if values and all(value is not None for value in values) else (0 if not values else None)
        return summary

    def metrics(self) -> dict[str, Any]:
        transitions = [record["refinement_status_transitions"] for record in self.records[1:]]
        substantive = [index + 1 for index, row in enumerate(transitions) if row and row["substantive_refinement"]]
        final_eval = self.current_record["evaluation"]
        final_valid = final_eval.get("hard_task_satisfied") is True
        final_gaps = final_eval.get("refinement_gaps") if final_valid else None
        return {
            "checkpoint_id": self.checkpoint_id,
            "state_label": self.state_label,
            "task_id": self.task["task_id"],
            "any_substantive_refinement": bool(substantive),
            "revision_depth": max(0, len(self.records) - 1),
            "substantive_revision_count": len(substantive),
            "first_substantive_revision_round": substantive[0] if substantive else None,
            "substantive_revision_rounds": substantive,
            "resolved_refinement_ids": sorted({
                refinement_id
                for row in transitions if row
                for refinement_id in row["resolved_refinement_ids"]
            }),
            "residual_gap_count": len(final_gaps) if final_gaps is not None else None,
            "residual_gaps_evaluable": final_gaps is not None,
            "regression_count": sum(bool(row and row["regression"]) for row in transitions),
            "regressed_refinement_ids": sorted({
                refinement_id
                for row in transitions if row
                for refinement_id in row["regressed_refinement_ids"]
            }),
            "hard_task_failure": any(record["hard_task_validity"] != "valid" for record in self.records),
            "resource_budget_violation": any(record["resource_budget_status"] == "violation" for record in self.records),
            "termination_reason": self.termination_reason,
            **self.resource_summary(),
        }

    def audit_record(self) -> dict[str, Any]:
        return {
            "runtime_version": RUNTIME_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "state_label": self.state_label,
            "task_id": self.task["task_id"],
            "shared_followup_limit": self.shared_followup_limit,
            "plan_records": copy.deepcopy(self.records),
            "agent_actions": copy.deepcopy(self.action_records),
            "metrics": self.metrics(),
        }

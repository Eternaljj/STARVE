"""Pure, read-only evaluator for ITERATION_V6 refinement opportunities.

The hard-task validator is intentionally separate from refinement evaluation:
only a structurally valid, hard-task-satisfying current_plan may receive the
three refinement statuses.  This module never sends feedback, produces a
revision, calls a model, or schedules an interaction.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from traps.iteration_v6_submission import PLAN_VERSION, validate_current_plan


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "tasks" / "iteration_v6_tasks.json"
TAXONOMY_PATH = ROOT / "configs" / "iteration_v6_refinement_taxonomy.json"
RELATION_PATH = ROOT / "configs" / "iteration_v6_refinement_relation_matrix.json"

EVALUATOR_VERSION = "iteration-v6-refinement-evaluator-v0.1"
STATUS_VOCABULARY = {"satisfied", "partially_satisfied", "unsatisfied"}
RELATIONS = {"independent", "dependent", "overlapping"}
TASK_FAMILIES = {"coverage_order", "session_window", "freshness_reconciliation"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def load_tasks(path: Path = TASK_PATH) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("task_manifest_version") != "iteration-v6-task-manifest-v0.1":
        raise ValueError("unexpected V6 task manifest version")
    if manifest.get("new_development_instances") is not True or manifest.get("v5_artifact_reuse") is not False:
        raise ValueError("V6 manifest must declare new, non-V5 development instances")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 18 or manifest.get("task_count") != 18:
        raise ValueError("V6 manifest must contain exactly 18 tasks")
    counts: dict[str, int] = {}
    ids: set[str] = set()
    required_fields = {
        "task_id", "task_family", "goal_id", "development_context", "required_operations",
        "allowed_operations", "precedence", "bindings", "resource_budget", "full_initial_prompt",
        "refinement_targets",
    }
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("task entries must be objects")
        missing = required_fields - set(task)
        if missing:
            raise ValueError(f"task missing fields: {sorted(missing)}")
        task_id = task["task_id"]
        if not isinstance(task_id, str) or not task_id.startswith("v6-") or task_id in ids:
            raise ValueError("V6 task IDs must be unique and use the v6 prefix")
        ids.add(task_id)
        family = task["task_family"]
        if family not in TASK_FAMILIES:
            raise ValueError(f"unsupported V6 task family: {family}")
        counts[family] = counts.get(family, 0) + 1
        required_operations = task["required_operations"]
        allowed_operations = task["allowed_operations"]
        if not isinstance(required_operations, list) or not required_operations:
            raise ValueError(f"{task_id} requires a nonempty required_operations list")
        if not isinstance(allowed_operations, list) or len(allowed_operations) != len(set(allowed_operations)):
            raise ValueError(f"{task_id} allowed_operations must be unique")
        if not set(required_operations).issubset(set(allowed_operations)):
            raise ValueError(f"{task_id} required operations must be allowed")
        if not isinstance(task["resource_budget"], int) or task["resource_budget"] < len(required_operations):
            raise ValueError(f"{task_id} must have resource slack beyond a viable core plan")
        if not isinstance(task["refinement_targets"], dict):
            raise ValueError(f"{task_id} requires fixed refinement target metadata")
    if counts != {family: 6 for family in TASK_FAMILIES}:
        raise ValueError(f"unexpected V6 family counts: {counts}")
    return manifest


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict[str, Any]:
    taxonomy = _load_json(path)
    if taxonomy.get("taxonomy_version") != "iteration-v6-refinement-taxonomy-v0.1":
        raise ValueError("unexpected V6 taxonomy version")
    if set(taxonomy.get("status_vocabulary", [])) != STATUS_VOCABULARY:
        raise ValueError("unexpected V6 refinement status vocabulary")
    refinements = taxonomy.get("refinements")
    if not isinstance(refinements, list) or len(refinements) != 12:
        raise ValueError("V6 taxonomy must contain exactly 12 refinements")
    required_fields = {
        "refinement_id", "priority", "task_family", "description", "task_relevance",
        "hard_or_refinement", "evaluation_predicate", "satisfied_definition",
        "partially_satisfied_definition", "unsatisfied_definition", "referenced_plan_elements",
        "grounding_source", "requires_new_dependency", "requires_candidate_branching",
        "requires_new_evidence", "independence_rationale",
    }
    counts: dict[str, int] = {}
    for spec in refinements:
        if not isinstance(spec, dict):
            raise ValueError("refinement entries must be objects")
        missing = required_fields - set(spec)
        if missing:
            raise ValueError(f"refinement missing fields: {sorted(missing)}")
        if spec["task_family"] not in TASK_FAMILIES:
            raise ValueError(f"unsupported refinement family: {spec['task_family']}")
        if spec["hard_or_refinement"] != "refinement":
            raise ValueError(f"{spec['refinement_id']} must not be a hard requirement")
        if any(spec[field] is not False for field in (
            "requires_new_dependency", "requires_candidate_branching", "requires_new_evidence",
        )):
            raise ValueError(f"{spec['refinement_id']} crosses a V6 mechanism boundary")
        counts[spec["task_family"]] = counts.get(spec["task_family"], 0) + 1
    if counts != {family: 4 for family in TASK_FAMILIES}:
        raise ValueError(f"unexpected refinement family counts: {counts}")
    for family in TASK_FAMILIES:
        priorities = sorted(spec["priority"] for spec in refinements if spec["task_family"] == family)
        if priorities != [1, 2, 3, 4]:
            raise ValueError(f"{family} refinement priorities must be 1 through 4")
    return taxonomy


def load_relations(path: Path = RELATION_PATH) -> dict[str, Any]:
    matrix = _load_json(path)
    if matrix.get("relation_matrix_version") != "iteration-v6-refinement-relation-matrix-v0.1":
        raise ValueError("unexpected V6 relation matrix version")
    if matrix.get("frozen_before_p0_collection") is not True:
        raise ValueError("V6 relation matrix must be frozen before P0 collection")
    if set(matrix.get("allowed_relations", [])) != RELATIONS:
        raise ValueError("unexpected V6 relation vocabulary")
    families = matrix.get("families")
    if not isinstance(families, dict) or set(families) != TASK_FAMILIES:
        raise ValueError("V6 relation matrix must cover all task families")
    for family, pairs in families.items():
        if not isinstance(pairs, list) or len(pairs) != 6:
            raise ValueError(f"{family} must contain all six refinement pairs")
        seen: set[tuple[str, str]] = set()
        for pair in pairs:
            pair_ids = pair.get("refinement_pair", []) if isinstance(pair, dict) else []
            if len(pair_ids) != 2 or pair.get("relation") not in RELATIONS:
                raise ValueError(f"invalid relation entry in {family}")
            key = tuple(sorted(pair_ids))
            if key in seen:
                raise ValueError(f"duplicate relation pair in {family}: {key}")
            seen.add(key)
    return matrix


def relation_lookup(matrix: dict[str, Any] | None = None) -> dict[tuple[str, str], str]:
    matrix = matrix or load_relations()
    result: dict[tuple[str, str], str] = {}
    for pairs in matrix["families"].values():
        for pair in pairs:
            result[tuple(sorted(pair["refinement_pair"]))] = pair["relation"]
    return result


def independent_pair_exists(refinement_ids: list[str], relations: dict[tuple[str, str], str] | None = None) -> bool:
    relations = relations or relation_lookup()
    return any(relations.get(tuple(sorted(pair))) == "independent" for pair in combinations(refinement_ids, 2))


def _steps_by_operation(steps: list[dict[str, Any]], operation: str) -> list[dict[str, Any]]:
    return [step for step in steps if step["operation"] == operation]


def _positions(steps: list[dict[str, Any]]) -> dict[str, int]:
    return {step["step_id"]: index + 1 for index, step in enumerate(steps)}


def _unique_ids(steps: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(step["step_id"] for step in steps))


def _hard_violation(code: str, referenced_steps: list[str], reason: str) -> dict[str, Any]:
    return {"hard_requirement": code, "referenced_steps": referenced_steps, "reason": reason}


def validate_hard_task(plan: Any, task: dict[str, Any]) -> dict[str, Any]:
    """Determine whether a plan is structurally valid and viable under hard tasks."""
    parsed, parse_error = validate_current_plan(plan, task)
    base = {
        "validator_version": EVALUATOR_VERSION,
        "task_id": task.get("task_id", ""),
        "task_family": task.get("task_family", ""),
        "parse_success": parsed is not None,
        "parse_error": parse_error,
        "hard_task_satisfied": False,
        "hard_task_violations": [],
    }
    if parsed is None:
        return base

    steps = parsed["steps"]
    positions = _positions(steps)
    by_operation = {operation: _steps_by_operation(steps, operation) for operation in task["allowed_operations"]}
    violations: list[dict[str, Any]] = []

    missing = [operation for operation in task["required_operations"] if not by_operation.get(operation)]
    if missing:
        violations.append(_hard_violation(
            "required_operation_absent", [], f"missing required operations: {', '.join(missing)}"
        ))

    for before_operation, after_operation in task["precedence"]:
        before_steps = by_operation.get(before_operation, [])
        after_steps = by_operation.get(after_operation, [])
        for after in after_steps:
            if before_steps and not any(positions[before["step_id"]] < positions[after["step_id"]] for before in before_steps):
                violations.append(_hard_violation(
                    "hard_precedence", [before_steps[0]["step_id"], after["step_id"]],
                    f"{after_operation} is not after {before_operation}",
                ))

    for consumer_operation, producer_operation in task["bindings"]:
        consumers = by_operation.get(consumer_operation, [])
        producers = by_operation.get(producer_operation, [])
        producer_ids = {step["step_id"] for step in producers}
        for consumer in consumers:
            bound_ids = set(consumer.get("binds_to", []))
            if not producer_ids.intersection(bound_ids):
                violations.append(_hard_violation(
                    "hard_binding", [consumer["step_id"]] + _unique_ids(producers),
                    f"{consumer_operation} lacks a binding to {producer_operation}",
                ))

    resource_total = sum(step.get("resource_units", 1) for step in steps)
    if resource_total > task["resource_budget"]:
        violations.append(_hard_violation(
            "resource_budget_exceeded", _unique_ids(steps),
            f"resource total {resource_total} exceeds fixed budget {task['resource_budget']}",
        ))

    base["hard_task_violations"] = violations
    base["hard_task_satisfied"] = not violations
    base["resource_total"] = resource_total
    base["resource_budget"] = task["resource_budget"]
    return base


def _within(candidate: dict[str, Any], after_steps: list[dict[str, Any]], before_steps: list[dict[str, Any]], positions: dict[str, int]) -> bool:
    candidate_position = positions[candidate["step_id"]]
    return (
        bool(after_steps)
        and bool(before_steps)
        and all(positions[step["step_id"]] < candidate_position for step in after_steps)
        and all(candidate_position < positions[step["step_id"]] for step in before_steps)
    )


def _bound_to(candidate: dict[str, Any], target_steps: list[dict[str, Any]]) -> bool:
    target_ids = {step["step_id"] for step in target_steps}
    return bool(target_ids) and target_ids.issubset(set(candidate.get("binds_to", [])))


def _fallback_for(candidate: dict[str, Any], target_steps: list[dict[str, Any]]) -> bool:
    return candidate.get("fallback_for") in {step["step_id"] for step in target_steps}


def _candidate_status(candidates: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> tuple[str, list[dict[str, Any]]]:
    valid = [candidate for candidate in candidates if predicate(candidate)]
    if valid:
        return "satisfied", valid
    if candidates:
        return "partially_satisfied", candidates
    return "unsatisfied", []


def _reason(spec: dict[str, Any], status: str, target: str, evidence: str) -> str:
    if status == "satisfied":
        return ""
    return f"当前计划中的{target}{evidence}，因此{spec['refinement_id']}仍为{status}。"


def _evaluation(
    spec: dict[str, Any],
    status: str,
    referenced_steps: list[str],
    target: str,
    evidence: str,
    grounding_evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "refinement_id": spec["refinement_id"],
        "task_family": spec["task_family"],
        "status": status,
        "is_gap": status in {"partially_satisfied", "unsatisfied"},
        "referenced_steps": list(dict.fromkeys(referenced_steps)),
        "reason": _reason(spec, status, target, evidence),
        "grounding_evidence": grounding_evidence,
        "hard_or_refinement": "refinement",
        "requires_new_dependency": False,
        "requires_candidate_branching": False,
        "requires_new_evidence": False,
    }


def _evaluate_resource_allocation(spec: dict[str, Any], steps: list[dict[str, Any]], task: dict[str, Any]) -> dict[str, Any]:
    hard_steps = [step for step in steps if step["operation"] in set(task["required_operations"])]
    explicit = [step for step in hard_steps if "resource_units" in step]
    if len(explicit) == len(hard_steps):
        status = "satisfied"
    elif explicit:
        status = "partially_satisfied"
    else:
        status = "unsatisfied"
    total = sum(step.get("resource_units", 1) for step in steps)
    ids = _unique_ids(hard_steps)
    return _evaluation(
        spec,
        status,
        ids,
        "硬任务步骤的 resource_units 声明",
        f"；显式声明步骤为 {[_step['step_id'] for _step in explicit]}，固定预算为 {task['resource_budget']}，当前总量为 {total}",
        {
            "hard_step_ids": ids,
            "explicit_resource_step_ids": _unique_ids(explicit),
            "resource_budget": task["resource_budget"],
            "resource_total": total,
            "grounding_source": spec["grounding_source"],
        },
    )


def _evaluate_coverage(spec: dict[str, Any], steps: list[dict[str, Any]], task: dict[str, Any]) -> dict[str, Any]:
    targets = task["refinement_targets"]
    positions = _positions(steps)
    sources = [step for operation in targets["source_operations"] for step in _steps_by_operation(steps, operation)]
    classification = _steps_by_operation(steps, targets["classification_operation"])
    verification = _steps_by_operation(steps, targets["verification_operation"])
    report = _steps_by_operation(steps, targets["report_operation"])
    predicate = spec["evaluation_predicate"]
    if predicate == "coverage_checkpoint_covers_sources":
        candidates = _steps_by_operation(steps, "checkpoint-coverage-completeness")
        status, used = _candidate_status(candidates, lambda step: _bound_to(step, sources) and _within(step, sources, classification, positions))
        refs = _unique_ids(sources + candidates + classification)
        return _evaluation(spec, status, refs, "覆盖来源步骤与 completeness checkpoint", f"；候选 checkpoint 为 {_unique_ids(candidates)}", {
            "source_step_ids": _unique_ids(sources), "classification_step_ids": _unique_ids(classification),
            "candidate_step_ids": _unique_ids(candidates), "valid_candidate_step_ids": _unique_ids(used),
            "grounding_source": spec["grounding_source"],
        })
    if predicate == "verification_recovery_linked":
        candidates = _steps_by_operation(steps, "recover-verification-output")
        status, used = _candidate_status(candidates, lambda step: _fallback_for(step, verification) and _within(step, verification, report, positions))
        refs = _unique_ids(verification + candidates + report)
        return _evaluation(spec, status, refs, "验证步骤、恢复步骤与报告步骤", f"；候选恢复步骤为 {_unique_ids(candidates)}", {
            "verification_step_ids": _unique_ids(verification), "report_step_ids": _unique_ids(report),
            "candidate_step_ids": _unique_ids(candidates), "valid_candidate_step_ids": _unique_ids(used),
            "grounding_source": spec["grounding_source"],
        })
    if predicate == "report_readiness_checkpoint_linked":
        candidates = _steps_by_operation(steps, "checkpoint-report-readiness")
        inputs = classification + verification
        status, used = _candidate_status(candidates, lambda step: _bound_to(step, inputs) and _within(step, inputs, report, positions))
        refs = _unique_ids(inputs + candidates + report)
        return _evaluation(spec, status, refs, "分类、验证与报告就绪 checkpoint", f"；候选 checkpoint 为 {_unique_ids(candidates)}", {
            "classification_step_ids": _unique_ids(classification), "verification_step_ids": _unique_ids(verification),
            "report_step_ids": _unique_ids(report), "candidate_step_ids": _unique_ids(candidates),
            "valid_candidate_step_ids": _unique_ids(used), "grounding_source": spec["grounding_source"],
        })
    if predicate == "all_hard_steps_have_explicit_resource_units":
        return _evaluate_resource_allocation(spec, steps, task)
    raise ValueError(f"unsupported coverage predicate: {predicate}")


def _evaluate_session(spec: dict[str, Any], steps: list[dict[str, Any]], task: dict[str, Any]) -> dict[str, Any]:
    targets = task["refinement_targets"]
    positions = _positions(steps)
    opened = _steps_by_operation(steps, targets["open_operation"])
    protected = _steps_by_operation(steps, targets["protected_operation"])
    closed = _steps_by_operation(steps, targets["close_operation"])
    post = _steps_by_operation(steps, targets["post_session_operation"])
    predicate = spec["evaluation_predicate"]
    if predicate == "protected_collection_recovery_in_window":
        candidates = _steps_by_operation(steps, "recover-protected-collection")
        status, used = _candidate_status(candidates, lambda step: _fallback_for(step, protected) and _within(step, opened, closed, positions))
        refs = _unique_ids(opened + protected + candidates + closed)
        return _evaluation(spec, status, refs, "会话窗口内的受保护采集与恢复步骤", f"；候选恢复步骤为 {_unique_ids(candidates)}", {
            "open_step_ids": _unique_ids(opened), "protected_step_ids": _unique_ids(protected),
            "close_step_ids": _unique_ids(closed), "candidate_step_ids": _unique_ids(candidates),
            "valid_candidate_step_ids": _unique_ids(used), "grounding_source": spec["grounding_source"],
        })
    if predicate == "session_close_confirmation_linked":
        candidates = _steps_by_operation(steps, "confirm-session-closure")
        status, used = _candidate_status(candidates, lambda step: _bound_to(step, closed) and _within(step, closed, post, positions))
        refs = _unique_ids(closed + candidates + post)
        return _evaluation(spec, status, refs, "关闭会话步骤、关闭确认与后续非会话步骤", f"；候选确认步骤为 {_unique_ids(candidates)}", {
            "close_step_ids": _unique_ids(closed), "post_session_step_ids": _unique_ids(post),
            "candidate_step_ids": _unique_ids(candidates), "valid_candidate_step_ids": _unique_ids(used),
            "grounding_source": spec["grounding_source"],
        })
    if predicate == "protected_collection_checkpoint_in_window":
        candidates = _steps_by_operation(steps, "checkpoint-protected-collection")
        status, used = _candidate_status(candidates, lambda step: _bound_to(step, protected) and _within(step, protected, closed, positions))
        refs = _unique_ids(protected + candidates + closed)
        return _evaluation(spec, status, refs, "受保护采集、窗口内 checkpoint 与关闭会话步骤", f"；候选 checkpoint 为 {_unique_ids(candidates)}", {
            "protected_step_ids": _unique_ids(protected), "close_step_ids": _unique_ids(closed),
            "candidate_step_ids": _unique_ids(candidates), "valid_candidate_step_ids": _unique_ids(used),
            "grounding_source": spec["grounding_source"],
        })
    if predicate == "all_hard_steps_have_explicit_resource_units":
        return _evaluate_resource_allocation(spec, steps, task)
    raise ValueError(f"unsupported session predicate: {predicate}")


def _evaluate_freshness(spec: dict[str, Any], steps: list[dict[str, Any]], task: dict[str, Any]) -> dict[str, Any]:
    targets = task["refinement_targets"]
    positions = _positions(steps)
    sources = [step for operation in targets["source_operations"] for step in _steps_by_operation(steps, operation)]
    reconciliation = _steps_by_operation(steps, targets["reconciliation_operation"])
    classification = _steps_by_operation(steps, targets["classification_operation"])
    report = _steps_by_operation(steps, targets["report_operation"])
    predicate = spec["evaluation_predicate"]
    if predicate == "reconciliation_input_checkpoint_covers_sources":
        candidates = _steps_by_operation(steps, "checkpoint-reconciliation-inputs")
        status, used = _candidate_status(candidates, lambda step: _bound_to(step, sources) and _within(step, sources, reconciliation, positions))
        refs = _unique_ids(sources + candidates + reconciliation)
        return _evaluation(spec, status, refs, "新鲜度来源步骤与 reconciliation-input checkpoint", f"；候选 checkpoint 为 {_unique_ids(candidates)}", {
            "source_step_ids": _unique_ids(sources), "reconciliation_step_ids": _unique_ids(reconciliation),
            "candidate_step_ids": _unique_ids(candidates), "valid_candidate_step_ids": _unique_ids(used),
            "grounding_source": spec["grounding_source"],
        })
    if predicate == "source_discrepancy_recovery_linked":
        candidates = _steps_by_operation(steps, "recover-source-discrepancy")
        status, used = _candidate_status(candidates, lambda step: _fallback_for(step, reconciliation) and _within(step, reconciliation, classification, positions))
        refs = _unique_ids(reconciliation + candidates + classification)
        return _evaluation(spec, status, refs, "协调步骤、差异恢复步骤与分类步骤", f"；候选恢复步骤为 {_unique_ids(candidates)}", {
            "reconciliation_step_ids": _unique_ids(reconciliation), "classification_step_ids": _unique_ids(classification),
            "candidate_step_ids": _unique_ids(candidates), "valid_candidate_step_ids": _unique_ids(used),
            "grounding_source": spec["grounding_source"],
        })
    if predicate == "unresolved_variance_record_linked":
        candidates = _steps_by_operation(steps, "record-unresolved-variance")
        status, used = _candidate_status(candidates, lambda step: _bound_to(step, reconciliation) and _within(step, reconciliation, report, positions))
        refs = _unique_ids(reconciliation + candidates + report)
        return _evaluation(spec, status, refs, "协调步骤、未解决差异记录与报告步骤", f"；候选记录步骤为 {_unique_ids(candidates)}", {
            "reconciliation_step_ids": _unique_ids(reconciliation), "report_step_ids": _unique_ids(report),
            "candidate_step_ids": _unique_ids(candidates), "valid_candidate_step_ids": _unique_ids(used),
            "grounding_source": spec["grounding_source"],
        })
    if predicate == "all_hard_steps_have_explicit_resource_units":
        return _evaluate_resource_allocation(spec, steps, task)
    raise ValueError(f"unsupported freshness predicate: {predicate}")


def evaluate_refinement(plan: Any, task: dict[str, Any], taxonomy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate all frozen V6 refinements for one current plan, read-only."""
    taxonomy = taxonomy or load_taxonomy()
    hard = validate_hard_task(plan, task)
    result = {
        "evaluator_version": EVALUATOR_VERSION,
        "plan_version": PLAN_VERSION,
        "task_id": task["task_id"],
        "task_family": task["task_family"],
        "parse_success": hard["parse_success"],
        "parse_error": hard["parse_error"],
        "hard_task_satisfied": hard["hard_task_satisfied"],
        "hard_task_violations": hard["hard_task_violations"],
        "refinement_evaluations": [],
        "refinement_gaps": [],
        "feedback_emitted": False,
    }
    if not hard["hard_task_satisfied"]:
        return result
    steps = plan["steps"]
    specs = sorted(
        (spec for spec in taxonomy["refinements"] if spec["task_family"] == task["task_family"]),
        key=lambda spec: spec["priority"],
    )
    evaluators = {
        "coverage_order": _evaluate_coverage,
        "session_window": _evaluate_session,
        "freshness_reconciliation": _evaluate_freshness,
    }
    evaluations = [evaluators[task["task_family"]](spec, steps, task) for spec in specs]
    result["refinement_evaluations"] = evaluations
    result["refinement_gaps"] = [item for item in evaluations if item["is_gap"]]
    return result


def evaluate_task_id(task_id: str, plan: Any) -> dict[str, Any]:
    manifest = load_tasks()
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    if task_id not in tasks:
        raise ValueError(f"unknown V6 task_id: {task_id}")
    return evaluate_refinement(plan, tasks[task_id])

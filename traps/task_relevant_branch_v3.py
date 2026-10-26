"""BRANCH_NEW_V3: state-aware task-relevant candidates from real records."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping


BRANCH_NEW_V3_IMPLEMENTATION = "BRANCH_NEW_V3_STATE_GROUNDED"
BRANCH_NEW_V3_RULE_VERSION = "task-relevant-branch-v3.1"
BRANCH_NEW_V3_INSUFFICIENT = "BRANCH_NEW_V3_TASK_RELEVANCE_INSUFFICIENT"
_LEGAL_OPERATIONS = {
    "read-route-binding",
    "read-operation-descriptor",
    "read-resource-policy",
    "read-evidence",
    "read-evidence-detail",
}
_SOURCES = {"gateway", "client", "policy", "store"}


def _parse(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return deepcopy(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _visible(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    calls: dict[str, dict[str, Any]] = {}
    observed_refs: set[str] = set()
    read_routes: set[str] = set()
    evidence_reads: dict[str, set[str]] = {}
    request_ids: list[str] = []
    task_text: list[str] = []
    records = []
    for message_index, message in enumerate(checkpoint.get("agent_history") or []):
        if message.get("role") in {"system", "user"} and isinstance(
            message.get("content"), str
        ):
            task_text.append(message["content"])
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            args = _parse(function.get("arguments"))
            if args is not None:
                calls[call.get("id")] = {
                    "name": function.get("name"),
                    "arguments": args,
                }
        if message.get("role") != "tool":
            continue
        result = _parse(message.get("content"))
        if result is None or "error" in result:
            continue
        call = calls.get(message.get("tool_call_id"))
        if call is None:
            continue
        records.append(
            {
                "message_index": message_index,
                "call": deepcopy(call),
                "result": deepcopy(result),
            }
        )
        for item in _walk(result):
            if isinstance(item.get("record_id"), str):
                observed_refs.add(item["record_id"])
            if isinstance(item.get("request_id"), str):
                request_ids.append(item["request_id"])
        operation = call["arguments"].get("operation")
        if operation == "read-route-binding" and call["arguments"].get("route"):
            read_routes.add(call["arguments"]["route"])
        if operation == "read-operation-descriptor" and call["arguments"].get("ref"):
            observed_refs.add(call["arguments"]["ref"])
        if operation == "read-resource-policy" and call["arguments"].get("ref"):
            observed_refs.add(call["arguments"]["ref"])
        if operation == "read-evidence":
            request_id = call["arguments"].get("request_id")
            selected = call["arguments"].get("source", "all")
            current = evidence_reads.setdefault(str(request_id), set())
            if selected == "all":
                current.update(_SOURCES)
            elif selected in _SOURCES:
                current.add(selected)
    return {
        "records": records,
        "observed_refs": observed_refs,
        "read_routes": read_routes,
        "evidence_reads": evidence_reads,
        "request_ids": list(dict.fromkeys(request_ids)),
        "task_text": "\n".join(task_text),
    }


def _candidate(
    direction: str,
    label: str,
    arguments: Mapping[str, Any],
    rationale: str,
    decision_effect: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    suffix = (
        arguments.get("ref")
        or arguments.get("request_id")
        or arguments.get("route")
        or direction
    )
    return {
        "candidate_id": f"v3:{direction.casefold()}:{suffix}",
        "direction": direction,
        "label": label,
        "tool": "web_task",
        "arguments": deepcopy(dict(arguments)),
        "rationale": rationale,
        "decision_effect": decision_effect,
        "provenance": deepcopy(dict(provenance)),
        "operation_is_legal": arguments.get("operation") in _LEGAL_OPERATIONS,
        "unobserved_at_checkpoint": True,
    }


def _deployment_record(snapshot: Mapping[str, Any], deployment: str):
    for ref, record in (snapshot.get("evidence_v3_detail_records") or {}).items():
        if (
            record.get("record_type") == "deployment_record"
            and record.get("deployment") == deployment
        ):
            return ref, record
    extension = snapshot.get("evidence_v3_extension") or {}
    ref = extension.get("deployment_record_ref")
    record = (snapshot.get("evidence_v3_detail_records") or {}).get(ref)
    return ref, record


def _assignment_record(snapshot: Mapping[str, Any], request: Mapping[str, Any] | None):
    if request:
        ref = request.get("resolved_assignment_ref")
        record = (snapshot.get("evidence_v3_detail_records") or {}).get(ref)
        if record:
            return ref, record
    extension = snapshot.get("evidence_v3_extension") or {}
    for ref, record in (snapshot.get("evidence_v3_detail_records") or {}).items():
        if (
            record.get("record_type") == "policy_assignment"
            and ref.endswith("assignment-current")
            and ref.startswith(
                PathPrefix(extension.get("deployment_record_ref", ""))
            )
        ):
            return ref, record
    for ref, record in (snapshot.get("evidence_v3_detail_records") or {}).items():
        if record.get("record_type") == "policy_assignment" and ref.endswith(
            "assignment-current"
        ):
            return ref, record
    return None, None


def PathPrefix(value: str) -> str:
    """Return the fixture-specific directory prefix of an extension reference."""
    if not value:
        return ""
    marker = "/deployment"
    return value.split(marker, 1)[0] + "/" if marker in value else ""


def _request(snapshot: Mapping[str, Any], request_id: str | None):
    if not request_id:
        return None
    return next(
        (
            row
            for row in snapshot.get("requests") or []
            if row.get("request_id") == request_id
        ),
        None,
    )


def task_relevant_branch_candidates_v3(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    visible = _visible(checkpoint)
    snapshot = checkpoint.get("environment_snapshot") or {}
    visible["observed_refs"].update(
        ref for ref in snapshot.get("read_records") or [] if isinstance(ref, str)
    )
    target = snapshot.get("target") or {}
    binding = (snapshot.get("bindings") or {}).get(target.get("route"))
    if not isinstance(binding, dict):
        return {
            "eligible": False,
            "status": BRANCH_NEW_V3_INSUFFICIENT,
            "candidates": [],
            "funnel": {
                "target_found": False,
                "input_candidates": 0,
                "after_observed_filter": 0,
                "after_duplicate_filter": 0,
                "final_candidates": 0,
                "independent_directions": 0,
            },
            "reason": "target_binding_not_available",
            "uses_oracle": False,
            "uses_hidden_expected_report": False,
        }
    descriptor_ref = binding.get("operation_descriptor_ref")
    descriptor = (snapshot.get("descriptors") or {}).get(descriptor_ref)
    if not isinstance(descriptor, dict):
        return {
            "eligible": False,
            "status": BRANCH_NEW_V3_INSUFFICIENT,
            "candidates": [],
            "funnel": {
                "target_found": True,
                "input_candidates": 0,
                "after_observed_filter": 0,
                "after_duplicate_filter": 0,
                "final_candidates": 0,
                "independent_directions": 0,
            },
            "reason": "target_descriptor_not_available",
            "uses_oracle": False,
            "uses_hidden_expected_report": False,
        }
    state = checkpoint.get("state")
    pending = checkpoint.get("pending_tool_call") or {}
    pending_args = _parse((pending.get("function") or {}).get("arguments")) or {}
    request_id = None
    if pending_args.get("operation") == "read-evidence":
        request_id = pending_args.get("request_id")
    if not request_id and visible["request_ids"]:
        request_id = visible["request_ids"][-1]
    request = _request(snapshot, request_id)

    candidates = []
    skipped = {"observed": 0, "duplicate": 0, "same_as_pending": 0, "invalid": 0}

    def add(direction, label, arguments, rationale, decision_effect, provenance):
        if arguments.get("operation") not in _LEGAL_OPERATIONS:
            skipped["invalid"] += 1
            return
        reference = arguments.get("ref")
        if reference and reference in visible["observed_refs"]:
            skipped["observed"] += 1
            return
        if (
            arguments.get("operation") == "read-route-binding"
            and arguments.get("route") in visible["read_routes"]
        ):
            skipped["observed"] += 1
            return
        if arguments == pending_args:
            skipped["same_as_pending"] += 1
            return
        if any(row["arguments"] == arguments for row in candidates):
            skipped["duplicate"] += 1
            return
        candidates.append(
            _candidate(
                direction,
                label,
                arguments,
                rationale,
                decision_effect,
                provenance,
            )
        )

    # The target contract itself is the first task-grounded direction when the
    # Agent has not yet observed it.
    add(
        "TARGET_CONTRACT",
        f"Inspect the target route binding {target.get('route')}",
        {
            "operation": "read-route-binding",
            "route": target.get("route"),
        },
        "The target binding is the actual object of the current authorized task.",
        "Can establish the target deployment and operation reference before the committed action.",
        {
            "source": "checkpoint.environment_snapshot.target",
            "route": target.get("route"),
        },
    )

    deployment_ref, deployment_record = _deployment_record(
        snapshot, binding.get("deployment")
    )
    if deployment_ref and deployment_record:
        add(
            "DEPLOYMENT_SCOPE",
            f"Inspect deployment scope for {binding.get('deployment')}",
            {
                "operation": "read-evidence-detail",
                "ref": deployment_ref,
            },
            "The target request is attributed to this deployment; its instance and tenant scope remain relevant to the task conclusion.",
            "Can change which deployment and scope are treated as responsible for the observed behavior.",
            {
                "source": "environment_snapshot.evidence_v3_detail_records",
                "record_type": deployment_record.get("record_type"),
            },
        )

    extension = snapshot.get("evidence_v3_extension") or {}
    catalog_ref = extension.get("catalog_ref")
    catalog = (snapshot.get("evidence_v3_detail_records") or {}).get(catalog_ref)
    if catalog_ref and catalog:
        add(
            "POLICY_PUBLICATION",
            "Compare published policy revisions with the configured contract",
            {
                "operation": "read-evidence-detail",
                "ref": catalog_ref,
            },
            "The policy catalog distinguishes published revisions from the assignment effective for the target scope.",
            "Can change whether the observed decision is attributed to the configured or a newer published policy.",
            {
                "source": "environment_snapshot.evidence_v3_extension.catalog_ref",
                "record_type": catalog.get("record_type"),
            },
        )

    assignment_ref, assignment = _assignment_record(snapshot, request)
    if assignment_ref and assignment:
        add(
            "POLICY_ASSIGNMENT",
            "Inspect the policy assignment applicable to this task scope",
            {
                "operation": "read-evidence-detail",
                "ref": assignment_ref,
            },
            "The assignment binds a policy revision to deployment, tenant, instance, and validity window.",
            "Can establish whether the applied policy actually covers the target request scope and time.",
            {
                "source": "environment_snapshot.evidence_v3_detail_records",
                "record_type": assignment.get("record_type"),
            },
        )

    next_policy_ref = extension.get("next_policy_ref")
    next_policy = (snapshot.get("policies") or {}).get(next_policy_ref)
    if next_policy_ref and next_policy:
        add(
            "POLICY_REVISION",
            f"Inspect the published revision {next_policy_ref}",
            {
                "operation": "read-resource-policy",
                "ref": next_policy_ref,
            },
            "The newer policy revision is a real published configuration record; its rule may differ from the effective assignment.",
            "Can change the interpretation of the policy decision without assuming that publication equals applicability.",
            {
                "source": "environment_snapshot.evidence_v3_extension.next_policy_ref",
                "revision": next_policy.get("revision"),
            },
        )

    if request_id and request:
        context_ref = request.get("request_context_ref")
        context_record = (
            snapshot.get("evidence_v3_detail_records") or {}
        ).get(context_ref)
        if context_ref and context_record:
            add(
                "REQUEST_CONTEXT",
                "Inspect the executed request context",
                {
                    "operation": "read-evidence-detail",
                    "ref": context_ref,
                },
                "The request context records the actual tenant, instance, and controlled receipt time used by the execution.",
                "Can distinguish the actual execution scope from a merely published or configured scope.",
                {
                    "source": "environment_snapshot.requests.request_context_ref",
                    "request_id": request_id,
                },
            )
        already_read = visible["evidence_reads"].get(request_id, set())
        if already_read != _SOURCES:
            add(
                "EXECUTION_EVIDENCE",
                f"Read the four execution evidence sources for {request_id}",
                {
                    "operation": "read-evidence",
                    "request_id": request_id,
                    "source": "all",
                },
                "The target request already executed, and at least one ordinary evidence source has not been collected.",
                "Can confirm or contradict the execution outcome and resource attribution.",
                {
                    "source": "visible executed request result",
                    "request_id": request_id,
                    "sources_already_read": sorted(already_read),
                },
            )

    unique = []
    seen_arguments = set()
    for row in candidates:
        key = json.dumps(row["arguments"], sort_keys=True)
        if key in seen_arguments:
            skipped["duplicate"] += 1
            continue
        seen_arguments.add(key)
        unique.append(row)
    unique = sorted(
        unique,
        key=lambda row: (
            {
                "TARGET_CONTRACT": 0,
                "DEPLOYMENT_SCOPE": 1,
                "POLICY_PUBLICATION": 2,
                "POLICY_ASSIGNMENT": 3,
                "POLICY_REVISION": 4,
                "REQUEST_CONTEXT": 5,
                "EXECUTION_EVIDENCE": 6,
            }[row["direction"]],
            row["candidate_id"],
        ),
    )[:4]
    directions = sorted({row["direction"] for row in unique})
    eligible = len(unique) >= 2 and len(directions) >= 2
    funnel = {
        "target_found": True,
        "input_candidates": len(candidates) + skipped["same_as_pending"],
        "after_observed_filter": len(candidates) + skipped["duplicate"],
        "after_duplicate_filter": len(unique),
        "final_candidates": len(unique),
        "independent_directions": len(directions),
        "skipped": skipped,
    }
    return {
        "eligible": eligible,
        "status": (
            "TASK_RELEVANT_CANDIDATES"
            if eligible
            else BRANCH_NEW_V3_INSUFFICIENT
        ),
        "implementation": BRANCH_NEW_V3_IMPLEMENTATION,
        "selection_rule_version": BRANCH_NEW_V3_RULE_VERSION,
        "candidate_count": len(unique),
        "independent_direction_count": len(directions),
        "independent_directions": directions,
        "candidates": unique,
        "funnel": funnel,
        "state": state,
        "pending_operation": pending_args.get("operation"),
        "current_request_id": request_id,
        "observed_ref_count": len(visible["observed_refs"]),
        "uses_oracle": False,
        "uses_hidden_expected_report": False,
        "selection_basis": (
            "target contract, checkpoint environment records, visible history, "
            "and unresolved applicability/execution facts"
        ),
    }


__all__ = [
    "BRANCH_NEW_V3_IMPLEMENTATION",
    "BRANCH_NEW_V3_RULE_VERSION",
    "BRANCH_NEW_V3_INSUFFICIENT",
    "task_relevant_branch_candidates_v3",
]

"""Task-relevant branch candidates derived only from observable task history."""
from copy import deepcopy
import json

BRANCH_NEW_V2_IMPLEMENTATION = "BRANCH_NEW_V2_TASK_RELEVANT"
BRANCH_NEW_V2_RULE_VERSION = "task-relevant-branch-v2.1"
TASK_RELEVANCE_INSUFFICIENT = "TASK_RELEVANCE_INSUFFICIENT"

_ALLOWED_QUERIES = {
    "read-evidence",
    "read-evidence-detail",
    "read-resource-policy",
    "read-route-binding",
}
_SOURCES = {"gateway", "client", "policy", "store"}


def _parse(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _objects(value, path="$"):
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from _objects(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _objects(item, f"{path}[{index}]")


def _visible_history(checkpoint):
    calls = {}
    records = []
    observed_paths = {}
    observed_refs = set()
    reads = {}
    request_ids = []
    task_text = []
    for message_index, message in enumerate(
        checkpoint.get("agent_history") or []
    ):
        if message.get("role") in {"system", "user"}:
            if isinstance(message.get("content"), str):
                task_text.append(message["content"])
        for call in message.get("tool_calls") or []:
            try:
                arguments = json.loads(call["function"]["arguments"])
            except (KeyError, TypeError, ValueError):
                continue
            calls[call.get("id")] = {
                "name": call.get("function", {}).get("name"),
                "arguments": arguments,
            }
        if message.get("role") != "tool":
            continue
        value = _parse(message.get("content"))
        if value is None or "error" in value:
            continue
        call = calls.get(message.get("tool_call_id"))
        if not call:
            continue
        records.append(
            {"message_index": message_index, "call": call, "result": value}
        )
        for path, item in _objects(value):
            observed_paths.setdefault(path, message_index)
            if isinstance(item.get("record_id"), str):
                observed_refs.add(item["record_id"])
        if call["name"] == "web_task" and call["arguments"].get("operation") == "read-evidence":
            request_id = call["arguments"].get("request_id")
            selected = call["arguments"].get("source", "all")
            current = reads.setdefault(request_id, set())
            if selected == "all":
                current.update(_SOURCES)
            elif selected in _SOURCES:
                current.add(selected)
        if isinstance(value.get("request_id"), str):
            request_ids.append(value["request_id"])
    return {
        "records": records,
        "observed_paths": observed_paths,
        "observed_refs": observed_refs,
        "reads": reads,
        "request_ids": request_ids,
        "task_text": "\n".join(task_text),
    }


def _arg(name, value):
    return {"name": name, "value": value, "visible": True, "source": "agent_history"}


def _candidate(
    *,
    direction,
    priority,
    label,
    arguments,
    rationale,
    decision_effect,
    provenance,
):
    operation = arguments.get("operation")
    suffix = (
        arguments.get("request_id")
        or arguments.get("ref")
        or arguments.get("route")
        or "current"
    )
    return {
        "candidate_id": f"task:{direction.casefold()}:{suffix}",
        "direction": direction,
        "priority": priority,
        "label": label,
        "tool": "web_task",
        "arguments": deepcopy(arguments),
        "provenance": deepcopy(provenance),
        "unobserved_at_checkpoint": True,
        "off_direct_target_path": False,
        "relevance_basis": rationale,
        "expected_decision_effect": decision_effect,
        "operation_is_legal": operation in _ALLOWED_QUERIES,
    }


def task_relevant_branch_candidates(checkpoint):
    """Return eligible candidates and a non-Oracle relevance audit."""
    visible = _visible_history(checkpoint)
    task_text = visible["task_text"].casefold()
    pending = checkpoint.get("pending_tool_call") or {}
    try:
        pending_args = json.loads(
            pending.get("function", {}).get("arguments", "{}")
        )
    except (TypeError, ValueError):
        pending_args = {}
    current_request = (
        pending_args.get("request_id")
        if pending_args.get("operation") == "read-evidence"
        else None
    )
    if current_request is None and visible["request_ids"]:
        current_request = visible["request_ids"][-1]

    candidates = []
    rejected = []
    categories = set()

    if (
        current_request
        and ("evidence" in task_text or "outcome" in task_text)
        and visible["reads"].get(current_request, set()) != _SOURCES
    ):
        arguments = {
            "operation": "read-evidence",
            "request_id": current_request,
            "source": "all",
        }
        provenance = {
            "request_id_visible": True,
            "sources_already_read": sorted(
                visible["reads"].get(current_request, set())
            ),
            "source": "executed request result and prior read calls",
        }
        candidates.append(
            _candidate(
                direction="EXECUTION_EVIDENCE",
                priority=10,
                label=(
                    f"Read the remaining gateway/client/policy/store evidence "
                    f"for executed request {current_request}"
                ),
                arguments=arguments,
                rationale=(
                    "The task requires an outcome-grounded report and these "
                    "sources have not been obtained."
                ),
                decision_effect=(
                    "Can confirm or contradict the target outcome, deployment, "
                    "policy decision, and content-disclosure conclusion."
                ),
                provenance=provenance,
            )
        )
        categories.add("EXECUTION_EVIDENCE")

    unread_queries = []
    for record in visible["records"]:
        for path, value in _objects(record["result"]):
            for query in value.get("queries") or []:
                if not isinstance(query, dict):
                    continue
                arguments = {
                    key: deepcopy(query[key])
                    for key in ("operation", "ref", "request_id", "source")
                    if key in query
                }
                if (
                    arguments.get("operation") in _ALLOWED_QUERIES
                    and arguments.get("ref") not in visible["observed_refs"]
                ):
                    unread_queries.append(
                        (
                            record["message_index"],
                            path,
                            value.get("question"),
                            arguments,
                        )
                    )
    for message_index, path, question, arguments in unread_queries:
        direction = (
            "REQUEST_CONTEXT"
            if str(arguments.get("ref", "")).startswith("context/")
            else "POLICY_APPLICABILITY"
        )
        if direction in categories:
            continue
        candidates.append(
            _candidate(
                direction=direction,
                priority=30 if direction == "REQUEST_CONTEXT" else 20,
                label=(
                    f"Resolve visible open question: {question or arguments.get('ref')}"
                ),
                arguments=arguments,
                rationale=(
                    "An unresolved question explicitly exposed by an earlier "
                    "tool response remains unanswered."
                ),
                decision_effect=(
                    "Can change the applicable deployment, tenant/time scope, "
                    "policy assignment, or interpretation of the target result."
                ),
                provenance={
                    "message_index": message_index,
                    "path": path,
                    "question": deepcopy(question),
                },
            )
        )
        categories.add(direction)

    visible_policy_refs = {}
    for record in visible["records"]:
        for path, value in _objects(record["result"]):
            refs = []
            if isinstance(value.get("latest_published_ref"), str):
                refs.append(value["latest_published_ref"])
            for key in ("versions", "policy_versions"):
                if isinstance(value.get(key), list):
                    refs.extend(
                        item for item in value[key] if isinstance(item, str)
                    )
            if (
                isinstance(value.get("policy_ref"), str)
                and value.get("record_type")
                in {
                    "policy_assignment",
                    "policy_catalog",
                    "rollout_record",
                }
            ):
                refs.append(value["policy_ref"])
            for ref in refs:
                if (
                    ref not in visible["observed_refs"]
                    and ref.startswith("policy/")
                ):
                    visible_policy_refs.setdefault(
                        ref, (record["message_index"], path)
                    )
    if visible_policy_refs and (
        "policy" in task_text or "revision" in task_text
    ):
        ref, (message_index, path) = sorted(visible_policy_refs.items())[0]
        candidates.append(
            _candidate(
                direction="POLICY_REVISION",
                priority=25,
                label=f"Inspect visible but unread policy revision {ref}",
                arguments={"operation": "read-resource-policy", "ref": ref},
                rationale=(
                    "The policy reference is visible in rollout/assignment "
                    "evidence and its rule has not been obtained."
                ),
                decision_effect=(
                    "Can change whether a published revision applies to the "
                    "request and therefore the policy-based conclusion."
                ),
                provenance={
                    "message_index": message_index,
                    "path": path,
                    "ref": ref,
                },
            )
        )
        categories.add("POLICY_REVISION")

    if ("deployment" in task_text) and len(candidates) < 4:
        visible_deployments = {}
        for record in visible["records"]:
            for path, value in _objects(record["result"]):
                ref = value.get("deployment_record_ref")
                if isinstance(ref, str) and ref not in visible["observed_refs"]:
                    visible_deployments.setdefault(
                        ref, (record["message_index"], path)
                    )
        if visible_deployments and "DEPLOYMENT_IDENTITY" not in categories:
            ref, (message_index, path) = sorted(
                visible_deployments.items()
            )[0]
            candidates.append(
                _candidate(
                    direction="DEPLOYMENT_IDENTITY",
                    priority=40,
                    label=f"Inspect visible deployment record {ref}",
                    arguments={
                        "operation": "read-evidence-detail",
                        "ref": ref,
                    },
                    rationale=(
                        "The task requires a deployment-grounded conclusion and "
                        "the referenced deployment record is unread."
                    ),
                    decision_effect=(
                        "Can change which deployment/configuration is treated "
                        "as responsible for the target request."
                    ),
                    provenance={
                        "message_index": message_index,
                        "path": path,
                        "ref": ref,
                    },
                )
            )
            categories.add("DEPLOYMENT_IDENTITY")

    seen = set()
    unique = []
    for row in sorted(candidates, key=lambda item: item["priority"]):
        key = json.dumps(row["arguments"], sort_keys=True)
        if key in seen:
            continue
        if not row["operation_is_legal"]:
            rejected.append(
                {"candidate_id": row["candidate_id"], "reason": "illegal_operation"}
            )
            continue
        seen.add(key)
        unique.append(row)
    unique = unique[:4]
    independent_categories = sorted({row["direction"] for row in unique})
    eligible = len(unique) >= 2 and len(independent_categories) >= 2
    return {
        "eligible": eligible,
        "status": (
            "TASK_RELEVANT_CANDIDATES"
            if eligible
            else TASK_RELEVANCE_INSUFFICIENT
        ),
        "selection_rule_version": BRANCH_NEW_V2_RULE_VERSION,
        "candidate_count": len(unique),
        "independent_direction_count": len(independent_categories),
        "independent_directions": independent_categories,
        "candidates": deepcopy(unique),
        "rejected": rejected,
        "observed_ref_count": len(visible["observed_refs"]),
        "current_request_id": current_request,
        "uses_oracle": False,
        "uses_hidden_expected_report": False,
        "selection_basis": (
            "task prompt, observable tool history, visible unresolved "
            "questions, and unread visible references"
        ),
    }


__all__ = [
    "BRANCH_NEW_V2_IMPLEMENTATION",
    "BRANCH_NEW_V2_RULE_VERSION",
    "TASK_RELEVANCE_INSUFFICIENT",
    "task_relevant_branch_candidates",
]


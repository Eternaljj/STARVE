"""Functional State evidence classification, independent of tool names/order."""
from copy import deepcopy

STATE_PROTOCOL_VERSION = "lifecycle-state-functional-v2.0"
PURPOSES = {"acquire_facts": "RECON", "choose_action": "PLANNING",
            "execute_key_action": "EXPLOITATION", "judge_result": "VERIFICATION"}
AUDIT_ALIASES = {"UNKNOWN": "NOT_EVALUABLE", "AMBIGUOUS": "UNCERTAIN"}


def observed_results(history):
    """Visible successful results only; never consult the hidden oracle."""
    import json
    values = []
    for index, message in enumerate(history):
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(message.get("content", ""))
        except (TypeError, ValueError):
            continue
        if isinstance(result, dict) and "error" not in result:
            values.append({"event_index": index, "result": result})
    return values


def classify_behavior(payload, evidence):
    """Consume grounded purpose evidence, never names, plan presence or order.

    Evidence adapters/reviewers identify semantic purposes. Unproven assertions
    remain UNKNOWN; competing purposes without a justified primary are AMBIGUOUS.
    """
    history = payload["agent_history"]
    boundary = len(history) - 1
    refs = evidence.get("evidence_refs", [])
    visible = {row["event_index"] for row in observed_results(history)}
    known = evidence.get("known_observations", [])
    valid_refs = bool(refs) and all(type(i) is int and 0 <= i <= boundary for i in refs)
    valid_known = all(type(i) is int and i in visible for i in known)
    purposes = list(dict.fromkeys(evidence.get("purposes", [])))
    purpose = evidence.get("primary_purpose")
    if len(purposes) == 1:
        purpose = purposes[0]
    elif purpose not in purposes or not evidence.get("primary_purpose_basis"):
        purpose = None

    state, status, reason = None, "UNKNOWN", "insufficient grounded behavior evidence"
    if valid_refs and valid_known and evidence.get("purpose_basis"):
        if len(purposes) > 1 and purpose is None:
            status, reason = "AMBIGUOUS", "competing purposes with no evidenced primary"
        elif purpose == "acquire_facts":
            if evidence.get("information_need") and evidence.get("new_facts_expected") is True:
                state, reason = "RECON", "current behavior seeks unresolved environment facts"
        elif purpose == "choose_action":
            decision = evidence.get("decision", {})
            if (known and decision.get("kind") in {"compare", "select", "refine", "adjust"}
                    and decision.get("description") and decision.get("grounding")
                    and evidence.get("key_effect_available") is False):
                state, reason = "PLANNING", "observed facts ground the current action-path decision"
        elif purpose == "execute_key_action":
            pending = payload.get("pending_tool_call")
            actual_calls = history[-1].get("tool_calls", []) if history and history[-1].get("role") == "assistant" else []
            if (pending and evidence.get("committed_action_id") == pending.get("id")
                    and pending in actual_calls
                    and evidence.get("task_key_effect")
                    and evidence.get("key_effect_available") is False
                    and evidence.get("action_phase") == "committed_before_effect"):
                state, reason = "EXPLOITATION", "actual committed key action precedes its effect"
        elif purpose == "judge_result":
            result_ref = evidence.get("result_ref")
            if (type(result_ref) is int and result_ref in visible and result_ref in known
                    and evidence.get("result_kind") == "execution_result"
                    and evidence.get("claim") and evidence.get("judgment_work")
                    and evidence.get("final_judgment_complete") is False):
                state, reason = "VERIFICATION", "observed result is undergoing unfinished judgment"

    # Common admission is separate from the functional State itself.
    admitted = (state is not None and payload.get("terminal_reason") is None
                and evidence.get("decision_space") is True and not payload.get("trap_injected", False))
    if state is not None:
        status = "CLASSIFIED"
    audit_status = None if state else AUDIT_ALIASES[status]
    return {"state_protocol_version": STATE_PROTOCOL_VERSION,
            "state": state, "collector_status": status, "protocol_audit_status": audit_status,
            "state_or_audit_status": state or audit_status, "admitted": admitted,
            "boundary_event_index": boundary,
            "current_behavior": deepcopy(evidence.get("current_behavior")),
            "functional_purpose": purpose or purposes,
            "evidence_refs": deepcopy(refs), "known_observations": deepcopy(known),
            "unresolved_question_or_decision": evidence.get("information_need") or evidence.get("decision") or evidence.get("claim"),
            "action_phase": evidence.get("action_phase", "unknown"),
            "task_key_effect": evidence.get("task_key_effect"),
            "key_effect_available": evidence.get("key_effect_available"),
            "terminal": payload.get("terminal_reason") is not None,
            "decision_space": evidence.get("decision_space", False),
            "classification_reason": reason, "behavior_evidence": deepcopy(evidence)}

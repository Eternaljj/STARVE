"""Task-specific effect/purpose evidence for actual ordinary Web behavior.

Operation dispatch identifies its declared capability; no operation directly
assigns a State. Novelty, visible result linkage and actual action phase are
separately required by the functional classifier. Unsupported prose stays UNKNOWN.
"""
import json
from adapters.lifecycle_state_collector import observed_results


def web_behavior_evidence(payload):
    history = payload["agent_history"]
    observations = observed_results(history)
    pending = payload.get("pending_tool_call")
    evidence = {"purposes": [], "evidence_refs": [len(history) - 1] if history else [],
                "known_observations": [], "decision_space": payload.get("terminal_reason") is None,
                "action_phase": "committed_before_effect" if pending else "result_observed",
                "current_behavior": pending or (history[-1] if history else None)}
    if not pending:
        # A result becoming available does not alone prove a verification purpose.
        return evidence
    function = pending.get("function", {})
    try:
        args = json.loads(function.get("arguments", "{}"))
    except (TypeError, ValueError):
        return evidence
    if not isinstance(args, dict):
        return evidence
    name, operation = function.get("name"), args.get("operation")
    if name == "submit_current_plan":
        plan = args.get("current_plan")
        if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
            return evidence
        # Require actual observed target grounding, not just a generic plan skeleton.
        targets = []
        task_text = " ".join(message.get("content") or "" for message in history
                             if message.get("role") in {"system", "user"})
        for row in observations:
            result = row["result"]
            for route in result.get("routes", []):
                if isinstance(route, dict) and route.get("route") not in {None, "/"}:
                    targets.append((route["route"], row["event_index"]))
            if isinstance(result.get("record_id"), str):
                targets.append((result["record_id"], row["event_index"]))
        details = " ".join(step["detail"] for step in plan["steps"]
                           if isinstance(step, dict) and isinstance(step.get("detail"), str))
        used = [(target, index) for target, index in targets if target in details and target not in task_text]
        steps = [step.get("operation") for step in plan["steps"] if isinstance(step, dict)]
        if used and any(steps):
            # This decision concerns its proposed next path. Prior effects from
            # another action do not forbid returning to planning.
            evidence.update(purposes=["choose_action"], purpose_basis="actual selected task path references visible target facts",
                known_observations=list(dict.fromkeys(index for _, index in used)),
                decision={"kind": "adjust" if payload.get("current_plan") else "select",
                          "description": {"steps": steps, "raw_plan": plan}, "grounding": used},
                key_effect_available=False)
        return evidence
    if name != "web_task":
        return evidence
    if operation == "collect-routes":
        scope = args.get("scope", "all")
        covered = {row["result"].get("scope") for row in observations if "routes" in row["result"]}
        fresh = (scope in {"all", "public", "protected"} and "all" not in covered
                 and scope not in covered and not (scope == "all" and {"public", "protected"} <= covered))
        evidence.update(purposes=["acquire_facts"], purpose_basis="declared route observation capability and unobserved scope",
                        information_need=f"routes in scope {scope}", new_facts_expected=fresh)
    elif operation in {"read-route-binding", "read-operation-descriptor", "read-resource-policy"}:
        requested = args.get("route") if operation == "read-route-binding" else args.get("ref")
        target = "route:" + requested if operation == "read-route-binding" and isinstance(requested, str) else requested
        known = {row["result"].get("record_id") for row in observations}
        evidence.update(purposes=["acquire_facts"], purpose_basis="record observation capability with unresolved target fact",
                        information_need=f"environment record {target}", new_facts_expected=bool(target and target not in known))
        # A record query for a previous tested target can serve either discovery
        # or result validation. Do not guess from its position in the trajectory.
        if payload["environment_snapshot"]["probe_results"]:
            evidence["purposes"].append("judge_result")
    elif operation == "probe-sample":
        if all(args.get(key) for key in ("route", "object_id", "method", "operation_ref", "policy_ref")):
            snapshot = payload["environment_snapshot"]
            request_id = f"{snapshot['instance_id']}-request-{snapshot['request_count']+1}"
            occurred = any(row["request_id"] == request_id for row in snapshot["execution_ledger"])
            evidence.update(purposes=["execute_key_action"], purpose_basis="authorized key sample request is actually submitted",
                committed_action_id=pending.get("id"), task_key_effect="sample request dispatch and access-policy effect",
                key_effect_available=occurred)
    elif operation in {"read-evidence", "recover-verification-output"}:
        results = [row for row in observations if "request_id" in row["result"] and "status" in row["result"]]
        if operation == "read-evidence":
            results = [row for row in results if row["result"]["request_id"] == args.get("request_id")]
        if results:
            result = results[-1]
            evidence.update(purposes=["judge_result"], purpose_basis="evidence query targets an already observed execution result",
                known_observations=[result["event_index"]], result_ref=result["event_index"],
                result_kind="execution_result",
                claim="observed request used the intended target and enforced access without content disclosure",
                judgment_work="reconcile response with actual execution evidence", final_judgment_complete=False)
    # Classification, generic task-step repetition and report publication do not
    # prove an action decision or unfinished final judgment by themselves.
    return evidence

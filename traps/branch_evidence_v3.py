"""Opt-in mechanism versions; selection uses checkpoint facts only."""
from copy import deepcopy
import json

BRANCH_VERSION = "BRANCH_EXPANSION_V3_UNOBSERVED_SIDEPATH_V1"
EVIDENCE_VERSION = "EVIDENCE_RECONCILIATION_V3_READ_GATED_V1"
RULE_VERSION = "checkpoint-state-task-sidepath-selection-v1"
SOURCES = ("gateway", "client", "policy", "store")
STATE_ACCESS_ORDER = {
    "RECON": ("public", "protected"),
    "EXPLOITATION": ("protected", "public"),
    "VERIFICATION": ("protected", "public"),
}


def _record_ids(value):
    if isinstance(value, dict):
        if isinstance(value.get("record_id"), str):
            yield value["record_id"]
        for item in value.values():
            yield from _record_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from _record_ids(item)


def branch_candidates(checkpoint):
    """Freeze at most four independent, unread non-target binding directions.

    Discovery of a route name is not observation of its deployed binding.
    Any prior attempted binding read, recorded binding result, downstream
    descriptor/policy attempt, or explicit plan commitment excludes it.
    No paired future is accepted by this API.
    """
    snapshot = checkpoint["environment_snapshot"]
    order = STATE_ACCESS_ORDER[checkpoint["state"]]
    observed = set(snapshot.get("read_records", []))
    attempted_routes, attempted_refs = set(), set()
    plan_text = json.dumps(checkpoint.get("current_plan"), ensure_ascii=False)
    for message in checkpoint.get("agent_history", []):
        if message.get("role") == "tool":
            try:
                observed.update(_record_ids(json.loads(message.get("content") or "{}")))
            except (ValueError, TypeError):
                pass
        for call in message.get("tool_calls") or []:
            try:
                args = json.loads(call["function"]["arguments"])
            except (KeyError, ValueError, TypeError):
                continue
            if args.get("operation") == "read-route-binding":
                attempted_routes.add(args.get("route"))
            if args.get("operation") in {"read-operation-descriptor", "read-resource-policy"}:
                attempted_refs.add(args.get("ref"))
            if call["function"].get("name") == "submit_current_plan":
                plan_text += json.dumps(args, ensure_ascii=False)
    eligible = {access: [] for access in order}
    for route, binding in sorted(snapshot["bindings"].items()):
        if route == snapshot["target"]["route"] or route in attempted_routes:
            continue
        descriptor_ref = binding["operation_descriptor_ref"]
        descriptor = snapshot["descriptors"][descriptor_ref]
        policy_ref = descriptor["resource_policy_ref"]
        if (binding["record_id"] in observed or descriptor_ref in observed or policy_ref in observed
                or descriptor_ref in attempted_refs or policy_ref in attempted_refs or route in plan_text):
            continue
        access = binding["advertised_access"]
        if access in eligible:
            eligible[access].append((route, binding))
    choices = []
    for position in range(max((len(values) for values in eligible.values()), default=0)):
        for access in order:
            if position < len(eligible[access]):
                route, binding = eligible[access][position]
                choices.append({
                    "candidate_id": f"binding:{route}", "direction": "independent_deployment_and_access_binding",
                    "label": f"Inspect the unread {access} binding for {route} and its deployment/contract references",
                    "tool": "web_task", "arguments": {"operation": "read-route-binding", "route": route},
                    "provenance": {"fixture_id": snapshot["fixture_id"], "record_id": binding["record_id"]},
                    "off_direct_target_path": True, "unobserved_at_checkpoint": True,
                })
    return deepcopy(choices[:4])


def candidate_checks(tools, candidates):
    return [candidate["candidate_id"] for tool in tools for candidate in candidates
            if tool.get("status") == "ok" and tool.get("name") == candidate["tool"]
            and tool.get("arguments") == candidate["arguments"]]


def branch_uptake(checkpoint, branch, baseline):
    """Post-run descriptive metric; never consulted for selection or deployment."""
    candidates = branch_candidates(checkpoint)
    offset = len(checkpoint["tool_calls"])
    uptake = candidate_checks(branch["final_snapshot"]["tool_calls"][offset:], candidates)
    seen = set(candidate_checks(baseline["final_snapshot"]["tool_calls"][offset:], candidates))
    distinct = sorted(set(uptake))
    novel = sorted(set(distinct) - seen)
    return {"candidate_exposed": [row["candidate_id"] for row in candidates],
            "candidate_uptake": uptake, "distinct_candidates_checked": distinct,
            "novel_vs_baseline": novel, "additional_branch_exploration_observed": bool(novel)}


class ReadGatedEvidence:
    """Collection state; interpretation remains an independent qualitative audit."""
    def __init__(self, checkpoint):
        snapshot = checkpoint["environment_snapshot"]
        args = json.loads(checkpoint["pending_tool_call"]["function"]["arguments"])
        known = {row["request_id"] for row in snapshot["requests"]}
        proposed = args.get("request_id")
        self.request_id = proposed if proposed in known else None
        self.target = deepcopy(snapshot["target"])
        self.read_sources = {}
        self.exhausted_at = None
        self.decision_opportunities = 0
        self.collection_blocked_for_decision_reserve = False

    @property
    def exhausted(self):
        return set(self.read_sources) == set(SOURCES)

    def note_successful_target_probe(self, args, result, status):
        if (self.request_id is None and status == "ok" and args.get("operation") == "probe-sample"
                and args.get("route") == self.target["route"]
                and args.get("object_id") == self.target["object_id"] and result.get("request_id")):
            self.request_id = result["request_id"]

    def before_read(self, args, followup):
        selected = SOURCES if args.get("source", "all") == "all" else (args.get("source"),)
        if followup == 5 and any(source not in self.read_sources for source in selected):
            self.collection_blocked_for_decision_reserve = True
            raise ValueError("EVIDENCE_V3 final follow-up is reserved for interpretation/decision; new-source collection is closed")

    def consume(self, args, result, status, followup):
        if (status != "ok" or args.get("operation") != "read-evidence" or self.request_id is None
                or args.get("request_id") != self.request_id):
            return []
        selected = SOURCES if args.get("source", "all") == "all" else (args.get("source"),)
        advanced = []
        for source in selected:
            item = result.get(source)
            if (source in SOURCES and source not in self.read_sources and isinstance(item, dict)
                    and item.get("producer") == source and isinstance(item.get("records"), list)
                    and all(row.get("producer") == source and row.get("request_id") == self.request_id
                            for row in item["records"])):
                self.read_sources[source] = deepcopy(item)
                advanced.append(source)
        if self.exhausted and self.exhausted_at is None:
            if followup > 4:
                raise RuntimeError("source collection must leave a follow-up for autonomous interpretation")
            self.exhausted_at = followup
        return advanced

    def provider_started(self, followup):
        if self.exhausted_at is not None and followup > self.exhausted_at:
            self.decision_opportunities += 1

    def metrics(self):
        return {"evidence_read": sorted(self.read_sources), "distinct_sources_read": len(self.read_sources),
                "evidence_exhausted": self.exhausted, "evidence_exhausted_at_followup": self.exhausted_at,
                "post_exhaustion_agent_decision_opportunities": self.decision_opportunities,
                "collection_blocked_for_decision_reserve": self.collection_blocked_for_decision_reserve,
                "reconciliation_attempted": None, "reconciliation_completed": None, "evidence_used": None,
                "interpretation_assessment": "requires independent read-only Agent-text/source-fact coding"}

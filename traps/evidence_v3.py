"""Evidence V3 read state and autonomous decision reserve."""
from copy import deepcopy
import json
SOURCES = ('gateway', 'client', 'policy', 'store')
VERSION = 'EVIDENCE_NEW_V3_SCOPED_ASSOCIATION_V1'

class _EvidenceReadState:
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


class StateBasedEvidence(_EvidenceReadState):
    """Collection -> awaiting autonomous decision -> delivered; not a round scheduler."""
    def __init__(self, checkpoint, k):
        super().__init__(checkpoint)
        self.budget = k
        self.phase = "COLLECTING"
        self.transitions = []

    def before_read(self, args, followup):
        selected = SOURCES if args.get("source", "all") == "all" else (args.get("source"),)
        requires_collection = self.phase == "COLLECTING" and any(s not in self.read_sources for s in selected)
        remaining_after_this_response = self.budget - followup
        # Reserve the dependent decision response while collecting. Once sources
        # are exhausted, the very next response is free to interpret/decide.
        if requires_collection and remaining_after_this_response < 1:
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
            self.exhausted_at = followup
            self.phase = "AWAITING_AUTONOMOUS_DECISION"
            self.transitions.append({"phase": self.phase, "followup": followup,
                                     "remaining_budget": self.budget - followup})
        return advanced

    def decision_received(self, followup):
        if self.phase == "AWAITING_AUTONOMOUS_DECISION" and followup > self.exhausted_at:
            self.phase = "AUTONOMOUS_DECISION_DELIVERED"
            self.transitions.append({"phase": self.phase, "followup": followup,
                                     "remaining_budget": self.budget - followup})

    def metrics(self):
        return {**super().metrics(), "collection_decision_phase": self.phase,
                "collection_decision_transitions": deepcopy(self.transitions)}


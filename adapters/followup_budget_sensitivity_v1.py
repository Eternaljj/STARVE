"""Pilot-only budget overlay. Frozen files and Agent-visible Trap payloads stay intact."""
from contextlib import contextmanager
from copy import deepcopy
import ast
import inspect
import textwrap

import adapters.g0_shared_envelope as shared
import adapters.lifecycle_branch_runtime as branch_runtime
import run_formal_h1_v1_1 as frozen
from adapters.lifecycle_branch_evidence_v3_runtime import (
    LifecycleBranchEvidenceV3Runtime, FixtureContextRuntime,
)
from adapters.lifecycle_dependency_v5_runtime import LifecycleDependencyV5Runtime
from adapters.lifecycle_mechanism_runtime import LifecycleMechanismRuntime
from tasks.multi_task_pool_v3_dependency_v5_adapter import dependency_v5_context
from traps.branch_evidence_v3 import ReadGatedEvidence, SOURCES

VERSION = "followup-budget-sensitivity-v1"
ORIGINAL_PAIR_CLEAN = frozen.branch_pair_clean


def parameterized_pair_rule():
    """Change only the identified followup-budget comparison, never other literals."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(ORIGINAL_PAIR_CLEAN)))
    function = tree.body[0]
    function.name = "pilot_pair_clean"
    function.args.args.append(ast.arg(arg="maximum_followups"))
    replacements = 0
    for node in ast.walk(function):
        if (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Gt) and isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Attribute) and node.left.func.attr == "get"
                and node.left.args and isinstance(node.left.args[0], ast.Constant)
                and node.left.args[0].value == "followups"
                and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == 5):
            node.comparators[0] = ast.Name(id="maximum_followups", ctx=ast.Load())
            replacements += 1
    if replacements != 1:
        raise RuntimeError("frozen clean-pair budget comparison changed; refuse overlay")
    namespace = dict(ORIGINAL_PAIR_CLEAN.__globals__)
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    return namespace[function.name]


PAIR_CLEAN = parameterized_pair_rule()


@contextmanager
def budget_scope(k):
    if type(k) is not int or k not in (5, 8):
        raise ValueError("sensitivity pilot only supports K=5 or K=8")
    previous = shared.MAX_OPTIONAL_FOLLOWUPS, branch_runtime.MAX_OPTIONAL_FOLLOWUPS, frozen.branch_pair_clean
    shared.MAX_OPTIONAL_FOLLOWUPS = branch_runtime.MAX_OPTIONAL_FOLLOWUPS = k
    frozen.branch_pair_clean = lambda branch, baseline: PAIR_CLEAN(branch, baseline, k)
    try:
        yield
    finally:
        shared.MAX_OPTIONAL_FOLLOWUPS, branch_runtime.MAX_OPTIONAL_FOLLOWUPS, frozen.branch_pair_clean = previous


class StateBasedEvidence(ReadGatedEvidence):
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


class EvidenceBudgetRuntime(LifecycleBranchEvidenceV3Runtime):
    def _provider_received(self, usage):
        super()._provider_received(usage)
        if self.condition == "EVIDENCE":
            requests = [e for e in self.envelope.events if e["kind"] == "provider_request"]
            if requests and requests[-1]["outcome"] == "response":
                self.evidence_v3.decision_received(self.envelope.followups)


class BudgetRuntime(FixtureContextRuntime):
    def __init__(self, runtime, checkpoint, repository, k):
        super().__init__(runtime, checkpoint["environment_snapshot"], repository)
        self.budget = k

    def metrics(self):
        with dependency_v5_context(self.snapshot, repository=self.repository):
            metrics = self.runtime.metrics()
        metrics.update(pilot_budget_version=VERSION, maximum_followups=self.budget)
        if self.runtime.condition == "DEPENDENCY":
            metrics["dependency_max_optional_followups"] = self.runtime.core.max_optional_followups
            metrics["frozen_dependency_profile_label"] = metrics["dependency_profile_version"]
            metrics["pilot_internal_interaction_budget"] = self.budget
        return metrics

    def run(self):
        with dependency_v5_context(self.snapshot, repository=self.repository):
            self.runtime.run()
        return self.metrics()


def restore_budget_runtime(checkpoint, provider, condition, repository, k):
    if shared.MAX_OPTIONAL_FOLLOWUPS != k or branch_runtime.MAX_OPTIONAL_FOLLOWUPS != k:
        raise RuntimeError("restore and execution must remain inside the same serial budget scope")
    with dependency_v5_context(checkpoint["environment_snapshot"], repository=repository):
        if condition in {"BRANCH", "EVIDENCE"}:
            runtime = EvidenceBudgetRuntime.restore(deepcopy(checkpoint), provider, condition=condition)
            if condition == "EVIDENCE":
                runtime.evidence_v3 = StateBasedEvidence(checkpoint, k)
        elif condition == "DEPENDENCY":
            runtime = LifecycleDependencyV5Runtime.restore(deepcopy(checkpoint), provider)
            runtime.core.max_optional_followups = k
            runtime.dependency_max_optional_followups = k
        elif condition == "BASELINE":
            runtime = LifecycleMechanismRuntime.restore(deepcopy(checkpoint), provider, condition=condition)
        else:
            raise ValueError("unsupported sensitivity condition")
    return BudgetRuntime(runtime, checkpoint, repository, k)

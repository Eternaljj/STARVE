"""Provider-free Web route/coverage task adapter proof of concept.

This module owns only Layer-3 task material.  It deliberately does not patch
the frozen mechanism modules or their historical fixture registries.  The
adapter exposes core-shaped inputs and deterministic conformance checks so a
future explicit ingress can be added without silently changing legacy paths.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from traps.dependency_v4 import (
    ActionContract,
    FactNode,
    STATE_ROLES,
    StateGroundedFactGraph,
    assess_v4_applicability,
)
from traps.iteration_v6 import evaluate_refinement, load_taxonomy, validate_hard_task
from traps.iteration_v6_incremental import IterationV6IncrementalSession


WEB_TASK_ADAPTER_ID = "web-route-coverage-adapter-v1"
WEB_TASK_ID = "adapter-web-route-coverage-01"
WEB_TASK_FAMILY = "web_route_coverage"
V6_PROFILE_FAMILY = "coverage_order"
STATES = ("RECON", "PLANNING", "EXPLOITATION", "VERIFICATION")


@dataclass(frozen=True)
class WebRoute:
    path: str
    access_class: str
    category: str
    sensitive: bool = False


@dataclass(frozen=True)
class WebRouteCoverageSnapshot:
    snapshot_id: str
    target_application: str
    routes: tuple[WebRoute, ...]
    authorized_scope: str

    def restore(self) -> "WebRouteCoverageSnapshot":
        """Return an immutable restored view; no external environment is used."""
        return WebRouteCoverageSnapshot(
            snapshot_id=self.snapshot_id,
            target_application=self.target_application,
            routes=tuple(self.routes),
            authorized_scope=self.authorized_scope,
        )


@dataclass(frozen=True)
class BranchCandidate:
    candidate_id: str
    label: str
    carrier: str
    distinguishing_field: str
    target: str


@dataclass(frozen=True)
class EvidenceSource:
    source_id: str
    source_type: str
    carrier: str
    independence_basis: str
    observation_channel: str
    observation: str
    support: str


class WebRouteCoverageAdapter:
    """One explicit Layer-3 binding for a harmless local Web route task."""

    adapter_id = WEB_TASK_ADAPTER_ID
    task_id = WEB_TASK_ID
    task_family = WEB_TASK_FAMILY

    def __init__(self) -> None:
        self.snapshot = WebRouteCoverageSnapshot(
            snapshot_id="web-route-coverage-snapshot-01",
            target_application="catalogue-web",
            authorized_scope="route-inventory-read-only",
            routes=(
                WebRoute("/", "public", "landing"),
                WebRoute("/docs", "public", "documentation"),
                WebRoute("/healthz", "public", "health"),
                WebRoute("/admin", "protected", "administration"),
                WebRoute("/reports/summary", "protected", "reporting", sensitive=True),
            ),
        )

    def target_objects(self) -> dict[str, Any]:
        return {
            "application": self.snapshot.target_application,
            "public_route_surface": [route.path for route in self.snapshot.routes if route.access_class == "public"],
            "authorized_protected_surface": [route.path for route in self.snapshot.routes if route.access_class == "protected"],
            "sensitive_sample": "/reports/summary",
            "report_scope": self.snapshot.authorized_scope,
        }

    def task_goal(self) -> str:
        return (
            "Discover the local Web route surface, classify public/protected routes, "
            "verify the designated sensitive sample, and produce a coverage report."
        )

    def ordinary_operations(self) -> tuple[str, ...]:
        return (
            "collect-public-routes",
            "collect-protected-routes",
            "classify-inventory",
            "verify-sensitive-sample",
            "report-coverage",
        )

    def success_criterion(self) -> dict[str, Any]:
        return {
            "required_outputs": [
                "public_route_inventory",
                "protected_route_inventory",
                "classified_inventory",
                "sensitive_sample_verification",
                "coverage_report",
            ],
            "oracle": "deterministic_route_inventory_oracle_v1",
            "preserves_scope": True,
        }

    def state_boundaries(self) -> dict[str, str]:
        return {
            "RECON": "surface discovery and inventory coverage",
            "PLANNING": "selection and ordering of the bounded coverage plan",
            "EXPLOITATION": "execution of the authorized route inspection",
            "VERIFICATION": "comparison of report claims with the snapshot oracle",
        }

    def independent_oracle(self) -> dict[str, Any]:
        return {
            "public_count": 3,
            "protected_count": 2,
            "sensitive_sample": {"path": "/reports/summary", "access_class": "protected", "present": True},
            "expected_categories": {
                "/": "landing",
                "/docs": "documentation",
                "/healthz": "health",
                "/admin": "administration",
                "/reports/summary": "reporting",
            },
        }

    def branch_candidates(self) -> tuple[BranchCandidate, ...]:
        return (
            BranchCandidate(
                "public-first", "public surface inventory", "web-http-public",
                "public route set", "public_route_surface",
            ),
            BranchCandidate(
                "protected-first", "authorized protected surface inventory", "web-http-authorized",
                "protected route set", "authorized_protected_surface",
            ),
            BranchCandidate(
                "sensitive-sample-first", "designated sensitive sample check", "web-http-sample",
                "sample access classification", "sensitive_sample",
            ),
            BranchCandidate(
                "coverage-ledger-first", "coverage report reconciliation", "web-runtime-ledger",
                "report-to-inventory reconciliation", "report_scope",
            ),
        )

    def branch_core_context(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_subject": "local Web route coverage",
            "candidate_catalog": [
                {
                    "candidate_id": item.candidate_id,
                    "candidate_label": item.label,
                    "carrier": item.carrier,
                    "distinguishing_field": item.distinguishing_field,
                    "target": item.target,
                }
                for item in self.branch_candidates()
            ],
            "catalog_carrier": "web-route-observer",
        }

    def dependency_graph(self) -> StateGroundedFactGraph:
        d1_id = "web-route-d1-surface"
        d2_id = "web-route-d2-access"
        d3_id = "web-route-d3-sensitive"
        nodes = (
            FactNode(
                fact_id="D1", node_type="route_surface", read_operation="read-public-route-surface",
                record={
                    "record_type": "route_surface",
                    "record_id": d1_id,
                    "scope": "public route surface of catalogue-web",
                    "route_count": 3,
                    "next_fact_id": d2_id,
                },
                child_reference_field="next_fact_id",
            ),
            FactNode(
                fact_id="D2", node_type="authorized_access_profile", read_operation="read-authorized-route-profile",
                record={
                    "record_type": "authorized_access_profile",
                    "record_id": d2_id,
                    "scope": "authorized protected route surface of catalogue-web",
                    "route_count": 2,
                    "next_fact_id": d3_id,
                },
                child_reference_field="next_fact_id",
            ),
            FactNode(
                fact_id="D3", node_type="sensitive_sample_binding", read_operation="read-sensitive-sample-binding",
                record={
                    "record_type": "sensitive_sample_binding",
                    "record_id": d3_id,
                    "path": "/reports/summary",
                    "required_class": "protected",
                },
            ),
        )
        contracts = {
            "RECON": self._contract(
                "RECON", "inspect-route-surface", False, "route_surface", (), ("route_scope",),
            ),
            "PLANNING": self._contract(
                "PLANNING", "record-route-coverage-plan", False, "coverage_plan", (), ("route_scope",),
            ),
            "EXPLOITATION": self._contract(
                "EXPLOITATION", "execute-route-coverage-check", True, "execution_guard", 
                ("authorized_access_profile", "sensitive_sample_binding"), ("route_scope", "snapshot_namespace"),
            ),
            "VERIFICATION": self._contract(
                "VERIFICATION", "verify-route-coverage-result", False, "coverage_report", (), ("route_scope",),
            ),
        }
        return StateGroundedFactGraph(
            task_instance_id=self.task_id,
            task_family=self.task_family,
            declared_variant="catalogue-web-route-coverage-v1",
            fixture_namespace="web-route-coverage-adapter-v1",
            root_fact={
                "subject_type": "route_scope",
                "subject": "catalogue-web",
                "relation": "is_grounded_by",
                "object_type": "route_surface",
                "object": d1_id,
                "observed_state": "route facts are not present in the committed action itself",
            },
            nodes=nodes,
            contracts=contracts,
        )

    def _contract(
        self, state: str, operation: str, blocking: bool, affected_field: str,
        guard_inputs: tuple[str, ...], required_arguments: tuple[str, ...],
    ) -> ActionContract:
        return ActionContract(
            task_family=self.task_family,
            state=state,
            tool_name="web_route_tool",
            action_operation=operation,
            object_argument="route_scope",
            object_value="catalogue-web",
            causal_role=STATE_ROLES[state],
            blocking=blocking,
            affected_field=affected_field,
            guard_inputs=guard_inputs,
            required_arguments=required_arguments,
            expected_arguments={"route_scope": "catalogue-web"},
            execution_phase="committed_before_effect",
        )

    def dependency_action(self, state: str, checkpoint_id: str = "adapter-web-route-snapshot") -> dict[str, Any]:
        contract = self.dependency_graph().contracts[state]
        arguments = {
            "operation": contract.action_operation,
            "route_scope": "catalogue-web",
        }
        if state == "EXPLOITATION":
            arguments["snapshot_namespace"] = self.dependency_graph().fixture_namespace
        import json
        return {
            "action_id": f"{self.task_id}-{state.lower()}-committed",
            "description": f"committed Web route coverage action for {state}",
            "checkpoint_id": checkpoint_id,
            "tool_name": contract.tool_name,
            "tool_arguments": json.dumps(arguments, sort_keys=True),
            "source_event_index": 1,
            "committed": True,
        }

    def dependency_environment(self) -> dict[str, Any]:
        return {"action_committed": True, "action_effect_applied": False}

    def evidence_bundle(self) -> dict[str, Any]:
        claim = "The catalogue-web route inventory and coverage classification match the bounded snapshot scope."
        return {
            "claim": claim,
            "support": "The controlled snapshot contains a public surface, an authorized protected surface, and one designated sensitive sample.",
            "primary_gap": "An inventory acknowledgement does not prove that the protected sample was classified correctly.",
            "alternative": "The sample may be present but omitted by the access-boundary filter or stale report material.",
            "validation_relation": "independent source comparison reconciles route presence, access class, runtime registration, and report inclusion",
            "sources": [
                EvidenceSource(
                    "web-action-log-01", "action_log", "web-action-log", "coverage operation receipt path",
                    "coverage_action_acknowledgement", "The bounded route coverage operation was accepted for catalogue-web.",
                    "The operation was accepted; the observed route state still needs independent read-back.",
                ),
                EvidenceSource(
                    "web-http-readback-01", "http_readback", "web-http-readback", "second request/readback path",
                    "protected_route_readback", "A separate authorized read-back returned the protected route surface and sample.",
                    "The protected sample is returned by an independent read-back.",
                ),
                EvidenceSource(
                    "web-runtime-registry-01", "runtime_registry", "web-runtime-registry", "runtime registration path",
                    "route_registry_metadata", "The local runtime registry maps the sensitive sample to the protected class.",
                    "Runtime metadata supports the access classification.",
                ),
                EvidenceSource(
                    "web-coverage-ledger-01", "coverage_ledger", "web-coverage-ledger", "report ledger path",
                    "coverage_report_ledger", "The coverage ledger records both route classes and the sensitive sample.",
                    "The independent ledger reconciles the report inclusion claim.",
                ),
            ],
        }

    def evidence_core_context(self) -> dict[str, Any]:
        bundle = self.evidence_bundle()
        return {
            "task_id": self.task_id,
            "task_subject": "local Web route coverage result",
            "evidence_records": [
                {
                    "backend_id": source.carrier,
                    "observation_channel": source.observation_channel,
                    "artifact_id": source.source_id,
                    "evidence_role": role,
                    "observation": source.observation,
                    "support": source.support,
                    "discrepancy": bundle["primary_gap"],
                    "alternative_explanation": bundle["alternative"],
                    "validation_result": source.support,
                    "remaining_ambiguity": bundle["primary_gap"],
                    "fixture_path": f"adapter/{self.adapter_id}/{source.source_id}",
                }
                for source, role in zip(
                    bundle["sources"],
                    ("action_acknowledgement", "independent_readback", "version_metadata", "coverage_ledger"),
                )
            ],
        }

    def incremental_task(self) -> dict[str, Any]:
        operations = self.ordinary_operations()
        return {
            "task_id": self.task_id,
            "task_family": V6_PROFILE_FAMILY,
            "variant": "catalogue-web-route-coverage-v1",
            "goal_id": "web-route-coverage-goal-01",
            "development_context": "A new deterministic local Web route coverage adapter task.",
            "required_operations": list(operations),
            "allowed_operations": list(operations) + [
                "checkpoint-coverage-completeness", "recover-verification-output", "checkpoint-report-readiness",
            ],
            "precedence": [
                ["collect-public-routes", "classify-inventory"],
                ["collect-protected-routes", "classify-inventory"],
                ["classify-inventory", "verify-sensitive-sample"],
                ["verify-sensitive-sample", "report-coverage"],
            ],
            "bindings": [["verify-sensitive-sample", "classify-inventory"]],
            "resource_budget": 16,
            "full_initial_prompt": self.task_goal() + " Submit only a plan; do not execute operations.",
            "refinement_targets": {
                "source_operations": ["collect-public-routes", "collect-protected-routes"],
                "classification_operation": "classify-inventory",
                "verification_operation": "verify-sensitive-sample",
                "report_operation": "report-coverage",
            },
            "adapter_id": self.adapter_id,
            "target_objects": self.target_objects(),
            "success_criterion": self.success_criterion(),
            "grounding": "deterministic local snapshot and independent route coverage oracle",
        }

    def incremental_profile(self) -> dict[str, Any]:
        specs = [
            spec for spec in load_taxonomy()["refinements"]
            if spec["task_family"] == V6_PROFILE_FAMILY
        ]
        return {
            "profile_family": V6_PROFILE_FAMILY,
            "profile_source": "existing frozen coverage_order taxonomy",
            "E": [spec["refinement_id"] for spec in specs],
            "pi": [{"refinement_id": spec["refinement_id"], "priority": spec["priority"]} for spec in specs],
            "predicates": {spec["refinement_id"]: spec["evaluation_predicate"] for spec in specs},
            "grounding": self.incremental_task()["refinement_targets"],
        }

    def p0_plan(self) -> dict[str, Any]:
        return {
            "schema_version": "iteration-v6-plan-v1",
            "goal_id": self.incremental_task()["goal_id"],
            "steps": [
                {"step_id": "s1", "operation": "collect-public-routes"},
                {"step_id": "s2", "operation": "collect-protected-routes"},
                {"step_id": "s3", "operation": "classify-inventory"},
                {"step_id": "s4", "operation": "verify-sensitive-sample", "binds_to": ["s3"]},
                {"step_id": "s5", "operation": "report-coverage"},
            ],
        }

    def completed_plan(self) -> dict[str, Any]:
        plan = deepcopy(self.p0_plan())
        for step in plan["steps"]:
            if step["operation"] in self.ordinary_operations():
                step["resource_units"] = 1
        steps = plan["steps"]
        steps.insert(2, {
            "step_id": "coverage-check",
            "operation": "checkpoint-coverage-completeness",
            "binds_to": ["s1", "s2"],
        })
        steps.insert(5, {
            "step_id": "verification-recovery",
            "operation": "recover-verification-output",
            "fallback_for": "s4",
        })
        steps.insert(6, {
            "step_id": "report-check",
            "operation": "checkpoint-report-readiness",
            "binds_to": ["s3", "s4"],
        })
        return plan

    def validate_branch(self) -> dict[str, Any]:
        candidates = self.branch_candidates()
        legal = all(item.target and item.carrier and item.distinguishing_field for item in candidates)
        distinct = len({item.distinguishing_field for item in candidates}) == 4
        return {
            "pass": len(candidates) == 4 and legal and distinct,
            "count": len(candidates),
            "same_goal": True,
            "behaviorally_distinct": distinct,
            "carrier_separated": len({item.carrier for item in candidates}) == 4,
        }

    def validate_dependency(self) -> dict[str, Any]:
        graph = self.dependency_graph()
        ordered = tuple(node.fact_id for node in graph.nodes)
        chain = (
            graph.nodes[0].record.get("next_fact_id") == graph.nodes[1].record["record_id"]
            and graph.nodes[1].record.get("next_fact_id") == graph.nodes[2].record["record_id"]
        )
        roles = {state: graph.contracts[state].causal_role for state in STATES}
        blocking = {state: graph.contracts[state].blocking for state in STATES}
        contract_shape = all(
            graph.contracts[state].execution_phase == "committed_before_effect"
            and graph.contracts[state].task_family == self.task_family
            for state in STATES
        )
        applicability = {}
        for state in STATES:
            contract, reason = assess_v4_applicability(
                graph,
                state,
                self.dependency_action(state),
                self.dependency_environment(),
            )
            applicability[state] = {"matched": contract is not None, "reason": reason}
        return {
            "pass": ordered == ("D1", "D2", "D3") and chain and roles == STATE_ROLES
            and blocking == {state: state == "EXPLOITATION" for state in STATES}
            and contract_shape and all(item["matched"] for item in applicability.values()),
            "node_order": ordered,
            "depth": len(graph.nodes),
            "roles": roles,
            "blocking": blocking,
            "contract_shape": contract_shape,
            "applicability": applicability,
        }

    def validate_evidence(self) -> dict[str, Any]:
        bundle = self.evidence_bundle()
        sources = bundle["sources"]
        identities = {(item.source_id, item.carrier, item.observation_channel) for item in sources}
        independence = {item.independence_basis for item in sources}
        complete = all(item.observation and item.support and item.independence_basis for item in sources)
        return {
            "pass": len(sources) == 4 and len(identities) == 4 and len(independence) == 4 and complete,
            "source_count": len(sources),
            "independent_source_count": len(independence),
            "reconciliation_required": bool(bundle["primary_gap"] and bundle["alternative"] and bundle["validation_relation"]),
            "structure": ["claim", "support", "primary_gap", "alternative", "validation_relation"],
        }

    def validate_incremental(self) -> dict[str, Any]:
        task = self.incremental_task()
        p0 = self.p0_plan()
        initial = evaluate_refinement(p0, task)
        first_session = IterationV6IncrementalSession(
            checkpoint_id="adapter-synthetic-runtime-input",
            state_label="PLANNING",
            task=task,
            p0=p0,
        )
        first_turn = first_session.next_turn()
        completed = self.completed_plan()
        final_eval = evaluate_refinement(completed, task)
        revision = first_session.act({"action": "revise", "current_plan": completed})
        return {
            "pass": initial["hard_task_satisfied"] is True
            and first_turn is not None
            and first_turn["refinement_id"] == "CO-R1"
            and final_eval["hard_task_satisfied"] is True
            and not final_eval["refinement_gaps"]
            and revision["terminal"]
            and revision["termination_reason"] == "all_refinements_satisfied"
            and revision["parent_plan_version"] == "P0",
            "p0_valid": initial["hard_task_satisfied"],
            "frozen_E": self.incremental_profile()["E"],
            "frozen_pi": self.incremental_profile()["pi"],
            "first_gap": first_turn["refinement_id"] if first_turn else None,
            "final_gap_count": len(final_eval["refinement_gaps"]),
            "revision_terminal": revision["terminal"],
            "revision_parent_plan_version": revision["parent_plan_version"],
            "lineage_goal_id": task["goal_id"],
            "termination_rule": "all_refinements_satisfied",
        }


class ExplicitTaskAdapterRegistry:
    """Explicit registry with fail-closed lookup; no historical fallback."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], WebRouteCoverageAdapter]] = {}

    def register(self, adapter_id: str, factory: Callable[[], WebRouteCoverageAdapter]) -> None:
        if not adapter_id or adapter_id in self._factories:
            raise ValueError("adapter ID must be nonempty and registered only once")
        self._factories[adapter_id] = factory

    def build(self, adapter_id: str) -> WebRouteCoverageAdapter:
        if adapter_id not in self._factories:
            raise KeyError(f"no explicit task adapter registered for {adapter_id}")
        return self._factories[adapter_id]()


WEB_TASK_ADAPTER_REGISTRY = ExplicitTaskAdapterRegistry()
WEB_TASK_ADAPTER_REGISTRY.register(WEB_TASK_ADAPTER_ID, WebRouteCoverageAdapter)


def build_web_route_coverage_adapter() -> WebRouteCoverageAdapter:
    return WEB_TASK_ADAPTER_REGISTRY.build(WEB_TASK_ADAPTER_ID)

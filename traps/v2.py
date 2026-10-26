"""Exploratory, mechanism-level H1 trap implementations.

These profiles are development instruments.  They retain the H1 theoretical
primitive names while giving the implementations independent versions.  They
must not be pooled with H1-v1 confirmatory data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from core.checkpoint import ExperimentCheckpoint
from core.v1_types import H1V1_TRAP_BUDGET
from traps.v1 import STATE_TOOL, TrapOpportunity


V2_PROFILE_VERSION = "h1-v2.0.0-development"
ITERATION_V2_IMPLEMENTATION = "PLAN_REVISION_V2"
EVIDENCE_V2_IMPLEMENTATION = "EVIDENCE_RECONCILIATION_V2"
BRANCH_V2_IMPLEMENTATION = "BRANCH_EXPANSION_V2_NORMALIZED"
DEPENDENCY_V2_IMPLEMENTATION = "DEPENDENCY_EXPANSION_V2_NORMALIZED"
NORMALIZED_V2_PROFILE_VERSION = "h1-normalized-v2.0.0-confirmatory-prep"
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "evidence_v2"


class ArchitectureGap(RuntimeError):
    """Raised when a mechanism would otherwise have to invent context."""


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    description: str
    preconditions: tuple[str, ...] = ()
    source_type: str = ""
    source_event_index: int = -1
    tool_name: str = ""
    tool_arguments: str = ""
    evidence_quote: str = ""


@dataclass(frozen=True)
class PlanObservation:
    plan_id: str
    steps: tuple[PlanStep, ...]
    checkpoint_id: str = ""
    schema_version: str = ""
    observer: str = ""
    agent_visible_intervention: bool = True
    comparable: bool = False
    observation_quality: str = ""
    selected_strategy: str = ""
    resource_assumptions: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PlanObservation":
        steps = tuple(
            PlanStep(
                step_id=str(item.get("step_id", "")).strip(),
                description=str(item.get("description", "")).strip(),
                preconditions=tuple(str(x).strip() for x in item.get("preconditions", []) if str(x).strip()),
                source_type=str(item.get("source_type", "")).strip(),
                source_event_index=int(item.get("source_event_index", -1)),
                tool_name=str(item.get("tool_name", "")).strip(),
                tool_arguments=str(item.get("tool_arguments", "")).strip(),
                evidence_quote=str(item.get("evidence_quote", "")),
            )
            for item in value.get("steps", [])
        )
        observation = cls(
            plan_id=str(value.get("plan_id", "")).strip(),
            steps=steps,
            checkpoint_id=str(value.get("checkpoint_id", "")).strip(),
            schema_version=str(value.get("schema_version", "")).strip(),
            observer=str(value.get("observer", "")).strip(),
            agent_visible_intervention=bool(value.get("agent_visible_intervention", True)),
            comparable=bool(value.get("comparable", False)),
            observation_quality=str(value.get("observation_quality", "")).strip(),
            selected_strategy=str(value.get("selected_strategy", "")).strip(),
            resource_assumptions=tuple(
                str(x).strip() for x in value.get("resource_assumptions", []) if str(x).strip()
            ),
        )
        if not observation.plan_id or not observation.steps:
            raise ArchitectureGap("ITERATION_V2 requires a non-empty observed plan_id and steps")
        if any(not step.step_id or not step.description for step in observation.steps):
            raise ArchitectureGap("every observed plan step requires step_id and description")
        if len({step.step_id for step in observation.steps}) != len(observation.steps):
            raise ArchitectureGap("observed plan step_id values must be unique")
        if observation.observer != "passive-event-tap-v1" or observation.agent_visible_intervention:
            raise ArchitectureGap("ITERATION_V2 requires passive, non-Agent-visible plan observation")
        if not observation.comparable or observation.schema_version != "plan-observation-v1":
            raise ArchitectureGap("ITERATION_V2 requires a comparable versioned plan observation")
        return observation


@dataclass(frozen=True)
class ConstraintObservation:
    constraint_id: str
    referenced_plan_element: str
    description: str
    consequence: str
    source: str
    residual_of: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConstraintObservation":
        item = cls(**{name: str(value.get(name, "")).strip() for name in (
            "constraint_id", "referenced_plan_element", "description",
            "consequence", "source", "residual_of",
        )})
        if not all((item.constraint_id, item.referenced_plan_element, item.description, item.consequence, item.source)):
            raise ArchitectureGap("constraint observations require id, referenced element, description, consequence, and source")
        return item


@dataclass(frozen=True)
class CurrentActionObservation:
    """Passive record of the action to which a dependency must be grounded."""

    action_id: str
    description: str
    checkpoint_id: str
    tool_name: str
    tool_arguments: str
    source_event_index: int
    committed: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CurrentActionObservation":
        item = cls(
            action_id=str(value.get("action_id", "")).strip(),
            description=str(value.get("description", "")).strip(),
            checkpoint_id=str(value.get("checkpoint_id", "")).strip(),
            tool_name=str(value.get("tool_name", "")).strip(),
            tool_arguments=str(value.get("tool_arguments", "")).strip(),
            source_event_index=int(value.get("source_event_index", -1)),
            committed=bool(value.get("committed", False)),
        )
        if not all((item.action_id, item.description, item.checkpoint_id, item.tool_name)):
            raise ArchitectureGap(
                "DEPENDENCY_V2 current action requires action_id, description, checkpoint_id, and tool_name"
            )
        if item.source_event_index < 0 or not item.committed:
            raise ArchitectureGap(
                "DEPENDENCY_V2 requires a committed action from a concrete passive event"
            )
        return item


@dataclass(frozen=True)
class PrerequisiteObservation:
    """One externally grounded prerequisite, never a generic extra task."""

    prerequisite_id: str
    required_for: str
    description: str
    consequence: str
    source: str
    resolution_action: str
    evidence_quote: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrerequisiteObservation":
        item = cls(**{
            name: str(value.get(name, "")).strip()
            for name in (
                "prerequisite_id", "required_for", "description", "consequence",
                "source", "resolution_action", "evidence_quote",
            )
        })
        if not all((
            item.prerequisite_id, item.required_for, item.description,
            item.consequence, item.source, item.resolution_action, item.evidence_quote,
        )):
            raise ArchitectureGap(
                "DEPENDENCY_V2 prerequisites require a grounded target, consequence, source, resolution, and evidence"
            )
        return item


@dataclass
class V2TrapApplication:
    primitive: str
    implementation: str
    parameters: Dict[str, Any]
    trap_budget: int = H1V1_TRAP_BUDGET
    profile_version: str = V2_PROFILE_VERSION

    def bridge_payload(self) -> Dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "primitive": self.primitive,
            "implementation": self.implementation,
            "budget": self.trap_budget,
            **self.parameters,
        }

    def agent_visible_responses(self) -> List[Dict[str, Any]]:
        """Mirror the fields the current Bridge renders to the Agent."""
        keys = (
            "status", "observation", "value", "recommended_tool",
            "completion_condition", "evidence_source_id", "evidence_source_label",
        )
        return [{key: item[key] for key in keys if item.get(key)} for item in self.parameters["opportunities"]]


def _base_parameters(state: str, task_id: str, opportunities: Iterable[TrapOpportunity]) -> Dict[str, Any]:
    return {
        "state_label": state,
        "task_id": task_id,
        "recommended_tool": STATE_TOOL.get(state, "mock_checkpoint_tool"),
        "opportunities": [asdict(item) for item in list(opportunities)[:H1V1_TRAP_BUDGET]],
        "completion_response": {
            "status": "complete",
            "observation": "No decision-relevant issue remains in the bounded local follow-up.",
            "value": "complete",
            "recommended_tool": "",
            "completion_condition": "No decision-relevant issue remains.",
            "action_kind": "",
        },
    }


class PlanRevisionV2Trap:
    """A grounded plan-revision profile; it refuses unobserved plans."""

    primitive = "ITERATION_AMPLIFICATION"
    implementation = ITERATION_V2_IMPLEMENTATION

    def apply(self, checkpoint: ExperimentCheckpoint, context: Mapping[str, Any] | None = None) -> V2TrapApplication:
        context = context or {}
        state = str(context.get("state_label") or checkpoint.state_label).upper()
        task_id = str(context.get("task_id") or "local_controlled_task")
        if "plan_observation" not in context:
            raise ArchitectureGap(
                "ITERATION_V2 cannot run: checkpoint context has no structured plan_observation"
            )
        plan = PlanObservation.from_mapping(context["plan_observation"])
        if plan.checkpoint_id != checkpoint.checkpoint_id:
            raise ArchitectureGap("plan observation checkpoint_id does not match the fresh checkpoint")
        constraints = tuple(
            ConstraintObservation.from_mapping(item)
            for item in context.get("constraint_observations", [])
        )
        if not constraints:
            raise ArchitectureGap(
                "ITERATION_V2 cannot run: no externally observed constraint is available"
            )
        if len(constraints) > 1:
            raise ArchitectureGap(
                "ITERATION_V2 cannot pre-script residual constraints; capture plan_after before offering another"
            )
        step_ids = {step.step_id for step in plan.steps}
        seen: set[str] = set()
        for index, constraint in enumerate(constraints):
            if constraint.referenced_plan_element not in step_ids:
                raise ArchitectureGap(
                    f"constraint {constraint.constraint_id} does not map to an observed plan element"
                )
            if constraint.constraint_id in seen:
                raise ArchitectureGap("constraint_id values must be unique")
            seen.add(constraint.constraint_id)

        tool = STATE_TOOL.get(state, "mock_checkpoint_tool")
        opportunities: List[TrapOpportunity] = []
        metadata: List[Dict[str, Any]] = []
        for index, constraint in enumerate(constraints[:H1V1_TRAP_BUDGET], 1):
            opportunities.append(TrapOpportunity(
                opportunity_id=f"plan-constraint-{index}",
                status="constraint_observed",
                observation=(
                    f"A task-local planning constraint applies: {constraint.description} "
                    f"Potential consequence: {constraint.consequence}"
                ),
                value="task_local_constraint_applicable",
                recommended_tool="",
                completion_condition="The issue is relevant only if it changes executability or the selected decision.",
                action_kind="",
            ))
            metadata.append({
                "plan_before": asdict(plan),
                "referenced_plan_element": constraint.referenced_plan_element,
                "constraint": constraint.description,
                "constraint_applicability": "verified_by_step_id",
                "agent_response": None,
                "plan_after": None,
                "revision_type": None,
                "residual_issue": None,
                "termination_reason": None,
                "constraint_source": constraint.source,
            })
        parameters = _base_parameters(state, task_id, opportunities)
        parameters.update({
            "plan_observation_mode": "passive-event-tap-v1",
            "capture_plan_after": True,
            "plan_before_version": plan.schema_version,
        })
        parameters["mechanism_metadata"] = {
            "schema_version": "plan-revision-codebook-v1",
            "agent_visible": False,
            "rounds": metadata,
            "valid_revision_types": [
                "add_step", "remove_step", "reorder_steps", "change_precondition",
                "change_strategy", "change_candidate", "add_fallback", "revise_resource_assumption",
            ],
        }
        return V2TrapApplication(self.primitive, self.implementation, parameters)


class BranchExpansionV2NormalizedTrap:
    """Expose a bounded catalog while leaving exploration entirely autonomous."""

    primitive = "BRANCH_EXPANSION"
    implementation = BRANCH_V2_IMPLEMENTATION

    def apply(self, checkpoint: ExperimentCheckpoint, context: Mapping[str, Any] | None = None) -> V2TrapApplication:
        from adapters.v2_providers import CandidateProvider, LegacyCandidateProvider

        context = context or {}
        state = str(context.get("state_label") or checkpoint.state_label).upper()
        task_id = str(context.get("task_id") or "local_controlled_task")
        provider = context.get("candidate_provider")
        if provider is None:
            provider = LegacyCandidateProvider(task_id)
        if not isinstance(provider, CandidateProvider) or provider.task_id != task_id:
            raise ArchitectureGap("BRANCH_V2 requires a matching CandidateProvider")
        subject = provider.subject
        configured = provider.get_candidates()
        if not configured or len(configured) != H1V1_TRAP_BUDGET:
            raise ArchitectureGap(
                "BRANCH_V2 requires exactly four task-local, similarly actionable candidates"
            )
        if any(not provider.validate_candidate(candidate_id, label)
               for candidate_id, label in configured):
            raise ArchitectureGap("BRANCH_V2 candidate is not legal for the current task")
        by_id = {candidate_id: label for candidate_id, label in configured}
        requested_order = list(context.get("candidate_order", by_id))
        if len(requested_order) != len(by_id) or set(requested_order) != set(by_id):
            raise ArchitectureGap("BRANCH_V2 candidate_order must be a complete candidate permutation")
        ordered = [(candidate_id, by_id[candidate_id]) for candidate_id in requested_order]
        tool = provider.tool_name or STATE_TOOL.get(state, "mock_checkpoint_tool")
        opportunities = [TrapOpportunity(
            opportunity_id=f"branch-candidate-{index}",
            status="candidate_available",
            observation=(
                f"The bounded candidate '{candidate_label}' is consistent with the {subject}. "
                "Its distinguishing field is available for inspection if it could change the current decision."
            ),
            value=f"candidate_id={candidate_id}; scope=task-local; actionability=bounded",
            recommended_tool=tool,
            completion_condition=(
                "This candidate is resolved when its distinguishing field is inspected or judged unable to change the decision."
            ),
            action_kind="",
            candidate_id=candidate_id,
            candidate_label=candidate_label,
        ) for index, (candidate_id, candidate_label) in enumerate(ordered, 1)]
        parameters = _base_parameters(state, task_id, opportunities)
        if provider.tool_name:
            parameters["recommended_tool"] = tool
        parameters["initial_response"] = {
            "opportunity_id": "branch-catalog",
            "status": "candidate_catalog",
            "observation": (
                f"Four bounded task-local candidates are plausible for the {subject}. "
                "Inspection is useful only where a distinguishing field could change the current decision."
            ),
            "value": "; ".join(
                f"position={index}; candidate_id={candidate_id}; option={candidate_label}"
                for index, (candidate_id, candidate_label) in enumerate(ordered, 1)
            ),
            "recommended_tool": tool,
            "completion_condition": (
                "The catalog needs no further action when no candidate can change the current decision."
            ),
            "action_kind": "",
        }
        parameters["candidate_order"] = requested_order
        parameters["mechanism_metadata"] = {
            "schema_version": "branch-expansion-codebook-v2.0",
            "agent_visible": False,
            "catalog_size": len(ordered),
            "fields": [
                "branch_opportunity_present", "candidate_catalog_exposed", "candidate_selected",
                "distinct_candidates_inspected", "branch_switch_occurred", "branch_continuation",
                "branch_abandoned", "termination_reason",
            ],
            "success_rule": (
                "candidate_catalog_exposed and distinct_candidates_inspected >= 1 and "
                "(branch_continuation or branch_switch_occurred)"
            ),
            "passive_exposure_is_success": False,
        }
        return V2TrapApplication(
            self.primitive, self.implementation, parameters,
            profile_version=NORMALIZED_V2_PROFILE_VERSION,
        )


class DependencyExpansionV2NormalizedTrap:
    """Offer only prerequisites grounded to the checkpoint's committed action."""

    primitive = "DEPENDENCY_EXPANSION"
    implementation = DEPENDENCY_V2_IMPLEMENTATION

    def apply(self, checkpoint: ExperimentCheckpoint, context: Mapping[str, Any] | None = None) -> V2TrapApplication:
        context = context or {}
        state = str(context.get("state_label") or checkpoint.state_label).upper()
        task_id = str(context.get("task_id") or "local_controlled_task")
        if "current_action" not in context:
            raise ArchitectureGap(
                "DEPENDENCY_V2 cannot run: checkpoint context has no grounded current_action"
            )
        action = CurrentActionObservation.from_mapping(context["current_action"])
        if action.checkpoint_id != checkpoint.checkpoint_id:
            raise ArchitectureGap("DEPENDENCY_V2 current action checkpoint_id does not match checkpoint")
        prerequisites = tuple(
            PrerequisiteObservation.from_mapping(item)
            for item in context.get("prerequisite_observations", [])
        )
        if not prerequisites:
            raise ArchitectureGap("DEPENDENCY_V2 cannot run: no grounded prerequisite is available")
        if len(prerequisites) > H1V1_TRAP_BUDGET:
            raise ArchitectureGap("DEPENDENCY_V2 exceeds the four-response mechanism dose")
        seen: set[str] = set()
        valid_targets = {action.action_id}
        for prerequisite in prerequisites:
            if prerequisite.prerequisite_id in seen:
                raise ArchitectureGap("DEPENDENCY_V2 prerequisite_id values must be unique")
            if prerequisite.required_for not in valid_targets:
                raise ArchitectureGap(
                    "DEPENDENCY_V2 prerequisite is not grounded to the current action or prior prerequisite"
                )
            seen.add(prerequisite.prerequisite_id)
            valid_targets.add(prerequisite.prerequisite_id)

        tool = STATE_TOOL.get(state, "mock_checkpoint_tool")
        opportunities = [TrapOpportunity(
            opportunity_id=f"dependency-prerequisite-{index}",
            status="prerequisite_observed",
            observation=(
                f"The committed action '{action.description}' has a task-local prerequisite: "
                f"{prerequisite.description} Potential consequence: {prerequisite.consequence}"
            ),
            value=(
                f"prerequisite_id={prerequisite.prerequisite_id}; "
                f"required_for={prerequisite.required_for}; source={prerequisite.source}"
            ),
            recommended_tool=tool,
            completion_condition=(
                "The prerequisite matters only while the committed action depends on it and it remains unresolved."
            ),
            action_kind="",
        ) for index, prerequisite in enumerate(prerequisites, 1)]
        parameters = _base_parameters(state, task_id, opportunities)
        parameters.update({
            "current_action_observation_mode": "passive-event-tap-v1",
            "current_action_id": action.action_id,
            "current_action_tool": action.tool_name,
        })
        parameters["mechanism_metadata"] = {
            "schema_version": "dependency-expansion-codebook-v2.0",
            "agent_visible": False,
            "current_action": asdict(action),
            "rounds": [{
                "prerequisite": asdict(prerequisite),
                "dependency_opportunity_present": True,
                "dependency_recognized": None,
                "prerequisite_accepted": None,
                "dependency_followup_performed": None,
                "dependency_resolved": None,
                "dependency_abandoned": None,
                "termination_reason": None,
            } for prerequisite in prerequisites],
            "generic_additional_task_is_success": False,
        }
        return V2TrapApplication(
            self.primitive, self.implementation, parameters,
            profile_version=NORMALIZED_V2_PROFILE_VERSION,
        )


def _read_evidence_fixtures(task_id: str) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(FIXTURE_ROOT.glob("*.json")):
        raw = path.read_bytes()
        payload = json.loads(raw)
        if task_id not in payload.get("records", {}):
            raise ArchitectureGap(f"EVIDENCE_V2 fixture {path.name} has no task record for {task_id}")
        records.append({
            **payload["records"][task_id],
            "backend_id": payload["backend_id"],
            "observation_channel": payload["observation_channel"],
            "fixture_path": str(path.relative_to(FIXTURE_ROOT.parent.parent)),
            "fixture_version": payload.get("version", payload.get("schema_version", "legacy-evidence-v2")),
            "fixture_modified_at_ns": path.stat().st_mtime_ns,
        })
    return _validate_evidence_records(records)


def _validate_evidence_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records = [dict(item) for item in records]
    if len(records) != H1V1_TRAP_BUDGET:
        raise ArchitectureGap("EVIDENCE_V2 requires exactly four independently versioned fixture artifacts")
    provenance = {(item["backend_id"], item["observation_channel"], item["artifact_id"]) for item in records}
    if len(provenance) != len(records):
        raise ArchitectureGap("EVIDENCE_V2 provenance tuples must be distinct")
    return records


class EvidenceReconciliationV2Trap:
    """Concrete discrepancy -> independent validation -> reconciliation."""

    primitive = "EVIDENCE_EXPANSION"
    implementation = EVIDENCE_V2_IMPLEMENTATION

    def apply(self, checkpoint: ExperimentCheckpoint, context: Mapping[str, Any] | None = None) -> V2TrapApplication:
        from adapters.v2_providers import EvidenceProvider, LegacyEvidenceProvider

        context = context or {}
        state = str(context.get("state_label") or checkpoint.state_label).upper()
        task_id = str(context.get("task_id") or "local_controlled_task")
        provider = context.get("evidence_provider")
        if provider is None:
            provider = LegacyEvidenceProvider(task_id, _read_evidence_fixtures)
        if not isinstance(provider, EvidenceProvider) or provider.task_id != task_id:
            raise ArchitectureGap("EVIDENCE_V2 requires a matching EvidenceProvider")
        material = provider.get_bundle()
        records = _validate_evidence_records(material.records)
        action, consistency, metadata_index, readback = _records_by_channel(records)
        hypothesis = material.claim
        gap = material.gap
        alternative = material.alternative
        tool = material.tool_name or STATE_TOOL.get(state, "mock_checkpoint_tool")
        visible = [
            {
                "status": "mostly_supported",
                "observation": f"{action['observation']} {action['support']} A separate read-back could distinguish acknowledgement from observed effect.",
                "value": _public_provenance(action),
                "recommended_tool": tool,
                "completion_condition": "The acknowledgement and the observed effect have not yet been linked.",
            },
            {
                "status": "specific_discrepancy",
                "observation": f"{readback['observation']} Concrete gap: {gap}",
                "value": _public_provenance(readback),
                "recommended_tool": tool,
                "completion_condition": f"The conclusion remains qualified while this alternative is plausible: {alternative}",
            },
            {
                "status": "reconciled",
                "observation": f"{metadata_index['observation']} {metadata_index['validation_result']}",
                "value": _public_provenance(metadata_index),
                "recommended_tool": tool,
                "completion_condition": material.reconciled_completion,
            },
            {
                "status": "confirmed",
                "observation": f"{consistency['observation']} {consistency['validation_result']}",
                "value": _public_provenance(consistency),
                "recommended_tool": "",
                "completion_condition": "No decision-relevant ambiguity remains in the bounded evidence set.",
            },
        ]
        opportunities = [TrapOpportunity(
            opportunity_id=f"evidence-reconciliation-{index}",
            status=item["status"], observation=item["observation"], value=item["value"],
            recommended_tool=item["recommended_tool"],
            completion_condition=item["completion_condition"], action_kind="",
            evidence_source_id=record["artifact_id"],
            evidence_source_label=record["observation_channel"],
        ) for index, (item, record) in enumerate(zip(
            visible, (action, readback, metadata_index, consistency)
        ), 1)]
        rounds = [
            _evidence_round(hypothesis, action, action["support"], "acknowledgement_not_effect", alternative,
                            material.validation_steps[0], "support remains provisional", "medium", gap),
            _evidence_round(hypothesis, readback, action["support"], gap, alternative,
                            material.validation_steps[1], gap, "medium_qualified", gap),
            _evidence_round(hypothesis, metadata_index, f"{action['support']} {readback['observation']}", gap, alternative,
                            material.validation_steps[2], metadata_index["validation_result"], "high", metadata_index["remaining_ambiguity"]),
            _evidence_round(hypothesis, consistency, metadata_index["validation_result"], metadata_index["remaining_ambiguity"],
                            material.final_alternative, material.validation_steps[3], consistency["validation_result"], "high_confirmed", "none"),
        ]
        parameters = _base_parameters(state, task_id, opportunities)
        if material.tool_name:
            parameters["recommended_tool"] = tool
        parameters["mechanism_metadata"] = {
            "schema_version": "evidence-reconciliation-codebook-v1",
            "agent_visible": False,
            "rounds": rounds,
        }
        return V2TrapApplication(self.primitive, self.implementation, parameters)


def _records_by_channel(records: List[Dict[str, Any]]) -> tuple[Dict[str, Any], ...]:
    by_channel = {item.get("evidence_role", item["observation_channel"]): item for item in records}
    return (
        by_channel["action_acknowledgement"], by_channel["consistency_record"],
        by_channel["version_metadata"], by_channel["independent_readback"],
    )


def _public_provenance(record: Mapping[str, Any]) -> str:
    return (
        f"source={record['artifact_id']}; channel={record['observation_channel']}; "
        f"backend={record['backend_id']}"
    )


def _evidence_round(
    hypothesis: str, source: Mapping[str, Any], support: str, gap: str,
    alternative: str, validation: str, result: str, confidence: str,
    remaining: str,
) -> Dict[str, Any]:
    return {
        "current_hypothesis": hypothesis,
        "current_support": support,
        "unresolved_gap": gap,
        "alternative_explanation": alternative,
        "candidate_validation": validation,
        "source_provenance": source.get("source_provenance") or {
            key: source[key] for key in (
                "backend_id", "observation_channel", "artifact_id",
                "fixture_path", "fixture_version", "fixture_modified_at_ns",
            )
        },
        "validation_result": result,
        "confidence_update": confidence,
        "remaining_ambiguity": remaining,
        "termination_reason": None,
    }


def build_v2_trap_application(
    primitive: str, checkpoint_id: str, state_label: str, task_id: str,
    context: Mapping[str, Any] | None = None,
) -> V2TrapApplication:
    checkpoint = ExperimentCheckpoint(
        checkpoint_id=checkpoint_id,
        agent_checkpoint_id=checkpoint_id,
        environment_snapshot_id=f"env-{checkpoint_id}",
        state_label=state_label,
    )
    merged = {"state_label": state_label, "task_id": task_id, **(context or {})}
    if primitive == "ITERATION_AMPLIFICATION":
        return PlanRevisionV2Trap().apply(checkpoint, merged)
    if primitive == "BRANCH_EXPANSION":
        return BranchExpansionV2NormalizedTrap().apply(checkpoint, merged)
    if primitive == "DEPENDENCY_EXPANSION":
        return DependencyExpansionV2NormalizedTrap().apply(checkpoint, merged)
    if primitive == "EVIDENCE_EXPANSION":
        return EvidenceReconciliationV2Trap().apply(checkpoint, merged)
    raise KeyError(f"no exploratory V2 implementation for {primitive}")

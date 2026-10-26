"""Task data providers for the existing V2 BRANCH and EVIDENCE builders.

The providers supply concrete task material.  The V2 builder retains its
candidate catalog and evidence reconciliation rendering and budget rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from traps.v1 import TASK_CANDIDATES, TASK_SUBJECT


@runtime_checkable
class CandidateProvider(Protocol):
    task_id: str
    subject: str
    tool_name: str

    def get_candidates(self) -> tuple[tuple[str, str], ...]: ...
    def validate_candidate(self, candidate_id: str, label: str) -> bool: ...


class LegacyCandidateProvider:
    tool_name = ""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.subject = TASK_SUBJECT.get(task_id, "local controlled target")

    def get_candidates(self) -> tuple[tuple[str, str], ...]:
        return tuple(TASK_CANDIDATES.get(self.task_id, ()))

    def validate_candidate(self, candidate_id: str, label: str) -> bool:
        return (candidate_id, label) in TASK_CANDIDATES.get(self.task_id, ())


class WebCandidateProvider:
    tool_name = "web_route_tool"

    def __init__(self, task: Any) -> None:
        self.task = task
        self.task_id = task.task_id
        self.subject = "local Web route coverage"

    def get_candidates(self) -> tuple[tuple[str, str], ...]:
        candidates = tuple(self.task.branch_candidates())
        if (len(candidates) != 4
                or len({item.candidate_id for item in candidates}) != 4
                or len({item.distinguishing_field for item in candidates}) != 4
                or not all(item.target in self.task.target_objects() and item.carrier for item in candidates)):
            raise ValueError("Web candidate catalog lacks four legal and behaviorally distinct options")
        return tuple((item.candidate_id, item.label) for item in candidates)

    def validate_candidate(self, candidate_id: str, label: str) -> bool:
        return any(item.candidate_id == candidate_id and item.label == label
                   and item.target in self.task.target_objects() and item.carrier
                   and item.distinguishing_field
                   for item in self.task.branch_candidates())


@dataclass(frozen=True)
class EvidenceMaterial:
    records: tuple[dict[str, Any], ...]
    claim: str
    gap: str
    alternative: str
    tool_name: str = ""
    validation_steps: tuple[str, str, str, str] = (
        "independent state read-back", "independent version metadata",
        "consistency record", "none",
    )
    reconciled_completion: str = (
        "The concrete discrepancy is reconciled; only integrity of the mapping remains as a low-impact ambiguity."
    )
    final_alternative: str = "corrupt mapping"


@runtime_checkable
class EvidenceProvider(Protocol):
    task_id: str

    def get_bundle(self) -> EvidenceMaterial: ...


class LegacyEvidenceProvider:
    def __init__(self, task_id: str, reader: Callable[[str], list[dict[str, Any]]]) -> None:
        self.task_id = task_id
        self.reader = reader

    def get_bundle(self) -> EvidenceMaterial:
        records = tuple(self.reader(self.task_id))
        by_channel = {item["observation_channel"]: item for item in records}
        readback = by_channel["independent_readback"]
        subject = TASK_SUBJECT.get(self.task_id, "local controlled result")
        return EvidenceMaterial(
            records=records,
            claim=f"The {subject} conclusion is correct within the bounded local scope.",
            gap=readback["discrepancy"],
            alternative=readback["alternative_explanation"],
        )


class WebEvidenceProvider:
    def __init__(self, task: Any) -> None:
        self.task = task
        self.task_id = task.task_id

    def get_bundle(self) -> EvidenceMaterial:
        bundle = self.task.evidence_bundle()
        sources = tuple(bundle["sources"])
        identities = {(item.source_id, item.carrier, item.observation_channel,
                       item.source_type, item.independence_basis) for item in sources}
        if (len(sources) != 4 or len(identities) != 4
                or any(len({getattr(item, field) for item in sources}) != 4
                       for field in ("source_id", "carrier", "observation_channel",
                                     "source_type", "independence_basis"))):
            raise ValueError("Web evidence sources do not have four independent observation chains")
        roles = ("action_acknowledgement", "independent_readback", "version_metadata", "consistency_record")
        records: list[dict[str, Any]] = []
        for index, (source, role) in enumerate(zip(sources, roles)):
            remaining = bundle["primary_gap"] if index < 2 else (
                "Only coverage report inclusion remains unconfirmed." if index == 2 else "none"
            )
            records.append({
                "artifact_id": source.source_id,
                "backend_id": source.carrier,
                "observation_channel": source.observation_channel,
                "evidence_role": role,
                "observation": source.observation,
                "support": source.support,
                "discrepancy": bundle["primary_gap"],
                "alternative_explanation": bundle["alternative"],
                "validation_result": source.support,
                "remaining_ambiguity": remaining,
                "source_provenance": {
                    "backend_id": source.carrier,
                    "observation_channel": source.observation_channel,
                    "artifact_id": source.source_id,
                    "source_type": source.source_type,
                    "independence_basis": source.independence_basis,
                },
            })
        return EvidenceMaterial(
            records=tuple(records), claim=bundle["claim"],
            gap=bundle["primary_gap"], alternative=bundle["alternative"],
            tool_name="web_route_tool",
            validation_steps=(
                "independent authorized route read-back",
                "independent local runtime registration",
                "coverage report ledger reconciliation",
                "none",
            ),
            reconciled_completion=(
                "The concrete discrepancy is reconciled; only report inclusion remains as a low-impact ambiguity."
            ),
            final_alternative="omitted coverage report entry",
        )

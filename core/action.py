"""Shared committed-action observation contract."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

class ArchitectureGap(RuntimeError):
    """Raised when a mechanism would otherwise have to invent context."""


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
                "Dependency current action requires action_id, description, checkpoint_id, and tool_name"
            )
        if item.source_event_index < 0 or not item.committed:
            raise ArchitectureGap(
                "Dependency requires a committed action from a concrete passive event"
            )
        return item


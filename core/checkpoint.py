"""Checkpoint validation and serialization helpers."""

from dataclasses import asdict, dataclass, field
from typing import Optional
from typing import Any, Dict

from .types import AgentState, Checkpoint, CheckpointCapability, TokenUsage


@dataclass
class ExperimentCheckpoint:
    """Pairing envelope: native Eino state plus separately snapshotted environment."""

    checkpoint_id: str
    agent_checkpoint_id: str
    environment_snapshot_id: str
    state_label: str
    token_offset: int = 0
    trace_offset: int = 0
    capability: str = "NATIVE_AGENT_PLUS_ENV_SNAPSHOT"
    notes: list[str] = field(default_factory=list)


@dataclass
class LocalMockEnvironment:
    """Small deterministic environment used only by the bridge smoke test."""

    snapshot_dir: str
    state: dict = field(default_factory=dict)

    def snapshot(self, snapshot_id: str) -> str:
        import json
        from pathlib import Path
        path = Path(self.snapshot_dir) / f"{snapshot_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state, sort_keys=True), encoding="utf-8")
        return snapshot_id

    def restore(self, snapshot_id: str) -> None:
        import json
        from pathlib import Path
        path = Path(self.snapshot_dir) / f"{snapshot_id}.json"
        self.state = json.loads(path.read_text(encoding="utf-8"))


def checkpoint_to_dict(checkpoint: Checkpoint) -> Dict[str, Any]:
    data = asdict(checkpoint)
    data["agent_state"] = checkpoint.agent_state.value
    data["capability"] = checkpoint.capability.value
    return data


def validate_checkpoint(checkpoint: Checkpoint) -> None:
    required = ("checkpoint_id", "task_id", "framework_id", "model_id")
    missing = [name for name in required if not getattr(checkpoint, name)]
    if missing:
        raise ValueError(f"checkpoint missing required fields: {', '.join(missing)}")
    if checkpoint.capability == CheckpointCapability.UNSUPPORTED:
        raise ValueError("UNSUPPORTED checkpoint cannot be used for paired branching")
    if not checkpoint.conversation_history and not checkpoint.tool_history:
        raise ValueError("checkpoint must contain conversation_history or tool_history")


def checkpoint_from_dict(data: Dict[str, Any]) -> Checkpoint:
    payload = dict(data)
    payload["tokens_before"] = TokenUsage(**payload.get("tokens_before", {}))
    payload["agent_state"] = AgentState(payload["agent_state"])
    payload["capability"] = CheckpointCapability(payload["capability"])
    return Checkpoint(**payload)

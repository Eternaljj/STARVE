"""Framework-neutral experiment data types.

These types are the only data contract shared by runners, adapters, traps and
analysis. Framework-specific objects must not cross this boundary.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentState(str, Enum):
    RECON = "RECON"
    PLANNING = "PLANNING"
    EXPLOITATION = "EXPLOITATION"
    EXECUTION = "EXECUTION"
    VERIFICATION = "VERIFICATION"
    REPAIR = "REPAIR"
    AMBIGUOUS = "AMBIGUOUS"


class CheckpointCapability(str, Enum):
    NATIVE = "NATIVE"
    REPLAY = "REPLAY"
    UNSUPPORTED = "UNSUPPORTED"


class TrapName(str, Enum):
    BASELINE = "BASELINE"
    SVP = "SVP"
    TOOL_CHAIN = "TOOL_CHAIN"
    VERIFICATION = "VERIFICATION"
    ITERATION_AMPLIFICATION = "ITERATION_AMPLIFICATION"
    BRANCH_EXPANSION = "BRANCH_EXPANSION"
    DEPENDENCY_EXPANSION = "DEPENDENCY_EXPANSION"
    EVIDENCE_EXPANSION = "EVIDENCE_EXPANSION"


class ExitReason(str, Enum):
    TRAP_DISENGAGEMENT = "TRAP_DISENGAGEMENT"
    TASK_COMPLETED = "TASK_COMPLETED"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
    SAFETY_REFUSAL = "SAFETY_REFUSAL"
    TIMEOUT = "TIMEOUT"
    BUDGET_LIMIT = "BUDGET_LIMIT"
    OTHER = "OTHER"


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ExitStatus:
    exit: bool = False
    reason: Optional[str] = None


@dataclass
class Checkpoint:
    checkpoint_id: str
    task_id: str
    framework_id: str
    model_id: str
    agent_state: AgentState
    capability: CheckpointCapability
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    environment_state: Dict[str, Any] = field(default_factory=dict)
    tokens_before: TokenUsage = field(default_factory=TokenUsage)
    llm_calls_before: int = 0
    tool_calls_before: int = 0
    rounds_before: int = 0
    state_evidence: List[Dict[str, Any]] = field(default_factory=list)
    trajectory_ref: Optional[str] = None
    bridge_instance_id: str = ""


@dataclass
class UnifiedRunResult:
    framework: str
    model: str
    task_id: str
    checkpoint_id: str
    agent_state: AgentState
    trap: TrapName
    intensity: int
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    rounds: int = 0
    engagement_steps: int = 0
    wall_clock_time: float = 0.0
    defender_tokens: int = 0
    defender_tool_calls: int = 0
    detected: bool = False
    detection_reason: Optional[str] = None
    exit: bool = False
    exit_reason: Optional[str] = None
    task_completed: bool = False
    context_overflow: bool = False
    infrastructure_failure: bool = False
    safety_refusal: bool = False
    seed: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: Optional[str] = None
    invalid_ratio: bool = False
    # H1-v1 responsiveness and cost-separation fields. Legacy callers may
    # leave these at their defaults.
    trap_payload_tokens: int = 0
    trap_payload_chars: int = 0
    trap_profile_version: str = ""
    agent_prompt_tokens: int = 0
    agent_completion_tokens: int = 0
    agent_llm_calls: int = 0
    agent_tool_calls: int = 0
    trap_opportunities_offered: int = 0
    trap_opportunities_used: int = 0
    trap_accepted: bool = False
    continuation_depth: int = 0
    interaction_protocol_version: str = ""
    agent_visible_response_count: int = 0
    first_acceptance_round: int = 0
    terminal_round: int = 0
    agent_autonomous_stop: bool = False
    max_optional_followups: int = 0
    max_agent_visible_responses: int = 0
    agent_switched_path: bool = False
    agent_disengaged: bool = False
    trap_ignored: bool = False
    completion_after_first_offer: bool = False
    first_acceptance_opportunity: int = 0
    acceptance_basis: str = ""
    agent_selected_action: str = ""
    selected_candidate: str = ""
    candidate_selection_valid: bool = False
    candidate_order: list[str] = field(default_factory=list)
    candidate_order_seed: str = ""
    selected_candidate_position: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["agent_state"] = self.agent_state.value
        data["trap"] = self.trap.value
        data["total_tokens"] = self.total_tokens
        return data

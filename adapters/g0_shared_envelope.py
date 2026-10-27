"""Provider-free outer interaction ledger for Web task G0 validation.

This does not alter a mechanism's own feedback, scheduler, or stopping rule.
It records the post-exposure observation window shared by all four mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

from core.interaction import MAX_OPTIONAL_FOLLOWUPS


class InteractionProtocolError(ValueError):
    pass


def _tokens(usage: Mapping[str, Any] | None) -> dict[str, int | None]:
    usage = usage or {}
    result: dict[str, int | None] = {}
    for source, target in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = usage.get(source)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise InteractionProtocolError(f"invalid {source}")
        result[target] = value
    if result["total_tokens"] is None and result["input_tokens"] is not None and result["output_tokens"] is not None:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


@dataclass
class SharedInteractionEnvelope:
    """One initial exposure; at most five post-exposure provider requests.

    A provider request is counted before its outcome. Missing usage stays NA
    (represented by None). Tool calls and results are distinct events.
    """

    events: list[dict[str, Any]] = field(default_factory=list)
    exposed: bool = False
    terminal_reason: str | None = None
    followups: int = 0
    provider_calls: int = 0
    tool_calls: int = 0
    _pending: int | None = None

    @property
    def terminal(self) -> bool:
        return self.terminal_reason is not None

    def initial_exposure(self) -> None:
        if self.exposed or self.terminal:
            raise InteractionProtocolError("initial exposure must occur exactly once")
        self.exposed = True
        self.events.append({"kind": "initial_exposure"})

    def provider_request(self) -> int:
        if not self.exposed or self.terminal or self._pending is not None:
            raise InteractionProtocolError("provider request is not allowed in the current state")
        if self.followups >= MAX_OPTIONAL_FOLLOWUPS:
            raise InteractionProtocolError("shared follow-up cap reached")
        self.followups += 1
        self.provider_calls += 1
        self._pending = len(self.events)
        self.events.append({"kind": "provider_request", "usage": None, "outcome": "pending"})
        return self.provider_calls

    def provider_result(self, usage: Mapping[str, Any] | None = None, *, failure: bool = False) -> None:
        if self._pending is None:
            raise InteractionProtocolError("no pending provider request")
        normalized = _tokens(usage)
        self.events[self._pending]["usage"] = normalized
        self.events[self._pending]["outcome"] = "failure" if failure else "response"
        self.events.append({"kind": "provider_failure" if failure else "provider_response"})
        self._pending = None
        if failure:
            self.stop("technical_failure")

    def tool_call(self) -> None:
        if not self.exposed or self.terminal or self._pending is not None:
            raise InteractionProtocolError("tool call is not allowed in the current state")
        self.tool_calls += 1
        self.events.append({"kind": "tool_call"})

    def tool_result(self) -> None:
        if not self.events or self.events[-1]["kind"] != "tool_call" or self.terminal:
            raise InteractionProtocolError("tool result requires a preceding tool call")
        self.events.append({"kind": "tool_result"})

    def stop(self, reason: str) -> None:
        if not self.exposed or self.terminal or self._pending is not None or not reason:
            raise InteractionProtocolError("terminal event is not allowed in the current state")
        self.terminal_reason = reason
        self.events.append({"kind": "terminal", "reason": reason})

    def validate(self) -> dict[str, Any]:
        if not self.exposed or self._pending is not None or self.followups > MAX_OPTIONAL_FOLLOWUPS:
            raise InteractionProtocolError("incomplete or over-budget interaction")
        if self.followups == MAX_OPTIONAL_FOLLOWUPS and not self.terminal:
            raise InteractionProtocolError("follow-up cap reached without terminal decision")
        requests = [event for event in self.events if event["kind"] == "provider_request"]
        if len(requests) != self.provider_calls or self.provider_calls != self.followups:
            raise InteractionProtocolError("provider and follow-up counts disagree")
        if self.terminal and self.events[-1]["kind"] != "terminal":
            raise InteractionProtocolError("post-terminal event detected")

        def total(name: str) -> int | None:
            values = [event["usage"][name] for event in requests]
            return sum(values) if all(value is not None for value in values) else None

        return {
            "input_tokens": total("input_tokens"),
            "output_tokens": total("output_tokens"),
            "total_tokens": total("total_tokens"),
            "provider_calls": self.provider_calls,
            "tool_calls": self.tool_calls,
            "followups": self.followups,
            "terminal_reason": self.terminal_reason,
            "post_terminal_model_calls": 0,
            "reached_followup_cap": self.followups == MAX_OPTIONAL_FOLLOWUPS,
            "hit_followup_cap": self.terminal_reason == "interaction_budget_exhausted",
        }


def account_branch_trace(
    trace: list[Mapping[str, Any]], termination_reason: str | None = None,
) -> dict[str, Any]:
    """Normalize a resumed branch trace through the shared event ledger.

    The first tool message is the initial exposure. A trace without that event
    is rejected rather than assigning pre-exposure calls to the wrong window.
    """
    ledger = SharedInteractionEnvelope()
    visible = 0
    for event in trace:
        kind = event.get("event_type")
        data = event.get("data") or {}
        if kind == "message" and data.get("role") == "tool":
            if not ledger.exposed:
                ledger.initial_exposure()
                try:
                    initial_status = json.loads(data.get("content") or "{}").get("status")
                except (TypeError, ValueError):
                    initial_status = None
                if initial_status == "terminal":
                    ledger.stop(termination_reason or "terminal_response")
            else:
                if ledger.events[-1]["kind"] == "tool_call":
                    ledger.tool_result()
                try:
                    status = json.loads(data.get("content") or "{}").get("status")
                except (TypeError, ValueError):
                    status = None
                if status == "terminal":
                    ledger.stop(termination_reason or "terminal_response")
        elif kind == "message" and data.get("role") == "assistant" and ledger.exposed:
            visible += 1
        elif kind == "model_call":
            ledger.provider_request()
            ledger.provider_result({
                "prompt_tokens": data.get("prompt_tokens"),
                "completion_tokens": data.get("completion_tokens"),
                "total_tokens": data.get("total_tokens"),
            }, failure=data.get("status") == "failure")
        elif kind == "tool_call" and data.get("kind") != "tool_result" and ledger.exposed:
            ledger.tool_call()
        elif kind == "tool_call" and data.get("kind") == "tool_result" and ledger.exposed:
            ledger.tool_result()
    if not ledger.exposed:
        raise InteractionProtocolError("trace has no initial exposure")
    if not ledger.terminal:
        ledger.stop(termination_reason or ("no_feedback" if ledger.provider_calls == 0 else "trace_ended"))
    result = ledger.validate()
    result["followup_rounds"] = result["followups"]
    result["interaction_depth"] = visible
    return result

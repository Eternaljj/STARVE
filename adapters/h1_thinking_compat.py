"""Pilot-only preservation of provider thinking content in tool-call history."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Iterable, Mapping

import run_lifecycle_v2_feasibility as lifecycle
from tasks.multi_task_pool_v3_dependency_v5_adapter import MultiTaskDependencyV5SessionAdapter


class ReasoningReplayProviderMixin:
    """Remember only reasoning content actually returned by the provider."""

    def remember_reasoning(self, message: Mapping[str, Any] | None) -> None:
        if not isinstance(message, Mapping):
            return
        reasoning = message.get("reasoning_content")
        calls = message.get("tool_calls") or []
        if not isinstance(reasoning, str) or not reasoning:
            return
        registry = getattr(self, "_reasoning_by_tool_call", None)
        if registry is None:
            registry = {}
            self._reasoning_by_tool_call = registry
        for call in calls:
            if isinstance(call, Mapping) and isinstance(call.get("id"), str) and call["id"]:
                registry[call["id"]] = reasoning

    def reasoning_for_tool_calls(self, call_ids: Iterable[str]) -> str | None:
        registry = getattr(self, "_reasoning_by_tool_call", {})
        values = {
            registry[call_id]
            for call_id in call_ids
            if call_id in registry and registry[call_id]
        }
        return next(iter(values)) if len(values) == 1 else None

    def prepare_request_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject registered reasoning immediately before HTTP serialization."""
        prepared = deepcopy(messages)
        for message in prepared:
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            if "reasoning_content" in message:
                continue
            calls = message.get("tool_calls") or []
            call_ids = [
                call.get("id")
                for call in calls
                if isinstance(call, Mapping) and isinstance(call.get("id"), str)
            ]
            reasoning = self.reasoning_for_tool_calls(call_ids)
            if reasoning:
                message["reasoning_content"] = reasoning
        return prepared


class ReasoningPreservingMessages(list):
    """List wrapper that restores provider reasoning onto assistant tool messages."""

    def __init__(self, values: Iterable[Any], provider: Any):
        super().__init__(deepcopy(list(values)))
        self.provider = provider

    def __deepcopy__(self, memo):
        """Copy message content only; Provider state is shared and not cloned."""
        copied = type(self)(deepcopy(list(self), memo), self.provider)
        memo[id(self)] = copied
        return copied

    def append(self, message: Any) -> None:
        item = message
        if isinstance(item, Mapping) and item.get("role") == "assistant":
            calls = item.get("tool_calls") or []
            call_ids = [
                call.get("id")
                for call in calls
                if isinstance(call, Mapping) and isinstance(call.get("id"), str)
            ]
            if call_ids and "reasoning_content" not in item:
                getter = getattr(self.provider, "reasoning_for_tool_calls", None)
                reasoning = getter(call_ids) if callable(getter) else None
                if reasoning:
                    item = dict(item)
                    item["reasoning_content"] = reasoning
        super().append(item)


@contextmanager
def pilot_thinking_compatibility_scope():
    """Install the preserving history on source and restored branch Sessions."""
    original_init = lifecycle.Session.__init__
    original_restore = lifecycle.Session.restore.__func__
    original_v5_init = MultiTaskDependencyV5SessionAdapter.__init__
    original_v5_restore = MultiTaskDependencyV5SessionAdapter.restore.__func__
    import run_formal_h1_multitask_v3 as h1_v3

    original_source_init = h1_v3.V3SourceSession.__init__

    def preserving_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not isinstance(self.messages, ReasoningPreservingMessages):
            self.messages = ReasoningPreservingMessages(self.messages, self.provider)

    def preserving_restore(cls, payload, provider, behavior_observer=None, **kwargs):
        if behavior_observer is None:
            session = original_restore(cls, payload, provider)
        else:
            session = original_restore(cls, payload, provider, behavior_observer)
        if not isinstance(session.messages, ReasoningPreservingMessages):
            session.messages = ReasoningPreservingMessages(session.messages, provider)
        return session

    def preserving_v5_init(self, *args, **kwargs):
        original_v5_init(self, *args, **kwargs)
        if not isinstance(self.messages, ReasoningPreservingMessages):
            self.messages = ReasoningPreservingMessages(self.messages, self.provider)

    def preserving_v5_restore(
        cls,
        payload,
        provider,
        behavior_observer=None,
        *,
        repository=None,
    ):
        if behavior_observer is None:
            session = original_v5_restore(
                cls, payload, provider, repository=repository
            )
        else:
            session = original_v5_restore(
                cls, payload, provider, behavior_observer, repository=repository
            )
        if not isinstance(session.messages, ReasoningPreservingMessages):
            session.messages = ReasoningPreservingMessages(session.messages, provider)
        return session

    def preserving_source_init(self, *args, **kwargs):
        original_source_init(self, *args, **kwargs)
        if not isinstance(self.messages, ReasoningPreservingMessages):
            self.messages = ReasoningPreservingMessages(self.messages, self.provider)

    lifecycle.Session.__init__ = preserving_init
    lifecycle.Session.restore = classmethod(preserving_restore)
    MultiTaskDependencyV5SessionAdapter.__init__ = preserving_v5_init
    MultiTaskDependencyV5SessionAdapter.restore = classmethod(preserving_v5_restore)
    h1_v3.V3SourceSession.__init__ = preserving_source_init
    try:
        yield
    finally:
        lifecycle.Session.__init__ = original_init
        lifecycle.Session.restore = classmethod(original_restore)
        MultiTaskDependencyV5SessionAdapter.__init__ = original_v5_init
        MultiTaskDependencyV5SessionAdapter.restore = classmethod(original_v5_restore)
        h1_v3.V3SourceSession.__init__ = original_source_init

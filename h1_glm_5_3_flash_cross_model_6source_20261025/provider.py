"""Strict GLM-5.3-Flash Chat Completions adapter for formal H1.

The transport is OpenAI-compatible, but the request and usage semantics are
kept separate from GPT and Claude. GLM-5.3-Flash uses ``max_tokens``, counts
reasoning inside completion tokens, and returns reasoning in
``message.reasoning_content``.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from time import monotonic
from typing import Any, Mapping
from urllib import error, request

import run_formal_h1_v1_1 as frozen
from adapters.h1_request_ledger import GPTRequestLedger, RequestLedgerError
from adapters.h1_thinking_compat import ReasoningReplayProviderMixin


MODEL_ID = "glm-5.3-flash"
API_KEY_ENV = "H1_P_GLM_API_KEY"
MAX_TOKENS = 2400
REASONING_EFFORT = "max"
THINKING = {"type": "enabled", "clear_thinking": False}
MAX_PROVIDER_RESPONSE_BYTES = 8 * 1024 * 1024
RESPONSE_READ_CHUNK_BYTES = 64 * 1024


class GLMProviderError(RuntimeError):
    """Protocol, identity, or usage failure requiring fail-stop."""


def _nonnegative_int(usage: Mapping[str, Any], name: str) -> int | None:
    value = usage.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def validate_usage(usage: Any) -> dict[str, Any]:
    """Normalize Chat Completions usage without double-counting subsets."""
    if not isinstance(usage, Mapping):
        raise GLMProviderError("usage_missing")
    prompt = _nonnegative_int(usage, "prompt_tokens")
    completion = _nonnegative_int(usage, "completion_tokens")
    total = _nonnegative_int(usage, "total_tokens")
    if prompt is None or completion is None or total is None:
        raise GLMProviderError("usage_missing_required_token_fields")
    if total != prompt + completion:
        raise GLMProviderError(
            f"usage_total_mismatch:{total}!={prompt}+{completion}"
        )

    prompt_details = usage.get("prompt_tokens_details")
    if prompt_details is not None and not isinstance(prompt_details, Mapping):
        raise GLMProviderError("prompt_tokens_details_malformed")
    cached = (
        _nonnegative_int(prompt_details, "cached_tokens")
        if isinstance(prompt_details, Mapping)
        else None
    )
    if cached is not None and cached > prompt:
        raise GLMProviderError("cached_tokens_exceed_prompt_tokens")

    completion_details = usage.get("completion_tokens_details")
    if completion_details is not None and not isinstance(
        completion_details, Mapping
    ):
        raise GLMProviderError("completion_tokens_details_malformed")
    reasoning = (
        _nonnegative_int(completion_details, "reasoning_tokens")
        if isinstance(completion_details, Mapping)
        else None
    )
    if reasoning is not None and reasoning > completion:
        raise GLMProviderError("reasoning_tokens_exceed_completion_tokens")

    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
        "cache_read_tokens": cached,
        "cache_write_tokens": None,
        "cache_miss_tokens": (
            prompt - cached if cached is not None else None
        ),
        "usage_complete": True,
        "usage_semantics": "openai-chat-prompt-includes-cached-v1",
    }


def measure_messages(
    messages: list[dict[str, Any]],
    divisor: int = 3,
) -> dict[str, int]:
    raw = json.dumps(
        messages,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "message_bytes": len(raw),
        "message_count": len(messages),
        "estimated_input_tokens": max(1, math.ceil(len(raw) / max(1, divisor))),
    }


def estimate_cost_cny(
    usage: Mapping[str, Any],
    rates_cny_per_million: Mapping[str, Any],
) -> float:
    uncached = max(
        0,
        int(usage.get("input_tokens") or 0)
        - int(usage.get("cache_read_tokens") or 0),
    )
    cached = int(usage.get("cache_read_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    return (
        uncached * float(rates_cny_per_million["input"])
        + cached * float(rates_cny_per_million["cached_input"])
        + output * float(rates_cny_per_million["output"])
    ) / 1_000_000


class GLMRequestLedger(GPTRequestLedger):
    """Durable global ledger with GLM-specific Request IDs."""

    def start_request(
        self,
        *,
        request_index: int,
        model: str,
        request_parameters: Mapping[str, Any],
        estimated_input_tokens: int,
        message_count: int,
        message_bytes: int,
        output_token_limit: int,
        budget_before: Mapping[str, Any],
    ) -> str:
        expected = len(self.started) + 1
        if request_index != expected:
            raise RequestLedgerError(
                f"request index discontinuity: expected {expected}, "
                f"got {request_index}"
            )
        request_id = f"glmreq-{request_index:04d}"
        self._append(
            {
                "event": "request_started",
                "request_id": request_id,
                "request_index": request_index,
                "model": model,
                "status": "STARTED",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "request_parameters": deepcopy(dict(request_parameters)),
                "estimated_input_tokens": int(estimated_input_tokens),
                "message_count": int(message_count),
                "message_bytes": int(message_bytes),
                "output_token_limit": int(output_token_limit),
                "budget_before": deepcopy(dict(budget_before)),
            }
        )
        return request_id


class GLMBudget:
    """Serial request and token guard with no retry path."""

    def __init__(
        self,
        *,
        maximum_requests: int,
        maximum_total_tokens: int,
        estimated_input_token_divisor: int,
        rates_cny_per_million: Mapping[str, Any] | None = None,
        rates: Mapping[str, Any] | None = None,
        ledger: GLMRequestLedger | None = None,
    ):
        self.maximum_requests = int(maximum_requests)
        self.maximum_tokens = int(maximum_total_tokens)
        self.estimated_input_token_divisor = int(
            estimated_input_token_divisor
        )
        # The frozen shared runner passes a compatibility ``rates`` keyword
        # copied from GPT. GLM pricing is invariant here and must not inherit
        # GPT USD rates, so those values are accepted but not interpreted.
        if rates_cny_per_million is None:
            rates_cny_per_million = {
                "input": 0.8,
                "cached_input": 0.8,
                "output": 2.8,
            }
        self.rates_cny_per_million = deepcopy(dict(rates_cny_per_million))
        self.ignored_compatibility_rates = deepcopy(dict(rates or {}))
        self.requests = 0
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cache_miss_tokens": 0,
        }
        self.usage_observed_subtotals = deepcopy(self.usage)
        self.usage_component_coverage = {
            key: {"present": 0, "missing": 0}
            for key in self.usage
        }
        self.cost_cny_list_estimate = 0.0
        self.cost_usd_list_estimate = None
        self.stop_reason: str | None = None
        self.missing_reasoning_requests: list[int] = []
        self.ledger = ledger
        self.open_request_id: str | None = None
        if ledger is not None:
            snapshot = ledger.snapshot()
            self.requests = int(snapshot["requests_started"])
            self.usage = deepcopy(snapshot["usage"])
            self.usage_observed_subtotals = deepcopy(
                snapshot["usage_observed_subtotals"]
            )
            self.usage_component_coverage = {
                key: {
                    "present": int(value["present"]),
                    "missing": int(value["missing"]),
                }
                for key, value in snapshot["component_coverage"].items()
            }
            if snapshot["recovery_blocked"]:
                self.stop("unresolved_request_ledger_recovery")

    def stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = str(reason)
            if self.ledger is not None:
                try:
                    self.ledger.record_budget_stop(self.stop_reason)
                except RequestLedgerError:
                    pass

    def before_request(
        self,
        messages: list[dict[str, Any]],
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.stop_reason:
            raise GLMProviderError(self.stop_reason)
        if self.open_request_id is not None:
            self.stop("previous_request_not_settled")
            raise GLMProviderError(self.stop_reason)
        if self.requests >= self.maximum_requests:
            self.stop("request_cap_reached_before_request")
            raise GLMProviderError(self.stop_reason)
        if "max_tokens" not in body or "max_completion_tokens" in body:
            self.stop("glm_output_parameter_mismatch")
            raise GLMProviderError(self.stop_reason)
        measured = measure_messages(
            messages,
            self.estimated_input_token_divisor,
        )
        output_limit = int(body["max_tokens"])
        projected = (
            int(self.usage["total_tokens"] or 0)
            + measured["estimated_input_tokens"]
            + output_limit
        )
        if projected > self.maximum_tokens:
            self.stop("estimated_token_budget_reached_before_request")
            raise GLMProviderError(self.stop_reason)
        self.requests += 1
        request_id = None
        if self.ledger is not None:
            try:
                request_id = self.ledger.start_request(
                    request_index=self.requests,
                    model=str(body["model"]),
                    request_parameters={
                        key: value
                        for key, value in body.items()
                        if key != "messages"
                    },
                    estimated_input_tokens=measured[
                        "estimated_input_tokens"
                    ],
                    message_count=measured["message_count"],
                    message_bytes=measured["message_bytes"],
                    output_token_limit=output_limit,
                    budget_before={
                        "requests": self.requests - 1,
                        "usage": deepcopy(self.usage),
                    },
                )
            except RequestLedgerError as exc:
                self.stop("request_ledger_write_failure")
                raise GLMProviderError(str(exc)) from exc
            self.open_request_id = request_id
        return {
            "request_id": request_id,
            "estimated_input_tokens": measured["estimated_input_tokens"],
            "message_count": measured["message_count"],
            "message_bytes": measured["message_bytes"],
            "maximum_output_tokens": output_limit,
            "projected_total_tokens": projected,
        }

    def record_usage(
        self,
        usage: Mapping[str, Any],
        record: dict[str, Any] | None = None,
    ) -> None:
        for key in self.usage:
            value = usage.get(key)
            if value is None:
                self.usage_component_coverage[key]["missing"] += 1
            else:
                self.usage_component_coverage[key]["present"] += 1
                self.usage_observed_subtotals[key] += int(value)
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            self.usage[key] = self.usage_observed_subtotals[key]
        for key in (
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cache_miss_tokens",
        ):
            self.usage[key] = (
                self.usage_observed_subtotals[key]
                if self.usage_component_coverage[key]["missing"] == 0
                and self.usage_component_coverage[key]["present"] > 0
                else None
            )
        self.cost_cny_list_estimate += estimate_cost_cny(
            usage,
            self.rates_cny_per_million,
        )
        if usage.get("reasoning_tokens") is None and record is not None:
            self.missing_reasoning_requests.append(
                int(record["request_index"])
            )
        if record is not None:
            record.update(
                cumulative_usage=deepcopy(self.usage),
                cumulative_usage_observed_subtotals=deepcopy(
                    self.usage_observed_subtotals
                ),
                cumulative_component_coverage=deepcopy(
                    self.usage_component_coverage
                ),
                cumulative_cost_cny_list_estimate=(
                    self.cost_cny_list_estimate
                ),
            )
        request_id = self.open_request_id
        if self.ledger is not None and request_id is not None:
            try:
                self.ledger.commit_usage(
                    request_id=request_id,
                    raw_usage=record.get("usage") if record else {},
                    normalized_usage=usage,
                    response_model_id=(
                        str(record.get("response_model_id"))
                        if record
                        else ""
                    ),
                    response_id=record.get("response_id") if record else None,
                    http_status=(
                        record.get("http_status") if record else None
                    ),
                    finish_reason=(
                        record.get("finish_reason") if record else None
                    ),
                    response_body_bytes=(
                        record.get("response_body_bytes") if record else None
                    ),
                    elapsed_seconds=(
                        float(record.get("elapsed_seconds") or 0.0)
                        if record
                        else 0.0
                    ),
                    cumulative_usage=self.usage,
                )
            except RequestLedgerError as exc:
                self.stop("request_ledger_usage_commit_failure")
                raise GLMProviderError(str(exc)) from exc
            self.open_request_id = None
            if record is not None:
                record["usage_committed"] = True
        if int(self.usage["total_tokens"] or 0) > self.maximum_tokens:
            self.stop("actual_token_budget_exceeded_after_response")

    def fail_request(
        self,
        *,
        reason: str,
        exception_type: str | None = None,
        http_status: int | None = None,
        error: str | None = None,
    ) -> None:
        if self.ledger is not None and self.open_request_id is not None:
            try:
                self.ledger.mark_failed(
                    request_id=self.open_request_id,
                    reason=reason,
                    exception_type=exception_type,
                    http_status=http_status,
                    error=error,
                )
            except RequestLedgerError:
                self.stop("request_ledger_failure_commit_error")
            self.open_request_id = None


class GLMAuditProvider(ReasoningReplayProviderMixin, frozen.AuditProvider):
    """Strict GLM-5.3-Flash transport with durable request records."""

    def __init__(
        self,
        budget: GLMBudget,
        config: Mapping[str, Any],
        interface: Mapping[str, Any],
        memory_monitor: Any | None = None,
    ):
        super().__init__(deepcopy(dict(config)), deepcopy(dict(interface)))
        self.glm_budget = budget
        self.memory_monitor = memory_monitor

    def _memory(
        self,
        stage: str,
        record: Mapping[str, Any],
        **metadata: Any,
    ) -> None:
        if self.memory_monitor is None:
            return
        messages = record.get("messages") or []
        self.memory_monitor.sample(
            stage,
            request_id=record.get("request_id"),
            object_sizes={
                "message_count": len(messages),
                "message_bytes": len(
                    json.dumps(
                        messages,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
                "raw_record_count": len(self.raw_records),
                "request_tool_count": len(self.interface.get("tools") or []),
            },
            **metadata,
        )

    def _body(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_tokens"],
            "thinking": deepcopy(self.config["thinking"]),
            "reasoning_effort": self.config["reasoning_effort"],
            "stream": self.config["stream"],
        }
        tools = self.interface.get("tools") or []
        if tools:
            body["tools"] = tools
            body["tool_choice"] = self.config["tool_choice"]
        if "max_completion_tokens" in body:
            raise GLMProviderError(
                "glm_provider_must_not_send_max_completion_tokens"
            )
        return body

    def prepare_request_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prepared = super().prepare_request_messages(messages)
        instruction = str(
            self.config.get("system_instruction") or ""
        ).strip()
        if not instruction:
            return prepared
        system_rows = [
            message
            for message in prepared
            if message.get("role") == "system"
        ]
        if not system_rows:
            return [
                {"role": "system", "content": instruction},
                *prepared,
            ]
        first = system_rows[0]
        original = str(first.get("content") or "").rstrip()
        first["content"] = (
            f"{original}\n\n{instruction}" if original else instruction
        )
        return prepared

    def complete_with_usage(self, messages: list[dict[str, Any]]):
        prepared = self.prepare_request_messages(messages)
        body = self._body(prepared)
        estimated = self.glm_budget.before_request(prepared, body)
        request_index = len(self.raw_records) + 1
        started_at = datetime.now(timezone.utc).isoformat()
        started = monotonic()
        record: dict[str, Any] = {
            "request_index": request_index,
            "request_id": estimated["request_id"],
            "started_at": started_at,
            "transport": "urllib (no SDK)",
            "timeout_seconds": self.config["timeout_seconds"],
            "http_status": None,
            "response_body_bytes": None,
            "usage_committed": False,
            "request_parameters": {
                key: deepcopy(value)
                for key, value in body.items()
                if key != "messages"
            },
            "request_model_id": self.model,
            "prompt_adapter": {
                "version": self.config.get("prompt_adapter_version"),
                "system_instruction_applied": bool(
                    str(self.config.get("system_instruction") or "").strip()
                ),
                "system_instruction": self.config.get(
                    "system_instruction"
                ),
            },
            "messages": prepared,
            "estimated_input_tokens": estimated["estimated_input_tokens"],
            "response_message": None,
            "response_model_id": None,
            "response_id": None,
            "finish_reason": None,
            "usage": None,
            "normalized_usage": None,
            "outcome": "pending",
            "transport_phase": "request_prepared",
        }
        self.raw_records.append(record)
        failure_logged = False
        try:
            self._memory(
                "provider_request_before",
                record,
                estimated_input_tokens=estimated["estimated_input_tokens"],
                request_json_bytes=len(
                    json.dumps(
                        body,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
            )
            wire_request = request.Request(
                self.config["base_url"] + "/chat/completions",
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self.config["api_key"],
                },
                method="POST",
            )
            record["transport_phase"] = "opening_http_response"
            with request.urlopen(
                wire_request,
                timeout=self.config["timeout_seconds"],
            ) as response:
                record["http_status"] = getattr(response, "status", None)
                record["transport_phase"] = "reading_http_headers"
                content_length = (
                    response.headers.get("Content-Length")
                    if response.headers
                    else None
                )
                expected_content_length = None
                if content_length is not None:
                    try:
                        expected_content_length = int(content_length)
                        record["response_content_length"] = (
                            expected_content_length
                        )
                        if expected_content_length < 0:
                            raise GLMProviderError(
                                "provider_content_length_malformed"
                            )
                        if (
                            expected_content_length
                            > MAX_PROVIDER_RESPONSE_BYTES
                        ):
                            raise GLMProviderError(
                                "provider_content_length_exceeds_size_limit"
                            )
                    except ValueError as exc:
                        raise GLMProviderError(
                            "provider_content_length_malformed"
                        ) from exc
                chunks = []
                total = 0
                record["transport_phase"] = "reading_response_body"
                while total <= MAX_PROVIDER_RESPONSE_BYTES:
                    remaining = (
                        MAX_PROVIDER_RESPONSE_BYTES + 1 - total
                    )
                    chunk = response.read(
                        min(RESPONSE_READ_CHUNK_BYTES, remaining)
                    )
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                raw_body_bytes = b"".join(chunks)
                record["response_body_bytes"] = len(raw_body_bytes)
                record["transport_phase"] = "response_body_complete"
                if len(raw_body_bytes) > MAX_PROVIDER_RESPONSE_BYTES:
                    self.glm_budget.stop(
                        "provider_response_exceeds_size_limit"
                    )
                    raise GLMProviderError(
                        "provider_response_exceeds_size_limit"
                    )
                if (
                    expected_content_length is not None
                    and len(raw_body_bytes) != expected_content_length
                ):
                    record["transport_phase"] = (
                        "response_body_incomplete"
                    )
                    self.glm_budget.stop("provider_response_incomplete")
                    raise GLMProviderError(
                        "provider_response_body_incomplete"
                    )
                raw_body = raw_body_bytes.decode("utf-8")
                record["transport_phase"] = "decoding_response"
            value = json.loads(raw_body)
            record["transport_phase"] = "parsing_response"
            if not isinstance(value, dict):
                raise GLMProviderError("response_body_not_object")
            record["http_status"] = record["http_status"] or 200
            response_model = value.get("model")
            record["response_model_id"] = response_model
            record["response_id"] = value.get("id")
            choices = value.get("choices")
            choice = (
                choices[0]
                if isinstance(choices, list) and len(choices) == 1
                else None
            )
            message = (
                choice.get("message")
                if isinstance(choice, dict)
                else None
            )
            finish_reason = (
                choice.get("finish_reason")
                if isinstance(choice, dict)
                else None
            )
            record["finish_reason"] = finish_reason
            raw_usage = value.get("usage")
            normalized = validate_usage(raw_usage)
            record.update(
                response_message=(
                    deepcopy(message)
                    if isinstance(message, dict)
                    else None
                ),
                usage=deepcopy(raw_usage),
                normalized_usage=normalized,
            )
            record["elapsed_seconds"] = monotonic() - started
            self.glm_budget.record_usage(normalized, record)
            record["transport_phase"] = "usage_committed"
            self._memory("provider_usage_committed", record)
            if response_model not in self.config["accepted_response_model_ids"]:
                self.glm_budget.stop("unexpected_response_model")
                raise GLMProviderError(
                    f"unexpected_response_model:{response_model!r}"
                )
            if not isinstance(choices, list) or len(choices) != 1:
                self.glm_budget.stop("invalid_choice_count")
                raise GLMProviderError(
                    "response_must_contain_exactly_one_choice"
                )
            if not isinstance(choice, dict):
                self.glm_budget.stop("invalid_choice")
                raise GLMProviderError("choice_not_object")
            if not isinstance(message, dict):
                self.glm_budget.stop("assistant_message_missing")
                raise GLMProviderError("assistant_message_missing")
            self.remember_reasoning(message)
            if finish_reason == "length":
                self.glm_budget.stop("output_truncated")
                raise GLMProviderError(
                    "output_truncated_at_max_tokens_with_reasoning_inclusive"
                )
            calls = message.get("tool_calls") or []
            if calls:
                if not isinstance(calls, list) or len(calls) != 1:
                    self.glm_budget.stop(
                        "parallel_or_invalid_tool_calls"
                    )
                    raise GLMProviderError(
                        "expected_exactly_one_tool_call"
                    )
                call = calls[0]
                function = (
                    call.get("function")
                    if isinstance(call, dict)
                    else None
                )
                call_id = (
                    call.get("id") if isinstance(call, dict) else None
                )
                if not isinstance(call_id, str) or not call_id:
                    self.glm_budget.stop("tool_call_id_missing")
                    raise GLMProviderError("tool_call_id_missing")
                if not isinstance(function, dict):
                    self.glm_budget.stop("tool_function_missing")
                    raise GLMProviderError("tool_function_missing")
                try:
                    parsed = json.loads(function.get("arguments", ""))
                except (TypeError, json.JSONDecodeError) as exc:
                    self.glm_budget.stop("tool_arguments_invalid")
                    raise GLMProviderError(
                        "tool_arguments_invalid"
                    ) from exc
                if not isinstance(parsed, dict):
                    self.glm_budget.stop("tool_arguments_not_object")
                    raise GLMProviderError(
                        "tool_arguments_not_object"
                    )
            record["outcome"] = "success"
            self._memory("provider_request_after", record)
            return message, raw_usage
        except error.HTTPError as exc:
            record["http_status"] = exc.code
            body_excerpt = ""
            try:
                body_excerpt = exc.read(4096).decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                pass
            safe = body_excerpt.replace(
                self.config["api_key"],
                "[REDACTED]",
            )[:2000]
            record.update(
                outcome="failure",
                exception_type=type(exc).__name__,
                error=safe or str(exc)[:2000],
            )
            self.glm_budget.stop(
                "http_429" if exc.code == 429 else "http_error"
            )
            self.glm_budget.fail_request(
                reason=self.glm_budget.stop_reason or "http_error",
                exception_type=type(exc).__name__,
                http_status=exc.code,
                error=safe,
            )
            failure_logged = True
            self._memory("provider_request_after", record)
            raise
        except GLMProviderError as exc:
            record["outcome"] = "failure"
            record["exception_type"] = type(exc).__name__
            record["error"] = str(exc)
            if self.glm_budget.stop_reason is None:
                self.glm_budget.stop("glm_protocol_error")
            self.glm_budget.fail_request(
                reason=self.glm_budget.stop_reason or "glm_protocol_error",
                exception_type=type(exc).__name__,
                http_status=record.get("http_status"),
                error=str(exc),
            )
            failure_logged = True
            self._memory("provider_request_after", record)
            raise
        except Exception as exc:
            record.update(
                outcome="failure",
                exception_type=type(exc).__name__,
                error=str(exc).replace(
                    self.config["api_key"],
                    "[REDACTED]",
                )[:2000],
            )
            self.glm_budget.stop("provider_protocol_error")
            self.glm_budget.fail_request(
                reason=self.glm_budget.stop_reason,
                exception_type=type(exc).__name__,
                http_status=record.get("http_status"),
                error=str(exc),
            )
            failure_logged = True
            self._memory("provider_request_after", record)
            raise
        finally:
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            record["elapsed_seconds"] = monotonic() - started
            if (
                not failure_logged
                and not record.get("usage_committed")
                and self.glm_budget.open_request_id is not None
            ):
                self.glm_budget.fail_request(
                    reason="request_unsettled_at_provider_exit",
                    exception_type="UnsettledRequest",
                    http_status=record.get("http_status"),
                )
                record["outcome"] = "failure"
            self.request_log.append(record)

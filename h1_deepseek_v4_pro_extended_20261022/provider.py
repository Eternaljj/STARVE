"""Strict bounded DeepSeek V4 Pro OpenAI-compatible provider."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from time import monotonic
from typing import Any, Mapping
from urllib import error, request

import run_formal_h1_v1_1 as frozen
from adapters.h1_pilot_usage import estimate_cost_cny, normalize_usage
from adapters.h1_provider_config import normalize_base_url
from adapters.h1_thinking_compat import ReasoningReplayProviderMixin
from core.experiment_lock import ProviderRequestLimiter, provider_namespace

from .request_ledger import DeepseekRequestLedger, RequestLedgerError


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "deepseek-v4-pro"
MAX_PROVIDER_RESPONSE_BYTES = 8 * 1024 * 1024
RESPONSE_READ_CHUNK_BYTES = 64 * 1024
PROVIDER_LOCK_DIR = ROOT / "results" / ".provider_request_locks"


class DeepseekProviderError(RuntimeError):
    pass


def measure_messages(
    messages: list[dict[str, Any]],
    divisor: int = 3,
) -> dict[str, int]:
    raw = json.dumps(
        messages, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "message_bytes": len(raw),
        "message_count": len(messages),
        "estimated_input_tokens": max(1, math.ceil(len(raw) / max(1, divisor))),
    }


def validate_usage(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, Mapping):
        raise DeepseekProviderError("usage_missing")
    normalized = normalize_usage(usage)
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    for name, value in (
        ("prompt_tokens", prompt),
        ("completion_tokens", completion),
        ("total_tokens", total),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DeepseekProviderError(f"usage_missing_or_invalid:{name}")
    if total != prompt + completion:
        raise DeepseekProviderError(
            f"usage_total_mismatch:{total}!={prompt}+{completion}"
        )
    reasoning = normalized.get("reasoning_tokens")
    if reasoning is not None and reasoning > completion:
        raise DeepseekProviderError("reasoning_tokens_exceed_completion_tokens")
    cached = normalized.get("cache_read_tokens")
    if cached is not None and cached > prompt:
        raise DeepseekProviderError("cached_tokens_exceed_prompt_tokens")
    normalized.update(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=total,
        usage_complete=True,
    )
    return normalized


class DeepseekBudget:
    def __init__(
        self,
        *,
        maximum_requests: int,
        maximum_total_tokens: int,
        estimated_input_token_divisor: int,
        rates: Mapping[str, Any],
        ledger: DeepseekRequestLedger | None = None,
        batch_state: "BatchState | None" = None,
    ):
        self.maximum_requests = int(maximum_requests)
        self.maximum_tokens = int(maximum_total_tokens)
        self.estimated_input_token_divisor = int(estimated_input_token_divisor)
        self.rates = deepcopy(dict(rates))
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
        self.observed = deepcopy(self.usage)
        self.coverage = {
            key: {"present": 0, "missing": 0} for key in self.usage
        }
        self.cost_cny = 0.0
        self.stop_reason: str | None = None
        self.ledger = ledger
        self.batch_state = batch_state
        self.open_request_id: str | None = None
        if ledger is not None:
            snapshot = ledger.snapshot()
            self.requests = int(snapshot["requests_started"])
            self.usage = deepcopy(snapshot["usage"])
            self.observed = deepcopy(snapshot["usage_observed_subtotals"])
            self.coverage = {
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
            raise DeepseekProviderError(self.stop_reason)
        if self.open_request_id is not None:
            self.stop("previous_request_not_settled")
            raise DeepseekProviderError(self.stop_reason)
        if self.requests >= self.maximum_requests:
            self.stop("source_request_cap_reached")
            raise DeepseekProviderError(self.stop_reason)
        if self.batch_state and self.batch_state.stop_reason:
            raise DeepseekProviderError(self.batch_state.stop_reason)
        if self.batch_state and self.batch_state.requests >= self.batch_state.maximum_requests:
            self.batch_state.stop("batch_request_cap_reached")
            raise DeepseekProviderError(self.batch_state.stop_reason)
        measured = measure_messages(messages, self.estimated_input_token_divisor)
        output_limit = int(body["max_tokens"])
        projected = (
            int(self.usage["total_tokens"] or 0)
            + measured["estimated_input_tokens"]
            + output_limit
        )
        if projected > self.maximum_tokens:
            self.stop("estimated_source_token_budget_reached")
            raise DeepseekProviderError(self.stop_reason)
        if self.batch_state:
            batch_projected = (
                int(self.batch_state.usage["total_tokens"])
                + measured["estimated_input_tokens"]
                + output_limit
            )
            if batch_projected > self.batch_state.maximum_tokens:
                self.batch_state.stop("estimated_batch_token_budget_reached")
                raise DeepseekProviderError(self.batch_state.stop_reason)
        request_id = None
        next_index = self.requests + 1
        if self.ledger is not None:
            try:
                request_id = self.ledger.start_request(
                    request_index=next_index,
                    model=str(body["model"]),
                    request_parameters={
                        key: value for key, value in body.items() if key != "messages"
                    },
                    estimated_input_tokens=measured["estimated_input_tokens"],
                    message_count=measured["message_count"],
                    message_bytes=measured["message_bytes"],
                    output_token_limit=output_limit,
                    budget_before={
                        "requests": self.requests,
                        "usage": deepcopy(self.usage),
                    },
                )
            except RequestLedgerError as exc:
                self.stop("request_ledger_write_failure")
                raise DeepseekProviderError(str(exc)) from exc
            self.open_request_id = request_id
        self.requests = next_index
        if self.batch_state:
            self.batch_state.requests += 1
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
                self.coverage[key]["missing"] += 1
                if self.batch_state:
                    self.batch_state.coverage[key]["missing"] += 1
            else:
                self.coverage[key]["present"] += 1
                self.observed[key] += int(value)
                if self.batch_state:
                    self.batch_state.observed[key] += int(value)
                    self.batch_state.coverage[key]["present"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            self.usage[key] = self.observed[key]
            if self.batch_state:
                self.batch_state.usage[key] = self.batch_state.observed[key]
        for key in (
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cache_miss_tokens",
        ):
            self.usage[key] = (
                self.observed[key]
                if self.coverage[key]["present"] > 0
                and self.coverage[key]["missing"] == 0
                else None
            )
            if self.batch_state:
                self.batch_state.usage[key] = (
                    self.batch_state.observed[key]
                    if self.batch_state.coverage[key]["present"] > 0
                    and self.batch_state.coverage[key]["missing"] == 0
                    else None
                )
        cost = estimate_cost_cny(usage, self.rates)
        self.cost_cny += cost
        if self.batch_state:
            self.batch_state.cost_cny += cost
        request_id = self.open_request_id
        if self.ledger is not None and request_id is not None:
            try:
                self.ledger.commit_usage(
                    request_id=request_id,
                    raw_usage=record.get("usage") if record else {},
                    normalized_usage=usage,
                    response_model_id=str(record.get("response_model_id")) if record else "",
                    response_id=record.get("response_id") if record else None,
                    http_status=record.get("http_status") if record else None,
                    finish_reason=record.get("finish_reason") if record else None,
                    response_body_bytes=record.get("response_body_bytes") if record else None,
                    elapsed_seconds=float(record.get("elapsed_seconds") or 0.0) if record else 0.0,
                    cumulative_usage=self.usage,
                )
            except RequestLedgerError as exc:
                self.stop("request_ledger_usage_commit_failure")
                raise DeepseekProviderError(str(exc)) from exc
            self.open_request_id = None
            if record is not None:
                record["usage_committed"] = True
        if int(self.usage["total_tokens"] or 0) > self.maximum_tokens:
            self.stop("actual_source_token_budget_exceeded")
        if (
            self.batch_state
            and int(self.batch_state.usage["total_tokens"] or 0)
            > self.batch_state.maximum_tokens
        ):
            self.batch_state.stop("actual_batch_token_budget_exceeded")
        if record is not None:
            record["cumulative_usage"] = deepcopy(self.usage)
            record["cumulative_cost_cny"] = self.cost_cny

    def fail_request(
        self,
        *,
        reason: str,
        exception_type: str | None = None,
        http_status: int | None = None,
        error_message: str | None = None,
        transport_phase: str | None = None,
    ) -> None:
        if self.ledger is not None and self.open_request_id is not None:
            try:
                self.ledger.mark_failed(
                    request_id=self.open_request_id,
                    reason=reason,
                    exception_type=exception_type,
                    http_status=http_status,
                    error=error_message,
                    transport_phase=transport_phase,
                )
            except RequestLedgerError:
                self.stop("request_ledger_failure_commit_error")
            self.open_request_id = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "maximum_requests": self.maximum_requests,
            "requests": self.requests,
            "maximum_total_tokens": self.maximum_tokens,
            "usage": deepcopy(self.usage),
            "usage_observed_subtotals": deepcopy(self.observed),
            "component_coverage": deepcopy(self.coverage),
            "cost_cny": self.cost_cny,
            "stop_reason": self.stop_reason,
        }


class BatchState:
    def __init__(self, limits: Mapping[str, Any]):
        self.maximum_requests = int(limits["batch_maximum_requests"])
        self.maximum_tokens = int(limits["batch_maximum_total_tokens"])
        self.requests = 0
        self.usage = {
            key: 0
            for key in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "reasoning_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "cache_miss_tokens",
            )
        }
        self.observed = deepcopy(self.usage)
        self.coverage = {
            key: {"present": 0, "missing": 0} for key in self.usage
        }
        self.cost_cny = 0.0
        self.stop_reason: str | None = None

    def stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = str(reason)

    def snapshot(self) -> dict[str, Any]:
        return {
            "maximum_requests": self.maximum_requests,
            "requests": self.requests,
            "maximum_total_tokens": self.maximum_tokens,
            "usage": deepcopy(self.usage),
            "usage_observed_subtotals": deepcopy(self.observed),
            "component_coverage": deepcopy(self.coverage),
            "cost_cny": self.cost_cny,
            "stop_reason": self.stop_reason,
        }


def provider_config(
    source: Mapping[str, Any],
    *,
    api_key: str,
) -> dict[str, Any]:
    provider = source["provider"]
    if provider["model_id"] != MODEL_ID:
        raise DeepseekProviderError("provider model must be deepseek-v4-pro")
    base_url = normalize_base_url(provider["base_url"])
    provider_id = provider_namespace(
        str(provider.get("name") or "deepseek-v4"),
        MODEL_ID,
        base_url,
    )
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": MODEL_ID,
        "accepted_response_model_ids": [MODEL_ID],
        "temperature": float(provider["temperature"]),
        "max_tokens": int(provider["max_output_tokens"]),
        "timeout_seconds": int(provider["timeout_seconds"]),
        "tool_choice": str(provider["tool_choice"]),
        "parallel_tool_calls": bool(provider["parallel_tool_calls"]),
        "stream": bool(provider["stream"]),
        "provider_rate_limit": {
            "lock_dir": str(PROVIDER_LOCK_DIR),
            "provider_id": provider_id,
            "max_concurrent_requests": 1,
            "slot_timeout_seconds": int(provider["timeout_seconds"]) + 60,
        },
    }


def public_provider(config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(config))
    value["api_key"] = "[PRESENT]" if value.get("api_key") else "[MISSING]"
    return value


class DeepseekAuditProvider(ReasoningReplayProviderMixin, frozen.AuditProvider):
    def __init__(
        self,
        budget: DeepseekBudget,
        config: Mapping[str, Any],
        interface: Mapping[str, Any],
        memory_monitor: Any | None = None,
    ):
        super().__init__(deepcopy(dict(config)), deepcopy(dict(interface)))
        self.deepseek_budget = budget
        self.memory_monitor = memory_monitor
        limit = self.config.get("provider_rate_limit")
        self.provider_request_limiter = (
            ProviderRequestLimiter(
                lock_dir=limit["lock_dir"],
                provider_id=limit["provider_id"],
                max_concurrent_requests=int(
                    limit.get("max_concurrent_requests", 1)
                ),
            )
            if isinstance(limit, Mapping)
            else None
        )

    def _memory(self, stage: str, record: Mapping[str, Any], **metadata: Any) -> None:
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
            },
            **metadata,
        )

    def _body(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"],
            "parallel_tool_calls": self.config["parallel_tool_calls"],
            "stream": self.config["stream"],
        }
        tools = self.interface.get("tools") or []
        if tools:
            body["tools"] = deepcopy(tools)
            body["tool_choice"] = self.config["tool_choice"]
        if "max_completion_tokens" in body:
            raise DeepseekProviderError("DeepSeek must use max_tokens")
        return body

    @contextmanager
    def _open_provider_response(
        self,
        wire_request: request.Request,
        record: dict[str, Any],
    ):
        if self.provider_request_limiter is None:
            with request.urlopen(
                wire_request,
                timeout=self.config["timeout_seconds"],
            ) as response:
                yield response
            return
        limit = self.config["provider_rate_limit"]
        with self.provider_request_limiter.request_slot(
            timeout_seconds=limit.get("slot_timeout_seconds")
        ) as slot:
            record["provider_rate_limit"] = deepcopy(slot)
            with request.urlopen(
                wire_request,
                timeout=self.config["timeout_seconds"],
            ) as response:
                yield response

    def complete_with_usage(self, messages: list[dict[str, Any]]):
        prepared = self.prepare_request_messages(messages)
        body = self._body(prepared)
        estimated = self.deepseek_budget.before_request(prepared, body)
        started_at = datetime.now(timezone.utc).isoformat()
        started = monotonic()
        record: dict[str, Any] = {
            "request_index": len(self.raw_records) + 1,
            "request_id": estimated["request_id"],
            "started_at": started_at,
            "transport": "urllib (no SDK)",
            "timeout_seconds": self.config["timeout_seconds"],
            "http_status": None,
            "response_body_bytes": None,
            "usage_committed": False,
            "request_parameters": {
                key: deepcopy(value) for key, value in body.items() if key != "messages"
            },
            "request_model_id": self.model,
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
            self._memory("provider_request_before", record)
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
            with self._open_provider_response(wire_request, record) as response:
                record["http_status"] = getattr(response, "status", None)
                record["transport_phase"] = "reading_http_headers"
                content_length = (
                    response.headers.get("Content-Length")
                    if response.headers
                    else None
                )
                expected_length = None
                if content_length is not None:
                    try:
                        expected_length = int(content_length)
                    except ValueError as exc:
                        raise DeepseekProviderError(
                            "provider_content_length_malformed"
                        ) from exc
                    record["response_content_length"] = expected_length
                    if expected_length < 0:
                        raise DeepseekProviderError(
                            "provider_content_length_malformed"
                        )
                    if expected_length > MAX_PROVIDER_RESPONSE_BYTES:
                        raise DeepseekProviderError(
                            "provider_content_length_exceeds_size_limit"
                        )
                chunks = []
                total = 0
                record["transport_phase"] = "reading_response_body"
                while total <= MAX_PROVIDER_RESPONSE_BYTES:
                    remaining = MAX_PROVIDER_RESPONSE_BYTES + 1 - total
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
                    self.deepseek_budget.stop(
                        "provider_response_exceeds_size_limit"
                    )
                    raise DeepseekProviderError(
                        "provider_response_exceeds_size_limit"
                    )
                if (
                    expected_length is not None
                    and len(raw_body_bytes) != expected_length
                ):
                    record["transport_phase"] = "response_body_incomplete"
                    self.deepseek_budget.stop("provider_response_incomplete")
                    raise DeepseekProviderError(
                        "provider_response_body_incomplete"
                    )
                raw_body = raw_body_bytes.decode("utf-8")
                record["transport_phase"] = "decoding_response"
            value = json.loads(raw_body)
            record["transport_phase"] = "parsing_response"
            if not isinstance(value, dict):
                raise DeepseekProviderError("response_body_not_object")
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
            if not isinstance(choice, dict):
                raise DeepseekProviderError(
                    "response_must_contain_exactly_one_choice"
                )
            message = choice.get("message")
            finish_reason = choice.get("finish_reason")
            record["finish_reason"] = finish_reason
            usage = value.get("usage")
            normalized = validate_usage(usage)
            record.update(
                response_message=deepcopy(message) if isinstance(message, dict) else None,
                usage=deepcopy(usage),
                normalized_usage=normalized,
            )
            record["elapsed_seconds"] = monotonic() - started
            self.deepseek_budget.record_usage(normalized, record)
            record["transport_phase"] = "usage_committed"
            if response_model not in self.config["accepted_response_model_ids"]:
                self.deepseek_budget.stop("unexpected_response_model")
                raise DeepseekProviderError(
                    f"unexpected_response_model:{response_model!r}"
                )
            if not isinstance(message, dict):
                self.deepseek_budget.stop("assistant_message_missing")
                raise DeepseekProviderError("assistant_message_missing")
            self.remember_reasoning(message)
            if finish_reason == "length":
                self.deepseek_budget.stop("output_truncated")
                raise DeepseekProviderError("output_truncated_at_max_tokens")
            calls = message.get("tool_calls") or []
            if calls:
                if not isinstance(calls, list) or len(calls) != 1:
                    self.deepseek_budget.stop("parallel_or_invalid_tool_calls")
                    raise DeepseekProviderError(
                        "expected_exactly_one_tool_call"
                    )
                call = calls[0]
                function = call.get("function") if isinstance(call, dict) else None
                call_id = call.get("id") if isinstance(call, dict) else None
                if not isinstance(call_id, str) or not call_id:
                    self.deepseek_budget.stop("tool_call_id_missing")
                    raise DeepseekProviderError("tool_call_id_missing")
                if not isinstance(function, dict):
                    self.deepseek_budget.stop("tool_function_missing")
                    raise DeepseekProviderError("tool_function_missing")
                try:
                    parsed = json.loads(function.get("arguments", ""))
                except (TypeError, json.JSONDecodeError) as exc:
                    self.deepseek_budget.stop("tool_arguments_invalid")
                    raise DeepseekProviderError("tool_arguments_invalid") from exc
                if not isinstance(parsed, dict):
                    self.deepseek_budget.stop("tool_arguments_not_object")
                    raise DeepseekProviderError("tool_arguments_not_object")
            record["outcome"] = "success"
            return message, usage
        except error.HTTPError as exc:
            record["http_status"] = exc.code
            body_excerpt = ""
            try:
                body_excerpt = exc.read(4096).decode("utf-8", errors="replace")
            except Exception:
                pass
            safe = body_excerpt.replace(
                self.config["api_key"], "[REDACTED]"
            )[:2000]
            record.update(
                outcome="failure",
                exception_type=type(exc).__name__,
                error=safe or str(exc)[:2000],
            )
            self.deepseek_budget.stop(
                "http_429" if exc.code == 429 else "http_error"
            )
            self.deepseek_budget.fail_request(
                reason=self.deepseek_budget.stop_reason or "http_error",
                exception_type=type(exc).__name__,
                http_status=exc.code,
                error_message=safe,
                transport_phase=record["transport_phase"],
            )
            failure_logged = True
            raise
        except DeepseekProviderError as exc:
            record.update(
                outcome="failure",
                exception_type=type(exc).__name__,
                error=str(exc),
            )
            if self.deepseek_budget.stop_reason is None:
                self.deepseek_budget.stop("deepseek_protocol_error")
            self.deepseek_budget.fail_request(
                reason=self.deepseek_budget.stop_reason,
                exception_type=type(exc).__name__,
                http_status=record.get("http_status"),
                error_message=str(exc),
                transport_phase=record["transport_phase"],
            )
            failure_logged = True
            raise
        except Exception as exc:
            safe = str(exc).replace(self.config["api_key"], "[REDACTED]")[:2000]
            record.update(
                outcome="failure",
                exception_type=type(exc).__name__,
                error=safe,
            )
            self.deepseek_budget.stop("provider_protocol_error")
            self.deepseek_budget.fail_request(
                reason=self.deepseek_budget.stop_reason,
                exception_type=type(exc).__name__,
                http_status=record.get("http_status"),
                error_message=safe,
                transport_phase=record["transport_phase"],
            )
            failure_logged = True
            raise
        finally:
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            record["elapsed_seconds"] = monotonic() - started
            if (
                not failure_logged
                and not record.get("usage_committed")
                and self.deepseek_budget.open_request_id is not None
            ):
                self.deepseek_budget.fail_request(
                    reason="request_unsettled_at_provider_exit",
                    exception_type="UnsettledRequest",
                    http_status=record.get("http_status"),
                    transport_phase=record["transport_phase"],
                )
                record["outcome"] = "failure"
            self.request_log.append(record)

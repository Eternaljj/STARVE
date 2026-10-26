"""GPT-5.6 Sol provider adaptation for the H1 source-10 pilot.

This module is intentionally separate from the frozen DeepSeek/SiliconFlow
provider. It sends only Chat Completions fields supported by the GPT adapter,
records request and response model identities, and rejects incomplete usage.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
from time import monotonic
from typing import Any, Mapping
from urllib import error, request

import run_formal_h1_v1_1 as frozen
from adapters.h1_pilot_usage import normalize_usage
from adapters.h1_provider_config import H1ProviderConfigError, normalize_base_url
from adapters.h1_request_ledger import GPTRequestLedger, RequestLedgerError
from adapters.h1_thinking_compat import ReasoningReplayProviderMixin


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "h1_gpt_5_6_sol_source10_pilot_v1.json"
MODEL_ID = "gpt-5.6-sol"
OUTPUT_TOKEN_PARAMETER = "max_completion_tokens"
REASONING_EFFORT = "medium"
MAX_PROVIDER_RESPONSE_BYTES = 8 * 1024 * 1024
RESPONSE_READ_CHUNK_BYTES = 64 * 1024


class GPTProviderError(RuntimeError):
    """A protocol or usage failure that must stop subsequent Pilot requests."""


def load_gpt_pilot_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("config_version") != "h1-gpt-single-source-pilot-v1":
        raise H1ProviderConfigError("unexpected GPT H1 pilot config version")
    provider = config["provider"]
    if provider["model_id"] != MODEL_ID:
        raise H1ProviderConfigError("GPT pilot must use gpt-5.6-sol exactly")
    if provider["reasoning_effort"] != REASONING_EFFORT:
        raise H1ProviderConfigError("GPT pilot reasoning_effort must remain medium")
    if provider["output_token_parameter"] != OUTPUT_TOKEN_PARAMETER:
        raise H1ProviderConfigError("GPT pilot must use max_completion_tokens")
    if provider["max_completion_tokens"] != 2400:
        raise H1ProviderConfigError("GPT compatibility preflight requires max_completion_tokens=2400")
    if provider["accepted_response_model_ids"] != [MODEL_ID]:
        raise H1ProviderConfigError("unknown response model aliases are not permitted")
    limits = config["limits"]
    if (
        limits["maximum_requests"] != 140
        or limits["maximum_total_tokens"] != 800_000
        or limits["automatic_retry"]
        or limits["automatic_replacement"]
    ):
        raise H1ProviderConfigError("GPT Pilot limits or no-retry policy changed")
    if config["source_index"] != 10:
        raise H1ProviderConfigError("GPT Pilot source must remain source 10")
    return config


def provider_config(
    pilot: Mapping[str, Any],
    *,
    require_key: bool,
    environ: Mapping[str, str] | None = None,
    send_temperature: bool | None = None,
) -> dict[str, Any]:
    import os

    env = os.environ if environ is None else environ
    provider = pilot["provider"]
    names = provider["environment"]
    model = str(env.get(names["model_id"], "")).strip() or provider["model_id"]
    if model != MODEL_ID:
        raise H1ProviderConfigError("H1_MODEL_ID must be gpt-5.6-sol")
    base_url = str(env.get(names["base_url"], "")).strip() or provider["base_url"]
    api_key = str(env.get(names["api_key"], "")).strip()
    if require_key and not api_key:
        raise H1ProviderConfigError(
            f"{names['api_key']} is required to start the GPT Provider"
        )
    if send_temperature is None:
        send_temperature = False
    temperature = float(provider["temperature_candidate"]) if send_temperature else None
    return {
        "api_key": api_key,
        "base_url": normalize_base_url(base_url),
        "model": model,
        "accepted_response_model_ids": list(provider["accepted_response_model_ids"]),
        "temperature": temperature,
        "send_temperature": bool(send_temperature),
        "reasoning_effort": str(provider["reasoning_effort"]),
        "output_token_parameter": str(provider["output_token_parameter"]),
        "max_completion_tokens": int(provider["max_completion_tokens"]),
        "timeout_seconds": int(provider["timeout_seconds"]),
        "tool_choice": str(provider["tool_choice"]),
        "parallel_tool_calls": bool(provider["parallel_tool_calls"]),
        "stream": bool(provider["stream"]),
        "provider_interface": str(provider["interface"]),
    }


def provider_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(config))
    value["api_key"] = "[PRESENT]" if value.get("api_key") else "[MISSING]"
    return value


def _int_field(usage: Mapping[str, Any], name: str) -> int | None:
    value = usage.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def validate_usage(usage: Any) -> dict[str, Any]:
    """Normalize GPT usage and reject malformed or internally inconsistent totals."""
    if not isinstance(usage, Mapping):
        raise GPTProviderError("usage_missing")
    normalized = normalize_usage(usage)
    prompt = _int_field(usage, "prompt_tokens")
    completion = _int_field(usage, "completion_tokens")
    total = _int_field(usage, "total_tokens")
    if prompt is None or completion is None or total is None:
        raise GPTProviderError("usage_missing_required_token_fields")
    if total != prompt + completion:
        raise GPTProviderError(
            f"usage_total_mismatch:{total}!={prompt}+{completion}"
        )
    details = usage.get("completion_tokens_details")
    if details is not None and not isinstance(details, Mapping):
        raise GPTProviderError("completion_tokens_details_malformed")
    reasoning = normalized.get("reasoning_tokens")
    if reasoning is not None and reasoning > completion:
        raise GPTProviderError("reasoning_tokens_exceed_completion_tokens")
    prompt_details = usage.get("prompt_tokens_details")
    if prompt_details is not None and not isinstance(prompt_details, Mapping):
        raise GPTProviderError("prompt_tokens_details_malformed")
    cached = normalized.get("cache_read_tokens")
    if cached is not None and cached > prompt:
        raise GPTProviderError("cached_tokens_exceed_prompt_tokens")
    normalized.update(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=total,
        usage_complete=True,
    )
    return normalized


def measure_messages(
    messages: list[dict[str, Any]], divisor: int = 3
) -> dict[str, int]:
    raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "message_bytes": len(raw),
        "message_count": len(messages),
        "estimated_input_tokens": max(1, math.ceil(len(raw) / max(1, divisor))),
    }


def estimate_input_tokens(messages: list[dict[str, Any]], divisor: int = 3) -> int:
    return measure_messages(messages, divisor)["estimated_input_tokens"]


def install_process_memory_limit(maximum_bytes: int) -> dict[str, Any]:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    requested = int(maximum_bytes)
    target = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    return {
        "limit_bytes": target,
        "soft_before": soft,
        "hard": hard,
    }


def estimate_cost_usd(usage: Mapping[str, Any], rates: Mapping[str, Any]) -> float:
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cached = int(usage.get("cache_read_tokens") or 0)
    written = int(usage.get("cache_write_tokens") or 0)
    uncached = max(0, input_tokens - cached - written)
    return (
        uncached * float(rates["input"])
        + cached * float(rates["cached_input"])
        + written * float(rates["cache_write"])
        + output_tokens * float(rates["output"])
    ) / 1_000_000


class GPTBudget:
    """Serial request/token guard with explicit estimate and post-response settlement."""

    def __init__(
        self,
        *,
        maximum_requests: int,
        maximum_total_tokens: int,
        estimated_input_token_divisor: int,
        rates: Mapping[str, Any],
        ledger: GPTRequestLedger | None = None,
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
        self.usage_observed_subtotals = deepcopy(self.usage)
        self.usage_component_coverage = {
            key: {"present": 0, "missing": 0}
            for key in self.usage
        }
        self.cost_usd_list_estimate = 0.0
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
            self.cost_usd_list_estimate = estimate_cost_usd(
                self.usage_observed_subtotals, self.rates
            )
            if snapshot["recovery_blocked"]:
                self.stop("unresolved_request_ledger_recovery")
            if self.requests >= self.maximum_requests:
                self.stop("request_cap_reached_during_recovery")
            if int(self.usage.get("total_tokens") or 0) >= self.maximum_tokens:
                self.stop("token_threshold_reached_during_recovery")

    @classmethod
    def from_pilot(
        cls,
        pilot: Mapping[str, Any],
        *,
        ledger_path: Path | str | None = None,
    ) -> "GPTBudget":
        limits = pilot["limits"]
        return cls(
            maximum_requests=int(limits["maximum_requests"]),
            maximum_total_tokens=int(limits["maximum_total_tokens"]),
            estimated_input_token_divisor=int(limits["estimated_input_token_divisor"]),
            rates=pilot["cost_estimation"]["rates_usd_per_million_tokens"],
            ledger=GPTRequestLedger(ledger_path) if ledger_path is not None else None,
        )

    def stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
            if self.ledger is not None:
                try:
                    self.ledger.record_budget_stop(reason)
                except RequestLedgerError:
                    pass

    def before_request(
        self,
        messages: list[dict[str, Any]],
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.stop_reason:
            raise GPTProviderError(self.stop_reason)
        if self.open_request_id is not None:
            self.stop("previous_request_not_settled")
            raise GPTProviderError(self.stop_reason)
        if self.requests >= self.maximum_requests:
            self.stop("request_cap_reached_before_request")
            raise GPTProviderError(self.stop_reason)
        measured = measure_messages(messages, self.estimated_input_token_divisor)
        estimated_input = measured["estimated_input_tokens"]
        maximum_output = int(body["max_completion_tokens"])
        projected = int(self.usage["total_tokens"] or 0) + estimated_input + maximum_output
        if projected > self.maximum_tokens:
            self.stop("estimated_token_budget_reached_before_request")
            raise GPTProviderError(self.stop_reason)
        self.requests += 1
        request_id = None
        if self.ledger is not None:
            try:
                request_id = self.ledger.start_request(
                    request_index=self.requests,
                    model=str(body["model"]),
                    request_parameters={
                        key: value for key, value in body.items() if key != "messages"
                    },
                    estimated_input_tokens=estimated_input,
                    message_count=measured["message_count"],
                    message_bytes=measured["message_bytes"],
                    output_token_limit=maximum_output,
                    budget_before={
                        "requests": self.requests - 1,
                        "usage": deepcopy(self.usage),
                    },
                )
            except RequestLedgerError as exc:
                self.stop("request_ledger_write_failure")
                raise GPTProviderError(str(exc)) from exc
            self.open_request_id = request_id
        return {
            "request_id": request_id,
            "estimated_input_tokens": estimated_input,
            "message_count": measured["message_count"],
            "message_bytes": measured["message_bytes"],
            "maximum_output_tokens": maximum_output,
            "projected_total_tokens": projected,
            "remaining_before_request": self.maximum_tokens
            - int(self.usage["total_tokens"] or 0),
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
        self.cost_usd_list_estimate += estimate_cost_usd(usage, self.rates)
        if usage.get("reasoning_tokens") is None and record is not None:
            self.missing_reasoning_requests.append(int(record["request_index"]))
        if record is not None:
            record.update(
                cumulative_usage=deepcopy(self.usage),
                cumulative_usage_observed_subtotals=deepcopy(
                    self.usage_observed_subtotals
                ),
                cumulative_component_coverage=deepcopy(
                    self.usage_component_coverage
                ),
                cumulative_cost_usd_list_estimate=self.cost_usd_list_estimate,
            )
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
                raise GPTProviderError(str(exc)) from exc
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


class GPTAuditProvider(ReasoningReplayProviderMixin, frozen.AuditProvider):
    """Strict GPT Chat Completions transport used by smoke and Pilot runs."""

    def __init__(
        self,
        budget: Any,
        config: Mapping[str, Any],
        interface: Mapping[str, Any],
        memory_monitor: Any | None = None,
    ):
        super().__init__(deepcopy(dict(config)), deepcopy(dict(interface)))
        self.gpt_budget = budget
        self.memory_monitor = memory_monitor

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
                    json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
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
            "reasoning_effort": self.config["reasoning_effort"],
            "max_completion_tokens": self.config["max_completion_tokens"],
            "parallel_tool_calls": self.config["parallel_tool_calls"],
            "stream": self.config["stream"],
        }
        tools = self.interface.get("tools") or []
        if tools:
            body["tools"] = tools
            body["tool_choice"] = self.config["tool_choice"]
        if self.config.get("send_temperature"):
            body["temperature"] = self.config["temperature"]
        if "max_tokens" in body:
            raise GPTProviderError("gpt_provider_must_not_send_max_tokens")
        return body

    def complete_with_usage(self, messages: list[dict[str, Any]]):
        prepared = self.prepare_request_messages(messages)
        body = self._body(prepared)
        estimated = self.gpt_budget.before_request(prepared, body)
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
            self._memory(
                "provider_request_before",
                record,
                estimated_input_tokens=estimated["estimated_input_tokens"],
                request_json_bytes=len(
                    json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
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
            with request.urlopen(wire_request, timeout=self.config["timeout_seconds"]) as response:
                record["http_status"] = getattr(response, "status", None)
                record["transport_phase"] = "reading_http_headers"
                content_length = response.headers.get("Content-Length") if response.headers else None
                expected_content_length = None
                if content_length is not None:
                    try:
                        expected_content_length = int(content_length)
                        record["response_content_length"] = expected_content_length
                        if expected_content_length < 0:
                            raise GPTProviderError("provider_content_length_malformed")
                        if expected_content_length > MAX_PROVIDER_RESPONSE_BYTES:
                            raise GPTProviderError("provider_content_length_exceeds_size_limit")
                    except ValueError:
                        raise GPTProviderError("provider_content_length_malformed")
                chunks = []
                total = 0
                record["transport_phase"] = "reading_response_body"
                while total <= MAX_PROVIDER_RESPONSE_BYTES:
                    remaining = MAX_PROVIDER_RESPONSE_BYTES + 1 - total
                    chunk = response.read(min(RESPONSE_READ_CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                raw_body_bytes = b"".join(chunks)
                record["response_body_bytes"] = len(raw_body_bytes)
                record["transport_phase"] = "response_body_complete"
                if len(raw_body_bytes) > MAX_PROVIDER_RESPONSE_BYTES:
                    self.gpt_budget.stop("provider_response_exceeds_size_limit")
                    raise GPTProviderError("provider_response_exceeds_size_limit")
                if (
                    expected_content_length is not None
                    and len(raw_body_bytes) != expected_content_length
                ):
                    record["transport_phase"] = "response_body_incomplete"
                    self.gpt_budget.stop("provider_response_incomplete")
                    raise GPTProviderError("provider_response_body_incomplete")
                self._memory(
                    "provider_response_read_after",
                    record,
                    response_body_bytes=len(raw_body_bytes),
                )
                raw_body = raw_body_bytes.decode("utf-8")
                record["transport_phase"] = "decoding_response"
            value = json.loads(raw_body)
            record["transport_phase"] = "parsing_response"
            if not isinstance(value, dict):
                raise GPTProviderError("response_body_not_object")
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
            message = choice.get("message") if isinstance(choice, dict) else None
            finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
            record["finish_reason"] = finish_reason
            usage = value.get("usage")
            normalized = validate_usage(usage)
            record.update(
                response_message=deepcopy(message) if isinstance(message, dict) else None,
                usage=deepcopy(usage),
                normalized_usage=normalized,
            )
            record["elapsed_seconds"] = monotonic() - started
            self.gpt_budget.record_usage(normalized, record)
            record["transport_phase"] = "usage_committed"
            self._memory("provider_usage_committed", record)
            if response_model not in self.config["accepted_response_model_ids"]:
                self.gpt_budget.stop("unexpected_response_model")
                raise GPTProviderError(
                    f"unexpected_response_model:{response_model!r}"
                )
            if not isinstance(choices, list) or len(choices) != 1:
                self.gpt_budget.stop("invalid_choice_count")
                raise GPTProviderError("response_must_contain_exactly_one_choice")
            if not isinstance(choice, dict):
                self.gpt_budget.stop("invalid_choice")
                raise GPTProviderError("choice_not_object")
            if not isinstance(message, dict):
                self.gpt_budget.stop("assistant_message_missing")
                raise GPTProviderError("assistant_message_missing")
            self.remember_reasoning(message)
            if finish_reason == "length":
                self.gpt_budget.stop("output_truncated")
                raise GPTProviderError("output_truncated_at_max_completion_tokens")
            calls = message.get("tool_calls") or []
            if calls:
                if not isinstance(calls, list) or len(calls) != 1:
                    self.gpt_budget.stop("parallel_or_invalid_tool_calls")
                    raise GPTProviderError("expected_exactly_one_tool_call")
                call = calls[0]
                function = call.get("function") if isinstance(call, dict) else None
                call_id = call.get("id") if isinstance(call, dict) else None
                if not isinstance(call_id, str) or not call_id:
                    self.gpt_budget.stop("tool_call_id_missing")
                    raise GPTProviderError("tool_call_id_missing")
                if not isinstance(function, dict):
                    self.gpt_budget.stop("tool_function_missing")
                    raise GPTProviderError("tool_function_missing")
                try:
                    parsed = json.loads(function.get("arguments", ""))
                except (TypeError, json.JSONDecodeError) as exc:
                    self.gpt_budget.stop("tool_arguments_invalid")
                    raise GPTProviderError("tool_arguments_invalid") from exc
                if not isinstance(parsed, dict):
                    self.gpt_budget.stop("tool_arguments_not_object")
                    raise GPTProviderError("tool_arguments_not_object")
            record["outcome"] = "success"
            self._memory("provider_request_after", record)
            return message, usage
        except error.HTTPError as exc:
            record["http_status"] = exc.code
            body_excerpt = ""
            try:
                body_excerpt = exc.read(4096).decode("utf-8", errors="replace")
            except Exception:
                pass
            safe = body_excerpt.replace(self.config["api_key"], "[REDACTED]")[:2000]
            record.update(
                outcome="failure",
                exception_type=type(exc).__name__,
                error=safe or str(exc)[:2000],
            )
            self.gpt_budget.stop("http_429" if exc.code == 429 else "http_error")
            self.gpt_budget.fail_request(
                reason=self.gpt_budget.stop_reason or "http_error",
                exception_type=type(exc).__name__,
                http_status=exc.code,
                error=safe,
            )
            failure_logged = True
            self._memory("provider_request_after", record)
            raise
        except GPTProviderError as exc:
            record["outcome"] = "failure"
            record["exception_type"] = type(exc).__name__
            record["error"] = str(exc)
            if self.gpt_budget.stop_reason is None:
                self.gpt_budget.stop("gpt_protocol_error")
            self.gpt_budget.fail_request(
                reason=self.gpt_budget.stop_reason or "gpt_protocol_error",
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
                error=str(exc).replace(self.config["api_key"], "[REDACTED]")[:2000],
            )
            self.gpt_budget.stop("provider_protocol_error")
            self.gpt_budget.fail_request(
                reason=self.gpt_budget.stop_reason,
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
                and self.gpt_budget.open_request_id is not None
            ):
                self.gpt_budget.fail_request(
                    reason="request_unsettled_at_provider_exit",
                    exception_type="UnsettledRequest",
                    http_status=record.get("http_status"),
                )
                record["outcome"] = "failure"
            self.request_log.append(record)

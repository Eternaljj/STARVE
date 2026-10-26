"""Anthropic Messages provider, exact usage accounting, and durable ledger."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from time import monotonic
from typing import Any, Mapping
from urllib import error, request

import run_formal_h1_v1_1 as frozen
from adapters.h1_gpt_provider import GPTBudget
from adapters.h1_request_ledger import GPTRequestLedger, RequestLedgerError
from adapters.h1_provider_config import H1ProviderConfigError, normalize_base_url


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
API_KEY_ENV = "CLAUDE_H1_API_KEY"
MAX_PROVIDER_RESPONSE_BYTES = 8 * 1024 * 1024
RESPONSE_READ_CHUNK_BYTES = 64 * 1024

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


class ClaudeProviderError(RuntimeError):
    """Provider, usage, or protocol failure that must stop the current run."""


class ClaudeUpstreamRefusal(ClaudeProviderError):
    """The Messages response completed normally with stop_reason=refusal."""


REQUEST_ID_HEADERS = (
    "anthropic-request-id",
    "x-request-id",
    "request-id",
    "x-amzn-requestid",
    "cf-ray",
)
SAFE_RESPONSE_HEADERS = {
    "content-type",
    "content-length",
    "date",
    "server",
    *REQUEST_ID_HEADERS,
}


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = read_json(path)
    if config.get(
        "config_version"
    ) != "h1-claude-sonnet-4-6-9source-20261022":
        raise H1ProviderConfigError("unexpected Claude H1 config version")
    provider = config["provider"]
    if provider["interface"] != "anthropic-messages-v1":
        raise H1ProviderConfigError("Claude H1 requires Anthropic Messages")
    if provider["api_key_environment"] != API_KEY_ENV:
        raise H1ProviderConfigError("Claude H1 must use CLAUDE_H1_API_KEY exactly")
    candidate = str(provider.get("model_candidate", "")).strip()
    if not candidate or provider.get("accepted_response_model_ids") != [candidate]:
        raise H1ProviderConfigError("one exact Claude model ID must be declared")
    limits = config["limits"]
    if (
        limits["per_source_maximum_requests"] != 140
        or limits["per_source_maximum_total_tokens"] != 800_000
        or limits["batch_maximum_requests"] != 1_260
        or limits["batch_maximum_total_tokens"] != 7_200_000
        or limits["automatic_retry"]
        or limits["automatic_replacement"]
    ):
        raise H1ProviderConfigError("Claude H1 budget or no-retry policy changed")
    analysis = config["analysis"]
    if (
        analysis["states"] != ["RECON", "EXPLOITATION", "VERIFICATION"]
        or analysis["conditions"]
        != ["BASELINE", "BRANCH_NEW", "DEPENDENCY_V6", "EVIDENCE_NEW_V3"]
        or analysis["expected_slots"] != 108
        or analysis["expected_baselines"] != 27
        or analysis["expected_traps"] != 81
        or analysis["expected_pairs"] != 81
        or analysis["followup_budget"] != 12
        or analysis["pre_effect_version"] != "pre-effect-v1"
        or analysis["cross_model_comparison"]
    ):
        raise H1ProviderConfigError("Claude H1 frozen analysis design changed")
    if config["selected_source_indices"] != [1, 6, 2, 11, 16, 12, 21, 26, 22]:
        raise H1ProviderConfigError("frozen nine-source order changed")
    return config


def provider_config(
    config: Mapping[str, Any],
    *,
    require_key: bool,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    provider = config["provider"]
    model = str(
        env.get("CLAUDE_H1_MODEL_ID", "")
    ).strip() or str(provider["model_candidate"])
    if model not in provider["accepted_response_model_ids"]:
        raise H1ProviderConfigError(
            "CLAUDE_H1_MODEL_ID differs from the declared exact model ID"
        )
    api_key = str(env.get(API_KEY_ENV, "")).strip()
    if require_key and not api_key:
        raise H1ProviderConfigError(
            "CLAUDE_H1_API_KEY is required to start the Claude H1 run"
        )
    return {
        "name": str(provider["name"]),
        "api_key": api_key,
        "base_url": normalize_base_url(provider["base_url"]),
        "model": model,
        "accepted_response_model_ids": list(
            provider["accepted_response_model_ids"]
        ),
        "anthropic_version": str(provider["anthropic_version"]),
        "authorization_scheme": str(provider["authorization_scheme"]),
        "temperature": float(provider["temperature"]),
        "max_tokens": int(provider["max_tokens"]),
        "timeout_seconds": int(provider["timeout_seconds"]),
        "tool_choice": str(provider["tool_choice"]),
        "disable_parallel_tool_use": bool(
            provider["disable_parallel_tool_use"]
        ),
        "stream": bool(provider["stream"]),
        "provider_interface": str(provider["interface"]),
        "usage_semantics": str(provider["usage_semantics"]["version"]),
    }


def public_provider(config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(config))
    value["api_key"] = "[PRESENT]" if value.get("api_key") else "[MISSING]"
    return value


def _redact_text(value: str, secret: str) -> tuple[str, bool]:
    if not secret:
        return value, False
    return value.replace(secret, "[REDACTED]"), secret in value


def _safe_response_headers(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    result = {}
    for name in SAFE_RESPONSE_HEADERS:
        value = headers.get(name)
        if value is not None:
            result[name] = str(value)
    return result


def _header_request_id(headers: Mapping[str, str]) -> str | None:
    lowered = {str(key).lower(): value for key, value in headers.items()}
    for name in REQUEST_ID_HEADERS:
        value = lowered.get(name)
        if value:
            return str(value)
    return None


def _int_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def normalize_anthropic_usage(
    raw_usage: Any,
) -> dict[str, Any]:
    """Normalize Messages usage without double-counting cache tokens.

    Anthropic reports uncached input in ``input_tokens``. Cache creation and
    cache read are separate additive fields. If neither cache field is present,
    caching was not reported and both remain NA while contributing zero.
    """
    if not isinstance(raw_usage, Mapping):
        raise ClaudeProviderError("usage_missing")
    raw_input = _int_count(raw_usage.get("input_tokens"))
    output = _int_count(raw_usage.get("output_tokens"))
    if raw_input is None or output is None:
        raise ClaudeProviderError("usage_missing_required_token_fields")
    creation_present = "cache_creation_input_tokens" in raw_usage
    read_present = "cache_read_input_tokens" in raw_usage
    if creation_present != read_present:
        raise ClaudeProviderError("usage_cache_fields_incomplete")
    creation = (
        _int_count(raw_usage.get("cache_creation_input_tokens"))
        if creation_present
        else None
    )
    cache_read = (
        _int_count(raw_usage.get("cache_read_input_tokens"))
        if read_present
        else None
    )
    if creation_present and (creation is None or cache_read is None):
        raise ClaudeProviderError("usage_cache_fields_invalid")
    total_input = raw_input + (creation or 0) + (cache_read or 0)
    total = total_input + output
    normalized = {
        "input_tokens": total_input,
        "output_tokens": output,
        "prompt_tokens": total_input,
        "completion_tokens": output,
        "total_tokens": total,
        "uncached_input_tokens": raw_input,
        "cache_creation_input_tokens": creation,
        "cache_read_input_tokens": cache_read,
        "cache_write_tokens": creation,
        "cache_read_tokens": cache_read,
        "cache_miss_tokens": raw_input,
        "reasoning_tokens": None,
        "usage_complete": True,
        "usage_semantics": "anthropic-messages-usage-v1",
        "raw_usage": deepcopy(dict(raw_usage)),
    }
    return normalized


def estimate_input_tokens(
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


class ClaudeRequestLedger(GPTRequestLedger):
    """The shared durable ledger with Claude-specific request IDs."""

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
                f"request index discontinuity: expected {expected}, got {request_index}"
            )
        request_id = f"clreq-{request_index:04d}"
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


class ClaudeBudget(GPTBudget):
    """Request/token guard for the Anthropic Messages body schema."""

    def before_request(
        self,
        messages: list[dict[str, Any]],
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.stop_reason:
            raise ClaudeProviderError(self.stop_reason)
        if self.open_request_id is not None:
            self.stop("previous_request_not_settled")
            raise ClaudeProviderError(self.stop_reason)
        if self.requests >= self.maximum_requests:
            self.stop("request_cap_reached_before_request")
            raise ClaudeProviderError(self.stop_reason)
        measured = estimate_input_tokens(
            messages,
            self.estimated_input_token_divisor,
        )
        maximum_output = int(body["max_tokens"])
        projected = (
            int(self.usage["total_tokens"] or 0)
            + measured["estimated_input_tokens"]
            + maximum_output
        )
        if projected > self.maximum_tokens:
            self.stop("estimated_token_budget_reached_before_request")
            raise ClaudeProviderError(self.stop_reason)
        self.requests += 1
        request_id = None
        if self.ledger is not None:
            try:
                request_id = self.ledger.start_request(
                    request_index=self.requests,
                    model=str(body["model"]),
                    request_parameters={
                        key: deepcopy(value)
                        for key, value in body.items()
                        if key not in {"messages", "system"}
                    },
                    estimated_input_tokens=measured["estimated_input_tokens"],
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
                raise ClaudeProviderError(str(exc)) from exc
            self.open_request_id = request_id
        return {
            "request_id": request_id,
            **measured,
            "maximum_output_tokens": maximum_output,
            "projected_total_tokens": projected,
            "remaining_before_request": self.maximum_tokens
            - int(self.usage["total_tokens"] or 0),
        }


def _anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            if isinstance(content, str):
                system_parts.append(content)
            continue
        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": message.get("content") or "",
                            **(
                                {"is_error": True}
                                if message.get("is_error")
                                else {}
                            ),
                        }
                    ],
                }
            )
            continue
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            content = message.get("content")
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    arguments = json.loads(
                        function.get("arguments") or "{}"
                    )
                except json.JSONDecodeError as exc:
                    raise ClaudeProviderError(
                        "assistant ToolCall arguments are not JSON"
                    ) from exc
                if not isinstance(arguments, dict):
                    raise ClaudeProviderError(
                        "assistant ToolCall arguments are not an object"
                    )
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": function["name"],
                        "input": arguments,
                    }
                )
            converted.append(
                {"role": "assistant", "content": blocks}
            )
            continue
        if role == "user":
            converted.append(
                {
                    "role": "user",
                    "content": message.get("content") or "",
                }
            )
            continue
        raise ClaudeProviderError("unsupported provider message role")
    return (
        "\n\n".join(system_parts) if system_parts else None,
        converted,
    )


def _anthropic_tools(
    interface: dict[str, Any],
) -> list[dict[str, Any]]:
    tools = []
    for item in interface.get("tools") or []:
        function = item["function"]
        tools.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "input_schema": deepcopy(function["parameters"]),
            }
        )
    return tools


class AnthropicAuditProvider(frozen.AuditProvider):
    """Serial Anthropic Messages transport with complete request auditing."""

    def __init__(
        self,
        budget: ClaudeBudget,
        config: Mapping[str, Any],
        interface: Mapping[str, Any],
        *,
        memory_monitor: Any | None = None,
        condition: str | None = None,
    ):
        super().__init__(deepcopy(dict(config)), deepcopy(dict(interface)))
        self.claude_budget = budget
        self.memory_monitor = memory_monitor
        self.condition = condition

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

    def _body(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        system_prompt, converted = _anthropic_messages(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"],
            "messages": converted,
            "tools": _anthropic_tools(self.interface),
            "tool_choice": self.config.get(
                "forced_tool_choice",
                {
                    "type": "auto",
                    "disable_parallel_tool_use": True,
                },
            ),
            "stream": False,
        }
        if system_prompt is not None:
            body["system"] = system_prompt
        return body

    def complete_with_usage(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.claude_budget.stop_reason:
            raise ClaudeProviderError(self.claude_budget.stop_reason)
        body = self._body(messages)
        estimated = self.claude_budget.before_request(messages, body)
        request_index = self.claude_budget.requests
        started = monotonic()
        record: dict[str, Any] = {
            "request_index": request_index,
            "request_id": estimated["request_id"],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "condition": self.condition,
            "transport": "urllib-anthropic-messages",
            "provider_interface": "anthropic-messages-v1",
            "anthropic_version": self.config["anthropic_version"],
            "timeout_seconds": self.config["timeout_seconds"],
            "messages": deepcopy(messages),
            "request_parameters": {
                key: deepcopy(value)
                for key, value in body.items()
                if key not in {"messages", "system"}
            },
            "request_model_id": self.model,
            "estimated_input_tokens": estimated["estimated_input_tokens"],
            "message_count": estimated["message_count"],
            "message_bytes": estimated["message_bytes"],
            "http_status": None,
            "response_body_bytes": None,
            "response_content_length": None,
            "response_id": None,
            "response_model_id": None,
            "finish_reason": None,
            "response_message": None,
            "usage": None,
            "normalized_usage": None,
            "usage_committed": False,
            "outcome": "pending",
            "transport_phase": "request_prepared",
        }
        self.raw_records.append(record)
        self.request_log.append(record)
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
                self.config["base_url"] + "/v1/messages",
                data=json.dumps(
                    body,
                    ensure_ascii=False,
                ).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": self.config["anthropic_version"],
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
                sanitized_headers = _safe_response_headers(
                    response.headers
                )
                record["response_headers_sanitized"] = sanitized_headers
                record["upstream_request_id"] = _header_request_id(
                    sanitized_headers
                )
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
                        self.claude_budget.stop(
                            "provider_content_length_malformed"
                        )
                        raise ClaudeProviderError(
                            "provider_content_length_malformed"
                        ) from exc
                    record["response_content_length"] = expected_length
                    if expected_length < 0:
                        self.claude_budget.stop(
                            "provider_content_length_malformed"
                        )
                        raise ClaudeProviderError(
                            "provider_content_length_malformed"
                        )
                    if expected_length > MAX_PROVIDER_RESPONSE_BYTES:
                        self.claude_budget.stop(
                            "provider_content_length_exceeds_size_limit"
                        )
                        raise ClaudeProviderError(
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
                raw_bytes = b"".join(chunks)
                record["response_body_bytes"] = len(raw_bytes)
                if len(raw_bytes) > MAX_PROVIDER_RESPONSE_BYTES:
                    self.claude_budget.stop(
                        "provider_response_exceeds_size_limit"
                    )
                    raise ClaudeProviderError(
                        "provider_response_exceeds_size_limit"
                    )
                if (
                    expected_length is not None
                    and len(raw_bytes) != expected_length
                ):
                    record["transport_phase"] = "response_body_incomplete"
                    self.claude_budget.stop(
                        "provider_response_body_incomplete"
                    )
                    raise ClaudeProviderError(
                        "provider_response_body_incomplete"
                    )
                try:
                    raw_body = raw_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    record["transport_phase"] = "response_decode_error"
                    raise ClaudeProviderError(
                        "provider_response_not_utf8"
                    ) from exc
                sanitized_body, body_redacted = _redact_text(
                    raw_body,
                    self.config["api_key"],
                )
                record["response_body_text_sanitized"] = sanitized_body
                record["response_body_redacted"] = body_redacted
                record["transport_phase"] = "decoding_response"
            payload = json.loads(raw_body)
            if not isinstance(payload, dict):
                raise ClaudeProviderError("response_body_not_object")
            record["http_status"] = record["http_status"] or 200
            response_model = payload.get("model")
            record["response_model_id"] = response_model
            record["response_id"] = payload.get("id")
            record["finish_reason"] = payload.get("stop_reason")
            record["transport_phase"] = "parsing_response"
            if payload.get("type") != "message":
                raise ClaudeProviderError("invalid_anthropic_message_type")
            raw_usage = payload.get("usage")
            record["usage"] = deepcopy(raw_usage)
            normalized = normalize_anthropic_usage(raw_usage)
            record["normalized_usage"] = normalized
            record["elapsed_seconds"] = monotonic() - started
            self.claude_budget.record_usage(normalized, record)
            record["usage_committed"] = True
            record["transport_phase"] = "usage_committed"
            self._memory("provider_usage_committed", record)
            if response_model not in self.config["accepted_response_model_ids"]:
                self.claude_budget.stop("unexpected_response_model")
                raise ClaudeProviderError(
                    f"unexpected_response_model:{response_model!r}"
                )
            content = payload.get("content")
            if not isinstance(content, list):
                raise ClaudeProviderError("provider_content_not_array")
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            block_profile = []
            for block in content:
                if not isinstance(block, dict):
                    raise ClaudeProviderError("provider_content_block_invalid")
                block_type = block.get("type")
                if block_type == "text" and isinstance(
                    block.get("text"),
                    str,
                ):
                    text_parts.append(block["text"])
                    block_profile.append(
                        {"type": "text", "chars": len(block["text"])}
                    )
                elif block_type in {"thinking", "redacted_thinking"}:
                    block_profile.append({"type": block_type})
                elif (
                    block_type == "tool_use"
                    and isinstance(block.get("id"), str)
                    and block.get("id")
                    and isinstance(block.get("name"), str)
                    and isinstance(block.get("input"), dict)
                ):
                    tool_calls.append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(
                                    block["input"],
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                    )
                    block_profile.append(
                        {
                            "type": "tool_use",
                            "id": block["id"],
                            "name": block["name"],
                            "input_keys": sorted(block["input"]),
                        }
                    )
                else:
                    raise ClaudeProviderError(
                        f"unsupported_provider_content_block:{block_type!r}"
                    )
            record["content_blocks"] = block_profile
            record["tool_call_ids"] = [
                call["id"] for call in tool_calls
            ]
            record["tool_names"] = [
                call["function"]["name"] for call in tool_calls
            ]
            if len(tool_calls) > 1:
                self.claude_budget.stop(
                    "parallel_or_invalid_tool_calls"
                )
                raise ClaudeProviderError(
                    "expected_exactly_one_tool_call"
                )
            message = {
                "role": "assistant",
                "content": "".join(text_parts) or None,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            }
            record["response_message"] = deepcopy(message)
            if payload.get("stop_reason") == "refusal":
                record["outcome"] = "refusal"
                record["upstream_refusal"] = True
                record["refusal_origin_evidence"] = (
                    "external_messages_response_boundary"
                )
                self.claude_budget.stop("upstream_refusal")
                raise ClaudeUpstreamRefusal("UPSTREAM_REFUSAL")
            if payload.get("stop_reason") == "max_tokens":
                record["outcome"] = "truncated"
                self.claude_budget.stop("output_truncated")
                raise ClaudeProviderError("output_truncated_at_max_tokens")
            if tool_calls and payload.get("stop_reason") != "tool_use":
                self.claude_budget.stop("tool_use_stop_reason_mismatch")
                raise ClaudeProviderError(
                    "tool_call_without_tool_use_stop_reason"
                )
            if not tool_calls and payload.get("stop_reason") == "tool_use":
                self.claude_budget.stop("tool_use_content_missing")
                raise ClaudeProviderError(
                    "tool_use_stop_reason_without_tool_call"
                )
            record["outcome"] = "success"
            self._memory("provider_request_after", record)
            return message, normalized
        except error.HTTPError as exc:
            record["http_status"] = exc.code
            sanitized_headers = _safe_response_headers(exc.headers)
            record["response_headers_sanitized"] = sanitized_headers
            record["upstream_request_id"] = _header_request_id(
                sanitized_headers
            )
            excerpt = ""
            try:
                excerpt = exc.read(4096).decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                pass
            safe, excerpt_redacted = _redact_text(
                excerpt,
                self.config["api_key"],
            )
            safe = safe[:2000]
            record["response_body_text_sanitized"] = safe
            record["response_body_redacted"] = excerpt_redacted
            record.update(
                outcome="failure",
                exception_type=type(exc).__name__,
                error=safe or str(exc)[:2000],
            )
            self.claude_budget.stop(
                "http_429" if exc.code == 429 else "http_error"
            )
            self.claude_budget.fail_request(
                reason=self.claude_budget.stop_reason or "http_error",
                exception_type=type(exc).__name__,
                http_status=exc.code,
                error=safe,
            )
            failure_logged = True
            self._memory("provider_request_after", record)
            raise ClaudeProviderError(
                f"provider_http_status_{exc.code}"
            ) from exc
        except ClaudeProviderError as exc:
            record["outcome"] = record.get("outcome") or "failure"
            record["exception_type"] = type(exc).__name__
            record["error"] = str(exc)
            if self.claude_budget.stop_reason is None:
                self.claude_budget.stop("claude_protocol_error")
            self.claude_budget.fail_request(
                reason=self.claude_budget.stop_reason
                or "claude_protocol_error",
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
            self.claude_budget.stop("provider_protocol_error")
            self.claude_budget.fail_request(
                reason=self.claude_budget.stop_reason,
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
                and self.claude_budget.open_request_id is not None
            ):
                self.claude_budget.fail_request(
                    reason="request_unsettled_at_provider_exit",
                    exception_type="UnsettledRequest",
                    http_status=record.get("http_status"),
                )
                record["outcome"] = "failure"


def request_usage_row(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = record.get("normalized_usage") or {}
    raw_usage = record.get("usage") if isinstance(
        record.get("usage"),
        Mapping,
    ) else {}
    return {
        "request_index": record.get("request_index"),
        "request_id": record.get("request_id"),
        "condition": record.get("condition"),
        "request_model_id": record.get("request_model_id"),
        "response_model_id": record.get("response_model_id"),
        "response_id": record.get("response_id"),
        "http_status": record.get("http_status"),
        "finish_reason": record.get("finish_reason"),
        "outcome": record.get("outcome"),
        "usage_committed": record.get("usage_committed"),
        "raw_input_tokens_uncached": raw_usage.get("input_tokens"),
        "raw_output_tokens": raw_usage.get("output_tokens"),
        "cache_creation_input_tokens": raw_usage.get(
            "cache_creation_input_tokens"
        ),
        "cache_read_input_tokens": raw_usage.get(
            "cache_read_input_tokens"
        ),
        "normalized_input_tokens": normalized.get("input_tokens"),
        "normalized_output_tokens": normalized.get("output_tokens"),
        "normalized_total_tokens": normalized.get("total_tokens"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "response_body_bytes": record.get("response_body_bytes"),
        "transport_phase": record.get("transport_phase"),
        "error": record.get("error"),
    }

"""Durable append-only request ledger for the independent DeepSeek H1 run."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping


class RequestLedgerError(RuntimeError):
    pass


def _forbidden_key(value: Any, path: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            current = f"{path}.{key}" if path else str(key)
            if str(key).lower() in {"api_key", "authorization", "bearer"}:
                return current
            found = _forbidden_key(item, current)
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _forbidden_key(item, f"{path}[{index}]")
            if found:
                return found
    return None


class DeepseekRequestLedger:
    """Every state transition is flush+fsync before returning to the caller."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        self.started: dict[str, dict[str, Any]] = {}
        self.completed_usage: dict[str, dict[str, Any]] = {}
        self.failed: dict[str, dict[str, Any]] = {}
        self.unknown_request_ids: list[str] = []
        self.truncated_tail = False
        self.corrupt = False
        self._recover()

    def _recover(self) -> None:
        if not self.path.exists():
            return
        for index, line in enumerate(self.path.read_bytes().splitlines(keepends=True)):
            if not line.endswith(b"\n"):
                self.truncated_tail = True
                break
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.corrupt = True
                raise RequestLedgerError(
                    f"request ledger JSON is corrupt at line {index + 1}"
                ) from exc
            self._apply_event(event)
        self._refresh_unknown()

    def _apply_event(self, event: Mapping[str, Any]) -> None:
        kind = event.get("event")
        request_id = event.get("request_id")
        if kind == "request_started":
            if not isinstance(request_id, str) or not request_id:
                raise RequestLedgerError("request_started lacks request_id")
            if request_id in self.started:
                raise RequestLedgerError(f"duplicate request_started: {request_id}")
            self.started[request_id] = deepcopy(dict(event))
        elif kind == "request_usage_committed":
            if request_id not in self.started:
                raise RequestLedgerError(f"usage without request_started: {request_id}")
            if request_id in self.completed_usage:
                raise RequestLedgerError(f"duplicate usage commit: {request_id}")
            self.completed_usage[request_id] = deepcopy(dict(event))
        elif kind == "request_failed":
            if request_id not in self.started:
                raise RequestLedgerError(f"failure without request_started: {request_id}")
            self.failed[request_id] = deepcopy(dict(event))
        elif kind == "budget_stopped":
            pass
        else:
            raise RequestLedgerError(f"unknown ledger event: {kind!r}")
        self.events.append(deepcopy(dict(event)))

    def _refresh_unknown(self) -> None:
        self.unknown_request_ids = sorted(
            request_id
            for request_id in self.started
            if request_id not in self.completed_usage
        )

    def _append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if self.truncated_tail or self.corrupt:
            raise RequestLedgerError("request ledger is not append-safe")
        forbidden = _forbidden_key(event)
        if forbidden:
            raise RequestLedgerError(
                f"credential-like field forbidden in ledger: {forbidden}"
            )
        payload = deepcopy(dict(event))
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self._apply_event(payload)
        self._refresh_unknown()
        return payload

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
        request_id = f"dsreq-{request_index:04d}"
        self._append({
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
        })
        return request_id

    def commit_usage(
        self,
        *,
        request_id: str,
        raw_usage: Mapping[str, Any],
        normalized_usage: Mapping[str, Any],
        response_model_id: str,
        response_id: str | None,
        http_status: int | None,
        finish_reason: str | None,
        response_body_bytes: int | None,
        elapsed_seconds: float,
        cumulative_usage: Mapping[str, Any],
    ) -> None:
        if request_id not in self.started:
            raise RequestLedgerError("cannot commit usage for unknown request")
        if request_id in self.completed_usage:
            raise RequestLedgerError("usage already committed for request")
        self._append({
            "event": "request_usage_committed",
            "request_id": request_id,
            "request_index": self.started[request_id]["request_index"],
            "status": "COMPLETED",
            "committed_at": datetime.now(timezone.utc).isoformat(),
            "response_model_id": response_model_id,
            "response_id": response_id,
            "http_status": http_status,
            "finish_reason": finish_reason,
            "response_body_bytes": response_body_bytes,
            "elapsed_seconds": float(elapsed_seconds),
            "usage": deepcopy(dict(raw_usage)),
            "normalized_usage": deepcopy(dict(normalized_usage)),
            "cumulative_usage": deepcopy(dict(cumulative_usage)),
        })

    def mark_failed(
        self,
        *,
        request_id: str,
        reason: str,
        exception_type: str | None = None,
        http_status: int | None = None,
        error: str | None = None,
        transport_phase: str | None = None,
    ) -> None:
        if request_id not in self.started:
            raise RequestLedgerError("cannot fail unknown request")
        status = (
            "UNKNOWN"
            if request_id not in self.completed_usage
            else "FAILED_WITH_USAGE"
        )
        self._append({
            "event": "request_failed",
            "request_id": request_id,
            "request_index": self.started[request_id]["request_index"],
            "status": status,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "reason": str(reason)[:500],
            "exception_type": exception_type,
            "http_status": http_status,
            "error": str(error)[:1000] if error else None,
            "transport_phase": transport_phase,
        })

    def record_budget_stop(self, reason: str) -> None:
        if self.events and self.events[-1].get("event") == "budget_stopped":
            if self.events[-1].get("reason") == str(reason)[:500]:
                return
        self._append({
            "event": "budget_stopped",
            "status": "STOPPED",
            "reason": str(reason)[:500],
            "stopped_at": datetime.now(timezone.utc).isoformat(),
        })

    def snapshot(self) -> dict[str, Any]:
        required = ("input_tokens", "output_tokens", "total_tokens")
        optional = (
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cache_miss_tokens",
        )
        observed = {key: 0 for key in (*required, *optional)}
        coverage = {
            key: {"present": 0, "missing": 0, "request_ids_missing": []}
            for key in (*required, *optional)
        }
        for event in self.completed_usage.values():
            normalized = event.get("normalized_usage") or {}
            for key in (*required, *optional):
                value = normalized.get(key)
                if value is None:
                    coverage[key]["missing"] += 1
                    coverage[key]["request_ids_missing"].append(event["request_id"])
                    continue
                coverage[key]["present"] += 1
                observed[key] += int(value)
        usage = {key: observed[key] for key in required}
        for key in optional:
            usage[key] = (
                observed[key]
                if coverage[key]["present"] > 0 and coverage[key]["missing"] == 0
                else None
            )
        return {
            "ledger_path": str(self.path),
            "requests_started": len(self.started),
            "requests_with_usage": len(self.completed_usage),
            "unknown_request_ids": list(self.unknown_request_ids),
            "failed_requests": deepcopy(self.failed),
            "usage": usage,
            "usage_observed_subtotals": observed,
            "component_coverage": coverage,
            "truncated_tail": self.truncated_tail,
            "corrupt": self.corrupt,
            "recovery_blocked": bool(
                self.unknown_request_ids or self.truncated_tail or self.corrupt
            ),
        }

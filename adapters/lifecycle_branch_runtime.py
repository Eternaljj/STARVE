"""BASELINE/BRANCH continuation at the shared tool-response exposure boundary.

Restore an intercepted tool call, execute it once, then expose its normal
result with optional frozen Trap information. Only then request the Agent's
next decision. No free-standing user message can substitute for that result.
"""
from __future__ import annotations

from copy import deepcopy
import json
from time import monotonic
from typing import Any

from adapters.g0_shared_envelope import SharedInteractionEnvelope, shared_application
from adapters.lifecycle_state_collector import STATE_PROTOCOL_VERSION
from adapters.tool_response_exposure import EXPOSURE_BOUNDARY, ToolResponseExposure, compose_tool_response
from run_lifecycle_v2_feasibility import INTERFACE_VERSION, Session, neutral_interface
from traps.protocol_v3 import MAX_OPTIONAL_FOLLOWUPS
from traps.v2 import BRANCH_V2_IMPLEMENTATION


class LifecycleBranchRuntime:
    """One restored branch; run is bounded by the shared ledger, not MAX_CALLS."""

    @classmethod
    def restore(cls, checkpoint: dict, provider: Any, *, application: Any = None, interface: dict | None = None):
        if (checkpoint.get("version") != "lifecycle-v2-natural-runtime-v1"
                or checkpoint.get("interface_version") != INTERFACE_VERSION
                or checkpoint.get("state_protocol_version") != STATE_PROTOCOL_VERSION):
            raise ValueError("unsupported checkpoint protocol or interface")
        if checkpoint.get("terminal_reason") is not None or checkpoint.get("trap_injected"):
            raise ValueError("branch requires a non-terminal, pre-intervention checkpoint")
        if checkpoint.get("state") not in {"RECON", "EXPLOITATION", "VERIFICATION"}:
            raise ValueError("branch requires an observable functional-v2 State")
        expected_sampling = {
            "temperature": provider.config["temperature"],
            "max_tokens": provider.config.get("max_tokens", 2400),
        }
        if checkpoint.get("sampling") != expected_sampling:
            raise ValueError("checkpoint sampling mismatch")
        expected_interface = interface if interface is not None else neutral_interface()
        if getattr(provider, "interface", neutral_interface()) != expected_interface:
            raise ValueError("checkpoint neutral interface mismatch")
        pending = checkpoint.get("pending_tool_call")
        if pending is None:
            raise ValueError("tool-response boundary requires an intercepted pending tool call")
        history = checkpoint.get("agent_history") or []
        if pending is not None and (not history or history[-1].get("tool_calls") != [pending]):
            raise ValueError("pending action is not committed in checkpoint history")
        if pending is not None and (not isinstance(pending, dict) or not pending.get("id")
                        or not isinstance(pending.get("function"), dict)
                        or not isinstance(pending["function"].get("name"), str)
                        or not isinstance(pending["function"].get("arguments"), str)):
            raise ValueError("malformed checkpoint pending action")
        obj = cls()
        obj.application = None
        if application is not None:
            payload = shared_application(application).bridge_payload()
            tools = {item["function"]["name"] for item in expected_interface["tools"]}
            if (payload["primitive"] != "BRANCH_EXPANSION"
                    or payload["implementation"] != BRANCH_V2_IMPLEMENTATION
                    or payload.get("state_label") != checkpoint["state"]
                    or payload.get("recommended_tool") not in tools
                    or payload.get("initial_response", {}).get("status") != "candidate_catalog"
                    or len(payload.get("opportunities", [])) != 4):
                raise ValueError("application must be a validated BRANCH V2 binding for this runtime")
            # Build via the original builder/provider before calling this ingress.
            # This layer does not replace that builder's legality validation.
            obj.application = payload
        obj.session = Session.restore(deepcopy(checkpoint), provider)
        obj.checkpoint_id = checkpoint.get("checkpoint_id")
        obj.condition = "BRANCH" if application is not None else "BASELINE"
        obj.source_provider_calls = checkpoint["provider_calls"]
        obj.envelope = SharedInteractionEnvelope()
        obj.exposure = None
        obj.exposure_delivered = False
        obj.initial_tool_calls = 0
        obj.lifecycle_events = []
        obj.interaction_depth = 0
        obj.technical_error = None
        obj.wall_time = 0.0
        return obj

    # Runtime hooks transport core decisions; the legacy BRANCH defaults stay unchanged.
    def _before_tool_execution(self, call, *, initial):
        pass

    def _execute_task_tool(self, call, args):
        return self.session.execute_pending()

    def _response_exposure(self, call, args, result, status, *, initial):
        return self.exposure if initial else None

    def _mechanism_terminal_reason(self):
        return None

    def _provider_started(self):
        pass

    def _provider_received(self, usage):
        pass

    def _provider_failed(self, reason):
        pass

    def _autonomous_stop(self):
        pass

    def _invalid_agent_response(self, reason):
        pass

    def prepare_response(self):
        """Bind core information while the real intercepted action remains pending."""
        if self.session.terminal_reason is not None:
            raise RuntimeError("terminal branch cannot prepare a response")
        if self.exposure_delivered or self.envelope.exposed:
            raise RuntimeError("initial tool response has already been exposed")
        if any(event["kind"] == "response_prepared" for event in self.lifecycle_events):
            return
        self.exposure = ToolResponseExposure("BRANCH", BRANCH_V2_IMPLEMENTATION,
            deepcopy(self.application["initial_response"])) if self.application else None
        self.lifecycle_events.append({
            "kind": "response_prepared", "condition": self.condition,
            "pending_tool_call": deepcopy(self.session.pending),
            "environment_snapshot": self.session.env.snapshot(),
            "catalog": deepcopy(self.exposure.content) if self.exposure else None,
        })

    def initial_exposure(self):
        """Return the real initial tool response, then open the follow-up window."""
        if self.envelope.exposed or self.session.terminal_reason is not None:
            raise RuntimeError("initial tool response cannot be exposed twice or after terminal")
        self.prepare_response()
        self._execute_pending(initial=True)

    def _stop(self, reason):
        self.session.terminal_reason = reason
        self.envelope.stop(reason)
        self.lifecycle_events.append({"kind": "terminal", "reason": reason})

    def _execute_pending(self, *, initial=False):
        if self.envelope.terminal or self.session.terminal_reason:
            raise RuntimeError("terminal branch cannot execute a tool")
        if not initial and not self.envelope.exposed:
            raise RuntimeError("follow-up tool execution requires the initial tool response")
        call = deepcopy(self.session.pending)
        self._before_tool_execution(call, initial=initial)
        function = call["function"]
        if initial:
            self.initial_tool_calls += 1
        else:
            self.envelope.tool_call()
        self.lifecycle_events.append({"kind": "tool_call", "call": deepcopy(call)})
        args = None
        try:
            args = json.loads(function["arguments"])
            if not isinstance(args, dict):
                raise ValueError("arguments must be object")
            result = self._execute_task_tool(call, args)
            status = "ok"
        except (ValueError, KeyError, TypeError) as exc:
            result, status = {"error": str(exc)}, "tool_error"
        except Exception as exc:
            self.technical_error = f"tool failure: {type(exc).__name__}"
            result, status = {"error": self.technical_error}, "technical_failure"
        exposure = self._response_exposure(call, args, result, status, initial=initial)
        visible = compose_tool_response(result, exposure)
        self.session.tool_calls.append({
            "name": function["name"], "arguments": deepcopy(args),
            "result": deepcopy(result), "status": status,
            "provider_call": self.session.provider_calls,
        })
        self.session.messages.append({"role": "tool", "tool_call_id": call["id"],
                                      "content": json.dumps(visible, ensure_ascii=False)})
        self.session.pending = None
        if initial:
            self.envelope.initial_exposure()
            self.exposure_delivered = True
            self.lifecycle_events.append({"kind": "initial_exposure", "boundary": EXPOSURE_BOUNDARY,
                                          "tool_call_id": call["id"], "visible_result": deepcopy(visible)})
        else:
            self.envelope.tool_result()
        self.lifecycle_events.append({"kind": "tool_result", "result": deepcopy(result),
                                      "visible_result": deepcopy(visible), "status": status})
        reason = "technical_failure" if status == "technical_failure" else (
            self.session.terminal_reason or self._mechanism_terminal_reason())
        if reason:
            self._stop(reason)

    def resume_pending(self):
        if self.envelope.terminal or self.session.terminal_reason:
            raise RuntimeError("terminal branch cannot resume an action")
        if not self.envelope.exposed:
            self.initial_exposure()
            return
        if self.session.pending is not None:
            self._execute_pending()

    def run(self):
        if self.envelope.terminal or self.session.terminal_reason:
            raise RuntimeError("terminal branch cannot continue")
        started = monotonic()
        try:
            if not self.envelope.exposed:
                self.initial_exposure()
            while not self.envelope.terminal and self.envelope.followups < MAX_OPTIONAL_FOLLOWUPS:
                self.envelope.provider_request()
                self.session.provider_calls += 1
                try:
                    self._provider_started()
                    history = deepcopy(self.session.messages)
                    provider = self.session.provider
                    if hasattr(provider, "complete_with_usage"):
                        message, usage = provider.complete_with_usage(history)
                    else:
                        message, usage = provider.complete(history), None
                    if not isinstance(message, dict):
                        raise ValueError("invalid assistant message")
                except Exception as exc:
                    self.technical_error = f"provider failure: {type(exc).__name__}"
                    request_log = getattr(self.session.provider, "request_log", [])
                    detail = deepcopy(request_log[-1]) if request_log else {"exception_type": type(exc).__name__}
                    self.lifecycle_events.append({"kind": "provider_failure_detail",
                        "branch_request_index": self.envelope.provider_calls,
                        "source_plus_branch_request_index": self.session.provider_calls,
                        "detail": detail})
                    failed_usage = detail.get("usage")
                    if failed_usage is not None:
                        self._provider_received(failed_usage)
                    self._provider_failed(self.technical_error)
                    try:
                        self.envelope.provider_result(failed_usage, failure=True)
                    except (ValueError, TypeError, AttributeError):
                        detail["usage_validation_error"] = "unusable usage retained in provider request log"
                        self.envelope.provider_result(failure=True)
                    self.session.terminal_reason = "technical_failure"
                    self.lifecycle_events.append({"kind": "terminal", "reason": "technical_failure"})
                    break
                try:
                    self.envelope.provider_result(usage)
                except (ValueError, TypeError, AttributeError):
                    # Invalid usage is an accounting failure, never a fabricated zero.
                    self.technical_error = "invalid provider usage"
                    self._provider_failed(self.technical_error)
                    self.envelope.provider_result(failure=True)
                    self.session.terminal_reason = "technical_failure"
                    self.lifecycle_events.append({"kind": "terminal", "reason": "technical_failure"})
                    break
                self._provider_received(usage)
                self.interaction_depth += 1
                calls = message.get("tool_calls") or []
                self.session.messages.append({"role": "assistant", "content": message.get("content"),
                                              **({"tool_calls": deepcopy(calls)} if calls else {})})
                if not calls:
                    self._autonomous_stop()
                    self._stop("autonomous_stop")
                elif len(calls) != 1:
                    self._invalid_agent_response("unsupported_parallel_calls")
                    self._stop("unsupported_parallel_calls")
                elif (not isinstance(calls[0], dict) or not calls[0].get("id")
                      or not isinstance(calls[0].get("function"), dict)
                      or not isinstance(calls[0]["function"].get("name"), str)
                      or not isinstance(calls[0]["function"].get("arguments"), str)):
                    self._invalid_agent_response("invalid_tool_call")
                    self._stop("invalid_tool_call")
                else:
                    self.session.pending = deepcopy(calls[0])
                    self._execute_pending()
            if not self.envelope.terminal:
                self._stop("interaction_budget_exhausted")
        finally:
            self.wall_time += monotonic() - started
        return self.metrics()

    def metrics(self):
        return {**self.envelope.validate(), "condition": self.condition,
                "exposure_boundary": EXPOSURE_BOUNDARY,
                "initial_tool_calls": self.initial_tool_calls,
                "post_exposure_tool_calls": self.envelope.tool_calls,
                "tool_calls": self.initial_tool_calls + self.envelope.tool_calls,
                "checkpoint_id": self.checkpoint_id,
                "source_provider_calls": self.source_provider_calls,
                "followup_rounds": self.envelope.followups,
                "interaction_depth": self.interaction_depth,
                "wall_time": self.wall_time,
                "task_success": self.session.terminal_reason == "task_completed",
                "technical_failure": self.technical_error is not None,
                "technical_error": self.technical_error}

    def close(self):
        self.session.env.close()

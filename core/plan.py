"""Ordinary current-plan submission schema and autonomous-stop tool."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INTERFACE_CONTRACT_PATH = ROOT / "configs" / "plan_interface.json"

PLAN_VERSION = "iteration-v6-plan-v1"
FUNCTION_NAME = "submit_current_plan"
ROOT_PLAN_FIELDS = {"schema_version", "goal_id", "steps"}
STEP_FIELDS = {"step_id", "operation", "binds_to", "fallback_for", "resource_units", "detail"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def load_interface_contract(path: Path = INTERFACE_CONTRACT_PATH) -> dict[str, Any]:
    """Load and minimally validate the V6-only interface contract."""
    contract = _load_json(path)
    if contract.get("interface_contract_version") != "iteration-v6-p0-interface-contract-v0.1":
        raise ValueError("unexpected V6 interface contract version")
    if contract.get("current_plan_schema_version") != PLAN_VERSION:
        raise ValueError("interface contract plan version mismatch")
    if contract.get("submission_transport") != "openai_compatible_function_tool":
        raise ValueError("V6 requires the frozen function-tool transport")
    if contract.get("function_name") != FUNCTION_NAME:
        raise ValueError("interface contract function name mismatch")
    if contract.get("step_required_fields") != ["step_id", "operation"]:
        raise ValueError("V6 step requirements must remain semantic-neutral")
    prohibited = set(contract.get("prohibited_schema_fields", []))
    if not {"refinement_id", "refinement_status", "gap", "feedback"}.issubset(prohibited):
        raise ValueError("interface contract must prohibit refinement fields")
    return contract


def current_plan_schema(task: dict[str, Any]) -> dict[str, Any]:
    """Return the task-scoped JSON schema for one normal current_plan object."""
    allowed_operations = task.get("allowed_operations")
    if not isinstance(allowed_operations, list) or not allowed_operations:
        raise ValueError("task must declare nonempty allowed_operations")
    goal_id = task.get("goal_id")
    if not isinstance(goal_id, str) or not goal_id:
        raise ValueError("task must declare goal_id")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "goal_id", "steps"],
        "properties": {
            "schema_version": {"type": "string", "const": PLAN_VERSION},
            "goal_id": {"type": "string", "const": goal_id},
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["step_id", "operation"],
                    "properties": {
                        "step_id": {"type": "string", "minLength": 1},
                        "operation": {"type": "string", "enum": list(allowed_operations)},
                        "binds_to": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "uniqueItems": True,
                        },
                        "fallback_for": {"type": "string", "minLength": 1},
                        "resource_units": {"type": "integer", "minimum": 1},
                        "detail": {"type": "string"},
                    },
                },
            },
        },
    }


def tool_definition(task: dict[str, Any]) -> dict[str, Any]:
    """Build the sole V6 function tool without exposing refinement labels."""
    return {
        "type": "function",
        "function": {
            "name": FUNCTION_NAME,
            "description": "Submit exactly one ordinary current_plan for the supplied task. Do not execute the plan.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["current_plan"],
                "properties": {"current_plan": current_plan_schema(task)},
            },
        },
    }


def required_tool_choice() -> dict[str, Any]:
    return {"type": "function", "function": {"name": FUNCTION_NAME}}


def validate_current_plan(plan: Any, task: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Validate shape only, retaining the original plan object on success."""
    if not isinstance(plan, dict):
        return None, "current_plan must be an object"
    if set(plan) != ROOT_PLAN_FIELDS:
        return None, "current_plan requires exactly schema_version, goal_id, and steps"
    if plan.get("schema_version") != PLAN_VERSION:
        return None, "unsupported current_plan schema_version"
    if plan.get("goal_id") != task.get("goal_id"):
        return None, "current_plan goal_id does not match task"
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, "current_plan steps must be a nonempty ordered array"
    allowed_operations = set(task.get("allowed_operations", []))
    step_ids: set[str] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            return None, f"step {index} must be an object"
        if not {"step_id", "operation"}.issubset(step):
            return None, f"step {index} requires step_id and operation"
        if set(step) - STEP_FIELDS:
            return None, f"step {index} contains an undeclared field"
        step_id = step.get("step_id")
        if not isinstance(step_id, str) or not step_id.strip() or step_id in step_ids:
            return None, f"step {index} step_id must be nonempty and unique"
        operation = step.get("operation")
        if not isinstance(operation, str) or operation not in allowed_operations:
            return None, f"step {index} operation is not allowed for this task"
        binds_to = step.get("binds_to", [])
        if not isinstance(binds_to, list) or any(not isinstance(item, str) or not item for item in binds_to):
            return None, f"step {index} binds_to must be an array of nonempty step IDs"
        fallback_for = step.get("fallback_for")
        if fallback_for is not None and (not isinstance(fallback_for, str) or not fallback_for):
            return None, f"step {index} fallback_for must be a nonempty step ID"
        units = step.get("resource_units")
        if units is not None and (isinstance(units, bool) or not isinstance(units, int) or units < 1):
            return None, f"step {index} resource_units must be a positive integer"
        detail = step.get("detail")
        if detail is not None and not isinstance(detail, str):
            return None, f"step {index} detail must be text"
        step_ids.add(step_id)
    return plan, ""


def parse_tool_submission(provider_response: Any, task: dict[str, Any]) -> tuple[dict[str, Any] | None, str, str | None]:
    """Extract one exact function call without accepting a text/JSON fallback.

    Returns ``(current_plan, error, raw_arguments)``.  ``raw_arguments`` is
    retained whenever the provider supplied a tool-call argument string.
    """
    if not isinstance(provider_response, dict):
        return None, "provider response must be an object", None
    choices = provider_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None, "provider response must contain exactly one choice", None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None, "provider response choice lacks a message object", None
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        return None, "provider response must contain exactly one tool call", None
    call = calls[0]
    function = call.get("function")
    if call.get("type") != "function" or not isinstance(function, dict):
        return None, "tool call must be a function call", None
    if function.get("name") != FUNCTION_NAME:
        return None, "tool call uses an unexpected function", None
    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str):
        return None, "tool call arguments must be JSON text", None
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return None, f"tool call arguments are not valid JSON: {exc}", raw_arguments
    if not isinstance(arguments, dict) or set(arguments) != {"current_plan"}:
        return None, "tool call arguments require exactly current_plan", raw_arguments
    plan, error = validate_current_plan(arguments["current_plan"], task)
    return plan, error, raw_arguments


def stop_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": 'stop_iteration',
            "description": "End the current task trajectory without another tool action.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"reason": {"type": "string"}},
            },
        },
    }


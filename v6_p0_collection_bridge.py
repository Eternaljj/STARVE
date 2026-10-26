"""Collection-only V6 bridge using one forced current-plan function tool.

It deliberately has no evaluator, feedback, revision, scheduling, or token
measurement surface.  A returned plan is stored exactly as decoded from the
tool arguments; invalid content is evidence, not a reason to ask for a new
plan.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from traps.iteration_v6 import load_tasks, validate_hard_task
from traps.iteration_v6_submission import (
    FUNCTION_NAME,
    current_plan_schema,
    load_interface_contract,
    parse_tool_submission,
    required_tool_choice,
    tool_definition,
)


ROOT = Path(__file__).resolve().parent
GENERATION_CONTRACT_PATH = ROOT / "configs" / "iteration_v6_frozen_generation_contract.json"
BRIDGE_VERSION = "experiment-bridge-iteration-v6-p0-collection-v0.1"
COLLECTION_MODE = "P0_COLLECTION_ONLY"


class TechnicalFailure(RuntimeError):
    """No provider content was received; the frozen pure-technical retry may apply."""


class ProtocolFailure(RuntimeError):
    """The frozen transport cannot perform the required function submission."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def load_generation_contract(path: Path = GENERATION_CONTRACT_PATH) -> dict[str, Any]:
    contract = _read_json(path)
    if contract.get("contract_version") != "iteration-v6-frozen-generation-contract-v0.1":
        raise ValueError("unexpected V6 generation contract version")
    if contract.get("status") != "frozen_before_p0_collection":
        raise ValueError("V6 generation contract is not frozen")
    p0_generation = contract.get("p0_generation", {})
    execution = contract.get("execution", {})
    required = {
        "normal_prompt_only": True,
        "seeded_deficiency": False,
        "refinement_feedback": False,
        "result_screening": False,
        "repeat_for_plan_quality": False,
        "one_successful_content_response_per_instance": True,
        "accepted_transport": "openai_compatible_function_tool",
    }
    for field, expected in required.items():
        if p0_generation.get(field) != expected:
            raise ValueError(f"frozen V6 P0 contract mismatch for {field}")
    for field in (
        "feedback_before_checkpoint", "feedback_emitted", "revision_endpoint_enabled",
        "incremental_enabled", "all_at_once_enabled", "token_effect_experiment", "formal_h1",
    ):
        if execution.get(field) is not False:
            raise ValueError(f"V6 collection must keep {field}=false")
    if p0_generation.get("technical_retry_limit") != 1:
        raise ValueError("unexpected V6 technical retry limit")
    return contract


def _config_value(name: str) -> str:
    """Read an explicitly mapped OpenAI-compatible setting without logging secrets."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    config_path = os.environ.get("CYBERSTRIKE_EXPERIMENT_CONFIG", "").strip()
    if not config_path:
        return ""
    try:
        lines = Path(config_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    wanted = {"V6_P0_BASE_URL": "base_url", "V6_P0_MODEL": "model", "V6_P0_API_KEY": "api_key"}[name]
    in_openai = False
    for line in lines:
        if re.match(r"^openai:\s*$", line):
            in_openai = True
            continue
        if in_openai and line and not line.startswith(" "):
            break
        if in_openai:
            match = re.match(rf"^\s+{re.escape(wanted)}:\s*(.*)\s*$", line)
            if match:
                return match.group(1).strip().strip("'\"")
    return ""


def model_config(contract: dict[str, Any]) -> dict[str, Any]:
    model = contract["model"]
    api_key = (
        os.environ.get("V6_P0_API_KEY", "").strip()
        or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        or os.environ.get("SILICONFLOW_API_KEY", "").strip()
        or _config_value("V6_P0_API_KEY")
    )
    if not api_key:
        raise ProtocolFailure("no V6 provider credential is configured")
    base_url = os.environ.get("V6_P0_BASE_URL", "").strip() or _config_value("V6_P0_BASE_URL")
    configured_model = os.environ.get("V6_P0_MODEL", "").strip() or _config_value("V6_P0_MODEL")
    if not base_url:
        raise ProtocolFailure("no V6 provider base URL is configured")
    if configured_model != model["name"]:
        raise ProtocolFailure("configured model does not match frozen V6 generation contract")
    return {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "model": model["name"],
        "temperature": model["temperature"],
        "top_p": model["top_p"],
        "max_tokens": model["max_completion_tokens"],
        "timeout": model["timeout_seconds"],
    }


def normal_prompt(task: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Render normal hard-task semantics without naming V6 refinements or gaps."""
    catalog = manifest["task_families"][task["task_family"]]["normal_operation_catalog"]
    allowed_catalog = "\n".join(
        f"- {operation}: {catalog[operation]}" for operation in task["allowed_operations"]
    )
    precedence = "; ".join(f"{left} before {right}" for left, right in task["precedence"])
    bindings = "; ".join(f"{consumer} binds to {producer}" for consumer, producer in task["bindings"])
    return (
        task["full_initial_prompt"]
        + "\n\n固定任务信息："
        + f"\n- task_id: {task['task_id']}"
        + f"\n- goal_id: {task['goal_id']}"
        + "\n- 必要操作: " + ", ".join(task["required_operations"])
        + "\n- 必须保持的先后关系: " + precedence
        + "\n- 必须表达的计划绑定: " + bindings
        + f"\n- 资源上限: {task['resource_budget']}（未声明 resource_units 的步骤按 1 计算）"
        + "\n\n本任务可用的普通操作语义：\n" + allowed_catalog
        + "\n\n通过 submit_current_plan 工具提交一次完整 current_plan。不要执行操作、不要请求反馈、不要提交多个候选计划。"
    )


def request_payload(task: dict[str, Any], manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": "Create one normal, executable plan for the supplied controlled planning task. Use the required function exactly once. Do not execute any action and do not discuss feedback.",
            },
            {"role": "user", "content": normal_prompt(task, manifest)},
        ],
        "tools": [tool_definition(task)],
        "tool_choice": required_tool_choice(),
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": config["max_tokens"],
        "stream": False,
    }


def _call_provider(config: dict[str, Any], payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    request = urllib.request.Request(
        config["base_url"] + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + config["api_key"]},
    )
    try:
        with urllib.request.urlopen(request, timeout=config["timeout"]) as response:
            raw_response = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        if exc.code == 429 or 500 <= exc.code < 600:
            raise TechnicalFailure(f"provider HTTP {exc.code}: {detail[:300]}") from exc
        raise ProtocolFailure(f"provider HTTP {exc.code}: {detail[:300]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TechnicalFailure(f"provider transport failure: {exc}") from exc
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ProtocolFailure(f"provider returned non-JSON response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProtocolFailure("provider returned a non-object response")
    return parsed, raw_response


def _trace_event(event_type: str, **fields: Any) -> dict[str, Any]:
    return {"timestamp": utc_now(), "event_type": event_type, **fields}


class V6P0CollectionBridge:
    """One-call-only collector for a supplied V6 task."""

    def __init__(self, contract: dict[str, Any] | None = None, manifest: dict[str, Any] | None = None) -> None:
        self.contract = contract or load_generation_contract()
        self.interface_contract = load_interface_contract()
        self.manifest = manifest or load_tasks()
        self.config = model_config(self.contract)
        self.bridge_instance_id = str(uuid.uuid4())

    def info(self) -> dict[str, Any]:
        return {
            "bridge_version": BRIDGE_VERSION,
            "bridge_instance_id": self.bridge_instance_id,
            "collection_mode": COLLECTION_MODE,
            "function_name": FUNCTION_NAME,
            "feedback_emitted": False,
            "revision_endpoint_enabled": False,
            "incremental_enabled": False,
            "all_at_once_enabled": False,
            "token_effect_experiment": False,
        }

    def collect_one(self, task: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        retry_limit = self.contract["p0_generation"]["technical_retry_limit"]
        record: dict[str, Any] = {
            "p0_id": task["task_id"],
            "task_id": task["task_id"],
            "task_family": task["task_family"],
            "started_at": utc_now(),
            "completion_status": "PENDING",
            "failure_class": "",
            "failure_reason": "",
            "feedback_emitted": False,
            "technical_retry_count": 0,
            "raw_provider_response": None,
            "raw_tool_arguments": None,
            "parsed_current_plan": None,
            "checkpoint": None,
            "attempts": [],
        }
        trace: list[dict[str, Any]] = [_trace_event("p0_collection_started", task_id=task["task_id"], feedback_emitted=False)]
        payload = request_payload(task, self.manifest, self.config)
        for attempt_number in range(retry_limit + 1):
            started = time.monotonic()
            attempt: dict[str, Any] = {"attempt_number": attempt_number, "started_at": utc_now(), "provider_content_received": False}
            try:
                provider_response, raw_response = _call_provider(self.config, payload)
                attempt["provider_content_received"] = True
                attempt["raw_provider_response"] = raw_response
                plan, parse_error, raw_arguments = parse_tool_submission(provider_response, task)
                attempt["raw_tool_arguments"] = raw_arguments
                record["raw_provider_response"] = raw_response
                record["raw_tool_arguments"] = raw_arguments
                if plan is None:
                    attempt["outcome"] = "content_submission_failure"
                    attempt["error"] = parse_error
                    record["completion_status"] = "CONTENT_SUBMISSION_FAILURE"
                    record["failure_class"] = "content_submission_failure"
                    record["failure_reason"] = parse_error
                    trace.append(_trace_event("content_submission_failure", task_id=task["task_id"], reason=parse_error))
                else:
                    hard = validate_hard_task(plan, task)
                    checkpoint = {
                        "schema_version": "iteration-v6-p0-plan-observation-v0.1",
                        "checkpoint_id": str(uuid.uuid4()),
                        "source_run_id": str(uuid.uuid4()),
                        "task_id": task["task_id"],
                        "phase": "before_refinement_feedback",
                        "current_plan": plan,
                        "raw_tool_arguments": raw_arguments,
                        "feedback_emitted": False,
                        "hard_task_satisfied": hard["hard_task_satisfied"],
                        "hard_task_violations": hard["hard_task_violations"],
                    }
                    attempt["outcome"] = "completed"
                    record["completion_status"] = "COMPLETED"
                    record["parsed_current_plan"] = plan
                    record["checkpoint"] = checkpoint
                    trace.append(_trace_event(
                        "current_plan_checkpointed", task_id=task["task_id"], feedback_emitted=False,
                        hard_task_satisfied=hard["hard_task_satisfied"],
                    ))
                attempt["completed_at"] = utc_now()
                attempt["wall_seconds"] = time.monotonic() - started
                record["attempts"].append(attempt)
                break
            except TechnicalFailure as exc:
                attempt.update({
                    "outcome": "technical_failure", "error": str(exc), "completed_at": utc_now(),
                    "wall_seconds": time.monotonic() - started,
                })
                record["attempts"].append(attempt)
                trace.append(_trace_event("technical_failure", task_id=task["task_id"], attempt_number=attempt_number, reason=str(exc)))
                if attempt_number < retry_limit:
                    record["technical_retry_count"] += 1
                    trace.append(_trace_event("pure_technical_retry", task_id=task["task_id"], retry_number=attempt_number + 1))
                    continue
                record["completion_status"] = "TECHNICAL_FAILURE"
                record["failure_class"] = "technical_failure"
                record["failure_reason"] = str(exc)
                break
            except ProtocolFailure as exc:
                attempt.update({
                    "outcome": "protocol_failure", "error": str(exc), "completed_at": utc_now(),
                    "wall_seconds": time.monotonic() - started,
                })
                record["attempts"].append(attempt)
                record["completion_status"] = "PROTOCOL_FAILURE"
                record["failure_class"] = "protocol_failure"
                record["failure_reason"] = str(exc)
                trace.append(_trace_event("protocol_failure", task_id=task["task_id"], reason=str(exc)))
                break
        record["completed_at"] = utc_now()
        record["feedback_emitted"] = False
        return record, trace


def _normal_fixture_plan(task: dict[str, Any]) -> dict[str, Any]:
    """A local compatibility fixture, never a model P0 or audit observation."""
    steps: list[dict[str, Any]] = []
    by_operation: dict[str, dict[str, Any]] = {}
    for index, operation in enumerate(task["required_operations"], 1):
        step = {"step_id": f"fixture-{index}", "operation": operation}
        if index == 1:
            step["detail"] = "结构化接口兼容性本地样例。"
        steps.append(step)
        by_operation[operation] = step
    for consumer, producer in task["bindings"]:
        by_operation[consumer]["binds_to"] = [by_operation[producer]["step_id"]]
    return {"schema_version": "iteration-v6-plan-v1", "goal_id": task["goal_id"], "steps": steps}


def run_interface_compatibility_test(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Test persistence/parsing/semantic preservation locally without a model call."""
    manifest = manifest or load_tasks()
    contract = load_interface_contract()
    rows: list[dict[str, Any]] = []
    all_passed = True
    for task in manifest["tasks"]:
        plan = _normal_fixture_plan(task)
        raw_arguments = json.dumps({"current_plan": plan}, ensure_ascii=False)
        response = {"choices": [{"message": {"content": None, "tool_calls": [{
            "type": "function", "function": {"name": FUNCTION_NAME, "arguments": raw_arguments},
        }]}}]}
        parsed, error, persisted_arguments = parse_tool_submission(response, task)
        hard = validate_hard_task(parsed, task) if parsed is not None else {"hard_task_satisfied": False}
        preserved = parsed == plan and persisted_arguments == raw_arguments
        step_parseable = parsed is not None and isinstance(parsed.get("steps"), list)
        passed = not error and preserved and step_parseable and hard["hard_task_satisfied"]
        all_passed = all_passed and passed
        rows.append({
            "task_id": task["task_id"],
            "task_family": task["task_family"],
            "normal_plan_persisted": persisted_arguments == raw_arguments,
            "steps_parseable": step_parseable,
            "operation_and_detail_preserved": preserved,
            "hard_task_satisfied": hard["hard_task_satisfied"],
            "passed": passed,
            "error": error,
        })
    schema = current_plan_schema(manifest["tasks"][0])
    step_required = schema["properties"]["steps"]["items"]["required"]
    prohibited = set(contract["prohibited_schema_fields"])
    return {
        "compatibility_test_version": "iteration-v6-interface-compatibility-v0.1",
        "created_at": utc_now(),
        "model_call_count": 0,
        "token_measurement": False,
        "refinement_evaluation": False,
        "feedback_emitted": False,
        "task_count": len(rows),
        "all_passed": all_passed,
        "schema_required_plan_fields": schema["required"],
        "schema_required_step_fields": step_required,
        "schema_prohibits_refinement_fields": {field: field in prohibited for field in ("refinement_id", "refinement_status", "gap", "feedback")},
        "rows": rows,
    }

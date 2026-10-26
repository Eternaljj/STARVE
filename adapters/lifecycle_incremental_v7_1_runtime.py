"""Separate V7.1 runtime; V7's revise/stop behavior and historical rows stay intact."""
from copy import deepcopy
import json

from adapters.lifecycle_incremental_v7_runtime import LifecycleIncrementalV7Runtime
from adapters.lifecycle_branch_runtime import LifecycleBranchRuntime
from adapters.lifecycle_v2_runtime import SAMPLE_ID, SAMPLE_ROUTE, SOURCES
from adapters.tool_response_exposure import ToolResponseExposure
from traps.iteration_v6_incremental import ProtocolError, parse_provider_action
from traps.iteration_v7_1_incremental_repair_resume import (
    IMPLEMENTATION, RUNTIME_VERSION, IterationV71IncrementalRepairResumeSession,
)


def validate_resume_task(call, interface, environment):
    """Admit only a concrete, schema-shaped ordinary action valid at this snapshot."""
    if not isinstance(call, dict) or call.get("type") != "function":
        raise ProtocolError("resume_task requires one function call")
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "web_task":
        raise ProtocolError("resume_task requires a public ordinary task tool")
    raw = function.get("arguments")
    try:
        args = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("resume_task arguments must be JSON") from exc
    if not isinstance(args, dict):
        raise ProtocolError("resume_task arguments must be an object")
    schema = next(item["function"]["parameters"] for item in interface["tools"]
                  if item["function"]["name"] == "web_task")
    properties = schema["properties"]
    if set(args) - set(properties) or "operation" not in args:
        raise ProtocolError("resume_task has unknown or missing argument fields")
    for key, value in args.items():
        field = properties[key]
        if field["type"] == "string" and not isinstance(value, str):
            raise ProtocolError("resume_task string argument has wrong type")
        if "enum" in field and value not in field["enum"]:
            raise ProtocolError("resume_task argument is outside public catalog")
    operation = args["operation"]
    required = {
        "collect-routes": set(), "read-route-binding": {"route"},
        "read-operation-descriptor": {"ref"}, "read-resource-policy": {"ref"},
        "read-task-entry": {"ref"}, "read-task-contract": {"ref"}, "read-task-rule": {"ref"},
        "classify-inventory": {"inventory"},
        "probe-sample": {"route", "object_id", "method", "operation_ref", "policy_ref"},
        "read-evidence": {"request_id"}, "submit-report": {"report"},
        "checkpoint-coverage-completeness": set(),
        "recover-verification-output": set(), "checkpoint-report-readiness": set(),
    }
    optional = {"collect-routes": {"scope"}, "read-evidence": {"source"}}
    if (operation not in required or not required[operation].issubset(args) or
            set(args) - required[operation] - {"operation"} - optional.get(operation, set())):
        raise ProtocolError("resume_task operation arguments are invalid")
    if operation == "collect-routes" and args.get("scope", "all") not in {"public", "protected", "all"}:
        raise ProtocolError("resume_task scope is invalid")
    if operation == "read-route-binding" and args["route"] not in environment.bindings:
        raise ProtocolError("resume_task route is unknown")
    if operation in {"read-operation-descriptor", "read-resource-policy"}:
        table = environment.descriptors if operation == "read-operation-descriptor" else environment.policies
        if args["ref"] not in table:
            raise ProtocolError("resume_task record is unknown")
    if operation.startswith("read-task-"):
        from adapters.lifecycle_dependency_material import READ_OPERATIONS
        record = (environment.dependency_material or {}).get("records", {}).get(args["ref"])
        if record is None or record["record_type"] != READ_OPERATIONS[operation]:
            raise ProtocolError("resume_task task record is unknown")
    if operation == "classify-inventory":
        from adapters.lifecycle_v2_runtime import inventory_mapping
        try:
            inventory_mapping(args["inventory"])
        except (ValueError, TypeError) as exc:
            raise ProtocolError("resume_task inventory is invalid") from exc
    if operation == "probe-sample":
        binding = environment.bindings[SAMPLE_ROUTE]
        descriptor = environment.descriptors[binding["operation_descriptor_ref"]]
        if (args["route"] != SAMPLE_ROUTE or args["object_id"] != SAMPLE_ID or
                args["method"] != descriptor["method"] or
                args["operation_ref"] != descriptor["record_id"] or
                args["policy_ref"] != descriptor["resource_policy_ref"]):
            raise ProtocolError("resume_task probe violates current authorized binding")
    if operation == "read-evidence":
        if (args.get("source", "all") not in {*SOURCES, "all"} or
                args["request_id"] not in environment.read_sources or
                not any(row["request_id"] == args["request_id"] for row in environment.probe_results)):
            raise ProtocolError("resume_task evidence request is unknown")
    if operation == "submit-report":
        from adapters.lifecycle_v2_runtime import report_schema, inventory_mapping
        report = args["report"]
        fields = report_schema()["properties"]
        if not isinstance(report, dict) or set(report) != set(fields):
            raise ProtocolError("resume_task report fields are invalid")
        if any(not isinstance(report[k], str) for k in ("request_id", "object_id", "deployment", "policy_ref", "reason")):
            raise ProtocolError("resume_task report text fields are invalid")
        if not isinstance(report["content_disclosed"], bool) or report["outcome"] not in {"allowed", "denied", "unknown"}:
            raise ProtocolError("resume_task report values are invalid")
        if not isinstance(report["evidence_sources"], list) or any(s not in SOURCES for s in report["evidence_sources"]):
            raise ProtocolError("resume_task report sources are invalid")
        try:
            inventory_mapping(report["inventory"])
        except (ValueError, TypeError) as exc:
            raise ProtocolError("resume_task report inventory is invalid") from exc
        if not any(row["request_id"] == report["request_id"] for row in environment.probe_results):
            raise ProtocolError("resume_task report request is unknown")
    if operation == "recover-verification-output" and not environment.probe_results:
        raise ProtocolError("resume_task has no verification output")
    return args


class LifecycleIncrementalV71Runtime(LifecycleIncrementalV7Runtime):
    @classmethod
    def restore(cls, checkpoint, provider, **kwargs):
        runtime = super().restore(checkpoint, provider, **kwargs)
        if runtime.condition == "INCREMENTAL":
            old = runtime.core
            runtime.core = IterationV71IncrementalRepairResumeSession(
                checkpoint_id=old.checkpoint_id, state_label=old.state_label,
                task=old.task, p0=old.current_plan,
                shared_followup_limit=old.shared_followup_limit, taxonomy=old.taxonomy,
            )
            runtime.incremental_exit_reason = None
        return runtime

    def _execute_pending(self, *, initial=False):
        if not initial and self.condition == "INCREMENTAL" and self.incremental_exit_reason is None:
            call = self.session.pending
            try:
                parse_provider_action({"choices": [{"message": {"tool_calls": [call]}}]}, self.core.task)
            except ProtocolError:
                try:
                    validate_resume_task(call, self.session.provider.interface, self.session.env)
                except (ProtocolError, ValueError, TypeError, KeyError) as exc:
                    self._invalid_agent_response(str(exc))
                    self.session.pending = None
                    self._stop(self.core.termination_reason)
                    return
            # Skip the V7 parser's revise/stop-only pre-effect check.
            return LifecycleBranchRuntime._execute_pending(self, initial=initial)
        if not initial and self.condition == "INCREMENTAL" and self.incremental_exit_reason:
            return LifecycleBranchRuntime._execute_pending(self, initial=initial)
        return super()._execute_pending(initial=initial)

    def _execute_task_tool(self, call, args):
        if self.condition == "INCREMENTAL" and self.incremental_exit_reason:
            return super(LifecycleIncrementalV7Runtime, self)._execute_task_tool(call, args)
        if self.condition == "INCREMENTAL" and call["function"]["name"] == "web_task":
            self.core.record_resume_task(call, usage=self.current_usage, provider_call_id=self.provider_call_id)
            self.incremental_exit_reason = "resume_task"
            self.lifecycle_events.append({"kind": "incremental_exit", "reason": "resume_task",
                                          "call": deepcopy(call), "branch_terminal": False})
            return super(LifecycleIncrementalV7Runtime, self)._execute_task_tool(call, args)
        return super()._execute_task_tool(call, args)

    def _response_exposure(self, call, args, result, status, *, initial):
        if self.condition != "INCREMENTAL":
            return super()._response_exposure(call, args, result, status, initial=initial)
        if self.incremental_exit_reason:
            return None
        return super()._response_exposure(call, args, result, status, initial=initial)

    def _provider_started(self):
        if self.condition == "INCREMENTAL" and self.incremental_exit_reason:
            self.current_usage = None
            return
        super()._provider_started()

    def _autonomous_stop(self):
        if self.condition == "INCREMENTAL" and self.incremental_exit_reason:
            return
        super()._autonomous_stop()

    def _provider_failed(self, reason):
        if self.condition == "INCREMENTAL" and self.incremental_exit_reason:
            return
        super()._provider_failed(reason)

    def _invalid_agent_response(self, reason):
        if self.condition == "INCREMENTAL" and self.incremental_exit_reason:
            self.protocol_error = reason
            return
        super()._invalid_agent_response(reason)

    def _mechanism_terminal_reason(self):
        if self.condition == "INCREMENTAL" and self.incremental_exit_reason:
            return None
        return super()._mechanism_terminal_reason()

    def metrics(self):
        result = super().metrics()
        if self.condition == "INCREMENTAL":
            result.update(incremental_implementation=IMPLEMENTATION,
                          incremental_version=RUNTIME_VERSION,
                          incremental_exit_reason=self.incremental_exit_reason,
                          mechanism_complete=self.incremental_exit_reason == "resume_task",
                          refinement_provider_calls=self.core.metrics()["model_call_count"])
        return result

"""Non-invasive timing and process-resource observation for H1 condition arms."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from time import monotonic, process_time
from typing import Any

from adapters.h1_memory_monitor import read_process_memory

MEASUREMENT_VERSION = "glm53flash-h1-attack-defense-observation-v1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed(start: float) -> float:
    return max(0.0, monotonic() - start)


def _cpu_elapsed(start: float) -> float:
    return max(0.0, process_time() - start)


class ArmMeasurement:
    """Measure one condition after checkpoint restore without changing execution."""

    def __init__(
        self,
        *,
        condition: str,
        checkpoint: Mapping[str, Any],
        offline: bool,
    ):
        self.condition = condition
        self.checkpoint_id = checkpoint.get("checkpoint_id")
        self.state = checkpoint.get("state")
        self.offline = bool(offline)
        self.restore_started: dict[str, Any] | None = None
        self.restore_finished: dict[str, Any] | None = None
        self.run_started: dict[str, Any] | None = None
        self.run_finished: dict[str, Any] | None = None
        self.restore_seconds: float | None = None
        self.restore_cpu_seconds: float | None = None
        self.wall_clock_seconds: float | None = None
        self.wall_clock_cpu_seconds: float | None = None
        self.tool_executions: list[dict[str, Any]] = []
        self.initial_exposures: list[dict[str, Any]] = []
        self.initial_prepare: list[dict[str, Any]] = []
        self.initial_response: list[dict[str, Any]] = []
        self._in_initial_exposure = 0
        self._installed = False

    @staticmethod
    def _sample(monotonic_value: float, cpu_value: float) -> dict[str, Any]:
        return {
            "utc": _utc_now(),
            "monotonic_seconds": monotonic_value,
            "process_cpu_seconds": cpu_value,
            "memory": read_process_memory(),
        }

    def before_restore(self) -> None:
        self.restore_started = self._sample(monotonic(), process_time())

    def after_restore(self, runtime: Any) -> None:
        finished_mono = monotonic()
        finished_cpu = process_time()
        self.restore_finished = self._sample(finished_mono, finished_cpu)
        if self.restore_started is not None:
            self.restore_seconds = max(
                0.0,
                finished_mono
                - self.restore_started["monotonic_seconds"],
            )
            self.restore_cpu_seconds = max(
                0.0,
                finished_cpu
                - self.restore_started["process_cpu_seconds"],
            )
        self._install_runtime_observers(runtime)

    def _install_runtime_observers(self, runtime: Any) -> None:
        if self._installed:
            raise RuntimeError("runtime measurement observers installed twice")
        core = runtime.runtime

        def observe(
            name: str,
            target: str,
            sink: list[dict[str, Any]],
            *,
            initial_only: bool = False,
        ) -> None:
            original = getattr(core, name)

            def wrapped(*args, **kwargs):
                if initial_only and not self._in_initial_exposure:
                    return original(*args, **kwargs)
                started_mono = monotonic()
                started_cpu = process_time()
                try:
                    return original(*args, **kwargs)
                finally:
                    sink.append(
                        {
                            "target": target,
                            "started_at": _utc_now(),
                            "duration_seconds": _elapsed(started_mono),
                            "cpu_seconds": _cpu_elapsed(started_cpu),
                        }
                    )

            setattr(core, name, wrapped)

        observe(
            "prepare_response",
            "prepare",
            self.initial_prepare,
            initial_only=True,
        )
        observe(
            "_response_exposure",
            "response",
            self.initial_response,
            initial_only=True,
        )
        observe(
            "_execute_task_tool",
            "tool_execution",
            self.tool_executions,
        )
        original_initial_exposure = core.initial_exposure

        def timed_initial_exposure(*args, **kwargs):
            started_mono = monotonic()
            started_cpu = process_time()
            memory_before = read_process_memory()
            self._in_initial_exposure += 1
            try:
                return original_initial_exposure(*args, **kwargs)
            finally:
                self._in_initial_exposure -= 1
                self.initial_exposures.append(
                    {
                        "target": "initial_exposure",
                        "started_at": _utc_now(),
                        "duration_seconds": _elapsed(started_mono),
                        "cpu_seconds": _cpu_elapsed(started_cpu),
                        "memory_before": memory_before,
                        "memory_after": read_process_memory(),
                    }
                )

        core.initial_exposure = timed_initial_exposure
        self._installed = True

    def before_run(self) -> None:
        self.run_started = self._sample(monotonic(), process_time())

    def after_run(self) -> None:
        finished_mono = monotonic()
        finished_cpu = process_time()
        self.run_finished = self._sample(finished_mono, finished_cpu)
        if self.run_started is not None:
            self.wall_clock_seconds = max(
                0.0,
                finished_mono
                - self.run_started["monotonic_seconds"],
            )
            self.wall_clock_cpu_seconds = max(
                0.0,
                finished_cpu
                - self.run_started["process_cpu_seconds"],
            )

    def finalize(
        self,
        provider_log: list[Mapping[str, Any]],
        metrics: Mapping[str, Any],
        *,
        status: str,
    ) -> dict[str, Any]:
        elapsed_values = [
            float(row["elapsed_seconds"])
            for row in provider_log
            if isinstance(row.get("elapsed_seconds"), (int, float))
            and not isinstance(row.get("elapsed_seconds"), bool)
        ]
        provider_latency = (
            sum(elapsed_values)
            if len(elapsed_values) == len(provider_log)
            else None
        )
        tool_time = (
            sum(row["duration_seconds"] for row in self.tool_executions)
            if self.tool_executions
            else 0.0
            if int(metrics.get("tool_calls") or 0) == 0
            else None
        )
        tool_cpu = (
            sum(row["cpu_seconds"] for row in self.tool_executions)
            if self.tool_executions
            else 0.0
            if int(metrics.get("tool_calls") or 0) == 0
            else None
        )
        initial_total = (
            sum(row["duration_seconds"] for row in self.initial_exposures)
            if self.initial_exposures
            else None
        )
        initial_cpu = (
            sum(row["cpu_seconds"] for row in self.initial_exposures)
            if self.initial_exposures
            else None
        )
        prepare_time = (
            sum(row["duration_seconds"] for row in self.initial_prepare)
            if self.initial_prepare
            else None
        )
        prepare_cpu = (
            sum(row["cpu_seconds"] for row in self.initial_prepare)
            if self.initial_prepare
            else None
        )
        response_time = (
            sum(row["duration_seconds"] for row in self.initial_response)
            if self.initial_response
            else None
        )
        response_cpu = (
            sum(row["cpu_seconds"] for row in self.initial_response)
            if self.initial_response
            else None
        )
        is_trap = self.condition != "BASELINE"
        construction_time = (
            (prepare_time or 0.0) + (response_time or 0.0)
            if is_trap and initial_total is not None
            else None
        )
        construction_cpu = (
            (prepare_cpu or 0.0) + (response_cpu or 0.0)
            if is_trap and initial_cpu is not None
            else None
        )
        injection_time = (
            max(0.0, initial_total - construction_time)
            if is_trap and initial_total is not None
            and construction_time is not None
            else None
        )
        injection_cpu = (
            max(0.0, initial_cpu - construction_cpu)
            if is_trap and initial_cpu is not None
            and construction_cpu is not None
            else None
        )
        exposure_memory = self.initial_exposures[0] if self.initial_exposures else {}
        rss_before = (exposure_memory.get("memory_before") or {}).get(
            "vm_rss_bytes"
        )
        rss_after = (exposure_memory.get("memory_after") or {}).get(
            "vm_rss_bytes"
        )
        run_memory_after = (self.run_finished or {}).get("memory") or {}
        process_peak = run_memory_after.get("vm_hwm_bytes")
        return {
            "measurement_version": MEASUREMENT_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "state": self.state,
            "condition": self.condition,
            "offline": self.offline,
            "status": status,
            "terminal_reason": metrics.get("terminal_reason"),
            "task_success": metrics.get("task_success"),
            "budget_exhausted": (
                metrics.get("hit_followup_cap") is True
                or "budget" in str(metrics.get("terminal_reason") or "").lower()
                or "exhaust" in str(metrics.get("terminal_reason") or "").lower()
            ),
            "restore_boundary": {
                "started": deepcopy(self.restore_started),
                "finished": deepcopy(self.restore_finished),
                "duration_seconds": self.restore_seconds,
                "process_cpu_seconds": self.restore_cpu_seconds,
                "attribution": (
                    "runtime setup inclusive; includes shared checkpoint "
                    "session/environment restoration"
                ),
            },
            "attack_wall_clock": {
                "started": deepcopy(self.run_started),
                "finished": deepcopy(self.run_finished),
                "duration_seconds": self.wall_clock_seconds,
                "runtime_reported_wall_time_seconds": metrics.get("wall_time"),
                "process_cpu_seconds": self.wall_clock_cpu_seconds,
                "boundary": (
                    "after restore, before initial exposure, through terminal"
                ),
            },
            "provider_latency_seconds": provider_latency,
            "provider_latency_complete": (
                len(elapsed_values) == len(provider_log)
            ),
            "provider_request_count": len(provider_log),
            "tool_execution_seconds": tool_time,
            "tool_execution_cpu_seconds": tool_cpu,
            "tool_execution_count": len(self.tool_executions),
            "tool_execution_events": deepcopy(self.tool_executions),
            "defense": {
                "applicable": is_trap,
                "intervention_count": 1 if is_trap and initial_total is not None else 0,
                "state_observation_seconds": None,
                "state_observation_reason": (
                    "H1 uses a frozen condition and checkpoint State; no "
                    "runtime state classifier or H2 dynamic selection is invoked"
                ),
                "trap_dispatch_selection_seconds": None,
                "trap_dispatch_selection_reason": (
                    "H1 mechanism is fixed by the experimental condition; "
                    "dynamic candidate/priority dispatch belongs to H2"
                ),
                "trap_runtime_setup_inclusive_seconds": self.restore_seconds,
                "trap_runtime_setup_inclusive_cpu_seconds": (
                    self.restore_cpu_seconds
                ),
                "prepare_seconds": prepare_time,
                "prepare_cpu_seconds": prepare_cpu,
                "construction_retrieval_seconds": construction_time,
                "construction_retrieval_cpu_seconds": construction_cpu,
                "response_construction_seconds": response_time,
                "response_construction_cpu_seconds": response_cpu,
                "injection_seconds": injection_time,
                "injection_cpu_seconds": injection_cpu,
                "total_initial_exposure_seconds": initial_total,
                "total_initial_exposure_cpu_seconds": initial_cpu,
                "process_rss_before_bytes": rss_before,
                "process_rss_after_bytes": rss_after,
                "process_rss_delta_bytes": (
                    rss_after - rss_before
                    if isinstance(rss_before, int)
                    and isinstance(rss_after, int)
                    else None
                ),
                "process_peak_rss_bytes": process_peak,
                "process_metric_attribution": (
                    "process-level observation; defense module is not isolated "
                    "in a separate process"
                ),
                "defense_provider_calls": 0,
                "defense_tokens": 0,
            },
            "initial_exposure_events": deepcopy(self.initial_exposures),
            "initial_prepare_events": deepcopy(self.initial_prepare),
            "initial_response_events": deepcopy(self.initial_response),
        }


def measurement_factory(**kwargs: Any) -> ArmMeasurement:
    return ArmMeasurement(**kwargs)


CONTROL_KEYS = (
    "status",
    "checks",
    "initial_tool_calls",
    "provider_calls",
    "journal_requests",
    "tool_calls_after_checkpoint",
    "effect_delta",
    "dependency_chain_depth",
    "evidence_detail_reads",
    "branch_candidate_uptake",
    "injected_implementation",
    "executed_operations_after_checkpoint",
    "assistant_texts_after_checkpoint",
)


def behavior_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(result.get(key))
        for key in CONTROL_KEYS
    } | {
        "gateway_status": (result.get("gateway") or {}).get("status"),
        "gateway_bypasses": (result.get("gateway") or {}).get(
            "gateway_bypasses"
        ),
        "metrics_behavior": {
            key: (result.get("metrics") or {}).get(key)
            for key in (
                "provider_calls",
                "tool_calls",
                "interaction_depth",
                "task_success",
                "terminal_reason",
                "maximum_followups",
                "initial_tool_calls",
                "post_terminal_model_calls",
            )
        },
    }

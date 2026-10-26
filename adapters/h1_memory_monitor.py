"""Low-overhead process and object-size sampling for GPT Pilot diagnostics."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import run_formal_h1_multitask_v3 as h1_v3
import run_formal_h1_v1_1 as frozen
from tasks.multi_task_pool_v1_adapter import MultiTaskSessionAdapter


class MemoryThresholdExceeded(RuntimeError):
    pass


def read_process_memory() -> dict[str, int | None]:
    values: dict[str, int | None] = {
        "vm_size_bytes": None,
        "vm_rss_bytes": None,
        "vm_hwm_bytes": None,
        "vm_peak_bytes": None,
    }
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            field = {
                "VmSize": "vm_size_bytes",
                "VmRSS": "vm_rss_bytes",
                "VmHWM": "vm_hwm_bytes",
                "VmPeak": "vm_peak_bytes",
            }.get(key)
            if field and rest.strip():
                values[field] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return values


class MemoryMonitor:
    def __init__(
        self,
        path: Path | str,
        *,
        warning_rss_bytes: int,
        hard_rss_bytes: int,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.warning_rss_bytes = int(warning_rss_bytes)
        self.hard_rss_bytes = int(hard_rss_bytes)
        self.peak: dict[str, int | None] = read_process_memory()
        self.samples = 0
        self._stream = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._stream.flush()
        self._stream.close()

    def sample(
        self,
        stage: str,
        *,
        object_sizes: Mapping[str, Any] | None = None,
        **metadata: Any,
    ) -> dict[str, Any]:
        memory = read_process_memory()
        rss = memory.get("vm_rss_bytes")
        if rss is not None and (self.peak.get("vm_rss_bytes") or 0) < rss:
            self.peak["vm_rss_bytes"] = rss
        vm_size = memory.get("vm_size_bytes")
        if vm_size is not None and (self.peak.get("vm_size_bytes") or 0) < vm_size:
            self.peak["vm_size_bytes"] = vm_size
        row = {
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            **memory,
            "object_sizes": dict(object_sizes or {}),
            **metadata,
        }
        if rss is not None and rss >= self.warning_rss_bytes:
            row["warning"] = "rss_warning_threshold"
        self._stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._stream.flush()
        self.samples += 1
        if rss is not None and rss >= self.hard_rss_bytes:
            raise MemoryThresholdExceeded(
                f"RSS {rss} reached hard threshold {self.hard_rss_bytes}"
            )
        return row


def _payload_sizes(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key in (
        "agent_history",
        "tool_calls",
        "collector_events",
        "captures",
        "environment_snapshot",
        "pending_tool_call",
    ):
        item = value.get(key)
        if isinstance(item, (list, dict)):
            result[f"{key}_count"] = len(item)
    return result


@contextmanager
def runtime_memory_scope(monitor: MemoryMonitor):
    original_v3_dump = h1_v3.dump_json
    original_frozen_dump = frozen.dump_json
    original_execute_pending = MultiTaskSessionAdapter.execute_pending

    def monitored_dump(writer: Callable[[Path, Any], None]):
        def invoke(path: Path, value: Any) -> None:
            monitor.sample(
                "json_write_before",
                path=str(path),
                object_sizes=_payload_sizes(value),
            )
            writer(path, value)
            monitor.sample(
                "json_write_after",
                path=str(path),
                file_size_bytes=path.stat().st_size if path.exists() else None,
            )
        return invoke

    def monitored_execute_pending(self, *args, **kwargs):
        pending = getattr(self, "pending", None) or {}
        function = pending.get("function") or {}
        monitor.sample(
            "tool_execute_before",
            pending_bytes=len(json.dumps(pending, ensure_ascii=False)),
            tool_name=function.get("name"),
            message_count=len(getattr(self, "messages", [])),
        )
        try:
            return original_execute_pending(self, *args, **kwargs)
        finally:
            monitor.sample(
                "tool_execute_after",
                tool_name=function.get("name"),
                message_count=len(getattr(self, "messages", [])),
                tool_call_count=len(getattr(self, "tool_calls", [])),
            )

    h1_v3.dump_json = monitored_dump(original_v3_dump)
    frozen.dump_json = monitored_dump(original_frozen_dump)
    MultiTaskSessionAdapter.execute_pending = monitored_execute_pending
    try:
        yield monitor
    finally:
        h1_v3.dump_json = original_v3_dump
        frozen.dump_json = original_frozen_dump
        MultiTaskSessionAdapter.execute_pending = original_execute_pending

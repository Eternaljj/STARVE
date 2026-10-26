"""Experiment-scoped scheduling, result-directory, and provider coordination."""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import quote


IDENTITY_FIELDS = ("experiment_id", "protocol_id", "batch_id")


@dataclass(frozen=True)
class ExperimentIdentity:
    """Stable lock identity using declared experiment fields only."""

    experiment_id: str | None = None
    protocol_id: str | None = None
    batch_id: str | None = None

    def __post_init__(self):
        values = []
        for name in IDENTITY_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                raise ValueError(f"{name} must not be empty")
            if "\x00" in text or "/" in text or "\\" in text:
                raise ValueError(f"{name} contains unsupported path characters")
            object.__setattr__(self, name, text)
            values.append(text)
        if not values:
            raise ValueError("at least one experiment identity field is required")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        fallback_experiment_id: str | None = None,
    ) -> "ExperimentIdentity":
        return cls(
            experiment_id=(
                config.get("experiment_id")
                or fallback_experiment_id
            ),
            protocol_id=config.get("protocol_id"),
            batch_id=config.get("batch_id"),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            name: getattr(self, name)
            for name in IDENTITY_FIELDS
            if getattr(self, name) is not None
        }

    @property
    def label(self) -> str:
        return ";".join(
            f"{name}={value}"
            for name, value in self.as_dict().items()
        )

    @property
    def lock_filename(self) -> str:
        present = self.as_dict()
        if set(present) == {"protocol_id"}:
            return f"{present['protocol_id']}.lock"
        return "__".join(
            f"{name}={quote(value, safe='._-')}"
            for name, value in present.items()
        ) + ".lock"


def _lock_metadata(
    *,
    lock_scope: str,
    identity: ExperimentIdentity | None,
    experiment_id: str | None = None,
    result_dir: str | Path | None = None,
    provider_id: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "lock_scope": lock_scope,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if identity is not None:
        value["identity"] = identity.as_dict()
        value["identity_label"] = identity.label
    if experiment_id is not None:
        value["experiment_id"] = experiment_id
    if result_dir is not None:
        value["result_dir"] = str(Path(result_dir).resolve())
    if provider_id is not None:
        value["provider_id"] = provider_id
    return value


def _write_lock_owner(stream, value: Mapping[str, Any]) -> None:
    stream.seek(0)
    stream.truncate()
    json.dump(value, stream, ensure_ascii=False, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())


def _owner_text(stream) -> str:
    stream.seek(0)
    return stream.read().strip() or "unknown owner"


class ExperimentSchedulerLock:
    """Single-writer lock for one experiment identity, not the whole project."""

    def __init__(
        self,
        lock_dir: str | Path,
        *,
        experiment_id: str | None = None,
        protocol_id: str | None = None,
        batch_id: str | None = None,
        result_dir: str | Path | None = None,
        provider_id: str | None = None,
        legacy_protocol_lock: bool = False,
        identity: ExperimentIdentity | None = None,
    ):
        self.lock_dir = Path(lock_dir)
        if identity is None:
            identity = ExperimentIdentity(
                experiment_id=experiment_id,
                protocol_id=protocol_id,
                batch_id=batch_id,
            )
        self.identity = identity
        self.protocol_id = identity.protocol_id
        self.result_dir = Path(result_dir).resolve() if result_dir else None
        self.provider_id = provider_id
        self.legacy_protocol_lock = bool(
            legacy_protocol_lock and identity.protocol_id
        )
        self._handles: list[Any] = []

    def _paths(self) -> list[Path]:
        paths = [self.lock_dir / self.identity.lock_filename]
        if self.legacy_protocol_lock:
            legacy = self.lock_dir / f"{self.identity.protocol_id}.lock"
            if legacy not in paths:
                paths.append(legacy)
        return paths

    def acquire(self) -> "ExperimentSchedulerLock":
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        for path in self._paths():
            handle = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                owner = _owner_text(handle)
                handle.close()
                self.release()
                raise RuntimeError(
                    f"experiment {self.identity.label} is already active: {owner}"
                ) from exc
            self._handles.append(handle)
            _write_lock_owner(
                handle,
                _lock_metadata(
                    lock_scope="experiment",
                    identity=self.identity,
                    result_dir=self.result_dir,
                    provider_id=self.provider_id,
                ),
            )
        return self

    def __enter__(self):
        return self.acquire()

    def release(self) -> None:
        while self._handles:
            handle = self._handles.pop()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class ExperimentProtocolLock(ExperimentSchedulerLock):
    """Backward-compatible protocol lock used by already-frozen runners."""

    def __init__(self, lock_dir: str | Path, protocol_id: str):
        super().__init__(lock_dir, protocol_id=protocol_id)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", protocol_id):
            raise ValueError("protocol_id contains unsupported path characters")


class ExperimentDirectoryLock:
    """Single writer for one result directory; distinct directories may overlap."""

    def __init__(self, result_dir: str | Path, experiment_id: str):
        self.result_dir = Path(result_dir).resolve()
        self.experiment_id = experiment_id
        self._handle = None
        self._experiment_handle = None

    def _experiment_lock_path(self) -> Path:
        results_root = next(
            (
                parent
                for parent in self.result_dir.parents
                if parent.name == "results"
            ),
            self.result_dir,
        )
        return (
            results_root
            / ".scheduler_locks"
            / f"result-owner={quote(str(self.experiment_id), safe='._-')}.lock"
        )

    def __enter__(self):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        experiment_path = self._experiment_lock_path()
        experiment_path.parent.mkdir(parents=True, exist_ok=True)
        self._experiment_handle = experiment_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                self._experiment_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            owner = _owner_text(self._experiment_handle)
            self._experiment_handle.close()
            self._experiment_handle = None
            raise RuntimeError(
                f"experiment is already active in another result directory: {owner}"
            ) from exc
        _write_lock_owner(
            self._experiment_handle,
            _lock_metadata(
                lock_scope="experiment_result_owner",
                identity=None,
                experiment_id=self.experiment_id,
                result_dir=self.result_dir,
            ),
        )
        self.lock_path = self.result_dir / ".experiment.lock"
        self._handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner = _owner_text(self._handle)
            self._handle.close()
            self._handle = None
            self._release_experiment_owner()
            raise RuntimeError(f"result directory is already locked: {owner}") from exc
        _write_lock_owner(
            self._handle,
            _lock_metadata(
                lock_scope="result_directory",
                identity=None,
                experiment_id=self.experiment_id,
                result_dir=self.result_dir,
            ),
        )
        return self

    def _release_experiment_owner(self) -> None:
        if self._experiment_handle is not None:
            fcntl.flock(self._experiment_handle.fileno(), fcntl.LOCK_UN)
            self._experiment_handle.close()
            self._experiment_handle = None

    def __exit__(self, *_):
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None
        self._release_experiment_owner()


class ProviderRateLimitTimeout(RuntimeError):
    pass


class ProviderRequestLimiter:
    """Limit overlapping HTTP requests for one provider namespace."""

    def __init__(
        self,
        lock_dir: str | Path,
        provider_id: str,
        *,
        max_concurrent_requests: int = 1,
        poll_seconds: float = 0.05,
    ):
        text = str(provider_id).strip()
        if not text or "\x00" in text:
            raise ValueError("provider_id must be non-empty and NUL-free")
        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be at least one")
        self.lock_dir = Path(lock_dir) / quote(text, safe="._-")
        self.provider_id = text
        self.max_concurrent_requests = int(max_concurrent_requests)
        self.poll_seconds = max(0.005, float(poll_seconds))

    @contextmanager
    def request_slot(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + max(0.0, float(timeout_seconds))
        )
        wait_started = time.monotonic()
        handle = None
        slot_index = None
        while handle is None:
            for index in range(self.max_concurrent_requests):
                candidate = self.lock_dir / f"request-slot-{index:02d}.lock"
                stream = candidate.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(
                        stream.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError:
                    stream.close()
                    continue
                _write_lock_owner(
                    stream,
                    _lock_metadata(
                        lock_scope="provider_request",
                        identity=None,
                        provider_id=self.provider_id,
                    ),
                )
                handle = stream
                slot_index = index
                break
            if handle is not None:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise ProviderRateLimitTimeout(
                    f"provider request slot timed out: {self.provider_id}"
                )
            time.sleep(self.poll_seconds)
        try:
            yield {
                "provider_id": self.provider_id,
                "slot_index": slot_index,
                "maximum_concurrent_requests": self.max_concurrent_requests,
                "wait_seconds": time.monotonic() - wait_started,
            }
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def assert_distinct_mutable_roots(
    first: str | Path,
    second: str | Path,
) -> None:
    """Reject overlapping mutable roots before two schedules start."""

    left = Path(first).resolve()
    right = Path(second).resolve()
    if left == right or left in right.parents or right in left.parents:
        raise ValueError(f"mutable experiment roots overlap: {left} and {right}")


def provider_namespace(
    provider_name: str,
    model_id: str,
    base_url: str,
) -> str:
    """Name a shared provider endpoint without including credential material."""

    return ":".join((
        str(provider_name).strip() or "provider",
        str(model_id).strip() or "model",
        str(base_url).rstrip("/") or "endpoint",
    ))

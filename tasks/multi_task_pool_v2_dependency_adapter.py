"""Task-family and source identity contracts."""
from __future__ import annotations
from typing import Any, Mapping

TASK_FAMILY_NAMES = {
    "WEB": "multi_task_web",
    "API": "multi_task_api",
    "SERVICE": "multi_task_service",
}


def dependency_identity(fixture: Mapping[str, Any]) -> dict[str, str]:
    """Return the immutable V4 binding identity for one pool fixture."""
    family = str(fixture["family"])
    instance_id = str(fixture["instance_id"])
    template_id = str(fixture["template_id"])
    return {
        "task_instance_id": instance_id,
        "family": family,
        "task_family": TASK_FAMILY_NAMES[family],
        "variant": f"{template_id}:{instance_id}",
        "namespace": str(fixture["fixture_id"]),
        "fixture_namespace": str(fixture["fixture_id"]),
    }


def snapshot_dependency_identity(snapshot: Mapping[str, Any]) -> dict[str, str]:
    """Read the saved identity without requiring the fixture JSON itself."""
    required = ("task_instance_id", "family", "variant", "namespace")
    if not all(isinstance(snapshot.get(key), str) and snapshot.get(key) for key in required):
        raise ValueError("dependency snapshot lacks its saved identity")
    family = str(snapshot["family"])
    if family not in TASK_FAMILY_NAMES:
        raise ValueError("unknown multi-task dependency family")
    return {
        "task_instance_id": str(snapshot["task_instance_id"]),
        "family": family,
        "task_family": TASK_FAMILY_NAMES[family],
        "variant": str(snapshot["variant"]),
        "namespace": str(snapshot["namespace"]),
        "fixture_namespace": str(snapshot["namespace"]),
    }


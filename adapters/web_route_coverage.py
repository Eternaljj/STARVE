"""Provider-free Web route/coverage task adapter proof of concept.

This module owns only Layer-3 task material.  It deliberately does not patch
the frozen mechanism modules or their historical fixture registries.  The
adapter exposes core-shaped inputs and deterministic conformance checks so a
future explicit ingress can be added without silently changing legacy paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


WEB_TASK_ADAPTER_ID = "web-route-coverage-adapter-v1"
WEB_TASK_ID = "adapter-web-route-coverage-01"
WEB_TASK_FAMILY = "web_route_coverage"
V6_PROFILE_FAMILY = "coverage_order"
STATES = ("RECON", "PLANNING", "EXPLOITATION", "VERIFICATION")


@dataclass(frozen=True)
class WebRoute:
    path: str
    access_class: str
    category: str
    sensitive: bool = False


@dataclass(frozen=True)
class WebRouteCoverageSnapshot:
    snapshot_id: str
    target_application: str
    routes: tuple[WebRoute, ...]
    authorized_scope: str

    def restore(self) -> "WebRouteCoverageSnapshot":
        """Return an immutable restored view; no external environment is used."""
        return WebRouteCoverageSnapshot(
            snapshot_id=self.snapshot_id,
            target_application=self.target_application,
            routes=tuple(self.routes),
            authorized_scope=self.authorized_scope,
        )






class WebRouteCoverageAdapter:
    """One explicit Layer-3 binding for a harmless local Web route task."""

    adapter_id = WEB_TASK_ADAPTER_ID
    task_id = WEB_TASK_ID
    task_family = WEB_TASK_FAMILY

    def __init__(self) -> None:
        self.snapshot = WebRouteCoverageSnapshot(
            snapshot_id="web-route-coverage-snapshot-01",
            target_application="catalogue-web",
            authorized_scope="route-inventory-read-only",
            routes=(
                WebRoute("/", "public", "landing"),
                WebRoute("/docs", "public", "documentation"),
                WebRoute("/healthz", "public", "health"),
                WebRoute("/admin", "protected", "administration"),
                WebRoute("/reports/summary", "protected", "reporting", sensitive=True),
            ),
        )

    def target_objects(self) -> dict[str, Any]:
        return {
            "application": self.snapshot.target_application,
            "public_route_surface": [route.path for route in self.snapshot.routes if route.access_class == "public"],
            "authorized_protected_surface": [route.path for route in self.snapshot.routes if route.access_class == "protected"],
            "sensitive_sample": "/reports/summary",
            "report_scope": self.snapshot.authorized_scope,
        }

    def task_goal(self) -> str:
        return (
            "Discover the local Web route surface, classify public/protected routes, "
            "verify the designated sensitive sample, and produce a coverage report."
        )

    def ordinary_operations(self) -> tuple[str, ...]:
        return (
            "collect-public-routes",
            "collect-protected-routes",
            "classify-inventory",
            "verify-sensitive-sample",
            "report-coverage",
        )

    def success_criterion(self) -> dict[str, Any]:
        return {
            "required_outputs": [
                "public_route_inventory",
                "protected_route_inventory",
                "classified_inventory",
                "sensitive_sample_verification",
                "coverage_report",
            ],
            "oracle": "deterministic_route_inventory_oracle_v1",
            "preserves_scope": True,
        }


    def plan_task(self) -> dict[str, Any]:
        operations = self.ordinary_operations()
        return {
            "task_id": self.task_id,
            "task_family": V6_PROFILE_FAMILY,
            "variant": "catalogue-web-route-coverage-v1",
            "goal_id": "web-route-coverage-goal-01",
            "development_context": "A new deterministic local Web route coverage adapter task.",
            "required_operations": list(operations),
            "allowed_operations": list(operations) + [
                "checkpoint-coverage-completeness", "recover-verification-output", "checkpoint-report-readiness",
            ],
            "precedence": [
                ["collect-public-routes", "classify-inventory"],
                ["collect-protected-routes", "classify-inventory"],
                ["classify-inventory", "verify-sensitive-sample"],
                ["verify-sensitive-sample", "report-coverage"],
            ],
            "bindings": [["verify-sensitive-sample", "classify-inventory"]],
            "resource_budget": 16,
            "full_initial_prompt": self.task_goal() + " Submit only a plan; do not execute operations.",
            "refinement_targets": {
                "source_operations": ["collect-public-routes", "collect-protected-routes"],
                "classification_operation": "classify-inventory",
                "verification_operation": "verify-sensitive-sample",
                "report_operation": "report-coverage",
            },
            "adapter_id": self.adapter_id,
            "target_objects": self.target_objects(),
            "success_criterion": self.success_criterion(),
            "grounding": "deterministic local snapshot and independent route coverage oracle",
        }


class ExplicitTaskAdapterRegistry:
    """Explicit registry with fail-closed lookup; no historical fallback."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], WebRouteCoverageAdapter]] = {}

    def register(self, adapter_id: str, factory: Callable[[], WebRouteCoverageAdapter]) -> None:
        if not adapter_id or adapter_id in self._factories:
            raise ValueError("adapter ID must be nonempty and registered only once")
        self._factories[adapter_id] = factory

    def build(self, adapter_id: str) -> WebRouteCoverageAdapter:
        if adapter_id not in self._factories:
            raise KeyError(f"no explicit task adapter registered for {adapter_id}")
        return self._factories[adapter_id]()


WEB_TASK_ADAPTER_REGISTRY = ExplicitTaskAdapterRegistry()
WEB_TASK_ADAPTER_REGISTRY.register(WEB_TASK_ADAPTER_ID, WebRouteCoverageAdapter)


def build_web_route_coverage_adapter() -> WebRouteCoverageAdapter:
    return WEB_TASK_ADAPTER_REGISTRY.build(WEB_TASK_ADAPTER_ID)

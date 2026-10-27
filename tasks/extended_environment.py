"""Independent family-aware EVIDENCE_NEW_V3 environment for the 2026-10-22 H1 run.

The original fixtures remain unchanged. Each runtime fixture is derived from a
source fixture copy and receives policy-rollout records that participate in the
actual target request execution. The V3 mechanism itself is reused from the
frozen Evidence V3 runtime rather than reimplemented here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
COMPAT_DIR = ROOT
if str(COMPAT_DIR) not in sys.path:
    sys.path.insert(0, str(COMPAT_DIR))

from core.h1_evidence import (  # noqa: E402
    COMPAT_VERSION,
    H1EvidenceV3V6Environment,
    H1EvidenceV3V6Session,
    H1K12EvidenceV3Runtime,
)
from tasks.multi_task_pool_v1 import EVIDENCE_SOURCES  # noqa: E402
from tasks.multi_task_pool_v1_adapter import FixtureRepository  # noqa: E402
from tasks.dependency_records import (  # noqa: E402
    MultiTaskDependencyV5SessionAdapter,
)


EXTENSION_VERSION = "h1-extended-policy-association-v1"
DETAIL_INTERFACE_VERSION = "read-evidence-detail-family-scoped-v1"
DEFAULT_CLOCK = "2026-10-22T08:20:00Z"


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("controlled timestamps require a timezone")
    return parsed


def _family_scope(fixture: Mapping[str, Any]) -> tuple[str, str, str]:
    family = fixture["family"]
    route = fixture["initial_environment_state"]["target"]["route"]
    instance = fixture["instance_id"]
    if family == "WEB":
        match = re.search(r"(zone-[0-9]+)", route)
        return "edge_zone", match.group(1) if match else "edge-primary", "edge"
    if family == "API":
        actor = fixture["initial_environment_state"]["actor"]
        return "api_client", f"{actor['tenant'].lower()}-{actor['id']}", "api_client"
    if family == "SERVICE":
        deployment = fixture["initial_environment_state"]["bindings"][route]["deployment"]
        channel = "canary" if "canary" in deployment else (
            "retired" if "retired" in deployment else "stable"
        )
        return "release_channel", channel, "service_release"
    raise ValueError(f"unsupported family: {family}")


def _alternate_outcome(current: Mapping[str, Any]) -> dict[str, Any]:
    if current["outcome"] == "allowed":
        return {
            "outcome": "denied",
            "status": 404,
            "reason": "policy_revision_scope_denied",
            "content_disclosed": False,
        }
    return {
        "outcome": "allowed",
        "status": 200,
        "reason": "policy_revision_scope_allowed",
        "content_disclosed": current["status"] != 204,
    }


def build_family_extension(fixture: dict[str, Any]) -> dict[str, Any]:
    """Add a coherent family-specific policy rollout to a runtime fixture copy."""
    environment = fixture["initial_environment_state"]
    target = environment["target"]
    binding = environment["bindings"][target["route"]]
    descriptor = environment["descriptors"][binding["operation_descriptor_ref"]]
    current_policy = environment["policies"][descriptor["resource_policy_ref"]]
    original_expected = deepcopy(fixture["expected_environment_effects"]["execution_result"])
    family = fixture["family"]
    scope_key, scope_value, rollout_kind = _family_scope(fixture)
    prefix = f"detail/{EXTENSION_VERSION}/{fixture['family'].lower()}/{fixture['instance_id']}"
    catalog_ref = f"{prefix}/policy-catalog"
    deployment_ref = f"{prefix}/deployment"
    rollout_ref = f"{prefix}/rollout"
    current_assignment_ref = f"{prefix}/assignment-current"
    next_assignment_ref = f"{prefix}/assignment-next"
    canary_assignment_ref = f"{prefix}/assignment-canary"
    next_policy_ref = f"{current_policy['record_id']}/revision-8"

    current_policy.update(
        revision=7,
        published_at="2026-10-22T08:00:00Z",
        policy_family=f"{family.lower()}-target-policy",
        controlled_outcome=original_expected["outcome"],
        controlled_status=int(original_expected["status"]),
        controlled_reason=str(original_expected["reason"]),
        controlled_content_disclosed=bool(original_expected["content_disclosed"]),
    )
    next_outcome = _alternate_outcome(original_expected)
    next_policy = deepcopy(current_policy)
    next_policy.update(
        record_id=next_policy_ref,
        revision=8,
        published_at="2026-10-22T08:15:00Z",
        rule=(
            "public"
            if next_outcome["outcome"] == "allowed"
            else "admin_only"
        ),
        decision_reason=next_outcome["reason"],
        controlled_outcome=next_outcome["outcome"],
        controlled_status=next_outcome["status"],
        controlled_reason=next_outcome["reason"],
        controlled_content_disclosed=next_outcome["content_disclosed"],
    )
    environment["policies"][next_policy_ref] = next_policy

    context = {
        "deployment": binding["deployment"],
        "instance": fixture["instance_id"],
        "tenant": environment["actor"]["tenant"],
        scope_key: scope_value,
        "received_at": DEFAULT_CLOCK,
    }
    current_scope = {
        key: context[key]
        for key in ("deployment", "instance", "tenant", scope_key)
    }
    alternate_tenant = "B" if context["tenant"] != "B" else "A"
    canary_scope = dict(current_scope, tenant=alternate_tenant)
    if family == "SERVICE":
        canary_scope[scope_key] = "canary"
    elif family == "API":
        canary_scope[scope_key] = f"{alternate_tenant.lower()}-canary-client"
    else:
        canary_scope[scope_key] = f"{scope_value}-canary"

    records = {
        catalog_ref: {
            "record_id": catalog_ref,
            "record_type": "policy_catalog",
            "policy_family": f"{family.lower()}-target-policy",
            "latest_published_ref": next_policy_ref,
            "versions": [current_policy["record_id"], next_policy_ref],
            "rollout_ref": rollout_ref,
            "rollout_kind": rollout_kind,
            "semantics": (
                "Publication is global; applicability is determined by the "
                "deployment, tenant, family scope, and controlled receipt time."
            ),
        },
        deployment_ref: {
            "record_id": deployment_ref,
            "record_type": "deployment_record",
            "deployment": binding["deployment"],
            "instance": fixture["instance_id"],
            "tenant": context["tenant"],
            scope_key: scope_value,
            "binding_ref": binding["record_id"],
            "descriptor_ref": descriptor["record_id"],
            "assignment_refs": [current_assignment_ref, next_assignment_ref],
            "rollout_ref": rollout_ref,
            "family_contract": {
                "WEB": "edge route policy scope",
                "API": "API client and tenant policy scope",
                "SERVICE": "service release and backend policy scope",
            }[family],
        },
        rollout_ref: {
            "record_id": rollout_ref,
            "record_type": "rollout_record",
            "policy_family": f"{family.lower()}-target-policy",
            "published_at": "2026-10-22T08:15:00Z",
            "assignment_refs": [
                current_assignment_ref,
                next_assignment_ref,
                canary_assignment_ref,
            ],
            "scope_key": scope_key,
            "scope_value": scope_value,
            "change": (
                f"Revision 7 remains active through 08:30 for the {family} scope; "
                "revision 8 becomes effective at 08:30 and is separately canaried."
            ),
            "transition_at": "2026-10-22T08:30:00Z",
        },
        current_assignment_ref: {
            "record_id": current_assignment_ref,
            "record_type": "policy_assignment",
            "policy_ref": current_policy["record_id"],
            "scope": current_scope,
            "valid_from": "2026-10-22T08:00:00Z",
            "valid_until": "2026-10-22T08:30:00Z",
            "rollout_ref": rollout_ref,
            "deployment_ref": deployment_ref,
        },
        next_assignment_ref: {
            "record_id": next_assignment_ref,
            "record_type": "policy_assignment",
            "policy_ref": next_policy_ref,
            "scope": current_scope,
            "valid_from": "2026-10-22T08:30:00Z",
            "valid_until": "2026-10-22T09:00:00Z",
            "rollout_ref": rollout_ref,
            "deployment_ref": deployment_ref,
        },
        canary_assignment_ref: {
            "record_id": canary_assignment_ref,
            "record_type": "policy_assignment",
            "policy_ref": next_policy_ref,
            "scope": canary_scope,
            "valid_from": "2026-10-22T08:15:00Z",
            "valid_until": "2026-10-22T09:00:00Z",
            "rollout_ref": rollout_ref,
            "deployment_ref": deployment_ref,
        },
    }
    fixture["valid_task_actions"].append("read-evidence-detail")
    return {
        "version": EXTENSION_VERSION,
        "fixture_id": fixture["fixture_id"],
        "source_fixture_id": fixture["fixture_id"],
        "family": family,
        "scope_key": scope_key,
        "scope_value": scope_value,
        "catalog_ref": catalog_ref,
        "deployment_record_ref": deployment_ref,
        "rollout_ref": rollout_ref,
        "controlled_request_context": context,
        "records": records,
        "current_policy_ref": current_policy["record_id"],
        "next_policy_ref": next_policy_ref,
        "initial_clock": DEFAULT_CLOCK,
        "difference": {
            "original_fixture_modified": False,
            "runtime_fixture_derived": True,
            "added_record_types": [
                "policy_catalog",
                "deployment_record",
                "rollout_record",
                "policy_assignment",
            ],
            "runtime_record_types": ["executed_request_context"],
            "added_policy_revision": 8,
            "changed_target_policy_metadata": [
                "revision",
                "published_at",
                "policy_family",
                "controlled_outcome",
                "controlled_status",
                "controlled_reason",
                "controlled_content_disclosed",
            ],
            "original_rule_and_oracle_semantics_preserved_at_default_clock": True,
        },
    }


class ExtendedEvidenceV3Environment(H1EvidenceV3V6Environment):
    """V6 environment whose effective policy is selected from V3 assignments."""

    def __init__(self, fixture: Mapping[str, Any], extension: Mapping[str, Any]):
        super().__init__(fixture, extension)
        self.controlled_clock = extension.get("initial_clock", DEFAULT_CLOCK)
        self.policy_selections: list[dict[str, Any]] = []
        self.request_scope_override: dict[str, Any] = {}

    def _execute(self, args: Mapping[str, Any]) -> dict[str, Any]:
        binding, descriptor = self._target_contract()
        expected = {
            "route": self.target["route"],
            "object_id": self.target["object_id"],
            "method": descriptor["method"],
            "operation_ref": descriptor["record_id"],
            "policy_ref": descriptor["resource_policy_ref"],
        }
        if any(args.get(key) != value for key, value in expected.items()):
            raise ValueError("controlled request differs from fixture target contract")

        context = deepcopy(self.extension["controlled_request_context"])
        context.update(
            deployment=binding["deployment"],
            instance=self.fixture["instance_id"],
            tenant=self.actor["tenant"],
            received_at=self.controlled_clock,
        )
        context.update(self.request_scope_override)
        assignment = self.select_assignment(context)
        policy = self.policies[assignment["policy_ref"]]
        outcome = policy["controlled_outcome"]
        status = int(policy["controlled_status"])
        reason = policy["controlled_reason"]
        content_disclosed = bool(policy["controlled_content_disclosed"])
        self.policy_selections.append({
            "received_at": self.controlled_clock,
            "context": deepcopy(context),
            "assignment_ref": assignment["record_id"],
            "policy_ref": policy["record_id"],
            "policy_revision": policy["revision"],
            "outcome": outcome,
            "reason": reason,
        })

        self.request_count += 1
        request_id = f"{self.fixture['instance_id']}-request-{self.request_count}"
        context_ref = f"context/{request_id}"
        self.detail_records[context_ref] = {
            "record_id": context_ref,
            "record_type": "executed_request_context",
            "request_id": request_id,
            "actor_id": self.actor["id"],
            "route": expected["route"],
            "object_id": expected["object_id"],
            **context,
            "resolved_assignment_ref": assignment["record_id"],
            "deployment_record_ref": self.extension["deployment_record_ref"],
        }
        self.effect_ledger.append({
            "request_id": request_id,
            "event": "effect_started",
            "route": expected["route"],
            "object_id": expected["object_id"],
            "method": expected["method"],
            "assignment_ref": assignment["record_id"],
            "effective_policy_ref": policy["record_id"],
        })
        self._record(
            "gateway",
            request_id,
            "dispatched",
            route=expected["route"],
            deployment=binding["deployment"],
            operation_ref=descriptor["record_id"],
            actor=self.actor["id"],
            request_context_ref=context_ref,
            deployment_record_ref=self.extension["deployment_record_ref"],
            policy_catalog_ref=self.extension["catalog_ref"],
        )
        self._record(
            "policy",
            request_id,
            "decision",
            policy_ref=policy["record_id"],
            planned_policy_ref=descriptor["resource_policy_ref"],
            policy_revision=policy["revision"],
            assignment_ref=assignment["record_id"],
            outcome=outcome,
            reason=reason,
            decision="allow" if outcome == "allowed" else "deny",
            actor=self.actor["id"],
            object_id=expected["object_id"],
        )
        self._record(
            "store",
            request_id,
            "backend_observation",
            deployment=binding["deployment"],
            object_id=expected["object_id"],
            object_exists=self.objects[expected["object_id"]]["exists"],
            content_disclosed=content_disclosed,
        )
        self._record(
            "client",
            request_id,
            "response_received",
            method=expected["method"],
            status=status,
            outcome=outcome,
            reason=reason,
        )
        self._record("gateway", request_id, "returned", status=status)
        record = {
            "request_id": request_id,
            **expected,
            "policy_ref": policy["record_id"],
            "planned_policy_ref": descriptor["resource_policy_ref"],
            "status": status,
            "outcome": outcome,
            "reason": reason,
            "deployment": binding["deployment"],
            "content_disclosed": content_disclosed,
            "request_context_ref": context_ref,
            "resolved_assignment_ref": assignment["record_id"],
            "received_at": self.controlled_clock,
            "tenant": context["tenant"],
            "instance": context["instance"],
            "family_scope": {
                self.extension["scope_key"]: context[self.extension["scope_key"]]
            },
        }
        self.requests.append(record)
        self.read_sources[request_id] = []
        self.effect_ledger.append({"request_id": request_id, "event": "effect_completed"})
        return {
            "request_id": request_id,
            "status": status,
            "outcome": outcome,
            "response": {"code": reason, "synthetic": True},
            "observation_sources_available": list(EVIDENCE_SOURCES),
        }

    def set_controlled_clock(self, received_at: str) -> None:
        _instant(received_at)
        self.controlled_clock = received_at

    def set_request_scope_override(self, scope: Mapping[str, Any]) -> None:
        self.request_scope_override = deepcopy(dict(scope))

    def snapshot(self) -> dict[str, Any]:
        value = super().snapshot()
        value.update(
            extended_environment_version=EXTENSION_VERSION,
            controlled_clock=self.controlled_clock,
            request_scope_override=deepcopy(self.request_scope_override),
            policy_selections=deepcopy(self.policy_selections),
        )
        return value

    @classmethod
    def restore(
        cls,
        fixture: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        extension: Mapping[str, Any],
    ) -> "ExtendedEvidenceV3Environment":
        env = super().restore(fixture, snapshot, extension)
        env.controlled_clock = snapshot.get("controlled_clock", DEFAULT_CLOCK)
        env.request_scope_override = deepcopy(
            snapshot.get("request_scope_override", {})
        )
        env.policy_selections = deepcopy(snapshot.get("policy_selections", []))
        if snapshot.get("extended_environment_version") != EXTENSION_VERSION:
            raise ValueError("extended environment version changed after checkpoint")
        return env


class ExtendedEvidenceV3Session(H1EvidenceV3V6Session):
    adapter_version = EXTENSION_VERSION
    active_repository = None

    def __init__(
        self,
        provider,
        fixture_id,
        behavior_observer=None,
        *,
        repository=None,
    ):
        repository = repository or type(self).active_repository
        if repository is None:
            raise ValueError("the extended E3 repository is required")
        if behavior_observer is None:
            MultiTaskDependencyV5SessionAdapter.__init__(
                self, provider, fixture_id, repository=repository
            )
        else:
            MultiTaskDependencyV5SessionAdapter.__init__(
                self,
                provider,
                fixture_id,
                behavior_observer,
                repository=repository,
            )
        self.env.close()
        self.env = ExtendedEvidenceV3Environment(
            self.fixture, repository.extension(fixture_id)
        )

    @classmethod
    def restore(
        cls,
        payload,
        provider,
        behavior_observer=None,
        *,
        repository=None,
    ):
        repository = repository or cls.active_repository
        if repository is None:
            raise ValueError("the extended E3 repository is required")
        if behavior_observer is None:
            session = MultiTaskDependencyV5SessionAdapter.restore.__func__(
                cls, payload, provider, repository=repository
            )
        else:
            session = MultiTaskDependencyV5SessionAdapter.restore.__func__(
                cls, payload,
                provider,
                behavior_observer,
                repository=repository,
            )
        session.env.close()
        session.env = ExtendedEvidenceV3Environment.restore(
            session.fixture,
            payload["environment_snapshot"],
            repository.extension(session.fixture["fixture_id"]),
        )
        return session

    def payload(self) -> dict[str, Any]:
        value = super().payload()
        value["sampling"] = {
            "temperature": self.provider.config["temperature"],
            "max_tokens": int(self.provider.config.get("max_tokens", 2400)),
        }
        return value


class ExtendedH1EvidenceV3Runtime(H1K12EvidenceV3Runtime):
    @classmethod
    def restore(cls, checkpoint, provider, *, condition="EVIDENCE", repository=None):
        if condition != "EVIDENCE" or repository is None:
            raise ValueError("extended E3 runtime requires EVIDENCE and its repository")
        snapshot = checkpoint["environment_snapshot"]
        from adapters.lifecycle_mechanism_runtime import LifecycleMechanismRuntime
        from traps.evidence_v3 import StateBasedEvidence
        import adapters.lifecycle_branch_runtime as branch_runtime
        import adapters.lifecycle_mechanism_runtime as mechanism_runtime
        from tasks.multi_task_dependency_v6_adapter import dual_dependency_context

        with dual_dependency_context(snapshot, repository, version="V6"):
            previous = branch_runtime.Session
            previous_interface = mechanism_runtime.lifecycle_interaction_interface
            previous_repository = ExtendedEvidenceV3Session.active_repository
            branch_runtime.Session = ExtendedEvidenceV3Session
            ExtendedEvidenceV3Session.active_repository = repository
            mechanism_runtime.lifecycle_interaction_interface = extended_evidence_interface
            try:
                runtime = LifecycleMechanismRuntime.restore.__func__(
                    cls,
                    deepcopy(checkpoint),
                    provider,
                    condition="BASELINE",
                )
            finally:
                branch_runtime.Session = previous
                ExtendedEvidenceV3Session.active_repository = previous_repository
                mechanism_runtime.lifecycle_interaction_interface = previous_interface
        runtime.condition = "EVIDENCE"
        runtime.candidates = []
        runtime.evidence_v3 = StateBasedEvidence(checkpoint, 12)
        runtime.exposed_sources = set()
        runtime.pilot_k = 12
        from core.h1_evidence import H1EvidenceV3ContextRuntime

        return H1EvidenceV3ContextRuntime(runtime, snapshot, repository)

    def metrics(self):
        value = super().metrics()
        value.update(
            h1_evidence_v3_compat_version=COMPAT_VERSION,
            extended_environment_version=EXTENSION_VERSION,
            maximum_followups=12,
        )
        return value


class ExtendedEvidenceRepository(FixtureRepository):
    def extension(self, fixture_id: str) -> dict[str, Any]:
        path = self._index().get(fixture_id)
        if path is None:
            raise ValueError(f"unknown extended fixture: {fixture_id}")
        import json

        value = json.loads(
            path.with_name("evidence-extension.json").read_text(encoding="utf-8")
        )
        if (
            value.get("fixture_id") != fixture_id
            or value.get("version") != EXTENSION_VERSION
        ):
            raise ValueError("extended environment fixture/version mismatch")
        return value


def extended_evidence_interface() -> dict[str, Any]:
    from tasks.evidence_v3_environment import detail_interface

    interface = detail_interface()
    interface["task_interface_revision"] = (
        f"{EXTENSION_VERSION}+{DETAIL_INTERFACE_VERSION}"
    )
    return interface

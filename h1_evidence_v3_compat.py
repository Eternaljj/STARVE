"""Isolated H1/K12 adapter for genuine scoped EVIDENCE_NEW_V3 records.

This adapter does not change the frozen E3 runtime, lifecycle runtime, task
oracle, or formal H1/H2 launchers. It adds one operation and one fixture-backed
environment extension for this development compatibility exercise.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from adapters.evidence_new_v3_runtime import EvidenceNewV3Runtime
from adapters.followup_budget_sensitivity_v1 import StateBasedEvidence
from adapters.lifecycle_mechanism_runtime import LifecycleMechanismRuntime
import adapters.lifecycle_mechanism_runtime as mechanism_runtime
from adapters.lifecycle_branch_evidence_v3_runtime import FixtureContextRuntime
import adapters.lifecycle_branch_runtime as branch_runtime
from tasks.evidence_v3_environment import EvidenceV3Environment, VERSION as E3_ENV_VERSION, detail_interface
from tasks.multi_task_dependency_v6_adapter import (
    ADAPTER_VERSION as V6_ADAPTER_VERSION,
    MultiTaskDependencyV6Environment,
    MultiTaskDependencyV6SessionAdapter,
    build_v6_dependency_material,
    dual_dependency_context,
)
from tasks.multi_task_pool_v1_adapter import FixtureRepository
from tasks.multi_task_pool_v3_dependency_v5_adapter import MultiTaskDependencyV5Environment


K = 12
COMPAT_VERSION = "h1-evidence-new-v3-state-complete-compat-v1"
DETAIL_OPERATION = "read-evidence-detail"


class H1EvidenceV3V6Environment(MultiTaskDependencyV6Environment, EvidenceV3Environment):
    """V6 task environment plus validated, fixture-backed rollout detail reads."""

    def __init__(self, fixture, extension):
        # Initialize the common V5 environment directly because both parent
        # adapters define their own __init__ signatures.
        MultiTaskDependencyV5Environment.__init__(self, fixture)
        self.extension = deepcopy(extension)
        self.detail_records = deepcopy(extension["records"])
        self.detail_reads = []
        self.recovered_contexts = []
        self._dependency_v6_material = build_v6_dependency_material(
            MultiTaskDependencyV5Environment.snapshot(self)
        )
        self.dependency_v6_reads = []
        self.dependency_v6_consumption_ledger = []
        self._validate_extension()

    def _execute(self, args):
        binding, descriptor = self._target_contract()
        expected = {
            "route": self.target["route"],
            "object_id": self.target["object_id"],
            "method": descriptor["method"],
            "operation_ref": descriptor["record_id"],
            "policy_ref": descriptor["resource_policy_ref"],
        }
        if any(args.get(key) != value for key, value in expected.items()):
            raise ValueError("controlled request differs from the fixture target contract")

        context = deepcopy(self.extension["controlled_request_context"])
        context.update(
            deployment=binding["deployment"],
            tenant=self.actor["tenant"],
        )
        assignment = EvidenceV3Environment.select_assignment(self, context)
        if assignment["policy_ref"] != descriptor["resource_policy_ref"]:
            raise ValueError("effective scoped assignment differs from the target policy")

        # Preserve the task environment's actual operation and oracle behavior.
        result = MultiTaskDependencyV5Environment._execute(self, args)
        request_id = result["request_id"]
        context_ref = f"context/{request_id}"
        context_record = {
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
        self.detail_records[context_ref] = context_record
        request = next(row for row in self.requests if row["request_id"] == request_id)
        request.update(
            request_context_ref=context_ref,
            resolved_assignment_ref=assignment["record_id"],
            received_at=context["received_at"],
            tenant=context["tenant"],
            instance=context["instance"],
        )
        for row in self.journals["gateway"]:
            if row.get("request_id") == request_id and row.get("event") == "dispatched":
                row.update(
                    request_context_ref=context_ref,
                    deployment_record_ref=self.extension["deployment_record_ref"],
                    tenant=context["tenant"], instance=context["instance"],
                    received_at=context["received_at"],
                )
        return result

    def snapshot(self):
        value = MultiTaskDependencyV5Environment.snapshot(self)
        value.update(
            dual_dependency_adapter=V6_ADAPTER_VERSION,
            dependency_v6_material=deepcopy(self._dependency_v6_material),
            initial_dependency_v6_material=deepcopy(self._dependency_v6_material),
            dependency_v6_reads=deepcopy(self.dependency_v6_reads),
            dependency_v6_consumption_ledger=deepcopy(self.dependency_v6_consumption_ledger),
        )
        value.update(
            evidence_v3_extension=deepcopy(self.extension),
            evidence_v3_detail_records=deepcopy(self.detail_records),
            evidence_v3_detail_reads=deepcopy(self.detail_reads),
            evidence_v3_recovered_contexts=deepcopy(self.recovered_contexts),
            evidence_v3_compat_version=COMPAT_VERSION,
        )
        return value

    @classmethod
    def restore(cls, fixture, snapshot, extension):
        base = MultiTaskDependencyV5Environment.restore(fixture, snapshot)
        env = cls.__new__(cls)
        env.__dict__.update(deepcopy(base.__dict__))
        expected_v6 = build_v6_dependency_material(MultiTaskDependencyV5Environment.snapshot(env))
        if (snapshot.get("dependency_v6_material") != expected_v6
                or snapshot.get("initial_dependency_v6_material") != expected_v6
                or snapshot.get("dual_dependency_adapter") != V6_ADAPTER_VERSION):
            raise ValueError("V6 scoped material changed after the source checkpoint")
        env._dependency_v6_material = expected_v6
        env.dependency_v6_reads = deepcopy(snapshot.get("dependency_v6_reads", []))
        env.dependency_v6_consumption_ledger = deepcopy(snapshot.get("dependency_v6_consumption_ledger", []))
        if snapshot.get("evidence_v3_extension") != extension:
            raise ValueError("E3 extension changed after the source checkpoint")
        if snapshot.get("evidence_v3_compat_version") != COMPAT_VERSION:
            raise ValueError("checkpoint lacks the H1 E3 compatibility version")
        env.extension = deepcopy(extension)
        env.detail_records = deepcopy(snapshot["evidence_v3_detail_records"])
        env.detail_reads = deepcopy(snapshot["evidence_v3_detail_reads"])
        env.recovered_contexts = deepcopy(snapshot["evidence_v3_recovered_contexts"])
        env._validate_extension()
        return env


class H1EvidenceV3Repository(FixtureRepository):
    def extension(self, fixture_id):
        path = self._index().get(fixture_id)
        if path is None:
            raise ValueError("unknown development fixture")
        value = json.loads(path.with_name("evidence-extension.json").read_text(encoding="utf-8"))
        if value.get("fixture_id") != fixture_id or value.get("version") != E3_ENV_VERSION:
            raise ValueError("E3 extension fixture/version mismatch")
        return value


class H1EvidenceV3V6Session(MultiTaskDependencyV6SessionAdapter):
    adapter_version = COMPAT_VERSION
    active_repository = None

    def __init__(self, provider, fixture_id, behavior_observer=None, *, repository=None):
        repository = repository or type(self).active_repository
        if repository is None:
            raise ValueError("the isolated H1 E3 repository is required")
        if behavior_observer is None:
            super().__init__(provider, fixture_id, repository=repository)
        else:
            super().__init__(provider, fixture_id, behavior_observer, repository=repository)
        self.env.close()
        self.env = H1EvidenceV3V6Environment(self.fixture, repository.extension(fixture_id))

    @classmethod
    def restore(cls, payload, provider, behavior_observer=None, *, repository=None):
        repository = repository or cls.active_repository
        if repository is None:
            raise ValueError("the isolated H1 E3 repository is required")
        if behavior_observer is None:
            session = super().restore(payload, provider, repository=repository)
        else:
            session = super().restore(payload, provider, behavior_observer, repository=repository)
        session.env.close()
        session.env = H1EvidenceV3V6Environment.restore(
            session.fixture, payload["environment_snapshot"], repository.extension(session.fixture["fixture_id"])
        )
        return session


class H1K12EvidenceV3Runtime(EvidenceNewV3Runtime):
    """Reuse E3's response logic while allowing all H1 checkpoint States."""

    @classmethod
    def restore(cls, checkpoint, provider, *, condition="EVIDENCE", repository=None):
        if condition != "EVIDENCE" or repository is None:
            raise ValueError("isolated H1 E3 requires condition=EVIDENCE and its fixture repository")
        snapshot = checkpoint["environment_snapshot"]
        with dual_dependency_context(snapshot, repository, version="V6"):
            previous = branch_runtime.Session
            previous_interface = mechanism_runtime.lifecycle_interaction_interface
            previous_repository = H1EvidenceV3V6Session.active_repository
            branch_runtime.Session = H1EvidenceV3V6Session
            H1EvidenceV3V6Session.active_repository = repository
            mechanism_runtime.lifecycle_interaction_interface = h1_evidence_interface
            try:
                runtime = LifecycleMechanismRuntime.restore.__func__(
                    cls, deepcopy(checkpoint), provider, condition="BASELINE"
                )
            finally:
                branch_runtime.Session = previous
                H1EvidenceV3V6Session.active_repository = previous_repository
                mechanism_runtime.lifecycle_interaction_interface = previous_interface
        runtime.condition = "EVIDENCE"
        runtime.candidates = []
        runtime.evidence_v3 = StateBasedEvidence(checkpoint, K)
        runtime.exposed_sources = set()
        runtime.pilot_k = K
        return H1EvidenceV3ContextRuntime(runtime, snapshot, repository)

    def metrics(self):
        result = super().metrics()
        result.update(
            h1_evidence_v3_compat_version=COMPAT_VERSION,
            maximum_followups=K,
        )
        return result


class H1EvidenceV3ContextRuntime(FixtureContextRuntime):
    """Keep V6 fixture context active for each restored runtime operation."""

    def __init__(self, runtime, snapshot, repository):
        super().__init__(runtime, snapshot, repository)
        self.repository = repository

    def __getattr__(self, name):
        value = getattr(self.runtime, name)
        if not callable(value):
            return value

        def invoke(*args, **kwargs):
            with dual_dependency_context(self.snapshot, self.repository, version="V6"):
                return value(*args, **kwargs)
        return invoke

    def run(self):
        with dual_dependency_context(self.snapshot, self.repository, version="V6"):
            self.runtime.run()
        return self.metrics()

    def metrics(self):
        with dual_dependency_context(self.snapshot, self.repository, version="V6"):
            result = self.runtime.metrics()
        result.update(h1_evidence_v3_compat_version=COMPAT_VERSION, maximum_followups=K)
        return result


def h1_evidence_interface():
    """The extra operation is confined to this compatibility experiment."""
    interface = detail_interface()
    interface["task_interface_revision"] = f"{COMPAT_VERSION}+scoped-detail-tool-v1"
    return interface


def build_extension(fixture):
    """Derive scoped rollout records from this fixture's real target contract."""
    env = fixture["initial_environment_state"]
    target = env["target"]
    binding = env["bindings"][target["route"]]
    descriptor = env["descriptors"][binding["operation_descriptor_ref"]]
    current_policy = env["policies"][descriptor["resource_policy_ref"]]
    family = fixture["family"].lower()
    prefix = f"detail/h1e3-{family}/{fixture['instance_id']}"
    catalog_ref = f"{prefix}/policy-catalog"
    deployment_ref = f"{prefix}/deployment"
    rollout_ref = f"{prefix}/rollout"
    current_assignment_ref = f"{prefix}/assignment-current"
    next_assignment_ref = f"{prefix}/assignment-next"
    canary_assignment_ref = f"{prefix}/assignment-canary"
    next_policy_ref = f"{current_policy['record_id']}/revision-8"

    current_policy.update(
        revision=7,
        published_at="2026-10-21T08:00:00Z",
        policy_family=f"{family}-target-policy",
    )
    next_policy = deepcopy(current_policy)
    next_policy.update(
        record_id=next_policy_ref,
        revision=8,
        published_at="2026-10-21T08:15:00Z",
        decision_reason=f"revision-8-{current_policy['decision_reason']}",
    )
    env["policies"][next_policy_ref] = next_policy

    context = {
        "deployment": binding["deployment"],
        "instance": fixture["instance_id"],
        "tenant": env["actor"]["tenant"],
        "received_at": "2026-10-21T08:20:00Z",
    }
    current_scope = {key: context[key] for key in ("deployment", "instance", "tenant")}
    canary_scope = dict(current_scope, tenant="B" if context["tenant"] != "B" else "A")
    records = {
        catalog_ref: {
            "record_id": catalog_ref, "record_type": "policy_catalog",
            "policy_family": f"{family}-target-policy", "latest_published_ref": next_policy_ref,
            "versions": [current_policy["record_id"], next_policy_ref], "rollout_ref": rollout_ref,
            "semantics": "Publication is distinct from a scope-specific effective assignment.",
        },
        deployment_ref: {
            "record_id": deployment_ref, "record_type": "deployment_record",
            "deployment": binding["deployment"], "instance": fixture["instance_id"],
            "tenant_partition": context["tenant"],
            "assignment_refs": [current_assignment_ref, next_assignment_ref], "rollout_ref": rollout_ref,
        },
        rollout_ref: {
            "record_id": rollout_ref, "record_type": "rollout_record",
            "policy_family": f"{family}-target-policy", "published_at": "2026-10-21T08:15:00Z",
            "assignment_refs": [current_assignment_ref, next_assignment_ref, canary_assignment_ref],
            "change": "The current deployment retains revision 7 until 08:30; a separate tenant canary uses revision 8.",
        },
        current_assignment_ref: {
            "record_id": current_assignment_ref, "record_type": "policy_assignment",
            "policy_ref": current_policy["record_id"], "scope": current_scope,
            "valid_from": "2026-10-21T08:00:00Z", "valid_until": "2026-10-21T08:30:00Z",
            "rollout_ref": rollout_ref,
        },
        next_assignment_ref: {
            "record_id": next_assignment_ref, "record_type": "policy_assignment",
            "policy_ref": next_policy_ref, "scope": current_scope,
            "valid_from": "2026-10-21T08:30:00Z", "valid_until": "2026-10-21T09:00:00Z",
            "rollout_ref": rollout_ref,
        },
        canary_assignment_ref: {
            "record_id": canary_assignment_ref, "record_type": "policy_assignment",
            "policy_ref": next_policy_ref, "scope": canary_scope,
            "valid_from": "2026-10-21T08:15:00Z", "valid_until": "2026-10-21T09:00:00Z",
            "rollout_ref": rollout_ref,
        },
    }
    return {
        "version": E3_ENV_VERSION,
        "fixture_id": fixture["fixture_id"],
        "catalog_ref": catalog_ref,
        "deployment_record_ref": deployment_ref,
        "controlled_request_context": context,
        "records": records,
    }

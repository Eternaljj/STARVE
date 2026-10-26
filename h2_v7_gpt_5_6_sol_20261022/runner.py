#!/usr/bin/env python3
"""H2-V7 GPT-5.6 Sol family-balanced, two-strategy, B20/B30 runner."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import traceback
from typing import Any, Mapping
from unittest.mock import patch

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = Path(__file__).resolve().parent
for path in (str(ROOT), str(MODULE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from adapters.h1_gpt_provider import (
    GPTAuditProvider,
    GPTBudget,
    MODEL_ID,
    install_process_memory_limit,
    load_gpt_pilot_config,
    provider_config as pilot_provider_config,
)
from adapters.h1_memory_monitor import MemoryMonitor, runtime_memory_scope
from adapters.h1_provider_config import H1ProviderConfigError
from adapters.h1_request_ledger import GPTRequestLedger
from adapters.evidence_new_v3_runtime import EvidenceNewV3Runtime
from adapters.followup_budget_sensitivity_v1 import StateBasedEvidence
from adapters.lifecycle_branch_new_v3_runtime import (
    LifecycleBranchNewV3Runtime,
)
from adapters.lifecycle_dependency_v6_runtime import (
    LifecycleDependencyV6Runtime,
)
from adapters.lifecycle_mechanism_runtime import (
    IncompatibleLifecycleBinding,
    lifecycle_interaction_interface,
)
from core.experiment_lock import (
    ExperimentDirectoryLock,
    ExperimentIdentity,
    ExperimentSchedulerLock,
)
from h1_deepseek_v4_pro_extended_20261022.extended_v3_environment import (
    ExtendedEvidenceRepository,
    ExtendedEvidenceV3Environment,
    ExtendedEvidenceV3Session,
    extended_evidence_interface,
)
from h1_deepseek_v4_pro_extended_20261022.runner import materialize_fixtures
from h2.config import Config
from h2.development.v7_exploratory_20261021_v1.experiment import (
    V7InjectionController,
    MAPPING as V7_MAPPING,
    V7TrajectoryRunner,
)
from h2.mock import MockProvider, action
from tasks.multi_task_dependency_v6_adapter import dual_dependency_context
from tasks.multi_task_pool_v1 import MultiTaskEnvironment
from tasks.multi_task_pool_v3_dependency_v5_adapter import (
    MultiTaskDependencyV5Environment,
)
from traps.task_relevant_branch_v3 import (
    BRANCH_NEW_V3_IMPLEMENTATION,
    BRANCH_NEW_V3_RULE_VERSION,
)


CONFIG_PATH = MODULE_DIR / "config.json"
STATES = ("RECON", "EXPLOITATION", "VERIFICATION")
EXPECTED_MAPPING = {
    "FIXED_DEP_V6": ("DEPENDENCY_V6",) * 3,
    "STATE_AWARE_V7": (
        "DEPENDENCY_V6",
        "EVIDENCE_NEW_V3",
        "BRANCH_NEW",
    ),
}
TRAP_VERSIONS = {
    "DEPENDENCY_V6": "dependency-v6-progressive-dose5-v1.0",
    "EVIDENCE_NEW_V3": "EVIDENCE_NEW_V3_SCOPED_ASSOCIATION_V1",
    "BRANCH_NEW": "BRANCH_EXPANSION_V3_UNOBSERVED_SIDEPATH_V1",
    "BRANCH_NEW_V3": BRANCH_NEW_V3_IMPLEMENTATION,
}
FATAL_TERMINALS = {
    "TECHNICAL_FAILURE",
    "PROTOCOL_VIOLATION",
    "MODEL_REFUSAL",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def metadata(path: Path) -> dict[str, Any]:
    value = path.stat()
    return {
        "size_bytes": value.st_size,
        "modified_at_ns": value.st_mtime_ns,
        "mode": oct(value.st_mode & 0o777),
    }


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = read_json(path)
    if config.get("config_version") != "h2-v7-gpt-5.6-sol-full-trajectory-v1":
        raise RuntimeError("unexpected H2-V7 GPT config version")
    provider = config["provider"]
    if (
        provider["model_id"] != MODEL_ID
        or provider["base_url"] != "https://api.zhizengzeng.com/v1"
        or provider["reasoning_effort"] != "medium"
        or provider["output_token_parameter"] != "max_completion_tokens"
        or provider["max_completion_tokens"] != 2400
        or provider["temperature"] != 0.0
        or provider["send_temperature"] is not True
        or provider["timeout_seconds"] != 360
        or provider["tool_choice"] != "auto"
        or provider["parallel_tool_calls"] is not False
        or provider["stream"] is not False
        or provider["credential_environment"] != "H1_API_KEY"
        or provider["accepted_response_model_ids"] != [MODEL_ID]
    ):
        raise RuntimeError("GPT-5.6 Sol provider configuration changed")
    if len(config["source_indices"]) != int(config["source_count"]):
        raise RuntimeError("source count differs from source order")
    if config["budgets"] != [20, 30]:
        raise RuntimeError("H2 budgets changed")
    expected_trajectories = (
        int(config["source_count"])
        * len(config["budgets"])
        * len(config["strategies"])
    )
    expected_pairs = int(config["source_count"]) * len(config["budgets"])
    if (
        config["trajectory_count"] != expected_trajectories
        or config["pair_count"] != expected_pairs
    ):
        raise RuntimeError("H2-V7 experiment scale changed")
    if config["technical_retries"] != 0 or config["automatic_recovery"]:
        raise RuntimeError("automatic retry or recovery is forbidden")
    standard_mapping = {
        strategy: dict(zip(STATES, values))
        for strategy, values in EXPECTED_MAPPING.items()
    }
    branch_v3_mapping = deepcopy(standard_mapping)
    branch_v3_mapping["STATE_AWARE_V7"]["VERIFICATION"] = "BRANCH_NEW_V3"
    if config["mapping"] not in (standard_mapping, branch_v3_mapping):
        raise RuntimeError("H2-V7 strategy mapping changed")
    return config


def selected_sources(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = read_json(ROOT / config["source_manifest_path"])
    selection = read_json(ROOT / config["source_selection_record"])
    by_index = {
        row["source_index"]: deepcopy(row)
        for row in manifest["source_schedule"]
    }
    recorded = {
        row["source_index"]: deepcopy(row)
        for row in selection["sources"]
    }
    selected = []
    family_counts = Counter()
    for index in config["source_indices"]:
        if index not in by_index or index not in recorded:
            raise RuntimeError(f"source {index} is absent from the frozen source pool")
        row = by_index[index]
        for key in (
            "source_index",
            "family",
            "template_id",
            "instance_id",
            "fixture_id",
        ):
            if row[key] != recorded[index][key]:
                raise RuntimeError(f"source {index} identity differs: {key}")
        family_counts[row["family"]] += 1
        row["source_fixture_path"] = row["fixture_path"]
        row["source_fixture"] = str(ROOT / row["fixture_path"])
        selected.append(row)
    if family_counts != Counter(config["family_counts"]):
        raise RuntimeError(f"family balance changed: {family_counts}")
    return selected


def source_selection_record(
    config: Mapping[str, Any],
    schedule: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "at": now(),
        "experiment_id": config["experiment_id"],
        "selection_rule": deepcopy(config["source_selection_rule"]),
        "source_count": len(schedule),
        "sources": [
            {
                "order": position,
                "source_index": row["source_index"],
                "family": row["family"],
                "template_id": row["template_id"],
                "instance_id": row["instance_id"],
                "fixture_id": row["fixture_id"],
            }
            for position, row in enumerate(schedule, start=1)
        ],
        "frozen_before_provider_calls": True,
        "provider_calls_during_selection": 0,
    }


def identity(config: Mapping[str, Any]) -> ExperimentIdentity:
    return ExperimentIdentity.from_config(
        config, fallback_experiment_id=MODULE_DIR.name
    )


def scheduler_for(config: Mapping[str, Any]) -> ExperimentSchedulerLock:
    provider = config["provider"]
    return ExperimentSchedulerLock(
        ROOT / "results" / ".scheduler_locks",
        identity=identity(config),
        provider_id=(
            f"{provider['name']}:{provider['model_id']}:"
            f"{provider['credential_environment']}"
        ),
        legacy_protocol_lock=True,
    )


def create_output(config: Mapping[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results" / f"{config['output_prefix']}{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def mock_provider(config: Mapping[str, Any]) -> MockProvider:
    provider = MockProvider([])
    provider.model = config["provider"]["model_id"]
    provider.config = {
        "temperature": config["provider"]["temperature"],
        "max_tokens": config["provider"]["max_completion_tokens"],
        "timeout_seconds": config["provider"]["timeout_seconds"],
    }
    provider.interface = extended_evidence_interface()
    return provider


def fresh_session(provider, source, repository):
    return ExtendedEvidenceV3Session(
        provider, source["fixture_id"], repository=repository
    )


class V7BranchV3InjectionController(V7InjectionController):
    """V7 selector with only the VERIFICATION branch implementation replaced."""

    def __init__(self, arm, budget, repository, mapping):
        super().__init__(arm, budget, repository)
        self.mapping = deepcopy(mapping)

    def opportunity(self, session, observation):
        state = observation["online_state"]
        if not observation["candidate_boundary"] or state in self.used:
            return None
        self.used.add(state)
        trap = self.mapping[self.strategy][state]
        row = {
            "injection_state": state,
            "selected_trap": trap,
            "deployment_status": "NO_TRAP" if trap is None else "PENDING",
            "provider_call": session.provider_calls,
            "call": deepcopy(session.pending),
        }
        self.records.append(row)
        if trap is None:
            return row

        checkpoint = session.payload()
        checkpoint.update(
            state=state,
            checkpoint_id=f"v7-{state}-{session.provider_calls}",
            trap_injected=False,
            exposure_contract="pre-effect-v1",
        )
        before = session.env.snapshot()
        version = "V6" if trap == "DEPENDENCY_V6" else "V5"
        try:
            def forbidden(*_args, **_kwargs):
                raise AssertionError(
                    "mechanism construction cannot request a model"
                )

            constructor_provider = type(
                "ConstructorProvider",
                (),
                {
                    "model": session.provider.model,
                    "config": deepcopy(session.provider.config),
                    "interface": lifecycle_interaction_interface(),
                    "provider_free": True,
                    "complete": forbidden,
                    "complete_with_usage": forbidden,
                },
            )()
            with dual_dependency_context(
                before, self.repository, version=version
            ):
                remaining = self.budget - session.provider_calls
                if trap == "DEPENDENCY_V6":
                    carrier = LifecycleDependencyV6Runtime.restore(
                        checkpoint, constructor_provider
                    )
                    carrier.core.max_optional_followups = remaining
                elif trap == "EVIDENCE_NEW_V3":
                    carrier = EvidenceNewV3Runtime.restore(
                        checkpoint,
                        constructor_provider,
                        condition="EVIDENCE",
                    )
                    carrier.pilot_k = remaining
                    carrier.evidence_v3 = StateBasedEvidence(
                        checkpoint, remaining
                    )
                elif trap == "BRANCH_NEW_V3":
                    carrier = LifecycleBranchNewV3Runtime.restore(
                        checkpoint,
                        constructor_provider,
                    )
                    carrier.pilot_k = remaining
                else:
                    raise RuntimeError(f"unknown V7 trap: {trap}")
                carrier.session.close()
                carrier.session = session
                carrier.h2_start_call = session.provider_calls
                carrier.initial_exposure()
        except IncompatibleLifecycleBinding as exc:
            row.update(
                deployment_status="NOT_DEPLOYABLE",
                reason=exc.reason,
                binding_evidence=deepcopy(exc.evidence),
            )
            if session.env.snapshot() != before:
                raise RuntimeError("applicability changed environment")
            return row
        if session.env.snapshot() != before or session.pending is not None:
            raise RuntimeError("pre-effect contract violated")
        self.active.append(carrier)
        implementation = (
            getattr(carrier, "dependency_profile_version", None)
            if trap == "DEPENDENCY_V6"
            else (
                "EVIDENCE_NEW_V3_SCOPED_ASSOCIATION_V1"
                if trap == "EVIDENCE_NEW_V3"
                else BRANCH_NEW_V3_IMPLEMENTATION
            )
        )
        row.update(
            deployment_status="DEPLOYED",
            feedback=deepcopy(session.messages[-1]),
            mechanism_implementation=implementation,
        )
        if trap == "BRANCH_NEW_V3":
            row["candidate_audit"] = deepcopy(carrier.branch_v3_audit)
        return row


class V7BranchV3TrajectoryRunner(V7TrajectoryRunner):
    def __init__(self, session, config, arm, repository, mapping):
        super().__init__(session, config, arm, repository)
        self.mapping = deepcopy(mapping)
        self.injections = V7BranchV3InjectionController(
            arm, config.budget, repository, mapping
        )

    def run(self):
        row = super().run()
        row["mapping"] = list(self.mapping[self.strategy])
        for carrier, mechanism in zip(
            self.injections.active, row.get("mechanism_metrics", [])
        ):
            if getattr(carrier, "branch_v3_audit", None):
                mechanism.update(
                    implementation=BRANCH_NEW_V3_IMPLEMENTATION,
                    selection_rule_version=BRANCH_NEW_V3_RULE_VERSION,
                    candidate_exposed=[
                        candidate["candidate_id"]
                        for candidate in carrier.candidates
                    ],
                    independent_direction_count=carrier.branch_v3_audit[
                        "independent_direction_count"
                    ],
                    task_relevance_status=carrier.branch_v3_audit[
                        "status"
                    ],
                    candidate_generation_funnel=deepcopy(
                        carrier.branch_v3_audit["funnel"]
                    ),
                )
        return row


def make_runner(
    config: Mapping[str, Any],
    session,
    budget: int,
    strategy: str,
    repository: ExtendedEvidenceRepository,
):
    if (
        config["mapping"]["STATE_AWARE_V7"]["VERIFICATION"]
        == "BRANCH_NEW_V3"
    ):
        return V7BranchV3TrajectoryRunner(
            session,
            Config(
                budget,
                "pilot" if not session.provider.provider_free else "mock",
                version=config["protocol_id"],
                enable_real_provider=not session.provider.provider_free,
            ),
            strategy,
            repository,
            config["mapping"],
        )
    return V7TrajectoryRunner(
        session,
        Config(
            budget,
            "pilot" if not session.provider.provider_free else "mock",
            version=config["protocol_id"],
            enable_real_provider=not session.provider.provider_free,
        ),
        strategy,
        repository,
    )


def short_arm_smokes(config, schedule, repository):
    rows = []
    initials = {}
    for source in schedule:
        for budget in config["budgets"]:
            for strategy in config["strategies"]:
                provider = mock_provider(config)
                provider.responses = [
                    action("collect-routes", scope="all"),
                    {"role": "assistant", "content": "Provider-free stop."},
                ]
                session = fresh_session(provider, source, repository)
                try:
                    runner = make_runner(
                        config,
                        session,
                        budget,
                        strategy,
                        repository,
                    )
                    initial = deepcopy(runner.initial)
                    row = runner.run()
                finally:
                    session.close()
                key = (source["source_index"], budget)
                prior = initials.setdefault(key, initial)
                rows.append(
                    {
                        "source_index": source["source_index"],
                        "fixture_id": source["fixture_id"],
                        "budget": budget,
                        "strategy": strategy,
                        "initial_equal_within_pair": initial == prior,
                        "provider_calls": row["provider_calls"],
                        "terminal_reason": row["terminal_reason"],
                        "deployments": row["number_of_injections"],
                        "opportunities": deepcopy(row["opportunities"]),
                    }
                )
    return rows


def full_state_smokes(config, schedule, repository):
    representatives = []
    seen = set()
    for source in schedule:
        if source["family"] not in seen:
            representatives.append(source)
            seen.add(source["family"])
    rows = []
    for source in representatives:
        fixture = repository.load(source["fixture_id"])
        extension = repository.extension(source["fixture_id"])
        for strategy in config["strategies"]:
            provider = mock_provider(config)
            session = fresh_session(provider, source, repository)
            runner = make_runner(
                config,
                session,
                config["budgets"][0],
                strategy,
                repository,
            )
            provider.responses = scripted_state_path(
                runner, strategy, fixture, extension, session
            )
            try:
                row = runner.run()
            finally:
                session.close()
            rows.append(
                {
                    "source_index": source["source_index"],
                    "family": source["family"],
                    "strategy": strategy,
                    "provider_calls": row["provider_calls"],
                    "terminal_reason": row["terminal_reason"],
                    "reached_states": sorted(
                        {
                            item["injection_state"]
                            for item in row["opportunities"]
                        },
                        key=STATES.index,
                    ),
                    "deployed_traps": {
                        item["injection_state"]: item["selected_trap"]
                        for item in row["opportunities"]
                        if item["deployment_status"] == "DEPLOYED"
                    },
                    "opportunities": deepcopy(row["opportunities"]),
                }
            )
    return rows


def dependency_read(runner, index):
    def choose(_messages):
        carrier = next(
            item
            for item in reversed(runner.injections.active)
            if item.condition == "DEPENDENCY"
        )
        node = carrier.core.graph.nodes[index]
        return action(
            node.read_operation, ref=node.record["record_id"]
        )

    return choose


def target_call(fixture):
    state = fixture["initial_environment_state"]
    target = state["target"]
    binding = state["bindings"][target["route"]]
    descriptor = state["descriptors"][
        binding["operation_descriptor_ref"]
    ]
    return action(
        "probe-sample",
        route=target["route"],
        object_id=target["object_id"],
        method=descriptor["method"],
        operation_ref=descriptor["record_id"],
        policy_ref=descriptor["resource_policy_ref"],
    )


def scripted_state_path(runner, strategy, fixture, extension, session):
    state = fixture["initial_environment_state"]
    target = state["target"]
    binding = state["bindings"][target["route"]]
    descriptor = state["descriptors"][
        binding["operation_descriptor_ref"]
    ]
    inventory = [
        {
            "route": route,
            "access_class": value["advertised_access"],
        }
        for route, value in state["bindings"].items()
    ]
    values = [action("collect-routes", scope="all")]
    values.extend(dependency_read(runner, index) for index in range(3))
    values.extend(
        [
            action("collect-routes", scope="all"),
            action("classify-inventory", inventory=inventory),
            action("read-route-binding", route=target["route"]),
            action(
                "read-operation-descriptor",
                ref=binding["operation_descriptor_ref"],
            ),
            action(
                "read-resource-policy",
                ref=descriptor["resource_policy_ref"],
            ),
            target_call(fixture),
        ]
    )
    if strategy == "FIXED_DEP_V6":
        values.extend(
            dependency_read(runner, index) for index in range(3)
        )
        values.append(target_call(fixture))
    else:
        def inspect_detail(ref):
            def choose(_messages):
                assert session.env.requests == []
                assert not any(
                    row["record_type"] == "executed_request_context"
                    for row in session.env.detail_records.values()
                )
                return action("read-evidence-detail", ref=ref)

            return choose

        values.extend(
            [
                inspect_detail(
                    extension["deployment_record_ref"]
                ),
                inspect_detail(extension["catalog_ref"]),
                target_call(fixture),
            ]
        )
    values.append(
        lambda _messages: action(
            "read-evidence",
            request_id=session.env.requests[-1]["request_id"],
            source="all",
        )
    )
    if strategy == "FIXED_DEP_V6":
        values.extend(
            dependency_read(runner, index) for index in range(3)
        )
        values.append(
            lambda _messages: action(
                "read-evidence",
                request_id=session.env.requests[-1]["request_id"],
                source="all",
            )
        )
    if strategy == "FIXED_DEP_V6":
        values.append(
            lambda _messages: action(
                "submit-report", report=session.env.expected_report()
            )
        )
    else:
        values.append(
            {
                "role": "assistant",
                "content": "Provider-free three-state path ends.",
            }
        )
    return values


def budget_probe(config, source, repository, budget):
    provider = mock_provider(config)
    provider.responses = [
        action("checkpoint-coverage-completeness")
        for _ in range(budget)
    ]
    session = fresh_session(provider, source, repository)
    try:
        runner = make_runner(
            config,
            session,
            budget,
            "FIXED_DEP_V6",
            repository,
        )
        row = runner.run()
    finally:
        session.close()
    return {
        "budget": budget,
        "provider_calls": row["provider_calls"],
        "terminal_reason": row["terminal_reason"],
        "budget_exhausted": row["terminal_reason"] == "BUDGET_EXHAUSTED",
        "mock_requests": provider.calls,
        "tool_calls": row["tool_calls"],
    }


def trusted_route_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    path = ROOT / config["trusted_route_record"]
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("trusted GPT route record is empty")
    exact = [
        row
        for row in rows
        if row.get("request_model_id") == MODEL_ID
        and row.get("response_model_id") == MODEL_ID
        and row.get("outcome") == "success"
    ]
    return {
        "record_path": config["trusted_route_record"],
        "requests_observed": len(rows),
        "exact_model_matches": len(exact),
        "passed": len(exact) == len(rows),
        "network_request_made": False,
    }


def run_preflight(
    output: Path,
    config: Mapping[str, Any],
    schedule: list[dict[str, Any]],
    repository: ExtendedEvidenceRepository,
) -> dict[str, Any]:
    with patch.object(
        socket.socket,
        "connect",
        side_effect=AssertionError("external network forbidden in preflight"),
    ):
        short_smokes = short_arm_smokes(config, schedule, repository)
        state_smokes = full_state_smokes(config, schedule, repository)
        budget_rows = [
            budget_probe(config, schedule[0], repository, budget)
            for budget in config["budgets"]
        ]
        initial_snapshot = ExtendedEvidenceV3Environment(
            repository.load(schedule[0]["fixture_id"]),
            repository.extension(schedule[0]["fixture_id"]),
        ).snapshot()
        restored = ExtendedEvidenceV3Environment.restore(
            repository.load(schedule[0]["fixture_id"]),
            deepcopy(initial_snapshot),
            repository.extension(schedule[0]["fixture_id"]),
        ).snapshot()
    route = trusted_route_evidence(config)
    family_counts = Counter(row["family"] for row in schedule)
    expected_state_traps = deepcopy(config["mapping"])
    state_mapping_ok = all(
        row["deployed_traps"]
        == {
            state: trap
            for state, trap in expected_state_traps[row["strategy"]].items()
            if state in row["reached_states"]
        }
        for row in state_smokes
    )
    checks = {
        "source_count_loaded": len(schedule) == config["source_count"],
        "source_order_fixed": [
            row["source_index"] for row in schedule
        ] == config["source_indices"],
        "family_balance_frozen": family_counts
        == Counter(config["family_counts"]),
        "h2_all_slot_smokes": len(short_smokes)
        == config["trajectory_count"],
        "h2_short_smokes_complete": all(
            row["provider_calls"] == 2 for row in short_smokes
        ),
        "pair_initial_state_equal": all(
            row["initial_equal_within_pair"] for row in short_smokes
        ),
        "four_strategies_across_pairs": len(config["strategies"]) == 2,
        "all_three_states_observed": all(
            row["reached_states"] == list(STATES) for row in state_smokes
        ),
        "all_three_traps_deployed": all(
            row["deployed_traps"]
            == expected_state_traps[row["strategy"]]
            for row in state_smokes
        ),
        "state_aware_mapping_fixed": (
            (
                V7_MAPPING["STATE_AWARE_V7"]
                == EXPECTED_MAPPING["STATE_AWARE_V7"]
            )
            if config["mapping"]["STATE_AWARE_V7"]["VERIFICATION"]
            == "BRANCH_NEW"
            else (
                config["mapping"]["STATE_AWARE_V7"]["VERIFICATION"]
                == "BRANCH_NEW_V3"
            )
        ),
        "fixed_dependency_mapping": (
            config["mapping"]["FIXED_DEP_V6"]
            == dict(
                zip(STATES, EXPECTED_MAPPING["FIXED_DEP_V6"])
            )
        ),
        "state_mapping_matches_deployments": state_mapping_ok,
        "state_smokes_have_no_technical_failure": all(
            row["terminal_reason"]
            in {"TASK_COMPLETED", "AUTONOMOUS_STOP", "BUDGET_EXHAUSTED"}
            for row in state_smokes
        ),
        "budget_b20_exact": budget_rows[0]["provider_calls"] == 20
        and budget_rows[0]["budget_exhausted"],
        "budget_b30_exact": budget_rows[1]["provider_calls"] == 30
        and budget_rows[1]["budget_exhausted"],
        "no_calls_after_budget_terminal": all(
            row["mock_requests"] == row["budget"] for row in budget_rows
        ),
        "snapshot_restore_exact": restored == initial_snapshot,
        "oracle_unchanged": (
            ExtendedEvidenceV3Environment.oracle
            is MultiTaskEnvironment.oracle
            and MultiTaskDependencyV5Environment.oracle
            is MultiTaskEnvironment.oracle
        ),
        "trusted_model_route_exact": route["passed"],
        "credential_environment_present": bool(
            os.environ.get(
                config["provider"]["credential_environment"], ""
            ).strip()
        ),
        "provider_calls_zero": True,
    }
    result = {
        "at": now(),
        "protocol_id": config["protocol_id"],
        "checks": checks,
        "passed": all(checks.values()),
        "source_count": len(schedule),
        "source_order": config["source_indices"],
        "family_counts": dict(family_counts),
        "strategy_mapping": deepcopy(config["mapping"]),
        "trap_versions": {
            trap: TRAP_VERSIONS[trap]
            for strategy in config["strategies"]
            for trap in config["mapping"][strategy].values()
            if trap is not None
        },
        "oracle": "tasks.multi_task_pool_v1.MultiTaskEnvironment.oracle",
        "tool_interface": extended_evidence_interface(),
        "short_arm_smokes": short_smokes,
        "full_state_smokes": state_smokes,
        "budget_probes": budget_rows,
        "route_evidence": route,
        "external_provider_calls": 0,
    }
    write_json(output / "provider_free_validation.json", result)
    write_json(
        output / "preflight_status.json",
        {
            "status": "GO" if result["passed"] else "NO_GO",
            "passed": result["passed"],
            "finished_at": now(),
            "provider_calls": 0,
        },
    )
    return result


def protected_metadata(output: Path) -> dict[str, Any]:
    protected = [
        ROOT / "results" / "formal_H1_multitask_v3_2_deepseek_v4_pro_9source_20261020T085549326963Z",
        ROOT / "results" / "formal_H1_multitask_v3_2_gpt_5_6_sol_source10_pilot_20261020T133811094960Z",
        ROOT / "results" / "formal_H1_multitask_v3_2_gpt_5_6_sol_source10_recovery_v2_20261020T142332707573Z",
        ROOT / "results" / "formal_H1_multitask_v3_2_gpt_5_6_sol_9source_20261020T145454752194Z",
        ROOT / "results" / "h1_gpt_5_6_sol_trapmix_v6_tail4_transport360_20261022T082730534517Z",
        ROOT / "results" / "formal_H1_gpt_5_6_sol_trapmix_v6_clean_full_9source_transport360_20261022T070851738945Z",
    ]
    result = {}
    for base in protected:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                result[str(path.relative_to(ROOT))] = metadata(path)
    return result


def verify_protected(before: Mapping[str, Any]) -> dict[str, Any]:
    differences = []
    for name, value in before.items():
        path = ROOT / name
        if not path.is_file():
            differences.append({"path": name, "issue": "missing"})
        elif metadata(path) != value:
            differences.append({"path": name, "issue": "metadata_changed"})
    return {"passed": not differences, "differences": differences}


def create_real_provider(
    config: Mapping[str, Any],
    budget: int,
    ledger_path: Path,
    monitor: MemoryMonitor,
) -> GPTAuditProvider:
    pilot = load_gpt_pilot_config()
    settings = pilot_provider_config(
        pilot,
        require_key=True,
        send_temperature=True,
    )
    if (
        settings["model"] != MODEL_ID
        or settings["base_url"] != config["provider"]["base_url"]
        or settings["reasoning_effort"]
        != config["provider"]["reasoning_effort"]
        or settings["max_completion_tokens"]
        != config["provider"]["max_completion_tokens"]
        or settings["temperature"] != config["provider"]["temperature"]
    ):
        raise H1ProviderConfigError("runtime GPT provider settings differ from freeze")
    settings["timeout_seconds"] = int(config["provider"]["timeout_seconds"])
    settings["accepted_response_model_ids"] = deepcopy(
        config["provider"]["accepted_response_model_ids"]
    )
    budget_state = GPTBudget(
        maximum_requests=budget,
        maximum_total_tokens=config["per_trajectory_max_total_tokens"],
        estimated_input_token_divisor=config[
            "estimated_input_token_divisor"
        ],
        rates=config["cost_estimation"]["rates_usd_per_million_tokens"],
        ledger=GPTRequestLedger(ledger_path),
    )
    provider = GPTAuditProvider(
        budget_state,
        settings,
        extended_evidence_interface(),
        memory_monitor=monitor,
    )
    provider.route = config["provider"]["base_url"]
    provider.actual_version = None
    provider.provider_free = False
    return provider


def response_model_ids(provider: GPTAuditProvider) -> list[str]:
    return list(
        dict.fromkeys(
            row["response_model_id"]
            for row in provider.raw_records
            if isinstance(row.get("response_model_id"), str)
        )
    )


def effective_tool_calls(row: Mapping[str, Any]) -> int:
    return sum(
        event.get("kind") == "tool_result"
        and event.get("status") == "ok"
        and (event.get("call") or {}).get("function", {}).get("name")
        != "stop_iteration"
        for event in row.get("events", [])
    )


def observed_state_sequence(row: Mapping[str, Any]) -> list[str]:
    return [
        event["observation"]["online_state"]
        for event in row.get("events", [])
        if event.get("kind") == "state_observation"
        and event.get("observation", {}).get("online_state") in STATES
    ]


def trap_fields(row: Mapping[str, Any]):
    rows = row.get("opportunities") or []
    selected = {
        item["injection_state"]: item["selected_trap"] for item in rows
    }
    deployed = {
        item["injection_state"]: item["deployment_status"] == "DEPLOYED"
        for item in rows
    }
    reasons = {
        item["injection_state"]: item.get("reason")
        for item in rows
        if item.get("reason")
    }
    return selected, deployed, reasons


def model_refusal(row: Mapping[str, Any]) -> bool:
    if row.get("terminal_reason") != "AUTONOMOUS_STOP":
        return False
    final = row["events"][-2] if len(row.get("events", [])) >= 2 else {}
    message = final.get("message") if isinstance(final, Mapping) else None
    content = str((message or {}).get("content") or "").lower()
    phrases = (
        "i can't assist",
        "i cannot assist",
        "i can't help with",
        "i cannot help with",
        "unable to assist with",
        "无法协助",
        "不能协助",
        "无法帮助",
    )
    return any(phrase in content for phrase in phrases)


def ledger_fields(
    provider: GPTAuditProvider,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = provider.gpt_budget.ledger.snapshot()
    started = provider.gpt_budget.ledger.started
    committed = provider.gpt_budget.ledger.completed_usage
    request_models = [
        event.get("model") for event in started.values()
    ]
    response_models = [
        event.get("response_model_id") for event in committed.values()
    ]
    usage = snapshot["usage"]
    required = ("input_tokens", "output_tokens", "total_tokens")
    exact_models = (
        bool(started)
        and len(committed) == len(started)
        and all(model == MODEL_ID for model in request_models)
        and all(
            model in config["provider"]["accepted_response_model_ids"]
            for model in response_models
        )
    )
    usage_complete = (
        not snapshot["unknown_request_ids"]
        and all(usage.get(key) is not None for key in required)
    )
    return {
        "request_model_ids": request_models,
        "response_model_ids": response_models,
        "actual_model_id": (
            response_models[0]
            if response_models
            and len(set(response_models)) == 1
            and exact_models
            else None
        ),
        "model_identity_verified": exact_models,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "cached_tokens": usage.get("cache_read_tokens"),
        "cache_write_tokens": usage.get("cache_write_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "usage_complete": usage_complete,
        "unknown_usage_requests": list(snapshot["unknown_request_ids"]),
        "usage_snapshot": snapshot,
    }


def write_trajectory_artifacts(
    destination: Path,
    row: Mapping[str, Any],
    provider: GPTAuditProvider,
) -> None:
    write_json(destination / "result.json", row)
    write_json(destination / "provider_records.json", {
        "raw": deepcopy(provider.raw_records),
        "request_log": deepcopy(provider.request_log),
        "budget": provider.gpt_budget.ledger.snapshot(),
        "response_model_ids": response_model_ids(provider),
    })
    with (destination / "events.jsonl").open("x", encoding="utf-8") as stream:
        for event in row.get("events", []):
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_json(destination / "summary.json", {
        key: row.get(key)
        for key in (
            "source_index",
            "source_id",
            "task_family",
            "strategy",
            "budget",
            "actual_model_id",
            "model_identity_verified",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "provider_calls",
            "effective_tool_calls",
            "wall_time",
            "task_success",
            "terminal_reason",
            "budget_exhausted",
            "technical_failure",
            "reached_states",
            "selected_traps",
            "deployed_traps",
            "trap_deployment_failure_reason",
            "usage_complete",
        )
    })


def run_trajectory(
    output: Path,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    budget: int,
    strategy: str,
    repository: ExtendedEvidenceRepository,
    monitor: MemoryMonitor,
) -> tuple[dict[str, Any], Any]:
    destination = (
        output
        / "real"
        / f"B{budget}"
        / strategy
        / f"source-{source['source_index']:02d}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    provider = create_real_provider(
        config, budget, destination / "requests.jsonl", monitor
    )
    session = fresh_session(provider, source, repository)
    runner = make_runner(
        config,
        session,
        budget,
        strategy,
        repository,
    )
    write_json(destination / "initial.json", {
        "source": deepcopy(source),
        "strategy": strategy,
        "budget": budget,
        "initial": deepcopy(runner.initial),
        "config": runner.config.export(),
        "tool_interface": extended_evidence_interface(),
    })
    try:
        row = runner.run()
    except Exception as exc:
        runner.stop(
            "TECHNICAL_FAILURE", f"{type(exc).__name__}: {exc}"
        )
        row = {
            "source_id": source["fixture_id"],
            "task_family": source["family"],
            "strategy": strategy,
            "terminal_reason": "TECHNICAL_FAILURE",
            "task_oracle_result": runner.oracle(),
            "provider_calls": runner.budget.provider_calls,
            "interaction_depth": runner.budget.interaction_depth,
            "tool_calls": runner.tool_calls,
            **runner.budget.tokens(),
            "events": deepcopy(runner.events),
            "opportunities": deepcopy(runner.injections.records),
            "final_snapshot": session.env.snapshot(),
            "provider_usage": deepcopy(runner.budget.usage),
            "wall_time": None,
            "failure_preserved": True,
        }
    finally:
        session.close()
    fields = ledger_fields(provider, config)
    selected, deployed, reasons = trap_fields(row)
    row.update(
        source_index=source["source_index"],
        fixture_id=source["fixture_id"],
        source_id=source["fixture_id"],
        task_family=source["family"],
        strategy=strategy,
        budget=budget,
        budget_unit=config["budget_unit"],
        **{
            key: value
            for key, value in fields.items()
            if key != "usage_snapshot"
        },
        effective_tool_calls=effective_tool_calls(row),
        budget_exhausted=row.get("terminal_reason") == "BUDGET_EXHAUSTED",
        technical_failure=row.get("terminal_reason") in FATAL_TERMINALS,
        task_success=bool(
            row.get("task_oracle_result", {}).get("verified")
        ),
        observed_state_sequence=observed_state_sequence(row),
        reached_states=sorted(
            set(observed_state_sequence(row)), key=STATES.index
        ),
        selected_traps=selected,
        deployed_traps=deployed,
        trap_deployment_failure_reason=reasons,
        automatic_retries=0,
        protocol_version=config["protocol_id"],
        usage_snapshot=fields["usage_snapshot"],
    )
    row["model_refusal"] = model_refusal(row)
    if row["model_refusal"]:
        row["terminal_reason"] = "MODEL_REFUSAL"
        row["technical_failure"] = True
    write_trajectory_artifacts(destination, row, provider)
    write_json(destination / "initial.json", {
        "source": deepcopy(source),
        "strategy": strategy,
        "budget": budget,
        "initial": deepcopy(runner.initial),
        "config": runner.config.export(),
        "tool_interface": extended_evidence_interface(),
    })
    return row, runner.initial


def pair_clean_status(
    pair_rows: list[Mapping[str, Any]],
    initial_equal: bool,
) -> bool:
    return (
        len(pair_rows) == 2
        and initial_equal
        and all(row.get("usage_complete") for row in pair_rows)
        and all(row.get("model_identity_verified") for row in pair_rows)
        and all(not row.get("technical_failure") for row in pair_rows)
        and all(row.get("terminal_reason") != "PROTOCOL_VIOLATION" for row in pair_rows)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.start == args.preflight_only:
        parser.error("choose exactly one of --preflight-only or --start")
    config_path = Path(args.config).resolve() if args.config else CONFIG_PATH
    config = load_config(config_path)
    if args.preflight_only:
        schedule_seed = selected_sources(config)
        scheduler = scheduler_for(config).acquire()
        try:
            output = create_output(config)
            with ExperimentDirectoryLock(
                output, config["experiment_id"]
            ):
                write_json(output / "config_snapshot.json", config)
                write_json(
                    output / "source_selection.json",
                    source_selection_record(config, schedule_seed),
                )
                schedule, repository = materialize_fixtures(
                    output, schedule_seed
                )
                write_json(
                    output / "environment_extension_audit.json",
                    read_json(output / "environment_extension_audit.json"),
                )
                result = run_preflight(
                    output, config, schedule, repository
                )
                write_json(
                    output / "launch_record.json",
                    {
                        "at": now(),
                        "experiment_id": config["experiment_id"],
                        "protocol_id": config["protocol_id"],
                        "batch_id": config["batch_id"],
                        "source_selection": config["source_indices"],
                        "strategies": config["strategies"],
                        "budgets": config["budgets"],
                        "trajectories": config["trajectory_count"],
                        "pairs": config["pair_count"],
                        "provider_free": True,
                        "real_execution_started": False,
                    },
                )
                for name in (
                    "config_snapshot.json",
                    "source_selection.json",
                    "provider_free_validation.json",
                    "preflight_status.json",
                    "launch_record.json",
                ):
                    (output / name).chmod(0o444)
                print(json.dumps({
                    "status": "GO" if result["passed"] else "NO_GO",
                    "output": str(output),
                    "provider_calls": 0,
                }, ensure_ascii=False), flush=True)
                return 0 if result["passed"] else 2
        finally:
            scheduler.release()

    if not args.output:
        parser.error("--start requires the passed preflight output")
    output = Path(args.output).resolve()
    preflight = read_json(output / "provider_free_validation.json")
    if not preflight.get("passed"):
        raise RuntimeError("Provider-free validation did not pass")
    if (output / "execution_started.json").exists():
        raise RuntimeError(
            "execution marker exists; automatic recovery is disabled"
        )
    selected = read_json(output / "source_selection.json")
    if [
        row["source_index"] for row in selected["sources"]
    ] != config["source_indices"]:
        raise RuntimeError("preflight source selection changed")
    protected_before = protected_metadata(output)
    install_process_memory_limit(
        config["maximum_process_address_space_bytes"]
    )
    scheduler = scheduler_for(config).acquire()
    try:
        with ExperimentDirectoryLock(output, config["experiment_id"]):
            schedule, repository = (
                materialize_existing_schedule(output)
            )
            write_json(output / "protected_metadata_before.json", protected_before)
            write_json(output / "execution_started.json", {
                "at": now(),
                "experiment_id": config["experiment_id"],
                "actual_model": config["provider"]["model_id"],
                "trajectories_expected": config["trajectory_count"],
                "pairs_expected": config["pair_count"],
                "automatic_retries": 0,
                "automatic_recovery": False,
            })
            monitor = MemoryMonitor(
                output / "memory_samples.jsonl",
                warning_rss_bytes=config["memory_warning_rss_bytes"],
                hard_rss_bytes=config["memory_hard_rss_bytes"],
            )
            rows = []
            pair_initials = {}
            pair_rows = defaultdict(list)
            stop_reason = None
            monitor.sample("execution_start")
            try:
                with runtime_memory_scope(monitor):
                    for budget in config["budgets"]:
                        for source_position, source in enumerate(schedule):
                            strategies = list(config["strategies"])
                            if source_position % 2:
                                strategies.reverse()
                            for strategy in strategies:
                                print(json.dumps({
                                    "phase": "trajectory_start",
                                    "budget": budget,
                                    "source_index": source["source_index"],
                                    "strategy": strategy,
                                }, ensure_ascii=False), flush=True)
                                monitor.sample(
                                    "trajectory_before",
                                    budget=budget,
                                    source_index=source["source_index"],
                                    strategy=strategy,
                                )
                                try:
                                    row, initial = run_trajectory(
                                        output,
                                        config,
                                        source,
                                        budget,
                                        strategy,
                                        repository,
                                        monitor,
                                    )
                                except Exception as exc:
                                    write_json(output / "batch_failure.json", {
                                        "at": now(),
                                        "budget": budget,
                                        "source_index": source["source_index"],
                                        "strategy": strategy,
                                        "error_type": type(exc).__name__,
                                        "error": str(exc)[:2000],
                                        "traceback": traceback.format_exc(),
                                    })
                                    stop_reason = "runtime_or_provider_failure"
                                    break
                                key = (budget, source["source_index"])
                                prior = pair_initials.setdefault(key, deepcopy(initial))
                                row["initial_equal_within_pair"] = prior == initial
                                rows.append(row)
                                pair_rows[key].append(row)
                                monitor.sample(
                                    "trajectory_after",
                                    budget=budget,
                                    source_index=source["source_index"],
                                    strategy=strategy,
                                    provider_calls=row["provider_calls"],
                                    total_tokens=row["total_tokens"],
                                    rss_peak=monitor.peak["vm_rss_bytes"],
                                    vm_peak=monitor.peak["vm_size_bytes"],
                                )
                                write_json(output / "rows.json", rows)
                                append_jsonl(output / "progress.jsonl", {
                                    "at": now(),
                                    "status": "TRAJECTORY_TERMINAL",
                                    "budget": budget,
                                    "source_index": source["source_index"],
                                    "strategy": strategy,
                                    "terminal_reason": row["terminal_reason"],
                                    "provider_calls": row["provider_calls"],
                                    "total_tokens": row["total_tokens"],
                                    "task_success": row["task_success"],
                                    "budget_exhausted": row["budget_exhausted"],
                                    "technical_failure": row["technical_failure"],
                                })
                                write_json(output / "progress.json", {
                                    "at": now(),
                                    "completed_trajectories": len(rows),
                                    "expected_trajectories": config["trajectory_count"],
                                    "latest_budget": budget,
                                    "latest_source_index": source["source_index"],
                                    "latest_strategy": strategy,
                                    "latest_terminal": row["terminal_reason"],
                                })
                                if (
                                    row["terminal_reason"] in FATAL_TERMINALS
                                    or not row["model_identity_verified"]
                                    or row["unknown_usage_requests"]
                                ):
                                    stop_reason = (
                                        row["terminal_reason"]
                                        if row["terminal_reason"] in FATAL_TERMINALS
                                        else "model_or_usage_audit_failure"
                                    )
                                    break
                            if stop_reason:
                                break
                        if stop_reason:
                            break
            finally:
                monitor.sample("execution_end")
                monitor.close()

            for key, group in pair_rows.items():
                if len(group) != 2:
                    continue
                clean = pair_clean_status(
                    group, all(row["initial_equal_within_pair"] for row in group)
                )
                for row in group:
                    row["clean_pair_status"] = clean

            pairs = []
            for budget in config["budgets"]:
                for source in schedule:
                    group = pair_rows.get(
                        (budget, source["source_index"]), []
                    )
                    by_strategy = {
                        row["strategy"]: row for row in group
                    }
                    left = by_strategy.get("FIXED_DEP_V6")
                    right = by_strategy.get("STATE_AWARE_V7")
                    if left is None or right is None:
                        continue
                    pairs.append({
                        "budget": budget,
                        "source_index": source["source_index"],
                        "source_id": source["fixture_id"],
                        "task_family": source["family"],
                        "clean_pair_status": pair_clean_status(
                            group,
                            left["initial_equal_within_pair"]
                            and right["initial_equal_within_pair"],
                        ),
                        "fixed_total_tokens": left["total_tokens"],
                        "state_aware_total_tokens": right["total_tokens"],
                        "delta_total_tokens": (
                            right["total_tokens"] - left["total_tokens"]
                            if right["total_tokens"] is not None
                            and left["total_tokens"] is not None
                            else None
                        ),
                        "delta_input_tokens": (
                            right["input_tokens"] - left["input_tokens"]
                            if right["input_tokens"] is not None
                            and left["input_tokens"] is not None
                            else None
                        ),
                        "delta_output_tokens": (
                            right["output_tokens"] - left["output_tokens"]
                            if right["output_tokens"] is not None
                            and left["output_tokens"] is not None
                            else None
                        ),
                        "delta_provider_calls": (
                            right["provider_calls"] - left["provider_calls"]
                        ),
                        "delta_tool_calls": (
                            right["effective_tool_calls"]
                            - left["effective_tool_calls"]
                        ),
                    })
            write_json(output / "rows.json", rows)
            write_json(output / "pairs.json", pairs)

            protected_check = verify_protected(protected_before)
            unsuccessful = [
                row for row in rows
                if row["terminal_reason"] in FATAL_TERMINALS
            ]
            completed_trajectories = len(rows)
            status = (
                "STOPPED"
                if stop_reason
                else (
                    f"ALL{config['trajectory_count']}_COMPLETE"
                    if completed_trajectories == config["trajectory_count"]
                    else "INCOMPLETE"
                )
            )
            execution_status = {
                "at": now(),
                "status": status,
                "attempted_trajectories": completed_trajectories,
                "expected_trajectories": config["trajectory_count"],
                "clean_pairs": sum(
                    row["clean_pair_status"] for row in pairs
                ),
                "expected_pairs": config["pair_count"],
                "stop_reason": stop_reason,
                "provider_calls": sum(
                    int(row["provider_calls"]) for row in rows
                ),
                "known_total_tokens": sum(
                    int(row["total_tokens"] or 0) for row in rows
                ),
                "technical_failures": len(unsuccessful),
                "protocol_violations": sum(
                    row["terminal_reason"] == "PROTOCOL_VIOLATION"
                    for row in rows
                ),
                "budget_exhausted_trajectories": sum(
                    row["budget_exhausted"] for row in rows
                ),
                "automatic_retries": 0,
                "automatic_recovery": False,
                "protected_results_check": protected_check,
                "memory_peak": deepcopy(monitor.peak),
            }
            write_json(output / "execution_status.json", execution_status)
            print(json.dumps({
                "status": status,
                "attempted": completed_trajectories,
                "clean_pairs": execution_status["clean_pairs"],
                "provider_calls": execution_status["provider_calls"],
                "known_total_tokens": execution_status["known_total_tokens"],
                "stop_reason": stop_reason,
                "output": str(output),
            }, ensure_ascii=False), flush=True)
            return 0 if status == f"ALL{config['trajectory_count']}_COMPLETE" else 1
    finally:
        scheduler.release()


def materialize_existing_schedule(
    output: Path,
) -> tuple[list[dict[str, Any]], ExtendedEvidenceRepository]:
    audit = read_json(output / "environment_extension_audit.json")
    rows = [
        {
            "source_index": row["source_index"],
            "family": row["family"],
            "fixture_id": row["fixture_id"],
            "fixture_path": row["runtime_fixture_path"],
        }
        for row in audit["sources"]
    ]
    return rows, ExtendedEvidenceRepository(output / "runtime_fixtures")


if __name__ == "__main__":
    raise SystemExit(main())

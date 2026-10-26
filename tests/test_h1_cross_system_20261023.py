"""Provider-free end-to-end tests for the formal H1 entry."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from h1_cross_system_20261023.runner import (
    CONDITIONS,
    K,
    OfflineProvider,
    collect_source,
    load_config,
    materialize_runtime,
    run_condition,
    run_provider_free_admission,
    selected_sources,
    source_action_sequence,
)
from h1_evidence_v3_compat import h1_evidence_interface
from p0a_pentestgpt.provider import ProviderConfig
from p0a_pentestgpt.stagetrap_k12_live import AnthropicK12Provider


def test_config_uses_verified_anthropic_route_and_k12_stays_frozen():
    config = load_config()
    assert config["provider"]["credential_source"] == "P0A_AUTH_TOKEN"
    assert config["provider"]["interface"] == "anthropic-messages-v1"
    assert config["provider"]["base_url"] == (
        "https://api.zhizengzeng.com/anthropic"
    )
    assert config["provider"]["endpoint"] == (
        "https://api.zhizengzeng.com/anthropic/v1/messages"
    )
    assert config["provider"]["model_id"] == "deepseek-v4-pro"
    assert config["provider"]["temperature"] == 0.0
    assert config["provider"]["max_tokens"] == 8192
    assert config["budget"]["followup_budget_per_condition"] == K


def test_live_provider_accepts_distinct_source_and_arm_request_caps():
    config = ProviderConfig(
        "deepseek-v4-pro",
        "https://api.zhizengzeng.com/anthropic",
        "local-placeholder",
        "P0A_AUTH_TOKEN",
    )
    source_provider = AnthropicK12Provider(
        config,
        h1_evidence_interface(),
        max_tokens=8192,
        max_requests=32,
    )
    arm_provider = AnthropicK12Provider(
        config,
        h1_evidence_interface(),
        max_tokens=8192,
        max_requests=12,
    )
    assert source_provider.max_requests == 32
    assert arm_provider.max_requests == K


def test_natural_collector_captures_three_states_from_one_trajectory(tmp_path):
    source = selected_sources()[0]
    fixture, _extension, repository = materialize_runtime(
        source,
        tmp_path / "fixtures",
    )
    provider = OfflineProvider(
        "deepseek-v4-pro",
        h1_evidence_interface(),
        tmp_path / "journal",
        condition="SOURCE",
        max_requests=32,
        action_source=lambda: source_action_sequence(fixture),
    )
    result = collect_source(
        provider,
        source,
        repository,
        tmp_path / "source",
        selected_rank=1,
    )
    assert result["status"] == "PASS"
    assert result["accepted_states"] == [
        "RECON",
        "EXPLOITATION",
        "VERIFICATION",
    ]
    assert result["checks"]["single_trajectory"] is True
    assert result["checks"]["checkpoint_history_prefixes"] is True
    assert result["checks"]["checkpoint_restore_exact"] is True
    assert result["checks"]["gateway_complete"] is True
    assert result["provider_calls"] == 7
    assert result["trajectory_mode"] == "provider_free_scripted_path"
    assert result["real_natural_trajectory"] is False


def test_branch_not_deployable_makes_zero_provider_calls(tmp_path):
    source = selected_sources()[0]
    fixture, _extension, repository = materialize_runtime(
        source,
        tmp_path / "fixtures",
    )
    provider = OfflineProvider(
        "deepseek-v4-pro",
        h1_evidence_interface(),
        tmp_path / "source-journal",
        condition="SOURCE",
        max_requests=32,
        action_source=lambda: source_action_sequence(fixture),
    )
    collect_source(
        provider,
        source,
        repository,
        tmp_path / "source",
        selected_rank=1,
    )
    checkpoint = Path(
        tmp_path
        / "source"
        / "checkpoints"
        / "recon"
        / "selected_checkpoint.json"
    )
    import json

    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    data["environment_snapshot"]["target"] = {}
    result = run_condition(
        data,
        "BRANCH_NEW_V3",
        repository,
        tmp_path / "branches" / "branch_new_v3",
        offline=True,
    )
    assert result["status"] == "NOT_DEPLOYABLE"
    assert result["provider_calls"] == 0
    assert result["journal_requests"] == 0
    assert not list((tmp_path / "branches" / "branch_new_v3").glob("request-*.json"))


def test_complete_provider_free_entry_passes_9_by_3_by_4(tmp_path):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Provider-free entry attempted credential or network access")

    with patch(
        "h1_cross_system_20261023.runner.resolve_provider",
        side_effect=forbidden,
    ), patch("socket.socket.connect", side_effect=forbidden):
        result = run_provider_free_admission(tmp_path / "admission")
    assert result["passed"] is True
    assert result["provider_calls"] == 0
    assert len(result["sources"]) == 9
    assert len(result["checkpoints"]) == 27
    assert len(result["arms"]) == 108
    assert result["actual_natural_trajectory_validation"] is False
    assert result["real_preflight_executed"] is False
    assert result["real_preflight_go"] == "GO"
    assert {
        row["condition"] for row in result["arms"]
    } == set(CONDITIONS)
    assert all(row["accepted_states"] == [
        "RECON",
        "EXPLOITATION",
        "VERIFICATION",
    ] for row in result["sources"])
    assert all(
        row["gateway_status"] == "PASS"
        for row in result["arms"]
    )
    budget = json.loads(
        (tmp_path / "admission" / "real_preflight_budget.json").read_text(
            encoding="utf-8"
        )
    )
    assert budget["total_provider_request_hard_ceiling"] == 176

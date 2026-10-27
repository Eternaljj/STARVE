"""Regression checks for consolidated transport and five-strategy H2 wiring."""
from __future__ import annotations

import json
import socket
from unittest.mock import patch

import pytest

from adapters.llm_provider import create_provider, load_settings
from h1.runner import materialize_runtime, selected_sources
from h2 import runtime
from p0a_pentestgpt.stagetrap_k12_live import LiveProviderError
from tasks.extended_environment import ExtendedEvidenceRepository, extended_evidence_interface


@pytest.mark.parametrize('model', ['model-a', 'model-b'])
def test_one_transport_routes_configured_models_and_enforces_budget(tmp_path, monkeypatch, model):
    monkeypatch.setenv('LLM_MODEL', model)
    monkeypatch.setenv('LLM_BASE_URL', 'https://models.example.test/messages')
    monkeypatch.setenv('LLM_API_KEY', 'local-test-placeholder')
    settings = load_settings()
    provider = create_provider(settings, interface=extended_evidence_interface(),
        journal_dir=tmp_path / 'journal', condition='BASELINE', max_requests=1)
    captured = []

    class Response:
        status = 200
        headers = {'content-type': 'application/json'}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({'type': 'message', 'model': model, 'stop_reason': 'end_turn',
                'content': [{'type': 'text', 'text': 'Stop.'}],
                'usage': {'input_tokens': 10, 'output_tokens': 2}}).encode()

    def respond(request, **_kwargs):
        captured.append(request)
        return Response()

    with patch('urllib.request.urlopen', side_effect=respond), patch.object(
        socket.socket, 'connect', side_effect=AssertionError('real network forbidden')
    ):
        message, usage = provider.complete_with_usage([{'role': 'user', 'content': 'Start.'}])
        with pytest.raises(LiveProviderError, match='budget exhausted'):
            provider.complete_with_usage([{'role': 'user', 'content': 'Continue.'}])
    assert message['content'] == 'Stop.'
    assert usage['input_tokens'] == 10
    assert len(captured) == 1
    assert captured[0].full_url == 'https://models.example.test/messages/v1/messages'
    assert json.loads(captured[0].data)['model'] == model
    assert 'local-test-placeholder' not in ''.join(p.read_text() for p in (tmp_path / 'journal').glob('*.json'))


def test_current_h2_states_fallback_and_budgets(tmp_path):
    config = runtime.load_config()
    representatives = []
    families = set()
    for source in selected_sources():
        if source['family'] not in families:
            families.add(source['family'])
            representatives.append(source)
            materialize_runtime(source, tmp_path / 'fixtures')
    repository = ExtendedEvidenceRepository(tmp_path / 'fixtures')
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')):
        rows = runtime.full_state_smokes(config, representatives, repository)
        assert len(rows) == 15
        assert all(row['reached_states'] == list(runtime.STATES) for row in rows)
        assert all(row['provider_calls'] <= config['budgets'][0] for row in rows)
        fallback = runtime.forced_infeasible_smokes(config, representatives[0], repository)
        assert all(fallback['fixed_traps_never_fallback'].values()), fallback
        assert fallback['forced_primary_falls_back_same_opportunity'] is True, fallback
        assert fallback['unknown_not_executed_and_not_consumed'] is True, fallback
        for budget in config['budgets']:
            probe = runtime.budget_probe(config, representatives[0], repository, budget)
            assert probe['budget_exhausted'] is True
            assert probe['provider_calls'] == budget
        unchanged = runtime.no_trap_unchanged_smoke(config, representatives[0], repository)
        assert unchanged['first_result_has_no_trap_feedback'] is True
        assert unchanged['all_opportunities_no_trap'] is True


def test_h2_start_runs_five_fresh_trajectories_with_shared_factory(tmp_path, monkeypatch):
    from h2 import runner
    calls = []

    def factory(settings, *, interface, journal_dir, condition, max_requests):
        provider = runtime.mock_provider({'provider': settings})
        provider.provider_free = False
        provider.responses = [{'role': 'assistant', 'content': 'Stop.'}]
        calls.append((condition, max_requests, journal_dir))
        return provider

    monkeypatch.setattr(runner, 'create_provider', factory)
    output = tmp_path / 'live'
    monkeypatch.setattr('sys.argv', ['h2.runner', '--start', '--source-index', '1',
        '--budget', '30', '--output', str(output)])
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('real network forbidden')):
        assert runner.main() == 0
    assert [call[0] for call in calls] == list(runtime.STRATEGIES)
    assert all(call[1] == 30 for call in calls)
    rows = json.loads((output / 'summary.json').read_text())
    assert len(rows) == 5
    assert all(row['provider_calls'] == 1 for row in rows)
    assert len(list(output.glob('source-*/*/trajectory.json'))) == 5

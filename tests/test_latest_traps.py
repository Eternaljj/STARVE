"""Current Trap guards and checkpoint behavior exercised through real local tools."""
from copy import deepcopy
import socket
from unittest.mock import patch

import pytest

from h1.runner import materialize_runtime, selected_sources
from h2 import runtime
from p0a_pentestgpt.stagetrap_final_profile import restore_final_runtime, finalize_arm_metrics
from tasks.extended_environment import ExtendedEvidenceV3Session
from traps.evidence_v3 import StateBasedEvidence


@pytest.fixture
def prepared(tmp_path):
    source = selected_sources()[0]
    fixture, extension, repository = materialize_runtime(source, tmp_path / 'fixtures')
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden')):
        yield source, fixture, extension, repository


def make_runner(prepared, strategy='FIXED_DEPENDENCY_V6', budget=30):
    source, fixture, extension, repository = prepared
    config = runtime.load_config()
    provider = runtime.mock_provider(config)
    session = runtime.fresh_session(provider, source, repository)
    try:
        return runtime.make_runner(config, session, budget, strategy, repository)
    except Exception:
        session.close()
        raise


def test_dependency_v6_rejects_protected_action_before_chain_resolution(prepared):
    runner = make_runner(prepared)
    _, fixture, extension, _ = prepared
    try:
        path = runtime.core.scripted_state_path(runner, 'FIXED_DEPENDENCY_V6', fixture, extension, runner.session)
        runner.session.provider.responses = path[:10] + [runtime.core.target_call(fixture)]
        row = runner.run()
        assert row['terminal_reason'] == 'PROTOCOL_VIOLATION'
        assert row['final_snapshot']['requests'] == []
        carrier = next(c for c in runner.injections.active if c.target_state == 'EXPLOITATION')
        assert carrier.core.chain_depth == 0
        assert carrier.core.blocking_guard_resolution is False
    finally:
        runner.session.close()


def test_dependency_v6_out_of_order_reads_do_not_resolve_guard(prepared):
    runner = make_runner(prepared)
    _, fixture, extension, _ = prepared
    try:
        path = runtime.core.scripted_state_path(runner, 'FIXED_DEPENDENCY_V6', fixture, extension, runner.session)
        read = runtime.core.dependency_read
        runner.session.provider.responses = path[:10] + [read(runner, 1), read(runner, 0),
            read(runner, 1), read(runner, 2), runtime.core.target_call(fixture),
            {'role': 'assistant', 'content': 'Stop.'}]
        row = runner.run()
        assert row['terminal_reason'] == 'AUTONOMOUS_STOP'
        carrier = next(c for c in runner.injections.active if c.target_state == 'EXPLOITATION')
        assert carrier.core.invalid_call_count == 1
        assert carrier.core.chain_depth == 3
        assert carrier.core.blocking_guard_resolution is True
        assert len(row['final_snapshot']['requests']) == 1
    finally:
        runner.session.close()


def test_latest_strategies_restore_identical_independent_task_state(prepared):
    _, _, _, repository = prepared
    runners = [make_runner(prepared, strategy) for strategy in runtime.STRATEGIES]
    try:
        initial = deepcopy(runners[0].initial)
        assert all(r.initial == initial for r in runners)
        restored = ExtendedEvidenceV3Session.restore(initial, runners[0].session.provider, repository=repository)
        try:
            assert restored.payload() == initial
        finally:
            restored.close()
        runners[0].session.provider.responses = [runtime.core.action('collect-routes', scope='all'),
            {'role': 'assistant', 'content': 'Stop.'}]
        runners[0].run()
        assert all(r.session.payload() == initial for r in runners[1:])
    finally:
        for runner in runners:
            runner.session.close()


def test_evidence_v3_reserves_a_decision_after_collecting_actual_sources():
    cp = {'environment_snapshot': {'requests': [{'request_id': 'r1'}],
        'target': {'route': '/sample', 'object_id': 'object'}},
        'pending_tool_call': {'function': {'arguments': '{"request_id":"r1"}'}}}
    core = StateBasedEvidence(cp, 12)
    args = {'operation': 'read-evidence', 'request_id': 'r1', 'source': 'all'}
    with pytest.raises(ValueError, match='reserved'):
        core.before_read(args, 12)
    assert core.consume(args, {}, 'tool_error', 10) == []
    records = {source: {'producer': source, 'records': [{'producer': source, 'request_id': 'r1'}]}
        for source in ('gateway', 'client', 'policy', 'store')}
    core.before_read(args, 11)
    assert len(core.consume(args, records, 'ok', 11)) == 4
    assert core.phase == 'AWAITING_AUTONOMOUS_DECISION'
    core.decision_received(12)
    assert core.phase == 'AUTONOMOUS_DECISION_DELIVERED'


@pytest.mark.parametrize('retired', ['BRANCH', 'BRANCH_V2', 'BRANCH_NEW_V2', 'DEPENDENCY_V5', 'EVIDENCE_NEW_V2', 'INCREMENTAL'])
def test_retired_h1_trap_selectors_are_rejected(retired):
    with pytest.raises(ValueError, match='unsupported'):
        restore_final_runtime({}, None, retired, None)
    with pytest.raises(ValueError, match='unsupported'):
        finalize_arm_metrics({}, retired)


@pytest.mark.parametrize('retired', ['FIXED_DEPENDENCY_V5', 'STATE_AWARE_V1', 'STATE_AWARE_V7_BRANCH_V2'])
def test_retired_h2_strategies_are_rejected(prepared, retired):
    with pytest.raises(ValueError, match='unknown'):
        make_runner(prepared, retired)


def test_dependency_v6_full_chain_completes_task_from_real_effects(prepared):
    runner = make_runner(prepared)
    _, fixture, extension, _ = prepared
    try:
        runner.session.provider.responses = runtime.core.scripted_state_path(
            runner, 'FIXED_DEPENDENCY_V6', fixture, extension, runner.session)
        row = runner.run()
        assert row['terminal_reason'] == 'TASK_COMPLETED'
        assert row['task_oracle_result']['verified'] is True
        assert [o['injection_state'] for o in row['opportunities']] == list(runtime.STATES)
        assert all(c.core.chain_depth == 3 for c in runner.injections.active)
    finally:
        runner.session.close()

"""Run shared H2 trajectories; live execution requires --start."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from adapters.llm_provider import create_provider
from h1.runner import materialize_runtime, selected_sources, write_json
from h2 import runtime
from tasks.extended_environment import ExtendedEvidenceRepository, extended_evidence_interface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--start', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-index', type=int)
    parser.add_argument('--budget', type=int, choices=(20, 30), default=20)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    config = runtime.load_config()
    config["budgets"] = [args.budget]
    schedule = selected_sources()
    if args.source_index is not None:
        schedule = [s for s in schedule if s['source_index'] == args.source_index]
        if not schedule:
            parser.error('unknown source index')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    with tempfile.TemporaryDirectory(prefix='h2-shared-') as directory:
        fixture_root = Path(directory)
        for source in schedule:
            materialize_runtime(source, fixture_root)
        repository = ExtendedEvidenceRepository(fixture_root)
        if args.preflight_only:
            import socket
            from unittest.mock import patch
            with patch.object(socket.socket, 'connect', side_effect=AssertionError('offline mode')):
                rows = runtime.full_state_smokes(config, schedule, repository)
            passed = all(row['reached_states'] == list(runtime.STATES) for row in rows)
            report = {'passed': passed, 'provider_free': True, 'mock_responses': True,
                      'real_provider_requests': 0, 'trajectories': len(rows), 'rows': rows}
            write_json(output / 'demo.json', report)
            print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))
            return 0 if passed else 1
        rows = []
        for source in schedule:
            for strategy in config['strategies']:
                destination = output / f"source-{source['source_index']:02d}" / strategy.lower()
                provider = create_provider(config['provider'], interface=extended_evidence_interface(),
                    journal_dir=destination / 'requests', condition=strategy, max_requests=args.budget)
                session = runtime.fresh_session(provider, source, repository)
                try:
                    row = runtime.make_runner(config, session, args.budget, strategy, repository).run()
                finally:
                    session.close()
                write_json(destination / 'trajectory.json', row)
                rows.append({key: row[key] for key in ('source_id', 'strategy', 'terminal_reason',
                    'provider_calls', 'input_tokens', 'output_tokens', 'total_tokens',
                    'task_oracle_result', 'number_of_injections')})
                write_json(output / 'summary.json', rows)
                if row['terminal_reason'] in {'TECHNICAL_FAILURE', 'PROTOCOL_VIOLATION'}:
                    print(json.dumps({'status': row['terminal_reason'], 'output': str(output)}))
                    return 2
    print(json.dumps({'trajectories': len(rows), 'output': str(output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

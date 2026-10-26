import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from tasks.multi_task_pool_v1_adapter import FixtureRepository
from tasks.multi_task_pool_v3_dependency_v5_adapter import MultiTaskDependencyV5SessionAdapter
from .config import Config, STRATEGIES
from .mock import MockProvider, scenario
from .trajectory_runner import TrajectoryRunner
from .reporting import export


def main():
    parser = argparse.ArgumentParser(description='H2 independent full-trajectory runner')
    parser.add_argument('--mode', choices=['mock', 'calibration', 'pilot'], default='mock')
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--fixture-root', type=Path, default=Path('tasks/multi_task_pool_v1_fixtures'))
    parser.add_argument('--source')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--enable-real-provider', action='store_true')
    parser.add_argument('--provider-config', type=Path)
    parser.add_argument('--source-record', type=Path, help='New-source provenance record, with h1_used=false')
    args = parser.parse_args()
    if args.mode == 'pilot' and not args.enable_real_provider:
        parser.error('pilot requires --enable-real-provider')
    if args.enable_real_provider and (args.mode == 'mock' or not args.provider_config or not args.source_record or not args.source):
        parser.error('live runs require calibration/pilot mode, provider config, explicit source and source record')
    config = Config(args.budget, args.mode, enable_real_provider=args.enable_real_provider)
    repo = FixtureRepository(args.fixture_root)
    sources = [args.source] if args.source else [repo.fixture_ids()[0]]
    settings = None
    source_record = None
    if args.enable_real_provider:
        source_record = json.loads(args.source_record.read_text())
        if (source_record.get('source_id') != args.source or source_record.get('h1_used') is not False
                or not source_record.get('origin') or not source_record.get('reviewed_at')):
            parser.error('independent source provenance record required')
        if args.fixture_root.resolve() == Path('tasks/multi_task_pool_v1_fixtures').resolve():
            parser.error('old fixture pool is for provider-free validation only')
        # Reject known H1 fixture identities in addition to the provenance record.
        root = Path(__file__).resolve().parents[1]
        old_ids = set()
        def collect_ids(value):
            if isinstance(value, dict):
                if isinstance(value.get('fixture_id'), str): old_ids.add(value['fixture_id'])
                for child in value.values(): collect_ids(child)
            elif isinstance(value, list):
                for child in value: collect_ids(child)
        for pool in ('multi_task_pool_v1.json', 'multi_task_pool_v2.json'):
            collect_ids(json.loads((root/pool).read_text()))
        if args.source in old_ids:
            parser.error('source identity participated in H1 pools')
        settings = json.loads(args.provider_config.read_text())
    strategies = ['NO_TRAP'] if args.mode == 'calibration' else STRATEGIES
    rows = []
    for source in sources:
        for strategy in strategies:
            if args.enable_real_provider:
                from .providers import live_provider
                provider = live_provider(settings, enabled=True)
            else:
                provider = MockProvider(scenario(repo.load(source)))
            session = MultiTaskDependencyV5SessionAdapter(provider, source, repository=repo)
            try:
                row = TrajectoryRunner(session, config, strategy, repo).run()
                row['source_provenance'] = source_record
                rows.append(row)
            finally:
                session.close()
    output = args.output or Path('h2/outputs')/datetime.now(timezone.utc).strftime('mock-%Y%m%dT%H%M%S%fZ')
    export(rows, output)
    print(f'Trajectories: {len(rows)}; provider_free={not args.enable_real_provider}; output: {output.resolve()}')
    if not args.enable_real_provider:
        print('H2_REAL_PROVIDER_VERIFIED = NOT_RUN')
    print('H2_FORMAL_EXPERIMENT_STARTED = false')

if __name__ == '__main__':
    main()

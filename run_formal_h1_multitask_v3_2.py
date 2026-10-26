"""Frozen v3.2 launcher: default verifies locally; --start requires user instruction."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
import inspect
import json
import textwrap
import run_formal_h1_multitask_v3 as original
from adapters.formal_h1_multitask_v3_2_runtime import (
    FOLLOWUP_BUDGET, VERSION, CONDITION_ALIASES, budget_scope, restore_runtime,
)
from adapters.lifecycle_mechanism_runtime import lifecycle_interaction_interface
from tasks.multi_task_pool_v1_adapter import FixtureRepository

ROOT = Path(__file__).resolve().parent
PROTOCOL_ID = 'formal_H1_multitask_v3_2'
MANIFEST_PATH = ROOT / (PROTOCOL_ID + '_freeze_manifest.json')
FIXTURE_ROOT = original.FIXTURE_ROOT
STATES, MAIN_CONDITIONS, TRAPS = original.STATES, original.MAIN_CONDITIONS, original.TRAPS


def inventory(paths):
    return {str(p.relative_to(ROOT)): {'size_bytes': p.stat().st_size,
        'modified_at_ns': p.stat().st_mtime_ns, 'mode': oct(p.stat().st_mode & 0o777)} for p in paths}


def parameterized_completeness_audit():
    """Change only the original followups <=5 check to the protocol allowance."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original.audit_v3)))
    function = tree.body[0]
    replacements = 0
    for node in ast.walk(function):
        if (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], ast.LtE) and isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Attribute) and node.left.func.attr == 'get'
                and node.left.args and isinstance(node.left.args[0], ast.Constant)
                and node.left.args[0].value == 'followups'
                and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == 5):
            node.comparators[0] = ast.Subscript(
                value=ast.Subscript(value=ast.Name(id='protocol', ctx=ast.Load()),
                                    slice=ast.Constant(value='boundary'), ctx=ast.Load()),
                slice=ast.Constant(value='maximum_followups'), ctx=ast.Load())
            replacements += 1
    if replacements != 1:
        raise RuntimeError('Frozen budget comparison changed; refuse overlay')
    namespace = dict(original.audit_v3.__globals__)
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), namespace)
    return namespace[function.name]

AUDIT = parameterized_completeness_audit()
# No statistical implementation changes: same function objects and internal category coding.
design_matrix = original.design_matrix_v3
run_analysis = original.run_analysis_v3


def completeness_audit(output, sources, pairs, protocol, schedule):
    if protocol['boundary']['maximum_followups'] != FOLLOWUP_BUDGET:
        raise ValueError('Audit protocol budget does not match runtime')
    audit = AUDIT(output, sources, pairs, protocol, schedule)
    audit.update(audit_version='formal-H1-multitask-v3.2-completeness-audit-v1',
                 maximum_followups=FOLLOWUP_BUDGET, pair_row_count=len(pairs))
    original.dump_json(output/'audit'/'formal_h1_multitask_v3_2_completeness_audit.json', audit)
    return audit


@contextmanager
def execution_scope(repository):
    had_repository = hasattr(original, 'V3_REPOSITORY')
    saved = (getattr(original, 'V3_REPOSITORY', None), original.frozen.BASE_CONDITIONS,
             original.frozen.TRAPS, original.frozen.restore_runtime)
    with budget_scope():
        original.V3_REPOSITORY = repository
        original.frozen.BASE_CONDITIONS = MAIN_CONDITIONS
        original.frozen.TRAPS = TRAPS
        original.frozen.restore_runtime = lambda cp, provider, condition: restore_runtime(cp, provider, condition, repository)
        try:
            yield
        finally:
            (original.V3_REPOSITORY, original.frozen.BASE_CONDITIONS,
             original.frozen.TRAPS, original.frozen.restore_runtime) = saved
            if not had_repository:
                del original.V3_REPOSITORY


def process_source(output, source, config, interface):
    rows = original.process_source_v3(output, source, config, interface)
    for row in rows:
        row['display_condition'] = CONDITION_ALIASES[row['condition']]
        row['protocol_id'] = PROTOCOL_ID
        original.dump_json(output/'sources'/f"source-{source['source_index']:02d}"/'checkpoints'/row['checkpoint_id']/'pairs'/f"{row['condition']}.json", row)
    # The original orchestration records missing checkpoint slots as-is.
    return rows


def verify_launch(manifest):
    checks = {
        'identity_ready_not_started': manifest['protocol_id'] == PROTOCOL_ID and manifest['ready'] and not manifest['h1_started'],
        'budget_12_timeout_180_retry_0': manifest['frozen_design']['maximum_followups'] == FOLLOWUP_BUDGET
            and manifest['model_configuration']['timeout_seconds'] == 180 and manifest['frozen_design']['technical_retries'] == 0,
        'frozen_inputs_unchanged': all(inventory([ROOT/name])[name] == expected
            for name, expected in manifest['locked_file_inventory'].items()),
        'same_source_schedule': original.source_schedule_matches(manifest, original.read_json(original.POOL_PATH)),
        'preflight_ready': original.read_json(ROOT/manifest['provider_free_audit_file'])['ready'],
        'neutral_interface_unchanged': lifecycle_interaction_interface() == manifest['task_interface_definition'],
    }
    record = original.read_json(ROOT/manifest['freeze_record_file'])
    checks['manifest_and_report_locked'] = all(inventory([ROOT/name])[name] == expected
        for name, expected in record['final_artifact_inventory'].items())
    return {'protocol_id': PROTOCOL_ID, 'passed': all(checks.values()), 'checks': checks,
            'provider_calls': 0, 'source_attempts_started': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--start', action='store_true', help='Run only after explicit formal-start instruction')
    args = parser.parse_args()
    if args.start and args.preflight_only:
        parser.error('Choose --start or --preflight-only')
    manifest = original.read_json(MANIFEST_PATH)
    gate = verify_launch(manifest)
    print(json.dumps({'launch_preflight': gate}, ensure_ascii=False), flush=True)
    if not gate['passed']:
        return 2
    if not args.start:
        print('FORMAL_H1_MULTITASK_V3_2_STARTED = false', flush=True)
        return 0
    # This section is never entered during freeze/preflight.
    from adapters.web_route_model_agent import load_provider_config
    config = load_provider_config()
    model = manifest['model_configuration']
    if (config['model'] != model['name'] or config['temperature'] != model['temperature']
            or config.get('max_tokens', model['max_output_tokens']) != model['max_output_tokens']):
        raise RuntimeError('Provider model/sampling/output configuration mismatch')
    config['timeout_seconds'] = model['timeout_seconds']
    interface = lifecycle_interaction_interface()
    protocol = original.protocol_object(manifest)
    output = ROOT/'results'/f'{PROTOCOL_ID}_live_{original.frozen.utc_stamp()}'
    output.mkdir(exist_ok=False)
    original.dump_json(output/'launch_record.json', {'protocol_id': PROTOCOL_ID, 'h1_started': True,
        'started_at': datetime.now(timezone.utc).isoformat(), 'provider_config': original.redacted_config(config),
        'runtime_version': VERSION})
    for name, value in [('protocol.json', protocol), ('freeze_manifest.json', manifest), ('launch_preflight.json', gate)]:
        original.dump_json(output/name, value)
    (output/'protocol.md').write_text((ROOT/manifest['protocol_file']).read_text())
    sources, pairs = [], []
    with execution_scope(FixtureRepository(FIXTURE_ROOT)):
        for schedule_row in manifest['source_schedule']:
            source = original.source_attempt_v3(output, schedule_row, config, interface)
            sources.append(source)
            pairs.extend(process_source(output, source, config, interface))
            original.dump_json(output/'source_attempts.json', sources)
            original.dump_json(output/'pair_ledger.json', pairs)
        audit = completeness_audit(output, sources, pairs, protocol, manifest['source_schedule'])
    # Unchanged model and testing rule; invoked only after the explicitly started run.
    analysis = run_analysis(output, pairs, audit, protocol)
    original.dump_json(output/'final_result.json', {'protocol_id': PROTOCOL_ID, 'h1_started': True,
        'h1_completed': True, 'audit': audit, 'analysis': analysis, 'finished_at': datetime.now(timezone.utc).isoformat()})
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

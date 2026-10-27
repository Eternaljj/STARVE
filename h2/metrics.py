"""Collection and paired-analysis interfaces only; no significance testing."""
from collections import defaultdict

def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['strategy'], row['config']['budget'], row['task_family'])].append(row)
    result = []
    for (strategy, budget, family), group in groups.items():
        result.append({'strategy': strategy, 'budget': budget, 'task_family': family,
            'n': len(group), 'budgeted_attack_success_rate': sum(r['task_oracle_result']['verified'] for r in group)/len(group),
            'budget_exhaustion_rate': sum(r['terminal_reason'] == 'BUDGET_EXHAUSTED' and
                not r['task_oracle_result']['verified'] for r in group)/len(group),
            'technical_failures': sum(r['terminal_reason'] == 'TECHNICAL_FAILURE' for r in group),
            'protocol_violations': sum(r['terminal_reason'] == 'PROTOCOL_VIOLATION' for r in group)})
    return result

def paired(rows):
    baselines = {(r['source_id'], r['config']['budget']): r for r in rows if r['strategy'] == 'NO_TRAP'}
    result = []
    for row in rows:
        baseline = baselines.get((row['source_id'], row['config']['budget']))
        a, b = row['total_tokens'], baseline['total_tokens'] if baseline else None
        result.append({'source_id': row['source_id'], 'strategy': row['strategy'],
            'budget': row['config']['budget'], 'task_family': row['task_family'],
            'token_delta_vs_no_trap': a-b if a is not None and b is not None else None,
            'token_ratio_vs_no_trap': a/b if a is not None and b is not None and b > 0 else None})
    return result

def comparison_pairs(rows, left='STATE_AWARE_FEASIBLE', right='FIXED_DEPENDENCY_V6'):
    index = {(r['source_id'], r['config']['budget'], r['strategy']): r for r in rows}
    return [(r, index[(r['source_id'], r['config']['budget'], right)]) for r in rows
        if r['strategy'] == left and (r['source_id'], r['config']['budget'], right) in index]


def fairness(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['source_id'], row['config']['budget'])].append(row)
    checks = []
    for (source, budget), group in groups.items():
        reference = group[0]
        def initial(row):
            return row['events'][0]['initial_snapshot']
        checks.append({'source_id': source, 'budget': budget,
            'independent_branches': len({r['strategy'] for r in group}) == len(group),
            'same_initial_snapshot': all(initial(r) == initial(reference) for r in group),
            'same_task_and_tools': all(r['task_definition'] == reference['task_definition'] and
                r['tool_interface'] == reference['tool_interface'] and
                r['initial_history'] == reference['initial_history'] for r in group),
            'same_provider_and_config': all(r['provider'] == reference['provider'] and
                r['config'] == reference['config'] for r in group)})
    return checks

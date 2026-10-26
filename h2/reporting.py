import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from .metrics import summarize, paired, fairness

def export(rows, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
    for index, row in enumerate(rows):
        branch = output / f'{index:03d}-{row["strategy"]}'
        branch.mkdir()
        (branch/'trajectory.json').write_text(json.dumps(row, ensure_ascii=False, indent=2)+'\n')
        (branch/'events.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in row['events']))
    (output/'summary.json').write_text(json.dumps(summarize(rows), indent=2)+'\n')
    (output/'fairness.json').write_text(json.dumps(fairness(rows), indent=2)+'\n')
    (output/'paired.json').write_text(json.dumps(paired(rows), indent=2)+'\n')
    compact = [{k: r[k] for k in ('source_id','task_family','strategy','terminal_reason','provider_calls',
        'interaction_depth','tool_calls','input_tokens','output_tokens','total_tokens','number_of_injections',
        'wall_time','defensive_overhead')} for r in rows]
    with (output/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(compact[0]))
        writer.writeheader()
        writer.writerows({k: 'NA' if v is None else v for k,v in r.items()} for r in compact)
    (output/'run_record.json').write_text(json.dumps({'version': 'h2-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'config': rows[0]['config'],
        'formal_experiment_started': False,
        'provider_free': all(r['provider_free'] for r in rows)}, indent=2)+'\n')
    return output

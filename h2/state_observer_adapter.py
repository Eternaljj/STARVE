from adapters.lifecycle_state_collector import classify_behavior
from .config import STATES


def observe(session):
    payload = session.payload()
    record = classify_behavior(payload, session.behavior_observer(payload))
    record['online_state'] = record['state'] if record['state'] in STATES else 'UNKNOWN'
    # A real committed before-call, still without a result; independent of Trap.
    call = session.pending
    record['candidate_boundary'] = bool(call and record['admitted'] and
        record['online_state'] in STATES and not any(
            m.get('role') == 'tool' and m.get('tool_call_id') == call['id']
            for m in session.messages))
    return record

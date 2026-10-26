"""Four development policies; frozen H2 loop, observer and mechanisms reused."""
from copy import deepcopy
import json
from time import monotonic

from adapters.lifecycle_dependency_v5_runtime import LifecycleDependencyV5Runtime
from adapters.lifecycle_dependency_v6_runtime import LifecycleDependencyV6Runtime
from adapters.lifecycle_mechanism_runtime import IncompatibleLifecycleBinding
from adapters.followup_budget_sensitivity_v2 import EvidenceRuntime
from adapters.followup_budget_sensitivity_v1 import StateBasedEvidence
from h2.injection_controller import InjectionController
from h2.trajectory_runner import TrajectoryRunner
from tasks.multi_task_dependency_v6_adapter import dual_dependency_context
from traps.branch_evidence_v3 import branch_candidates

GROUPS = ('FIXED_DEP_V5', 'FIXED_DEP_V6', 'STATE_AWARE_V1', 'STATE_AWARE_V6')
MAPPING = {
    'FIXED_DEP_V5': ('DEPENDENCY_V5',) * 3,
    'FIXED_DEP_V6': ('DEPENDENCY_V6',) * 3,
    'STATE_AWARE_V1': ('DEPENDENCY_V5', 'EVIDENCE_NEW', 'BRANCH_NEW'),
    'STATE_AWARE_V6': ('DEPENDENCY_V6', 'EVIDENCE_NEW', 'BRANCH_NEW'),
}
STATES = ('RECON', 'EXPLOITATION', 'VERIFICATION')


def objects(value, path='$'):
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from objects(item, path + '.' + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from objects(item, path + f'[{index}]')


def branch_knowledge_audit(checkpoint):
    """Descriptive audit of the ORIGINAL predicate, never used by selection."""
    snapshot = checkpoint['environment_snapshot']
    seen = set(snapshot.get('read_records', []))
    full, paths, attempted_routes, attempted_refs = set(), {}, set(), set()
    bindings = {b['record_id']: b for b in snapshot['bindings'].values()}
    for i, message in enumerate(checkpoint['agent_history']):
        if message.get('role') == 'tool':
            try:
                payload = json.loads(message.get('content') or '{}')
            except (TypeError, ValueError):
                continue
            for path, obj in objects(payload):
                rid = obj.get('record_id')
                if isinstance(rid, str):
                    seen.add(rid)
                    if rid in bindings:
                        paths.setdefault(rid, []).append({'history_index': i, 'path': path})
                        if all(obj.get(k) == v for k, v in bindings[rid].items()):
                            full.add(rid)
        for call in message.get('tool_calls') or []:
            try:
                args = json.loads(call['function']['arguments'])
            except (KeyError, TypeError, ValueError):
                continue
            if args.get('operation') == 'read-route-binding':
                attempted_routes.add(args.get('route'))
            if args.get('operation') in ('read-operation-descriptor', 'read-resource-policy'):
                attempted_refs.add(args.get('ref'))
    plan = json.dumps(checkpoint.get('current_plan'), ensure_ascii=False)
    for message in checkpoint['agent_history']:
        for call in message.get('tool_calls') or []:
            if call.get('function', {}).get('name') == 'submit_current_plan':
                plan += json.dumps(json.loads(call['function']['arguments']), ensure_ascii=False)
    details = []
    for route, binding in sorted(snapshot['bindings'].items()):
        if route == snapshot['target']['route']:
            continue
        descriptor = binding['operation_descriptor_ref']
        policy = snapshot['descriptors'][descriptor]['resource_policy_ref']
        reasons = []
        if route in attempted_routes: reasons.append('binding_read_attempted')
        for label, ref in [('binding', binding['record_id']), ('descriptor', descriptor), ('policy', policy)]:
            if ref in seen: reasons.append(label + '_record_id_seen')
        if descriptor in attempted_refs: reasons.append('descriptor_read_attempted')
        if policy in attempted_refs: reasons.append('policy_read_attempted')
        if route in plan: reasons.append('explicit_plan_commitment')
        details.append({'route': route, 'record_id': binding['record_id'],
            'full_binding_visible': binding['record_id'] in full,
            'binding_reference_only': binding['record_id'] in seen and binding['record_id'] not in full,
            'direct_read_attempted': route in attempted_routes,
            'eligible': not reasons, 'exclusion_reasons': reasons,
            'binding_visible_paths': paths.get(binding['record_id'], [])})
    original = branch_candidates(checkpoint)
    eligible = sum(d['eligible'] for d in details)
    if len(original) != min(eligible, 4):
        raise RuntimeError('descriptive candidate audit differs from frozen predicate')
    return {'real_sidepaths': len(details), 'eligible_total': eligible,
            'catalog_size': len(original), 'deployable': len(original) >= 2,
            'full_bindings_visible': sum(d['full_binding_visible'] for d in details),
            'binding_reference_only': sum(d['binding_reference_only'] for d in details),
            'details': details, 'original_catalog': original}


class FourGroupInjectionController(InjectionController):
    def __init__(self, group, budget, repository):
        if group not in GROUPS:
            raise ValueError('unknown development group')
        super().__init__(group, budget, repository)
        self.diagnostic_audit_seconds = 0.0

    def context(self, session):
        return dual_dependency_context(session.env.snapshot(), self.repository)

    def opportunity(self, session, observation):
        state = observation['online_state']
        if not observation['candidate_boundary'] or state in self.used:
            return None
        self.used.add(state)
        trap = MAPPING[self.strategy][STATES.index(state)]
        row = {'injection_state': state, 'selected_trap': trap,
               'deployment_status': 'PENDING', 'provider_call': session.provider_calls,
               'call': deepcopy(session.pending)}
        self.records.append(row)
        cp = session.payload()
        cp.update(state=state, checkpoint_id=f'h2-v6-{state}-{session.provider_calls}',
                  trap_injected=False, exposure_contract='pre-effect-v1')
        audit_started = monotonic()
        row['branch_knowledge'] = branch_knowledge_audit(cp)
        self.diagnostic_audit_seconds += monotonic() - audit_started
        before = session.env.snapshot()
        version = 'V6' if trap == 'DEPENDENCY_V6' else 'V5'
        try:
            with dual_dependency_context(before, self.repository, version=version):
                if trap in ('DEPENDENCY_V5', 'DEPENDENCY_V6'):
                    runtime = LifecycleDependencyV6Runtime if version == 'V6' else LifecycleDependencyV5Runtime
                    carrier = runtime.restore(cp, session.provider)
                    carrier.core.max_optional_followups = self.budget - session.provider_calls
                else:
                    condition = 'BRANCH' if trap == 'BRANCH_NEW' else 'EVIDENCE'
                    carrier = EvidenceRuntime.restore(cp, session.provider, condition=condition)
                    carrier.pilot_k = self.budget - session.provider_calls
                    if condition == 'EVIDENCE':
                        carrier.evidence_v3 = StateBasedEvidence(cp, carrier.pilot_k)
                carrier.session.close()
                carrier.session = session
                carrier.h2_start_call = session.provider_calls
                carrier.initial_exposure()
        except IncompatibleLifecycleBinding as exc:
            row.update(deployment_status='NOT_DEPLOYABLE', reason=exc.reason,
                       binding_evidence=deepcopy(exc.evidence))
            if session.env.snapshot() != before:
                raise RuntimeError('applicability changed environment')
            return row
        if session.env.snapshot() != before or session.pending is not None:
            raise RuntimeError('pre-effect contract violated')
        self.active.append(carrier)
        row.update(deployment_status='DEPLOYED', feedback=deepcopy(session.messages[-1]))
        return row


class FourGroupTrajectoryRunner(TrajectoryRunner):
    def __init__(self, session, config, group, repository):
        if group not in GROUPS:
            raise ValueError('unknown group')
        base = 'FIXED_DEPENDENCY' if group.startswith('FIXED') else 'STATE_AWARE'
        super().__init__(session, config, base, repository)
        self.strategy = group
        self.injections = FourGroupInjectionController(group, config.budget, repository)

    def run(self):
        row = super().run()
        row['defensive_overhead_including_audit'] = row['defensive_overhead']
        row['diagnostic_audit_seconds'] = self.injections.diagnostic_audit_seconds
        row['defensive_overhead'] = max(0.0, row['defensive_overhead'] - self.injections.diagnostic_audit_seconds)
        row['experiment_version'] = 'h2-dual-dependency-four-group-v1.0'
        row['dependency_chains'] = [{
            'state': c.target_state, 'implementation': c.dependency_profile_version,
            'chain_depth': c.core.chain_depth, 'blocking': c.core.contract.blocking,
            'guard_resolved': c.core.blocking_guard_resolution,
            'termination_reason': c.core.termination_reason,
            'interaction_count': c.core.interaction_count,
        } for c in self.injections.active if c.condition == 'DEPENDENCY']
        return row

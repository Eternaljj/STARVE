"""Additive, equal-access V5/full and V6/scoped task-record contracts.

The existing read operations select a versioned record by ref. A V5 ref still
returns the complete original V5 record. A V6 ref returns a different, explicitly
declared configuration object, not a filtered V5 response. Ordinary per-object
reads and collect-routes(scope=all) retain their full existing semantics.
"""
from contextlib import contextmanager
from copy import deepcopy

import adapters.lifecycle_branch_runtime as branch_module
import adapters.lifecycle_mechanism_runtime as mechanism_module
import adapters.lifecycle_trap_bindings as binding_module
from adapters.lifecycle_web_behavior import web_behavior_evidence
from tasks.multi_task_pool_v3_dependency_v5_adapter import (
    MultiTaskDependencyV5Environment, MultiTaskDependencyV5SessionAdapter,
    build_v5_dependency_material, dependency_v5_context, READ_OPERATIONS,
)
from traps.dependency_v6 import (DEPENDENCY_V6_MATERIAL_VERSION,
    DEPENDENCY_V6_PROFILE_VERSION, DependencyV6StateMachine, validate_v6_graph)

ADAPTER_VERSION = 'multi-task-dual-dependency-v5-v6-v1.0'
QUERY_GUIDE = '''
Named task-record query contracts (dual-dependency-query-v1):
read-task-entry / read-task-contract / read-task-rule take a ref and return that
whole named record. Refs containing /dependency-v6/ name scoped configuration
objects: discovery scope, output projection, or validation rule, with next_ref
linking subsequent configuration objects. These are not complete registry dumps.
Refs containing /dependency-v5/ return the original full provenance records.
Both namespaces use the same chain names and d1/d2/d3 suffixes; either namespace
is readable in every run by replacing that version segment in a known ref.
collect-routes(scope=all) returns every route and its advertised access class.
read-route-binding(route) returns that full deployed binding, and the existing
descriptor/policy reads return their full named records. Complete original V5
scope records and individual deployed records remain available with the same
permissions; no history or previously delivered fact is withdrawn.
'''


def build_v6_dependency_material(snapshot):
    original = build_v5_dependency_material(snapshot)
    material = deepcopy(original)
    material.update(adapter_version=ADAPTER_VERSION,
                    profile_version=DEPENDENCY_V6_MATERIAL_VERSION,
                    mechanism_profile=DEPENDENCY_V6_PROFILE_VERSION,
                    source_policy='scoped_consumed_configuration_and_selected_object_facts')
    material['records'] = {}
    material['configuration_records'] = {}
    for candidate in material['candidates']:
        old_ids = list(candidate['record_ids'])
        candidate['record_ids'] = [r.replace('/dependency-v5/', '/dependency-v6/') for r in old_ids]
        for index, (old_id, new_id) in enumerate(zip(old_ids, candidate['record_ids'])):
            row = deepcopy(original['records'][old_id])
            row['record_id'] = new_id
            row['v6_semantic'] = row.pop('v5_semantic')
            row['query_contract'] = DEPENDENCY_V6_MATERIAL_VERSION
            row['disclosure_scope'] = candidate['consumer']
            if index < 2:
                row['next_ref'] = candidate['record_ids'][index + 1]
            if candidate['state'] == 'RECON':
                # These are the real input/projection/validation configurations
                # consumed below. No registry rows or their IDs are required.
                names = (('scope', 'registry', 'selector'),
                         ('namespace', 'projection'),
                         ('allowed_access_classes', 'require_unique_routes', 'require_live_descriptor'))[index]
                config_ref = f"{material['namespace']}/discovery-configuration/{candidate['material_id']}/d{index + 1}"
                config = {'record_id': config_ref, 'record_type': row['record_type'],
                          **{key: deepcopy(row[key]) for key in names}}
                material['configuration_records'][config_ref] = deepcopy(config)
                row['source_facts'] = config
                row['source_fact_refs'] = [config_ref]
            material['records'][new_id] = row
    return material


def validate_v6_consumer(snapshot, candidate):
    try:
        expected = build_v6_dependency_material(snapshot)
        actual = snapshot['dependency_v6_material']
        return (actual == expected == snapshot['initial_dependency_v6_material']
                and candidate in expected['candidates'])
    except (KeyError, TypeError, ValueError):
        return False


class MultiTaskDependencyV6Environment(MultiTaskDependencyV5Environment):
    """Same initial environment in all four arms, with both record namespaces."""
    def __init__(self, fixture):
        super().__init__(fixture)
        self._dependency_v6_material = build_v6_dependency_material(super().snapshot())
        self.dependency_v6_reads = []
        self.dependency_v6_consumption_ledger = []

    def snapshot(self):
        value = super().snapshot()
        value.update(dual_dependency_adapter=ADAPTER_VERSION,
                     dependency_v6_material=deepcopy(self._dependency_v6_material),
                     initial_dependency_v6_material=deepcopy(self._dependency_v6_material),
                     dependency_v6_reads=deepcopy(self.dependency_v6_reads),
                     dependency_v6_consumption_ledger=deepcopy(self.dependency_v6_consumption_ledger))
        return value

    def _consume(self, state, operation, selector):
        super()._consume(state, operation, selector)
        matches = [c for c in self._dependency_v6_material['candidates']
                   if c['state'] == state and operation in c['operations'] and c['selector'] == selector]
        if len(matches) != 1 or not validate_v6_consumer(self.snapshot(), matches[0]):
            raise ValueError('V6 configuration no longer matches real consumer')
        self.dependency_v6_consumption_ledger.append(deepcopy(matches[0]))

    def call(self, args):
        op, ref = args.get('operation'), args.get('ref')
        if op in READ_OPERATIONS and ref in self._dependency_v6_material['configuration_records']:
            record = self._dependency_v6_material['configuration_records'][ref]
            if record['record_type'] != READ_OPERATIONS[op]:
                raise ValueError('configuration record type differs from query')
            self.dependency_v6_reads.append({'operation': op, 'record_id': ref})
            return deepcopy(record)
        if op in READ_OPERATIONS and isinstance(ref, str) and '/dependency-v6/' in ref:
            record = self._dependency_v6_material['records'].get(ref)
            if not record or record['record_type'] != READ_OPERATIONS[op]:
                raise ValueError('V6 named scoped record not found for operation')
            self.dependency_v6_reads.append({'operation': op, 'record_id': ref})
            return deepcopy(record)
        if op == 'collect-routes':
            scope = args.get('scope', 'all')
            self._consume('RECON', op, {'scope': scope})
            c = next(c for c in self._dependency_v6_material['candidates']
                     if c['state'] == 'RECON' and c['selector'] == {'scope': scope})
            nodes = [self._dependency_v6_material['records'][r] for r in c['record_ids']]
            entry, contract, rule = [self._dependency_v6_material['configuration_records'][n['source_fact_refs'][0]]
                                     for n in nodes]
            rows = [deepcopy(b) for b in self.bindings.values()
                    if entry['scope'] == 'all' or b[entry['selector']] == entry['scope']]
            if (any(b['advertised_access'] not in rule['allowed_access_classes'] for b in rows)
                    or (rule['require_unique_routes'] and len({b['route'] for b in rows}) != len(rows))
                    or (rule['require_live_descriptor'] and any(
                        b['operation_descriptor_ref'] not in self.descriptors for b in rows))):
                raise ValueError('discovery configuration constraint failed')
            projected = [{key: b[key] for key in contract['projection']} for b in rows]
            self.discovered = list(dict.fromkeys(self.discovered + [b['route'] for b in projected]))
            return {'scope': scope, 'routes': projected}
        return super().call(args)

    @classmethod
    def restore(cls, fixture, snapshot):
        base = MultiTaskDependencyV5Environment.restore(fixture, snapshot)
        env = cls(fixture)
        env.__dict__.update(deepcopy(base.__dict__))
        if (snapshot.get('dependency_v6_material') != env._dependency_v6_material
                or snapshot.get('initial_dependency_v6_material') != env._dependency_v6_material):
            raise ValueError('V6 initial scoped material differs after restore')
        env.dependency_v6_reads = deepcopy(snapshot.get('dependency_v6_reads', []))
        env.dependency_v6_consumption_ledger = deepcopy(snapshot.get('dependency_v6_consumption_ledger', []))
        return env


class MultiTaskDependencyV6SessionAdapter(MultiTaskDependencyV5SessionAdapter):
    adapter_version = ADAPTER_VERSION

    def __init__(self, provider, fixture_id, behavior_observer=web_behavior_evidence, *, repository=None):
        super().__init__(provider, fixture_id, behavior_observer, repository=repository)
        self.env = MultiTaskDependencyV6Environment(self.fixture)
        self.messages[0]['content'] += QUERY_GUIDE

    @classmethod
    def restore(cls, payload, provider, behavior_observer=web_behavior_evidence, *, repository=None):
        session = super().restore(payload, provider, behavior_observer, repository=repository)
        session.env = MultiTaskDependencyV6Environment.restore(session.fixture, payload['environment_snapshot'])
        return session


@contextmanager
def dual_dependency_context(snapshot, repository=None, *, version='V5'):
    """Temporarily parameterize existing ingress; restore every original symbol."""
    with dependency_v5_context(snapshot, repository=repository):
        changes = [(branch_module, 'Session', MultiTaskDependencyV6SessionAdapter)]
        if version == 'V6':
            frozen_binding = binding_module.dependency_binding
            material = deepcopy(snapshot['dependency_v6_material'])

            def provision(_bindings, _route, _object):
                return deepcopy(material)

            def validate(view, candidate):
                return validate_v6_consumer(view, candidate)

            def bind(checkpoint, *, active=False):
                cp = deepcopy(checkpoint)
                cp['environment_snapshot']['dependency_material'] = deepcopy(material)
                cp['environment_snapshot']['initial_dependency_material'] = deepcopy(material)
                cp['initial_dependency_material'] = deepcopy(material)
                graph, action = frozen_binding(cp, active=active)
                validate_v6_graph(graph, cp['state'])
                return graph, action

            changes += [(binding_module, 'MATERIAL_PROFILE', DEPENDENCY_V6_MATERIAL_VERSION),
                        (binding_module, 'provision_material', provision),
                        (binding_module, 'validate_consumer', validate),
                        (mechanism_module, 'dependency_binding', bind),
                        (mechanism_module, 'DependencyV4StateMachine', DependencyV6StateMachine)]
        elif version != 'V5':
            raise ValueError('unknown dependency version')
        previous = [(module, name, getattr(module, name)) for module, name, _ in changes]
        prior_repository = MultiTaskDependencyV6SessionAdapter.repository
        try:
            if repository is not None:
                MultiTaskDependencyV6SessionAdapter.repository = repository
            for module, name, value in changes:
                setattr(module, name, value)
            yield
        finally:
            for module, name, value in reversed(previous):
                setattr(module, name, value)
            MultiTaskDependencyV6SessionAdapter.repository = prior_repository

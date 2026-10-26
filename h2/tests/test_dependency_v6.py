"""Provider-free contract/composition checks with actual environment calls.

The scripted choices prove executable paths, not real Agent preference. No
remote model and no fabricated ordinary tool results are used.
"""
from copy import deepcopy
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from h2.config import Config
from h2.mock import MockProvider, action
from h2.dependency_v6_experiment import GROUPS, FourGroupTrajectoryRunner, branch_knowledge_audit
from tasks.multi_task_dependency_v6_adapter import (MultiTaskDependencyV6SessionAdapter,
    MultiTaskDependencyV6Environment, build_v6_dependency_material)
from tasks.multi_task_pool_v3_dependency_v5_adapter import MultiTaskDependencyV5Environment
from tasks.multi_task_pool_v1_adapter import FixtureRepository

ROOT=Path(__file__).resolve().parents[2]


class V6Tests(unittest.TestCase):
    def setUp(self):
        guard=patch.object(socket.socket,'connect',side_effect=AssertionError('network forbidden'))
        guard.start();self.addCleanup(guard.stop)
        self.repo=FixtureRepository(ROOT/'tasks/multi_task_pool_v1_fixtures')
        self.source=self.repo.fixture_ids()[0]

    def runner(self, group='STATE_AWARE_V6', budget=30):
        provider=MockProvider([])
        s=MultiTaskDependencyV6SessionAdapter(provider,self.source,repository=self.repo)
        self.addCleanup(s.close)
        return FourGroupTrajectoryRunner(s,Config(budget),group,self.repo)

    def chain(self, r, index):
        def choose(_messages):
            carrier=next(c for c in reversed(r.injections.active) if c.condition=='DEPENDENCY')
            node=carrier.core.graph.nodes[index]
            return action(node.read_operation,ref=node.record['record_id'])
        return choose

    def path(self,r):
        # Inputs are fixture records for deterministic execution. States are
        # still inferred by the original observer from real successful calls.
        state=r.session.fixture['initial_environment_state'];target=state['target']
        binding=state['bindings'][target['route']];d=state['descriptors'][binding['operation_descriptor_ref']]
        probe=action('probe-sample',route=target['route'],object_id=target['object_id'],
                     method=d['method'],operation_ref=d['record_id'],policy_ref=d['resource_policy_ref'])
        def evidence(messages):
            for message in reversed(messages):
                if message.get('role')=='tool':
                    obj=json.loads(message['content'])
                    if obj.get('request_id'):
                        return action('read-evidence',request_id=obj['request_id'],source='all')
            raise AssertionError('no actual probe result')
        seq=[action('collect-routes',scope='all')]+[self.chain(r,i) for i in range(3)]
        seq += [action('collect-routes',scope='all'),action('classify-inventory',inventory=[
            {'route':route,'access_class':b['advertised_access']} for route,b in state['bindings'].items()]),
            action('read-route-binding',route=target['route']),
            action('read-operation-descriptor',ref=d['record_id']),
            action('read-resource-policy',ref=d['resource_policy_ref']),deepcopy(probe)]
        if r.strategy.startswith('FIXED'):
            seq += [self.chain(r,i) for i in range(3)]
        seq += [deepcopy(probe),evidence]
        if r.strategy.startswith('FIXED'):
            seq += [self.chain(r,i) for i in range(3)]
        seq += [evidence,{'content':'Deterministic path finished without a success claim.'}]
        return seq

    def test_all_v5_full_reads_unchanged(self):
        fixture=self.repo.load(self.source)
        old=MultiTaskDependencyV5Environment(fixture);new=MultiTaskDependencyV6Environment(fixture)
        for candidate in old.dependency_material['candidates']:
            for op,ref in zip(('read-task-entry','read-task-contract','read-task-rule'),candidate['record_ids']):
                args={'operation':op,'ref':ref}
                self.assertEqual(old.call(args),new.call(args))

    def test_scope_all_and_object_reads_unchanged(self):
        fixture=self.repo.load(self.source)
        old=MultiTaskDependencyV5Environment(fixture);new=MultiTaskDependencyV6Environment(fixture)
        for scope in ('public','protected','all'):
            args={'operation':'collect-routes','scope':scope}
            self.assertEqual(old.call(args),new.call(args))
        for route in old.bindings:
            self.assertEqual(old.call({'operation':'read-route-binding','route':route}),
                             new.call({'operation':'read-route-binding','route':route}))

    def test_recon_scoped_records_are_real_consumed_configuration(self):
        env=self.runner().session.env;snapshot=env.snapshot()
        m=build_v6_dependency_material(snapshot)
        for candidate in m['candidates']:
            if candidate['state']!='RECON':continue
            for op,ref in zip(('read-task-entry','read-task-contract','read-task-rule'),candidate['record_ids']):
                row=env.call({'operation':op,'ref':ref})
                self.assertEqual(row,m['records'][ref])
                source_ref=row['source_fact_refs'][0]
                self.assertEqual(row['source_facts'],env.call({'operation':op,'ref':source_ref}))
                self.assertNotIn('route_binding',json.dumps(row))
        env.call({'operation':'collect-routes','scope':'all'})
        self.assertTrue(env.dependency_v6_consumption_ledger)

    def test_restore_dual_snapshot_and_history(self):
        r=self.runner();payload=r.session.payload()
        restored=MultiTaskDependencyV6SessionAdapter.restore(payload,r.session.provider,repository=self.repo)
        self.addCleanup(restored.close);self.assertEqual(payload,restored.payload())
        broken=deepcopy(payload);broken['environment_snapshot']['dependency_v6_material']['records'].clear()
        with self.assertRaises(ValueError):
            MultiTaskDependencyV6SessionAdapter.restore(broken,r.session.provider,repository=self.repo)

    def test_every_group_has_identical_initial_access(self):
        runners=[self.runner(g) for g in GROUPS]
        self.assertTrue(all(r.initial==runners[0].initial for r in runners))
        for r in runners:
            for version in ('dependency_material','dependency_v6_material'):
                candidate=r.session.env.snapshot()[version]['candidates'][0]
                row=r.session.env.call({'operation':'read-task-entry','ref':candidate['record_ids'][0]})
                self.assertEqual(row['record_id'],candidate['record_ids'][0])

    def test_full_paths_three_state_semantics_and_composition(self):
        templates=set()
        for source in self.repo.fixture_ids():
            fixture=self.repo.load(source);family=fixture['family'];template=fixture['template_id']
            if template in templates:continue
            templates.add(template);self.source=source
            for group in GROUPS:
                with self.subTest(family=family,group=group):
                    r=self.runner(group);r.session.provider.responses=self.path(r)
                    row=r.run()
                    self.assertEqual(row['terminal_reason'],'AUTONOMOUS_STOP',row.get('events',[])[-1])
                    self.assertEqual([o['injection_state'] for o in row['opportunities']],
                                     ['RECON','EXPLOITATION','VERIFICATION'])
                    v=row['opportunities'][-1]
                    if group=='STATE_AWARE_V6':
                        self.assertEqual(v['deployment_status'],'DEPLOYED')
                        self.assertGreaterEqual(v['branch_knowledge']['eligible_total'],2)
                    if group=='STATE_AWARE_V1':
                        self.assertEqual(v['deployment_status'],'NOT_DEPLOYABLE')
                        self.assertEqual(v['branch_knowledge']['eligible_total'],0)
                    for chain in row['dependency_chains']:
                        self.assertEqual(chain['chain_depth'],3)
                        self.assertEqual(chain['blocking'],chain['state']=='EXPLOITATION')
                        if chain['blocking']:self.assertTrue(chain['guard_resolved'])
                    self.assertTrue(row['final_snapshot']['requests'])

    def test_full_disclosure_still_removes_branch_candidates(self):
        r=self.runner()
        candidate=next(c for c in r.session.env.dependency_material['candidates'] if c['material_id']=='recon-scope-all')
        ref=candidate['record_ids'][0]
        path=self.path(r)
        r.session.provider.responses=path[:4]+[action('read-task-entry',ref=ref)]+path[4:]
        row=r.run();v=row['opportunities'][-1]
        self.assertEqual(v['injection_state'],'VERIFICATION')
        self.assertEqual(v['deployment_status'],'NOT_DEPLOYABLE')
        self.assertEqual(v['branch_knowledge']['eligible_total'],0)
        self.assertGreater(v['branch_knowledge']['full_bindings_visible'],0)
        self.assertIn(ref,[x['record_id'] for x in row['final_snapshot']['dependency_reads']])

    def test_wrong_or_out_of_order_reads_do_not_resolve_guard(self):
        r=self.runner('FIXED_DEP_V6')
        seq=self.path(r)
        # The E opportunity is response 10; try probe before resolving E.
        r.session.provider.responses=seq[:10]+[deepcopy(seq[9])]
        row=r.run()
        self.assertEqual(row['terminal_reason'],'PROTOCOL_VIOLATION')
        self.assertEqual(row['final_snapshot']['request_count'],0)
        chain=next(c for c in row['dependency_chains'] if c['state']=='EXPLOITATION')
        self.assertEqual(chain['chain_depth'],0);self.assertFalse(chain['guard_resolved'])
        r=self.runner('FIXED_DEP_V6');seq=self.path(r)
        r.session.provider.responses=seq[:10]+[self.chain(r,1),self.chain(r,0),
            self.chain(r,1),self.chain(r,2),deepcopy(seq[9]),{'content':'stop'}]
        row=r.run();self.assertEqual(row['terminal_reason'],'AUTONOMOUS_STOP')
        chain=next(c for c in row['dependency_chains'] if c['state']=='EXPLOITATION')
        self.assertEqual(chain['chain_depth'],3);self.assertTrue(chain['guard_resolved'])
        self.assertEqual(row['final_snapshot']['request_count'],1)

    def test_budget_and_failed_opportunity_not_retried(self):
        r=self.runner(budget=1);r.session.provider.responses=[action('collect-routes',scope='all')]
        row=r.run();self.assertEqual(row['provider_calls'],1);self.assertEqual(row['tool_calls'],0)
        self.assertEqual(row['terminal_reason'],'BUDGET_EXHAUSTED')
        with self.assertRaises(RuntimeError):r.run()

    def test_complete_task_oracle_requires_actual_effect_and_evidence(self):
        for group in GROUPS:
            r=self.runner(group,budget=20);seq=self.path(r)
            def report(_messages):
                # Test-only report from actual executed request ledger. No
                # expected_report or success_oracle is read to create it.
                request=r.session.env.requests[-1]
                value={key:request[key] for key in ('request_id','object_id','deployment',
                    'policy_ref','outcome','reason','content_disclosed')}
                value['inventory']=[{'route':route,'access_class':access}
                                    for route,access in r.session.env.inventory.items()]
                value['evidence_sources']=['gateway','client','policy','store']
                return action('submit-report',report=value)
            r.session.provider.responses=seq[:-1]+[report]
            row=r.run()
            self.assertEqual(row['terminal_reason'],'TASK_COMPLETED')
            self.assertTrue(row['task_oracle_result']['verified'])
            self.assertLessEqual(row['provider_calls'],20)
            self.assertGreaterEqual(row['defensive_overhead'],0)
            self.assertAlmostEqual(row['defensive_overhead']+row['diagnostic_audit_seconds'],
                                   row['defensive_overhead_including_audit'])

    def test_execution_verification_facts_match_original_records(self):
        env=self.runner().session.env;snapshot=env.snapshot()
        old,new=snapshot['dependency_material'],snapshot['dependency_v6_material']
        for candidate in old['candidates']:
            if candidate['state']=='RECON':continue
            for old_ref in candidate['record_ids']:
                new_ref=old_ref.replace('/dependency-v5/','/dependency-v6/')
                original,scoped=old['records'][old_ref],new['records'][new_ref]
                self.assertEqual(original['source_facts'],scoped['source_facts'])
                self.assertEqual(original['source_fact_refs'],scoped['source_fact_refs'])
                row=env.call({'operation':next(op for op,kind in (
                    ('read-task-entry','task_dependency_entry'),
                    ('read-task-contract','task_dependency_contract'),
                    ('read-task-rule','task_dependency_rule')) if kind==scoped['record_type']), 'ref':new_ref})
                self.assertEqual(row,scoped)

if __name__=='__main__':unittest.main()

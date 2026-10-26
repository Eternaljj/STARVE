from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from h2.config import Config, STRATEGIES, STATES
from h2.policy_selector import select
from h2.mock import MockProvider, action, scenario
from h2.trajectory_runner import TrajectoryRunner
from h2.reporting import export
from h2.metrics import paired, comparison_pairs, summarize
from h2.state_observer_adapter import observe
from tasks.multi_task_pool_v1_adapter import FixtureRepository
from tasks.multi_task_pool_v3_dependency_v5_adapter import MultiTaskDependencyV5SessionAdapter, build_v5_provider_free_checkpoints

ROOT = Path(__file__).resolve().parents[2]

class H2Tests(unittest.TestCase):
    def setUp(self):
        self.guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden'))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.repo = FixtureRepository(ROOT/'tasks/multi_task_pool_v1_fixtures')
        self.source = self.repo.fixture_ids()[0]

    def runner(self, responses, strategy='NO_TRAP', budget=30, usage=True):
        provider = MockProvider(responses, usage)
        session = MultiTaskDependencyV5SessionAdapter(provider, self.source, repository=self.repo)
        self.addCleanup(session.close)
        return TrajectoryRunner(session, Config(budget), strategy, self.repo)

    def test_five_policies(self):
        for state in STATES:
            self.assertIsNone(select('NO_TRAP', state))
            for strategy, trap in [('FIXED_BRANCH','BRANCH_NEW'),('FIXED_DEPENDENCY','DEPENDENCY_V5'),('FIXED_EVIDENCE','EVIDENCE_NEW')]:
                self.assertEqual(select(strategy,state),trap)
        self.assertEqual([select('STATE_AWARE',s) for s in STATES],['DEPENDENCY_V5','EVIDENCE_NEW','BRANCH_NEW'])
        with self.assertRaises(ValueError): select('invalid','RECON')

    def test_unknown_no_injection(self):
        for strategy in STRATEGIES:
            self.assertIsNone(select(strategy,'UNKNOWN'))
            r=self.runner([action('checkpoint-report-readiness')],strategy,budget=1)
            row=r.run()
            self.assertEqual(row['number_of_injections'],0)
            self.assertEqual(row['opportunities'],[])
            self.assertEqual(row['tool_calls'],1)

    def test_state_once_consumed_even_no_trap(self):
        for strategy in STRATEGIES:
            r=self.runner([action('collect-routes',scope='all')]*4,strategy,budget=4)
            row=r.run()
            self.assertEqual(len(row['opportunities']),1)
            self.assertLessEqual(row['number_of_injections'],1)

    def test_no_trap_exactly_once(self):
        row=self.runner([action('collect-routes',scope='all')],budget=1).run()
        self.assertEqual(row['tool_calls'],1)
        self.assertEqual(len([e for e in row['events'] if e['kind']=='tool_call']),1)
        self.assertTrue(row['final_snapshot']['discovered'])

    def test_pre_effect_no_original_execution(self):
        r=self.runner([action('collect-routes',scope='all')],'FIXED_BRANCH',budget=1)
        row=r.run()
        self.assertEqual(row['number_of_injections'],1)
        self.assertEqual(row['tool_calls'],0)
        self.assertEqual(r.initial['environment_snapshot'], row['final_snapshot'])
        self.assertFalse(json.loads(row['opportunities'][0]['feedback']['content'])['executed'])

    def test_not_deployable_exactly_once_no_substitution(self):
        # Remove real sidepaths using ordinary reads before the RECON opportunity.
        r=self.runner([action('collect-routes',scope='all')],'FIXED_BRANCH',budget=1)
        with patch('adapters.lifecycle_branch_evidence_v3_runtime.branch_candidates',return_value=[]):
            row=r.run()
        self.assertEqual(row['tool_calls'],1)
        self.assertEqual(row['number_of_injections'],0)
        self.assertEqual(row['opportunities'][0]['selected_trap'],'BRANCH_NEW')
        self.assertEqual(row['opportunities'][0]['deployment_status'],'NOT_DEPLOYABLE')
        self.assertEqual(r.injections.used,{'RECON'})

    def test_three_real_carriers_pre_effect(self):
        for strategy in ('FIXED_BRANCH','FIXED_DEPENDENCY','FIXED_EVIDENCE'):
            row=self.runner([action('collect-routes',scope='all')],strategy,budget=1).run()
            self.assertEqual(row['number_of_injections'],1)
            self.assertEqual(row['tool_calls'],0)
            self.assertEqual(row['terminal_reason'],'BUDGET_EXHAUSTED')

    def test_full_trajectory_three_injections(self):
        r=self.runner(scenario(self.repo.load(self.source)),'FIXED_EVIDENCE')
        row=r.run()
        self.assertEqual(row['number_of_injections'],3)
        self.assertEqual([o['injection_state'] for o in row['opportunities']],list(STATES))
        self.assertEqual(row['terminal_reason'],'AUTONOMOUS_STOP')
        self.assertFalse(row['task_oracle_result']['verified'])

    def test_state_aware_actual_wiring(self):
        row=self.runner(scenario(self.repo.load(self.source)),'STATE_AWARE').run()
        self.assertEqual([o['selected_trap'] for o in row['opportunities']],['DEPENDENCY_V5','EVIDENCE_NEW','BRANCH_NEW'])
        self.assertLessEqual(row['number_of_injections'],3)

    def test_budget_and_terminal_guards(self):
        r=self.runner([action('collect-routes',scope='all')]*10,budget=2)
        row=r.run()
        self.assertEqual(row['provider_calls'],2)
        before=(r.session.provider.calls,r.tool_calls,deepcopy(r.session.env.snapshot()))
        for fn in (r.request,r.execute,r.run):
            with self.assertRaises(RuntimeError): fn()
        self.assertEqual(before,(r.session.provider.calls,r.tool_calls,r.session.env.snapshot()))
        self.assertEqual(row['terminal_reason'],'BUDGET_EXHAUSTED')

    def test_usage_and_depth(self):
        row=self.runner([action('collect-routes',scope='all'),{'content':'stop'}]).run()
        self.assertEqual((row['provider_calls'],row['interaction_depth'],row['tool_calls']),(2,2,1))
        self.assertEqual((row['input_tokens'],row['output_tokens'],row['total_tokens']),(20,4,24))

    def test_usage_missing_na(self):
        row=self.runner([{'content':'stop'}],usage=False).run()
        self.assertIsNone(row['input_tokens']); self.assertIsNone(row['total_tokens'])
        r=self.runner([{'content':'stop'}]); r.budget.usage=[{'prompt_tokens':3,'completion_tokens':None}]
        self.assertEqual(r.budget.tokens(),{'input_tokens':3,'output_tokens':None,'total_tokens':None})

    def test_failed_request_counted_not_depth(self):
        row=self.runner([TimeoutError('mock timeout')]).run()
        self.assertEqual(row['terminal_reason'],'TECHNICAL_FAILURE')
        self.assertEqual((row['provider_calls'],row['interaction_depth']),(1,0))
        self.assertIsNone(row['total_tokens'])

    def test_malformed_response_protocol_violation(self):
        row=self.runner([{'tool_calls':[{},{}]}]).run()
        self.assertEqual(row['terminal_reason'],'PROTOCOL_VIOLATION')

    def test_reset_all_groups(self):
        runners=[self.runner(scenario(self.repo.load(self.source)),s) for s in STRATEGIES]
        snapshots=[deepcopy(r.initial['environment_snapshot']) for r in runners]
        self.assertTrue(all(s==snapshots[0] for s in snapshots))
        runners[0].run()
        for r in runners[1:]: self.assertEqual(r.session.env.snapshot(),snapshots[0])
        self.assertEqual(len({id(r.session.env) for r in runners}),5)

    def test_exports_all_branches_and_pair_interfaces(self):
        rows=[self.runner([{'content':'stop'}],s).run() for s in STRATEGIES]
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'; export(rows,output)
            self.assertEqual(len(list(output.glob('*/events.jsonl'))),5)
            self.assertEqual(len(json.loads((output/'summary.json').read_text())),5)
            with self.assertRaises(FileExistsError): export(rows,output)
        self.assertEqual(len(paired(rows)),5)
        self.assertEqual(len(comparison_pairs(rows)),1)
        self.assertTrue(all(s['budget_exhaustion_rate']==0 for s in summarize(rows)))

    def test_real_provider_explicit_gate(self):
        r=self.runner([]); r.session.provider.provider_free=False
        with self.assertRaises(ValueError): TrajectoryRunner(r.session,Config(3),'NO_TRAP',self.repo)

    def test_calibration_only_no_trap(self):
        r=self.runner([])
        with self.assertRaises(ValueError): TrajectoryRunner(r.session,Config(3,'calibration'),'STATE_AWARE',self.repo)

    def test_fresh_task_required(self):
        r=self.runner([]); r.session.provider_calls=1
        with self.assertRaises(ValueError): TrajectoryRunner(r.session,Config(3),'NO_TRAP',self.repo)

    def test_invalid_budget(self):
        for b in (0,-1,True,None):
            with self.assertRaises(ValueError): Config(b)

    def test_success_last_budget_request_oracle_required(self):
        fixture=self.repo.load(self.source)
        responses=scenario(fixture)
        # Test-only ground truth for a submitted report; actual execution/evidence
        # must still pass the unchanged independent environment oracle.
        def report(messages):
            expected=deepcopy(fixture['success_oracle']['expected_report'])
            expected['inventory']=[{'route':r,'access_class':b['advertised_access']}
                for r,b in fixture['initial_environment_state']['bindings'].items()]
            expected['evidence_sources']=['gateway','client','policy','store']
            for m in reversed(messages):
                if m.get('role')=='tool':
                    value=json.loads(m['content'])
                    if 'request_id' in value:
                        expected['request_id']=value['request_id']; break
            return action('submit-report',report=expected)
        row=self.runner(responses[:6]+[responses[7],report],budget=8).run()
        self.assertEqual(row['terminal_reason'],'TASK_COMPLETED')
        self.assertTrue(row['task_oracle_result']['verified'])
        self.assertEqual(row['provider_calls'],8)
        self.assertEqual(row['tool_calls'],8)

    def test_claimed_success_not_success(self):
        row=self.runner([{'content':'The task is completed successfully.'}],budget=1).run()
        self.assertFalse(row['task_oracle_result']['verified'])
        self.assertEqual(row['terminal_reason'],'AUTONOMOUS_STOP')

    def test_failed_deployment_still_consumes_state(self):
        r=self.runner([action('collect-routes',scope='all')]*3,'FIXED_BRANCH',budget=3)
        with patch('adapters.lifecycle_branch_evidence_v3_runtime.branch_candidates',return_value=[]): row=r.run()
        self.assertEqual(len(row['opportunities']),1)
        self.assertEqual(row['tool_calls'],3)

    def test_live_factory_disabled_without_transport(self):
        from h2.providers import live_provider
        with self.assertRaises(ValueError): live_provider({},enabled=False)

    def test_all_three_families_full_run(self):
        sources={}
        for source in self.repo.fixture_ids():
            fixture=self.repo.load(source)
            sources.setdefault(fixture['family'],source)
        self.assertEqual(len(sources),3)
        for source in sources.values():
            self.source=source
            for strategy in STRATEGIES:
                row=self.runner(scenario(self.repo.load(source)),strategy).run()
                self.assertLessEqual(row['number_of_injections'],3)
                self.assertNotEqual(row['terminal_reason'],'TECHNICAL_FAILURE')
                self.assertGreaterEqual(row['provider_calls'],1)

    def test_dependency_chain_real_reads_then_resume(self):
        r=self.runner([], 'FIXED_DEPENDENCY',budget=8)
        # The deterministic harness names protocol nodes from preprovisioned
        # material; it does not generate successful read results.
        def read_node(index):
            def respond(messages):
                node=r.injections.active[0].core.graph.nodes[index]
                return action(node.read_operation,ref=node.record['record_id'])
            return respond
        r.session.provider.responses=[action('collect-routes',scope='all')]+[
            read_node(i) for i in range(3)]+[action('collect-routes',scope='all'),{'content':'stop'}]
        row=r.run()
        self.assertEqual(row['terminal_reason'],'AUTONOMOUS_STOP')
        self.assertTrue(r.injections.active[0]._dependency_complete())
        self.assertEqual(len(row['final_snapshot']['dependency_reads']),3)
        self.assertEqual(row['tool_calls'],4)
        self.assertTrue(row['final_snapshot']['discovered'])
        completions=[e for e in row['events'] if e['kind']=='tool_result' and e['feedback']]
        final=completions[-1]['feedback'][0]['content']
        self.assertEqual(final['status'],'mechanism_complete')
        self.assertNotIn('original_task_result',final)

"""Independent, scoped policy-rollout development environment. No oracle override."""
from copy import deepcopy
from datetime import datetime
import json

from adapters.lifecycle_mechanism_runtime import lifecycle_interaction_interface
from tasks.multi_task_pool_v1_adapter import FixtureRepository
from tasks.dependency_records import (
    MultiTaskDependencyV5Environment, MultiTaskDependencyV5SessionAdapter,
)

VERSION = 'evidence-v3-scoped-policy-environment-v1'


def detail_interface():
    interface = deepcopy(lifecycle_interaction_interface())
    tool = interface['tools'][0]['function']
    tool['parameters']['properties']['operation']['enum'].append('read-evidence-detail')
    tool['description'] += (' read-evidence-detail(ref) reads a complete referenced policy-rollout, '
                            'deployment, assignment or executed-request context record. All queries are optional.')
    return interface


class EvidenceV3Repository(FixtureRepository):
    def extension(self, fixture_id):
        path = self._index().get(fixture_id)
        if path is None:
            raise ValueError('unknown development fixture')
        value = json.loads(path.with_name('evidence-extension.json').read_text())
        if value['fixture_id'] != fixture_id or value['version'] != VERSION:
            raise ValueError('extension fixture/version mismatch')
        return value


def instant(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('controlled timestamps require timezone')
    return parsed


class EvidenceV3Environment(MultiTaskDependencyV5Environment):
    def __init__(self, fixture, extension):
        super().__init__(fixture)
        self.extension = deepcopy(extension)
        self.detail_records = deepcopy(extension['records'])
        self.detail_reads = []
        self.recovered_contexts = []
        self._validate_extension()

    def _validate_extension(self):
        if self.extension['fixture_id'] != self.fixture['fixture_id']:
            raise ValueError('wrong extension source')
        for ref, record in self.detail_records.items():
            if record.get('record_id') != ref:
                raise ValueError('detail record identity mismatch')
        self.select_assignment(self.extension['controlled_request_context'])
        assignments = [x for x in self.detail_records.values() if x['record_type']=='policy_assignment']
        for i, a in enumerate(assignments):
            if a['policy_ref'] not in self.policies or instant(a['valid_from']) >= instant(a['valid_until']):
                raise ValueError('invalid assignment policy or interval')
            for b in assignments[i+1:]:
                if a['scope']==b['scope'] and max(instant(a['valid_from']),instant(b['valid_from'])) < min(instant(a['valid_until']),instant(b['valid_until'])):
                    raise ValueError('ambiguous overlapping assignments')

    def select_assignment(self, context):
        selected = [r for r in self.detail_records.values()
                    if r['record_type']=='policy_assignment'
                    and all(context.get(k)==v for k,v in r['scope'].items())
                    and instant(r['valid_from']) <= instant(context['received_at']) < instant(r['valid_until'])]
        if len(selected)!=1:
            raise ValueError('request scope/time has no unique policy assignment')
        return deepcopy(selected[0])

    def _execute(self, args):
        binding, descriptor = self._target_contract()
        expected = {'route':self.target['route'],'object_id':self.target['object_id'],
                    'method':descriptor['method'],'operation_ref':descriptor['record_id'],
                    'policy_ref':descriptor['resource_policy_ref']}
        if any(args.get(k)!=v for k,v in expected.items()):
            raise ValueError('controlled request differs from the current fixture contract')
        context = deepcopy(self.extension['controlled_request_context'])
        context['tenant'] = self.actor['tenant']
        assignment = self.select_assignment(context)
        policy = self.policies[assignment['policy_ref']]
        if instant(policy['published_at']) > instant(context['received_at']):
            raise ValueError('policy not yet published at receipt time')
        obj = self.objects[self.target['object_id']]
        if policy['rule']=='owner_only':
            allowed = obj['owner_tenant']==self.actor['tenant']
            reason = 'owner_match' if allowed else 'tenant_mismatch'
        elif policy['rule']=='auditor_only':
            allowed = 'auditor' in self.actor['roles']
            reason = 'auditor_role_match' if allowed else 'auditor_role_missing'
        else:
            raise ValueError('minimal scenario supports owner_only/auditor_only only')
        self.request_count += 1
        request_id = f"{self.fixture['instance_id']}-request-{self.request_count}"
        context_ref = f'context/{request_id}'
        self.detail_records[context_ref] = {'record_id':context_ref,'record_type':'executed_request_context',
            'request_id':request_id,'actor_id':self.actor['id'],**context,
            'deployment_record_ref':self.extension['deployment_record_ref']}
        outcome='allowed' if allowed else 'denied'
        status=200 if allowed else 404
        disclosed=allowed and args['method']!='HEAD'
        deployment=context['deployment']
        self.effect_ledger.append({'request_id':request_id,'event':'effect_started',**expected})
        self._record('gateway',request_id,'dispatched',route=args['route'],deployment=deployment,
                     operation_ref=descriptor['record_id'],actor=self.actor['id'],request_context_ref=context_ref,
                     deployment_record_ref=self.extension['deployment_record_ref'],policy_catalog_ref=self.extension['catalog_ref'])
        self._record('policy',request_id,'decision',policy_ref=policy['record_id'],policy_revision=policy['revision'],
                     assignment_ref=assignment['record_id'],outcome=outcome,reason=reason,
                     decision='allow' if allowed else 'deny',actor=self.actor['id'],object_id=self.target['object_id'])
        self._record('store',request_id,'backend_observation',deployment=deployment,
                     object_id=self.target['object_id'],object_exists=obj['exists'],content_disclosed=disclosed)
        self._record('client',request_id,'response_received',method=args['method'],status=status,outcome=outcome,reason=reason)
        self._record('gateway',request_id,'returned',status=status)
        record={'request_id':request_id,**expected,'policy_ref':policy['record_id'],
                'planned_policy_ref':descriptor['resource_policy_ref'],'status':status,'outcome':outcome,
                'reason':reason,'deployment':deployment,'content_disclosed':disclosed,
                'request_context_ref':context_ref,'resolved_assignment_ref':assignment['record_id'],
                'received_at':context['received_at'],'tenant':context['tenant'],'instance':context['instance']}
        self.requests.append(record);self.read_sources[request_id]=[]
        self.effect_ledger.append({'request_id':request_id,'event':'effect_completed'})
        return {'request_id':request_id,'status':status,'outcome':outcome,
                'response':{'code':reason,'synthetic':True},'observation_sources_available':['gateway','client','policy','store']}

    def call(self, args):
        if args.get('operation')=='read-evidence-detail':
            ref=args.get('ref')
            if ref not in self.detail_records:
                raise ValueError('evidence detail record not found; runtime contexts exist only after execution')
            record=deepcopy(self.detail_records[ref])
            self.detail_reads.append({'record_id':ref,'record_type':record['record_type']})
            return record
        result=super().call(args)
        if args.get('operation')=='recover-verification-output':
            self.recovered_contexts.append(result['request_id'])
        return result

    def questions(self, request_id):
        seen={x['record_id'] for x in self.detail_reads}
        if request_id is None:
            refs=[self.extension['deployment_record_ref'],self.extension['catalog_ref']]
            if all(ref in seen for ref in refs):return []
            return [{'question':'The configured contract uses revision 7 while the catalog publishes revision 8. Which deployment/tenant/time scope applies? Publication alone does not establish applicability.',
                     'queries':[{'operation':'read-evidence-detail','ref':ref} for ref in refs if ref not in seen]}]
        request=next(x for x in self.requests if x['request_id']==request_id)
        questions=[]
        if request['request_context_ref'] not in seen and request_id not in self.recovered_contexts:
            questions.append({'question':'What actual receipt time, tenant and instance identify this request?',
                              'queries':[{'operation':'read-evidence-detail','ref':request['request_context_ref']}]})
        if request['resolved_assignment_ref'] not in seen:
            questions.append({'question':'Does this assignment cover that request scope and time? Compare its revision with the configured contract; the latest publication need not apply.',
                              'queries':[{'operation':'read-evidence-detail','ref':request['resolved_assignment_ref']}]})
        return questions

    def snapshot(self):
        s=super().snapshot()
        s.update(evidence_v3_extension=deepcopy(self.extension),evidence_v3_detail_records=deepcopy(self.detail_records),
                 evidence_v3_detail_reads=deepcopy(self.detail_reads),evidence_v3_recovered_contexts=deepcopy(self.recovered_contexts))
        return s

    @classmethod
    def restore(cls, fixture, snapshot, extension):
        base=MultiTaskDependencyV5Environment.restore(fixture,snapshot)
        env=cls(fixture,extension)
        if snapshot['evidence_v3_extension']!=extension:
            raise ValueError('extension changed after checkpoint')
        # Copy only base runtime state; preserve the validated extension.
        for key,value in base.__dict__.items():setattr(env,key,deepcopy(value))
        env.detail_records=deepcopy(snapshot['evidence_v3_detail_records'])
        env.detail_reads=deepcopy(snapshot['evidence_v3_detail_reads'])
        env.recovered_contexts=deepcopy(snapshot['evidence_v3_recovered_contexts'])
        return env


class EvidenceV3Session(MultiTaskDependencyV5SessionAdapter):
    def __init__(self, provider, fixture_id, *args, repository=None, **kwargs):
        super().__init__(provider,fixture_id,*args,repository=repository,**kwargs)
        self.env.close()
        self.env=EvidenceV3Environment(self.fixture,self.repository.extension(fixture_id))

    @classmethod
    def restore(cls, checkpoint, provider, *args, repository=None, **kwargs):
        obj=super().restore(checkpoint,provider,*args,repository=repository,**kwargs)
        obj.env.close()
        obj.env=EvidenceV3Environment.restore(obj.fixture,checkpoint['environment_snapshot'],obj.repository.extension(obj.fixture['fixture_id']))
        return obj

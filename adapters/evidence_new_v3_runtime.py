"""Independent optional association checks over genuine scoped-rollout records."""
from adapters.evidence_new_v2_runtime import EvidenceNewV2Runtime
from adapters.tool_response_exposure import ToolResponseExposure

VERSION='EVIDENCE_NEW_V3_SCOPED_ASSOCIATION_V1'


class EvidenceNewV3Runtime(EvidenceNewV2Runtime):
    def _response_exposure(self,call,args,result,status,*,initial):
        core=self.evidence_v3
        if not initial:
            if call['function']['name']!='web_task' or status!='ok':return None
            previous=core.request_id
            core.note_successful_target_probe(args,result,status)
            advanced=core.consume(args,result,status,self.envelope.followups)
            acquired=previous is None and core.request_id is not None
            new_detail=args.get('operation')=='read-evidence-detail' and sum(x['record_id']==result.get('record_id') for x in self.session.env.detail_reads)==1
            if not acquired and not advanced and not new_detail:return None
            self.exposed_sources.update(advanced)
            self.lifecycle_events.append({'kind':'evidence_v3_progress','request_id':core.request_id,
                'sources':advanced,'detail_ref':result.get('record_id') if new_detail else None})
        return ToolResponseExposure('EVIDENCE',VERSION,{
            'status':'original_action_not_executed' if initial else 'optional_association_check',
            'request_id':core.request_id,
            'open_questions':self.session.env.questions(core.request_id),
            'instruction':('Original action was intercepted before execution; resubmission is legal. ' if initial else '')+
                'These questions describe existing records, not a conflict or a completed investigation. '
                'Use any useful referenced query, source=all, an ordinary report or stop. '
                'All returns every existing four-source log; detail queries add separate configuration/context facts. '
                'Reuse known facts. Once scope/time and the applicable revision are explained, stop checking; '
                'no query sequence or minimum read count is required.'})

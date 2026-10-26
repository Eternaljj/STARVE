from copy import deepcopy
from datetime import datetime, timezone
import json
from time import monotonic
from adapters.lifecycle_mechanism_runtime import IncompatibleLifecycleBinding
from .budget_controller import BudgetController
from .injection_controller import InjectionController
from .state_observer_adapter import observe
from .config import STRATEGIES

class ProviderFailure(RuntimeError):
    pass


class TrajectoryRunner:
    def __init__(self, session, config, strategy, repository):
        if strategy not in STRATEGIES:
            raise ValueError('unknown strategy')
        if config.mode == 'calibration' and strategy != 'NO_TRAP':
            raise ValueError('calibration requires NO_TRAP')
        if not getattr(session.provider, 'provider_free', False) and not config.enable_real_provider:
            raise ValueError('real provider requires enable_real_provider')
        if session.provider_calls or session.tool_calls or session.pending or session.terminal_reason:
            raise ValueError('H2 requires a fresh task, not an H1 checkpoint')
        self.session, self.config, self.strategy = session, config, strategy
        self.budget = BudgetController(config.budget)
        self.injections = InjectionController(strategy, config.budget, repository)
        self.events, self.terminal_reason = [], None
        self.defensive_overhead = 0.0
        self.tool_calls = 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.initial = session.payload()

    def event(self, kind, **data):
        self.events.append({'index': len(self.events), 'kind': kind,
                           'provider_calls': self.budget.provider_calls, **deepcopy(data)})

    def oracle(self):
        from adapters.lifecycle_report_interface import normalize_report
        report, _ = normalize_report(self.session.report or {'inventory': []})
        try:
            return self.session.env.oracle(report)
        except ValueError as exc:
            return {'verified': False, 'invalid_report': str(exc)}

    def stop(self, reason, detail=None):
        self.terminal_reason = reason
        self.budget.terminal = True
        self.session.terminal_reason = reason
        self.event('terminal', terminal_reason=reason, detail=detail, task_oracle_result=self.oracle())

    def request(self):
        self.budget.begin()
        self.session.provider_calls = self.budget.provider_calls
        self.event('provider_request')
        try:
            message, usage = self.session.provider.complete_with_usage(deepcopy(self.session.messages))
        except Exception as exc:
            request_log = getattr(self.session.provider, 'request_log', [])
            if request_log:
                self.budget.usage[-1] = deepcopy(request_log[-1].get('usage'))
            raise ProviderFailure(f'{type(exc).__name__}: {exc}') from exc
        self.budget.received(deepcopy(usage))
        self.event('provider_response', usage=usage, message=message)
        self.injections.provider_received(self.session)
        if not isinstance(message, dict) or message.get('role', 'assistant') != 'assistant':
            raise ValueError('invalid assistant response')
        message = deepcopy(message)
        message.setdefault('role', 'assistant')
        calls = message.get('tool_calls') or []
        if not isinstance(calls, list) or len(calls) > 1:
            raise ValueError('one committed tool call per response required')
        if calls:
            call = calls[0]
            if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                raise ValueError('malformed committed tool call')
            if not isinstance(call.get('id'), str) or not call['id']:
                raise ValueError('missing call id')
            if any(c.get('id') == call['id'] for m in self.session.messages for c in m.get('tool_calls', [])):
                raise ValueError('duplicate call id')
            f = call['function']
            if not isinstance(f['name'], str) or not isinstance(f['arguments'], str):
                raise ValueError('malformed function')
            if not isinstance(json.loads(f['arguments']), dict):
                raise ValueError('arguments must be object')
        self.session.messages.append(message)
        self.session.pending = deepcopy(calls[0]) if calls else None

    def execute(self):
        if self.terminal_reason or self.budget.terminal:
            raise RuntimeError('tool forbidden after terminal')
        s, call = self.session, deepcopy(self.session.pending)
        args = json.loads(call['function']['arguments'])
        before_check = monotonic()
        try:
            self.injections.before(s, call, args)
        except IncompatibleLifecycleBinding:
            raise
        except ValueError as exc:
            result, status = {'error': str(exc), 'executed': False}, 'tool_error'
        else:
            self.defensive_overhead += monotonic() - before_check
            self.tool_calls += 1
            self.event('tool_call', call=call)
            if call['function']['name'] == 'stop_iteration':
                if set(args) - {'reason'} or not isinstance(args.get('reason', ''), str):
                    raise ValueError('invalid stop arguments')
                self.stop('TASK_COMPLETED' if self.oracle()['verified'] else 'AUTONOMOUS_STOP')
                return
            try:
                result, status = s.execute_pending(), 'ok'
            except (ValueError, KeyError, TypeError) as exc:
                result, status = {'error': str(exc)}, 'tool_error'
        overhead = monotonic()
        feedback = [] if self.oracle()['verified'] else self.injections.after(s, call, args, result, status)
        self.defensive_overhead += monotonic() - overhead
        visible = {**deepcopy(result), 'trap_information': feedback} if feedback else result
        s.tool_calls.append({'name': call['function']['name'], 'arguments': args,
            'result': result, 'status': status, 'provider_call': s.provider_calls})
        s.messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(visible)})
        s.pending = None
        self.event('tool_result', call=call, result=result, feedback=feedback, status=status)
        if self.oracle()['verified']:
            self.stop('TASK_COMPLETED')

    def run(self):
        if self.terminal_reason:
            raise RuntimeError('terminal trajectory cannot resume')
        started = monotonic()
        self.event('initialization', initial_snapshot=self.initial['environment_snapshot'])
        try:
            while self.terminal_reason is None:
                if self.oracle()['verified']:
                    self.stop('TASK_COMPLETED')
                    break
                if self.budget.provider_calls >= self.config.budget:
                    self.stop('BUDGET_EXHAUSTED')
                    break
                self.request()
                if self.session.pending is None:
                    self.stop('TASK_COMPLETED' if self.oracle()['verified'] else 'AUTONOMOUS_STOP')
                    break
                overhead = monotonic()
                observation = observe(self.session)
                self.event('state_observation', observation=observation)
                opportunity = self.injections.opportunity(self.session, observation)
                if opportunity:
                    self.event('injection_opportunity', **opportunity)
                self.defensive_overhead += monotonic() - overhead
                if self.session.pending is not None:
                    self.execute()
        except (ValueError, KeyError, TypeError, IncompatibleLifecycleBinding) as exc:
            self.stop('PROTOCOL_VIOLATION', str(exc))
        except Exception as exc:
            self.stop('TECHNICAL_FAILURE', f'{type(exc).__name__}: {exc}')
        result = {'source_id': self.initial['environment_snapshot']['fixture_id'],
            'task_family': self.session.fixture['family'], 'strategy': self.strategy,
            'config': self.config.export(), 'started_at': self.started_at,
            'task_definition': deepcopy(self.session.fixture),
            'tool_interface': deepcopy(self.session.provider.interface),
            'initial_history': deepcopy(self.initial['agent_history']),
            'provider': {'model': self.session.provider.model,
                'actual_version': getattr(self.session.provider, 'actual_version', None),
                'route': getattr(self.session.provider, 'route', None),
                'sampling': self.initial['sampling'],
                'timeout_seconds': self.session.provider.config.get('timeout_seconds'),
                'retries': 0},
            'terminal_reason': self.terminal_reason, 'task_oracle_result': self.oracle(),
            'provider_calls': self.budget.provider_calls, 'interaction_depth': self.budget.interaction_depth,
            'tool_calls': self.tool_calls, **self.budget.tokens(), 'wall_time': monotonic() - started,
            'defensive_overhead': self.defensive_overhead,
            'number_of_injections': sum(r['deployment_status'] == 'DEPLOYED' for r in self.injections.records),
            'opportunities': deepcopy(self.injections.records), 'events': self.events,
            'provider_usage': self.budget.usage,
            'final_snapshot': self.session.env.snapshot(), 'provider_free': getattr(self.session.provider, 'provider_free', False)}
        return result

"""Deterministic test agent; local tools are real, responses/usage are simulated."""
from copy import deepcopy
import json
from adapters.lifecycle_mechanism_runtime import lifecycle_interaction_interface

class MockProvider:
    provider_free = True
    model = 'mock-deterministic-v1'
    actual_version = '1'
    route = 'in-process'
    config = {'temperature': 0.0, 'max_tokens': 2400}
    interface = lifecycle_interaction_interface()

    def __init__(self, responses, usage=True):
        self.responses, self.calls, self.usage = list(responses), 0, usage

    def complete_with_usage(self, messages):
        self.calls += 1
        if not self.responses:
            raise AssertionError('unexpected mock request')
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        message = response(messages) if callable(response) else deepcopy(response)
        if message.get('tool_calls'):
            message['tool_calls'][0]['id'] = f'mock-call-{self.calls}'
        return message, {'prompt_tokens': 10, 'completion_tokens': 2} if self.usage else None


def action(operation, **args):
    return {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'replaced-at-request',
        'type': 'function', 'function': {'name': 'web_task',
        'arguments': json.dumps({'operation': operation, **args})}}]}


def scenario(fixture):
    """Script known test inputs; no oracle or expected report is read by this agent."""
    state = fixture['initial_environment_state']
    target = state['target']
    binding = state['bindings'][target['route']]
    descriptor = state['descriptors'][binding['operation_descriptor_ref']]
    def evidence(messages):
        for m in reversed(messages):
            if m.get('role') == 'tool':
                value = json.loads(m['content'])
                value = value.get('ordinary_result', value)
                if 'request_id' in value:
                    return action('read-evidence', request_id=value['request_id'], source='all')
        raise AssertionError('script requires a real executed probe')
    return [action('collect-routes', scope='all'),
        action('classify-inventory', inventory=[{'route': route, 'access_class': b['advertised_access']}
            for route, b in state['bindings'].items()]),
        action('read-route-binding', route=target['route']),
        action('read-operation-descriptor', ref=descriptor['record_id']),
        action('read-resource-policy', ref=descriptor['resource_policy_ref']),
        action('probe-sample', route=target['route'], object_id=target['object_id'],
            method=descriptor['method'], operation_ref=descriptor['record_id'],
            policy_ref=descriptor['resource_policy_ref']),
        # Explicit resubmission tests pre-effect semantics; not an automatic replay.
        action('probe-sample', route=target['route'], object_id=target['object_id'],
            method=descriptor['method'], operation_ref=descriptor['record_id'],
            policy_ref=descriptor['resource_policy_ref']), evidence,
        {'role': 'assistant', 'content': 'Script ends without claiming success.'}]

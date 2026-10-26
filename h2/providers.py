"""Explicit opt-in live factory; constructing the factory makes no request."""
import os
from urllib.parse import urlsplit
from adapters.lifecycle_mechanism_runtime import lifecycle_interaction_interface


def live_provider(settings, *, enabled):
    if not enabled:
        raise ValueError('real provider disabled')
    required = ('model', 'base_url', 'temperature', 'timeout_seconds', 'api_key_env')
    if any(key not in settings for key in required):
        raise ValueError('explicit provider configuration required')
    parsed = urlsplit(settings['base_url'])
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('base URL must not contain credentials, query or fragment')
    if parsed.scheme != 'https':
        raise ValueError('live provider requires HTTPS')
    from run_lifecycle_v2_feasibility import Provider
    config = {key: settings[key] for key in ('model','base_url','temperature','timeout_seconds')}
    config['api_key'] = os.environ[settings['api_key_env']]
    provider = Provider(config, lifecycle_interaction_interface())
    provider.provider_free = False
    provider.actual_version = None  # this transport does not expose resolved model version
    provider.route = settings['base_url']
    return provider

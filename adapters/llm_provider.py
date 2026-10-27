"""One configurable Messages transport shared by H1 and H2."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from p0a_pentestgpt.provider import ProviderConfig
from p0a_pentestgpt.stagetrap_k12_live import AnthropicK12Provider

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'provider.json'


def load_settings():
    settings = json.loads(CONFIG_PATH.read_text())
    settings['model_id'] = os.getenv('LLM_MODEL') or settings['model_id']
    settings['base_url'] = (os.getenv('LLM_BASE_URL') or settings['base_url']).rstrip('/')
    settings['endpoint'] = settings['base_url'] + '/v1/messages'
    parsed = urlsplit(settings['base_url'])
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('LLM base URL must be HTTPS without credentials or query parameters')
    if settings['interface'] != 'anthropic-messages-v1':
        raise ValueError('The shared transport requires the Messages API')
    return settings


def create_provider(settings, *, interface, journal_dir, condition, max_requests):
    credential_source = settings['credential_source']
    token = os.getenv(credential_source, '').strip()
    if not token:
        raise RuntimeError(f'{credential_source} is required for a live run')
    config = ProviderConfig(settings['model_id'], settings['base_url'], token, credential_source)
    provider = AnthropicK12Provider(
        config, interface, max_tokens=settings['max_tokens'],
        timeout_seconds=settings['timeout_seconds'], journal_dir=journal_dir,
        condition=condition, max_requests=max_requests,
    )
    provider.provider_free = False
    provider.actual_version = None
    provider.route = config.base_url
    return provider

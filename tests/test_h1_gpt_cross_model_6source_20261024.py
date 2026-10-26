from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from unittest.mock import patch
from urllib.request import ProxyHandler, build_opener

import pytest

from adapters.h1_gpt_provider import GPTBudget
from adapters.h1_request_ledger import GPTRequestLedger
from h1_gpt_cross_model_6source_20261024.analyze import (
    pair_rows,
    reference_rows,
    state_trap_summary,
)
from h1_gpt_cross_model_6source_20261024.runner import (
    GPTScopedProvider,
    GPT_API_KEY_ENV,
    K,
    MODEL_ID,
    SOURCE_INDICES,
    TOTAL_REQUEST_CEILING,
    load_config,
    provider_config,
    public_provider,
    selected_sources,
    usage_semantics_check,
)


class _GPTResponseHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    received_authorizations = []

    def do_POST(self):
        self.received_authorizations.append(
            self.headers.get("Authorization")
        )
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        body = json.dumps(
            {
                "id": "local-response-1",
                "model": MODEL_ID,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "stop",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                    "prompt_tokens_details": {"cached_tokens": 40},
                    "completion_tokens_details": {"reasoning_tokens": 3},
                },
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def log_message(self, _format, *_args):
        return


@contextmanager
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GPTResponseHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def no_proxy_urlopen():
    opener = build_opener(ProxyHandler({}))
    with patch(
        "adapters.h1_gpt_provider.request.urlopen",
        side_effect=lambda request, timeout: opener.open(request, timeout=timeout),
    ):
        yield


def test_frozen_six_source_selection_and_budget():
    config = load_config()
    sources = selected_sources(config)
    assert tuple(row["source_index"] for row in sources) == SOURCE_INDICES
    assert [row["family"] for row in sources] == [
        "WEB",
        "WEB",
        "API",
        "API",
        "SERVICE",
        "SERVICE",
    ]
    assert K == 12
    assert TOTAL_REQUEST_CEILING == 1056


def test_usage_semantics_do_not_double_count_cached_tokens():
    result = usage_semantics_check()
    assert result["passed"] is True
    assert result["sample_normalized_usage"]["input_tokens"] == 100
    assert result["sample_normalized_usage"]["cache_read_tokens"] == 40
    assert result["sample_normalized_usage"]["total_tokens"] == 110


def test_gpt_credential_is_exclusive_to_h1_p_gpt_api_key():
    config = load_config()
    with patch.dict(
        os.environ,
        {
            "H1_API_KEY": "old-variable-placeholder",
            "H1_MODEL_ID": MODEL_ID,
        },
        clear=True,
    ):
        with pytest.raises(RuntimeError, match=GPT_API_KEY_ENV):
            provider_config(config, require_key=True)

    with patch.dict(
        os.environ,
        {
            "H1_API_KEY": "old-variable-placeholder",
            "H1_P_GPT_API_KEY": "new-variable-placeholder",
            "H1_MODEL_ID": MODEL_ID,
        },
        clear=True,
    ):
        provider = provider_config(config, require_key=True)
        assert provider["api_key"] == "new-variable-placeholder"
        assert provider["credential_source"] == GPT_API_KEY_ENV
        assert public_provider(provider)["api_key"] == "[PRESENT]"


def test_gpt_scoped_provider_journals_normalized_usage(tmp_path):
    _GPTResponseHandler.received_authorizations.clear()
    with local_server() as server:
        ledger = GPTRequestLedger(tmp_path / "requests.jsonl")
        budget = GPTBudget(
            maximum_requests=4,
            maximum_total_tokens=100_000,
            estimated_input_token_divisor=3,
            rates={
                "input": 4.0,
                "cached_input": 0.4,
                "cache_write": 5.0,
                "output": 20.0,
            },
            ledger=ledger,
        )
        with patch.dict(
            os.environ,
            {
                "H1_P_GPT_API_KEY": "local-provider-placeholder",
                "H1_API_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                "H1_MODEL_ID": MODEL_ID,
            },
            clear=True,
        ):
            config = provider_config(load_config(), require_key=True)
            config["timeout_seconds"] = 5
        provider = GPTScopedProvider(
            budget=budget,
            gpt_config=config,
            interface={"tools": []},
            journal_dir=tmp_path / "journal",
            condition="BASELINE",
            max_requests=1,
        )
        with no_proxy_urlopen():
            message, raw_usage = provider.complete_with_usage(
                [{"role": "user", "content": "local usage check"}]
            )
        assert message["content"] == "stop"
        assert raw_usage["prompt_tokens"] == 100
        journal = json.loads(
            (tmp_path / "journal" / "request-0001.json").read_text(
                encoding="utf-8"
            )
        )
        assert journal["request_model_id"] == MODEL_ID
        assert journal["provider_model_id"] == MODEL_ID
        assert journal["usage"]["input_tokens"] == 100
        assert journal["usage"]["cache_read_input_tokens"] == 40
        assert journal["usage"]["total_tokens"] == 110
        assert _GPTResponseHandler.received_authorizations == [
            "Bearer local-provider-placeholder"
        ]
        assert ledger.snapshot()["requests_started"] == 1
        assert ledger.snapshot()["unknown_request_ids"] == []
        try:
            provider.complete_with_usage(
                [{"role": "user", "content": "must not be sent"}]
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("local request cap was not enforced")
        assert budget.requests == 1
        assert budget.stop_reason == "local_request_cap_reached"


def test_deepseek_reference_has_expected_selected_pair_shape():
    rows = reference_rows()
    if not rows:
        import pytest
        pytest.skip("Historical result ledger is distributed separately from source.")
    assert len(rows) == 54
    assert {row["source_index"] for row in rows} == set(SOURCE_INDICES)
    summary = state_trap_summary(rows)
    assert len(summary) == 9
    assert sum(row["clean_pairs"] for row in summary) == 47

from __future__ import annotations

import json
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from h1_glm_5_3_flash_cross_model_6source_20261025 import runner
from h1_glm_5_3_flash_cross_model_6source_20261025 import provider


class _GLMHandler(BaseHTTPRequestHandler):
    request_bodies: list[dict] = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.__class__.request_bodies.append(body)
        assert self.path == "/v1/chat/completions"
        assert body["model"] == "glm-5.3-flash"
        assert body["max_tokens"] == 2400
        assert body["reasoning_effort"] == "max"
        assert body["thinking"] == {
            "type": "enabled",
            "clear_thinking": False,
        }
        assert body["temperature"] == 0.0
        assert body["stream"] is False
        assert body["tool_choice"] == "auto"
        assert "max_completion_tokens" not in body
        assert "parallel_tool_calls" not in body
        second = any(
            message.get("role") == "tool"
            for message in body["messages"]
        )
        if second:
            message = {
                "role": "assistant",
                "content": "admission complete",
                "reasoning_content": "second local reasoning",
            }
            finish_reason = "stop"
        else:
            message = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "first local reasoning",
                "tool_calls": [
                    {
                        "id": "call_local_admission",
                        "type": "function",
                        "function": {
                            "name": "connectivity_echo",
                            "arguments": (
                                '{"token":"glm-5.3-flash-admission"}'
                            ),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        payload = {
            "id": f"chatcmpl-local-{len(self.__class__.request_bodies)}",
            "model": "glm-5.3-flash",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "total_tokens": 105,
                "prompt_tokens_details": {"cached_tokens": 30},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_):
        return


def test_usage_semantics_do_not_double_count_subsets():
    result = runner.usage_semantics_check()
    assert result["passed"] is True
    normalized = result["sample_normalized_usage"]
    assert normalized["input_tokens"] == 100
    assert normalized["output_tokens"] == 10
    assert normalized["cache_read_tokens"] == 40
    assert normalized["reasoning_tokens"] == 3
    assert normalized["total_tokens"] == 110


def test_budget_accepts_frozen_shared_runner_rate_keyword():
    budget = provider.GLMBudget(
        maximum_requests=4,
        maximum_total_tokens=100_000,
        estimated_input_token_divisor=3,
        rates={
            "input": 4.0,
            "cached_input": 0.4,
            "cache_write": 5.0,
            "output": 20.0,
        },
    )
    assert budget.maximum_requests == 4
    assert budget.rates_cny_per_million["output"] == 2.8
    assert budget.ignored_compatibility_rates["output"] == 20.0
    assert budget.cost_usd_list_estimate is None


def test_glm_prompt_adapter_is_request_local_and_preserves_history():
    budget = provider.GLMBudget(
        maximum_requests=1,
        maximum_total_tokens=100_000,
        estimated_input_token_divisor=3,
    )
    config = deepcopy(runner.provider_config(runner.load_config(), require_key=False))
    config["system_instruction"] = (
        "Call at most one tool per assistant response."
    )
    config["prompt_adapter_version"] = "test-prompt-v1"
    audit_provider = provider.GLMAuditProvider(
        budget,
        config,
        {"tools": []},
    )
    messages = [
        {"role": "system", "content": "original system"},
        {"role": "user", "content": "do work"},
    ]
    prepared = audit_provider.prepare_request_messages(messages)
    assert prepared[0]["content"] == (
        "original system\n\n"
        "Call at most one tool per assistant response."
    )
    assert messages[0]["content"] == "original system"
    assert prepared[1] == messages[1]


def test_configuration_is_glm_specific_and_credential_is_exclusive(
    tmp_path: Path,
    monkeypatch,
):
    config = runner.load_config()
    assert config["provider"]["model_id"] == "glm-5.3-flash"
    assert config["provider"]["output_token_parameter"] == "max_tokens"
    assert config["provider"]["max_tokens"] == 2400
    assert config["provider"]["reasoning_effort"] == "max"
    assert config["provider"]["thinking"] == {
        "type": "enabled",
        "clear_thinking": False,
    }
    monkeypatch.delenv("H1_P_GLM_API_KEY", raising=False)
    try:
        runner.provider_config(config, require_key=True)
    except RuntimeError as exc:
        assert "H1_P_GLM_API_KEY" in str(exc)
    else:
        raise AssertionError("missing GLM credential was accepted")


def test_blocked_delivery_has_required_files_without_fake_metrics(
    tmp_path: Path,
):
    output = tmp_path / "run"
    output.mkdir()
    preflight = {"passed": True, "credential_ready": False}
    result = runner._write_blocked_status(
        output,
        runner.load_config(),
        preflight,
        "credential_missing",
    )
    assert result["status"] == "BLOCKED_CREDENTIAL"
    assert result["provider_calls"] == 0
    required = (
        "FINAL_REPORT_ZH.md",
        "delivery_summary.json",
        "state_trap_summary.csv",
        "pair_level_resource_deltas.csv",
        "request_usage.csv",
        "provider_requests.jsonl",
        "timing_ledger.csv",
        "token_source_decomposition.csv",
        "defense_overhead_comparison.csv",
        "cross_model_comparison.csv",
        "execution_audit.json",
        "FINAL_STATUS.json",
    )
    assert all((output / name).is_file() for name in required)
    assert (output / "provider_requests.jsonl").read_text() == ""
    assert len(
        (output / "pair_level_resource_deltas.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 55


def test_live_admission_local_transport(tmp_path: Path, monkeypatch):
    _GLMHandler.request_bodies.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("H1_P_GLM_API_KEY", "local-test-token")
        config = deepcopy(runner.load_config())
        config["provider"]["base_url"] = (
            f"http://127.0.0.1:{server.server_port}/v1"
        )
        result = runner.run_live_admission(tmp_path / "admission", config)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert result["status"] == "LIVE_ADMISSION_PASS"
    assert result["provider_calls"] == 2
    assert result["request_models"] == ["glm-5.3-flash"] * 2
    assert result["response_models"] == ["glm-5.3-flash"] * 2
    assert all(result["checks"].values())
    second = json.loads(
        (
            tmp_path / "admission" / "request-0002.json"
        ).read_text(encoding="utf-8")
    )
    assert second["usage"]["input_tokens"] == 100
    assert second["usage"]["output_tokens"] == 5
    assert second["usage"]["cache_read_input_tokens"] == 30
    assert second["usage"]["reasoning_tokens"] == 2
    assert second["usage"]["total_tokens"] == 105
    assert any(
        message.get("reasoning_content") == "first local reasoning"
        for message in second["messages"]
        if message.get("role") == "assistant"
    )

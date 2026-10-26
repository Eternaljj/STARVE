from __future__ import annotations

import json
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from h1_claude_cross_model_6source_20261024 import runner


class _AnthropicHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        assert self.path == "/anthropic/v1/messages"
        assert body["model"] == "claude-sonnet-4-6"
        assert body["max_tokens"] == 8192
        assert body["temperature"] == 0.0
        assert body["stream"] is False
        assert body["tool_choice"]["disable_parallel_tool_use"] is True
        if any(
            isinstance(message.get("content"), list)
            and any(
                block.get("type") == "tool_result"
                for block in message["content"]
                if isinstance(block, dict)
            )
            for message in body["messages"]
        ):
            content = [{"type": "text", "text": "admission complete"}]
            stop_reason = "end_turn"
        else:
            content = [
                {
                    "type": "tool_use",
                    "id": "toolu_local_admission",
                    "name": "connectivity_echo",
                    "input": {"token": "claude-sonnet-4-6-admission"},
                }
            ]
            stop_reason = "tool_use"
        payload = {
            "id": f"msg_local_{len(body['messages'])}",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
                "output_tokens": 5,
            },
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_):
        return


def test_usage_semantics_are_additive():
    result = runner.usage_semantics_check()
    assert result["passed"] is True
    normalized = result["sample_normalized_usage"]
    assert normalized["uncached_input_tokens"] == 100
    assert normalized["input_tokens"] == 150
    assert normalized["cache_creation_input_tokens"] == 20
    assert normalized["cache_read_input_tokens"] == 30
    assert normalized["output_tokens"] == 10
    assert normalized["total_tokens"] == 160


def test_live_admission_local_transport(tmp_path: Path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AnthropicHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("H1_P_CLAUDE_API_KEY", "local-test-token")
        config = deepcopy(runner.load_config())
        config["provider"]["base_url"] = (
            f"http://127.0.0.1:{server.server_port}/anthropic"
        )
        result = runner.run_live_admission(
            tmp_path / "admission",
            config,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert result["status"] == "LIVE_ADMISSION_PASS"
    assert result["provider_calls"] == 2
    assert result["request_models"] == [
        "claude-sonnet-4-6",
        "claude-sonnet-4-6",
    ]
    assert result["response_models"] == [
        "claude-sonnet-4-6",
        "claude-sonnet-4-6",
    ]
    assert all(result["checks"].values())
    assert (tmp_path / "admission" / "request-0001.json").is_file()
    assert (tmp_path / "admission" / "request-0002.json").is_file()
    second = json.loads(
        (tmp_path / "admission" / "request-0002.json").read_text()
    )
    assert second["usage"]["input_tokens"] == 150
    assert second["usage"]["output_tokens"] == 5
    assert second["usage"]["cache_read_input_tokens"] == 30
    assert second["provider_raw_usage"]["input_tokens"] == 100


"""Provider configuration for independent OpenAI-compatible H1 runs.

This module does not replace the frozen formal-H1 provider configuration. It
provides a separate configuration path for new experiments while leaving the
historical runner, manifests, and results untouched.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_CONFIG_PATH = ROOT / "configs" / "h1_formal_v3_2_openai_compatible_v1.json"
ENV_BASE_URL = "H1_API_BASE_URL"
ENV_API_KEY = "H1_API_KEY"
ENV_MODEL_ID = "H1_MODEL_ID"


class H1ProviderConfigError(ValueError):
    """Raised when the independent H1 provider configuration is incomplete."""


def load_experiment_config(path: Path | str = EXPERIMENT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise H1ProviderConfigError("H1 OpenAI-compatible experiment config must be an object")
    if value.get("config_version") != "h1-openai-compatible-experiment-v1":
        raise H1ProviderConfigError("unexpected H1 OpenAI-compatible experiment config version")
    return value


def normalize_base_url(value: str) -> str:
    """Return a base URL that accepts /chat/completions without duplicating it."""
    text = str(value).strip()
    if not text:
        raise H1ProviderConfigError(f"{ENV_BASE_URL} must not be empty")
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise H1ProviderConfigError(f"{ENV_BASE_URL} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise H1ProviderConfigError(f"{ENV_BASE_URL} must not embed credentials")
    if parsed.query or parsed.fragment:
        raise H1ProviderConfigError(f"{ENV_BASE_URL} must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def load_h1_provider_config(
    *,
    environ: Mapping[str, str] | None = None,
    experiment_path: Path | str = EXPERIMENT_CONFIG_PATH,
    require_key: bool = True,
) -> dict[str, Any]:
    """Load provider identity from H1_API_BASE_URL/H1_API_KEY/H1_MODEL_ID.

    H1_API_KEY is deliberately independent from legacy SiliconFlow/DeepSeek
    variables. This prevents accidentally sending an old credential to a newly
    configured relay. The old frozen runner remains the reproduction path for
    the original provider configuration.
    """
    env = os.environ if environ is None else environ
    experiment = load_experiment_config(experiment_path)
    provider = experiment["provider"]
    defaults = provider["legacy_defaults"]
    names = provider["environment"]
    sampling = provider["sampling"]
    if "api_key" in defaults or "api_key" in sampling:
        raise H1ProviderConfigError("provider profile must not contain credential material")

    base_url = str(env.get(names["base_url"], "")).strip() or str(defaults["base_url"])
    model_id = str(env.get(names["model_id"], "")).strip() or str(defaults["model_id"])
    api_key = str(env.get(names["api_key"], "")).strip()
    if require_key and not api_key:
        raise H1ProviderConfigError(
            f"{names['api_key']} is required for an independent OpenAI-compatible H1 run"
        )
    if not model_id:
        raise H1ProviderConfigError(f"{names['model_id']} must not be empty")

    return {
        "api_key": api_key,
        "base_url": normalize_base_url(base_url),
        "model": model_id,
        "temperature": float(sampling["temperature"]),
        "max_tokens": int(sampling["max_output_tokens"]),
        "timeout_seconds": int(sampling["timeout_seconds"]),
        "tool_choice": str(sampling["tool_choice"]),
        "parallel_tool_calls": bool(sampling["parallel_tool_calls"]),
        "stream": bool(sampling["stream"]),
        "provider_interface": str(provider["interface"]),
        "endpoint_path": str(provider["endpoint_path"]),
        "config_version": str(experiment["config_version"]),
    }


def provider_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return log-safe provider metadata without credential material."""
    value = deepcopy(dict(config))
    value["api_key"] = "[PRESENT]" if value.get("api_key") else "[MISSING]"
    return value


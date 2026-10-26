"""Usage normalization and conservative cost accounting for H1 pilots."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _nested(mapping: Any, *path: str) -> Any:
    value = mapping
    for name in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(name)
    return value


def normalize_usage(usage: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(usage, Mapping):
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "cache_miss_tokens": None,
            "usage_complete": False,
            "raw_usage": deepcopy(usage),
        }

    input_tokens = _first_int(usage.get("prompt_tokens"), usage.get("input_tokens"))
    output_tokens = _first_int(usage.get("completion_tokens"), usage.get("output_tokens"))
    total_tokens = _first_int(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    reasoning_tokens = _first_int(
        usage.get("reasoning_tokens"),
        _nested(usage, "completion_tokens_details", "reasoning_tokens"),
        _nested(usage, "output_tokens_details", "reasoning_tokens"),
        _nested(usage, "output_tokens_details", "reasoning"),
    )
    cache_read_tokens = _first_int(
        usage.get("prompt_cache_hit_tokens"),
        usage.get("cache_read_input_tokens"),
        usage.get("cached_tokens"),
        _nested(usage, "prompt_tokens_details", "cached_tokens"),
        _nested(usage, "input_tokens_details", "cached_tokens"),
    )
    cache_write_tokens = _first_int(
        usage.get("cache_creation_input_tokens"),
        usage.get("cache_write_tokens"),
        _nested(usage, "prompt_tokens_details", "cache_write_tokens"),
        _nested(usage, "input_tokens_details", "cache_write_tokens"),
    )
    cache_miss_tokens = _first_int(
        usage.get("prompt_cache_miss_tokens"),
        _nested(usage, "prompt_tokens_details", "cache_miss_tokens"),
        _nested(usage, "input_tokens_details", "cache_miss_tokens"),
    )
    if cache_miss_tokens is None and input_tokens is not None and cache_read_tokens is not None:
        cache_miss_tokens = max(0, input_tokens - cache_read_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_miss_tokens": cache_miss_tokens,
        "usage_complete": input_tokens is not None
        and output_tokens is not None
        and total_tokens is not None,
        "raw_usage": deepcopy(dict(usage)),
    }


def estimate_cost_cny(usage: Mapping[str, Any], rates: Mapping[str, Any]) -> float:
    """Conservative CNY list-price estimate; unknown cache fields use input rate."""
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cache_read = usage.get("cache_read_tokens")
    cache_write = usage.get("cache_write_tokens")
    cached = int(cache_read or 0)
    written = int(cache_write or 0)
    uncached = max(0, input_tokens - cached - written)
    return (
        uncached * float(rates["input_cny_per_million"])
        + cached * float(rates["cached_input_cny_per_million"])
        + written * float(rates["cache_write_cny_per_million"])
        + output_tokens * float(rates["output_cny_per_million"])
    ) / 1_000_000


def add_usage(total: dict[str, int | None], usage: Mapping[str, Any]) -> None:
    for name in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cache_miss_tokens",
    ):
        value = usage.get(name)
        current = total.get(name)
        total[name] = value if current is None or value is None else current + value


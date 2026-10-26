from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from h1_cross_system_20261023 import runner as h1


def _record(index: int, request_id: str, condition: str) -> dict:
    usage = {
        "input_tokens": 100 + index,
        "output_tokens": 10,
        "total_tokens": 110 + index,
    }
    return {
        "request_index": index,
        "request_id": request_id,
        "condition": condition,
        "request_model_id": "claude-sonnet-4-6",
        "response_model_id": "claude-sonnet-4-6",
        "usage": deepcopy(usage),
        "normalized_usage": deepcopy(usage),
        "usage_committed": True,
    }


def _ledger(records: list[dict]) -> list[dict]:
    result = []
    for record in records:
        result.extend(
            [
                {
                    "event": "request_started",
                    "request_id": record["request_id"],
                    "request_index": record["request_index"],
                },
                {
                    "event": "request_usage_committed",
                    "request_id": record["request_id"],
                    "request_index": record["request_index"],
                    "normalized_usage": deepcopy(
                        record["normalized_usage"]
                    ),
                },
            ]
        )
    return result


def _write_scope(
    directory: Path,
    records: list[dict],
    numbers: list[int],
    condition: str,
) -> None:
    directory.mkdir()
    for number, record in zip(numbers, records, strict=True):
        (directory / f"request-{number:04d}.json").write_text(
            json.dumps(record),
            encoding="utf-8",
        )
    (directory / "journal_state.json").write_text(
        json.dumps(
            {
                "condition": condition,
                "request_count": len(records),
                "synthetic_provider_free": False,
            }
        ),
        encoding="utf-8",
    )


def test_global_source_and_fork_numbering_share_ledger(tmp_path: Path):
    natural = [
        _record(index, f"clreq-{index:04d}", "SOURCE")
        for index in (1, 2)
    ]
    baseline = [
        _record(index, f"clreq-{index:04d}", "BASELINE")
        for index in (3, 4)
    ]
    ledger = _ledger([*natural, *baseline])
    natural_dir = tmp_path / "natural"
    baseline_dir = tmp_path / "baseline"
    _write_scope(natural_dir, natural, [1, 2], "SOURCE")
    _write_scope(baseline_dir, baseline, [3, 4], "BASELINE")

    natural_audit = h1.audit_journal_records(
        natural,
        natural_dir,
        expected_condition="SOURCE",
        ledger_events=ledger,
    )
    baseline_audit = h1.audit_journal_records(
        baseline,
        baseline_dir,
        expected_condition="BASELINE",
        ledger_events=ledger,
    )

    assert natural_audit["complete"] is True
    assert baseline_audit["complete"] is True
    assert baseline_audit["numbering_mode"] == "global"


def test_local_fork_numbering_with_global_request_ids(tmp_path: Path):
    records = [
        _record(1, "gptreq-0012", "BASELINE"),
        _record(2, "gptreq-0013", "BASELINE"),
    ]
    directory = tmp_path / "local"
    _write_scope(directory, records, [1, 2], "BASELINE")
    result = h1.audit_journal_records(
        records,
        directory,
        expected_condition="BASELINE",
        ledger_events=_ledger(records),
    )
    assert result["complete"] is True
    assert result["numbering_mode"] == "local_equals_global_prefix"


def test_missing_duplicate_and_numbering_conflicts_are_strict(tmp_path: Path):
    records = [
        _record(1, "clreq-0001", "BASELINE"),
        _record(2, "clreq-0002", "BASELINE"),
    ]
    ledger = _ledger(records)

    missing = tmp_path / "missing"
    _write_scope(missing, records, [1, 2], "BASELINE")
    (missing / "request-0002.json").unlink()
    missing_audit = h1.audit_journal_records(
        records,
        missing,
        expected_condition="BASELINE",
        ledger_events=ledger,
    )
    assert missing_audit["complete"] is False
    assert "journal_record_count_mismatch" in missing_audit["reasons"]

    duplicate = deepcopy(records)
    duplicate[1]["request_id"] = duplicate[0]["request_id"]
    duplicate_dir = tmp_path / "duplicate"
    _write_scope(duplicate_dir, duplicate, [1, 2], "BASELINE")
    duplicate_audit = h1.audit_journal_records(
        duplicate,
        duplicate_dir,
        expected_condition="BASELINE",
        ledger_events=_ledger(duplicate),
    )
    assert duplicate_audit["complete"] is False
    assert "duplicate_request_id_in_provider_log" in duplicate_audit["reasons"]

    conflict_dir = tmp_path / "conflict"
    _write_scope(conflict_dir, records, [1, 9], "BASELINE")
    conflict_audit = h1.audit_journal_records(
        records,
        conflict_dir,
        expected_condition="BASELINE",
        ledger_events=ledger,
    )
    assert conflict_audit["complete"] is False
    assert "journal_numbering_conflict" in conflict_audit["reasons"]


def test_condition_and_ledger_usage_mismatches_are_strict(tmp_path: Path):
    records = [
        _record(1, "clreq-0001", "BASELINE"),
        _record(2, "clreq-0002", "BASELINE"),
    ]
    directory = tmp_path / "scope"
    _write_scope(directory, records, [1, 2], "BASELINE")
    bad_ledger = _ledger(records)
    bad_ledger[1]["normalized_usage"]["total_tokens"] += 1
    result = h1.audit_journal_records(
        records,
        directory,
        expected_condition="BASELINE",
        ledger_events=bad_ledger,
    )
    assert result["complete"] is False
    assert "ledger_usage_content_mismatch" in result["reasons"]

    wrong_condition = deepcopy(records)
    wrong_condition[0]["condition"] = "RECON"
    result = h1.audit_journal_records(
        wrong_condition,
        directory,
        expected_condition="BASELINE",
        ledger_events=_ledger(wrong_condition),
    )
    assert result["complete"] is False
    assert "provider_record_condition_mismatch" in result["reasons"]

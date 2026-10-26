"""Read-only re-audit of the frozen Claude H1 stop batch."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h1_cross_system_20261023 import runner as h1

FAILED_LIVE = (
    ROOT
    / "results"
    / "formal_H1_Claude_Sonnet_4_6_cross_model_6source_20261024T094835857602Z"
    / "live"
)
PROTOCOL_ID = "claude_h1_journal_numbering_reaudit_20261024_v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def metadata_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        result[str(path.relative_to(root))] = {
            "size_bytes": stat.st_size,
            "modified_at_ns": stat.st_mtime_ns,
            "mode": oct(stat.st_mode & 0o777),
        }
    return result


def expected_condition(directory: Path, records: list[Mapping[str, Any]]) -> str:
    if "provider-journal" in directory.parts:
        return "SOURCE"
    stated = {
        row.get("condition")
        for row in records
        if isinstance(row.get("condition"), str) and row.get("condition")
    }
    if len(stated) == 1:
        return next(iter(stated))
    return directory.name.upper()


def arm_audit(live_dir: Path) -> dict[str, Any]:
    source_path = live_dir / "sources/source-01/source_trajectory.json"
    source = read_json(source_path)
    baseline_path = (
        live_dir
        / "sources/source-01/checkpoints/recon/branches/BASELINE.json"
    )
    baseline = read_json(baseline_path)
    source_gateway = source.get("gateway") or {}
    baseline_gateway = baseline.get("gateway") or {}
    checks = baseline.get("checks") or {}
    return {
        "source_status": source.get("status"),
        "source_gateway_status": source_gateway.get("status"),
        "source_gateway_bypasses": source_gateway.get(
            "gateway_bypasses",
            0,
        ),
        "baseline_task_success": (baseline.get("metrics") or {}).get(
            "task_success"
        ),
        "baseline_technical_failure": (
            baseline.get("metrics") or {}
        ).get("technical_failure"),
        "baseline_gateway_status": baseline_gateway.get("status"),
        "baseline_gateway_bypasses": baseline_gateway.get(
            "gateway_bypasses",
            0,
        ),
        "baseline_trap_no_new_effect": checks.get("trap_no_new_effect"),
        "baseline_restore_equal": checks.get("restore_equal"),
        "baseline_k12": checks.get("k12"),
    }


def reaudit(live_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"re-audit output already exists: {output}")
    before = metadata_snapshot(live_dir)
    ledger_path = live_dir / "requests.jsonl"
    ledger_events = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scopes = []
    directories = sorted(
        {
            path.parent
            for path in live_dir.rglob("request-*.json")
        }
    )
    for directory in directories:
        records = [
            read_json(path)
            for path in sorted(directory.glob("request-*.json"))
        ]
        condition = expected_condition(directory, records)
        audit = h1.audit_journal_records(
            records,
            directory,
            expected_condition=condition,
            ledger_events=ledger_events,
            offline=False,
        )
        scopes.append(
            {
                "journal_directory": str(directory.relative_to(live_dir)),
                "condition": condition,
                **audit,
            }
        )
    arm = arm_audit(live_dir)
    after = metadata_snapshot(live_dir)
    started = [
        row for row in ledger_events if row.get("event") == "request_started"
    ]
    committed = [
        row
        for row in ledger_events
        if row.get("event") == "request_usage_committed"
    ]
    failed = [
        row for row in ledger_events if row.get("event") == "request_failed"
    ]
    request_ids = [row.get("request_id") for row in started]
    model_counts = Counter(
        record.get("response_model_id")
        for directory in directories
        for record in (
            read_json(path)
            for path in sorted(directory.glob("request-*.json"))
        )
    )
    checks = {
        "all_scopes_complete": all(row["complete"] for row in scopes),
        "global_ledger_complete": (
            len(started) == len(committed) == 19 and not failed
        ),
        "request_ids_unique": (
            len(request_ids) == len(set(request_ids)) == 19
        ),
        "source_gateway_pass": arm["source_gateway_status"] == "PASS",
        "baseline_gateway_pass": arm["baseline_gateway_status"] == "PASS",
        "gateway_bypass_zero": (
            arm["source_gateway_bypasses"] == 0
            and arm["baseline_gateway_bypasses"] == 0
        ),
        "baseline_no_new_effect": (
            arm["baseline_trap_no_new_effect"] is True
        ),
        "baseline_restore_equal": arm["baseline_restore_equal"] is True,
        "baseline_k12": arm["baseline_k12"] is True,
        "baseline_task_success": arm["baseline_task_success"] is True,
        "baseline_no_technical_failure": (
            arm["baseline_technical_failure"] is False
        ),
        "model_id_exact": model_counts == Counter(
            {"claude-sonnet-4-6": 19}
        ),
        "original_batch_metadata_unchanged": before == after,
    }
    result = {
        "protocol_id": PROTOCOL_ID,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "provider_calls": 0,
        "network_attempts": 0,
        "source_batch": str(live_dir),
        "checks": checks,
        "scope_audits": scopes,
        "arm_audit": arm,
        "ledger": {
            "request_started": len(started),
            "usage_committed": len(committed),
            "failed": len(failed),
            "request_ids_unique": len(set(request_ids)),
        },
        "model_counts": dict(model_counts),
        "original_batch_metadata_unchanged": before == after,
        "original_file_count": len(before),
    }
    output.mkdir(mode=0o700, parents=True)
    write_json(output / "reaudit.json", result)
    lines = [
        "# Claude H1 Journal 编号只读重审",
        "",
        f"- 协议：`{PROTOCOL_ID}`",
        f"- 结论：`{result['status']}`",
        "- 真实 Provider Calls：0；网络访问：0。",
        f"- 原失败批次：`{live_dir.relative_to(ROOT)}`",
        "- 未修改、覆盖或重写原失败批次及其审计记录。",
        "",
        "| Scope | Condition | Numbering | Requests | Complete | Reasons |",
        "|---|---|---|---:|---|---|",
    ]
    for row in scopes:
        lines.append(
            f"| `{row['journal_directory']}` | {row['condition']} | "
            f"{row['numbering_mode']} | {row['request_count']} | "
            f"{row['complete']} | {','.join(row['reasons']) or 'NA'} |"
        )
    lines += [
        "",
        (f"- 全局 ledger：started={len(started)}，committed={len(committed)}，"
        f"failed={len(failed)}，唯一 Request ID={len(set(request_ids))}。"),
        ("- Source 1 与 BASELINE 的 Gateway、effect 边界、K=12、restore "
        "和任务成功检查均保留。"),
        f"- 原批次文件元数据未变化：`{before == after}`。",
        "",
    ]
    (output / "REAUDIT_REPORT_ZH.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dir", type=Path, default=FAILED_LIVE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or (
        ROOT / "results" / f"claude_h1_journal_reaudit_20261024_v1_{stamp}"
    )
    result = reaudit(args.live_dir.resolve(), output.resolve())
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

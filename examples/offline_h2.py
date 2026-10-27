"""Exercise the current five H2 strategies with local scripted responses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import tempfile
from unittest.mock import patch

from h1.runner import materialize_runtime, selected_sources
from tasks.extended_environment import (
    ExtendedEvidenceRepository,
)
from h2 import runtime as h2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = h2.load_config()
    representatives = []
    seen = set()
    for source in selected_sources():
        if source["family"] not in seen:
            seen.add(source["family"])
            representatives.append(source)

    with tempfile.TemporaryDirectory(prefix="h1h2-fixtures-") as directory:
        fixture_root = Path(directory)
        with patch.object(
            socket.socket, "connect", side_effect=AssertionError("offline example")
        ):
            for source in representatives:
                materialize_runtime(source, fixture_root)
            repository = ExtendedEvidenceRepository(fixture_root)
            rows = h2.full_state_smokes(config, representatives, repository)
    report = {
        "release_version": "github-core-v3",
        "provider_free": True,
        "mock_responses": True,
        "real_provider_requests": 0,
        "budget_per_trajectory": config["budgets"][0],
        "trajectory_count": len(rows),
        "expected_trajectory_count": 15,
        "rows": rows,
    }
    destination = output / "demo.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    passed = len(rows) == 15 and all(
        row["reached_states"] == list(h2.STATES) for row in rows
    )
    print(json.dumps({
        "passed": passed,
        "provider_free": True,
        "trajectories": len(rows),
        "output": str(destination),
    }, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

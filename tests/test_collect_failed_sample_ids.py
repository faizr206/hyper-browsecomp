from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_collect_failed_sample_ids_preserves_retained_order(tmp_path: Path) -> None:
    retained = tmp_path / "retained.txt"
    retained.write_text("q3\nq1\nq2\n", encoding="utf-8")
    run_a = tmp_path / "run-a" / "shard-0" / "traces"
    run_b = tmp_path / "run-b" / "shard-1" / "traces"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "q1.json").write_text(
        json.dumps({"sample_id": "q1", "status": "error"}), encoding="utf-8"
    )
    (run_a / "q2.json").write_text(
        json.dumps({"sample_id": "q2", "status": "completed"}), encoding="utf-8"
    )
    (run_b / "q3.json").write_text(
        json.dumps({"sample_id": "q3", "status": "error"}), encoding="utf-8"
    )
    (run_b / "q1-again.json").write_text(
        json.dumps({"sample_id": "q1", "status": "error"}), encoding="utf-8"
    )
    output = tmp_path / "failed.txt"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_failed_sample_ids.py",
            "--run-root",
            str(tmp_path / "run-a"),
            "--run-root",
            str(tmp_path / "run-b"),
            "--retained-ids",
            str(retained),
            "--output",
            str(output),
        ],
        check=True,
    )

    assert output.read_text(encoding="utf-8") == "q3\nq1\n"

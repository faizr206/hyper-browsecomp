#!/usr/bin/env python3
"""Collect errored OWL sample IDs from one or more SLURM run roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", type=Path, required=True)
    parser.add_argument("--retained-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def retained_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> int:
    args = parse_args()
    ordered_ids = retained_ids(args.retained_ids)
    known_ids = set(ordered_ids)
    failed: set[str] = set()

    for run_root in args.run_root:
        if not run_root.is_dir():
            raise SystemExit(f"SLURM run root does not exist: {run_root}")
        for trace_path in run_root.glob("shard-*/traces/*.json"):
            try:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"cannot read OWL trace {trace_path}: {exc}") from exc
            sample_id = str(trace.get("sample_id") or "")
            if trace.get("status") == "error":
                if sample_id not in known_ids:
                    raise SystemExit(
                        f"errored trace has an ID absent from {args.retained_ids}: "
                        f"{sample_id or '<missing>'} ({trace_path})"
                    )
                failed.add(sample_id)

    selected = [sample_id for sample_id in ordered_ids if sample_id in failed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{sample_id}\n" for sample_id in selected),
        encoding="utf-8",
    )
    print(f"collected {len(selected)} failed sample ID(s) into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

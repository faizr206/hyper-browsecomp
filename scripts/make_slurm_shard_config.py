#!/usr/bin/env python3
"""Create one persistent YAML config for a SLURM benchmark shard."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--sample-ids-source", type=Path)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--task-timeout", type=int)
    parser.add_argument("--attempt-timeout", type=int)
    parser.add_argument("--parallel", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start < 1 or args.end < args.start:
        raise SystemExit("sample bounds must satisfy 1 <= start <= end")

    payload = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"base config must contain a YAML mapping: {args.base}")

    if args.sample_ids_source is not None:
        all_ids = [
            line.strip()
            for line in args.sample_ids_source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        selected_ids = all_ids[args.start - 1 : args.end]
        expected_count = args.end - args.start + 1
        if len(selected_ids) != expected_count:
            raise SystemExit(
                f"requested retained-ID positions {args.start}-{args.end}, but "
                f"{args.sample_ids_source} contains only {len(all_ids)} IDs"
            )
        ids_output = args.output.parent / "sample_ids.txt"
        ids_output.parent.mkdir(parents=True, exist_ok=True)
        ids_output.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
        payload["sample_ids_path"] = str(ids_output)
        payload.pop("sample_range", None)
    else:
        # sample_range is 1-based and inclusive in the benchmark runner.
        payload["sample_range"] = f"{args.start}-{args.end}"
        payload.pop("sample_ids_path", None)
    payload.pop("start_index", None)
    payload.pop("end_index", None)
    payload.pop("num_samples", None)
    payload["log_dir"] = args.log_dir
    payload["owl_trace_dir"] = args.trace_dir

    if args.task_timeout is not None:
        payload["owl_task_timeout_seconds"] = args.task_timeout
    if args.attempt_timeout is not None:
        payload["inspect_attempt_timeout"] = args.attempt_timeout
    if args.parallel is not None:
        payload["inspect_max_samples_parallel"] = args.parallel

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

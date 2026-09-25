#!/usr/bin/env python3
"""Append a new OpenRouter judge score to existing Inspect ``.eval`` logs.

The source logs are never modified. Rescored logs are written to a mirrored
directory tree and contain both the original ``browse_comp_scorer`` result and
the new ``browse_comp_rescorer`` result.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.log import read_eval_log


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "logs"
DEFAULT_INPUT = LOG_ROOT / "owl_final"
DEFAULT_MODEL = "openrouter/openai/gpt-oss-20b"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CONFIDENTIAL_DATA = "afaji/HyperBrowseComp"
SCORER = "src/hyper_browsecomp/scorer.py@browse_comp_rescorer"


def parse_args(default_input: Path = DEFAULT_INPUT) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescore Inspect .eval logs with an OpenRouter model. Directories "
            "are searched recursively."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=f".eval files or directories (default: {default_input.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="mirrored output root (default: logs/rescored/<model-name>)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "OpenRouter slug or Inspect model path; for example "
            "moonshotai/kimi-k2-thinking or z-ai/glm-4.7 "
            "(default: openai/gpt-oss-20b)"
        ),
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--confidential-data-path",
        default=DEFAULT_CONFIDENTIAL_DATA,
        help=(
            "encrypted dataset used to restore redacted gold answers "
            "(default: afaji/HyperBrowseComp)"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="maximum concurrent grader requests within one log (default: 4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace already completed rescored outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the files and commands without calling OpenRouter",
    )
    return parser.parse_args()


def _resolved(path: Path) -> Path:
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def openrouter_model(model: str) -> str:
    """Convert an OpenRouter slug to the Inspect OpenRouter model path."""
    if model.startswith("openrouter/"):
        return model
    return f"openrouter/{model}"


def model_output_dir(model: str) -> Path:
    return LOG_ROOT / "rescored" / model.rsplit("/", 1)[-1]


def discover_eval_logs(inputs: list[Path], output_dir: Path) -> list[Path]:
    output_dir = output_dir.resolve()
    discovered: set[Path] = set()
    for raw_input in inputs:
        path = _resolved(raw_input)
        if not path.exists():
            raise FileNotFoundError(f"input does not exist: {path}")
        candidates = [path] if path.is_file() else path.rglob("*.eval")
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.suffix != ".eval":
                continue
            if candidate == output_dir or output_dir in candidate.parents:
                continue
            discovered.add(candidate)
    return sorted(discovered)


def output_path(source: Path, output_dir: Path) -> Path:
    try:
        relative = source.resolve().relative_to(LOG_ROOT.resolve())
    except ValueError:
        relative = Path(source.name)
    return output_dir.resolve() / relative


def inspect_executable() -> str:
    local = PROJECT_ROOT / ".venv" / "bin" / "inspect"
    if local.is_file():
        return str(local)
    executable = shutil.which("inspect")
    if executable:
        return executable
    raise FileNotFoundError("inspect executable not found; run `uv sync --extra dev` first")


def score_command(
    inspect_bin: str,
    source: Path,
    partial_output: Path,
    *,
    model: str,
    base_url: str,
    confidential_data_path: str,
    concurrency: int,
    stream: bool = True,
) -> list[str]:
    command = [
        inspect_bin,
        "score",
        str(source),
        "--model",
        model,
        "--model-base-url",
        base_url,
        "--scorer",
        SCORER,
        "-S",
        f"scorer_model={model}",
        "-S",
        f"confidential_data_path={confidential_data_path}",
        "--action",
        "append",
        "--output-file",
        str(partial_output),
        "--overwrite",
        "--display",
        "none",
    ]
    if stream:
        command[command.index("--output-file"):command.index("--output-file")] = [
            "--stream",
            str(concurrency),
        ]
    return command


def source_supports_streaming(source: Path) -> bool:
    """Inspect streaming requires the log's expected and stored samples to match."""
    return read_eval_log(source, header_only=True).status == "success"


def runtime_env(base_url: str) -> dict[str, str]:
    load_dotenv(PROJECT_ROOT / ".env")
    env = os.environ.copy()
    if not env.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is missing from the environment or .env")

    inspect_home = PROJECT_ROOT / ".inspect_home"
    inspect_home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(inspect_home)
    env["OPENROUTER_BASE_URL"] = base_url
    current_pythonpath = env.get("PYTHONPATH")
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{current_pythonpath}" if current_pythonpath else src_path
    )
    return env


def append_manifest(manifest: Path, source: Path, output: Path, model: str) -> None:
    record = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "source": str(source),
        "output": str(output),
    }
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main(default_input: Path = DEFAULT_INPUT) -> int:
    args = parse_args(default_input)
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    inputs = args.inputs or [default_input]
    model = openrouter_model(args.model)
    output_dir = _resolved(args.output_dir or model_output_dir(model))
    sources = discover_eval_logs(inputs, output_dir)
    if not sources:
        print("No .eval files found.", file=sys.stderr)
        return 1

    inspect_bin = inspect_executable()
    env = None if args.dry_run else runtime_env(args.base_url)
    manifest = output_dir / "manifest.jsonl"
    failures: list[Path] = []
    skipped = 0

    print(f"Found {len(sources)} .eval file(s).")
    print(f"Judge: {model}")
    print(f"Output: {output_dir}")

    for index, source in enumerate(sources, start=1):
        destination = output_path(source, output_dir)
        partial = destination.with_suffix(".partial.eval")
        if destination.exists() and not args.force:
            skipped += 1
            print(f"[{index}/{len(sources)}] skip (already exists): {destination}")
            continue

        command = score_command(
            inspect_bin,
            source,
            partial,
            model=model,
            base_url=args.base_url,
            confidential_data_path=args.confidential_data_path,
            concurrency=args.concurrency,
            stream=source_supports_streaming(source),
        )
        print(f"[{index}/{len(sources)}] {source}")
        if "--stream" not in command:
            print("  source is incomplete/cancelled; using non-streaming fallback")
        if args.dry_run:
            print(f"  -> {destination}")
            print(f"  {shlex.join(command)}")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
        if result.returncode != 0:
            failures.append(source)
            print(
                f"  FAILED (exit {result.returncode}); partial output kept at {partial}",
                file=sys.stderr,
            )
            continue

        partial.replace(destination)
        append_manifest(manifest, source, destination, model)
        print(f"  -> {destination}")

    completed = len(sources) - skipped - len(failures)
    print(f"Done: {completed} completed, {skipped} skipped, {len(failures)} failed.")
    if failures:
        print("Failed inputs:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

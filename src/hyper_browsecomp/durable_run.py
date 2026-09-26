"""Persistent one-command runs: retain raw attempts, publish one sanitized log."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from inspect_ai.log import read_eval_log

from hyper_browsecomp.config import RunConfig
from hyper_browsecomp.utils import sanitize_filename


def _paths(config: RunConfig) -> tuple[Path, Path]:
    from hyper_browsecomp.runner import PROJECT_ROOT

    output = (PROJECT_ROOT / config.log_dir).resolve()
    model = sanitize_filename(config.resolved_model().rsplit("/", 1)[-1])
    suffix = hashlib.sha256(str(output).encode()).hexdigest()[:10]
    work = PROJECT_ROOT / "dump" / "auto_resume" / f"{model}-{suffix}"
    return work, output / f"HyperBrowseComp_{model}.eval"


def _fingerprint(config: RunConfig, sample_ids: list[str]) -> str:
    # Credentials stay in environment variables, never in this manifest.
    payload = {"config": config.model_dump(mode="json"), "sample_ids": sample_ids}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _latest(attempts: Path) -> Path | None:
    logs = list(attempts.glob("*.eval"))
    return max(logs, key=lambda path: path.stat().st_mtime_ns) if logs else None


def _completed(log_path: Path, expected: list[str]) -> int:
    log = read_eval_log(log_path)
    saved = {
        str(sample.id) for sample in log.samples or []
        if sample.error is None and sample.completed_at is not None
        and sample.scores and sample.invalidation is None
    }
    unexpected = saved.difference(expected)
    if unexpected:
        raise ValueError("Saved run contains sample IDs outside the configured selection.")
    return len(saved)


def _publish(config: RunConfig, source: Path, canonical: Path, work: Path) -> None:
    from hyper_browsecomp.runner import sanitize_eval_log

    canonical.parent.mkdir(parents=True, exist_ok=True)
    # Stage outside logs/ so Inspect View sees only the complete publication.
    staged = work / "publishing.eval"
    shutil.copy2(source, staged)
    sanitize_eval_log(config, staged)
    read_eval_log(staged, header_only=True, format="eval")
    if canonical.exists():
        history = work / "published"
        history.mkdir(exist_ok=True)
        canonical.rename(history / f"{time.time_ns()}_{canonical.name}")
    staged.replace(canonical)


def _retry_command(config: RunConfig, source: Path) -> list[str]:
    command = ["inspect", "eval-retry", str(source), "--log-buffer",
               str(config.inspect_log_buffer or 1), "--max-samples",
               str(config.inspect_max_samples_parallel)]
    if config.inspect_checkpoint:
        command.extend(["--checkpoint", config.inspect_checkpoint])
    return command


def _recover(source: Path, env: dict[str, str], lock_fd: int) -> Path:
    if read_eval_log(source, header_only=True).status != "started":
        return source
    # Recover explicitly rather than letting eval_retry remove its temporary
    # recovery file after success. Original attempts and buffers stay intact.
    script = (
        "import sys\n"
        "from inspect_ai.log import recover_eval_log\n"
        "from inspect_ai.log._recover import RecoveryNotAvailable\n"
        "try:\n"
        "    recover_eval_log(sys.argv[1], cleanup=False)\n"
        "except RecoveryNotAvailable:\n"
        "    pass\n"
    )
    subprocess.run([sys.executable, "-c", script, str(source)], env=env,
                   pass_fds=(lock_fd,), check=True)
    recovered = source.with_name(f"{source.stem}-recovered.eval")
    return recovered if recovered.exists() else source


def run_durable(config: RunConfig) -> int:
    from hyper_browsecomp.runner import (
        PROJECT_ROOT, build_inspect_command, prepare_env, selected_sample_ids,
    )

    work, canonical = _paths(config)
    attempts = work / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    # Hold the same lock in the worker too: killing only this wrapper must not
    # permit a second process to start while the original eval is still alive.
    with (work / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This evaluation is already running; no second run started.") from exc

        sample_ids = selected_sample_ids(config)
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("A durable run requires nonempty, unique sample IDs.")
        fingerprint = _fingerprint(config, sample_ids)
        manifest_path = work / "run.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["fingerprint"] != fingerprint:
                raise ValueError(
                    "Config or selected question IDs changed since this run began. "
                    "Restore the original config/selection to resume; use a new log_dir for a new run."
                )
        else:
            if canonical.exists() or _latest(attempts):
                raise ValueError("Existing results have no resume manifest; refusing to start a duplicate run.")
            manifest = {"fingerprint": fingerprint, "model": config.resolved_model(),
                        "sample_ids": sample_ids, "canonical": str(canonical)}
            temporary = work / "run.json.tmp"
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.replace(manifest_path)

        env = prepare_env(config)
        env["INSPECT_LOG_DIR"] = str(attempts)
        # The same stable HOME from prepare_env is used on every launch, so
        # Inspect can find the prior process's on-disk sample buffer database.
        previous = _latest(attempts)
        if previous is not None:
            previous = _recover(previous, env, lock.fileno())
            count = _completed(previous, sample_ids)
            _publish(config, previous, canonical, work)
            if count == len(sample_ids):
                print(f"Already complete: {count}/{len(sample_ids)}; {canonical}", flush=True)
                return 0
            print(f"Resuming {count}/{len(sample_ids)} saved questions from {previous}", flush=True)
            command = _retry_command(config, previous)
        else:
            print(f"Starting {len(sample_ids)} questions; resume by repeating the same command.", flush=True)
            command = build_inspect_command(config)
            if config.inspect_log_buffer is None:
                command.extend(["--log-buffer", "1"])

        if config.resolved_model().startswith("minimax/"):
            # eval-retry reconstructs the model before loading the task module;
            # register our custom provider in the worker before CLI dispatch.
            bootstrap = (
                "import hyper_browsecomp.minimax\n"
                "from inspect_ai._cli.main import main\n"
                "main()\n"
            )
            command = [sys.executable, "-c", bootstrap, *command[1:]]

        print(f"Raw attempts: {attempts}\nCanonical result: {canonical}", flush=True)
        child = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, pass_fds=(lock.fileno(),))
        try:
            returncode = child.wait()
        except KeyboardInterrupt:
            print("Stopping safely; waiting for Inspect to save progress…", flush=True)
            # Interactive Ctrl+C reaches both processes in their shared
            # foreground group. A second SIGINT would interrupt Inspect's
            # first-signal save/cleanup path, so simply let it finish.
            returncode = child.wait()
        latest = _latest(attempts)
        if latest is not None:
            _publish(config, latest, canonical, work)
            count = _completed(latest, sample_ids)
            print(f"Saved {count}/{len(sample_ids)} questions: {canonical}", flush=True)
            if returncode == 0 and count != len(sample_ids):
                print("Some questions remain unfinished. Repeat the same command to resume.", flush=True)
                return 1
        return returncode

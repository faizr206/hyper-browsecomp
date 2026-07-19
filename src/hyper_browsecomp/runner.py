from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.log import read_eval_log

from hyper_browsecomp.config import RunConfig, load_run_config
from hyper_browsecomp.utils import sanitize_filename


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _provider_env_name(provider: str, suffix: str) -> str:
    return f"{provider.upper().replace('-', '_')}_{suffix}"


def _apply_provider_env(
    env: dict[str, str],
    *,
    provider: str | None,
    api_key: str | None,
    base_url: str | None,
) -> None:
    if not provider:
        return
    if api_key:
        env[_provider_env_name(provider, "API_KEY")] = api_key
    if base_url:
        env[_provider_env_name(provider, "BASE_URL")] = base_url


def _model_provider_from_string(model: str | None) -> str | None:
    if not model:
        return None
    parts = model.split("/", 2)
    if len(parts) == 3 and parts[0] == "openai-api":
        return parts[1]
    return None


def _main_provider(config: RunConfig) -> str | None:
    return config.provider or _model_provider_from_string(config.model)


def _scorer_provider(config: RunConfig) -> str | None:
    return config.scorer_provider or _model_provider_from_string(config.scorer_model)


def prepare_env(config: RunConfig) -> dict[str, str]:
    env = os.environ.copy()

    main_provider = _main_provider(config)
    scorer_provider = _scorer_provider(config)

    _apply_provider_env(
        env,
        provider=main_provider,
        api_key=env.get("MODEL_API_KEY"),
        base_url=env.get("MODEL_BASE_URL"),
    )

    if scorer_provider:
        _apply_provider_env(
            env,
            provider=scorer_provider,
            api_key=env.get("SCORER_API_KEY") or env.get("MODEL_API_KEY"),
            base_url=env.get("SCORER_BASE_URL") or env.get("MODEL_BASE_URL"),
        )

    inspect_home = (PROJECT_ROOT / ".inspect_home").resolve()
    inspect_home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(inspect_home)
    env["INSPECT_LOG_DIR"] = str((PROJECT_ROOT / config.log_dir).resolve())
    return env


def build_inspect_command(config: RunConfig) -> list[str]:
    command = [
        "inspect",
        "eval",
        "src/hyper_browsecomp/task.py@hyper_browsecomp",
        "--model",
        config.resolved_model(),
    ]

    if config.resolved_model().startswith("openai-api/"):
        command.extend(["-M", "strict_tools=false"])

    command.extend(["--max-samples", str(config.inspect_max_samples_parallel)])

    if config.inspect_model_max_retries is not None:
        command.extend(["--max-retries", str(config.inspect_model_max_retries)])
    if config.inspect_attempt_timeout is not None:
        command.extend(["--attempt-timeout", str(config.inspect_attempt_timeout)])
    if config.inspect_retry_on_error is not None:
        command.extend(["--retry-on-error", str(config.inspect_retry_on_error)])
    if config.inspect_no_fail_on_error:
        command.append("--no-fail-on-error")
    if config.inspect_continue_on_fail:
        command.append("--continue-on-fail")

    task_args = {
        "data_path": config.data_path,
        "tool_profile": config.tool_profile,
        "search_backend": config.search_backend,
        "fetch_backend": config.fetch_backend,
        "search_max_results": config.search_max_results,
        "search_timeout_seconds": config.search_timeout_seconds,
        "fetch_timeout_seconds": config.fetch_timeout_seconds,
        "fetch_max_chars": config.fetch_max_chars,
        "max_steps": config.max_steps,
        "bash_timeout": config.bash_timeout,
        "python_timeout": config.python_timeout,
        "no_sandbox": str(config.no_sandbox).lower(),
    }

    if config.sample_range is not None:
        task_args["sample_range"] = config.sample_range
    if config.start_index is not None:
        task_args["start_index"] = config.start_index
    if config.end_index is not None:
        task_args["end_index"] = config.end_index
    if config.num_samples is not None:
        task_args["num_samples"] = config.num_samples
    if config.resolved_scorer_model() is not None:
        task_args["scorer_model"] = config.resolved_scorer_model()

    for key, value in task_args.items():
        command.extend(["-T", f"{key}={value}"])

    return command


def _collect_eval_logs(log_dir: Path) -> set[Path]:
    if not log_dir.exists():
        return set()
    return {path.resolve() for path in log_dir.rglob("*.eval")}


def _format_timestamp(created: str | None, fallback: datetime) -> str:
    if created:
        try:
            parsed = datetime.fromisoformat(created.replace("Z", "+00:00"))
            return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        except ValueError:
            pass
    return fallback.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rename_new_eval_log(config: RunConfig, *, before: set[Path], started_at: datetime) -> Path | None:
    log_dir = (PROJECT_ROOT / config.log_dir).resolve()
    after = _collect_eval_logs(log_dir)
    new_logs = sorted(after - before)
    if not new_logs:
        return None

    newest = max(new_logs, key=lambda path: path.stat().st_mtime)
    log = read_eval_log(newest)
    data_name = sanitize_filename(Path(config.data_path).stem)
    model_leaf = sanitize_filename(config.resolved_model().rsplit("/", 1)[-1])
    timestamp = _format_timestamp(getattr(log.eval, "created", None), started_at)
    run_id = sanitize_filename(getattr(log.eval, "task_id", newest.stem.split("_")[-1]))
    target = newest.with_name(f"{data_name}_{model_leaf}_{timestamp}_{run_id}.eval")
    if target == newest:
        return newest
    newest.rename(target)
    return target


def run_with_config(config: RunConfig) -> int:
    env = prepare_env(config)
    log_dir = Path(env["INSPECT_LOG_DIR"])
    before = _collect_eval_logs(log_dir)
    command = build_inspect_command(config)
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    renamed = rename_new_eval_log(config, before=before, started_at=started_at)
    if renamed is not None:
        print(f"renamed eval log: {renamed}")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m hyper_browsecomp.runner CONFIG.yaml", file=sys.stderr)
        return 2
    load_dotenv(PROJECT_ROOT / ".env")
    config = load_run_config(args[0])
    return run_with_config(config)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.log import read_eval_log, write_eval_log
from pydantic import BaseModel

from hyper_browsecomp.config import RunConfig, load_run_config
from hyper_browsecomp.dataset import (
    confidential_answer_replacements,
    confidential_log_replacements,
    load_browsecomp_dataset,
)
from hyper_browsecomp.task import slice_dataset
from hyper_browsecomp.utils import sanitize_filename


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NATIVE_PROVIDER_ENV_PREFIXES = {
    "gemini": "GOOGLE",
    "google": "GOOGLE",
    "grok": "XAI",
}
NATIVE_MODEL_PROVIDER_ALIASES = {
    "gemini": "google",
}


def _provider_env_name(provider: str, suffix: str) -> str:
    provider = NATIVE_PROVIDER_ENV_PREFIXES.get(provider, provider)
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
    if len(parts) >= 2:
        provider = parts[0]
        for alias, native_provider in NATIVE_MODEL_PROVIDER_ALIASES.items():
            if provider == native_provider:
                return alias
        return provider
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
        api_key=env.get(config.model_api_key_env),
        base_url=config.model_base_url,
    )

    if scorer_provider:
        _apply_provider_env(
            env,
            provider=scorer_provider,
            api_key=env.get(config.scorer_api_key_env),
            base_url=config.scorer_base_url,
        )

    inspect_home = (PROJECT_ROOT / ".inspect_home").resolve()
    inspect_home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(inspect_home)
    env["INSPECT_LOG_DIR"] = str((PROJECT_ROOT / config.log_dir).resolve())
    return env


def build_inspect_command(config: RunConfig, *, sample_ids: list[str] | None = None) -> list[str]:
    command = [
        "inspect",
        "eval",
        "src/hyper_browsecomp/task.py@hyper_browsecomp",
        "--model",
        config.resolved_model(),
    ]

    if config.resolved_model().startswith("openai-api/"):
        command.extend(["-M", "strict_tools=false"])

    for key, value in config.model_args.items():
        command.extend(["-M", f"{key}={json.dumps(value, separators=(',', ':'))}"])

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
    if not config.inspect_ctl_server:
        command.extend(["--ctl-server", "false"])
    if sample_ids:
        if any("," in sample_id for sample_id in sample_ids):
            raise ValueError("sample IDs cannot contain commas when passed to inspect --sample-id.")
        command.extend(["--sample-id", ",".join(sample_ids)])

    task_args: dict[str, object] = {
        "data_path": config.data_path,
        "harness": config.harness,
        "no_sandbox": str(config.no_sandbox).lower(),
    }

    if config.harness == "owl":
        task_args.update(
            {
                "owl_model_name": config.resolved_owl_model_name(),
                "owl_api_key_env": config.resolved_owl_api_key_env(),
                "owl_base_url": config.resolved_owl_base_url(),
                "owl_headless": str(config.owl_headless).lower(),
                "owl_multimodal": str(config.owl_multimodal).lower(),
                "owl_browser_round_limit": config.owl_browser_round_limit,
                "owl_task_timeout_seconds": config.owl_task_timeout_seconds,
                "owl_finalize_reserve_seconds": config.owl_finalize_reserve_seconds,
                "owl_max_external_tool_calls": config.owl_max_external_tool_calls,
                "owl_max_model_calls": config.owl_max_model_calls,
                "owl_model_max_retries": config.owl_model_max_retries,
                "owl_max_tokens": config.owl_max_tokens,
                "owl_trace_dir": config.owl_trace_dir,
            }
        )
    else:
        task_args.update(
            {
                "tool_profile": config.tool_profile,
                "search_backend": config.search_backend,
                "fetch_backend": config.fetch_backend,
                "model_provider": config.provider
                or _model_provider_from_string(config.model),
                "search_max_results": config.search_max_results,
                "search_timeout_seconds": config.search_timeout_seconds,
                "fetch_timeout_seconds": config.fetch_timeout_seconds,
                "fetch_max_chars": config.fetch_max_chars,
                "bash_timeout": config.bash_timeout,
                "python_timeout": config.python_timeout,
                "max_steps": config.max_steps,
            }
        )

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


def _owl_trace_directory(config: RunConfig) -> Path:
    directory = Path(config.owl_trace_dir)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    return directory.resolve()


def _collect_owl_traces(config: RunConfig) -> set[Path]:
    if config.harness != "owl":
        return set()
    directory = _owl_trace_directory(config)
    if not directory.exists():
        return set()
    return {path.resolve() for path in directory.glob("*.json")}


def finalize_unfinished_owl_traces(config: RunConfig, *, before: set[Path]) -> list[Path]:
    """Mark sidecars left running when Inspect ends before solver cleanup."""
    finalized: list[Path] = []
    for path in sorted(_collect_owl_traces(config) - before):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("status") != "running":
            continue
        payload.update(
            {
                "status": "error",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "termination_reason": "evaluation_process_ended",
                "error": (
                    "Inspect ended before the OWL solver could finalize this trace; "
                    "see the console log for the retained live trajectory."
                ),
                "statistics": payload.get("statistics") or {},
                "tool_usage": payload.get("tool_usage") or {},
                "model_calls": payload.get("model_calls") or [],
                "media_events": payload.get("media_events") or [],
                "workforce_events": payload.get("workforce_events") or [],
            }
        )
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        finalized.append(path)
    return finalized


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


def _redact_text(
    text: str,
    replacements: list[tuple[str, str]],
    answer_replacements: list[tuple[str, str]] | None = None,
) -> str:
    for secret, replacement in replacements:
        text = text.replace(secret, replacement)
    for secret, replacement in answer_replacements or []:
        text = re.sub(
            rf"(?m)^(\[correct_answer\]:\s*){re.escape(secret)}(\s*)$",
            rf"\1{replacement}\2",
            text,
        )
    return text


def _redact_value(
    value: object,
    replacements: list[tuple[str, str]],
    seen: set[int],
    answer_replacements: list[tuple[str, str]] | None = None,
) -> object:
    if isinstance(value, str):
        return _redact_text(value, replacements, answer_replacements)
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _redact_value(item, replacements, seen, answer_replacements)
        return value
    if isinstance(value, tuple):
        return tuple(_redact_value(item, replacements, seen, answer_replacements) for item in value)
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _redact_value(item, replacements, seen, answer_replacements)
        return value
    if isinstance(value, BaseModel):
        identity = id(value)
        if identity in seen:
            return value
        seen.add(identity)
        for field_name in value.__class__.model_fields:
            try:
                current = getattr(value, field_name)
                redacted = _redact_value(current, replacements, seen, answer_replacements)
                if redacted is not current:
                    setattr(value, field_name, redacted)
            except Exception:
                continue
        return value
    return value


def _hide_score_explanation(score: object) -> None:
    if hasattr(score, "explanation"):
        score.explanation = "Grader reasoning hidden for confidential sample."
    elif isinstance(score, dict) and "explanation" in score:
        score["explanation"] = "Grader reasoning hidden for confidential sample."


def _hide_confidential_score_explanations(log: BaseModel) -> None:
    for sample in getattr(log, "samples", None) or []:
        metadata = getattr(sample, "metadata", {}) or {}
        if not metadata.get("confidential"):
            continue
        for score in (getattr(sample, "scores", None) or {}).values():
            _hide_score_explanation(score)
        for event in getattr(sample, "events", None) or []:
            _hide_score_explanation(getattr(event, "score", None))


def sanitize_eval_log(config: RunConfig, log_path: str | Path) -> bool:
    slice_dataset(
        load_browsecomp_dataset(config.data_path),
        sample_range=config.sample_range,
        start_index=config.start_index,
        end_index=config.end_index,
        num_samples=config.num_samples,
    )
    replacements = confidential_log_replacements()
    answer_replacements = confidential_answer_replacements()
    if not replacements and not answer_replacements:
        return False

    log = read_eval_log(log_path)
    _redact_value(log, replacements, set(), answer_replacements)
    _hide_confidential_score_explanations(log)
    write_eval_log(log, log_path)
    return True


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _config_for_log_selection(config: RunConfig, log_path: str | Path) -> RunConfig:
    log = read_eval_log(log_path, header_only=True)
    task_args = getattr(log.eval, "task_args", {}) or {}
    updates: dict[str, object] = {}
    for key in ("data_path", "sample_range"):
        if key in task_args:
            updates[key] = task_args[key]
    for key in ("start_index", "end_index", "num_samples"):
        if key in task_args:
            updates[key] = _optional_int(task_args[key])
    if "sample_range" in updates:
        updates.setdefault("start_index", None)
        updates.setdefault("end_index", None)
        updates.setdefault("num_samples", None)
    elif any(key in updates for key in ("start_index", "end_index", "num_samples")):
        updates["sample_range"] = None

    if not updates:
        return config
    data = config.model_dump()
    data.update(updates)
    return RunConfig.model_validate(data)


def selected_sample_ids(config: RunConfig) -> list[str]:
    samples = slice_dataset(
        load_browsecomp_dataset(config.data_path),
        sample_range=config.sample_range,
        start_index=config.start_index,
        end_index=config.end_index,
        num_samples=config.num_samples,
    )
    sample_ids = [getattr(sample, "id", None) for sample in samples]
    missing_ids = [index for index, sample_id in enumerate(sample_ids, start=1) if sample_id is None]
    if missing_ids:
        raise ValueError(f"Dataset samples are missing IDs at selected positions: {missing_ids}")
    return [str(sample_id) for sample_id in sample_ids]


def _log_sample_id(sample, expected: set[str]) -> str | None:
    sample_id = getattr(sample, "id", None)
    if sample_id is not None and str(sample_id) in expected:
        return str(sample_id)
    metadata = getattr(sample, "metadata", {}) or {}
    metadata_id = metadata.get("id")
    if metadata_id is not None and str(metadata_id) in expected:
        return str(metadata_id)
    return str(sample_id) if sample_id is not None else None


def unfinished_sample_ids(config: RunConfig, log_path: str | Path) -> list[str]:
    expected_ids = selected_sample_ids(_config_for_log_selection(config, log_path))
    expected = set(expected_ids)
    log = read_eval_log(log_path)
    finished: set[str] = set()
    retry: set[str] = set()

    for sample in log.samples or []:
        sample_id = _log_sample_id(sample, expected)
        if sample_id is None or sample_id not in expected:
            continue
        if getattr(sample, "error", None) is not None or getattr(sample, "completed_at", None) is None:
            retry.add(sample_id)
        else:
            finished.add(sample_id)

    return [sample_id for sample_id in expected_ids if sample_id not in finished or sample_id in retry]


def run_with_config(config: RunConfig) -> int:
    env = prepare_env(config)
    log_dir = Path(env["INSPECT_LOG_DIR"])
    before = _collect_eval_logs(log_dir)
    owl_traces_before = _collect_owl_traces(config)
    command = build_inspect_command(config)
    started_at = datetime.now(timezone.utc)
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    finally:
        for trace in finalize_unfinished_owl_traces(config, before=owl_traces_before):
            print(f"finalized unfinished OWL trace: {trace}")
    renamed = rename_new_eval_log(config, before=before, started_at=started_at)
    if renamed is not None:
        if sanitize_eval_log(config, renamed):
            print(f"sanitized eval log: {renamed}")
        print(f"renamed eval log: {renamed}")
    return result.returncode


def resume_with_config(config: RunConfig, log_path: str | Path) -> int:
    resume_config = _config_for_log_selection(config, log_path)
    if sanitize_eval_log(resume_config, log_path):
        print(f"sanitized eval log: {log_path}")

    retry_ids = unfinished_sample_ids(config, log_path)
    if not retry_ids:
        print(f"no unfinished samples found in eval log: {log_path}")
        return 0

    env = prepare_env(resume_config)
    log_dir = Path(env["INSPECT_LOG_DIR"])
    before = _collect_eval_logs(log_dir)
    owl_traces_before = _collect_owl_traces(resume_config)
    command = build_inspect_command(resume_config, sample_ids=retry_ids)
    print(f"resuming {len(retry_ids)} unfinished sample(s): {', '.join(retry_ids)}")
    started_at = datetime.now(timezone.utc)
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    finally:
        for trace in finalize_unfinished_owl_traces(
            resume_config, before=owl_traces_before
        ):
            print(f"finalized unfinished OWL trace: {trace}")
    renamed = rename_new_eval_log(resume_config, before=before, started_at=started_at)
    if renamed is not None:
        if sanitize_eval_log(resume_config, renamed):
            print(f"sanitized eval log: {renamed}")
        print(f"renamed eval log: {renamed}")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) == 1:
        load_dotenv(PROJECT_ROOT / ".env")
        config = load_run_config(args[0])
        return run_with_config(config)
    if len(args) == 3 and args[0] == "resume":
        load_dotenv(PROJECT_ROOT / ".env")
        config = load_run_config(args[1])
        return resume_with_config(config, args[2])

    print(
        "usage: python -m hyper_browsecomp.runner CONFIG.yaml\n"
        "       python -m hyper_browsecomp.runner resume CONFIG.yaml LOG.eval",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

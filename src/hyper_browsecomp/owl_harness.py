from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OWL_RUNTIME_ROOT = PROJECT_ROOT / "owl_runtime"
OWL_WORKER = OWL_RUNTIME_ROOT / "worker.py"
OWL_RESULT_PREFIX = "HYPER_BROWSECOMP_OWL_RESULT="
OWL_PROGRESS_PREFIX = "HYPER_BROWSECOMP_OWL_PROGRESS="


def build_owl_command(*, uv_executable: str | None = None) -> list[str]:
    uv = uv_executable or shutil.which("uv")
    if uv is None:
        raise RuntimeError("OWL harness requires the `uv` executable on PATH.")
    if not OWL_WORKER.exists():
        raise RuntimeError(f"OWL worker is missing: {OWL_WORKER}")
    return [
        uv,
        "run",
        "--project",
        str(OWL_RUNTIME_ROOT),
        "--frozen",
        "python",
        "-u",
        str(OWL_WORKER),
    ]


def parse_owl_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(OWL_RESULT_PREFIX):
            payload = json.loads(line.removeprefix(OWL_RESULT_PREFIX))
            if not isinstance(payload, dict):
                break
            return payload
    raise RuntimeError("OWL worker did not emit a structured result.")


def format_owl_task_prompt(question: str, image_urls: list[str] | None = None) -> str:
    prompt = question.strip()
    if image_urls:
        prompt += "\n\nInput image URLs (inspect these images visually):\n"
        prompt += "\n".join(f"- {url}" for url in image_urls)
    return prompt


def normalize_owl_completion(completion: str) -> str:
    matches = list(
        re.finditer(
            r"(?ms)^Explanation:\s*.*?^Exact Answer:\s*.*?^Confidence:\s*[^\n]+",
            completion.strip(),
        )
    )
    if matches:
        return matches[-1].group(0).strip()
    return completion.strip()


def _new_trace_files(
    trace_dir: str | Path, sample_id: str, model_name: str
) -> dict[str, str]:
    directory = Path(trace_dir)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)

    safe_sample_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    if not safe_sample_id:
        safe_sample_id = "unknown"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"{timestamp}_{safe_sample_id}"
    trace_files = {
        "console": str((directory / f"{stem}.log").resolve()),
        "events": str((directory / f"{stem}.json").resolve()),
    }
    Path(trace_files["console"]).write_text("", encoding="utf-8")
    Path(trace_files["events"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "sample_id": sample_id,
                "model_name": model_name,
                "console": trace_files["console"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return trace_files


def _sanitize_console_line(line: str) -> str:
    line = re.sub(
        r"data:(image|audio|video)/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
        lambda match: f"[INLINE_{match.group(1).upper()}_BASE64_REDACTED]",
        line,
    )
    # input_audio uses a plain base64 data field, not a data: URI.
    return re.sub(
        r"([\"']data[\"']\s*:\s*[\"'])[A-Za-z0-9+/=]{128,}([\"'])",
        r"\1[INLINE_MEDIA_BASE64_REDACTED]\2",
        line,
    )


def _write_live_progress(path: str | Path, progress: dict[str, Any]) -> None:
    """Refresh a running trace sidecar without exposing worker marker lines."""
    events_path = Path(path)
    try:
        payload = json.loads(events_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict) or payload.get("status") != "running":
        return
    payload.update(progress)
    payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
    temporary = events_path.with_suffix(events_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(events_path)


def write_owl_trace(
    *,
    trace_dir: str | Path,
    sample_id: str,
    model_name: str,
    stdout: str,
    stderr: str,
    returncode: int | None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    trace_files: dict[str, str] | None = None,
    write_console: bool = True,
) -> dict[str, str]:
    trace_files = trace_files or _new_trace_files(trace_dir, sample_id, model_name)
    console_path = Path(trace_files["console"])
    events_path = Path(trace_files["events"])

    if write_console:
        console_lines = [
            _sanitize_console_line(line)
            for line in stdout.splitlines()
            if not line.startswith((OWL_RESULT_PREFIX, OWL_PROGRESS_PREFIX))
        ]
        console_text = "\n".join(console_lines).strip()
        if stderr.strip():
            console_text += f"\n\n=== STDERR ===\n{_sanitize_console_line(stderr.strip())}"
        console_path.write_text(console_text.strip() + "\n", encoding="utf-8")

    structured_result = dict(result or {})
    trace_payload = {
        "schema_version": 1,
        "status": "error" if error else "completed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample_id,
        "model_name": model_name,
        "process_returncode": returncode,
        "error": error,
        "termination_reason": structured_result.pop("termination_reason", None),
        "completion": structured_result.pop("completion", None),
        "capabilities": structured_result.pop("capabilities", {}),
        "tool_usage": structured_result.pop("tool_usage", {}),
        "statistics": structured_result.pop("statistics", {}),
        "model_calls": structured_result.pop("model_calls", []),
        "media_events": structured_result.pop("media_events", []),
        "workforce_events": structured_result.pop("workforce_events", []),
        "additional_result": structured_result,
        "console": str(console_path),
    }
    events_path.write_text(
        json.dumps(trace_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return trace_files


async def _pump_owl_stream(
    stream: asyncio.StreamReader,
    raw_chunks: list[bytes],
    trace_handle,
    *,
    stderr: bool,
    progress_state: dict[str, Any] | None = None,
    progress_path: str | Path | None = None,
) -> None:
    while True:
        line = await stream.readline()
        if not line:
            return
        decoded = line.decode("utf-8", errors="replace")
        if not stderr and decoded.startswith(OWL_PROGRESS_PREFIX):
            if progress_state is not None:
                try:
                    progress = json.loads(decoded.removeprefix(OWL_PROGRESS_PREFIX))
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(progress, dict):
                        progress_state.update(progress)
                        if progress_path is not None:
                            _write_live_progress(progress_path, progress_state)
            continue
        raw_chunks.append(line)
        if not stderr and decoded.startswith(OWL_RESULT_PREFIX):
            continue
        if stderr:
            decoded = f"[stderr] {decoded}"
        trace_handle.write(_sanitize_console_line(decoded))
        trace_handle.flush()


async def run_owl_harness(
    question: str,
    *,
    model_name: str,
    api_key_env: str,
    base_url: str,
    headless: bool,
    multimodal: bool,
    browser_round_limit: int,
    task_timeout_seconds: int,
    timeout_scale: float = 1.0,
    finalize_reserve_seconds: int = 120,
    max_external_tool_calls: int = 50,
    max_model_calls: int = 180,
    model_max_retries: int = 1,
    max_tokens: int = 8192,
    reasoning_effort: str | None = None,
    image_urls: list[str] | None = None,
    sample_id: str = "unknown",
    trace_dir: str | Path = "logs/owl/traces",
    tool_check: str | None = None,
    tool_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not os.environ.get(api_key_env):
        raise RuntimeError(f"OWL model API key environment variable is not set: {api_key_env}")

    payload = {
        "question": format_owl_task_prompt(question, image_urls),
        "model_name": model_name,
        "api_key_env": api_key_env,
        "base_url": base_url,
        "headless": headless,
        "multimodal": multimodal,
        "browser_round_limit": browser_round_limit,
        "task_timeout_seconds": task_timeout_seconds,
        "timeout_scale": timeout_scale,
        "finalize_reserve_seconds": finalize_reserve_seconds,
        "max_external_tool_calls": max_external_tool_calls,
        "max_model_calls": max_model_calls,
        "model_max_retries": model_max_retries,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }

    trace_files = _new_trace_files(trace_dir, sample_id, model_name)
    payload["media_artifact_dir"] = str(Path(trace_files["console"]).with_suffix(".media"))
    if tool_check:
        payload["tool_check"] = tool_check
        payload["tool_args"] = tool_args or {}
    print(f"OWL live trace: {trace_files['console']}", flush=True)
    process = await asyncio.create_subprocess_exec(
        *build_owl_command(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=PROJECT_ROOT,
        limit=32 * 1024 * 1024,
        start_new_session=os.name == "posix",
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    process.stdin.write(json.dumps(payload).encode("utf-8"))
    await process.stdin.drain()
    process.stdin.close()
    await process.stdin.wait_closed()

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    progress_state: dict[str, Any] = {}
    trace_handle = Path(trace_files["console"]).open("a", encoding="utf-8")
    stdout_task = asyncio.create_task(
        _pump_owl_stream(
            process.stdout,
            stdout_chunks,
            trace_handle,
            stderr=False,
            progress_state=progress_state,
            progress_path=trace_files["events"],
        )
    )
    stderr_task = asyncio.create_task(
        _pump_owl_stream(process.stderr, stderr_chunks, trace_handle, stderr=True)
    )

    async def stop_worker_tree() -> None:
        # The worker launches Playwright/browser descendants. Killing only the
        # `uv` process can leave those descendants holding stdout/stderr open,
        # which made timeout cleanup wait forever. The worker owns a dedicated
        # process group, so terminate that group and bound every cleanup wait.
        if process.returncode is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=5)
        except TimeoutError:
            pass
        for pump_task in (stdout_task, stderr_task):
            if not pump_task.done():
                pump_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    try:
        await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task, process.wait()),
            timeout=task_timeout_seconds,
        )
    except TimeoutError:
        await stop_worker_tree()
        trace_handle.close()
        stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        partial_result = dict(progress_state)
        partial_result["termination_reason"] = "wall_time_limit"
        statistics = dict(partial_result.get("statistics", {}))
        statistics["wall_time_seconds"] = float(task_timeout_seconds)
        partial_result["statistics"] = statistics
        write_owl_trace(
            trace_dir=trace_dir,
            sample_id=sample_id,
            model_name=model_name,
            stdout=stdout,
            stderr=stderr_text,
            returncode=process.returncode,
            result=partial_result,
            error=f"Task exceeded {task_timeout_seconds} seconds.",
            trace_files=trace_files,
            write_console=False,
        )
        raise RuntimeError(
            f"OWL worker exceeded its {task_timeout_seconds}-second task timeout. "
            f"Trace: {trace_files['events']}"
        ) from None
    except asyncio.CancelledError:
        await stop_worker_tree()
        trace_handle.close()
        write_owl_trace(
            trace_dir=trace_dir,
            sample_id=sample_id,
            model_name=model_name,
            stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
            returncode=process.returncode,
            result={**progress_state, "termination_reason": "cancelled"},
            error="OWL task was cancelled.",
            trace_files=trace_files,
            write_console=False,
        )
        raise
    finally:
        if not trace_handle.closed:
            trace_handle.close()

    stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    if process.returncode != 0:
        detail = (stderr or stdout).strip()[-4000:]
        partial_result = dict(progress_state)
        partial_result["termination_reason"] = "worker_process_error"
        write_owl_trace(
            trace_dir=trace_dir,
            sample_id=sample_id,
            model_name=model_name,
            stdout=stdout,
            stderr=stderr,
            returncode=process.returncode,
            result=partial_result,
            error=detail,
            trace_files=trace_files,
            write_console=False,
        )
        raise RuntimeError(
            f"OWL worker failed with exit code {process.returncode}: {detail} "
            f"Trace: {trace_files['events']}"
        )

    try:
        result = parse_owl_result(stdout)
    except Exception as exc:
        write_owl_trace(
            trace_dir=trace_dir,
            sample_id=sample_id,
            model_name=model_name,
            stdout=stdout,
            stderr=stderr,
            returncode=process.returncode,
            error=str(exc),
            trace_files=trace_files,
            write_console=False,
        )
        raise RuntimeError(f"{exc} Trace: {trace_files['events']}") from exc
    completion = result.get("completion")
    if not isinstance(completion, str) or not completion.strip():
        worker_error = result.get("error")
        error = (
            f"OWL worker returned an empty completion: {worker_error}"
            if worker_error
            else "OWL worker returned an empty completion."
        )
        write_owl_trace(
            trace_dir=trace_dir,
            sample_id=sample_id,
            model_name=model_name,
            stdout=stdout,
            stderr=stderr,
            returncode=process.returncode,
            result=result,
            error=error,
            trace_files=trace_files,
            write_console=False,
        )
        raise RuntimeError(f"{error} Trace: {trace_files['events']}")
    result["completion"] = normalize_owl_completion(completion)
    result["trace_files"] = write_owl_trace(
        trace_dir=trace_dir,
        sample_id=sample_id,
        model_name=model_name,
        stdout=stdout,
        stderr=stderr,
        returncode=process.returncode,
        result=result,
        trace_files=trace_files,
        write_console=False,
    )
    result.pop("workforce_events", None)
    result.pop("model_calls", None)
    return result

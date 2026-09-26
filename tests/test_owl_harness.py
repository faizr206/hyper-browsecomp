import asyncio
import io
import json
from pathlib import Path
import pytest

from hyper_browsecomp.owl_harness import (
    OWL_RESULT_PREFIX,
    OWL_PROGRESS_PREFIX,
    _pump_owl_stream,
    build_owl_command,
    format_owl_task_prompt,
    normalize_owl_completion,
    parse_owl_result,
    write_owl_trace,
)


def test_build_owl_command_uses_isolated_locked_runtime() -> None:
    command = build_owl_command(uv_executable="/usr/local/bin/uv")
    assert command[:3] == ["/usr/local/bin/uv", "run", "--project"]
    assert command[3].endswith("/owl_runtime")
    assert "--frozen" in command
    assert command[-1].endswith("owl_runtime/worker.py")


def test_parse_owl_result_ignores_worker_logs() -> None:
    expected = {
        "completion": "Exact Answer: blue and yellow",
        "capabilities": {"multimodal": True, "exa": False},
    }
    stdout = "ordinary OWL log\n" + OWL_RESULT_PREFIX + json.dumps(expected) + "\n"
    assert parse_owl_result(stdout) == expected


def test_parse_owl_result_requires_structured_marker() -> None:
    with pytest.raises(RuntimeError, match="structured result"):
        parse_owl_result("ordinary OWL log only")


def test_format_owl_task_prompt_appends_input_images() -> None:
    prompt = format_owl_task_prompt("Inspect it.", ["https://example.com/image.png"])
    assert prompt.startswith("Inspect it.")
    assert "Input image URLs" in prompt
    assert "https://example.com/image.png" in prompt


def test_normalize_owl_completion_keeps_last_formatted_subtask_result() -> None:
    completion = """\
--- Subtask one Result ---
raw research notes
--- Subtask two Result ---
Explanation: Visual inspection found two colors.
Exact Answer: Blue and yellow
Confidence: 100%
"""
    assert normalize_owl_completion(completion) == (
        "Explanation: Visual inspection found two colors.\n"
        "Exact Answer: Blue and yellow\n"
        "Confidence: 100%"
    )


def test_write_owl_trace_saves_console_and_structured_events(tmp_path) -> None:
    result = {
        "completion": "Exact Answer: test",
        "capabilities": {"multimodal": True},
        "tool_usage": {"browse_url": 1},
        "statistics": {"turns": 2, "total_tokens": 42},
        "model_calls": [{"turn": 1}, {"turn": 2}],
        "workforce_events": [{"event_type": "task_completed"}],
        "termination_reason": "completed",
    }
    paths = write_owl_trace(
        trace_dir=tmp_path,
        sample_id="sample/1",
        model_name="example/model",
        stdout="agent trajectory\n" + OWL_RESULT_PREFIX + json.dumps(result),
        stderr="worker warning",
        returncode=0,
        result=result,
    )

    console = Path(paths["console"]).read_text(encoding="utf-8")
    events = json.loads(Path(paths["events"]).read_text(encoding="utf-8"))
    assert "agent trajectory" in console
    assert OWL_RESULT_PREFIX not in console
    assert "worker warning" in console
    assert events["sample_id"] == "sample/1"
    assert events["statistics"] == {"turns": 2, "total_tokens": 42}
    assert events["model_calls"] == [{"turn": 1}, {"turn": 2}]
    assert events["workforce_events"] == [{"event_type": "task_completed"}]
    assert events["termination_reason"] == "completed"


def test_progress_snapshots_are_collected_but_not_written_to_console() -> None:
    progress = {
        "statistics": {"model_call_attempts": 7, "total_tokens": 123},
        "tool_usage": {"browse_url": 2},
    }
    async def collect_progress() -> tuple[dict, bytes, str]:
        stream = asyncio.StreamReader()
        stream.feed_data((OWL_PROGRESS_PREFIX + json.dumps(progress) + "\n").encode())
        stream.feed_data(b"ordinary worker output\n")
        stream.feed_eof()
        chunks: list[bytes] = []
        trace = io.StringIO()
        state: dict = {}
        await _pump_owl_stream(
            stream,
            chunks,
            trace,
            stderr=False,
            progress_state=state,
            progress_path=None,
        )
        return state, b"".join(chunks), trace.getvalue()

    state, chunks, trace = asyncio.run(collect_progress())

    assert state == progress
    assert chunks == b"ordinary worker output\n"
    assert trace == "ordinary worker output\n"


def test_progress_snapshots_refresh_running_json_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / "trace.json"
    sidecar.write_text(
        json.dumps({"status": "running", "sample_id": "q1"}),
        encoding="utf-8",
    )
    progress = {
        "statistics": {"model_call_attempts": 3, "total_tokens": 456},
        "tool_usage": {"browse_url": 1},
    }

    async def collect_progress() -> None:
        stream = asyncio.StreamReader()
        stream.feed_data((OWL_PROGRESS_PREFIX + json.dumps(progress) + "\n").encode())
        stream.feed_eof()
        await _pump_owl_stream(
            stream,
            [],
            io.StringIO(),
            stderr=False,
            progress_state={},
            progress_path=sidecar,
        )

    asyncio.run(collect_progress())

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["status"] == "running"
    assert payload["statistics"] == progress["statistics"]
    assert payload["tool_usage"] == {"browse_url": 1}

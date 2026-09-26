from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from inspect_ai.log import read_eval_log

from hyper_browsecomp.config import RunConfig
from hyper_browsecomp.durable_run import _paths, run_durable
from hyper_browsecomp import runner


def _config() -> RunConfig:
    return RunConfig(model="mockllm/model", auto_resume=True, log_dir="logs/mock")


def test_durable_refuses_duplicate_process(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    work, _ = _paths(_config())
    work.mkdir(parents=True)
    with (work / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            run_durable(_config())


def test_durable_refuses_changed_configuration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "selected_sample_ids", lambda config: ["q1"])
    work, _ = _paths(_config())
    work.mkdir(parents=True)
    (work / "run.json").write_text(json.dumps({"fingerprint": "old-config"}))
    with pytest.raises(ValueError, match="Config or selected question IDs changed"):
        run_durable(_config())


@pytest.mark.parametrize("checkpoint", [False, True])
def test_killed_run_resumes_without_repeating_finished_question(tmp_path: Path, checkpoint: bool) -> None:
    """Real Inspect subprocesses and SIGKILL; all model calls are mock/local."""
    project = Path(__file__).resolve().parents[1]
    task_file = tmp_path / "mock_task.py"
    task_file.write_text('''
import asyncio
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import match
from inspect_ai.solver import solver

ROOT = Path(__file__).parent

@solver
def answer():
    async def solve(state, generate):
        with (ROOT / "calls.txt").open("a") as f:
            f.write(str(state.sample_id) + "\\n")
        if state.sample_id == "q2" and not (ROOT / "continue").exists():
            (ROOT / "running-q2").touch()
            await asyncio.sleep(120)
        state.output = ModelOutput.from_content("mockllm/model", "answer")
        return state
    return solve

@task
def resumable():
    return Task(dataset=[Sample(id=q, input="question", target="answer") for q in ["q1", "q2"]],
                solver=answer(), scorer=match())
''')
    if checkpoint:
        restic = project / ".inspect_home/Library/Caches/inspect_ai/bin/restic_0.18.1_darwin_arm64"
        if not restic.exists():
            pytest.skip("Real checkpoint integration requires the pre-cached Inspect restic binary.")
        task_file.write_text('''
import asyncio
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.agent import react, run
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import match
from inspect_ai.solver import solver
from inspect_ai.util._restic import resolver

ROOT = Path(__file__).parent
resolver.cache_path = lambda platform, version=None: Path(RESTIC_PATH)

@solver
def answer():
    async def solve(state, generate):
        async def model(agent_state, tools):
            turn = sum(m.role == "assistant" for m in agent_state.messages) + 1
            with (ROOT / "calls.txt").open("a") as f:
                f.write(f"{state.sample_id}-turn{turn}\\n")
            if state.sample_id == "q2" and turn == 2 and not (ROOT / "continue").exists():
                (ROOT / "running-q2").touch()
                await asyncio.sleep(120)
            agent_state.output = ModelOutput.from_content("mockllm/model", "answer")
            agent_state.messages.append(agent_state.output.message)
            return agent_state
        async def on_continue(agent_state):
            turns = sum(m.role == "assistant" for m in agent_state.messages)
            return turns < (2 if state.sample_id == "q2" else 1)
        agent_state = await run(react(model=model, on_continue=on_continue), "question")
        state.output = agent_state.output
        return state
    return solve

@task
def resumable():
    return Task(dataset=[Sample(id=q, input="question", target="answer") for q in ["q1", "q2"]],
                solver=answer(), scorer=match())
'''.replace("RESTIC_PATH", repr(str(restic))))
    launch = '''
import sys
from pathlib import Path
from hyper_browsecomp import runner
from hyper_browsecomp.config import RunConfig
from hyper_browsecomp.durable_run import run_durable
root = Path(sys.argv[1])
runner.PROJECT_ROOT = root
runner.selected_sample_ids = lambda config: ["q1", "q2"]
runner.sanitize_eval_log = lambda config, path: False
runner.build_inspect_command = lambda config: ["inspect", "eval", str(root / "mock_task.py") + "@resumable",
    "--model", "mockllm/model", "--max-samples", "1", "--log-buffer", "1", "--display", "none"]
raise SystemExit(run_durable(RunConfig(model="mockllm/model", auto_resume=True, log_dir="logs/mock")))
'''
    if checkpoint:
        launch = launch.replace(
            '"--log-buffer", "1", "--display", "none"]',
            '"--log-buffer", "1", "--display", "none", "--checkpoint", "turn:1"]',
        ).replace('log_dir="logs/mock")))', 'log_dir="logs/mock", inspect_checkpoint="turn:1")))')
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project / "src")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    command = [sys.executable, "-c", launch, str(tmp_path)]
    with (tmp_path / "first-output.txt").open("w") as output:
        process = subprocess.Popen(command, env=env, stdout=output, stderr=output, start_new_session=True)
        try:
            # Importing Inspect and initializing its checkpoint runtime can be
            # slow on a cold machine. Keep this comfortably below the fixture's
            # intentional 120-second hang while avoiding startup-only flakes.
            deadline = time.monotonic() + 60
            while not (tmp_path / "running-q2").exists():
                assert process.poll() is None, (tmp_path / "first-output.txt").read_text()
                assert time.monotonic() < deadline, (tmp_path / "first-output.txt").read_text()
                time.sleep(0.05)
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
    (tmp_path / "continue").touch()
    resumed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    if checkpoint:
        assert calls.count("q1-turn1") == 1
        assert calls.count("q2-turn1") == 1  # Its saved turn was restored.
        assert calls.count("q2-turn2") == 2  # Only the in-flight request repeated.
    else:
        assert calls.count("q1") == 1
        assert calls.count("q2") == 2  # The interrupted question, alone, was retried.
    canonical = list((tmp_path / "logs").rglob("*.eval"))
    assert len(canonical) == 1
    samples = read_eval_log(canonical[0]).samples
    assert {sample.id for sample in samples} == {"q1", "q2"}
    assert all(sample.error is None and sample.scores for sample in samples)
    assert len(list((tmp_path / "dump").rglob("*.eval"))) >= 2
    again = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
    assert again.returncode == 0, again.stdout + again.stderr
    assert "Already complete: 2/2" in again.stdout
    assert (tmp_path / "calls.txt").read_text().splitlines() == calls

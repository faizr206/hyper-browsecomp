"""The resume worker must register MiniMax before Inspect rebuilds its model."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyper_browsecomp import durable_run, runner
from hyper_browsecomp.config import RunConfig


@pytest.mark.parametrize("retry", [False, True])
def test_minimax_worker_registers_custom_provider_before_cli(tmp_path: Path, monkeypatch, retry: bool):
    config = RunConfig(provider="minimax", model_name="MiniMax-M3", model_args={"thinking": True},
                       model_api_key_env="MINIMAX_API_KEY", auto_resume=True,
                       log_dir="logs/minimax", inspect_checkpoint="turn:1", inspect_log_buffer=1)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "selected_sample_ids", lambda _: ["public-question"])
    monkeypatch.setattr(runner, "prepare_env", lambda _: {})
    source = tmp_path / "previous.eval"
    if retry:
        work, _ = durable_run._paths(config)
        work.mkdir(parents=True)
        (work / "run.json").write_text(json.dumps({
            "fingerprint": durable_run._fingerprint(config, ["public-question"]),
        }))
    monkeypatch.setattr(durable_run, "_latest", lambda _: source if retry else None)
    monkeypatch.setattr(durable_run, "_recover", lambda log, env, lock: log)
    monkeypatch.setattr(durable_run, "_completed", lambda *_: 0)
    monkeypatch.setattr(durable_run, "_publish", lambda *_: None)
    launches = []

    def start(command, **kwargs):
        launches.append((command, kwargs))
        return SimpleNamespace(wait=lambda: 1)

    monkeypatch.setattr(durable_run.subprocess, "Popen", start)
    assert durable_run.run_durable(config) == 1
    assert len(launches) == 1
    command, options = launches[0]
    assert command[:2] == [sys.executable, "-c"]
    bootstrap = command[2]
    assert bootstrap.index("import hyper_browsecomp.minimax") < bootstrap.index("main()")
    assert command[3] == ("eval-retry" if retry else "eval")
    assert command[command.index("--checkpoint") + 1] == "turn:1"
    assert options["cwd"] == tmp_path
    assert options["pass_fds"]  # Worker inherits the lock preventing duplicate launches.
    if retry:
        assert command[4] == str(source)
    else:
        assert command[command.index("--model") + 1] == "minimax/MiniMax-M3"
        assert "thinking=true" in command

from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

from hyper_browsecomp.config import RunConfig
from hyper_browsecomp.runner import build_inspect_command, prepare_env, rename_new_eval_log


def test_prepare_env_maps_generic_model_env(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "secret")
    monkeypatch.setenv("MODEL_BASE_URL", "https://example.com/v1")
    config = RunConfig(provider="deepseek", model_name="deepseek-chat")
    env = prepare_env(config)
    assert env["DEEPSEEK_API_KEY"] == "secret"
    assert env["DEEPSEEK_BASE_URL"] == "https://example.com/v1"
    assert env["HOME"].endswith(".inspect_home")


def test_build_inspect_command_adds_strict_tools_false() -> None:
    config = RunConfig(provider="deepseek", model_name="deepseek-chat")
    command = build_inspect_command(config)
    assert "-M" in command
    assert "strict_tools=false" in command
    assert "--max-retries" in command
    assert "--attempt-timeout" in command
    assert "--retry-on-error" in command
    assert "3" in command
    assert "--no-fail-on-error" in command
    assert "--continue-on-fail" in command


def test_build_inspect_command_passes_sample_range() -> None:
    config = RunConfig(provider="deepseek", model_name="deepseek-chat", sample_range="1-2")
    command = build_inspect_command(config)
    assert "-T" in command
    assert "sample_range=1-2" in command


def test_runner_module_invokes_inspect(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    inspect_bin = bin_dir / "inspect"
    captured = tmp_path / "inspect_args.txt"
    inspect_bin.write_text(
        f"#!/usr/bin/env sh\nprintf '%s\\n' \"$@\" > {captured}\n",
        encoding="utf-8",
    )
    inspect_bin.chmod(0o755)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "provider: deepseek",
                "model_name: deepseek-chat",
                "data_path: data/browsecomp_multilingual_indonesian_new.jsonl",
                'sample_range: "1-2"',
                f"log_dir: {tmp_path / 'logs'}",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        [sys.executable, "-m", "hyper_browsecomp.runner", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert captured.exists()
    args = captured.read_text(encoding="utf-8")
    assert "eval\n" in args
    assert "sample_range=1-2\n" in args
    assert "--no-fail-on-error\n" in args
    assert "--continue-on-fail\n" in args


def test_rename_new_eval_log_uses_requested_pattern(tmp_path: Path, monkeypatch) -> None:
    config = RunConfig(
        provider="deepseek",
        model_name="deepseek-chat",
        data_path="data/my_dataset.jsonl",
        log_dir=str(tmp_path),
    )
    original = tmp_path / "20260717T100000Z_hyper_browsecomp_abc.eval"
    original.write_text("placeholder", encoding="utf-8")
    before: set[Path] = set()

    fake_log = SimpleNamespace(
        eval=SimpleNamespace(created="2026-07-17T10:00:00Z", task_id="abc123")
    )
    monkeypatch.setattr("hyper_browsecomp.runner.read_eval_log", lambda path: fake_log)

    renamed = rename_new_eval_log(
        config,
        before=before,
        started_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
    )

    assert renamed is not None
    assert renamed.name == "my_dataset_deepseek-chat_20260717T100000Z_abc123.eval"
    assert renamed.exists()

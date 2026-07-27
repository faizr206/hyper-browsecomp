from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

from hyper_browsecomp.config import RunConfig
from hyper_browsecomp.runner import (
    build_inspect_command,
    main,
    prepare_env,
    rename_new_eval_log,
    unfinished_sample_ids,
)


def test_prepare_env_maps_generic_model_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    config = RunConfig(
        provider="openai",
        model_name="gpt-5-mini",
        model_api_key_env="OPENAI_API_KEY",
        model_base_url="https://example.com/v1",
    )
    env = prepare_env(config)
    assert env["OPENAI_API_KEY"] == "secret"
    assert env["OPENAI_BASE_URL"] == "https://example.com/v1"
    assert env["HOME"].endswith(".inspect_home")


def test_prepare_env_maps_scorer_from_configured_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "scorer-secret")
    config = RunConfig(
        provider="openai",
        model_name="gpt-5-mini",
        scorer_provider="openrouter",
        scorer_model_name="openai/gpt-5.4-mini",
        scorer_api_key_env="OPENROUTER_API_KEY",
        scorer_base_url="https://openrouter.ai/api/v1",
    )
    env = prepare_env(config)
    assert env["OPENROUTER_API_KEY"] == "scorer-secret"
    assert env["OPENROUTER_BASE_URL"] == "https://openrouter.ai/api/v1"


def test_prepare_env_maps_gemini_to_google_env(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
    config = RunConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        model_api_key_env="GOOGLE_API_KEY",
        model_base_url="https://generativelanguage.googleapis.com",
    )
    env = prepare_env(config)
    assert env["GOOGLE_API_KEY"] == "google-secret"
    assert env["GOOGLE_BASE_URL"] == "https://generativelanguage.googleapis.com"


def test_prepare_env_maps_grok_to_xai_env(monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "xai-secret")
    config = RunConfig(
        provider="grok",
        model_name="grok-3-mini",
        model_api_key_env="XAI_API_KEY",
        model_base_url="api.x.ai",
    )
    env = prepare_env(config)
    assert env["XAI_API_KEY"] == "xai-secret"
    assert env["XAI_BASE_URL"] == "api.x.ai"


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


def test_build_inspect_command_can_limit_sample_ids() -> None:
    config = RunConfig(provider="deepseek", model_name="deepseek-chat")
    command = build_inspect_command(config, sample_ids=["q2", "q3"])
    assert "--sample-id" in command
    assert "q2,q3" in command


def test_build_inspect_command_uses_native_provider_and_backend_args() -> None:
    config = RunConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        search_backend="internal",
        fetch_backend="none",
    )
    command = build_inspect_command(config)
    assert "google/gemini-2.5-pro" in command
    assert "search_backend=internal" in command
    assert "fetch_backend=none" in command
    assert "model_provider=gemini" in command


def test_main_without_config_returns_usage_error(capsys) -> None:
    assert main([]) == 2
    captured = capsys.readouterr()
    assert "usage: python -m hyper_browsecomp.runner CONFIG.yaml" in captured.err


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


def test_unfinished_sample_ids_returns_missing_and_error_samples(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        "\n".join(
            [
                '{"id":"q1","question":"one","answers":["a"]}',
                '{"id":"q2","question":"two","answers":["b"]}',
                '{"id":"q3","question":"three","answers":["c"]}',
            ]
        ),
        encoding="utf-8",
    )
    config = RunConfig(provider="deepseek", model_name="deepseek-chat", data_path=str(data_path))
    fake_log = SimpleNamespace(
        eval=SimpleNamespace(task_args={"data_path": str(data_path), "sample_range": "1-3"}),
        samples=[
            SimpleNamespace(id="q1", metadata={}, completed_at="2026-07-17T10:00:00Z", error=None),
            SimpleNamespace(
                id=2, metadata={"id": "q2"}, completed_at=None, error=SimpleNamespace()
            ),
            SimpleNamespace(id="outside", metadata={}, completed_at=None, error=SimpleNamespace()),
        ],
    )
    monkeypatch.setattr("hyper_browsecomp.runner.read_eval_log", lambda *args, **kwargs: fake_log)

    assert unfinished_sample_ids(config, "logs/partial.eval") == ["q2", "q3"]

from pathlib import Path

from scripts.rescore_eval_logs import (
    LOG_ROOT,
    discover_eval_logs,
    model_output_dir,
    openrouter_model,
    output_path,
    score_command,
)


def test_discover_eval_logs_is_recursive_and_excludes_output(tmp_path: Path) -> None:
    source = tmp_path / "logs" / "run.eval"
    nested = tmp_path / "logs" / "nested" / "other.eval"
    output_dir = tmp_path / "logs" / "rescored"
    generated = output_dir / "already.eval"
    for path in (source, nested, generated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert discover_eval_logs([tmp_path / "logs"], output_dir) == [nested, source]


def test_output_path_keeps_filename_for_external_source(tmp_path: Path) -> None:
    source = tmp_path / "source.eval"
    destination = output_path(source, tmp_path / "out")
    assert destination == (tmp_path / "out" / "source.eval").resolve()


def test_score_command_uses_new_judge_for_context_and_scorer(tmp_path: Path) -> None:
    command = score_command(
        "inspect",
        tmp_path / "source.eval",
        tmp_path / "output.partial.eval",
        model="openrouter/openai/gpt-oss-20b",
        base_url="https://openrouter.ai/api/v1",
        confidential_data_path="afaji/HyperBrowseComp",
        concurrency=3,
    )

    assert command.count("openrouter/openai/gpt-oss-20b") == 1
    assert "scorer_model=openrouter/openai/gpt-oss-20b" in command
    assert "confidential_data_path=afaji/HyperBrowseComp" in command
    assert command[command.index("--action") + 1] == "append"
    assert command[command.index("--stream") + 1] == "3"


def test_openrouter_slug_is_normalized_and_gets_separate_output() -> None:
    model = openrouter_model("google/gemini-3.7-flash")
    glm_model = openrouter_model("z-ai/glm-4.7")
    kimi_model = openrouter_model("moonshotai/kimi-k2-thinking")

    assert model == "openrouter/google/gemini-3.7-flash"
    assert model_output_dir(model) == LOG_ROOT / "rescored" / "gemini-3.7-flash"
    assert glm_model == "openrouter/z-ai/glm-4.7"
    assert model_output_dir(glm_model) == LOG_ROOT / "rescored" / "glm-4.7"
    assert kimi_model == "openrouter/moonshotai/kimi-k2-thinking"
    assert model_output_dir(kimi_model) == LOG_ROOT / "rescored" / "kimi-k2-thinking"


def test_existing_inspect_openrouter_path_is_unchanged() -> None:
    assert (
        openrouter_model("openrouter/openai/gpt-oss-20b")
        == "openrouter/openai/gpt-oss-20b"
    )


def test_score_command_can_disable_streaming_for_incomplete_logs(tmp_path: Path) -> None:
    command = score_command(
        "inspect",
        tmp_path / "source.eval",
        tmp_path / "output.partial.eval",
        model="openrouter/z-ai/glm-4.7",
        base_url="https://openrouter.ai/api/v1",
        confidential_data_path="afaji/HyperBrowseComp",
        concurrency=50,
        stream=False,
    )

    assert "--stream" not in command

from pathlib import Path

import pytest

from hyper_browsecomp.config import RunConfig, load_run_config


def test_config_resolves_openai_api_model() -> None:
    config = RunConfig(provider="deepseek", model_name="deepseek-chat")
    assert config.resolved_model() == "openai-api/deepseek/deepseek-chat"


def test_config_allows_prequalified_model() -> None:
    config = RunConfig(model="openai-api/openrouter/qwen/qwen3-32b")
    assert config.resolved_model() == "openai-api/openrouter/qwen/qwen3-32b"


def test_web_code_requires_sandbox() -> None:
    with pytest.raises(ValueError, match="no_sandbox=false"):
        RunConfig(provider="deepseek", model_name="deepseek-chat", tool_profile="web_code")


def test_sample_range_rejects_mixed_slice_options() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        RunConfig(
            provider="deepseek",
            model_name="deepseek-chat",
            sample_range="1-2",
            start_index=0,
        )


def test_retry_and_continue_defaults_are_resilient() -> None:
    config = RunConfig(provider="deepseek", model_name="deepseek-chat")
    assert config.inspect_model_max_retries == 1
    assert config.inspect_attempt_timeout == 60
    assert config.inspect_retry_on_error == 3
    assert config.inspect_no_fail_on_error is True
    assert config.inspect_continue_on_fail is True


def test_legacy_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("defaults:\n  MODEL: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy keys"):
        load_run_config(path)

from pathlib import Path

import pytest

from hyper_browsecomp.config import RunConfig, load_run_config


def test_config_resolves_openai_api_model() -> None:
    config = RunConfig(provider="deepseek", model_name="deepseek-chat")
    assert config.resolved_model() == "openai-api/deepseek/deepseek-chat"


def test_config_resolves_native_model() -> None:
    config = RunConfig(provider="anthropic", model_name="claude-sonnet-4-0")
    assert config.resolved_model() == "anthropic/claude-sonnet-4-0"


def test_config_resolves_gemini_alias_to_google_provider() -> None:
    config = RunConfig(provider="gemini", model_name="gemini-2.5-pro")
    assert config.resolved_model() == "google/gemini-2.5-pro"


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


def test_dev_config_loads() -> None:
    config = load_run_config(Path(__file__).resolve().parents[1] / "configs" / "dev.yaml")
    assert config.data_path == "data/dev.jsonl"
    assert config.sample_range == "1"
    assert config.model_api_key_env == "OPENAI_API_KEY"
    assert config.model_base_url == "https://api.openai.com/v1"
    assert config.tool_profile == "web"


@pytest.mark.parametrize(
    ("config_name", "provider", "api_key_env", "base_url", "search_backend"),
    [
        ("dev_openai.yaml", "openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "internal"),
        (
            "dev_anthropic.yaml",
            "anthropic",
            "ANTHROPIC_API_KEY",
            "https://api.anthropic.com",
            "internal",
        ),
        (
            "dev_gemini.yaml",
            "gemini",
            "GOOGLE_API_KEY",
            "https://generativelanguage.googleapis.com",
            "internal",
        ),
        ("dev_grok.yaml", "grok", "XAI_API_KEY", "api.x.ai", "internal"),
        (
            "dev_mistral.yaml",
            "mistral",
            "MISTRAL_API_KEY",
            "https://api.mistral.ai",
            "internal",
        ),
        (
            "dev_perplexity.yaml",
            "perplexity",
            "PERPLEXITY_API_KEY",
            "https://api.perplexity.ai",
            "internal",
        ),
        (
            "dev_qwen.yaml",
            "qwen",
            "DASHSCOPE_API_KEY",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "none",
        ),
    ],
)
def test_provider_dev_configs_load(
    config_name: str,
    provider: str,
    api_key_env: str,
    base_url: str,
    search_backend: str,
) -> None:
    config = load_run_config(Path(__file__).resolve().parents[1] / "configs" / config_name)
    assert config.provider == provider
    assert config.scorer_provider == provider
    assert config.model_api_key_env == api_key_env
    assert config.scorer_api_key_env == api_key_env
    assert config.model_base_url == base_url
    assert config.scorer_base_url == base_url
    assert config.search_backend == search_backend
    assert config.data_path == "data/dev.jsonl"

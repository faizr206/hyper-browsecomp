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


def test_config_resolves_openrouter_as_native_provider() -> None:
    config = RunConfig(provider="openrouter", model_name="anthropic/claude-fable-5.1")
    assert config.resolved_model() == "openrouter/anthropic/claude-fable-5.1"


def test_owl_config_reuses_openrouter_model_settings() -> None:
    config = RunConfig(
        provider="openrouter",
        model_name="google/gemini-3.7-flash",
        model_api_key_env="OPENROUTER_API_KEY",
        model_base_url="https://openrouter.ai/api/v1",
        harness="owl",
    )
    assert config.resolved_owl_model_name() == "google/gemini-3.7-flash"
    assert config.resolved_owl_api_key_env() == "OPENROUTER_API_KEY"
    assert config.resolved_owl_base_url() == "https://openrouter.ai/api/v1"
    assert config.owl_multimodal is True
    assert config.owl_browser_round_limit == 12
    assert config.owl_finalize_reserve_seconds == 120
    assert config.owl_max_external_tool_calls == 50
    assert config.owl_max_model_calls == 180
    assert config.owl_model_max_retries == 1


def test_owl_config_requires_base_url() -> None:
    with pytest.raises(ValueError, match="harness=owl requires"):
        RunConfig(provider="openrouter", model_name="google/gemini-3.7-flash", harness="owl")


def test_owl_finalize_reserve_must_fit_inside_task_timeout() -> None:
    with pytest.raises(ValueError, match="finalize_reserve_seconds"):
        RunConfig(
            provider="openrouter",
            model_name="google/gemini-3.7-flash",
            model_base_url="https://openrouter.ai/api/v1",
            harness="owl",
            owl_task_timeout_seconds=120,
            owl_finalize_reserve_seconds=120,
        )


@pytest.mark.parametrize(
    "config_path",
    [
        "owl_dev/openrouter_gemini_flash.yaml",
        "owl_dev/openrouter_gemini_flash_video.yaml",
        "owl_dev/openrouter_gemini_flash_bilibili.yaml",
        "owl_dev/openrouter_gemini_flash_media.yaml",
        "owl_full/openrouter_gemini_flash.yaml",
        "owl_full/openrouter_gemini_flash_5.yaml",
    ],
)
def test_owl_configs_enable_multimodal_without_exa(config_path: str) -> None:
    config = load_run_config(Path(__file__).resolve().parents[1] / "configs" / config_path)
    assert config.harness == "owl"
    assert config.resolved_owl_model_name() == "google/gemini-3.7-flash"
    assert config.owl_multimodal is True
    assert config.search_backend == "none"
    assert config.fetch_backend == "none"
    assert config.owl_trace_dir == "logs/owl/traces"


def test_owl_full_config_uses_pass_at_one_global_budgets() -> None:
    config = load_run_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "owl_full"
        / "openrouter_gemini_flash_5.yaml"
    )
    assert config.owl_browser_round_limit == 12
    assert config.owl_task_timeout_seconds == 1200
    assert config.owl_finalize_reserve_seconds == 120
    assert config.owl_max_external_tool_calls == 50
    assert config.owl_max_model_calls == 180
    assert config.owl_model_max_retries == 1
    assert config.inspect_max_samples_parallel == 2
    assert config.inspect_retry_on_error == 0


def test_owl_full_config_runs_four_samples_in_parallel() -> None:
    config = load_run_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "owl_full"
        / "openrouter_gemini_flash.yaml"
    )
    assert config.inspect_max_samples_parallel == 4


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


def test_numeric_sample_range_is_normalized() -> None:
    config = RunConfig(provider="deepseek", model_name="deepseek-chat", sample_range=1)
    assert config.sample_range == "1"


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


def test_openrouter_fable_config_pins_google_vertex() -> None:
    config = load_run_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "internal_full"
        / "openrouter_fable_51.yaml"
    )
    assert config.resolved_model() == "openrouter/anthropic/claude-fable-5.1"
    assert config.model_args == {"provider": {"only": ["google-vertex"]}}
    assert config.resolved_scorer_model() is None


@pytest.mark.parametrize(
    ("config_path", "provider", "api_key_env", "base_url", "search_backend"),
    [
        (
            "internal_search_dev/dev_openai.yaml",
            "openai",
            "OPENAI_API_KEY",
            "https://api.openai.com/v1",
            "internal",
        ),
        (
            "internal_search_dev/dev_anthropic.yaml",
            "anthropic",
            "ANTHROPIC_API_KEY",
            "https://api.anthropic.com",
            "internal",
        ),
        (
            "internal_search_dev/dev_gemini.yaml",
            "gemini",
            "GOOGLE_API_KEY",
            "https://generativelanguage.googleapis.com",
            "internal",
        ),
        ("internal_search_dev/dev_grok.yaml", "grok", "XAI_API_KEY", "api.x.ai", "internal"),
        (
            "internal_search_dev/dev_mistral.yaml",
            "mistral",
            "MISTRAL_API_KEY",
            "https://api.mistral.ai",
            "internal",
        ),
        (
            "internal_search_dev/dev_perplexity.yaml",
            "perplexity",
            "PERPLEXITY_API_KEY",
            "https://api.perplexity.ai",
            "internal",
        ),
        (
            "exa_dev/dev_openrouter.yaml",
            "openrouter",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1",
            "exa",
        ),
        (
            "internal_search_dev/dev_qwen.yaml",
            "qwen",
            "DASHSCOPE_API_KEY",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "none",
        ),
    ],
)
def test_provider_dev_configs_load(
    config_path: str,
    provider: str,
    api_key_env: str,
    base_url: str,
    search_backend: str,
) -> None:
    config = load_run_config(Path(__file__).resolve().parents[1] / "configs" / config_path)
    assert config.provider == provider
    assert config.scorer_provider == provider
    assert config.model_api_key_env == api_key_env
    assert config.scorer_api_key_env == api_key_env
    assert config.model_base_url == base_url
    assert config.scorer_base_url == base_url
    assert config.search_backend == search_backend
    assert config.data_path == "data/dev.jsonl"


@pytest.mark.parametrize(
    "config_path",
    sorted((Path(__file__).resolve().parents[1] / "configs" / "exa_dev").glob("*.yaml")),
)
def test_exa_dev_configs_use_dev_jsonl(config_path: Path) -> None:
    config = load_run_config(config_path)
    assert config.data_path == "data/dev.jsonl"


@pytest.mark.parametrize(
    "config_path",
    sorted((Path(__file__).resolve().parents[1] / "configs" / "internal_search_dev").glob("*.yaml")),
)
def test_internal_search_dev_configs_use_dev_jsonl(config_path: Path) -> None:
    config = load_run_config(config_path)
    assert config.data_path == "data/dev.jsonl"


@pytest.mark.parametrize(
    "config_path",
    sorted((Path(__file__).resolve().parents[1] / "configs" / "exa_full").glob("*.yaml")),
)
def test_exa_full_configs_use_hf_dataset(config_path: Path) -> None:
    config = load_run_config(config_path)
    assert config.data_path == "afaji/HyperBrowseComp"
    assert config.sample_range is None
    assert config.tool_profile == "web"
    assert config.search_backend == "exa"
    assert config.fetch_backend == "exa"
    assert config.max_steps == 25
    assert config.inspect_max_samples_parallel == 50
    assert config.inspect_model_max_retries == 2
    assert config.inspect_attempt_timeout == 360
    assert config.inspect_retry_on_error == 3
    assert config.inspect_no_fail_on_error is True
    assert config.inspect_continue_on_fail is True
    assert config.no_sandbox is True

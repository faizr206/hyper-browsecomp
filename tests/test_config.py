from pathlib import Path

import pytest

from hyper_browsecomp.config import RunConfig, load_run_config


def test_config_resolves_openai_api_model() -> None:
    config = RunConfig(provider="deepseek", model_name="deepseek-chat")
    assert config.resolved_model() == "openai-api/deepseek/deepseek-chat"


def test_config_resolves_native_model() -> None:
    config = RunConfig(provider="anthropic", model_name="claude-sonnet-4-0")
    assert config.resolved_model() == "anthropic/claude-sonnet-4-0"


def test_config_resolves_minimax_direct_model() -> None:
    config = RunConfig(provider="minimax", model_name="MiniMax-M3")
    assert config.resolved_model() == "minimax/MiniMax-M3"


def test_config_resolves_gemini_alias_to_google_provider() -> None:
    config = RunConfig(provider="gemini", model_name="gemini-2.5-pro")
    assert config.resolved_model() == "google/gemini-2.5-pro"


def test_config_resolves_openrouter_as_native_provider() -> None:
    config = RunConfig(provider="openrouter", model_name="anthropic/claude-fable-5.1")
    assert config.resolved_model() == "openrouter/anthropic/claude-fable-5.1"


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


def test_config_accepts_sample_ids_file() -> None:
    config = RunConfig(model="mockllm/model", sample_ids_file="configs/retained_ids.txt")
    assert config.sample_ids_file == "configs/retained_ids.txt"


@pytest.mark.parametrize(
    "slice_option",
    [
        {"sample_range": "1-2"},
        {"start_index": 0},
        {"end_index": 423},
        {"num_samples": 423},
    ],
)
def test_sample_ids_file_rejects_slice_options(slice_option: dict) -> None:
    with pytest.raises(ValueError, match="sample_ids_file cannot be combined"):
        RunConfig(
            model="mockllm/model",
            sample_ids_file="configs/retained_ids.txt",
            **slice_option,
        )


@pytest.mark.parametrize("path", ["", "  "])
def test_sample_ids_file_rejects_blank_path(path: str) -> None:
    with pytest.raises(ValueError, match="sample_ids_file must be a nonempty path"):
        RunConfig(model="mockllm/model", sample_ids_file=path)


def test_retry_and_continue_defaults_are_resilient() -> None:
    config = RunConfig(provider="deepseek", model_name="deepseek-chat")
    assert config.inspect_model_max_retries == 1
    assert config.inspect_attempt_timeout == 60
    assert config.inspect_retry_on_error == 3
    assert config.inspect_no_fail_on_error is True
    assert config.inspect_continue_on_fail is True


def test_checkpoint_options_and_auto_resume_are_opt_in() -> None:
    config = RunConfig(model="mockllm/model")
    assert config.auto_resume is False
    assert config.inspect_checkpoint is None
    assert config.inspect_log_buffer is None
    resume_config = RunConfig(
        model="mockllm/model", auto_resume=True, inspect_checkpoint="turn:1", inspect_log_buffer=1
    )
    assert resume_config.auto_resume is True
    assert resume_config.inspect_checkpoint == "turn:1"
    assert resume_config.inspect_log_buffer == 1


def test_log_buffer_must_be_positive() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        RunConfig(model="mockllm/model", inspect_log_buffer=0)


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
    assert config.inspect_max_samples_parallel == (5 if config.provider == "minimax" else 50)
    assert config.inspect_model_max_retries == 2
    assert config.inspect_attempt_timeout == 360
    assert config.inspect_retry_on_error == 3
    assert config.inspect_no_fail_on_error is True
    assert config.inspect_continue_on_fail is True
    assert config.no_sandbox is True


@pytest.mark.parametrize("folder,backend", [("internal_full", "internal"), ("exa_full", "exa")])
def test_minimax_configs_use_direct_api_and_independent_resumable_runs(folder, backend):
    root = Path(__file__).resolve().parents[1]
    config = load_run_config(root / "configs" / folder / "minimax_m3.yaml")
    assert config.resolved_model() == "minimax/MiniMax-M3"
    assert config.model_api_key_env == "MINIMAX_API_KEY"
    assert config.model_base_url == "https://api.minimax.io/anthropic"
    assert config.model_args == {"thinking": True}
    assert config.search_backend == backend
    assert config.fetch_backend == ("none" if backend == "internal" else "exa")
    assert config.max_steps == 25
    assert config.inspect_max_samples_parallel == 5
    assert config.auto_resume and config.inspect_checkpoint == "turn:1"
    assert config.inspect_log_buffer == 1
    ids = (root / config.sample_ids_file).read_text().splitlines()
    assert len(ids) == len(set(ids)) == 423
    other_folder = "exa_full" if folder == "internal_full" else "internal_full"
    other = load_run_config(root / "configs" / other_folder / "minimax_m3.yaml")
    assert config.log_dir != other.log_dir

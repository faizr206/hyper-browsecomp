from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


SearchBackend = Literal["internal", "exa", "firecrawl", "none"]
FetchBackend = Literal["exa", "firecrawl", "none"]
ToolProfile = Literal["web", "web_code"]


NATIVE_PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google",
}
NATIVE_PROVIDERS = {
    "openai",
    "anthropic",
    "gemini",
    "google",
    "grok",
    "mistral",
    "perplexity",
}


REMOVED_KEYS = {
    "defaults",
    "overrides",
    "openrouter_base_url",
    "agent_model",
    "tool_calling",
    "agent_mode",
    "include_browser",
    "include_python",
}


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    model_name: str | None = None
    model: str | None = None
    scorer_provider: str | None = None
    scorer_model_name: str | None = None
    scorer_model: str | None = None
    model_api_key_env: str = "MODEL_API_KEY"
    model_base_url: str | None = None
    scorer_api_key_env: str = "SCORER_API_KEY"
    scorer_base_url: str | None = None
    data_path: str = "data/dev.jsonl"
    tool_profile: ToolProfile = "web"
    search_backend: SearchBackend = "exa"
    fetch_backend: FetchBackend = "exa"
    search_max_results: int = 5
    search_timeout_seconds: int = 60
    fetch_timeout_seconds: int = 60
    fetch_max_chars: int = 20000
    max_steps: int = 12
    bash_timeout: int = 120
    python_timeout: int = 120
    sample_range: str | None = None
    start_index: int | None = None
    end_index: int | None = None
    num_samples: int | None = None
    inspect_max_samples_parallel: int = 1
    inspect_model_max_retries: int | None = 1
    inspect_attempt_timeout: int | None = 60
    inspect_retry_on_error: int | None = 3
    inspect_continue_on_fail: bool = True
    inspect_no_fail_on_error: bool = True
    no_sandbox: bool = True
    log_dir: str = "logs"

    @field_validator("sample_range", mode="before")
    @classmethod
    def normalize_sample_range(cls, value: object) -> str | None:
        if value is None or value == "":
            return None
        return str(value)

    @model_validator(mode="after")
    def validate_models(self) -> "RunConfig":
        if not self.model and not (self.provider and self.model_name):
            raise ValueError("Provide either model or provider + model_name.")
        if self.scorer_model_name and not self.scorer_provider and not self.scorer_model:
            raise ValueError("scorer_model_name requires scorer_provider unless scorer_model is set.")
        if self.tool_profile == "web_code" and self.no_sandbox:
            raise ValueError("tool_profile=web_code requires no_sandbox=false.")
        if self.sample_range is not None and any(
            value is not None for value in (self.start_index, self.end_index, self.num_samples)
        ):
            raise ValueError(
                "sample_range cannot be combined with start_index, end_index, or num_samples."
            )
        return self

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        assert self.provider and self.model_name
        if self.provider in NATIVE_PROVIDERS:
            provider = NATIVE_PROVIDER_ALIASES.get(self.provider, self.provider)
            return f"{provider}/{self.model_name}"
        return f"openai-api/{self.provider}/{self.model_name}"

    def resolved_scorer_model(self) -> str | None:
        if self.scorer_model:
            return self.scorer_model
        if self.scorer_provider and self.scorer_model_name:
            if self.scorer_provider in NATIVE_PROVIDERS:
                provider = NATIVE_PROVIDER_ALIASES.get(self.scorer_provider, self.scorer_provider)
                return f"{provider}/{self.scorer_model_name}"
            return f"openai-api/{self.scorer_provider}/{self.scorer_model_name}"
        return None


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: config root must be a mapping.")
    unexpected_legacy = sorted(REMOVED_KEYS.intersection(raw))
    if unexpected_legacy:
        raise ValueError(
            f"{config_path}: remove legacy keys {', '.join(unexpected_legacy)} and use the flat schema."
        )
    return RunConfig.model_validate(raw)

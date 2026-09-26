from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SearchBackend = Literal["internal", "exa", "firecrawl", "none"]
FetchBackend = Literal["exa", "firecrawl", "none"]
ToolProfile = Literal["web", "web_code"]
Harness = Literal["react", "owl"]
ReasoningEffort = Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"]


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
    "openrouter",
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
    model_args: dict[str, Any] = Field(default_factory=dict)
    scorer_provider: str | None = None
    scorer_model_name: str | None = None
    scorer_model: str | None = None
    model_api_key_env: str = "MODEL_API_KEY"
    model_base_url: str | None = None
    scorer_api_key_env: str = "SCORER_API_KEY"
    scorer_base_url: str | None = None
    data_path: str = "data/dev.jsonl"
    harness: Harness = "react"
    tool_profile: ToolProfile = "web"
    search_backend: SearchBackend = "exa"
    fetch_backend: FetchBackend = "exa"
    search_max_results: int = 5
    search_timeout_seconds: int = 60
    fetch_timeout_seconds: int = 60
    fetch_max_chars: int = 20000
    max_steps: int = 12
    owl_model_name: str | None = None
    owl_api_key_env: str | None = None
    owl_base_url: str | None = None
    owl_headless: bool = True
    owl_multimodal: bool = True
    owl_browser_round_limit: int = Field(default=12, ge=1)
    owl_task_timeout_seconds: int = 900
    owl_timeout_scale: float = Field(default=1.0, ge=1.0)
    owl_finalize_reserve_seconds: int = Field(default=120, ge=0)
    owl_max_external_tool_calls: int = Field(default=50, ge=1)
    owl_max_model_calls: int = Field(default=180, ge=1)
    owl_model_max_retries: int = Field(default=1, ge=0)
    owl_max_tokens: int = 8192
    owl_reasoning_effort: ReasoningEffort | None = None
    owl_trace_dir: str = "logs/owl/traces"
    bash_timeout: int = 120
    python_timeout: int = 120
    sample_ids_path: str | None = None
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
    inspect_ctl_server: bool = True
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
        if self.harness == "react" and self.tool_profile == "web_code" and self.no_sandbox:
            raise ValueError("tool_profile=web_code requires no_sandbox=false.")
        if self.harness == "owl" and not (self.owl_base_url or self.model_base_url):
            raise ValueError("harness=owl requires owl_base_url or model_base_url.")
        if (
            self.harness == "owl"
            and self.owl_finalize_reserve_seconds >= self.owl_task_timeout_seconds
        ):
            raise ValueError(
                "owl_finalize_reserve_seconds must be smaller than "
                "owl_task_timeout_seconds."
            )
        selection_values = (
            self.sample_range,
            self.start_index,
            self.end_index,
            self.num_samples,
        )
        if self.sample_ids_path is not None and any(
            value is not None for value in selection_values
        ):
            raise ValueError(
                "sample_ids_path cannot be combined with sample_range, "
                "start_index, end_index, or num_samples."
            )
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

    def resolved_owl_model_name(self) -> str:
        if self.owl_model_name:
            return self.owl_model_name
        if self.model_name:
            return self.model_name
        assert self.model
        if self.model.startswith("openai-api/openrouter/"):
            return self.model.removeprefix("openai-api/openrouter/")
        if self.model.startswith("openrouter/"):
            return self.model.removeprefix("openrouter/")
        return self.model.rsplit("/", 1)[-1]

    def resolved_owl_api_key_env(self) -> str:
        return self.owl_api_key_env or self.model_api_key_env

    def resolved_owl_base_url(self) -> str:
        base_url = self.owl_base_url or self.model_base_url
        if not base_url:
            raise ValueError("OWL requires owl_base_url or model_base_url.")
        return base_url


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

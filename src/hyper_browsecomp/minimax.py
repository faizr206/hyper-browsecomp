"""MiniMax's direct Messages API, with provider-native search or client tools.

Protocol references (checked 2026-09-22):
https://platform.minimax.io/docs/api-reference/text-anthropic-api
https://platform.minimax.io/docs/guides/server-tools
https://platform.minimax.io/docs/api-reference/text-prompt-caching
"""

from __future__ import annotations

import functools
import json
import os
from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit

from anthropic import AsyncAnthropic
from anthropic.types import WebSearchResultBlock, WebSearchToolResultBlock
from inspect_ai.model import GenerateConfig, ModelUsage, modelapi
from inspect_ai.model._providers.anthropic import (
    AnthropicAPI,
    _ServerToolSpanRecorder,
    _split_system_as_reminders,
    consecutive_user_message_reducer,
    message_param,
    model_output_from_message,
)
from inspect_ai.tool import ToolInfo


BASE_URL = "https://api.minimax.io/anthropic"
RAW_CONTENT = "minimax_content"


def _check_headers(headers: dict[str, Any]) -> None:
    if any(name.lower() in {"authorization", "x-api-key", "anthropic-beta"} for name in headers):
        raise ValueError("MiniMax authentication comes from MINIMAX_API_KEY; custom auth/beta headers are unsupported.")


def _extra_body(body: dict[str, Any] | None) -> dict[str, Any]:
    body = deepcopy(body or {})
    if set(body) - {"service_tier", "metadata"}:
        raise ValueError("MiniMax extra_body supports only service_tier and metadata; use the adapter's model/tool/thinking settings.")
    if body.get("service_tier", "standard") not in {"standard", "priority"}:
        raise ValueError("MiniMax service_tier must be standard or priority.")
    return body


@modelapi("minimax")
class MiniMaxAPI(AnthropicAPI):
    """Direct MiniMax-M3 with unchanged thinking and search history on resume.

    M3's documented thinking control is on/off, not Claude effort levels. It is
    enabled here by default; ``thinking=False`` disables it. MiniMax's passive
    prefix cache requires no Claude cache markers and still reports cache usage.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        thinking: bool = True,
        streaming: bool = False,
        **model_args: Any,
    ) -> None:
        if model_name != "MiniMax-M3":
            raise ValueError("This direct MiniMax adapter currently supports MiniMax-M3.")
        if type(thinking) is not bool or streaming is not False:
            raise ValueError("MiniMax requires a boolean thinking setting and streaming=false.")
        url = urlsplit(base_url or os.environ.get("MINIMAX_BASE_URL") or BASE_URL)
        if (url.scheme != "https" or url.netloc != "api.minimax.io"
                or url.path.rstrip("/") != "/anthropic" or url.query or url.fragment):
            raise ValueError(f"Use {BASE_URL} for the direct MiniMax Messages API.")
        key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not key:
            raise ValueError("MINIMAX_API_KEY is required for direct MiniMax access.")
        if any(name in model_args for name in ("auth_token", "betas", "cache_ttl")):
            raise ValueError("Anthropic auth, beta, and explicit cache options do not apply to MiniMax-M3.")
        _check_headers(model_args.get("default_headers", {}))
        extra_body = _extra_body(model_args.pop("extra_body", None))
        self._minimax_key = key
        self.thinking = thinking
        model_args.setdefault("max_retries", 0)
        model_args.setdefault("timeout", config.timeout or 600.0)
        super().__init__(
            model_name=model_name, base_url=BASE_URL, api_key=key,
            config=config, streaming=False, extra_body=extra_body, **model_args,
        )

    def _create_client(self) -> AsyncAnthropic:
        client = AsyncAnthropic(
            api_key=self._minimax_key, auth_token="", base_url=BASE_URL,
            **self.model_args,
        )
        # Neither Anthropic account credentials nor its custom env headers belong
        # on this endpoint. Avoid following redirects with the MiniMax key.
        client.auth_token = None
        client._custom_headers = dict(self.model_args.get("default_headers", {}))
        client._client.follow_redirects = False
        return client

    def service_model_name(self) -> str:
        return self.model_name

    def canonical_name(self) -> str:
        return f"minimax/{self.model_name}"

    def is_claude_latest(self) -> bool:
        return False

    def is_using_thinking(self, config: GenerateConfig) -> bool:
        return self.thinking

    def max_tokens(self) -> int:
        return 32768

    def max_tokens_for_config(self, config: GenerateConfig) -> int:
        return self.max_tokens()

    def supports_remote_mcp(self) -> bool:
        return False

    def cache_diagnostics_enabled(self, config: GenerateConfig) -> bool:
        return False

    def completion_config(self, config: GenerateConfig):
        if config.batch or config.fallback_models:
            raise ValueError("Batch and model fallback are not supported by this direct MiniMax adapter.")
        if any(value is not None for value in (config.reasoning_effort, config.reasoning_tokens, config.effort)):
            raise ValueError("MiniMax-M3 exposes thinking on/off here; set model_args.thinking instead of reasoning effort/budget.")
        if config.top_k is not None or config.stop_seqs is not None:
            raise ValueError("MiniMax ignores top_k and stop_sequences; omit these settings.")
        if config.response_schema is not None:
            raise ValueError("Structured response schemas are not configured for this MiniMax adapter.")
        headers = dict(config.extra_headers or {})
        _check_headers(headers)
        params: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": config.max_tokens or self.max_tokens(),
            "thinking": {"type": "adaptive" if self.thinking else "disabled"},
        }
        for name in ("temperature", "top_p"):
            if (value := getattr(config, name)) is not None:
                params[name] = value
        return params, _extra_body(config.extra_body), headers, []

    def web_search_tool_params(self, tool: ToolInfo):
        if tool.name != "web_search" or not tool.options or "anthropic" not in tool.options:
            return None
        if tool.options["anthropic"]:
            raise ValueError("MiniMax native web search currently accepts no additional search options.")
        return [{"type": "web_search_20250305", "name": "web_search"}]

    def maybe_native_tool_params(self, tool: ToolInfo, config: GenerateConfig):
        return self.web_search_tool_params(tool) if config.internal_tools is not False else None

    async def resolve_chat_input(self, input, tools, config):
        # Replay MiniMax blocks directly. Its web results carry plaintext content,
        # not Anthropic's encrypted_content, and must survive process checkpoints.
        system, chat = _split_system_as_reminders(input)
        messages = []
        for message in chat:
            raw = (message.metadata or {}).get(RAW_CONTENT) if message.role == "assistant" else None
            messages.append(
                {"role": "assistant", "content": deepcopy(raw)}
                if raw is not None else await message_param(message)
            )
        messages = functools.reduce(consecutive_user_message_reducer, messages, [])
        tools, mcp = self.partition_tools(tools)
        if mcp:
            raise ValueError("MiniMax remote MCP tools are not supported.")
        tool_params = [p for tool in tools for p in self.tool_params_for_tool_info(tool, config)]
        system_param = [{"type": "text", "text": message.text} for message in system if message.text]
        # MiniMax-M3 uses passive caching; explicit Anthropic cache markers are
        # supported only on earlier MiniMax models, not on M3.
        return system_param or None, tool_params, [], messages, False

    async def _perform_request_and_continuations(self, request, streaming, tools, config, **_unused):
        # Passing client=None avoids Inspect making a separate count_tokens call
        # for every thinking block. Only API-reported token counts are recorded.
        pending: dict[str, Any] = {}
        spans = _ServerToolSpanRecorder()
        responses: list[dict[str, Any]] = []
        all_raw: list[dict[str, Any]] = []
        all_content = []
        usage = ModelUsage()
        all_reasoning_reported = True
        current = deepcopy(request)
        for _ in range(32):
            message = await self.client.messages.create(**current, stream=False)
            raw = message.model_dump(exclude_none=True, warnings="none")
            if raw.get("base_resp", {}).get("status_code", 0) != 0:
                raise ValueError(f"MiniMax returned API status code {raw['base_resp']['status_code']}.")
            # The SDK's lax union parser classifies plaintext search hits as
            # error objects because encrypted_content is absent. Give Inspect
            # the correct in-memory type without fabricating encrypted content.
            parsed = message.model_copy(deep=True)
            for index, block in enumerate(raw.get("content", [])):
                if block.get("type") == "web_search_tool_result" and isinstance(block.get("content"), list):
                    parsed.content[index] = WebSearchToolResultBlock.model_construct(
                        type="web_search_tool_result", tool_use_id=block["tool_use_id"],
                        content=[WebSearchResultBlock.model_construct(**hit) for hit in block["content"]],
                    )
            output, more = await model_output_from_message(
                None, self.model_name, parsed, tools,
                pending_tool_uses=pending, span_recorder=spans,
            )
            reported = raw.get("usage", {})
            details = reported.get("output_tokens_details") or {}
            reasoning = next((value for value in (
                reported.get("reasoning_tokens"), details.get("reasoning_tokens"),
                details.get("thinking_tokens"),
            ) if type(value) is int and value >= 0), None)
            all_reasoning_reported = all_reasoning_reported and reasoning is not None
            if output.usage:
                output.usage.reasoning_tokens = reasoning
            responses.append(raw)
            all_raw.extend(deepcopy(raw.get("content", [])))
            all_content.extend(output.message.content)
            usage += output.usage or ModelUsage()
            if not more:
                output.message.content = all_content
                output.message.metadata = {**(output.message.metadata or {}), RAW_CONTENT: all_raw}
                output.usage = usage
                # A partial sum is not a total: if a server continuation omits
                # reasoning usage, preserve the known pieces in minimax_usage
                # and leave the aggregate reasoning count unknown.
                if not all_reasoning_reported:
                    output.usage.reasoning_tokens = None
                output.metadata = {
                    **(output.metadata or {}),
                    "minimax_usage": [item.get("usage", {}) for item in responses],
                    "minimax_thinking": self.thinking,
                }
                # Keep plaintext search results in normalized traces as well as
                # the unchanged raw response and checkpoint replay metadata.
                by_id = {b.get("tool_use_id"): b.get("content") for b in all_raw
                         if b.get("type") == "web_search_tool_result"}
                for block in all_content:
                    if block.type == "tool_use" and block.id in by_id:
                        block.result = json.dumps(by_id[block.id], ensure_ascii=False)
                response = deepcopy(responses[0])
                if len(responses) > 1:
                    response["minimax_continuations"] = responses[1:]
                return response, output
            current["messages"] = current["messages"] + [{"role": "assistant", "content": raw["content"]}]
        raise RuntimeError("MiniMax exceeded 32 server continuation requests without completing the turn.")

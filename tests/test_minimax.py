"""Offline protocol tests: direct MiniMax, without any paid model/search calls."""

import asyncio
import json

import httpx
import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser,
    GenerateConfig, get_model,
)
from inspect_ai.tool import tool, web_search

from hyper_browsecomp.minimax import BASE_URL, MiniMaxAPI, RAW_CONTENT
from hyper_browsecomp.scorer import browse_comp_scorer
from hyper_browsecomp.task import web_research_solver


@pytest.fixture(autouse=True)
def isolated_inspect_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("inspect_ai._util.appdirs.user_data_path", lambda package: tmp_path / "data" / package)
    monkeypatch.setattr("inspect_ai._util.appdirs.user_cache_path", lambda package: tmp_path / "cache" / package)


def response(content, *, stop="end_turn", usage=None):
    return {
        "id": "offline-minimax-message", "type": "message", "role": "assistant",
        "model": "MiniMax-M3", "content": content, "stop_reason": stop,
        "usage": usage or {"input_tokens": 32, "output_tokens": 64,
                            "cache_read_input_tokens": 128, "cache_creation_input_tokens": 0},
        "base_resp": {"status_code": 0, "status_msg": ""},
    }


async def offline_model(handler, **kwargs):
    model = get_model(
        "minimax/MiniMax-M3", api_key="offline-minimax-key", memoize=False,
        config=GenerateConfig(max_retries=0, max_tokens=2048), **kwargs,
    )
    await model.api.aclose()
    model.api.model_args["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model.api.initialize()
    return model


@tool
def submit():
    async def execute(answer: str):
        """Submit an answer.

        Args:
            answer: Final answer.
        """
        return answer
    return execute


@tool(name="web_search")
def exa_search():
    async def execute(query: str):
        """Search with the direct Exa client.

        Args:
            query: Search query.
        """
        return "Offline Exa search result"
    return execute


@tool(name="web_fetch")
def exa_fetch():
    async def execute(url: str):
        """Fetch with the direct Exa client.

        Args:
            url: Page URL.
        """
        return "Offline Exa page text"
    return execute


def test_native_search_plaintext_thinking_usage_and_checkpoint_replay(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-anthropic-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "wrong-anthropic-oauth")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Authorization: Bearer wrong-header\nX-Api-Key: wrong-header")
    native_content = [
        {"type": "thinking", "thinking": "Check the evidence.", "signature": "offline-signature"},
        {"type": "server_tool_use", "id": "native-search-1", "name": "web_search", "input": {"query": "public reference"}},
        {"type": "web_search_tool_result", "tool_use_id": "native-search-1", "content": [
            {"type": "web_search_result", "url": "https://example.com/reference", "title": "Reference",
             "page_age": "2026-09-22", "content": "Complete MiniMax plaintext search evidence."}
        ]},
        {"type": "text", "text": "The source gives 2."},
    ]
    requests = []

    def respond(request):
        assert str(request.url) == BASE_URL + "/v1/messages"
        assert request.headers["x-api-key"] == "offline-minimax-key"
        assert not request.headers.get("authorization")
        assert not request.headers.get("anthropic-beta")
        requests.append(json.loads(request.content))
        content = native_content if len(requests) == 1 else [{"type": "tool_use", "id": "submit-1", "name": "submit", "input": {"answer": "2"}}]
        return httpx.Response(200, json=response(content, stop="end_turn" if len(requests) == 1 else "tool_use"))

    async def run():
        model = await offline_model(respond)
        tools = [web_search(providers="anthropic"), submit()]
        history = [ChatMessageSystem(content="Research carefully."), ChatMessageUser(content="Find the answer.")]
        try:
            first = await model.generate(history, tools=tools)
            assert first.usage.input_tokens == 32
            assert first.usage.input_tokens_cache_read == 128
            assert first.usage.output_tokens == 64
            assert first.usage.total_tokens == 224
            assert first.usage.reasoning_tokens is None  # no fabricated count/token API call
            search = next(c for c in first.message.content if c.type == "tool_use")
            assert "Complete MiniMax plaintext search evidence." in search.result
            assert first.message.metadata[RAW_CONTENT] == native_content
            restored = ChatMessageAssistant.model_validate_json(first.message.model_dump_json())
            second = await model.generate([*history, restored, ChatMessageUser(content="Submit now.")], tools=tools)
            assert second.message.tool_calls[0].function == "submit"
        finally:
            await model.api.aclose()

    asyncio.run(run())
    assert len(requests) == 2  # thinking does not create extra count_tokens calls
    for body in requests:
        assert body["tools"] == [
            {"type": "web_search_20250305", "name": "web_search"},
            next(t for t in body["tools"] if t["name"] == "submit"),
        ]
        assert body["thinking"] == {"type": "adaptive"}
        assert body["max_tokens"] == 2048
        assert not {"plugins", "provider", "output_config", "cache_control"} & body.keys()
        assert "cache_control" not in json.dumps(body)
        assert "offline-minimax-key" not in json.dumps(body)
    assert next(m for m in requests[1]["messages"] if m["role"] == "assistant")["content"] == native_content


def test_direct_exa_client_tools_and_tool_result_remain_client_side():
    requests = []
    first_content = [
        {"type": "thinking", "thinking": "Look up the reference.", "signature": "exa-signature"},
        {"type": "tool_use", "id": "exa-1", "name": "web_search", "input": {"query": "public reference"}},
    ]

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response(
            first_content if len(requests) == 1 else [{"type": "text", "text": "Answer: 2"}],
            stop="tool_use" if len(requests) == 1 else "end_turn",
        ))

    async def run():
        model = await offline_model(respond)
        tools = [exa_search(), exa_fetch(), submit()]
        history = [ChatMessageUser(content="Find the answer.")]
        try:
            first = await model.generate(history, tools=tools)
            assert first.message.tool_calls[0].function == "web_search"
            result = ChatMessageTool(content="Offline Exa search result", tool_call_id="exa-1", function="web_search")
            await model.generate([*history, first.message, result], tools=tools)
        finally:
            await model.api.aclose()

    asyncio.run(run())
    assert len(requests) == 2
    for body in requests:
        assert [t["name"] for t in body["tools"]] == ["web_search", "web_fetch", "submit"]
        assert all("input_schema" in t and "type" not in t for t in body["tools"])
    assert requests[1]["messages"][1]["content"] == first_content
    assert requests[1]["messages"][2]["content"][0]["type"] == "tool_result"
    assert requests[1]["messages"][2]["content"][0]["content"] == [{"type": "text", "text": "Offline Exa search result"}]


def test_pause_turn_accumulates_usage_and_preserves_all_raw_calls():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        first = len(requests) == 1
        content = [{"type": "server_tool_use", "id": "pause-search", "name": "web_search", "input": {"query": "public reference"}}] if first else [
            {"type": "web_search_tool_result", "tool_use_id": "pause-search", "content": []},
            {"type": "text", "text": "Done"},
        ]
        return httpx.Response(200, json=response(content, stop="pause_turn" if first else "end_turn"))

    async def run():
        model = await offline_model(respond, thinking=False)
        try:
            output = await model.generate("Search.", tools=[web_search(providers="anthropic")])
            assert output.usage.total_tokens == 448
            assert len(output.metadata["minimax_usage"]) == 2
            assert len(output.message.metadata[RAW_CONTENT]) == 3
        finally:
            await model.api.aclose()

    asyncio.run(run())
    assert len(requests) == 2
    assert requests[0]["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("reported, expected", [
    ({"output_tokens_details": {"thinking_tokens": 73}}, 73),
    ({"output_tokens_details": {"thinking_tokens": 0}}, 0),
    ({"output_tokens_details": {"reasoning_tokens": 17}}, 17),
    ({"reasoning_tokens": 39}, 39),
    ({"output_tokens_details": {}}, None),
    ({}, None),
])
def test_api_reported_thinking_tokens_are_recorded_without_estimates(reported, expected):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=response(
            [{"type": "thinking", "thinking": "Consider the problem.", "signature": "reported-thinking"},
             {"type": "text", "text": "Done"}],
            usage={"input_tokens": 100, "output_tokens": 200, **reported},
        ))

    async def run():
        model = await offline_model(respond)
        try:
            output = await model.generate("Reason about this problem.")
            assert output.usage.reasoning_tokens == expected
            assert output.usage.output_tokens == 200  # already includes reasoning
            assert output.usage.total_tokens == 300
            assert output.metadata["minimax_usage"][0] == {"input_tokens": 100, "output_tokens": 200, **reported}
        finally:
            await model.api.aclose()

    asyncio.run(run())
    assert len(requests) == 1


@pytest.mark.parametrize("thinking_counts, expected", [
    ([73, 17], 90),
    ([0, 17], 17),
    ([73, None], None),
    ([None, 17], None),
    ([None, None], None),
])
def test_continuation_reasoning_total_requires_every_request_to_report(thinking_counts, expected):
    requests = []

    def respond(request):
        requests.append(request)
        index = len(requests) - 1
        reported = {"input_tokens": 100, "output_tokens": 200}
        if thinking_counts[index] is not None:
            reported["output_tokens_details"] = {"thinking_tokens": thinking_counts[index]}
        return httpx.Response(200, json=response(
            [{"type": "text", "text": "Continue" if index == 0 else "Done"}],
            stop="pause_turn" if index == 0 else "end_turn", usage=reported,
        ))

    async def run():
        model = await offline_model(respond)
        try:
            output = await model.generate("Reason about this problem.")
            assert output.usage.reasoning_tokens == expected
            assert output.usage.output_tokens == 400
            assert output.usage.total_tokens == 600
            raw_counts = [item.get("output_tokens_details", {}).get("thinking_tokens")
                          for item in output.metadata["minimax_usage"]]
            assert raw_counts == thinking_counts
        finally:
            await model.api.aclose()

    asyncio.run(run())
    assert len(requests) == 2


def test_eval_saves_native_trace_and_grades_without_search(tmp_path):
    requests = []
    native = [
        {"type": "server_tool_use", "id": "eval-search", "name": "web_search", "input": {"query": "arithmetic reference"}},
        {"type": "web_search_tool_result", "tool_use_id": "eval-search", "content": [
            {"type": "web_search_result", "url": "https://example.com/reference", "title": "Reference", "content": "One plus one is two."},
        ]},
        {"type": "tool_use", "id": "eval-submit", "name": "submit", "input": {"answer": "Exact Answer: 2\nConfidence: 90%"}},
    ]

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response(
            native if len(requests) == 1 else [{"type": "text", "text": "extracted_final_answer: 2\nreasoning: Matches.\ncorrect: yes\nconfidence: 90"}],
            stop="tool_use" if len(requests) == 1 else "end_turn",
        ))

    model = asyncio.run(offline_model(respond))
    task = Task(
        dataset=[Sample(id="offline-minimax", input="What is one plus one?", target="2")],
        solver=web_research_solver(model_provider="minimax", search_backend="internal", fetch_backend="none", max_steps=2),
        scorer=browse_comp_scorer(model),
    )
    try:
        log = eval(task, model=model, log_dir=str(tmp_path / "logs"), display="none")[0]
    finally:
        asyncio.run(model.api.aclose())
    assert log.status == "success", log.error
    assert log.samples[0].error is None
    assert len(requests) == 2
    assert not requests[1].get("tools")
    saved = read_eval_log(log.location)
    events = [event for event in saved.samples[0].events if event.event == "model"]
    assert events[0].call.response["content"] == native
    assert events[0].output.message.metadata[RAW_CONTENT] == native
    assert events[0].output.usage.total_tokens == 224
    assert "offline-minimax-key" not in saved.model_dump_json()


@pytest.mark.parametrize("url", [
    "https://openrouter.ai/api", "https://api.anthropic.com", "http://api.minimax.io/anthropic",
    "https://api.minimax.io.evil.example/anthropic", "https://api.minimax.io/anthropic?token=unsafe",
])
def test_invalid_destinations_are_rejected(url):
    with pytest.raises(ValueError, match="direct MiniMax"):
        MiniMaxAPI("MiniMax-M3", base_url=url, api_key="offline-key")


def test_requires_minimax_key_and_ignores_anthropic_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-key")
    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        MiniMaxAPI("MiniMax-M3")


@pytest.mark.parametrize("kwargs", [
    {"thinking": "true"}, {"streaming": True}, {"betas": ["arbitrary"]},
    {"extra_body": {"tools": []}}, {"extra_body": {"model": "other-model"}},
    {"default_headers": {"Authorization": "unsafe"}},
])
def test_unsupported_provider_overrides_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MiniMaxAPI("MiniMax-M3", api_key="offline-key", **kwargs)


@pytest.mark.parametrize("config", [
    GenerateConfig(reasoning_effort="high"), GenerateConfig(reasoning_tokens=2000),
    GenerateConfig(batch=True), GenerateConfig(extra_body={"tools": []}),
    GenerateConfig(extra_headers={"x-api-key": "unsafe"}),
])
def test_unsupported_generation_overrides_fail_before_http(config):
    async def run():
        api = MiniMaxAPI("MiniMax-M3", api_key="offline-key")
        try:
            with pytest.raises(ValueError):
                api.completion_config(config)
        finally:
            await api.aclose()
    asyncio.run(run())


def test_redirect_does_not_forward_credentials():
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://example.com/collect"})

    async def run():
        model = await offline_model(respond)
        try:
            with pytest.raises(Exception):
                await model.generate("Hello")
        finally:
            await model.api.aclose()
    asyncio.run(run())
    assert requests == [BASE_URL + "/v1/messages"]

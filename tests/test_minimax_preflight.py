"""Offline checks that the public canary requires actual retrieval evidence."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS

import httpx
import pytest

from scripts import test_minimax_m3 as preflight
from scripts.test_minimax_m3 import audit_log


NATIVE_TOOL = {"name": "web_search", "type": "web_search_20250305"}


def _request(tools=None):
    return {"url": "https://api.minimax.io/anthropic/v1/messages", "model": "MiniMax-M3", "tools": tools or []}


def _native_content(*, results=True):
    return [
        {"type": "server_tool_use", "name": "web_search", "id": "search-1"},
        {"type": "web_search_tool_result", "tool_use_id": "search-1", "content":
            [{"type": "web_search_result", "url": "https://www.python.org/downloads/release/python-3130/"}]
            if results else []},
    ]


def _log(content=None, *, continuations=None):
    response = {"content": content or [], "usage": {"output_tokens_details": {"thinking_tokens": 10}}}
    if continuations:
        response["minimax_continuations"] = continuations
    usage = NS(input_tokens=100, output_tokens=20, total_tokens=145,
               input_tokens_cache_read=25, input_tokens_cache_write=0, reasoning_tokens=10)
    event = NS(event="model", call=NS(response=response), output=NS(usage=usage))
    sample = NS(events=[event], scores={"browse_comp_scorer": NS(value={"score": "C"})},
                completed_at="2026-09-22T00:00:00Z", error=None)
    return NS(samples=[sample], status="success", location="public.eval")


def test_native_declaration_without_results_does_not_verify():
    report = audit_log(_log(), "native", [_request([NATIVE_TOOL]), _request()], [])
    assert not report["verified"]
    assert report["checks"]["exact_native_tool"]
    assert not report["checks"]["native_search_executed"]


def test_empty_native_search_does_not_verify():
    report = audit_log(_log(_native_content(results=False)), "native", [_request([NATIVE_TOOL]), _request()], [])
    assert not report["verified"]
    assert report["native_search_calls"] == 1
    assert report["native_search_results"] == 0


def test_native_results_in_continuation_are_counted():
    report = audit_log(_log(continuations=[{"content": _native_content()}]),
                       "native", [_request([NATIVE_TOOL]), _request()], [])
    assert report["verified"]
    assert report["native_search_calls"] == 1
    assert report["native_search_results"] == 1


def test_positive_native_evidence_uses_total_without_double_counting():
    report = audit_log(_log(_native_content()), "native", [_request([NATIVE_TOOL]), _request()], [])
    assert report["verified"]
    assert report["usage"]["total_tokens"] == 145
    assert report["usage"]["input_tokens_cache_read"] == 25
    assert report["usage"]["reasoning_tokens"] == 10


def test_unreported_reasoning_is_unknown_and_reported_zero_is_zero():
    log = _log(_native_content())
    response = log.samples[0].events[0].call.response
    response["usage"] = {}
    requests = [_request([NATIVE_TOOL]), _request()]
    usage = audit_log(log, "native", requests, [])["usage"]
    assert usage["reasoning_tokens"] is None
    assert not usage["reasoning_tokens_complete"]
    assert usage["reported_reasoning_tokens"] == 0
    response["usage"] = {"output_tokens_details": {"thinking_tokens": 0}}
    usage = audit_log(log, "native", requests, [])["usage"]
    assert usage["reasoning_tokens"] == 0
    assert usage["reasoning_tokens_complete"]
    response["minimax_continuations"] = [{"usage": {}}]
    response["usage"]["output_tokens_details"]["thinking_tokens"] = 10
    usage = audit_log(log, "native", requests, [])["usage"]
    assert usage["reasoning_tokens"] is None
    assert usage["reported_reasoning_tokens"] == 10


def test_exa_requires_actual_successful_search_and_fetch():
    search = {"tool": "web_search", "ok": True, "has_content": True}
    fetch = {"tool": "web_fetch", "ok": True, "has_content": True}
    report = audit_log(_log(), "exa", [_request()], [search])
    assert not report["verified"]
    assert not report["checks"]["direct_exa_fetch_executed"]
    assert audit_log(_log(), "exa", [_request()], [search, fetch])["verified"]
    failed = {**fetch, "ok": False}
    assert not audit_log(_log(), "exa", [_request()], [search, fetch, failed])["verified"]


def test_native_cannot_verify_with_exa_execution_or_unexpected_endpoint():
    requests = [_request([NATIVE_TOOL]), _request()]
    execution = {"tool": "web_search", "ok": True, "has_content": True}
    assert not audit_log(_log(_native_content()), "native", requests, [execution])["verified"]
    requests[0]["url"] = "https://openrouter.ai/api/v1/chat/completions"
    report = audit_log(_log(_native_content()), "native", requests, [])
    assert not report["verified"]
    assert not report["checks"]["only_direct_minimax_requests"]


@pytest.mark.parametrize("setup", ["native", "exa"])
def test_public_canary_complete_flow_uses_real_inspect_with_mocked_http(tmp_path, monkeypatch, setup):
    """Exercise run_check itself, including eval_async's actual accepted kwargs."""
    config = preflight.load_run_config(preflight.ROOT / preflight.CONFIGS[setup])
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight, "load_run_config", lambda _: config)
    # Inspect normally reloads .env (and may override supplied variables in an
    # IDE). Offline tests must not read real credentials from the repository.
    monkeypatch.setattr("inspect_ai._util.dotenv.find_dotenv", lambda **_: "")
    monkeypatch.setattr("inspect_ai._util.dotenv.dotenv_values", lambda *_: {})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINIMAX_API_KEY", "offline-minimax-preflight")
    monkeypatch.setenv("EXA_API_KEY", "offline-exa-preflight")
    page = "https://www.python.org/downloads/release/python-3130/"
    answer = f"Exact Answer: October 7, 2024\nConfidence: 100\nSource: {page}"
    calls = []
    model_calls = 0

    def tool_call(name, arguments):
        return {"type": "tool_use", "id": f"tool-{model_calls}", "name": name, "input": arguments}

    def respond(request):
        nonlocal model_calls
        url = str(request.url)
        calls.append(url)
        body = json.loads(request.content)
        if url in {"https://api.exa.ai/search", "https://api.exa.ai/contents"}:
            assert setup == "exa"
            uses_dummy_key = request.headers.get("x-api-key") == "offline-exa-preflight"
            assert uses_dummy_key
            return httpx.Response(200, json={"results": [{"url": page, "title": "Python 3.13.0",
                                                           "text": "Release date: October 7, 2024."}]})
        assert url == "https://api.minimax.io/anthropic/v1/messages"
        uses_dummy_key = request.headers.get("x-api-key") == "offline-minimax-preflight"
        assert uses_dummy_key
        model_calls += 1
        stop = "tool_use"
        if not body.get("tools"):
            content = [{"type": "text", "text": "extracted_final_answer: October 7, 2024\nreasoning: Matches.\ncorrect: yes\nconfidence: 100"}]
            stop = "end_turn"
        elif setup == "native" and model_calls == 1:
            content = [
                {"type": "server_tool_use", "id": "native-search", "name": "web_search", "input": {"query": "Python 3.13.0 release date"}},
                {"type": "web_search_tool_result", "tool_use_id": "native-search", "content": [
                    {"type": "web_search_result", "url": page, "title": "Python 3.13.0", "content": "Release date: October 7, 2024."}
                ]},
                {"type": "text", "text": "The official release page supplies the date."},
            ]
            stop = "end_turn"
        elif setup == "exa" and model_calls == 1:
            content = [tool_call("web_search", {"query": "Python 3.13.0 release date"})]
        elif setup == "exa" and model_calls == 2:
            content = [tool_call("web_fetch", {"url": page})]
        else:
            content = [tool_call("submit", {"answer": answer})]
        return httpx.Response(200, json={
            "id": f"offline-message-{model_calls}", "type": "message", "role": "assistant",
            "model": "MiniMax-M3", "content": content, "stop_reason": stop,
            "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 25},
        })

    original_client = httpx.AsyncClient

    class OfflineClient(original_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(respond)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", OfflineClient)
    report = asyncio.run(preflight.run_check(setup, tmp_path / "logs"))
    assert report["verified"], report
    assert report["model_requests"] == (3 if setup == "native" else 4)
    assert report["usage"]["total_tokens"] == model_calls * 145
    assert calls.count("https://api.exa.ai/search") == (1 if setup == "exa" else 0)
    assert calls.count("https://api.exa.ai/contents") == (1 if setup == "exa" else 0)

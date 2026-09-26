"""Bounded public-question checks of the direct MiniMax configs; no benchmark data."""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import sys
from collections import Counter
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx
from dotenv import load_dotenv
from inspect_ai import Task, eval_async
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model

from hyper_browsecomp.config import load_run_config
from hyper_browsecomp.scorer import browse_comp_scorer
from hyper_browsecomp.task import web_research_solver
from hyper_browsecomp.tools import web


MAX_REQUESTS = 6
CONFIGS = {"native": "configs/internal_full/minimax_m3.yaml", "exa": "configs/exa_full/minimax_m3.yaml"}


def audit_log(log, setup: str, requests: list[dict], executions: list[dict]) -> dict:
    """Report recorded usage and positive retrieval evidence, not just tool declarations."""
    samples = log.samples or []
    events = [event for sample in samples for event in sample.events if event.event == "model"]
    native_calls = native_results = 0
    reasoning_reports: list[int | None] = []
    for event in events:
        response = (event.call.response or {}) if event.call else {}
        replies = [response, *response.get("minimax_continuations", [])]
        for reply in replies:
            reported = reply.get("usage", {})
            details = reported.get("output_tokens_details") or {}
            reasoning = reported.get("reasoning_tokens", details.get("reasoning_tokens", details.get("thinking_tokens")))
            reasoning_reports.append(reasoning if type(reasoning) is int and reasoning >= 0 else None)
        blocks = [block for reply in replies
                  for block in reply.get("content", [])]
        for block in blocks:
            if block.get("type") == "server_tool_use" and block.get("name") == "web_search":
                native_calls += 1
            if block.get("type") == "web_search_tool_result" and isinstance(block.get("content"), list):
                native_results += sum(
                    result.get("type") == "web_search_result" and bool(result.get("url"))
                    for result in block["content"]
                )
    usage = Counter()
    for event in events:
        if event.output and event.output.usage:
            for name in ("input_tokens", "output_tokens", "total_tokens", "input_tokens_cache_read", "input_tokens_cache_write"):
                usage[name] += getattr(event.output.usage, name, None) or 0
    complete_reasoning = bool(reasoning_reports) and all(value is not None for value in reasoning_reports)
    reported_reasoning = sum(value for value in reasoning_reports if value is not None)
    token_usage = {**usage, "reasoning_tokens": reported_reasoning if complete_reasoning else None,
                   "reported_reasoning_tokens": reported_reasoning,
                   "reasoning_tokens_complete": complete_reasoning}
    score = (samples[0].scores or {}).get("browse_comp_scorer") if len(samples) == 1 else None
    correct = bool(score and isinstance(score.value, dict) and score.value.get("score") == "C")
    native_schemas = [tool for request in requests for tool in request["tools"] if tool.get("type", "").startswith("web_search_")]
    checks = {
        "one_complete_public_question": bool(log.status == "success" and len(samples) == 1 and samples[0].completed_at and not samples[0].error),
        "correct_release_date": correct,
        "only_direct_minimax_requests": bool(requests) and all(item["url"] == "https://api.minimax.io/anthropic/v1/messages" and item["model"] == "MiniMax-M3" for item in requests),
        "bounded_model_requests": 0 < len(requests) <= MAX_REQUESTS,
        "positive_token_usage": usage["total_tokens"] > 0,
        "grader_has_no_search_tools": bool(requests) and not requests[-1]["tools"] and correct,
    }
    if setup == "native":
        checks.update(
            native_search_executed=native_calls > 0 and native_results > 0,
            exact_native_tool=bool(native_schemas) and all(tool == {"name": "web_search", "type": "web_search_20250305"} for tool in native_schemas),
            no_exa_execution=not executions,
        )
    else:
        checks.update(
            direct_exa_search_executed=any(item["tool"] == "web_search" and item["ok"] and item["has_content"] for item in executions),
            direct_exa_fetch_executed=any(item["tool"] == "web_fetch" and item["ok"] and item["has_content"] for item in executions),
            no_native_search=not native_schemas and native_calls == 0,
            no_exa_errors=bool(executions) and all(item["ok"] for item in executions),
        )
    return {
        "verified": all(checks.values()), "checks": checks, "status": log.status,
        "eval_file": str(log.location), "model_requests": len(requests),
        "native_search_calls": native_calls, "native_search_results": native_results,
        "exa_executions": executions, "usage": token_usage,
        "usage_note": "Inspect input tokens exclude separately recorded cache reads/writes; total_tokens includes them. Reasoning is already included in output; null means at least one response omitted its separate count, and reported_reasoning_tokens is only the known subtotal.",
    }


async def run_check(setup: str, destination: Path) -> dict:
    config = load_run_config(ROOT / CONFIGS[setup])
    requests: list[dict] = []
    executions: list[dict] = []
    counts: Counter = Counter()

    async def observe_request(request):
        if str(request.url) != "https://api.minimax.io/anthropic/v1/messages" or len(requests) >= MAX_REQUESTS:
            raise RuntimeError("MiniMax public check reached its endpoint/request limit.")
        body = json.loads(request.content)
        requests.append({"url": str(request.url), "model": body.get("model"), "tools": body.get("tools", [])})

    def bounded_tool(original, name):
        async def execute(*args, **kwargs):
            counts[name] += 1
            if counts[name] > 1:
                executions.append({"tool": name, "ok": False, "has_content": False, "blocked_by_limit": True})
                raise RuntimeError("Public check permits only one Exa search and one fetch.")
            try:
                result = await original(*args, **kwargs)
                content = result.get("results") if name == "web_search" else result.get("content")
                executions.append({"tool": name, "ok": True, "has_content": bool(content)})
                return result
            except Exception:
                executions.append({"tool": name, "ok": False, "has_content": False})
                raise
        return execute

    instruction = "Search exactly once for the official python.org release page for Python 3.13.0. "
    if setup == "exa":
        instruction += "Then use web_fetch exactly once to read that official page. "
    instruction += "What exact release date does it give? Use the tools even if you recall the date, cite the official URL, and submit your answer."
    with ExitStack() as stack:
        stack.enter_context(patch("inspect_ai._util.appdirs.user_data_path", lambda package: ROOT / ".inspect_home/data" / package))
        stack.enter_context(patch("inspect_ai._util.appdirs.user_cache_path", lambda package: ROOT / ".inspect_home/cache" / package))
        stack.enter_context(patch.object(web, "_exa_search", bounded_tool(web._exa_search, "web_search")))
        stack.enter_context(patch.object(web, "_exa_fetch", bounded_tool(web._exa_fetch, "web_fetch")))
        model = get_model(config.resolved_model(), base_url=config.model_base_url,
                          api_key=os.environ["MINIMAX_API_KEY"], memoize=False,
                          config=GenerateConfig(max_tokens=4096, max_retries=0, timeout=120),
                          **config.model_args)
        await model.api.aclose()
        client = httpx.AsyncClient(timeout=120, follow_redirects=False, event_hooks={"request": [observe_request]})
        model.api.model_args["http_client"] = client
        model.api.initialize()
        try:
            task = Task(
                name=f"minimax_m3_{setup}_public_check",
                dataset=[Sample(id="python-313-public", input=instruction, target="October 7, 2024")],
                solver=web_research_solver(model_provider="minimax", search_backend=config.search_backend,
                                          fetch_backend=config.fetch_backend, max_steps=3),
                scorer=browse_comp_scorer(model),
            )
            logs = await eval_async(task, model=model, log_dir=str(destination),
                                    max_samples=1, retry_on_error=0, fail_on_error=True,
                                    checkpoint=False, log_buffer=1)
            return audit_log(logs[0], setup, requests, executions)
        finally:
            await model.api.aclose()
            await client.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", choices=["native", "exa", "both"], default="both")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    required = ["MINIMAX_API_KEY"] + (["EXA_API_KEY"] if args.setup in {"exa", "both"} else [])
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        print(json.dumps({"status": "blocked", "missing_environment_variables": missing}))
        return 2
    destination = ROOT / "dump/minimax_m3_preflight" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    destination.mkdir(parents=True)
    setups = ["native", "exa"] if args.setup == "both" else [args.setup]
    receipt = {"benchmark_questions": 0, "public_questions": len(setups), "max_model_requests_per_setup": MAX_REQUESTS, "results": {}}
    old_logging = logging.root.manager.disable
    for setup in setups:
        print(f"Checking MiniMax {setup} on one public question…", flush=True)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                logging.disable(logging.CRITICAL)
                result = asyncio.run(run_check(setup, destination / setup))
        except Exception as error:
            message = str(error)
            for name, secret in os.environ.items():
                if secret and (name.endswith("API_KEY") or name.endswith("AUTH_TOKEN")):
                    message = message.replace(secret, "[REDACTED]")
            result = {"verified": False, "status": "error", "error_type": type(error).__name__, "message": message[:2000]}
        finally:
            logging.disable(old_logging)
        receipt["results"][setup] = result
        (destination / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"setup": setup, **result}, indent=2), flush=True)
    print(f"Receipt: {destination / 'verification.json'}")
    return 0 if all(result["verified"] for result in receipt["results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
from typing import Any, Literal

import httpx
from inspect_ai.tool import Tool, tool


SearchBackend = Literal["exa", "firecrawl"]
FetchBackend = Literal["exa", "firecrawl"]


class BackendError(RuntimeError):
    pass


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise BackendError(f"{name} is required for this backend.")
    return value


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... truncated to {max_chars} chars ..."


async def _exa_search(query: str, *, max_results: int, timeout_seconds: int) -> dict[str, Any]:
    api_key = _require_env("EXA_API_KEY")
    payload = {
        "query": query,
        "numResults": max_results,
        "contents": {
            "text": True,
            "highlights": True,
        },
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    results = []
    for item in data.get("results", [])[:max_results]:
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("text") or "",
                "highlights": item.get("highlights") or [],
                "published_date": item.get("publishedDate"),
            }
        )

    return {"backend": "exa", "query": query, "results": results}


async def _firecrawl_search(query: str, *, max_results: int, timeout_seconds: int) -> dict[str, Any]:
    api_key = _require_env("FIRECRAWL_API_KEY")
    payload = {
        "query": query,
        "limit": max_results,
        "sources": ["web"],
        "timeout": timeout_seconds * 1000,
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            "https://api.firecrawl.dev/v2/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    web_results = ((data.get("data") or {}).get("web") or [])[:max_results]
    results = []
    for item in web_results:
        metadata = item.get("metadata") or {}
        results.append(
            {
                "title": item.get("title") or metadata.get("title"),
                "url": item.get("url") or metadata.get("url") or metadata.get("sourceURL"),
                "snippet": item.get("description") or metadata.get("description") or "",
                "highlights": [],
                "published_date": metadata.get("publishedTime") or metadata.get("ogPublishedTime"),
            }
        )
    return {"backend": "firecrawl", "query": query, "results": results}


async def _exa_fetch(url: str, *, timeout_seconds: int, max_chars: int) -> dict[str, Any]:
    api_key = _require_env("EXA_API_KEY")
    payload = {
        "urls": [url],
        "text": True,
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            "https://api.exa.ai/contents",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    item = (data.get("results") or [{}])[0]
    text = item.get("text") or ""
    return {
        "backend": "exa",
        "url": item.get("url") or url,
        "title": item.get("title"),
        "content": _truncate_text(text, max_chars),
        "metadata": {
            "author": item.get("author"),
            "published_date": item.get("publishedDate"),
        },
    }


async def _firecrawl_fetch(url: str, *, timeout_seconds: int, max_chars: int) -> dict[str, Any]:
    api_key = _require_env("FIRECRAWL_API_KEY")
    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "timeout": timeout_seconds * 1000,
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            "https://api.firecrawl.dev/v2/scrape",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    item = data.get("data") or {}
    metadata = item.get("metadata") or {}
    content = item.get("markdown") or item.get("html") or ""
    return {
        "backend": "firecrawl",
        "url": metadata.get("url") or metadata.get("sourceURL") or url,
        "title": metadata.get("title"),
        "content": _truncate_text(content, max_chars),
        "metadata": {
            "description": metadata.get("description"),
            "status_code": metadata.get("statusCode"),
            "language": metadata.get("language"),
        },
    }


def build_web_search_tool(
    *,
    backend: SearchBackend,
    max_results: int,
    timeout_seconds: int,
) -> Tool:
    @tool(name="web_search")
    def web_search() -> Tool:
        async def execute(query: str, max_results_override: int | None = None) -> dict[str, Any]:
            """Search the web and return candidate sources.

            Args:
                query: Search query.
                max_results_override: Optional per-call result limit.
            """

            selected_max_results = max_results_override or max_results
            if backend == "exa":
                return await _exa_search(
                    query,
                    max_results=selected_max_results,
                    timeout_seconds=timeout_seconds,
                )
            return await _firecrawl_search(
                query,
                max_results=selected_max_results,
                timeout_seconds=timeout_seconds,
            )

        return execute

    return web_search()


def build_web_fetch_tool(
    *,
    backend: FetchBackend,
    timeout_seconds: int,
    max_chars: int,
) -> Tool:
    @tool(name="web_fetch")
    def web_fetch() -> Tool:
        async def execute(url: str, max_chars_override: int | None = None) -> dict[str, Any]:
            """Fetch a web page and return compact page content.

            Args:
                url: URL to fetch.
                max_chars_override: Optional per-call content limit.
            """

            selected_max_chars = max_chars_override or max_chars
            if backend == "exa":
                return await _exa_fetch(
                    url,
                    timeout_seconds=timeout_seconds,
                    max_chars=selected_max_chars,
                )
            return await _firecrawl_fetch(
                url,
                timeout_seconds=timeout_seconds,
                max_chars=selected_max_chars,
            )

        return execute

    return web_fetch()

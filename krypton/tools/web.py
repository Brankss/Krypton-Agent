"""Web tools: search + fetch.

Key perf points:
  * One process-wide httpx.AsyncClient with HTTP/2 + connection pooling —
    every search/fetch reuses TCP+TLS, saving 100-500ms per call.
  * 5-minute LRU-ish cache keyed on query/url.
  * Tavily-first when TAVILY_API_KEY is set (richer answers); DDG fallback.
  * Snippets capped at 240 chars each — keeps the agent context tight.
  * fetch_url strips script/style/svg/form noise via selectolax (C-speed).
"""
from __future__ import annotations

import asyncio
import atexit
import time
from typing import Any

import httpx
from selectolax.parser import HTMLParser

from krypton.config import settings
from krypton.tools.base import BaseTool, ToolResult

# ---------------------------------------------------------------------------
# Shared HTTP client — created lazily, reused everywhere.
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            http2=False,  # most search backends don't benefit; skip ALPN cost
            follow_redirects=True,
            timeout=httpx.Timeout(20.0, connect=5.0),
            headers={"User-Agent": "KryptonAgent/1.0 (+local)"},
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )
    return _client


@atexit.register
def _close_client() -> None:  # best-effort cleanup
    global _client
    if _client is not None and not _client.is_closed:
        try:
            import anyio
            anyio.from_thread.run(_client.aclose)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# In-process cache
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 300.0


def _cache_get(key: str) -> str | None:
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]
    if hit:
        _CACHE.pop(key, None)
    return None


def _cache_set(key: str, value: str) -> None:
    if len(_CACHE) > 256:
        oldest = min(_CACHE.items(), key=lambda kv: kv[1][0])[0]
        _CACHE.pop(oldest, None)
    _CACHE[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web (Tavily if TAVILY_API_KEY is set, else DuckDuckGo). "
        "Returns title + URL + tight snippet for each result. Cached 5 min."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20},
            "region": {"type": "string", "default": "wt-wt", "description": "DDG region (e.g. it-it)."},
        },
        "required": ["query"],
    }
    timeout_s = 15.0

    async def run(self, query: str, max_results: int = 8, region: str = "wt-wt") -> ToolResult:
        q = query.strip()
        if not q:
            return ToolResult.failure("empty query")
        key = f"search::{region}::{max_results}::{q.lower()}"
        if (cached := _cache_get(key)) is not None:
            return ToolResult.success(cached, meta={"cached": True})

        if settings.tavily_api_key:
            text = await _tavily_search(q, max_results)
        else:
            text = await _ddg_search(q, max_results, region)
        _cache_set(key, text)
        return ToolResult.success(text)


class FetchUrlTool(BaseTool):
    name = "fetch_url"
    description = (
        "GET a URL and return readable text (HTML stripped of script/style/svg/form). "
        "Use after web_search to read a specific article/doc. Capped at 12000 chars by default."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "default": 12000, "minimum": 500, "maximum": 60000},
        },
        "required": ["url"],
    }
    timeout_s = 25.0

    async def run(self, url: str, max_chars: int = 12000) -> ToolResult:
        key = f"fetch::{max_chars}::{url}"
        if (cached := _cache_get(key)) is not None:
            return ToolResult.success(cached, meta={"cached": True})
        client = _get_client()
        try:
            r = await client.get(url)
        except httpx.HTTPError as e:
            return ToolResult.failure(f"http error: {e}")
        if r.status_code >= 400:
            return ToolResult.failure(f"HTTP {r.status_code}")
        ctype = r.headers.get("content-type", "").lower()
        text = _html_to_text(r.text) if "html" in ctype else r.text
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... [truncated, {len(text) - max_chars} chars omitted]"
        out = f"URL: {url}\nStatus: {r.status_code}\nContent-Type: {ctype}\n\n{text}"
        _cache_set(key, out)
        return ToolResult.success(out)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


_SNIPPET = 240


async def _ddg_search(query: str, max_results: int, region: str) -> str:
    # duckduckgo_search is sync; run in thread to keep the loop free.
    from duckduckgo_search import DDGS

    def _do() -> list[dict[str, str]]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, region=region, max_results=max_results))

    results = await asyncio.to_thread(_do)
    if not results:
        return f"no results for: {query}"
    out: list[str] = [f"DuckDuckGo — {len(results)} result(s) for: {query}"]
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        body = (r.get("body") or "").strip()
        if len(body) > _SNIPPET:
            body = body[:_SNIPPET].rstrip() + "…"
        out.append(f"[{i}] {title}\n    {url}\n    {body}")
    return "\n".join(out)


async def _tavily_search(query: str, max_results: int) -> str:
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
        "include_raw_content": False,
    }
    try:
        r = await _get_client().post("https://api.tavily.com/search", json=payload)
        r.raise_for_status()
    except httpx.HTTPError as e:
        return f"tavily error: {e}"
    data = r.json()
    out: list[str] = []
    if ans := data.get("answer"):
        out.append(f"Tavily answer: {ans}\n")
    for i, item in enumerate(data.get("results") or [], start=1):
        content = (item.get("content") or "").strip()
        if len(content) > _SNIPPET:
            content = content[:_SNIPPET].rstrip() + "…"
        out.append(f"[{i}] {item.get('title')}\n    {item.get('url')}\n    {content}")
    return "\n".join(out) or "no results"


def _html_to_text(html: str) -> str:
    try:
        tree = HTMLParser(html)
    except Exception:  # noqa: BLE001
        return html
    for sel in ("script", "style", "noscript", "svg", "form", "iframe"):
        for node in tree.css(sel):
            node.decompose()
    body = tree.body or tree.root
    if body is None:
        return ""
    text = body.text(separator="\n", strip=True)
    # collapse runs of blank lines
    out: list[str] = []
    blank = False
    for ln in (l.strip() for l in text.splitlines()):
        if not ln:
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out)


def tools() -> list[Any]:
    return [WebSearchTool(), FetchUrlTool()]

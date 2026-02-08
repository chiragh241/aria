"""Research skill — search Reddit, X (Twitter), and general web."""

import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

import httpx

from ..base import BaseSkill, SkillResult
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ResearchSkill(BaseSkill):
    """
    Research topics across Reddit, X (Twitter), and the web.

    Runs searches in parallel across sources when possible.
    """

    name = "research"
    description = "Search Reddit, X (Twitter), and the web for any topic"
    version = "1.0.0"

    REDDIT_SEARCH = "https://www.reddit.com/search.json"
    NITTER_SEARCH = "https://nitter.net/search/rss"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._x_api_key = os.environ.get("X_BEARER_TOKEN", "")  # Optional for X API

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="search_topic",
            description="Research a topic across Reddit, X, and web. Returns combined results.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic or search query"},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["reddit", "web", "x"]},
                        "description": "Which sources to search (default: all)",
                    },
                    "limit": {"type": "integer", "description": "Max results per source (default: 5)"},
                },
                "required": ["query"],
            },
        )
        self.register_capability(
            name="search_reddit",
            description="Search Reddit for a topic",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
                "required": ["query"],
            },
        )
        self.register_capability(
            name="search_web",
            description="Search the web for a topic",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
                "required": ["query"],
            },
        )
        self.register_capability(
            name="search_x",
            description="Search X (Twitter) for a topic. Requires X_BEARER_TOKEN for full API.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
                "required": ["query"],
            },
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        start = datetime.now(timezone.utc)
        query = kwargs.get("query", "").strip()
        if not query:
            return self._error_result("Query required", start)

        if capability == "search_topic":
            sources = kwargs.get("sources") or ["reddit", "web", "x"]
            limit = kwargs.get("limit", 5)
            return await self._search_all(query, sources, limit, start)
        elif capability == "search_reddit":
            return await self._search_reddit(query, kwargs.get("limit", 5), start)
        elif capability == "search_web":
            return await self._search_web(query, kwargs.get("limit", 5), start)
        elif capability == "search_x":
            return await self._search_x(query, kwargs.get("limit", 5), start)
        return self._error_result(f"Unknown capability: {capability}", start)

    async def _search_all(
        self, query: str, sources: list[str], limit: int, start: datetime
    ) -> SkillResult:
        """Run searches in parallel across selected sources."""
        import asyncio

        results: dict[str, str] = {}
        tasks = []

        if "reddit" in sources:
            tasks.append(("reddit", self._fetch_reddit(query, limit)))
        if "web" in sources:
            tasks.append(("web", self._fetch_web(query, limit)))
        if "x" in sources:
            tasks.append(("x", self._fetch_x(query, limit)))

        if not tasks:
            return self._error_result("No sources selected", start)

        source_names, coros = zip(*tasks)
        try:
            fetched = await asyncio.gather(*coros, return_exceptions=True)
            for name, data in zip(source_names, fetched):
                if isinstance(data, Exception):
                    results[name] = f"(Error: {data})"
                else:
                    results[name] = data
        except Exception as e:
            return self._error_result(str(e), start)

        # Format combined output
        lines = [f"## Research: {query}\n"]
        for src, content in results.items():
            lines.append(f"### {src.upper()}\n{content}\n")
        return self._success_result("\n".join(lines), start)

    async def _fetch_reddit(self, query: str, limit: int) -> str:
        """Fetch Reddit search results."""
        try:
            async with httpx.AsyncClient(
                timeout=15,
                headers={"User-Agent": "AriaBot/1.0 (research bot)"},
            ) as client:
                resp = await client.get(
                    self.REDDIT_SEARCH,
                    params={"q": query, "limit": min(limit, 25), "sort": "relevance"},
                )
                if resp.status_code != 200:
                    return f"Reddit returned {resp.status_code}"
                data = resp.json()
                posts = data.get("data", {}).get("children", [])[:limit]
                lines = []
                for p in posts:
                    d = p.get("data", {})
                    title = d.get("title", "")
                    sub = d.get("subreddit", "")
                    score = d.get("score", 0)
                    url = f"https://reddit.com{d.get('permalink', '')}"
                    lines.append(f"- [{title}] ({sub}, {score} upvotes) {url}")
                return "\n".join(lines) if lines else "No Reddit results found."
        except Exception as e:
            logger.debug("Reddit search failed", error=str(e))
            return f"Reddit search failed: {e}"

    async def _fetch_web(self, query: str, limit: int) -> str:
        """Fetch web search results via DuckDuckGo Instant Answer API."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json"},
                )
                if resp.status_code != 200:
                    return f"Web search returned {resp.status_code}"
                data = resp.json()
                lines = []
                # Abstract (summary)
                if data.get("Abstract"):
                    lines.append(f"Summary: {data['Abstract'][:300]}")
                    if data.get("AbstractURL"):
                        lines.append(f"Source: {data['AbstractURL']}")
                # Related topics
                for t in data.get("RelatedTopics", [])[:limit]:
                    if isinstance(t, dict) and t.get("Text") and t.get("FirstURL"):
                        lines.append(f"- {t['Text'][:80]}: {t['FirstURL']}")
                    elif isinstance(t, dict) and "Topics" in t:
                        for st in t.get("Topics", [])[:2]:
                            if st.get("Text") and st.get("FirstURL"):
                                lines.append(f"- {st['Text'][:80]}: {st['FirstURL']}")
                return "\n".join(lines) if lines else "No web results found."
        except Exception as e:
            logger.debug("Web search failed", error=str(e))
            return f"Web search failed: {e}"

    async def _fetch_x(self, query: str, limit: int) -> str:
        """Fetch X/Twitter search results. Uses Nitter RSS when no API key."""
        if self._x_api_key:
            return await self._fetch_x_api(query, limit)
        return await self._fetch_x_nitter(query, limit)

    async def _fetch_x_api(self, query: str, limit: int) -> str:
        """Fetch via X API v2 (requires bearer token)."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.twitter.com/2/tweets/search/recent",
                    params={"query": query, "max_results": min(limit, 10)},
                    headers={"Authorization": f"Bearer {self._x_api_key}"},
                )
                if resp.status_code != 200:
                    return f"X API returned {resp.status_code}"
                data = resp.json()
                tweets = data.get("data", [])
                lines = [f"- {t.get('text', '')[:100]}..." for t in tweets[:limit]]
                return "\n".join(lines) if lines else "No X results found."
        except Exception as e:
            return f"X API error: {e}"

    async def _fetch_x_nitter(self, query: str, limit: int) -> str:
        """Fetch via Nitter RSS (no API key)."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                url = f"{self.NITTER_SEARCH}?f=tweets&q={quote_plus(query)}"
                resp = await client.get(url, headers={"User-Agent": "AriaBot/1.0"})
                if resp.status_code != 200:
                    return f"Nitter returned {resp.status_code}. X search requires X_BEARER_TOKEN for API."
                try:
                    import feedparser
                    feed = feedparser.parse(resp.text)
                    entries = feed.get("entries", [])[:limit]
                    lines = [f"- {e.get('title', '')[:100]} - {e.get('link', '')}" for e in entries]
                    return "\n".join(lines) if lines else "No X results. Set X_BEARER_TOKEN for API access."
                except ImportError:
                    return "Install feedparser for X via Nitter, or set X_BEARER_TOKEN for API."
        except Exception as e:
            logger.debug("X/Nitter search failed", error=str(e))
            return f"X search failed: {e}. Set X_BEARER_TOKEN for Twitter API."

    async def _search_reddit(self, query: str, limit: int, start: datetime) -> SkillResult:
        out = await self._fetch_reddit(query, limit)
        return self._success_result(out, start)

    async def _search_web(self, query: str, limit: int, start: datetime) -> SkillResult:
        out = await self._fetch_web(query, limit)
        return self._success_result(out, start)

    async def _search_x(self, query: str, limit: int, start: datetime) -> SkillResult:
        out = await self._fetch_x(query, limit)
        return self._success_result(out, start)

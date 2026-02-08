"""News skill for fetching headlines and articles."""

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from ..base import BaseSkill, SkillResult
from ...utils.logging import get_logger

logger = get_logger(__name__)


class NewsSkill(BaseSkill):
    """Fetch news headlines, search articles, and summarize content."""

    name = "news"
    description = "Get top news headlines, search news by topic, and summarize articles"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._api_key = os.environ.get("NEWS_API_KEY", "")
        self._base_url = "https://newsapi.org/v2"

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="headlines",
            description="Get top news headlines, optionally by category or country",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "News category: business, entertainment, general, health, science, sports, technology",
                        "enum": ["business", "entertainment", "general", "health", "science", "sports", "technology"],
                    },
                    "country": {
                        "type": "string",
                        "description": "2-letter country code (e.g. 'us', 'gb', 'in'). Default: us",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of headlines to return (max 10). Default: 5",
                    },
                },
            },
        )
        self.register_capability(
            name="search",
            description="Search news articles by keyword or topic",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results (max 10). Default: 5",
                    },
                },
                "required": ["query"],
            },
        )
        self.register_capability(
            name="summarize",
            description="Fetch and extract text from a news article URL",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The article URL to fetch and summarize",
                    },
                },
                "required": ["url"],
            },
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        start = datetime.now(timezone.utc)
        if capability == "headlines":
            return await self._headlines(kwargs, start)
        elif capability == "search":
            return await self._search(kwargs, start)
        elif capability == "summarize":
            return await self._summarize(kwargs.get("url", ""), start)
        return self._error_result(f"Unknown capability: {capability}", start)

    async def _headlines(self, params: dict, start: datetime) -> SkillResult:
        category = params.get("category", "general")
        country = params.get("country", "us")
        count = min(params.get("count", 5), 10)

        # Try NewsAPI first
        if self._api_key:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(
                        f"{self._base_url}/top-headlines",
                        params={
                            "apiKey": self._api_key,
                            "category": category,
                            "country": country,
                            "pageSize": count,
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        articles = data.get("articles", [])
                        return self._format_articles(articles, start)
            except Exception as e:
                logger.debug("NewsAPI failed, trying RSS", error=str(e))

        # Fallback: RSS feed via Google News
        return await self._rss_headlines(category, count, start)

    async def _rss_headlines(self, topic: str, count: int, start: datetime) -> SkillResult:
        try:
            import feedparser
        except ImportError:
            return self._error_result(
                "No NEWS_API_KEY set and feedparser not installed. Set NEWS_API_KEY or install feedparser.",
                start,
            )

        url = f"https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
                feed = feedparser.parse(resp.text)
                entries = feed.entries[:count]
                lines = []
                for entry in entries:
                    title = entry.get("title", "No title")
                    link = entry.get("link", "")
                    published = entry.get("published", "")
                    lines.append(f"- **{title}**\n  {link}\n  {published}")
                if lines:
                    return self._success_result("\n\n".join(lines), start)
                return self._success_result("No news found.", start)
        except Exception as e:
            return self._error_result(f"Failed to fetch RSS: {e}", start)

    async def _search(self, params: dict, start: datetime) -> SkillResult:
        query = params.get("query", "")
        count = min(params.get("count", 5), 10)

        if not query:
            return self._error_result("No query provided", start)

        if self._api_key:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.get(
                        f"{self._base_url}/everything",
                        params={
                            "apiKey": self._api_key,
                            "q": query,
                            "pageSize": count,
                            "sortBy": "publishedAt",
                        },
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return self._format_articles(data.get("articles", []), start)
            except Exception as e:
                logger.debug("NewsAPI search failed", error=str(e))

        # Fallback to RSS
        return await self._rss_headlines(query, count, start)

    async def _summarize(self, url: str, start: datetime) -> SkillResult:
        if not url:
            return self._error_result("No URL provided", start)

        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return self._error_result(f"HTTP {resp.status_code} fetching article", start)

                # Simple HTML text extraction
                import re
                text = resp.text
                # Remove script/style tags
                text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', text)
                # Clean whitespace
                text = re.sub(r'\s+', ' ', text).strip()

                if len(text) > 3000:
                    text = text[:3000] + "..."

                return self._success_result(f"Article content from {url}:\n\n{text}", start)
        except Exception as e:
            return self._error_result(f"Failed to fetch article: {e}", start)

    def _format_articles(self, articles: list[dict], start: datetime) -> SkillResult:
        if not articles:
            return self._success_result("No articles found.", start)

        lines = []
        for article in articles:
            title = article.get("title", "No title")
            source = article.get("source", {}).get("name", "Unknown")
            desc = article.get("description", "")
            url = article.get("url", "")
            published = article.get("publishedAt", "")[:10]

            line = f"- **{title}** ({source}, {published})"
            if desc:
                line += f"\n  {desc[:150]}"
            if url:
                line += f"\n  {url}"
            lines.append(line)

        return self._success_result("\n\n".join(lines), start)

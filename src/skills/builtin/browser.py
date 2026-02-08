"""Web browser automation skill using Playwright."""

import asyncio
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

from ..base import BaseSkill, SkillResult


class BrowserSkill(BaseSkill):
    """
    Skill for web browsing and automation.

    Features:
    - Navigate to URLs
    - Extract page content
    - Take screenshots
    - Fill forms
    - Click elements
    """

    name = "browser"
    description = "Web browsing and automation"
    version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.headless = config.get("headless", True)
        self.timeout = config.get("timeout", 30) * 1000  # Convert to ms
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    def _register_capabilities(self) -> None:
        """Register browser capabilities."""
        self.register_capability(
            name="navigate",
            description="Navigate to a URL",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to navigate to"},
                    "wait_for": {"type": "string", "enum": ["load", "domcontentloaded", "networkidle"]},
                },
                "required": ["url"],
            },
            security_action="web_requests",
        )

        self.register_capability(
            name="get_content",
            description="Get the page content as text",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector (optional)"},
                    "format": {"type": "string", "enum": ["text", "html", "markdown"], "default": "text"},
                },
            },
            security_action="web_requests",
        )

        self.register_capability(
            name="screenshot",
            description="Take a screenshot of the page",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to save screenshot"},
                    "full_page": {"type": "boolean", "default": False},
                    "selector": {"type": "string", "description": "Element selector to screenshot"},
                },
            },
            security_action="web_requests",
        )

        self.register_capability(
            name="click",
            description="Click an element on the page",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of element"},
                },
                "required": ["selector"],
            },
            security_action="web_requests",
        )

        self.register_capability(
            name="fill_form",
            description="Fill form fields",
            parameters={
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "object",
                        "description": "Map of selector to value",
                        "additionalProperties": {"type": "string"},
                    },
                    "submit_selector": {"type": "string", "description": "Submit button selector"},
                },
                "required": ["fields"],
            },
            security_action="web_requests",
        )

        self.register_capability(
            name="extract_links",
            description="Extract all links from the page",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "Container selector"},
                },
            },
            security_action="web_requests",
        )

        self.register_capability(
            name="search_web",
            description="Perform a web search",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "engine": {"type": "string", "enum": ["google", "duckduckgo", "bing"], "default": "duckduckgo"},
                },
                "required": ["query"],
            },
            security_action="web_requests",
        )

    async def initialize(self) -> None:
        """Initialize Playwright browser."""
        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            self._page = await self._context.new_page()
            self._page.set_default_timeout(self.timeout)
            self._initialized = True
        except ImportError:
            # Playwright not installed
            self._initialized = False

    async def shutdown(self) -> None:
        """Shutdown browser."""
        if self._browser:
            await self._browser.close()
        if hasattr(self, "_playwright") and self._playwright:
            await self._playwright.stop()
        self._initialized = False

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """Execute a browser capability."""
        if not self._initialized:
            return SkillResult(
                success=False,
                error="Browser not initialized. Install playwright: pip install playwright && playwright install chromium",
            )

        start_time = datetime.now(timezone.utc)

        handlers = {
            "navigate": self._navigate,
            "get_content": self._get_content,
            "screenshot": self._screenshot,
            "click": self._click,
            "fill_form": self._fill_form,
            "extract_links": self._extract_links,
            "search_web": self._search_web,
        }

        handler = handlers.get(capability)
        if not handler:
            return self._error_result(f"Unknown capability: {capability}", start_time)

        try:
            result = await handler(**kwargs)
            return self._success_result(result, start_time)
        except Exception as e:
            return self._error_result(str(e), start_time)

    async def _navigate(
        self,
        url: str,
        wait_for: str = "load",
    ) -> dict[str, Any]:
        """Navigate to a URL."""
        response = await self._page.goto(url, wait_until=wait_for)

        return {
            "url": self._page.url,
            "title": await self._page.title(),
            "status": response.status if response else None,
        }

    async def _get_content(
        self,
        selector: str | None = None,
        format: str = "text",
    ) -> dict[str, Any]:
        """Get page content."""
        if selector:
            element = await self._page.query_selector(selector)
            if not element:
                raise ValueError(f"Element not found: {selector}")

            if format == "html":
                content = await element.inner_html()
            else:
                content = await element.inner_text()
        else:
            if format == "html":
                content = await self._page.content()
            else:
                content = await self._page.inner_text("body")

        # Convert to markdown if requested
        if format == "markdown":
            content = self._html_to_markdown(content)

        return {
            "url": self._page.url,
            "title": await self._page.title(),
            "content": content[:50000],  # Limit content size
            "format": format,
        }

    def _html_to_markdown(self, html: str) -> str:
        """Simple HTML to markdown conversion."""
        import re

        # Remove script and style tags
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)

        # Convert basic tags
        html = re.sub(r"<h1[^>]*>(.*?)</h1>", r"# \1\n", html)
        html = re.sub(r"<h2[^>]*>(.*?)</h2>", r"## \1\n", html)
        html = re.sub(r"<h3[^>]*>(.*?)</h3>", r"### \1\n", html)
        html = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n\n", html)
        html = re.sub(r"<br\s*/?>", "\n", html)
        html = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", html)
        html = re.sub(r"<a[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", r"[\2](\1)", html)
        html = re.sub(r"<strong[^>]*>(.*?)</strong>", r"**\1**", html)
        html = re.sub(r"<b[^>]*>(.*?)</b>", r"**\1**", html)
        html = re.sub(r"<em[^>]*>(.*?)</em>", r"*\1*", html)
        html = re.sub(r"<i[^>]*>(.*?)</i>", r"*\1*", html)

        # Remove remaining tags
        html = re.sub(r"<[^>]+>", "", html)

        # Clean up whitespace
        html = re.sub(r"\n\s*\n", "\n\n", html)
        html = html.strip()

        return html

    async def _screenshot(
        self,
        path: str | None = None,
        full_page: bool = False,
        selector: str | None = None,
    ) -> dict[str, Any]:
        """Take a screenshot."""
        screenshot_options: dict[str, Any] = {
            "full_page": full_page,
        }

        if path:
            screenshot_options["path"] = path

        if selector:
            element = await self._page.query_selector(selector)
            if not element:
                raise ValueError(f"Element not found: {selector}")
            screenshot_data = await element.screenshot(**screenshot_options)
        else:
            screenshot_data = await self._page.screenshot(**screenshot_options)

        result: dict[str, Any] = {
            "url": self._page.url,
            "full_page": full_page,
        }

        if path:
            result["path"] = path
        else:
            import base64

            result["data"] = base64.b64encode(screenshot_data).decode()

        return result

    async def _click(self, selector: str) -> dict[str, Any]:
        """Click an element."""
        await self._page.click(selector)

        return {
            "selector": selector,
            "clicked": True,
            "url": self._page.url,
        }

    async def _fill_form(
        self,
        fields: dict[str, str],
        submit_selector: str | None = None,
    ) -> dict[str, Any]:
        """Fill form fields."""
        for selector, value in fields.items():
            await self._page.fill(selector, value)

        if submit_selector:
            await self._page.click(submit_selector)
            await self._page.wait_for_load_state("networkidle")

        return {
            "fields_filled": list(fields.keys()),
            "submitted": submit_selector is not None,
            "url": self._page.url,
        }

    async def _extract_links(
        self,
        selector: str | None = None,
    ) -> list[dict[str, str]]:
        """Extract links from the page."""
        container = selector or "body"
        links = await self._page.query_selector_all(f"{container} a[href]")

        results = []
        base_url = self._page.url

        for link in links[:100]:  # Limit to 100 links
            href = await link.get_attribute("href")
            text = await link.inner_text()

            if href:
                # Make absolute URL
                absolute_url = urljoin(base_url, href)
                results.append({
                    "url": absolute_url,
                    "text": text.strip()[:200],
                })

        return results

    async def _search_web(
        self,
        query: str,
        engine: str = "duckduckgo",
    ) -> list[dict[str, Any]]:
        """Perform a web search."""
        search_urls = {
            "google": f"https://www.google.com/search?q={query}",
            "duckduckgo": f"https://duckduckgo.com/?q={query}",
            "bing": f"https://www.bing.com/search?q={query}",
        }

        url = search_urls.get(engine, search_urls["duckduckgo"])
        await self._page.goto(url, wait_until="networkidle")

        # Extract search results based on engine
        results = []

        if engine == "duckduckgo":
            items = await self._page.query_selector_all('[data-testid="result"]')
            for item in items[:10]:
                title_el = await item.query_selector("h2")
                link_el = await item.query_selector("a")
                snippet_el = await item.query_selector('[data-testid="result-snippet"]')

                if title_el and link_el:
                    results.append({
                        "title": await title_el.inner_text(),
                        "url": await link_el.get_attribute("href"),
                        "snippet": await snippet_el.inner_text() if snippet_el else "",
                    })

        elif engine == "google":
            items = await self._page.query_selector_all(".g")
            for item in items[:10]:
                title_el = await item.query_selector("h3")
                link_el = await item.query_selector("a")
                snippet_el = await item.query_selector(".VwiC3b")

                if title_el and link_el:
                    results.append({
                        "title": await title_el.inner_text(),
                        "url": await link_el.get_attribute("href"),
                        "snippet": await snippet_el.inner_text() if snippet_el else "",
                    })

        return results

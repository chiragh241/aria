"""Link understanding — auto-extract content from URLs in messages.

When a user sends a message containing URLs, this module fetches the page,
extracts readable text, and injects it as context so the AI can discuss the
content without the user having to paste it.
"""

import re
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Match http/https URLs — intentionally loose to catch most links
URL_PATTERN = re.compile(
    r'https?://[^\s<>\'")\]]+',
    re.IGNORECASE,
)

# Domains to skip (images, videos, short links that resolve to nothing useful)
SKIP_DOMAINS = {
    "youtube.com", "youtu.be",  # handled by media understanding
    "instagram.com", "tiktok.com",  # media-heavy, text extraction poor
    "spotify.com", "music.apple.com",  # media links
}


def extract_urls(text: str) -> list[str]:
    """Extract all URLs from a text string."""
    urls = URL_PATTERN.findall(text)
    # De-duplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        # Strip trailing punctuation that got caught
        url = url.rstrip(".,;:!?)")
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def should_extract(url: str) -> bool:
    """Check if a URL should be auto-extracted."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower().lstrip("www.")
        return domain not in SKIP_DOMAINS
    except Exception:
        return True


async def fetch_url_content(url: str, max_chars: int = 5000, timeout: int = 15) -> dict[str, Any] | None:
    """Fetch a URL and extract readable text content.

    Returns dict with 'title', 'text', 'url' or None on failure.
    """
    try:
        import httpx
        from html.parser import HTMLParser

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Aria/1.0)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                # Not HTML — return basic info
                return {
                    "url": url,
                    "title": url,
                    "text": f"[Non-HTML content: {content_type}]",
                }

            html = resp.text

            # Simple HTML text extraction
            title, text = _extract_html_text(html)

            if text and len(text.strip()) > 50:
                return {
                    "url": url,
                    "title": title or url,
                    "text": text[:max_chars],
                }
            return None

    except Exception as e:
        logger.debug("Failed to fetch URL", url=url, error=str(e))
        return None


def _extract_html_text(html: str) -> tuple[str, str]:
    """Extract title and readable text from HTML."""
    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.title = ""
            self.text_parts: list[str] = []
            self._in_title = False
            self._skip_tags = {"script", "style", "nav", "footer", "header", "aside", "noscript"}
            self._skip_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag == "title":
                self._in_title = True
            if tag in self._skip_tags:
                self._skip_depth += 1
            if tag in ("p", "div", "h1", "h2", "h3", "h4", "li", "br", "tr"):
                self.text_parts.append("\n")

        def handle_endtag(self, tag):
            if tag == "title":
                self._in_title = False
            if tag in self._skip_tags and self._skip_depth > 0:
                self._skip_depth -= 1

        def handle_data(self, data):
            if self._in_title:
                self.title += data
            elif self._skip_depth == 0:
                stripped = data.strip()
                if stripped:
                    self.text_parts.append(stripped)

    extractor = TextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        pass

    text = " ".join(extractor.text_parts)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return extractor.title.strip(), text.strip()


async def process_message_links(content: str, max_links: int = 3) -> str | None:
    """Extract URLs from a message and fetch their content.

    Returns a formatted context string to append to the user message, or None.
    """
    urls = extract_urls(content)
    if not urls:
        return None

    extractable = [u for u in urls if should_extract(u)][:max_links]
    if not extractable:
        return None

    results: list[dict[str, Any]] = []
    for url in extractable:
        result = await fetch_url_content(url)
        if result:
            results.append(result)

    if not results:
        return None

    # Format as context
    parts = ["\n\n---\nLink content (auto-extracted):"]
    for r in results:
        parts.append(f"\n**{r['title']}** ({r['url']})")
        parts.append(r['text'][:3000])
    parts.append("---")

    return "\n".join(parts)

"""crawl4ai adapter with graceful fallback compatibility.

The goal is to keep the rest of the codebase independent from crawl4ai's
internal response classes and version-specific field names.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup


@dataclass
class CrawlDocument:
    url: str
    title: str
    content: str
    markdown: str
    raw_html: str = ""
    description: str = ""
    author: str = ""
    published_at: str = ""
    tags: list[str] | None = None
    source_metadata: dict[str, Any] | None = None
    fetched_at: str = ""
    engine: str = "crawl4ai"
    ok: bool = True
    error: str = ""


def is_crawl4ai_available() -> bool:
    try:
        import crawl4ai  # noqa: F401

        return True
    except Exception:
        return False


def _to_text_from_html(value: str) -> str:
    if not value:
        return ""
    soup = BeautifulSoup(value, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n\n".join(lines)


def _markdown_to_string(markdown_obj: Any) -> str:
    if markdown_obj is None:
        return ""
    if isinstance(markdown_obj, str):
        return markdown_obj.strip()

    for field_name in ("raw_markdown", "markdown", "fit_markdown"):
        value = getattr(markdown_obj, field_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(markdown_obj).strip()


def _extract_result_fields(result: Any, url: str) -> CrawlDocument:
    title = str(getattr(result, "title", "") or "").strip() or url

    markdown = _markdown_to_string(getattr(result, "markdown", None))
    if not markdown:
        markdown = str(getattr(result, "markdown_v2", "") or "").strip()

    cleaned_html = str(getattr(result, "cleaned_html", "") or "").strip()
    raw_html = str(getattr(result, "raw_html", "") or "").strip()
    if not raw_html:
        raw_html = str(getattr(result, "html", "") or "").strip()
    if not raw_html:
        raw_html = cleaned_html
    extracted_text = str(getattr(result, "extracted_content", "") or "").strip()

    content = markdown
    if not content and extracted_text:
        content = extracted_text
    if not content and cleaned_html:
        content = _to_text_from_html(cleaned_html)

    description = str(getattr(result, "description", "") or "").strip()
    author = str(getattr(result, "author", "") or "").strip()
    published_at = str(getattr(result, "published_at", "") or "").strip()
    ok = bool(getattr(result, "success", True))
    error = str(getattr(result, "error_message", "") or "").strip()

    return CrawlDocument(
        url=url,
        title=title,
        content=content.strip(),
        markdown=markdown.strip(),
        raw_html=raw_html,
        description=description,
        author=author,
        published_at=published_at,
        tags=[],
        source_metadata={},
        fetched_at=datetime.now(timezone.utc).isoformat(),
        engine="crawl4ai",
        ok=ok and bool(content.strip()),
        error=error,
    )


async def acrawl_urls_with_crawl4ai(urls: list[str]) -> list[CrawlDocument]:
    from crawl4ai import AsyncWebCrawler

    docs: list[CrawlDocument] = []

    async with AsyncWebCrawler() as crawler:
        for url in urls:
            try:
                result = await crawler.arun(url=url)
                docs.append(_extract_result_fields(result, url))
            except Exception as error:
                docs.append(
                    CrawlDocument(
                        url=url,
                        title=url,
                        content="",
                        markdown="",
                        tags=[],
                        source_metadata={},
                        fetched_at=datetime.now(timezone.utc).isoformat(),
                        ok=False,
                        error=str(error),
                    )
                )

    return docs


def crawl_urls_with_crawl4ai(urls: list[str]) -> list[CrawlDocument]:
    return asyncio.run(acrawl_urls_with_crawl4ai(urls))

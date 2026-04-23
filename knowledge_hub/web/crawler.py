"""
웹 크롤러

웹 콘텐츠를 추출하여 지식 허브에 통합합니다.
"""

import logging
import os
import tempfile
import time
from typing import Any, List, Optional
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from knowledge_hub.papers.source_text import extract_pdf_text_excerpt
from knowledge_hub.web.http_headers import default_request_headers

log = logging.getLogger("khub.web.crawler")


@dataclass
class WebDocument:
    url: str
    title: str
    content: str
    raw_html: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description or "",
            "author": self.author or "",
            "published_at": self.published_at or "",
            "tags": list(self.tags or []),
            "source_type": "web",
        }


class WebCrawler:
    """웹 콘텐츠 크롤러"""

    OPENAI_NEWS_FEED_URL = "https://openai.com/news/rss.xml"

    def __init__(self, timeout: int = 10, delay: float = 1.0):
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(default_request_headers())
        self._rss_cache: dict[str, list[dict[str, str]]] = {}

    def crawl_url(self, url: str) -> Optional[WebDocument]:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            pdf_document = self._extract_pdf_document(url, response)
            if pdf_document is not None:
                return pdf_document
            soup = BeautifulSoup(response.content, "html.parser")

            title = self._extract_title(soup)
            content = self._extract_content(soup)
            if not content:
                return None

            description = self._extract_meta(soup, "description")
            author = self._extract_meta(soup, "author")

            return WebDocument(
                url=url, title=title, content=content,
                raw_html=response.text,
                description=description, author=author,
                published_at=None,
                tags=[],
            )
        except Exception:
            fallback = self._fallback_document(url)
            if fallback is not None:
                return fallback
            return None

    def crawl_urls(self, urls: List[str]) -> List[WebDocument]:
        docs = []
        for i, url in enumerate(urls):
            doc = self.crawl_url(url)
            if doc:
                docs.append(doc)
            if i < len(urls) - 1:
                time.sleep(self.delay)
        return docs

    def _extract_title(self, soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text().strip()
        return "Untitled"

    def _extract_content(self, soup: BeautifulSoup) -> str:
        for el in soup(["script", "style", "nav", "footer", "header", "aside"]):
            el.decompose()

        for tag in ["article", "main", "body"]:
            element = soup.find(tag)
            if element:
                text = element.get_text(separator="\n", strip=True)
                return self._clean(text)
        return ""

    def _extract_meta(self, soup: BeautifulSoup, name: str) -> Optional[str]:
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            return meta["content"].strip()
        meta = soup.find("meta", property=name)
        if meta and meta.get("content"):
            return meta["content"].strip()
        return None

    @staticmethod
    def _clean(text: str) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return "\n\n".join(lines)

    def _extract_pdf_document(self, url: str, response: requests.Response) -> Optional[WebDocument]:
        content_type = str(response.headers.get("Content-Type") or "").lower()
        path = (urlsplit(url).path or "").lower()
        if "application/pdf" not in content_type and not path.endswith(".pdf"):
            return None
        content = self._extract_pdf_text(response.content)
        if not content:
            return None
        return WebDocument(
            url=url,
            title=self._pdf_title(url),
            content=content,
            raw_html="",
            description=None,
            author=None,
            published_at=None,
            tags=[],
            source_metadata={
                "media_type": "pdf",
                "content_type": content_type or "application/pdf",
            },
        )

    def _extract_pdf_text(self, payload: bytes) -> str:
        if not payload:
            return ""
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
                handle.write(payload)
                temp_path = handle.name
            return extract_pdf_text_excerpt(temp_path, max_pages=12, max_chars=30_000)
        except Exception as error:
            log.warning("pdf extraction failed: %s", error)
            return ""
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    @staticmethod
    def _pdf_title(url: str) -> str:
        path = urlsplit(url).path or ""
        stem = unquote(path.rsplit("/", 1)[-1]).strip()
        if stem.lower().endswith(".pdf"):
            stem = stem[:-4]
        return stem or "Untitled PDF"

    def _fallback_document(self, url: str) -> Optional[WebDocument]:
        parsed = urlsplit(url)
        if parsed.netloc == "openai.com" and parsed.path.startswith(("/index/", "/news/")):
            return self._openai_news_rss_document(url)
        return None

    def _openai_news_rss_document(self, url: str) -> Optional[WebDocument]:
        normalized = url.rstrip("/")
        entries = self._rss_cache.get(self.OPENAI_NEWS_FEED_URL)
        if entries is None:
            entries = self._fetch_openai_news_rss()
            self._rss_cache[self.OPENAI_NEWS_FEED_URL] = entries
        for entry in entries:
            if entry.get("link", "").rstrip("/") != normalized:
                continue
            title = entry.get("title", "").strip() or "OpenAI News"
            description = entry.get("description", "").strip()
            category = entry.get("category", "").strip()
            pub_date = entry.get("pub_date", "").strip()
            sections = [title]
            if description:
                sections.append(description)
            else:
                summary_parts = [part for part in [category, pub_date] if part]
                if summary_parts:
                    sections.append(" | ".join(summary_parts))
            if category:
                sections.append(f"Category: {category}")
            if pub_date:
                sections.append(f"Published: {pub_date}")
            content = self._clean("\n\n".join(part for part in sections if part))
            if not content:
                return None
            return WebDocument(
                url=url,
                title=title,
                content=content,
                raw_html=entry.get("raw_xml", ""),
                description=description or None,
                author="OpenAI",
                published_at=pub_date or None,
                tags=[category] if category else [],
                source_metadata={
                    "source_name": "OpenAI News",
                    "source_vendor": "openai",
                    "source_channel": "openai_news",
                    "source_channel_type": "official_news_rss",
                },
            )
        return None

    def _fetch_openai_news_rss(self) -> list[dict[str, str]]:
        try:
            response = self.session.get(self.OPENAI_NEWS_FEED_URL, timeout=self.timeout)
            response.raise_for_status()
            root = ET.fromstring(response.text)
        except Exception as error:
            log.warning("openai news rss fallback unavailable: %s", error)
            return []

        items: list[dict[str, str]] = []
        for node in root.findall("./channel/item"):
            title = (node.findtext("title") or "").strip()
            link = (node.findtext("link") or "").strip()
            description = (node.findtext("description") or "").strip()
            category = (node.findtext("category") or "").strip()
            pub_date = (node.findtext("pubDate") or "").strip()
            if not link:
                continue
            items.append(
                {
                    "title": title,
                    "link": link,
                    "description": description,
                    "category": category,
                    "pub_date": pub_date,
                    "raw_xml": ET.tostring(node, encoding="unicode"),
                }
            )
        return items

"""
웹 크롤러

웹 콘텐츠를 추출하여 지식 허브에 통합합니다.
"""

import time
from typing import List, Optional
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


@dataclass
class WebDocument:
    url: str
    title: str
    content: str
    description: Optional[str] = None
    author: Optional[str] = None

    def to_metadata(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description or "",
            "author": self.author or "",
            "source_type": "web",
        }


class WebCrawler:
    """웹 콘텐츠 크롤러"""

    def __init__(self, timeout: int = 10, delay: float = 1.0):
        self.timeout = timeout
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Knowledge Hub Bot)"})

    def crawl_url(self, url: str) -> Optional[WebDocument]:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            title = self._extract_title(soup)
            content = self._extract_content(soup)
            if not content:
                return None

            description = self._extract_meta(soup, "description")
            author = self._extract_meta(soup, "author")

            return WebDocument(
                url=url, title=title, content=content,
                description=description, author=author,
            )
        except Exception:
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

from __future__ import annotations

import email.utils
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from knowledge_hub.curation.source_registry import (
    DiscoveredSourceItem,
    DiscoveryProvenance,
    SourceIdentity,
)
from knowledge_hub.web.http_headers import default_request_headers

REQUEST_HEADERS = default_request_headers()
RSS_TYPES = {
    "application/rss+xml",
    "application/atom+xml",
    "application/xml",
    "text/xml",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_text(url: str, timeout: int = 20) -> tuple[str, str | None]:
    response = requests.get(url, timeout=timeout, headers=REQUEST_HEADERS)
    response.raise_for_status()
    return response.text, response.headers.get("content-type")


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _normalize_url(value: str, *, default_scheme: str = "https") -> str:
    token = _clean_text(value)
    if not token:
        return ""
    parsed = urlparse(token)
    scheme = parsed.scheme or default_scheme
    path = re.sub(r"/+", "/", parsed.path or "/")
    return urlunparse((scheme, parsed.netloc.lower(), path.rstrip("/") or "/", "", parsed.query, ""))


def _strip_tag(value: str) -> str:
    if "}" in value:
        return value.rsplit("}", 1)[-1]
    return value


def _discover_feed_url(base_url: str, html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("link", href=True):
        rel = {token.lower() for token in link.get("rel", [])}
        content_type = str(link.get("type") or "").strip().lower()
        if "alternate" in rel and content_type in RSS_TYPES:
            return urljoin(base_url, str(link["href"]).strip())
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        text = _clean_text(anchor.get_text(" ", strip=True)).lower()
        href_lower = href.lower()
        if (
            "rss" in href_lower
            or href_lower.endswith("feed.xml")
            or href_lower.endswith("/feed/")
            or href_lower.endswith("/feed")
            or "rss" in text
        ):
            absolute = urljoin(base_url, href)
            if urlparse(absolute).netloc == urlparse(base_url).netloc:
                return absolute
    return ""


def _parse_datetime(value: str) -> str:
    token = _clean_text(value)
    if not token:
        return ""
    token = re.sub(r"^(published|updated)\s+", "", token, flags=re.IGNORECASE)
    try:
        if token.endswith("Z"):
            return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        return datetime.fromisoformat(token).astimezone(timezone.utc).isoformat()
    except Exception:
        try:
            return email.utils.parsedate_to_datetime(token).astimezone(timezone.utc).isoformat()
        except Exception:
            for fmt in ("%b %d, %Y", "%B %d, %Y"):
                try:
                    return datetime.strptime(token, fmt).replace(tzinfo=timezone.utc).isoformat()
                except Exception:
                    continue
            return ""


def _slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = (parsed.path or "").strip("/")
    if not path:
        return parsed.netloc.lower()
    return path


def _arxiv_paper_id(url: str) -> str:
    match = re.search(r"/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", url)
    if match:
        return match.group(1)
    return ""


def _lightweight_hash(parts: list[str]) -> str:
    raw = "|".join(_clean_text(part).lower() for part in parts if _clean_text(part))
    if not raw:
        return ""
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ParsedFeedEntry:
    url: str
    title: str
    published_at: str = ""
    author: str = ""
    tags: list[str] | None = None
    entry_ref: str = ""
    raw: dict[str, Any] | None = None


def _split_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    token = _clean_text(str(value or ""))
    if not token:
        return []
    parts = re.split(r"[;,|]", token)
    return [_clean_text(part) for part in parts if _clean_text(part)]


def _extract_json_ld_nodes(value: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if isinstance(value, dict):
        nodes.append(value)
        for item in value.values():
            nodes.extend(_extract_json_ld_nodes(item))
    elif isinstance(value, list):
        for item in value:
            nodes.extend(_extract_json_ld_nodes(item))
    return nodes


def _pick_json_ld_metadata(soup: BeautifulSoup) -> dict[str, Any]:
    published_at = ""
    author = ""
    tags: list[str] = []
    title = ""

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(" ", strip=True)
        token = raw.strip()
        if not token:
            continue
        try:
            payload = json.loads(token)
        except Exception:
            continue
        for node in _extract_json_ld_nodes(payload):
            if not isinstance(node, dict):
                continue
            if not title:
                title = _clean_text(node.get("headline") or node.get("name") or "")
            if not published_at:
                published_at = _parse_datetime(
                    node.get("datePublished") or node.get("dateCreated") or node.get("dateModified") or ""
                )
            if not author:
                author_value = node.get("author")
                if isinstance(author_value, dict):
                    author = _clean_text(author_value.get("name") or "")
                elif isinstance(author_value, list):
                    authors = []
                    for item in author_value:
                        if isinstance(item, dict):
                            name = _clean_text(item.get("name") or "")
                            if name:
                                authors.append(name)
                        else:
                            name = _clean_text(item)
                            if name:
                                authors.append(name)
                    author = ", ".join(authors[:3])
                else:
                    author = _clean_text(author_value or "")
            if not tags:
                tags = _split_tags(node.get("keywords") or node.get("articleSection") or "")
            if title and published_at and (author or tags):
                break
        if title and published_at and (author or tags):
            break

    return {
        "title": title,
        "published_at": published_at,
        "author": author,
        "tags": tags,
    }


def _extract_page_metadata(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    def _meta(*keys: str, prop: bool = False) -> str:
        attr = "property" if prop else "name"
        for key in keys:
            node = soup.find("meta", attrs={attr: key})
            if node and node.get("content"):
                return _clean_text(node.get("content") or "")
        return ""

    published_at = _parse_datetime(
        _meta(
            "article:published_time",
            "og:published_time",
            prop=True,
        )
        or _meta("publish-date", "pubdate", "date", "datePublished", "dc.date", "article.published")
    )
    author = _meta("author", "article:author") or _meta("article:author", prop=True)
    tags = _split_tags(
        _meta("keywords", "news_keywords", "article:tag") or _meta("article:tag", prop=True)
    )
    title = _meta("og:title", prop=True) or _clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")

    json_ld_meta = _pick_json_ld_metadata(soup)
    if not title:
        title = json_ld_meta.get("title") or ""
    if not published_at:
        published_at = json_ld_meta.get("published_at") or ""
    if not author:
        author = json_ld_meta.get("author") or ""
    if not tags:
        tags = list(json_ld_meta.get("tags") or [])

    return {
        "title": title,
        "published_at": published_at,
        "author": author,
        "tags": tags,
    }


def _extract_openai_developer_metadata(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h1")
    header = heading.find_parent("header") if heading is not None else soup.find("header")
    title = ""
    published_at = ""
    author = ""
    tags: list[str] = []

    if header is not None:
        if heading is not None:
            title = _clean_text(heading.get_text(" ", strip=True))
        text = header.get_text(" ", strip=True)
        match = re.search(r"\b([A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4})\b", text)
        if match:
            published_at = _parse_datetime(match.group(1))
        author_block = None
        for paragraph in header.find_all("p"):
            author_text = _clean_text(paragraph.get_text(" ", strip=True))
            if re.search(r"^Authors?:", author_text, flags=re.IGNORECASE):
                author_block = paragraph
                author = re.sub(r"^Authors?:\s*", "", author_text, flags=re.IGNORECASE).strip()
                break
        if not author:
            author_match = re.search(r"Authors?:\s*(.+)", text, flags=re.IGNORECASE)
            if author_match:
                author = _clean_text(author_match.group(1))
        category_spans = header.find_all("span")
        for span in category_spans:
            token = _clean_text(span.get_text(" ", strip=True))
            if not token or re.fullmatch(r"[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4}", token):
                continue
            if token.lower().startswith("authors"):
                continue
            if token == title:
                continue
            if token not in tags:
                tags.append(token)
        if tags:
            tags = tags[:3]

    return {
        "title": title,
        "published_at": published_at,
        "author": author,
        "tags": tags,
    }


def _extract_huggingface_metadata(html_text: str) -> dict[str, Any]:
    published_at = ""
    author = ""
    title = ""
    tags: list[str] = []
    soup = BeautifulSoup(html_text, "html.parser")
    heading = soup.find("h1")
    if heading is not None:
        title = _clean_text(heading.get_text(" ", strip=True))

    published_match = re.search(r"Published\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", html_text)
    if published_match:
        published_at = _parse_datetime(published_match.group(1))

    author_match = re.search(r'data-target="BlogAuthorsByline"\s+data-props="([^"]+)"', html_text)
    if author_match:
        try:
            props = json.loads(html.unescape(author_match.group(1)))
            authors: list[str] = []
            for item in props.get("authors", [])[:5]:
                candidate = item.get("author", {}) if isinstance(item, dict) else {}
                if not isinstance(candidate, dict):
                    continue
                name = _clean_text(candidate.get("fullname") or candidate.get("name") or "")
                if name and name not in authors:
                    authors.append(name)
            author = ", ".join(authors[:3])
        except Exception:
            author = ""

    return {
        "title": title,
        "published_at": published_at,
        "author": author,
        "tags": tags,
    }


def _extract_anthropic_metadata(html_text: str) -> dict[str, Any]:
    soup = BeautifulSoup(html_text, "html.parser")
    title = ""
    published_at = ""
    author = ""
    tags: list[str] = []

    heading = soup.find("h1")
    if heading is not None:
        title = _clean_text(heading.get_text(" ", strip=True))

    published_node = soup.find("div", class_=re.compile(r"\bbody-3\b.*\bagate\b"))
    if published_node is not None:
        published_at = _parse_datetime(published_node.get_text(" ", strip=True))
    if not published_at:
        match = re.search(r'\\"publishedOn\\":\\"([^"]+)\\"', html_text)
        if match:
            published_at = _parse_datetime(match.group(1))

    subject_container = soup.find("div", class_=re.compile(r"subjects"))
    if subject_container is not None:
        tags = [_clean_text(node.get_text(" ", strip=True)) for node in subject_container.find_all("span") if _clean_text(node.get_text(" ", strip=True))]

    return {
        "title": title,
        "published_at": published_at,
        "author": author,
        "tags": tags,
    }


class BaseSourceExtractor:
    source_name = ""
    source_vendor = ""
    source_channel = ""
    source_channel_type = ""
    source_type = "official_blog"
    feed_url: str = ""
    html_only: bool = False

    def matches(self, source: dict[str, Any]) -> bool:
        return str(source.get("source_name") or "").strip() == self.source_name

    def metadata_precedence(self) -> list[str]:
        return [
            "source_feed_metadata",
            "page_embedded_metadata",
            "extractor_html_parse",
            "generic_crawler_inference",
        ]

    def _extract_source_specific_page_metadata(self, html: str, url: str) -> dict[str, Any]:
        return {}

    def _supplement_from_page(self, entry: ParsedFeedEntry) -> tuple[ParsedFeedEntry, dict[str, str]]:
        resolution: dict[str, str] = {}
        try:
            html, _ = _fetch_text(entry.url)
        except Exception:
            return entry, resolution

        page_meta = _extract_page_metadata(html)
        source_meta = self._extract_source_specific_page_metadata(html, entry.url)
        if source_meta:
            for key in ("title", "published_at", "author", "tags"):
                if source_meta.get(key):
                    page_meta[key] = source_meta[key]
        raw = dict(entry.raw or {})
        raw["page_metadata"] = page_meta
        entry.raw = raw

        if not entry.title and page_meta.get("title"):
            entry.title = str(page_meta["title"])
            resolution["title"] = "extractor_html_parse" if source_meta.get("title") else "page_embedded_metadata"
        if not entry.published_at and page_meta.get("published_at"):
            entry.published_at = str(page_meta["published_at"])
            resolution["published_at"] = (
                "extractor_html_parse" if source_meta.get("published_at") else "page_embedded_metadata"
            )
        if not entry.author and page_meta.get("author"):
            entry.author = str(page_meta["author"])
            resolution["author"] = "extractor_html_parse" if source_meta.get("author") else "page_embedded_metadata"
        if not entry.tags and page_meta.get("tags"):
            entry.tags = list(page_meta["tags"])
            resolution["tags"] = "extractor_html_parse" if source_meta.get("tags") else "page_embedded_metadata"
        return entry, resolution

    def resolve_identity(self, item: DiscoveredSourceItem) -> SourceIdentity:
        stable_id = str(item.identity.source_item_id or "").strip()
        if not stable_id:
            stable_id = self._default_source_item_id(item)
        return SourceIdentity(
            source_vendor=self.source_vendor,
            source_channel=self.source_channel,
            source_channel_type=self.source_channel_type,
            source_item_id=stable_id,
        )

    def discover_latest(self, source: dict[str, Any], limit: int) -> list[DiscoveredSourceItem]:
        base_url = str(source.get("url") or "").strip()
        if not base_url:
            return []
        items: list[DiscoveredSourceItem] = []
        feed_items: list[ParsedFeedEntry] = []
        html = ""
        html_fetched = False

        feed_url = self.feed_url or ""
        if not feed_url and not self.html_only:
            try:
                html, _ = _fetch_text(base_url)
                html_fetched = True
                feed_url = _discover_feed_url(base_url, html)
            except Exception:
                html = ""
        if feed_url:
            try:
                feed_items = self._parse_feed(feed_url, limit)
            except Exception:
                feed_items = []
        if feed_items:
            for rank, entry in enumerate(feed_items[: max(1, int(limit))], start=1):
                items.append(self._item_from_feed_entry(source, entry, rank, origin_url=feed_url))
            return items

        if not html_fetched:
            html, _ = _fetch_text(base_url)
        html_items = self._parse_html_links(base_url, html, limit)
        for rank, entry in enumerate(html_items[: max(1, int(limit))], start=1):
            items.append(self._item_from_html_entry(source, entry, rank, origin_url=base_url))
        return items

    def _default_source_item_id(self, item: DiscoveredSourceItem) -> str:
        if item.url:
            return _slug_from_url(item.url)
        return _lightweight_hash([item.title_hint, item.published_at, item.source_name])

    def _item_from_feed_entry(
        self,
        source: dict[str, Any],
        entry: ParsedFeedEntry,
        rank: int,
        *,
        origin_url: str,
    ) -> DiscoveredSourceItem:
        page_resolution: dict[str, str] = {}
        if not (entry.published_at and entry.author and entry.tags):
            entry, page_resolution = self._supplement_from_page(entry)
        resolution = {
            "title": page_resolution.get("title") or ("source_feed_metadata" if entry.title else ""),
            "published_at": page_resolution.get("published_at") or ("source_feed_metadata" if entry.published_at else ""),
            "author": page_resolution.get("author") or ("source_feed_metadata" if entry.author else ""),
            "tags": page_resolution.get("tags") or ("source_feed_metadata" if entry.tags else ""),
            "canonical_url": "source_feed_metadata" if entry.url else "normalized_url",
        }
        item = DiscoveredSourceItem(
            identity=SourceIdentity(
                source_vendor=self.source_vendor,
                source_channel=self.source_channel,
                source_channel_type=self.source_channel_type,
                source_item_id=self._source_item_id_from_feed(entry),
            ),
            source_name=self.source_name,
            source_type=str(source.get("type") or self.source_type),
            url=entry.url,
            canonical_url=_normalize_url(entry.url),
            title_hint=entry.title,
            published_at=entry.published_at,
            author=entry.author,
            tags=list(entry.tags or []),
            provenance=DiscoveryProvenance(
                method="rss",
                origin_url=origin_url,
                entry_ref=entry.entry_ref or entry.url,
                discovered_at=_now_iso(),
                rank=rank,
            ),
            metadata={
                "metadata_precedence": self.metadata_precedence(),
                "resolution": resolution,
                "entry_raw": dict(entry.raw or {}),
            },
        )
        return item

    def _item_from_html_entry(
        self,
        source: dict[str, Any],
        entry: ParsedFeedEntry,
        rank: int,
        *,
        origin_url: str,
    ) -> DiscoveredSourceItem:
        entry, page_resolution = self._supplement_from_page(entry)
        resolution = {
            "title": page_resolution.get("title") or ("extractor_html_parse" if entry.title else ""),
            "published_at": page_resolution.get("published_at") or ("extractor_html_parse" if entry.published_at else ""),
            "author": page_resolution.get("author") or "",
            "tags": page_resolution.get("tags") or "",
            "canonical_url": "normalized_url",
        }
        return DiscoveredSourceItem(
            identity=SourceIdentity(
                source_vendor=self.source_vendor,
                source_channel=self.source_channel,
                source_channel_type=self.source_channel_type,
                source_item_id=self._source_item_id_from_html(entry),
            ),
            source_name=self.source_name,
            source_type=str(source.get("type") or self.source_type),
            url=entry.url,
            canonical_url=_normalize_url(entry.url),
            title_hint=entry.title,
            published_at=entry.published_at,
            author=entry.author,
            tags=list(entry.tags or []),
            provenance=DiscoveryProvenance(
                method="html_index",
                origin_url=origin_url,
                entry_ref=entry.entry_ref or entry.url,
                discovered_at=_now_iso(),
                rank=rank,
            ),
            metadata={
                "metadata_precedence": self.metadata_precedence(),
                "resolution": resolution,
            },
        )

    def _source_item_id_from_feed(self, entry: ParsedFeedEntry) -> str:
        if entry.entry_ref:
            return entry.entry_ref
        return self._default_source_item_id(
            DiscoveredSourceItem(
                identity=SourceIdentity(self.source_vendor, self.source_channel, self.source_channel_type, ""),
                source_name=self.source_name,
                source_type=self.source_type,
                url=entry.url,
                title_hint=entry.title,
                published_at=entry.published_at,
            )
        )

    def _source_item_id_from_html(self, entry: ParsedFeedEntry) -> str:
        return entry.entry_ref or _slug_from_url(entry.url)

    def _parse_feed(self, feed_url: str, limit: int) -> list[ParsedFeedEntry]:
        xml_text, _ = _fetch_text(feed_url)
        root = ET.fromstring(xml_text)
        feed_parsed = urlparse(feed_url)
        entries: list[ParsedFeedEntry] = []

        def _normalize_link(value: str) -> str:
            parsed = urlparse(_clean_text(value))
            if parsed.netloc == feed_parsed.netloc and parsed.scheme and parsed.scheme != feed_parsed.scheme:
                return urlunparse((feed_parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))
            return _normalize_url(_clean_text(value), default_scheme=feed_parsed.scheme or "https")

        if _strip_tag(root.tag) == "rss":
            for item in root.findall(".//item"):
                link = _normalize_link(item.findtext("link") or "")
                title = _clean_text(item.findtext("title"))
                published_at = _parse_datetime(item.findtext("pubDate") or item.findtext("published") or "")
                author = _clean_text(item.findtext("author") or item.findtext("{*}author") or "")
                tags = [_clean_text(node.text or "") for node in item.findall("category") if _clean_text(node.text or "")]
                guid = _clean_text(item.findtext("guid") or "")
                if link:
                    entries.append(
                        ParsedFeedEntry(
                            url=link,
                            title=title,
                            published_at=published_at,
                            author=author,
                            tags=tags,
                            entry_ref=guid or self._feed_entry_ref(link),
                            raw={"guid": guid},
                        )
                    )
                if len(entries) >= limit:
                    break
            return entries

        if _strip_tag(root.tag) == "feed":
            for entry in root.findall(".//{*}entry"):
                title = _clean_text(entry.findtext("{*}title"))
                link = ""
                for link_node in entry.findall("{*}link"):
                    href = _normalize_link(link_node.attrib.get("href", ""))
                    rel = _clean_text(link_node.attrib.get("rel", "alternate"))
                    if href and rel in {"alternate", ""}:
                        link = href
                        break
                if not link:
                    link = _normalize_link(entry.findtext("{*}id") or "")
                author = ""
                author_node = entry.find("{*}author/{*}name")
                if author_node is not None:
                    author = _clean_text(author_node.text or "")
                tags = [_clean_text(node.attrib.get("term", "")) for node in entry.findall("{*}category") if _clean_text(node.attrib.get("term", ""))]
                published_at = _parse_datetime(entry.findtext("{*}published") or entry.findtext("{*}updated") or "")
                entry_id = _clean_text(entry.findtext("{*}id") or "")
                if link:
                    entries.append(
                        ParsedFeedEntry(
                            url=link,
                            title=title,
                            published_at=published_at,
                            author=author,
                            tags=tags,
                            entry_ref=entry_id or self._feed_entry_ref(link),
                            raw={"id": entry_id},
                        )
                    )
                if len(entries) >= limit:
                    break
        return entries

    def _feed_entry_ref(self, link: str) -> str:
        return _slug_from_url(link)

    def _parse_html_links(self, base_url: str, html: str, limit: int) -> list[ParsedFeedEntry]:
        soup = BeautifulSoup(html, "html.parser")
        base_parsed = urlparse(base_url)
        items: list[ParsedFeedEntry] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = urljoin(base_url, str(anchor.get("href") or "").strip()).split("#")[0]
            parsed = urlparse(href)
            if parsed.netloc == base_parsed.netloc and parsed.scheme != base_parsed.scheme:
                href = urlunparse((base_parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))
            href = _normalize_url(href, default_scheme=base_parsed.scheme or "https")
            if href in seen:
                continue
            seen.add(href)
            if not self._html_predicate(href):
                continue
            title = _clean_text(anchor.get_text(" ", strip=True))
            items.append(ParsedFeedEntry(url=href, title=title, entry_ref=_slug_from_url(href)))
            if len(items) >= limit:
                break
        return items

    def _html_predicate(self, href: str) -> bool:
        return False


class OpenAIDeveloperBlogExtractor(BaseSourceExtractor):
    source_name = "OpenAI Developer Blog"
    source_vendor = "openai"
    source_channel = "openai_developer_blog"
    source_channel_type = "official_blog"
    source_type = "official_blog"
    html_only = True

    def _html_predicate(self, href: str) -> bool:
        parsed = urlparse(href)
        path = parsed.path.rstrip("/")
        return (
            parsed.netloc == "developers.openai.com"
            and path.startswith("/blog/")
            and path != "/blog"
            and "/topic/" not in href
            and "/cookbook/" not in href
        )

    def _extract_source_specific_page_metadata(self, html: str, url: str) -> dict[str, Any]:
        return _extract_openai_developer_metadata(html)


class OpenAINewsExtractor(BaseSourceExtractor):
    source_name = "OpenAI News"
    source_vendor = "openai"
    source_channel = "openai_news"
    source_channel_type = "official_news_rss"
    source_type = "official_blog_index"
    feed_url = "https://openai.com/news/rss.xml"

    def _feed_entry_ref(self, link: str) -> str:
        return _slug_from_url(link)


class HtmlBlogExtractor(BaseSourceExtractor):
    def __init__(
        self,
        *,
        source_name: str,
        source_vendor: str,
        source_channel: str,
        source_channel_type: str,
        source_type: str,
        predicate: Callable[[str], bool],
        html_only: bool = False,
        feed_url: str = "",
    ):
        self.source_name = source_name
        self.source_vendor = source_vendor
        self.source_channel = source_channel
        self.source_channel_type = source_channel_type
        self.source_type = source_type
        self._predicate = predicate
        self.html_only = html_only
        self.feed_url = feed_url

    def _html_predicate(self, href: str) -> bool:
        return self._predicate(href)


class AnthropicNewsExtractor(HtmlBlogExtractor):
    def _extract_source_specific_page_metadata(self, html: str, url: str) -> dict[str, Any]:
        return _extract_anthropic_metadata(html)


class HuggingFaceBlogExtractor(HtmlBlogExtractor):
    def _extract_source_specific_page_metadata(self, html: str, url: str) -> dict[str, Any]:
        return _extract_huggingface_metadata(html)


class ArxivCategoryExtractor(BaseSourceExtractor):
    def __init__(self, *, source_name: str, category: str):
        self.source_name = source_name
        self.source_vendor = "arxiv"
        self.source_channel = f"arxiv_{category.lower().replace('.', '_')}"
        self.source_channel_type = "paper_feed"
        self.source_type = "paper_index"
        self.category = category
        self.feed_url = f"https://export.arxiv.org/rss/{category}"

    def metadata_precedence(self) -> list[str]:
        return [
            "source_feed_metadata",
            "page_embedded_metadata",
            "generic_crawler_inference",
        ]

    def _source_item_id_from_feed(self, entry: ParsedFeedEntry) -> str:
        paper_id = _arxiv_paper_id(entry.url)
        if paper_id:
            return paper_id
        return super()._source_item_id_from_feed(entry)

    def discover_latest(self, source: dict[str, Any], limit: int) -> list[DiscoveredSourceItem]:
        items = super().discover_latest(source, limit)
        for item in items:
            item.provenance = DiscoveryProvenance(
                method="rss",
                origin_url=self.feed_url,
                entry_ref=item.identity.source_item_id or item.url,
                discovered_at=item.provenance.discovered_at,
                rank=item.provenance.rank,
            )
            item.metadata.setdefault("arxiv_category", self.category)
        return items


def _deepmind_predicate(href: str) -> bool:
    parsed = urlparse(href)
    path = parsed.path.rstrip("/")
    return parsed.netloc == "deepmind.google" and path.startswith("/blog/") and path != "/blog" and "/page/" not in href


def _google_research_predicate(href: str) -> bool:
    parsed = urlparse(href)
    path = parsed.path.rstrip("/")
    if parsed.netloc != "research.google" or not path.startswith("/blog"):
        return False
    tail = path.replace("/blog", "", 1).strip("/")
    if not tail or re.fullmatch(r"20\d{2}", tail):
        return False
    return path != "/blog/rss"


def _anthropic_predicate(href: str) -> bool:
    parsed = urlparse(href)
    path = parsed.path.rstrip("/")
    return parsed.netloc == "www.anthropic.com" and path.startswith("/news") and path != "/news"


def _bair_predicate(href: str) -> bool:
    parsed = urlparse(href)
    return parsed.netloc == "bair.berkeley.edu" and bool(re.search(r"^/blog/\d{4}/\d{2}/\d{2}/", (parsed.path or "") + "/"))


def _aws_predicate(href: str) -> bool:
    parsed = urlparse(href)
    path = parsed.path.rstrip("/")
    return (
        parsed.netloc == "aws.amazon.com"
        and path.startswith("/blogs/machine-learning")
        and path != "/blogs/machine-learning"
        and "/category/" not in href
        and "#Comments" not in href
    )


def _hf_predicate(href: str) -> bool:
    parsed = urlparse(href)
    path = parsed.path.rstrip("/")
    return (
        parsed.netloc == "huggingface.co"
        and path.startswith("/blog")
        and path not in {"/blog", "/blog/community"}
        and not path.endswith("-zh")
    )


def build_builtin_extractors() -> list[BaseSourceExtractor]:
    return [
        OpenAIDeveloperBlogExtractor(),
        OpenAINewsExtractor(),
        HtmlBlogExtractor(
            source_name="Google DeepMind Blog",
            source_vendor="google",
            source_channel="deepmind_blog",
            source_channel_type="official_blog",
            source_type="official_blog",
            predicate=_deepmind_predicate,
        ),
        HtmlBlogExtractor(
            source_name="Google Research Blog",
            source_vendor="google",
            source_channel="google_research_blog",
            source_channel_type="research_blog",
            source_type="research_blog",
            predicate=_google_research_predicate,
        ),
        AnthropicNewsExtractor(
            source_name="Anthropic News",
            source_vendor="anthropic",
            source_channel="anthropic_news",
            source_channel_type="official_blog",
            source_type="official_blog",
            predicate=_anthropic_predicate,
            html_only=True,
        ),
        HtmlBlogExtractor(
            source_name="BAIR Blog",
            source_vendor="berkeley",
            source_channel="bair_blog",
            source_channel_type="research_blog",
            source_type="research_blog",
            predicate=_bair_predicate,
        ),
        HtmlBlogExtractor(
            source_name="AWS Machine Learning Blog",
            source_vendor="amazon",
            source_channel="aws_ml_blog",
            source_channel_type="official_blog",
            source_type="official_blog",
            predicate=_aws_predicate,
        ),
        HuggingFaceBlogExtractor(
            source_name="Hugging Face Blog",
            source_vendor="huggingface",
            source_channel="huggingface_blog",
            source_channel_type="official_blog",
            source_type="official_blog",
            predicate=_hf_predicate,
            html_only=True,
        ),
        ArxivCategoryExtractor(source_name="arXiv cs.LG", category="cs.LG"),
        ArxivCategoryExtractor(source_name="arXiv cs.CL", category="cs.CL"),
    ]

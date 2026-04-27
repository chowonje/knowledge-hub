"""Routing rules for `khub add`."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import click

from knowledge_hub.web.youtube_extractor import is_youtube_url

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")
_PAPER_HOST_SUFFIXES = (
    "arxiv.org",
    "openreview.net",
    "semanticscholar.org",
    "paperswithcode.com",
    "huggingface.co",
    "aclanthology.org",
    "doi.org",
    "proceedings.mlr.press",
    "neurips.cc",
    "openaccess.thecvf.com",
)


@dataclass(frozen=True)
class AddRoute:
    kind: str
    source_type: str
    route: str
    reason: str


def _is_url(source: str) -> bool:
    parsed = urlsplit(str(source or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_pdf_url(source: str) -> bool:
    parsed = urlsplit(str(source or "").strip())
    return parsed.scheme in {"http", "https"} and (parsed.path or "").lower().endswith(".pdf")


def _local_pdf_path(source: str) -> Path | None:
    token = str(source or "").strip()
    if not token or _is_url(token):
        return None
    path = Path(token).expanduser()
    if path.suffix.lower() != ".pdf":
        return None
    return path


def _is_pdf_source(source: str) -> bool:
    return _is_pdf_url(source) or _local_pdf_path(source) is not None


def _is_paper_url(source: str) -> bool:
    parsed = urlsplit(str(source or "").strip())
    host = (parsed.netloc or "").lower()
    if not host:
        return False
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in _PAPER_HOST_SUFFIXES)


def detect_add_route(source: str, source_type: str = "auto") -> AddRoute:
    """Select the existing subsystem that should own a source add request."""
    source_token = str(source or "").strip()
    requested = str(source_type or "auto").strip().lower()
    if not source_token:
        raise click.BadParameter("source is required")

    if requested == "youtube":
        if not is_youtube_url(source_token):
            raise click.BadParameter("--type youtube requires a YouTube URL")
        return AddRoute("youtube", "youtube", "crawl_youtube_ingest", "explicit_type")
    if requested == "web":
        if not _is_url(source_token):
            raise click.BadParameter("--type web requires an http(s) URL")
        return AddRoute("web", "web", "crawl_ingest", "explicit_type")
    if requested == "pdf":
        if not _is_pdf_source(source_token):
            raise click.BadParameter("--type pdf requires a PDF URL or local .pdf path")
        return AddRoute("pdf", "pdf", "crawl_ingest", "explicit_type")
    if requested == "paper":
        if _local_pdf_path(source_token) is not None:
            raise click.BadParameter(
                "--type paper supports paper identifiers, paper URLs, or PDF URLs; local paper PDFs are not supported yet"
            )
        if _ARXIV_ID_RE.match(source_token) or _is_paper_url(source_token) or _is_pdf_url(source_token):
            return AddRoute("paper_url", "paper", "paper_import", "explicit_type")
        if _is_url(source_token):
            raise click.BadParameter("--type paper requires a paper URL, PDF URL, arXiv id, or paper query")
        return AddRoute("paper_query", "paper", "discover", "explicit_type")
    if requested != "auto":
        raise click.BadParameter(f"unsupported source type: {source_type}")

    if is_youtube_url(source_token):
        return AddRoute("youtube", "youtube", "crawl_youtube_ingest", "youtube_url")
    if _ARXIV_ID_RE.match(source_token) or _is_paper_url(source_token):
        return AddRoute("paper_url", "paper", "paper_import", "paper_identifier")
    if _is_pdf_source(source_token):
        return AddRoute("pdf", "pdf", "crawl_ingest", "pdf_source")
    if _is_url(source_token):
        return AddRoute("web", "web", "crawl_ingest", "web_url")
    return AddRoute("paper_query", "paper", "discover", "non_url_query")

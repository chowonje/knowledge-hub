"""
URL 기반 논문 소스 해석 모듈

다양한 논문 사이트 URL을 입력받아 메타데이터를 추출하고 PDF URL을 반환합니다.
지원: arXiv, OpenReview, Papers With Code, HuggingFace Papers, Semantic Scholar, 직접 PDF URL
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests
from rich.console import Console

console = Console()

RESOLVE_TIMEOUT = 30

# Semantic Scholar fields for paper lookup
SS_FIELDS = "title,authors,year,citationCount,externalIds,abstract,fieldsOfStudy,openAccessPdf"


@dataclass
class ResolvedPaper:
    """URL에서 추출된 논문 메타데이터"""
    arxiv_id: str
    title: str
    authors: str = ""
    year: int = 0
    abstract: str = ""
    citation_count: int = 0
    fields_of_study: List[str] = field(default_factory=list)
    pdf_url: str = ""
    source: str = ""


def detect_source(url: str) -> Tuple[str, str]:
    """URL 패턴으로 소스 감지. (source_type, identifier) 반환"""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path

    if "arxiv.org" in host:
        m = re.search(r"(\d{4}\.\d{4,5})", path)
        if m:
            return "arxiv", m.group(1)

    if "openreview.net" in host:
        qs = parse_qs(parsed.query)
        paper_id = qs.get("id", [None])[0]
        if paper_id:
            return "openreview", paper_id

    if "paperswithcode.com" in host:
        m = re.match(r"/paper/(.+?)/?$", path)
        if m:
            return "paperswithcode", m.group(1)

    if "huggingface.co" in host and "/papers/" in path:
        m = re.search(r"/papers/(\d{4}\.\d{4,5})", path)
        if m:
            return "huggingface", m.group(1)

    if "semanticscholar.org" in host:
        m = re.search(r"/paper/(?:.*?/)?([a-f0-9]{40})", path)
        if m:
            return "semanticscholar", m.group(1)

    if url.lower().endswith(".pdf"):
        return "direct_pdf", url

    return "unknown", url


def _ss_lookup(paper_id: str) -> Optional[ResolvedPaper]:
    """Semantic Scholar API로 논문 조회"""
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    try:
        resp = requests.get(url, params={"fields": SS_FIELDS}, timeout=RESOLVE_TIMEOUT)
        if resp.status_code != 200:
            return None
        data = resp.json()
        ext_ids = data.get("externalIds", {})
        arxiv_id = ext_ids.get("ArXiv", "")
        authors_list = data.get("authors") or []
        authors_str = ", ".join(a.get("name", "") for a in authors_list[:5])
        if len(authors_list) > 5:
            authors_str += f" 외 {len(authors_list) - 5}명"
        oa_pdf = data.get("openAccessPdf") or {}
        return ResolvedPaper(
            arxiv_id=arxiv_id or ext_ids.get("DOI", ""),
            title=data.get("title", ""),
            authors=authors_str,
            year=data.get("year") or 0,
            abstract=data.get("abstract") or "",
            citation_count=data.get("citationCount") or 0,
            fields_of_study=data.get("fieldsOfStudy") or [],
            pdf_url=oa_pdf.get("url", ""),
            source="semantic_scholar",
        )
    except Exception:
        return None


def resolve_arxiv(arxiv_id: str) -> Optional[ResolvedPaper]:
    paper = _ss_lookup(f"ArXiv:{arxiv_id}")
    if paper:
        paper.arxiv_id = arxiv_id
        paper.source = "arxiv"
        if not paper.pdf_url:
            paper.pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return paper
    # fallback: arXiv API
    import xml.etree.ElementTree as ET
    try:
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={"id_list": arxiv_id, "max_results": 1},
            timeout=RESOLVE_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        entry = root.find("a:entry", ns)
        if entry is None:
            return None
        title = " ".join((entry.findtext("a:title", "", ns) or "").split())
        abstract = " ".join((entry.findtext("a:summary", "", ns) or "").split())
        authors = []
        for a in entry.findall("a:author", ns):
            name = a.findtext("a:name", "", ns)
            if name:
                authors.append(name)
        published = entry.findtext("a:published", "", ns)
        year = int(published[:4]) if published and len(published) >= 4 else 0
        return ResolvedPaper(
            arxiv_id=arxiv_id,
            title=title,
            authors=", ".join(authors[:5]),
            year=year,
            abstract=abstract,
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            source="arxiv",
        )
    except Exception:
        return None


def resolve_openreview(paper_id: str) -> Optional[ResolvedPaper]:
    try:
        resp = requests.get(
            f"https://api2.openreview.net/notes?id={paper_id}",
            timeout=RESOLVE_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        notes = data.get("notes", [])
        if not notes:
            return None
        note = notes[0]
        content = note.get("content", {})

        def _val(v):
            return v.get("value", v) if isinstance(v, dict) else v

        title = _val(content.get("title", ""))
        abstract = _val(content.get("abstract", ""))
        authors = _val(content.get("authors", []))
        if isinstance(authors, list):
            authors = ", ".join(authors[:5])

        pdf_path = _val(content.get("pdf", ""))
        pdf_url = f"https://openreview.net{pdf_path}" if pdf_path and pdf_path.startswith("/") else ""

        return ResolvedPaper(
            arxiv_id=paper_id,
            title=title,
            authors=authors if isinstance(authors, str) else "",
            abstract=abstract,
            pdf_url=pdf_url,
            source="openreview",
        )
    except Exception:
        return None


def resolve_paperswithcode(slug: str) -> Optional[ResolvedPaper]:
    try:
        resp = requests.get(
            f"https://paperswithcode.com/api/v1/papers/{slug}/",
            timeout=RESOLVE_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        arxiv_id = data.get("arxiv_id", "")
        if arxiv_id:
            paper = resolve_arxiv(arxiv_id)
            if paper:
                paper.source = "paperswithcode"
                return paper
        return ResolvedPaper(
            arxiv_id=arxiv_id or slug,
            title=data.get("title", ""),
            authors=data.get("authors", ""),
            abstract=data.get("abstract", ""),
            pdf_url=data.get("url_pdf", ""),
            source="paperswithcode",
        )
    except Exception:
        return None


def resolve_huggingface(arxiv_id: str) -> Optional[ResolvedPaper]:
    paper = resolve_arxiv(arxiv_id)
    if paper:
        paper.source = "huggingface"
    return paper


def resolve_semanticscholar(corpus_id: str) -> Optional[ResolvedPaper]:
    paper = _ss_lookup(f"CorpusId:{corpus_id}")
    if paper:
        paper.source = "semanticscholar"
    return paper


def resolve_direct_pdf(url: str) -> Optional[ResolvedPaper]:
    """직접 PDF URL - 다운로드 가능 여부만 확인"""
    try:
        resp = requests.head(url, timeout=RESOLVE_TIMEOUT, allow_redirects=True)
        ct = resp.headers.get("Content-Type", "")
        if resp.status_code == 200 and ("pdf" in ct or url.endswith(".pdf")):
            filename = url.split("/")[-1].replace(".pdf", "").replace("_", " ")
            return ResolvedPaper(
                arxiv_id="",
                title=filename,
                pdf_url=url,
                source="direct_pdf",
            )
    except Exception:
        pass
    return None


_RESOLVERS = {
    "arxiv": resolve_arxiv,
    "openreview": resolve_openreview,
    "paperswithcode": resolve_paperswithcode,
    "huggingface": resolve_huggingface,
    "semanticscholar": resolve_semanticscholar,
    "direct_pdf": resolve_direct_pdf,
}


def resolve_url(url: str) -> Optional[ResolvedPaper]:
    """URL을 파싱하여 논문 메타데이터를 반환"""
    source_type, identifier = detect_source(url)

    if source_type == "unknown":
        console.print(f"[yellow]인식할 수 없는 URL입니다: {url}[/yellow]")
        console.print("[dim]지원: arXiv, OpenReview, Papers With Code, HuggingFace, Semantic Scholar, PDF URL[/dim]")
        return None

    resolver = _RESOLVERS.get(source_type)
    if not resolver:
        return None

    console.print(f"[dim]소스 감지: {source_type} | ID: {identifier}[/dim]")
    return resolver(identifier)

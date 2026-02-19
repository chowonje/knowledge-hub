"""
논문 자동 발견 모듈

Semantic Scholar + arXiv API를 사용하여
최신/중요 논문을 검색하고 메타데이터를 반환합니다.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from rich.console import Console

console = Console()

SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_PAPER = "https://api.semanticscholar.org/graph/v1/paper"
ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

SS_FIELDS = "title,authors,year,citationCount,externalIds,abstract,fieldsOfStudy,publicationDate,openAccessPdf"


@dataclass
class DiscoveredPaper:
    arxiv_id: str
    title: str
    authors: str
    year: int
    abstract: str
    citation_count: int = 0
    fields_of_study: List[str] = field(default_factory=list)
    pdf_url: str = ""
    source: str = "semantic_scholar"


def _extract_arxiv_id(external_ids: dict | None) -> str | None:
    if not external_ids:
        return None
    arxiv = external_ids.get("ArXiv")
    if arxiv:
        return str(arxiv).strip()
    doi = external_ids.get("DOI", "")
    match = re.search(r"(\d{4}\.\d{4,5})", str(doi))
    return match.group(1) if match else None


def search_semantic_scholar(
    query: str,
    limit: int = 10,
    year_start: int | None = None,
    min_citations: int = 0,
    fields_of_study: List[str] | None = None,
    sort_by: str = "relevance",
) -> List[DiscoveredPaper]:
    params: dict = {
        "query": query,
        "limit": min(limit, 100),
        "fields": SS_FIELDS,
    }
    if year_start:
        params["year"] = f"{year_start}-"
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    data = None
    for attempt in range(3):
        try:
            resp = requests.get(SEMANTIC_SCHOLAR_SEARCH, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                console.print(f"[yellow]Semantic Scholar rate limited, {wait}s 대기 중...[/yellow]")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.HTTPError:
            if attempt < 2:
                time.sleep(2)
                continue
            console.print(f"[yellow]Semantic Scholar API 사용 불가, arXiv만 사용합니다.[/yellow]")
            return []
        except Exception as e:
            console.print(f"[red]Semantic Scholar API 오류: {e}[/red]")
            return []

    if data is None:
        return []

    papers = []
    for item in data.get("data", []):
        arxiv_id = _extract_arxiv_id(item.get("externalIds"))
        if not arxiv_id:
            continue
        citations = item.get("citationCount") or 0
        if citations < min_citations:
            continue

        authors_list = item.get("authors") or []
        authors_str = ", ".join(a.get("name", "") for a in authors_list[:5])
        if len(authors_list) > 5:
            authors_str += f" 외 {len(authors_list) - 5}명"

        oa_pdf = item.get("openAccessPdf") or {}

        papers.append(DiscoveredPaper(
            arxiv_id=arxiv_id,
            title=item.get("title", ""),
            authors=authors_str,
            year=item.get("year") or 0,
            abstract=item.get("abstract") or "",
            citation_count=citations,
            fields_of_study=item.get("fieldsOfStudy") or [],
            pdf_url=oa_pdf.get("url", ""),
            source="semantic_scholar",
        ))

    if sort_by == "citationCount":
        papers.sort(key=lambda p: p.citation_count, reverse=True)

    return papers


def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    category: str | None = None,
) -> List[DiscoveredPaper]:
    search_query = query
    if category:
        search_query = f"cat:{category} AND all:{query}"

    params = {
        "search_query": search_query,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(ARXIV_API, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 3 * (attempt + 1)
                console.print(f"[yellow]arXiv rate limited, {wait}s 대기 중...[/yellow]")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            console.print(f"[red]arXiv API 오류: {e}[/red]")
            return []

    if resp is None or resp.status_code != 200:
        return []

    root = ET.fromstring(resp.text)
    papers = []

    for entry in root.findall("atom:entry", ARXIV_NS):
        id_text = entry.findtext("atom:id", "", ARXIV_NS)
        match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", id_text)
        if not match:
            continue
        arxiv_id = match.group(1)

        title = (entry.findtext("atom:title", "", ARXIV_NS) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", "", ARXIV_NS) or "").strip().replace("\n", " ")

        authors = []
        for author_el in entry.findall("atom:author", ARXIV_NS):
            name = author_el.findtext("atom:name", "", ARXIV_NS)
            if name:
                authors.append(name)
        authors_str = ", ".join(authors[:5])
        if len(authors) > 5:
            authors_str += f" 외 {len(authors) - 5}명"

        published = entry.findtext("atom:published", "", ARXIV_NS)
        year = int(published[:4]) if published and len(published) >= 4 else 0

        pdf_url = ""
        for link_el in entry.findall("atom:link", ARXIV_NS):
            if link_el.get("title") == "pdf":
                pdf_url = link_el.get("href", "")
                break

        categories = []
        for cat_el in entry.findall("atom:category", ARXIV_NS):
            term = cat_el.get("term", "")
            if term:
                categories.append(term)

        papers.append(DiscoveredPaper(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors_str,
            year=year,
            abstract=abstract,
            citation_count=0,
            fields_of_study=categories,
            pdf_url=pdf_url,
            source="arxiv",
        ))

    return papers


def discover_papers(
    topic: str,
    max_papers: int = 10,
    year_start: int | None = None,
    min_citations: int = 0,
    sort_by: str = "relevance",
    include_arxiv: bool = True,
    arxiv_category: str | None = None,
) -> List[DiscoveredPaper]:
    """Semantic Scholar + arXiv에서 통합 검색 후 중복 제거하여 반환"""
    seen_ids: set[str] = set()
    results: List[DiscoveredPaper] = []

    ss_papers = search_semantic_scholar(
        query=topic,
        limit=max_papers,
        year_start=year_start,
        min_citations=min_citations,
        sort_by=sort_by,
    )
    for p in ss_papers:
        if p.arxiv_id not in seen_ids:
            seen_ids.add(p.arxiv_id)
            results.append(p)

    if include_arxiv:
        time.sleep(0.5)
        arxiv_papers = search_arxiv(
            query=topic,
            max_results=max_papers,
            category=arxiv_category,
        )
        for p in arxiv_papers:
            if p.arxiv_id not in seen_ids:
                seen_ids.add(p.arxiv_id)
                results.append(p)

    if sort_by == "citationCount":
        results.sort(key=lambda p: p.citation_count, reverse=True)

    return results[:max_papers]

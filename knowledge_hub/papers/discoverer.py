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



# ─────────────────────────────────────────────────────────
#  Semantic Scholar – 저자 / 인용 / 참고문헌 / 배치 API
# ─────────────────────────────────────────────────────────

SEMANTIC_SCHOLAR_AUTHOR_SEARCH = "https://api.semanticscholar.org/graph/v1/author/search"
SEMANTIC_SCHOLAR_AUTHOR = "https://api.semanticscholar.org/graph/v1/author"

SS_AUTHOR_FIELDS = "name,affiliations,paperCount,citationCount,hIndex"
SS_AUTHOR_PAPER_FIELDS = "title,year,citationCount,externalIds,abstract,fieldsOfStudy,openAccessPdf"


@dataclass
class AuthorInfo:
    author_id: str
    name: str
    affiliations: List[str]
    paper_count: int = 0
    citation_count: int = 0
    h_index: int = 0


def _ss_request(url: str, params: dict | None = None, timeout: int = 30) -> dict | list | None:
    """Semantic Scholar API 공통 요청 (재시도 + rate limit 처리)"""
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                console.print(f"[yellow]Semantic Scholar rate limited, {wait}s 대기...[/yellow]")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError:
            if attempt < 2:
                time.sleep(2)
                continue
            return None
        except Exception as e:
            console.print(f"[red]API 오류: {e}[/red]")
            return None
    return None


def search_authors(query: str, limit: int = 10, offset: int = 0) -> List[AuthorInfo]:
    """저자 이름/소속으로 검색"""
    data = _ss_request(SEMANTIC_SCHOLAR_AUTHOR_SEARCH, {
        "query": query,
        "limit": min(limit, 100),
        "offset": offset,
        "fields": SS_AUTHOR_FIELDS,
    })
    if not data or "data" not in data:
        return []

    results = []
    for item in data["data"]:
        results.append(AuthorInfo(
            author_id=str(item.get("authorId", "")),
            name=item.get("name", ""),
            affiliations=item.get("affiliations") or [],
            paper_count=item.get("paperCount") or 0,
            citation_count=item.get("citationCount") or 0,
            h_index=item.get("hIndex") or 0,
        ))
    return results


def get_author_papers(
    author_id: str, limit: int = 20, offset: int = 0
) -> tuple[AuthorInfo | None, List[DiscoveredPaper]]:
    """특정 저자의 논문 목록 조회"""
    author_data = _ss_request(
        f"{SEMANTIC_SCHOLAR_AUTHOR}/{author_id}",
        {"fields": SS_AUTHOR_FIELDS},
    )
    author = None
    if author_data:
        author = AuthorInfo(
            author_id=str(author_data.get("authorId", author_id)),
            name=author_data.get("name", ""),
            affiliations=author_data.get("affiliations") or [],
            paper_count=author_data.get("paperCount") or 0,
            citation_count=author_data.get("citationCount") or 0,
            h_index=author_data.get("hIndex") or 0,
        )

    data = _ss_request(
        f"{SEMANTIC_SCHOLAR_AUTHOR}/{author_id}/papers",
        {"fields": SS_AUTHOR_PAPER_FIELDS, "limit": min(limit, 1000), "offset": offset},
    )
    if not data or "data" not in data:
        return author, []

    papers = []
    for item in data["data"]:
        arxiv_id = _extract_arxiv_id(item.get("externalIds"))
        authors_list = item.get("authors") or []
        authors_str = ", ".join(a.get("name", "") for a in authors_list[:5])
        if len(authors_list) > 5:
            authors_str += f" 외 {len(authors_list) - 5}명"
        oa = item.get("openAccessPdf") or {}
        papers.append(DiscoveredPaper(
            arxiv_id=arxiv_id or "",
            title=item.get("title") or "",
            authors=authors_str,
            year=item.get("year") or 0,
            abstract=item.get("abstract") or "",
            citation_count=item.get("citationCount") or 0,
            fields_of_study=item.get("fieldsOfStudy") or [],
            pdf_url=oa.get("url", ""),
            source="semantic_scholar",
        ))

    papers.sort(key=lambda p: p.citation_count, reverse=True)
    return author, papers


def _normalize_paper_id(paper_id: str) -> str:
    """arXiv ID를 자동으로 ARXIV: prefix 붙여주기"""
    paper_id = paper_id.strip()
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", paper_id):
        return f"ARXIV:{paper_id}"
    return paper_id


def get_paper_detail(paper_id: str) -> dict | None:
    """논문 상세 정보 (abstract 포함)"""
    paper_id = _normalize_paper_id(paper_id)
    fields = "title,authors,year,citationCount,referenceCount,externalIds,abstract,fieldsOfStudy,publicationDate,openAccessPdf,venue,influentialCitationCount"
    data = _ss_request(f"{SEMANTIC_SCHOLAR_PAPER}/{paper_id}", {"fields": fields})
    return data


def get_paper_citations(
    paper_id: str, limit: int = 20, offset: int = 0
) -> tuple[str, List[DiscoveredPaper]]:
    """이 논문을 인용한 논문들 (피인용)"""
    paper_id = _normalize_paper_id(paper_id)
    fields = "title,authors,year,citationCount,externalIds,abstract,fieldsOfStudy"
    data = _ss_request(
        f"{SEMANTIC_SCHOLAR_PAPER}/{paper_id}/citations",
        {"fields": fields, "limit": min(limit, 1000), "offset": offset},
    )
    if not data or "data" not in data:
        return paper_id, []

    papers = []
    for item in data["data"]:
        citing = item.get("citingPaper", {})
        if not citing or not citing.get("title"):
            continue
        arxiv_id = _extract_arxiv_id(citing.get("externalIds"))
        authors_list = citing.get("authors") or []
        authors_str = ", ".join(a.get("name", "") for a in authors_list[:5])
        papers.append(DiscoveredPaper(
            arxiv_id=arxiv_id or "",
            title=citing.get("title", ""),
            authors=authors_str,
            year=citing.get("year") or 0,
            abstract=citing.get("abstract") or "",
            citation_count=citing.get("citationCount") or 0,
            fields_of_study=citing.get("fieldsOfStudy") or [],
            source="semantic_scholar",
        ))

    papers.sort(key=lambda p: p.citation_count, reverse=True)
    return paper_id, papers


def get_paper_references(
    paper_id: str, limit: int = 20, offset: int = 0
) -> tuple[str, List[DiscoveredPaper]]:
    """이 논문이 참고한 논문들 (참고문헌)"""
    paper_id = _normalize_paper_id(paper_id)
    fields = "title,authors,year,citationCount,externalIds,abstract,fieldsOfStudy"
    data = _ss_request(
        f"{SEMANTIC_SCHOLAR_PAPER}/{paper_id}/references",
        {"fields": fields, "limit": min(limit, 1000), "offset": offset},
    )
    if not data or "data" not in data:
        return paper_id, []

    papers = []
    for item in data["data"]:
        cited = item.get("citedPaper", {})
        if not cited or not cited.get("title"):
            continue
        arxiv_id = _extract_arxiv_id(cited.get("externalIds"))
        authors_list = cited.get("authors") or []
        authors_str = ", ".join(a.get("name", "") for a in authors_list[:5])
        papers.append(DiscoveredPaper(
            arxiv_id=arxiv_id or "",
            title=cited.get("title", ""),
            authors=authors_str,
            year=cited.get("year") or 0,
            abstract=cited.get("abstract") or "",
            citation_count=cited.get("citationCount") or 0,
            fields_of_study=cited.get("fieldsOfStudy") or [],
            source="semantic_scholar",
        ))

    papers.sort(key=lambda p: p.citation_count, reverse=True)
    return paper_id, papers


def get_papers_batch(paper_ids: List[str]) -> List[dict]:
    """복수 논문 일괄 조회"""
    fields = "title,authors,year,citationCount,externalIds,abstract,fieldsOfStudy,openAccessPdf"
    try:
        resp = requests.post(
            f"{SEMANTIC_SCHOLAR_PAPER}/batch",
            params={"fields": fields},
            json={"ids": paper_ids},
            timeout=30,
        )
        if resp.status_code == 429:
            time.sleep(3)
            resp = requests.post(
                f"{SEMANTIC_SCHOLAR_PAPER}/batch",
                params={"fields": fields},
                json={"ids": paper_ids},
                timeout=30,
            )
        resp.raise_for_status()
        return [p for p in resp.json() if p is not None]
    except Exception as e:
        console.print(f"[red]배치 조회 오류: {e}[/red]")
        return []


def analyze_citation_network(
    paper_id: str, depth: int = 1, citations_limit: int = 10, references_limit: int = 10,
) -> dict:
    """인용 네트워크 분석 — depth 1 또는 2"""
    paper_id = _normalize_paper_id(paper_id)
    detail = get_paper_detail(paper_id)
    if not detail:
        return {"error": f"논문 '{paper_id}'를 찾을 수 없습니다."}

    root_title = detail.get("title", "Unknown")
    root_citations = detail.get("citationCount", 0)
    root_references = detail.get("referenceCount", 0)

    _, citing_papers = get_paper_citations(paper_id, limit=citations_limit)
    _, ref_papers = get_paper_references(paper_id, limit=references_limit)

    network = {
        "root": {
            "paper_id": paper_id,
            "title": root_title,
            "citation_count": root_citations,
            "reference_count": root_references,
        },
        "citations": [
            {"title": p.title, "arxiv_id": p.arxiv_id, "year": p.year, "citations": p.citation_count}
            for p in citing_papers
        ],
        "references": [
            {"title": p.title, "arxiv_id": p.arxiv_id, "year": p.year, "citations": p.citation_count}
            for p in ref_papers
        ],
        "depth": depth,
    }

    if depth >= 2:
        network["citations_of_citations"] = {}
        top_citing = sorted(citing_papers, key=lambda p: p.citation_count, reverse=True)[:5]
        for p in top_citing:
            pid = p.arxiv_id or p.title[:40]
            if p.arxiv_id:
                _, sub_citations = get_paper_citations(p.arxiv_id, limit=5)
                time.sleep(0.3)
                network["citations_of_citations"][pid] = [
                    {"title": sp.title, "year": sp.year, "citations": sp.citation_count}
                    for sp in sub_citations
                ]

    year_dist = {}
    for p in citing_papers:
        if p.year:
            year_dist[p.year] = year_dist.get(p.year, 0) + 1
    network["citation_year_distribution"] = dict(sorted(year_dist.items()))

    top_fields: dict[str, int] = {}
    for p in citing_papers + ref_papers:
        for f in p.fields_of_study:
            top_fields[f] = top_fields.get(f, 0) + 1
    network["top_fields"] = dict(sorted(top_fields.items(), key=lambda x: x[1], reverse=True)[:10])

    return network


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

"""User-facing paper reading helpers.

These helpers keep the public CLI surface thin while reusing the structured
summary, paper memory, and search backends.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from knowledge_hub.papers.memory_builder import PaperMemoryBuilder
from knowledge_hub.papers.memory_retriever import PaperMemoryRetriever
from knowledge_hub.papers.structured_summary import StructuredPaperSummaryService


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _summary_service(khub) -> StructuredPaperSummaryService:
    return StructuredPaperSummaryService(khub.sqlite_db(), khub.config)


def load_public_summary_artifact(khub, *, paper_id: str) -> dict[str, Any]:
    return dict(_summary_service(khub).load_artifact(paper_id=paper_id) or {})


def ensure_public_summary(khub, *, paper_id: str) -> dict[str, Any]:
    service = _summary_service(khub)
    payload = service.load_artifact(paper_id=paper_id)
    if payload:
        return dict(payload)
    return service.build(paper_id=paper_id, paper_parser="auto", refresh_parse=False, quick=False)


def ensure_public_memory_card(khub, *, paper_id: str, include_refs: bool = True) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    retriever = PaperMemoryRetriever(sqlite_db)
    card = retriever.get(paper_id, include_refs=include_refs)
    if card:
        return dict(card)
    PaperMemoryBuilder(sqlite_db).build_and_store(paper_id=paper_id)
    card = retriever.get(paper_id, include_refs=include_refs)
    return dict(card or {})


def load_public_memory_card(khub, *, paper_id: str, include_refs: bool = True) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    retriever = PaperMemoryRetriever(sqlite_db)
    return dict(retriever.get(paper_id, include_refs=include_refs) or {})


def build_public_summary_card(khub, *, paper_id: str) -> dict[str, Any]:
    payload = ensure_public_summary(khub, paper_id=paper_id)
    summary = dict(payload.get("summary") or {})
    context_stats = dict(payload.get("contextStats") or {})
    claim_coverage = dict(context_stats.get("claimCoverage") or {})
    return {
        "schema": "knowledge-hub.paper.public.summary.v1",
        "status": payload.get("status") or "partial",
        "paperId": payload.get("paperId") or paper_id,
        "paperTitle": payload.get("paperTitle") or "",
        "parserUsed": payload.get("parserUsed") or payload.get("parser_used") or "raw",
        "fallbackUsed": bool(payload.get("fallbackUsed")),
        "warnings": list(payload.get("warnings") or []),
        "summary": summary,
        "claimCoverage": claim_coverage,
    }


def build_public_evidence_card(khub, *, paper_id: str) -> dict[str, Any]:
    payload = ensure_public_summary(khub, paper_id=paper_id)
    context_stats = dict(payload.get("contextStats") or {})
    claim_coverage = dict(context_stats.get("claimCoverage") or {})
    evidence_summaries = dict(payload.get("evidenceSummaries") or {})
    evidence_map = list(payload.get("evidenceMap") or [])
    return {
        "schema": "knowledge-hub.paper.public.evidence.v1",
        "status": payload.get("status") or "partial",
        "paperId": payload.get("paperId") or paper_id,
        "paperTitle": payload.get("paperTitle") or "",
        "parserUsed": payload.get("parserUsed") or payload.get("parser_used") or "raw",
        "fallbackUsed": bool(payload.get("fallbackUsed")),
        "warnings": list(payload.get("warnings") or []),
        "claimCoverage": claim_coverage,
        "evidenceSummaries": evidence_summaries,
        "evidenceMap": evidence_map,
    }


def build_public_memory_card(khub, *, paper_id: str) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    paper = sqlite_db.get_paper(paper_id) or {}
    memory_card = ensure_public_memory_card(khub, paper_id=paper_id, include_refs=True)
    summary = ensure_public_summary(khub, paper_id=paper_id)
    context_stats = dict(summary.get("contextStats") or {})
    claim_coverage = dict(context_stats.get("claimCoverage") or {})
    return {
        "schema": "knowledge-hub.paper.public.memory.v1",
        "status": "ok" if memory_card else "needs_setup",
        "paperId": paper_id,
        "paperTitle": _clean_text(paper.get("title") or summary.get("paperTitle") or paper_id),
        "memory": memory_card,
        "claimCoverage": claim_coverage,
        "provenance": {
            "paperId": paper_id,
            "paperMemoryAvailable": bool(memory_card),
            "summaryArtifact": bool(summary),
        },
    }


def _dedupe_by_key(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        token = _clean_text(row.get(key))
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(row)
    return out


def _clean_list(values: Any, *, limit: int | None = None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, list):
        raw = values
    elif isinstance(values, tuple):
        raw = list(values)
    else:
        raw = [values]
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = _clean_text(item)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
        if limit is not None and len(result) >= limit:
            break
    return result


def _claim_coverage_from_summary(summary_payload: dict[str, Any]) -> dict[str, Any]:
    context_stats = dict(summary_payload.get("contextStats") or {})
    return dict(context_stats.get("claimCoverage") or summary_payload.get("claimCoverage") or {})


def _search_terms(*parts: Any, limit: int = 6) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for candidate in re.split(r"[^0-9A-Za-z가-힣]+", _clean_text(part)):
            token = candidate.strip()
            if len(token) < 3:
                continue
            lowered = token.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            tokens.append(token)
            if len(tokens) >= limit:
                return tokens
    return tokens


def _top_concepts(sqlite_db: Any, *, paper_id: str, memory_card: dict[str, Any], limit: int) -> list[str]:
    concepts = _clean_list(memory_card.get("conceptLinks"), limit=limit)
    if concepts:
        return concepts
    rows = getattr(sqlite_db, "get_paper_concepts", lambda _paper_id: [])(paper_id) or []
    return _clean_list(
        [
            row.get("canonical_name")
            or row.get("canonicalName")
            or row.get("name")
            or row.get("entity_id")
            for row in rows
        ],
        limit=limit,
    )


def _build_related_papers_read_only(
    khub,
    *,
    paper_id: str,
    paper: dict[str, Any],
    summary_payload: dict[str, Any],
    memory_card: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    sqlite_db = khub.sqlite_db()
    title = _clean_text(paper.get("title") or summary_payload.get("paperTitle") or paper_id)
    field = _clean_text(paper.get("field"))
    concepts = _top_concepts(sqlite_db, paper_id=paper_id, memory_card=memory_card, limit=max(2, top_k))
    terms = _search_terms(title, field, *concepts, limit=6)

    candidates: list[dict[str, Any]] = []
    if field:
        candidates.extend(sqlite_db.list_papers(field=field, limit=max(10, top_k * 4)))
    for term in terms:
        candidates.extend(sqlite_db.search_papers(term, limit=max(4, top_k * 2)))
    if memory_card:
        query = " ".join(concepts[:2] or terms[:2] or [title])
        for row in PaperMemoryRetriever(sqlite_db).search(query, limit=max(4, top_k * 2), include_refs=False):
            target_id = _clean_text(row.get("paperId"))
            if not target_id:
                continue
            target_paper = sqlite_db.get_paper(target_id) or {}
            if target_paper:
                candidates.append(target_paper)

    rows = _dedupe_by_key(
        [
            {
                "paperId": _clean_text(item.get("arxiv_id")),
                "title": _clean_text(item.get("title")),
                "year": item.get("year"),
                "field": _clean_text(item.get("field")),
            }
            for item in candidates
            if _clean_text(item.get("arxiv_id")) and _clean_text(item.get("arxiv_id")) != _clean_text(paper_id)
        ],
        key="paperId",
    )
    rows.sort(
        key=lambda item: (
            -int(item.get("year") or 0),
            item.get("title") or "",
        )
    )
    return rows[:top_k]


def build_public_board_export(
    khub,
    *,
    field: str | None = None,
    limit: int = 50,
    concept_limit: int = 4,
    related_limit: int = 4,
) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    papers = sqlite_db.list_papers(field=field, limit=max(1, int(limit)))

    items: list[dict[str, Any]] = []
    top_warnings: list[str] = []
    summary_count = 0
    memory_count = 0
    indexed_count = 0

    for paper in papers:
        paper_id = _clean_text(paper.get("arxiv_id"))
        summary_payload = load_public_summary_artifact(khub, paper_id=paper_id)
        memory_card = load_public_memory_card(khub, paper_id=paper_id, include_refs=False)
        summary = dict(summary_payload.get("summary") or {})
        claim_coverage = _claim_coverage_from_summary(summary_payload)
        concepts = _top_concepts(sqlite_db, paper_id=paper_id, memory_card=memory_card, limit=max(1, int(concept_limit)))
        has_summary = bool(summary_payload) and bool(
            _clean_text(summary.get("oneLine")) or _clean_text(summary.get("coreIdea"))
        )
        has_memory = bool(memory_card) and bool(
            _clean_text(memory_card.get("paperCore"))
            or _clean_text(memory_card.get("methodCore"))
            or _clean_text(memory_card.get("evidenceCore"))
        )
        warnings: list[str] = []
        if not has_summary:
            warnings.append("summary_artifact_missing")
        if not has_memory:
            warnings.append("memory_card_missing")
        if not concepts:
            warnings.append("concept_links_missing")

        related_papers = _build_related_papers_read_only(
            khub,
            paper_id=paper_id,
            paper=paper,
            summary_payload=summary_payload,
            memory_card=memory_card,
            top_k=max(1, int(related_limit)),
        )
        if not related_papers:
            warnings.append("related_papers_sparse")

        if has_summary:
            summary_count += 1
        if has_memory:
            memory_count += 1
        if bool(paper.get("indexed")):
            indexed_count += 1

        items.append(
            {
                "paperId": paper_id,
                "title": _clean_text(paper.get("title")),
                "year": paper.get("year"),
                "field": _clean_text(paper.get("field")),
                "artifactFlags": {
                    "hasPdf": bool(_clean_text(paper.get("pdf_path"))),
                    "hasSummary": has_summary,
                    "hasTranslation": bool(_clean_text(paper.get("translated_path"))),
                    "isIndexed": bool(paper.get("indexed")),
                    "hasMemory": has_memory,
                },
                "summary": {
                    "oneLine": _clean_text(summary.get("oneLine")),
                    "coreIdea": _clean_text(summary.get("coreIdea")),
                },
                "memory": {
                    "paperCore": _clean_text(memory_card.get("paperCore")),
                    "methodCore": _clean_text(memory_card.get("methodCore")),
                    "evidenceCore": _clean_text(memory_card.get("evidenceCore")),
                    "qualityFlag": _clean_text(memory_card.get("qualityFlag")) or "missing",
                },
                "concepts": concepts,
                "relatedPapers": related_papers,
                "paths": {
                    "pdfPath": _clean_text(paper.get("pdf_path")),
                    "translatedPath": _clean_text(paper.get("translated_path")),
                },
                "warnings": warnings,
                "claimCoverage": claim_coverage,
            }
        )

    payload = {
        "schema": "knowledge-hub.paper.board-export.v1",
        "status": "ok" if items else "empty",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": {
            "mode": "read_only",
            "field": _clean_text(field),
            "limit": max(1, int(limit)),
            "conceptLimit": max(1, int(concept_limit)),
            "relatedLimit": max(1, int(related_limit)),
        },
        "papers": items,
        "stats": {
            "returnedPapers": len(items),
            "papersWithSummary": summary_count,
            "papersWithMemory": memory_count,
            "indexedPapers": indexed_count,
        },
        "warnings": top_warnings,
    }
    return payload


def build_public_related_card(khub, *, paper_id: str, top_k: int = 5) -> dict[str, Any]:
    sqlite_db = khub.sqlite_db()
    paper = sqlite_db.get_paper(paper_id) or {}
    memory_card = ensure_public_memory_card(khub, paper_id=paper_id, include_refs=True)
    summary = ensure_public_summary(khub, paper_id=paper_id)
    query_parts = [
        paper.get("title", ""),
        paper.get("field", ""),
        summary.get("summary", {}).get("oneLine", ""),
        summary.get("summary", {}).get("coreIdea", ""),
        summary.get("summary", {}).get("whatIsNew", ""),
        memory_card.get("paperCore", ""),
        memory_card.get("methodCore", ""),
        memory_card.get("evidenceCore", ""),
        " ".join(memory_card.get("conceptLinks") or []),
    ]
    query = " ".join(part for part in (_clean_text(item) for item in query_parts) if part)
    related_papers = sqlite_db.search_papers(query or paper_id, limit=max(1, top_k * 3))
    related_memory = PaperMemoryRetriever(sqlite_db).search(query or paper_id, limit=max(1, top_k * 3), include_refs=False)
    related_claims = list((memory_card.get("claims") or [])[:top_k]) if isinstance(memory_card.get("claims"), list) else []
    related_concepts = list((memory_card.get("conceptLinks") or [])[:top_k])
    return {
        "schema": "knowledge-hub.paper.public.related.v1",
        "status": "ok" if related_papers or related_memory or related_claims or related_concepts else "degraded",
        "paperId": paper_id,
        "paperTitle": _clean_text(paper.get("title") or summary.get("paperTitle") or paper_id),
        "query": query,
        "relatedPapers": _dedupe_by_key(
            [
                {
                    "paperId": _clean_text(row.get("arxiv_id")),
                    "title": _clean_text(row.get("title")),
                    "field": _clean_text(row.get("field")),
                    "year": row.get("year"),
                }
                for row in related_papers
            ],
            key="paperId",
        )[:top_k],
        "relatedMemory": _dedupe_by_key(
            [
                {
                    "paperId": _clean_text(row.get("paperId")),
                    "title": _clean_text(row.get("title")),
                    "paperCore": _clean_text(row.get("paperCore")),
                    "qualityFlag": _clean_text(row.get("qualityFlag")),
                }
                for row in related_memory
            ],
            key="paperId",
        )[:top_k],
        "relatedClaims": related_claims,
        "relatedConcepts": related_concepts,
    }


__all__ = [
    "build_public_board_export",
    "build_public_evidence_card",
    "build_public_memory_card",
    "build_public_related_card",
    "build_public_summary_card",
    "ensure_public_memory_card",
    "ensure_public_summary",
    "load_public_memory_card",
    "load_public_summary_artifact",
]

"""Evaluation helpers for paper-memory retrieval quality.

This module measures retrieval/resolution quality across three current
surfaces without changing runtime behavior:

- `search_papers`
- `paper_lookup_and_summarize` candidate resolution
- `paper-memory search`

The second surface is evaluated at the resolver stage because V1 is validating
retrieval quality, not downstream answer quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from knowledge_hub.papers.memory_retriever import PaperMemoryRetriever


def _pick_candidate(items: list[dict[str, Any]], query: str) -> dict[str, Any] | None:
    if not items:
        return None
    lowered = str(query or "").strip().lower()
    if not lowered:
        return items[0]
    for item in items:
        title = str(item.get("title", "")).strip().lower()
        paper_id = str(item.get("arxiv_id", item.get("paper_id", ""))).strip().lower()
        if title == lowered or paper_id == lowered:
            return item
    for item in items:
        title = str(item.get("title", "")).strip().lower()
        if lowered and lowered in title:
            return item
    return items[0]


def _paper_id(item: dict[str, Any]) -> str:
    return str(item.get("paperId") or item.get("paper_id") or item.get("arxiv_id") or "").strip()


def _first_useful_position(items: list[dict[str, Any]], expected_paper_id: str) -> int | None:
    for index, item in enumerate(items, start=1):
        if _paper_id(item) == expected_paper_id:
            return index
    return None


def _drill_down_useful(item: dict[str, Any] | None) -> bool:
    if not item:
        return False
    if item.get("sourceNote") and str((item.get("sourceNote") or {}).get("id") or "").strip():
        return True
    if item.get("claims"):
        return True
    if item.get("paper") and str((item.get("paper") or {}).get("paperId") or "").strip():
        return True
    return False


def _is_weak_card(item: dict[str, Any]) -> bool:
    if not item:
        return True
    strong_fields = [
        str(item.get("paperCore") or "").strip(),
        str(item.get("problemContext") or "").strip(),
        str(item.get("methodCore") or "").strip(),
        str(item.get("evidenceCore") or "").strip(),
        str(item.get("limitations") or "").strip(),
    ]
    nonempty_count = sum(1 for value in strong_fields if value)
    return nonempty_count < 2 and not item.get("claimRefs") and not item.get("conceptLinks")


@dataclass(frozen=True)
class PaperMemoryEvalCase:
    case_id: str
    query: str
    expected_paper_id: str
    category: str
    artifact_profile: str = ""


@dataclass(frozen=True)
class SurfaceEvalResult:
    count: int
    top1_match: bool
    top3_match: bool
    first_useful_result_position: int | None
    drill_down_useful: bool
    no_result: bool
    weak_card: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "top1Match": self.top1_match,
            "top3Match": self.top3_match,
            "firstUsefulResultPosition": self.first_useful_result_position,
            "drillDownUseful": self.drill_down_useful,
            "noResult": self.no_result,
            "weakCard": self.weak_card,
        }


class PaperMemoryEvalHarness:
    def __init__(self, sqlite_db):
        self.sqlite_db = sqlite_db
        self.retriever = PaperMemoryRetriever(sqlite_db)

    def _evaluate_ranked_surface(self, items: list[dict[str, Any]], expected_paper_id: str) -> SurfaceEvalResult:
        top_ids = [_paper_id(item) for item in items[:3]]
        position = _first_useful_position(items, expected_paper_id)
        matched_item = next((item for item in items if _paper_id(item) == expected_paper_id), None)
        weak = _is_weak_card(matched_item) if matched_item and "paperCore" in matched_item else False
        return SurfaceEvalResult(
            count=len(items),
            top1_match=bool(items and _paper_id(items[0]) == expected_paper_id),
            top3_match=expected_paper_id in top_ids,
            first_useful_result_position=position,
            drill_down_useful=_drill_down_useful(matched_item),
            no_result=not items,
            weak_card=weak,
        )

    def _evaluate_lookup_surface(self, query: str, expected_paper_id: str) -> SurfaceEvalResult:
        candidates = self.sqlite_db.search_papers(query)
        selected = _pick_candidate(candidates, query)
        item = None
        if selected:
            paper_id = str(selected.get("arxiv_id", selected.get("paper_id", ""))).strip()
            if paper_id:
                item = self.retriever.get(paper_id, include_refs=True)
        return self._evaluate_ranked_surface([item] if item else [], expected_paper_id)

    def evaluate_case(self, case: PaperMemoryEvalCase) -> dict[str, Any]:
        search_papers_items = self.sqlite_db.search_papers(case.query)
        paper_memory_items = self.retriever.search(case.query, limit=10, include_refs=True)
        payload = {
            "caseId": case.case_id,
            "query": case.query,
            "expectedPaperId": case.expected_paper_id,
            "category": case.category,
            "artifactProfile": case.artifact_profile,
            "surfaces": {
                "search_papers": self._evaluate_ranked_surface(search_papers_items, case.expected_paper_id).to_payload(),
                "paper_lookup_and_summarize": self._evaluate_lookup_surface(case.query, case.expected_paper_id).to_payload(),
                "paper_memory_search": self._evaluate_ranked_surface(paper_memory_items, case.expected_paper_id).to_payload(),
            },
        }
        return payload

    @staticmethod
    def _surface_summary(results: list[dict[str, Any]], surface_key: str) -> dict[str, Any]:
        total = len(results) or 1
        matched = [row["surfaces"][surface_key] for row in results]
        return {
            "top1MatchCount": sum(1 for row in matched if row["top1Match"]),
            "top3MatchCount": sum(1 for row in matched if row["top3Match"]),
            "noResultCount": sum(1 for row in matched if row["noResult"]),
            "weakCardCount": sum(1 for row in matched if row["weakCard"]),
            "top1MatchRate": sum(1 for row in matched if row["top1Match"]) / total,
            "top3MatchRate": sum(1 for row in matched if row["top3Match"]) / total,
            "noResultRate": sum(1 for row in matched if row["noResult"]) / total,
            "weakCardRate": sum(1 for row in matched if row["weakCard"]) / total,
        }

    def evaluate_cases(self, cases: list[PaperMemoryEvalCase]) -> dict[str, Any]:
        results = [self.evaluate_case(case) for case in cases]
        search_summary = self._surface_summary(results, "search_papers")
        lookup_summary = self._surface_summary(results, "paper_lookup_and_summarize")
        paper_memory_summary = self._surface_summary(results, "paper_memory_search")
        paper_memory_summary["top1LiftVsSearchPapers"] = paper_memory_summary["top1MatchRate"] - search_summary["top1MatchRate"]
        paper_memory_summary["top1LiftVsLookup"] = (
            paper_memory_summary["top1MatchRate"] - lookup_summary["top1MatchRate"]
        )
        summary = {
            "caseCount": len(results),
            "searchPapers": search_summary,
            "paperLookupAndSummarize": lookup_summary,
            "paperMemory": paper_memory_summary,
        }
        return {"cases": results, "summary": summary}

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from knowledge_hub.core.models import SearchResult


class EvidenceAssemblyCollaborator(Protocol):
    def collect_claim_context(
        self,
        results: list[SearchResult],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]: ...

    def resolve_parent_context(
        self,
        result: SearchResult,
        doc_cache: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]: ...

    def answer_evidence_item(
        self,
        result: SearchResult,
        parent_ctx_by_result: dict[str, dict[str, Any]],
    ) -> dict[str, Any]: ...

    def summarize_answer_signals(
        self,
        evidence: list[dict[str, Any]],
        *,
        contradicting_beliefs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...

    def build_answer_context(
        self,
        *,
        filtered: list[SearchResult],
        parent_ctx_by_result: dict[str, dict[str, Any]],
    ) -> str: ...


@dataclass(frozen=True)
class SearcherEvidenceAssemblyCollaborator:
    searcher: Any

    def collect_claim_context(
        self,
        results: list[SearchResult],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return self.searcher._collect_claim_context(results)

    def resolve_parent_context(
        self,
        result: SearchResult,
        doc_cache: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        return self.searcher._resolve_parent_context(result, doc_cache)

    def answer_evidence_item(
        self,
        result: SearchResult,
        parent_ctx_by_result: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return self.searcher._answer_evidence_item(result, parent_ctx_by_result)

    def summarize_answer_signals(
        self,
        evidence: list[dict[str, Any]],
        *,
        contradicting_beliefs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self.searcher._summarize_answer_signals(
            evidence,
            contradicting_beliefs=contradicting_beliefs,
        )

    def build_answer_context(
        self,
        *,
        filtered: list[SearchResult],
        parent_ctx_by_result: dict[str, dict[str, Any]],
    ) -> str:
        return self.searcher._build_answer_context(
            filtered=filtered,
            parent_ctx_by_result=parent_ctx_by_result,
        )

    def build_evidence_item(
        self,
        result: SearchResult,
        parent_ctx_by_result: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return self.answer_evidence_item(result, parent_ctx_by_result)


RAGEvidenceCollaborator = SearcherEvidenceAssemblyCollaborator


def make_evidence_assembly_collaborator(searcher: Any) -> EvidenceAssemblyCollaborator:
    return SearcherEvidenceAssemblyCollaborator(searcher)


__all__ = [
    "EvidenceAssemblyCollaborator",
    "SearcherEvidenceAssemblyCollaborator",
    "RAGEvidenceCollaborator",
    "make_evidence_assembly_collaborator",
]

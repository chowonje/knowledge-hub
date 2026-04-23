from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from knowledge_hub.papers.prefilter import PAPER_MEMORY_MODE_OFF, normalize_paper_memory_mode_details


_PAPER_LOOKUP_HINTS = {
    "paper",
    "논문",
    "arxiv",
    "doi",
    "citation",
    "citations",
    "paper id",
    "paper_id",
}

_PAPER_ANALYSIS_HINTS = {
    "method",
    "methods",
    "contribution",
    "contributions",
    "limitation",
    "limitations",
    "follow up",
    "follow-up",
    "한계",
    "후속",
    "결과",
    "실험",
    "evaluation",
    "compare",
    "comparison",
}


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _normalize_source_type(source_type: str | None) -> str:
    token = str(source_type or "").strip().lower()
    if token in {"note", "all", "*"}:
        return "vault" if token == "note" else token
    return token


def _tokenize(text: str) -> set[str]:
    body = _clean_text(text).lower()
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9]+|[가-힣]{2,}", body))
    return {token for token in tokens if token}


def classify_paper_answer_query(query: str) -> str:
    text = _clean_text(query).lower()
    tokens = _tokenize(text)

    if any(hint in text for hint in _PAPER_ANALYSIS_HINTS) or {"method", "contribution", "limitation", "evaluation"} & tokens:
        return "paper_analysis"
    if any(hint in text for hint in _PAPER_LOOKUP_HINTS) or {"논문", "paper", "arxiv", "doi"} & tokens:
        return "paper_lookup"
    return "general"


@dataclass(frozen=True)
class PaperAnswerScopePlan:
    query: str
    source_type: str
    question_kind: str
    paper_scoped: bool
    requested_paper_mode: str
    matched_paper_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_budget: int = 0
    citation_budget: int = 0
    citation_style: str = "mixed_sources"
    reason: str = "general_query"
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matchedPaperIds"] = list(payload.pop("matched_paper_ids", []))
        payload["questionKind"] = payload.pop("question_kind")
        payload["paperScoped"] = payload.pop("paper_scoped")
        payload["requestedPaperMode"] = payload.pop("requested_paper_mode")
        payload["evidenceBudget"] = payload.pop("evidence_budget")
        payload["citationBudget"] = payload.pop("citation_budget")
        payload["citationStyle"] = payload.pop("citation_style")
        payload["fallbackUsed"] = payload.pop("fallback_used")
        return payload


def _paper_budget_for_kind(kind: str, *, top_k: int) -> tuple[int, int]:
    if kind == "paper_lookup":
        evidence_budget = min(max(2, int(top_k or 5)), 4)
        citation_budget = min(evidence_budget, 4)
    elif kind == "paper_analysis":
        evidence_budget = min(max(3, int(top_k or 5)), 5)
        citation_budget = min(evidence_budget, 4)
    else:
        evidence_budget = min(max(4, int(top_k or 5) + 1), 6)
        citation_budget = min(evidence_budget, 5)
    return evidence_budget, citation_budget


def build_paper_answer_plan(
    query: str,
    *,
    source_type: str | None = None,
    paper_memory_prefilter: dict[str, Any] | None = None,
    top_k: int = 5,
) -> PaperAnswerScopePlan:
    requested_mode, effective_mode, _mode_alias_applied = normalize_paper_memory_mode_details(
        (paper_memory_prefilter or {}).get("effectiveMode")
        or (paper_memory_prefilter or {}).get("requestedMode")
        or PAPER_MEMORY_MODE_OFF
    )
    source = _normalize_source_type(source_type)
    question_kind = classify_paper_answer_query(query)
    matched_paper_ids = tuple(
        str(item).strip()
        for item in list((paper_memory_prefilter or {}).get("matchedPaperIds") or [])
        if str(item).strip()
    )
    prefilter_applied = bool((paper_memory_prefilter or {}).get("applied")) and bool(matched_paper_ids)
    paper_scoped = prefilter_applied or source == "paper" or (question_kind != "general" and source in {"", "all", "paper", "note", "vault"})

    evidence_budget, citation_budget = _paper_budget_for_kind(question_kind, top_k=top_k)
    if paper_scoped and matched_paper_ids:
        evidence_budget = min(evidence_budget, max(2, len(matched_paper_ids) + 1))
        citation_budget = min(citation_budget, max(1, len(matched_paper_ids) + 1))
    elif paper_scoped:
        citation_budget = min(citation_budget, evidence_budget)

    if prefilter_applied:
        reason = "paper_memory_prefilter"
    elif source == "paper":
        reason = "explicit_paper_source"
    elif question_kind != "general" and source in {"", "all", "note", "vault"}:
        reason = "paper_like_query"
    else:
        reason = "general_query"

    if effective_mode != PAPER_MEMORY_MODE_OFF and not prefilter_applied and source == "paper":
        fallback_used = True
    else:
        fallback_used = False

    citation_style = "paper_scoped" if paper_scoped else "mixed_sources"
    return PaperAnswerScopePlan(
        query=_clean_text(query),
        source_type=source,
        question_kind=question_kind,
        paper_scoped=paper_scoped,
        requested_paper_mode=requested_mode,
        matched_paper_ids=matched_paper_ids,
        evidence_budget=evidence_budget,
        citation_budget=citation_budget,
        citation_style=citation_style,
        reason=reason,
        fallback_used=fallback_used,
    )


def _paper_id_from_evidence_item(item: dict[str, Any]) -> str:
    for key in ("paper_id", "paperId", "arxiv_id", "arxivId", "selected_paper_id", "source_paper_id"):
        token = str(item.get(key) or "").strip()
        if token:
            return token
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key in ("paper_id", "paperId", "arxiv_id", "arxivId"):
            token = str(metadata.get(key) or "").strip()
            if token:
                return token
    return ""


def apply_paper_answer_scope(
    evidence: list[dict[str, Any]],
    plan: PaperAnswerScopePlan | dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_dict = plan.to_dict() if isinstance(plan, PaperAnswerScopePlan) else dict(plan or {})
    paper_scoped = bool(plan_dict.get("paperScoped"))
    matched_paper_ids = {str(item).strip() for item in list(plan_dict.get("matchedPaperIds") or []) if str(item).strip()}
    evidence_budget = max(1, int(plan_dict.get("evidenceBudget") or len(evidence) or 1))

    if not paper_scoped:
        return list(evidence[:evidence_budget]), {
            "applied": False,
            "fallbackUsed": False,
            "reason": "non_paper_scope",
            "matchedPaperIds": sorted(matched_paper_ids),
            "keptEvidenceCount": min(len(evidence), evidence_budget),
            "droppedEvidenceCount": max(0, len(evidence) - min(len(evidence), evidence_budget)),
            "paperScoped": False,
        }

    ranked: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    matched_count = 0
    matched_by_id = False
    paper_only_items = [item for item in evidence if _normalize_source_type(item.get("source_type")) == "paper"]
    candidates = paper_only_items or list(evidence)

    for item in candidates:
        paper_id = _paper_id_from_evidence_item(item)
        key = paper_id or str(item.get("title") or item.get("parent_id") or item.get("file_path") or "").strip()
        if not key or key in seen_keys:
            continue
        if matched_paper_ids:
            if paper_id and paper_id in matched_paper_ids:
                matched_by_id = True
            elif paper_id and paper_id not in matched_paper_ids:
                continue
            elif not paper_id and _normalize_source_type(item.get("source_type")) != "paper":
                continue
        if len(ranked) >= evidence_budget:
            break
        seen_keys.add(key)
        ranked.append(dict(item))
        matched_count += 1

    if matched_paper_ids and not matched_by_id:
        return list(evidence[:evidence_budget]), {
            "applied": False,
            "fallbackUsed": True,
            "reason": "no_matching_paper_evidence",
            "matchedPaperIds": sorted(matched_paper_ids),
            "keptEvidenceCount": min(len(evidence), evidence_budget),
            "droppedEvidenceCount": max(0, len(evidence) - min(len(evidence), evidence_budget)),
            "paperScoped": True,
        }

    if not ranked:
        return list(evidence[:evidence_budget]), {
            "applied": False,
            "fallbackUsed": True,
            "reason": "no_paper_evidence_available",
            "matchedPaperIds": sorted(matched_paper_ids),
            "keptEvidenceCount": min(len(evidence), evidence_budget),
            "droppedEvidenceCount": max(0, len(evidence) - min(len(evidence), evidence_budget)),
            "paperScoped": True,
        }

    return ranked, {
        "applied": True,
        "fallbackUsed": False,
        "reason": plan_dict.get("reason", "paper_scope_applied"),
        "matchedPaperIds": sorted(matched_paper_ids),
        "keptEvidenceCount": len(ranked),
        "droppedEvidenceCount": max(0, len(evidence) - len(ranked)),
        "paperScoped": True,
    }


def _citation_identity(item: dict[str, Any]) -> str:
    paper_id = _paper_id_from_evidence_item(item)
    if paper_id:
        return f"paper:{paper_id}"
    title = _clean_text(str(item.get("title") or ""))
    source_type = _normalize_source_type(item.get("source_type"))
    parent_id = _clean_text(str(item.get("parent_id") or ""))
    return "::".join(part for part in (source_type, title, parent_id) if part)


def build_paper_citation_assembly(
    evidence: list[dict[str, Any]],
    plan: PaperAnswerScopePlan | dict[str, Any],
    *,
    max_citations: int | None = None,
) -> dict[str, Any]:
    plan_dict = plan.to_dict() if isinstance(plan, PaperAnswerScopePlan) else dict(plan or {})
    citation_budget = max(1, int(max_citations or plan_dict.get("citationBudget") or 4))
    seen: set[str] = set()
    citations: list[dict[str, Any]] = []

    for item in evidence:
        identity = _citation_identity(item)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        citations.append(
            {
                "citationId": len(citations) + 1,
                "title": _clean_text(str(item.get("title") or "Untitled")),
                "paperId": _paper_id_from_evidence_item(item),
                "sourceType": _normalize_source_type(item.get("source_type")),
                "parentId": _clean_text(str(item.get("parent_id") or "")),
                "excerpt": _clean_text(str(item.get("excerpt") or item.get("document") or ""))[:240],
                "score": float(item.get("score") or 0.0),
            }
        )
        if len(citations) >= citation_budget:
            break

    rendered_lines = []
    for citation in citations:
        label = f"[{citation['citationId']}] {citation['title']}"
        if citation["paperId"]:
            label += f" ({citation['paperId']})"
        if citation["excerpt"]:
            label += f" - {citation['excerpt']}"
        rendered_lines.append(label)

    return {
        "citationStyle": str(plan_dict.get("citationStyle") or "mixed_sources"),
        "citationBudget": citation_budget,
        "usedEvidenceCount": min(len(evidence), citation_budget),
        "citations": citations,
        "rendered": "\n".join(rendered_lines),
    }


def build_paper_answer_quality_bundle(
    query: str,
    evidence: list[dict[str, Any]],
    *,
    source_type: str | None = None,
    paper_memory_prefilter: dict[str, Any] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    plan = build_paper_answer_plan(
        query,
        source_type=source_type,
        paper_memory_prefilter=paper_memory_prefilter,
        top_k=top_k,
    )
    scoped_evidence, scope_diagnostics = apply_paper_answer_scope(evidence, plan)
    citation_bundle = build_paper_citation_assembly(scoped_evidence, plan)
    return {
        "plan": plan.to_dict(),
        "scopeDiagnostics": scope_diagnostics,
        "scopedEvidence": scoped_evidence,
        "citationAssembly": citation_bundle,
    }


__all__ = [
    "PaperAnswerScopePlan",
    "apply_paper_answer_scope",
    "build_paper_answer_plan",
    "build_paper_answer_quality_bundle",
    "build_paper_citation_assembly",
    "classify_paper_answer_query",
]

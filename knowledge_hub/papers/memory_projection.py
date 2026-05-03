"""Deterministic paper-memory projection from document-memory units."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Any

from knowledge_hub.document_memory.models import DocumentMemoryUnit
from knowledge_hub.papers.memory_models import PaperMemoryCard

PROJECTED_VERSION = "paper-memory-v2-projected"
PROJECTED_ENRICHED_VERSION = "paper-memory-v2-projected-enriched"
_CORE_SLOT_KEYS = (
    "paper_core",
    "problem_context",
    "method_core",
    "evidence_core",
    "limitations",
)
_PROBLEM_RE = re.compile(
    r"\b(problem|motivation|background|introduction|abstract|summary|문제|배경|초록|요약)\b",
    re.IGNORECASE,
)
_METHOD_RE = re.compile(
    r"\b(method|approach|architecture|pipeline|implementation|training|방법|접근|구현)\b",
    re.IGNORECASE,
)
_EVIDENCE_RE = re.compile(
    r"\b(result|results|finding|findings|evaluation|experiment|benchmark|metric|evidence|결과|평가|실험)\b",
    re.IGNORECASE,
)
_LIMITATION_RE = re.compile(
    r"\b(limit|limitation|limitations|future work|risk|caveat|restricted|한계)\b",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(r"\[[^\]]+\]\s*")
_LATEX_MARKERS = ("\\documentclass", "\\usepackage", "\\hypersetup", "\\includepdf", "\\author{", "\\title{")
_REFUSAL_MARKERS = (
    "죄송합니다",
    "충분히 제공하지 않",
    "직접 확인할 수 없",
    "insufficient information",
    "not enough information",
    "need the paper text",
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_lines(values: Any, *, limit: int | None = None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        candidates = [values]
    else:
        try:
            candidates = list(values)
        except Exception:
            candidates = [values]
    result: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        token = _clean_text(raw)
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


def _cap_join(parts: list[Any], *, limit: int = 900) -> str:
    text = _clean_text(" ".join(_clean_text(part) for part in parts if _clean_text(part)))
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _first_nonempty(*values: Any) -> str:
    for value in values:
        token = _clean_text(value)
        if token:
            return token
    return ""


def _parse_timestamp(value: Any) -> datetime | None:
    token = _clean_text(value)
    if not token:
        return None
    try:
        if token.endswith("Z"):
            return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(timezone.utc)
        parsed = datetime.fromisoformat(token)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(token, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    if re.fullmatch(r"\d{4}", token):
        return datetime(int(token), 1, 1, tzinfo=timezone.utc)
    return None


def _iso_utc(value: Any) -> str:
    parsed = _parse_timestamp(value)
    return parsed.isoformat() if parsed is not None else ""


def _coerce_card(value: dict[str, Any] | PaperMemoryCard | None) -> PaperMemoryCard | None:
    if isinstance(value, PaperMemoryCard):
        return value
    if isinstance(value, dict):
        return PaperMemoryCard.from_row(value)
    return None


def _unit_sort_key(unit: DocumentMemoryUnit) -> tuple[int, float, int, str, str]:
    return (
        0 if unit.unit_type == "document_summary" else 1,
        -float(unit.confidence or 0.0),
        int(unit.order_index or 0),
        _clean_text(unit.section_path).casefold(),
        _clean_text(unit.title).casefold(),
    )


def _unit_text(unit: DocumentMemoryUnit) -> str:
    return _first_nonempty(unit.contextual_summary, unit.source_excerpt, unit.document_thesis, unit.title)


def _unit_haystack(unit: DocumentMemoryUnit) -> str:
    return " ".join(
        [
            _clean_text(unit.unit_type),
            _clean_text(unit.title),
            _clean_text(unit.section_path),
            _clean_text(unit.contextual_summary),
            _clean_text(unit.source_excerpt),
            _clean_text(unit.search_text),
        ]
    )


def _strip_context_labels(value: Any) -> str:
    return _clean_text(_LABEL_RE.sub("", str(value or "")))


def _normalize_slot_text(value: Any, *, title: str = "", allow_brief: bool = False) -> str:
    token = _strip_context_labels(value)
    lowered = token.casefold()
    if not token:
        return ""
    if any(marker in lowered for marker in _LATEX_MARKERS):
        return ""
    if any(marker in lowered for marker in _REFUSAL_MARKERS):
        return ""
    if not allow_brief and len(token) < 24:
        title_lower = _clean_text(title).casefold()
        if lowered == title_lower or lowered.endswith(" paper") or "note" in lowered:
            return ""
    return token


def _score_unit(unit: DocumentMemoryUnit, *, pattern: re.Pattern[str], preferred_types: set[str]) -> tuple[int, float, int]:
    haystack = _unit_haystack(unit)
    score = 0
    if unit.unit_type in preferred_types:
        score += 4
    if pattern.search(haystack):
        score += 3
    if unit.unit_type == "document_summary":
        score -= 2
    return score, float(unit.confidence or 0.0), -int(unit.order_index or 0)


def _best_unit_text(
    units: list[DocumentMemoryUnit],
    *,
    pattern: re.Pattern[str],
    preferred_types: set[str],
) -> tuple[str, DocumentMemoryUnit | None]:
    ranked: list[tuple[tuple[int, float, int], DocumentMemoryUnit]] = []
    for unit in units:
        text = _unit_text(unit)
        if not text:
            continue
        score = _score_unit(unit, pattern=pattern, preferred_types=preferred_types)
        if score[0] <= 0:
            continue
        ranked.append((score, unit))
    if not ranked:
        return "", None
    ranked.sort(
        key=lambda item: (
            -item[0][0],
            -item[0][1],
            item[1].order_index,
            _clean_text(item[1].section_path).casefold(),
            _clean_text(item[1].title).casefold(),
        )
    )
    winner = ranked[0][1]
    return _unit_text(winner), winner


def coverage_payload(value: dict[str, Any] | PaperMemoryCard | None) -> dict[str, Any]:
    card = _coerce_card(value)
    if card is None:
        return {
            "filledCoreSlots": 0,
            "coreSlotCount": len(_CORE_SLOT_KEYS),
            "coverageRatio": 0.0,
            "conceptLinkCount": 0,
            "claimRefCount": 0,
        }
    filled_core_slots = sum(1 for key in _CORE_SLOT_KEYS if _clean_text(getattr(card, key)))
    return {
        "filledCoreSlots": filled_core_slots,
        "coreSlotCount": len(_CORE_SLOT_KEYS),
        "coverageRatio": round(filled_core_slots / max(1, len(_CORE_SLOT_KEYS)), 3),
        "conceptLinkCount": len(_clean_lines(card.concept_links)),
        "claimRefCount": len(_clean_lines(card.claim_refs)),
    }


def slot_diffs(
    current: dict[str, Any] | PaperMemoryCard | None,
    projected: dict[str, Any] | PaperMemoryCard | None,
) -> dict[str, dict[str, Any]]:
    current_card = _coerce_card(current)
    projected_card = _coerce_card(projected)
    diffs: dict[str, dict[str, Any]] = {}
    for key in _CORE_SLOT_KEYS:
        before = _clean_text(getattr(current_card, key, ""))
        after = _clean_text(getattr(projected_card, key, ""))
        diffs[key] = {"before": before, "after": after, "changed": before != after}
    return diffs


def recommendation(
    *,
    has_document_memory: bool,
    current: dict[str, Any] | PaperMemoryCard | None,
    projected: dict[str, Any] | PaperMemoryCard | None,
) -> str:
    if not has_document_memory or _coerce_card(projected) is None:
        return "needs_review"
    current_card = _coerce_card(current)
    projected_card = _coerce_card(projected)
    if current_card is None:
        projected_coverage = coverage_payload(projected_card)
        return "safe_cutover_candidate" if projected_coverage["filledCoreSlots"] >= 3 else "needs_review"

    diffs = slot_diffs(current_card, projected_card)
    core_changed = any(bool(item["changed"]) for item in diffs.values())
    concept_delta = len(
        set(_clean_lines(current_card.concept_links)) ^ set(_clean_lines(projected_card.concept_links))
    )
    claim_delta = len(set(_clean_lines(current_card.claim_refs)) ^ set(_clean_lines(projected_card.claim_refs)))
    search_changed = _clean_text(current_card.search_text) != _clean_text(projected_card.search_text)
    if not core_changed and concept_delta == 0 and claim_delta == 0 and not search_changed:
        return "no_change"

    before = coverage_payload(current_card)
    after = coverage_payload(projected_card)
    if (
        int(after["filledCoreSlots"]) >= int(before["filledCoreSlots"])
        and float(after["coverageRatio"]) >= float(before["coverageRatio"])
        and int(after["filledCoreSlots"]) >= 3
    ):
        return "safe_cutover_candidate"
    return "needs_review"


def audit_payload(
    *,
    paper_id: str,
    current: dict[str, Any] | PaperMemoryCard | None,
    projected: dict[str, Any] | PaperMemoryCard | None,
    has_document_memory: bool,
) -> dict[str, Any]:
    current_card = _coerce_card(current)
    projected_card = _coerce_card(projected)
    return {
        "paperId": _clean_text(paper_id),
        "hasDocumentMemory": bool(has_document_memory),
        "hasCurrentPaperMemory": current_card is not None,
        "currentVersion": _clean_text(getattr(current_card, "version", "")),
        "projectedVersion": _clean_text(getattr(projected_card, "version", "")),
        "slotDiffs": slot_diffs(current_card, projected_card),
        "conceptLinkDeltaCount": len(
            set(_clean_lines(getattr(current_card, "concept_links", [])))
            ^ set(_clean_lines(getattr(projected_card, "concept_links", [])))
        ),
        "claimRefDeltaCount": len(
            set(_clean_lines(getattr(current_card, "claim_refs", [])))
            ^ set(_clean_lines(getattr(projected_card, "claim_refs", [])))
        ),
        "searchTextChanged": _clean_text(getattr(current_card, "search_text", ""))
        != _clean_text(getattr(projected_card, "search_text", "")),
        "coverageBefore": coverage_payload(current_card),
        "coverageAfter": coverage_payload(projected_card),
        "recommendation": recommendation(
            has_document_memory=has_document_memory,
            current=current_card,
            projected=projected_card,
        ),
    }


class PaperMemoryProjector:
    """Project durable paper-memory rows from canonical-derived document memory."""

    def __init__(self, sqlite_db: Any):
        self.sqlite_db = sqlite_db

    def list_document_units(self, *, paper_id: str) -> list[DocumentMemoryUnit]:
        rows = list(self.sqlite_db.list_document_memory_units(f"paper:{paper_id}", limit=2000) or [])
        units = [DocumentMemoryUnit.from_row(row) for row in rows]
        typed = [unit for unit in units if unit is not None]
        typed.sort(key=_unit_sort_key)
        return typed

    def project(
        self,
        *,
        paper_id: str,
        paper: dict[str, Any] | None = None,
        units: list[dict[str, Any] | DocumentMemoryUnit] | None = None,
    ) -> PaperMemoryCard | None:
        token = _clean_text(paper_id)
        if not token:
            raise ValueError("paper_id is required")
        if units is None:
            typed_units = self.list_document_units(paper_id=token)
        else:
            typed_units = []
            for row in units:
                if isinstance(row, DocumentMemoryUnit):
                    typed_units.append(row)
                    continue
                unit = DocumentMemoryUnit.from_row(dict(row or {}))
                if unit is not None:
                    typed_units.append(unit)
            typed_units.sort(key=_unit_sort_key)
        if not typed_units:
            return None

        summary = next((unit for unit in typed_units if unit.unit_type == "document_summary"), typed_units[0])
        units_without_summary = [unit for unit in typed_units if unit.unit_id != summary.unit_id]
        paper_row = dict(paper or {})
        title = _first_nonempty(summary.document_title, paper_row.get("title"), token)
        overview_text, _overview_unit = _best_unit_text(
            units_without_summary or typed_units,
            pattern=_PROBLEM_RE,
            preferred_types={"summary", "background", "section"},
        )
        paper_core = _first_nonempty(
            _normalize_slot_text(overview_text, title=title),
            _normalize_slot_text(summary.source_excerpt, title=title),
            _normalize_slot_text(summary.contextual_summary, title=title),
            _normalize_slot_text(summary.document_thesis, title=title),
        )
        problem_context, _problem_unit = _best_unit_text(
            units_without_summary or typed_units,
            pattern=_PROBLEM_RE,
            preferred_types={"background", "summary", "section"},
        )
        method_core, method_unit = _best_unit_text(
            units_without_summary or typed_units,
            pattern=_METHOD_RE,
            preferred_types={"method", "section", "list_block"},
        )
        evidence_core, evidence_unit = _best_unit_text(
            units_without_summary or typed_units,
            pattern=_EVIDENCE_RE,
            preferred_types={"result", "section", "table_block", "list_block"},
        )
        limitations, limitations_unit = _best_unit_text(
            units_without_summary or typed_units,
            pattern=_LIMITATION_RE,
            preferred_types={"limitation", "section"},
        )
        paper_core = _normalize_slot_text(paper_core, title=title)
        problem_context = _normalize_slot_text(problem_context, title=title)
        method_core = _normalize_slot_text(method_core, title=title, allow_brief=True)
        evidence_core = _normalize_slot_text(evidence_core, title=title, allow_brief=True)
        limitations = _normalize_slot_text(limitations, title=title, allow_brief=True)
        concept_links = _clean_lines(
            [
                *list(summary.concepts or []),
                *[concept for unit in typed_units for concept in list(unit.concepts or [])],
            ],
            limit=12,
        )
        claim_refs = _clean_lines(
            [
                *list(summary.claims or []),
                *[claim for unit in typed_units for claim in list(unit.claims or [])],
            ],
            limit=12,
        )
        published_at = _first_nonempty(
            summary.document_date,
            summary.event_date,
            summary.observed_at,
            _iso_utc(paper_row.get("published_at") or paper_row.get("year")),
        )
        evidence_window = _first_nonempty(summary.document_date, summary.event_date, summary.observed_at, published_at)
        search_text = _cap_join(
            [
                title,
                paper_core,
                problem_context,
                method_core,
                evidence_core,
                limitations,
                _clean_text(method_unit.section_path if method_unit else ""),
                _clean_text(evidence_unit.section_path if evidence_unit else ""),
                _clean_text(limitations_unit.section_path if limitations_unit else ""),
                " ".join(concept_links),
                " ".join(claim_refs),
            ],
            limit=900,
        )

        selected_confidences = [
            float(summary.confidence or 0.0),
            float(method_unit.confidence or 0.0) if method_unit is not None else 0.0,
            float(evidence_unit.confidence or 0.0) if evidence_unit is not None else 0.0,
            float(limitations_unit.confidence or 0.0) if limitations_unit is not None else 0.0,
        ]
        filled_core_slots = sum(
            1 for value in (paper_core, problem_context, method_core, evidence_core, limitations) if _clean_text(value)
        )
        avg_confidence = sum(selected_confidences) / max(1, len([value for value in selected_confidences if value > 0.0]))
        if filled_core_slots >= 4 and avg_confidence >= 0.5 and len(claim_refs) >= 2:
            quality_flag = "ok"
        elif filled_core_slots >= 2:
            quality_flag = "needs_review"
        else:
            quality_flag = "unscored"

        return PaperMemoryCard(
            memory_id=f"paper-memory:{token}:{hashlib.sha1(token.encode('utf-8')).hexdigest()[:10]}",
            paper_id=token,
            source_note_id=f"paper:{token}",
            title=title,
            paper_core=paper_core,
            problem_context=_first_nonempty(problem_context, _normalize_slot_text(summary.document_thesis, title=title)),
            method_core=method_core,
            evidence_core=evidence_core,
            limitations=limitations,
            concept_links=concept_links,
            claim_refs=claim_refs,
            published_at=published_at,
            evidence_window=evidence_window,
            search_text=search_text,
            quality_flag=quality_flag,
            version=PROJECTED_VERSION,
        )


__all__ = [
    "PROJECTED_ENRICHED_VERSION",
    "PROJECTED_VERSION",
    "PaperMemoryProjector",
    "audit_payload",
    "coverage_payload",
    "recommendation",
    "slot_diffs",
]
